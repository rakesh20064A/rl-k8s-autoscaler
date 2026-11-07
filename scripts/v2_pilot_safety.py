"""scripts/v2_pilot_safety.py

V2 Phase 3A PART 5/6/7/8: live safety monitoring + per-step telemetry for
the one-rollout V2 PPO pilot. An SB3 `BaseCallback` subclass -- SB3 calls
`_on_step()` once after every `env.step()` inside `model.learn()`, which is
the only hook point available to observe/intervene during a real rollout
(the training script itself never gets per-step control once `.learn()` is
called).

Hard safety contract (task's own fixed numbers, never adjusted after
seeing results):
  - pod memory >= 230Mi for >= SUSTAINED_BREACH_SAMPLES consecutive samples
  - any pod restart during the pilot (RESTARTS increasing past the
    pre-pilot baseline recorded by the caller)
  - zero Ready replicas sustained

On any of these, `_on_step()` returns False, which stops `model.learn()`
early (SB3's own mechanism -- not a kill signal to Kubernetes). The caller
(`scripts/train_ppo_live_v2.py`) is responsible for all cleanup/restoration
in its own `finally` blocks regardless of how `.learn()` ends.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from stable_baselines3.common.callbacks import BaseCallback

from scripts.v2_traffic_orchestrator import V2TrafficSyncedWrapper

MEMORY_SAFETY_THRESHOLD_MIB = 230.0  # fixed for this pilot -- never changed after seeing results
SUSTAINED_BREACH_SAMPLES = 2

# V2 Phase 3B PART 1: the ONE authoritative Prometheus endpoint for the
# entire pilot -- LiveKubernetesBackend, V2TrafficOrchestrator's drain
# query, and this module's own safety/telemetry monitoring ALL use this
# same URL, wired explicitly from scripts/train_ppo_live_v2.py rather than
# depending on each component's own (coincidentally identical, but never
# verified together until Phase 3A's live failure) default. Phase 3A's
# physical attempt 1 failed because the pilot's own external monitoring
# port-forward was placed on 19090 while every internal client (Live
# KubernetesBackend, the orchestrator's drain query) still pointed at their
# unmodified default of 9090 -- there must be exactly one port used
# everywhere, never two "coincidentally compatible" paths.
PROMETHEUS_URL = "http://localhost:9090"

REWARD_COMPONENT_KEYS = (
    "reliability", "sla", "pressure", "waste", "action", "reversal", "boundary",
)


@dataclass(frozen=True)
class PodHealthSnapshot:
    name: str
    memory_bytes: Optional[float]
    restart_count: int
    ready: bool
    phase: str


def get_pod_health(namespace: str, label_selector: str, run=subprocess.run) -> List[PodHealthSnapshot]:
    """Reads live pod health directly from the Kubernetes API via kubectl
    (never from Prometheus for restart/ready -- those must reflect the
    Deployment/Pod object's own authoritative state, matching the
    project's established CORR-25 evidence-hierarchy convention)."""
    proc = run(
        ["kubectl", "get", "pods", "-n", namespace, "-l", label_selector, "-o", "json"],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        return []
    data = json.loads(proc.stdout or "{}")
    snapshots = []
    for item in data.get("items", []):
        name = item["metadata"]["name"]
        status = item.get("status", {})
        phase = status.get("phase", "Unknown")
        container_statuses = status.get("containerStatuses", [])
        restart_count = max((c.get("restartCount", 0) for c in container_statuses), default=0)
        ready = all(c.get("ready", False) for c in container_statuses) if container_statuses else False
        snapshots.append(PodHealthSnapshot(name=name, memory_bytes=None, restart_count=restart_count, ready=ready, phase=phase))
    return snapshots


def verify_prometheus_internal_paths(
    prom_url: str = PROMETHEUS_URL, namespace: str = "autoscaler-app", route: str = "/work",
) -> Dict[str, Any]:
    """V2 Phase 3B PART 2: tests the SAME `PrometheusClient` type/config
    `LiveKubernetesBackend` constructs internally (not merely an external
    `curl`), AND the exact PromQL `drain_in_flight_requests()` uses --
    called from the trainer's preflight, BEFORE any HPA removal or
    controller-lock acquisition. A query returning `(None, None)` (absent
    metric/empty series) is NOT a failure (collectors.metrics_collector.
    PrometheusClient.query()'s own documented contract); only a non-None
    error string is."""
    from collectors.metrics_collector import PrometheusClient

    client = PrometheusClient(base_url=prom_url)
    ready, ready_message = client.check_readiness()

    backend_value, backend_error = client.query("up")
    drain_promql = f'sum(demo_app_requests_in_progress{{namespace="{namespace}",route="{route}"}})'
    drain_value, drain_error = client.query(drain_promql)

    return {
        "prometheus_url": prom_url,
        "ready": ready,
        "ready_message": ready_message,
        "backend_query_ok": backend_error is None,
        "backend_query_value": backend_value,
        "backend_query_error": backend_error,
        "drain_query_ok": drain_error is None,
        "drain_query_value": drain_value,
        "drain_query_error": drain_error,
        "all_ok": ready and backend_error is None and drain_error is None,
    }


def _query_prometheus_scalar(prom_base_url: str, promql: str, timeout: float = 5.0) -> Optional[float]:
    url = f"{prom_base_url}/api/v1/query?query=" + urllib.parse.quote(promql)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
        result = data.get("data", {}).get("result", [])
        if not result:
            return None
        return float(result[0]["value"][1]) / 1048576.0
    except Exception:  # noqa: BLE001 -- a monitoring-query failure must never crash the pilot; treated as "unknown" this sample
        return None


def get_aggregate_memory_mib(prom_base_url: str, namespace: str, pod_regex: str, timeout: float = 5.0) -> Optional[float]:
    """REPORTING ONLY -- summed across every pod matching `pod_regex`. Must
    NEVER drive the OOM safety threshold (see V2 Phase 3C PART 1): this
    value scales up with replica count even when no individual pod is
    anywhere near its own per-pod memory limit, which is exactly what
    caused V2 Phase 3B's physical attempt 2 to safety-stop after only 7
    steps despite zero real OOM risk."""
    q = f'sum(container_memory_working_set_bytes{{namespace="{namespace}",pod=~"{pod_regex}"}})'
    return _query_prometheus_scalar(prom_base_url, q, timeout)


def get_max_per_pod_memory_mib(prom_base_url: str, namespace: str, pod_regex: str, timeout: float = 5.0) -> Optional[float]:
    """V2 Phase 3C PART 1/2 CORRECTION: the metric that actually drives the
    230Mi OOM safety threshold. `sum by (pod) (...)` collapses each pod's
    (possibly multi-container, though this deployment currently has
    exactly one) own container series into a single per-pod total FIRST,
    THEN `max(...)` takes the worst single pod across the whole deployment
    -- correctly answers "is any ONE pod close to its own 256Mi limit?"
    rather than "what does the whole deployment sum to?" (the aggregate
    query's question, appropriate only for reporting, never for this
    per-pod OOM guard). Safe even if a sidecar container is introduced
    later: `sum by (pod)` still correctly totals that one pod's own
    containers before comparing pods against each other."""
    q = f'max(sum by (pod) (container_memory_working_set_bytes{{namespace="{namespace}",pod=~"{pod_regex}"}}))'
    return _query_prometheus_scalar(prom_base_url, q, timeout)


class PilotSafetyTerminated(RuntimeError):
    """Raised by the caller (train_ppo_live_v2.py) after model.learn()
    returns early due to a tripped safety condition, to distinguish a
    safety-terminated pilot from a normal completion."""


class PilotSafetyCallback(BaseCallback):
    """Plain (non-dataclass) BaseCallback subclass -- SB3's BaseCallback has
    its own required `__init__(verbose)` contract (sets `self.model`,
    `self.training_env`, etc. when the callback is attached to
    `model.learn()`); mixing that with a dataclass-generated `__init__`
    risks silently skipping SB3's own setup, so a plain explicit `__init__`
    is used instead."""

    def __init__(
        self,
        namespace: str,
        label_selector: str,
        prom_base_url: str,
        pod_regex: str,
        baseline_restart_counts: Dict[str, int],
        telemetry_path: Path,
        synced_wrapper: V2TrafficSyncedWrapper,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.namespace = namespace
        self.label_selector = label_selector
        self.prom_base_url = prom_base_url
        self.pod_regex = pod_regex
        self.baseline_restart_counts = dict(baseline_restart_counts)
        self.telemetry_path = telemetry_path
        self._synced_wrapper = synced_wrapper

        self._consecutive_memory_breaches = 0
        self._termination_reason: Optional[str] = None
        self._max_per_pod_samples_mib: List[float] = []
        self._aggregate_samples_mib: List[float] = []
        self._provenance_row_written = False
        self._telemetry_rows: List[Dict[str, Any]] = []
        self._profile_counts: Dict[str, int] = {}
        self._action_counts = {0: 0, 1: 0, 2: 0}
        self._effective_action_counts = {0: 0, 1: 0, 2: 0}
        self._override_counts: Dict[str, int] = {}
        self._invalid_step_count = 0
        self._n_steps_seen = 0
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def termination_reason(self) -> Optional[str]:
        return self._termination_reason

    @property
    def max_per_pod_samples_mib(self) -> List[float]:
        return list(self._max_per_pod_samples_mib)

    @property
    def aggregate_samples_mib(self) -> List[float]:
        return list(self._aggregate_samples_mib)

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "n_steps_seen": self._n_steps_seen,
            "profile_counts": dict(self._profile_counts),
            "action_counts": dict(self._action_counts),
            "effective_action_counts": dict(self._effective_action_counts),
            "override_counts": dict(self._override_counts),
            "invalid_step_count": self._invalid_step_count,
            "invalid_step_rate": (self._invalid_step_count / self._n_steps_seen) if self._n_steps_seen else 0.0,
            "max_per_pod_memory_mib": max(self._max_per_pod_samples_mib) if self._max_per_pod_samples_mib else None,
            "max_aggregate_memory_mib": max(self._aggregate_samples_mib) if self._aggregate_samples_mib else None,
            "termination_reason": self._termination_reason,
        }

    def _on_step(self) -> bool:
        self._n_steps_seen += 1
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}
        actions = self.locals.get("actions")
        sampled_action = int(actions[0]) if actions is not None else None

        profile = self._synced_wrapper.current_profile_name
        if profile:
            self._profile_counts[profile] = self._profile_counts.get(profile, 0) + 1

        if sampled_action is not None:
            self._action_counts[sampled_action] = self._action_counts.get(sampled_action, 0) + 1
        effective_action = info.get("action_index")
        if effective_action is not None:
            self._effective_action_counts[effective_action] = self._effective_action_counts.get(effective_action, 0) + 1
        override_reason = info.get("action_override_reason", "none")
        self._override_counts[override_reason] = self._override_counts.get(override_reason, 0) + 1
        if not info.get("sample_valid", True):
            self._invalid_step_count += 1

        # V2 Phase 4A PART 3: the EXACT normalized 11-element vector SB3 just
        # received from env.step() (and will feed to the policy next) --
        # `self.locals["new_obs"]` is SB3's own per-step local, populated by
        # its vectorized-env step inside collect_rollouts(); never
        # recomputed/renormalized here, so this can never drift from
        # whatever `autoscaler.src.rl_env.observation.build_observation()`
        # (the frozen V2 observation contract) actually produced. Purely
        # read-only instrumentation -- adding this key changes no action,
        # reward, or training behavior.
        new_obs = self.locals.get("new_obs")
        normalized_observation = new_obs[0].tolist() if new_obs is not None else None

        row = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "profile": profile,
            "sampled_action": sampled_action,
            "effective_action": effective_action,
            "override_reason": override_reason,
            "reward_total": info.get("reward_total"),
            "reward_components": info.get("reward_components"),
            "raw_observation": info.get("raw_observation"),
            "normalized_observation": normalized_observation,
            "deployment_current_replicas_after": info.get("deployment_current_replicas_after"),
            "ready_replicas_after": info.get("ready_replicas_after"),
        }
        self._telemetry_rows.append(row)
        with open(self.telemetry_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

        # -- Hard safety checks --
        pods = get_pod_health(self.namespace, self.label_selector)
        for pod in pods:
            baseline = self.baseline_restart_counts.get(pod.name, pod.restart_count)
            if pod.restart_count > baseline:
                return self._trip_safety(f"pod_restart_detected:{pod.name}")
        if pods and not any(p.ready for p in pods):
            return self._trip_safety("zero_ready_replicas")

        # V2 Phase 3C PART 1 CORRECTION: the safety THRESHOLD is driven
        # exclusively by the worst SINGLE pod's memory, never the
        # deployment-wide aggregate (see get_max_per_pod_memory_mib's own
        # docstring for why -- V2 Phase 3B's physical attempt 2 safety-
        # stopped from aggregate-across-5-replicas crossing 230Mi while no
        # individual pod was ever close to its own 256Mi limit). The
        # aggregate is still recorded every sample, for reporting only.
        max_pod_mib = get_max_per_pod_memory_mib(self.prom_base_url, self.namespace, self.pod_regex)
        aggregate_mib = get_aggregate_memory_mib(self.prom_base_url, self.namespace, self.pod_regex)
        if aggregate_mib is not None:
            self._aggregate_samples_mib.append(aggregate_mib)
        if max_pod_mib is not None:
            self._max_per_pod_samples_mib.append(max_pod_mib)
            if max_pod_mib >= MEMORY_SAFETY_THRESHOLD_MIB:
                self._consecutive_memory_breaches += 1
            else:
                self._consecutive_memory_breaches = 0
            if self._consecutive_memory_breaches >= SUSTAINED_BREACH_SAMPLES:
                return self._trip_safety(f"memory_safety_threshold_breached:{max_pod_mib:.1f}MiB_per_pod")

        return True

    def _trip_safety(self, reason: str) -> bool:
        """Records the termination reason and writes the (idempotent,
        exactly-once) pilot-safety-interruption provenance row -- see
        autoscaler.src.rl_env.training_wrapper.LedgeredEpisodeWrapper.
        log_pilot_safety_interruption() for the duplicate-write guard
        (`_episode_completed`), which this method relies on rather than
        reimplementing its own."""
        self._termination_reason = reason
        if not self._provenance_row_written:
            ledgered_env = getattr(self._synced_wrapper, "env", None)
            if ledgered_env is not None and hasattr(ledgered_env, "log_pilot_safety_interruption"):
                ledgered_env.log_pilot_safety_interruption(reason)
            self._provenance_row_written = True
        return False
