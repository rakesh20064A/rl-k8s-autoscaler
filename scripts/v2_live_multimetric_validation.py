#!/usr/bin/env python3
"""scripts/v2_live_multimetric_validation.py

V2 Phase 4D-D1: a live, read-and-observe validation runner, NOT a training
or evaluation script -- it issues no PPO actions at all. Proves the V2
multi-metric contract on the real cluster before the pilot/training phases
by running five short (<=300s) scenarios and sampling every 30s:

  S1 idle_baseline : no traffic, 120s, demo-app-hpa-v2 present
  S2 fixed2_cpu    : demo-app-hpa-v2 REMOVED, replicas held at 2 (PPO's
                     own exclusivity lock acquired -- no HPA governs),
                     constant_cpu traffic -- the worst case "PPO holds at
                     the floor"; pod restarts here ARE the measurement,
                     never an abort condition.
  S3 hpa_cpu       : demo-app-hpa-v2 present, constant_cpu traffic
  S4 hpa_memory    : demo-app-hpa-v2 present, constant_memory traffic
  S5 hpa_mixed     : demo-app-hpa-v2 present, constant_mixed traffic

Reuses existing, already-tested code paths -- no new traffic or metric
logic:
  - `autoscaler.src.rl_env.live_backend.LiveKubernetesBackend` (V2 HPA
    identity, floor=2) for the deployment/metrics snapshot.
  - `autoscaler.src.rl_env.observation.build_observation()` -- the EXACT
    function `environment_v2.py` calls, with the SAME `ObservationConfig`
    `config_v2.load_env_config_v2_from_yaml()` returns -- so every
    normalized value here is numerically identical to what training would
    have computed from the same raw snapshot.
  - `scripts.traffic_orchestrator`'s generic V5 transport/generator
    lifecycle (`V5TransportManager`, `start_generator`/`stop_generator`/
    `verify_generator_started`) -- only the PROFILE RESOLUTION step is
    overridden (`ValidationTrafficOrchestrator`), to read from the new,
    separate `configs/traffic_profiles/v2_validation/` directory instead
    of either firewalled registry (V1's six-name list, V2's training
    allowlist) -- these are deliberately NOT training profiles.
  - `scripts.v2_pilot_safety`'s `get_pod_health`/`get_max_per_pod_memory_mib`/
    `MEMORY_SAFETY_THRESHOLD_MIB` (230Mi, unchanged) for the safety guard.
  - `autoscaler.src.rl_env.controller_exclusivity`'s existing lock
    semantics (`CONTROLLER_PPO` for S2's "nothing else may scale" manual
    hold, `CONTROLLER_CPU_MEMORY_HPA` for S3-S5).

Never calls `model.predict()`/`PPO.load()` and never writes an episode
ledger row -- this is observation-only against the live cluster.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoscaler.src.rl_env.config_v2 import load_env_config_v2_from_yaml  # noqa: E402
from autoscaler.src.rl_env.controller_exclusivity import (  # noqa: E402
    CONTROLLER_CPU_MEMORY_HPA,
    CONTROLLER_PPO,
    V2_HPA_MANIFEST_PATH,
    V2_HPA_NAME,
    ControllerConflictError,
    acquire_controller,
    release_controller,
)
from autoscaler.src.rl_env.live_backend import LiveKubernetesBackend  # noqa: E402
from autoscaler.src.rl_env.observation import HistoryState, build_observation  # noqa: E402
from scripts import rl_live_common as common  # noqa: E402
from scripts import traffic_orchestrator as to  # noqa: E402
from scripts.v2_pilot_safety import (  # noqa: E402
    MEMORY_SAFETY_THRESHOLD_MIB,
    PROMETHEUS_URL,
    get_max_per_pod_memory_mib,
    get_pod_health,
)
from loadgen.traffic_generator import parse_traffic_profile  # noqa: E402

APP_NAMESPACE = "autoscaler-app"
DEPLOYMENT = "demo-app"
POD_LABEL = "app=demo-app"
POD_REGEX = f"{DEPLOYMENT}-.*"

VALIDATION_PROFILE_DIR = PROJECT_ROOT / "configs" / "traffic_profiles" / "v2_validation"
OUTPUT_ROOT = PROJECT_ROOT / "persistent-data" / "traffic" / "v2_phase4dd_validation"

SAMPLE_INTERVAL_S = 30.0
IDLE_SETTLE_S = 120.0
READY_ZERO_ABORT_S = 60.0
DEPLOYMENT_READY_WAIT_S = 240.0

SCENARIOS: List[Dict[str, Any]] = [
    {"name": "idle_baseline", "kind": "idle", "profile": None, "duration_s": 120.0},
    {"name": "fixed2_cpu", "kind": "fixed2", "profile": "constant_cpu", "duration_s": 300.0},
    {"name": "hpa_cpu", "kind": "hpa", "profile": "constant_cpu", "duration_s": 300.0},
    {"name": "hpa_memory", "kind": "hpa", "profile": "constant_memory", "duration_s": 300.0},
    {"name": "hpa_mixed", "kind": "hpa", "profile": "constant_mixed", "duration_s": 300.0},
]


# -- profile resolution (new, firewall-free namespace) -----------------------


def resolve_validation_profile_metadata(
    profile_name: str, profile_dir: Path = VALIDATION_PROFILE_DIR,
) -> to.ProfileMetadata:
    """Deliberately NOT firewalled against V1's six-name registry or V2's
    training allowlist -- `configs/traffic_profiles/v2_validation/` is a
    separate, non-training namespace by design (never copied into either
    registry, never used by any training/evaluation script)."""
    path = profile_dir / f"{profile_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"validation profile not found: {path}")
    actual_sha256 = common.sha256_file(path)
    profile = parse_traffic_profile(path)
    return to.ProfileMetadata(
        name=profile_name, path=path, sha256=actual_sha256, seed=profile.seed,
        duration_seconds=profile.total_duration, n_phases=len(profile.phases),
    )


class ValidationTrafficOrchestrator(to.TrafficOrchestrator):
    """Identical to `to.TrafficOrchestrator` except profile resolution
    reads from `configs/traffic_profiles/v2_validation/` -- every other
    method (route lifecycle, generator process lifecycle, provenance,
    drain) is inherited unchanged."""

    def start_episode_traffic(self, profile_name: str, episode_id: str) -> to.GeneratorHandle:
        profile = resolve_validation_profile_metadata(profile_name)
        run_id = to.resolve_next_available_run_id(episode_id, self.traffic_base_dir)
        route = self.ensure_route_established(run_id)
        if not to.verify_route_alive(route, self.http_get):
            raise to.RouteEstablishmentFailure(f"route {route.target_url} not alive before scenario {episode_id}")
        handle = to.start_generator(profile, route, run_id, popen_factory=self.popen_factory, clock=self.clock)
        if not to.verify_generator_started(handle, sleep_fn=self.sleep_fn):
            to.stop_generator(handle)
            raise to.TrafficGeneratorFailure(f"generator failed to start for validation profile {profile_name!r}")
        return handle


def _default_http_get(url: str, timeout: float) -> "tuple[bool, str]":
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return 200 <= resp.status < 300, str(resp.status)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# -- pure, independently-testable helpers ------------------------------------


def evaluate_safety_abort(
    ready_replicas: int,
    ready_zero_since_monotonic: Optional[float],
    now_monotonic: float,
    max_per_pod_memory_mib: Optional[float],
    memory_threshold_mib: float = MEMORY_SAFETY_THRESHOLD_MIB,
    ready_zero_abort_s: float = READY_ZERO_ABORT_S,
) -> "tuple[Optional[float], Optional[str]]":
    """Pure function: returns (updated ready_zero_since_monotonic,
    abort_reason_or_None). Abort conditions (per scenario, task sec.4):
    ready replicas == 0 sustained for >= `ready_zero_abort_s`, OR any pod's
    own memory >= `memory_threshold_mib` (the project's fixed 230Mi OOM
    guard, unchanged). Pod RESTARTS are never an abort condition here --
    S2 deliberately measures them."""
    if ready_replicas == 0:
        if ready_zero_since_monotonic is None:
            ready_zero_since_monotonic = now_monotonic
        elif now_monotonic - ready_zero_since_monotonic >= ready_zero_abort_s:
            return ready_zero_since_monotonic, f"ready_replicas=0 for >= {ready_zero_abort_s}s"
    else:
        ready_zero_since_monotonic = None

    if max_per_pod_memory_mib is not None and max_per_pod_memory_mib >= memory_threshold_mib:
        return (
            ready_zero_since_monotonic,
            f"per-pod memory {max_per_pod_memory_mib:.1f}MiB >= {memory_threshold_mib}MiB safety guard",
        )
    return ready_zero_since_monotonic, None


def parse_hpa_v2_status(hpa_json: Optional[dict]) -> Optional[Dict[str, Any]]:
    """Pure function: extracts the fields this phase cares about from a raw
    `kubectl get hpa demo-app-hpa-v2 -o json` document (or None if absent).
    Returns None if the object is absent; never raises on a malformed/
    partial status (treats missing sub-fields as None)."""
    if hpa_json is None:
        return None
    status = hpa_json.get("status", {})
    cpu_pct = memory_pct = None
    for metric in status.get("currentMetrics", []) or []:
        resource = metric.get("resource", {})
        name = resource.get("name")
        current = resource.get("current", {})
        if name == "cpu":
            cpu_pct = current.get("averageUtilization")
        elif name == "memory":
            memory_pct = current.get("averageUtilization")
    return {
        "cpu_current_pct": cpu_pct,
        "memory_current_pct": memory_pct,
        "current_replicas": status.get("currentReplicas"),
        "desired_replicas": status.get("desiredReplicas"),
    }


def build_sample_row(
    *,
    scenario: str,
    timestamp_utc: str,
    elapsed_in_scenario_s: float,
    obs_result,
    replicas_desired: int,
    replicas_ready: int,
    hpa_status: Optional[Dict[str, Any]],
    pod_restart_counts: Dict[str, int],
    pod_request_rates: Optional[Dict[str, float]],
    max_per_pod_memory_mib: Optional[float] = None,
) -> Dict[str, Any]:
    """Pure function assembling one JSONL row. `obs_result` is an
    `observation.ObservationBuildResult` (the SAME type `environment_v2.py`
    gets from `build_observation()`). `max_per_pod_memory_mib` (V2 Phase
    4D-D3, Fix D) is the SAME value `evaluate_safety_abort()` already
    queries for the 230Mi guard -- persisted here too, so a post-hoc
    analysis of any run never has to approximate it from
    `memory_per_replica`'s own average-based raw observation."""
    return {
        "timestamp_utc": timestamp_utc,
        "scenario": scenario,
        "elapsed_in_scenario_s": round(elapsed_in_scenario_s, 1),
        "raw_observation": obs_result.raw,
        "normalized_observation": obs_result.normalized,
        "sample_valid": obs_result.validity.sample_valid,
        "latency_observation_state": obs_result.validity.latency_observation_state,
        "ready_replicas_used": obs_result.ready_replicas_used,
        "deployment_desired_replicas": replicas_desired,
        "deployment_ready_replicas": replicas_ready,
        "hpa_v2_present": hpa_status is not None,
        "hpa_v2_cpu_current_pct": hpa_status.get("cpu_current_pct") if hpa_status else None,
        "hpa_v2_memory_current_pct": hpa_status.get("memory_current_pct") if hpa_status else None,
        "hpa_v2_desired_replicas": hpa_status.get("desired_replicas") if hpa_status else None,
        "pod_restart_counts": pod_restart_counts,
        "pod_request_rates_rps": pod_request_rates,
        "max_per_pod_memory_mib": max_per_pod_memory_mib,
    }


def query_per_pod_request_rate(prom_base_url: str, namespace: str, route: str = "/work") -> Optional[Dict[str, float]]:
    """Best-effort, read-only: per-pod request rate via the SAME metric
    name/labels used elsewhere in this project (`demo_app_requests_total`).
    Returns None (never raises) on any query failure -- this is a reporting
    convenience, not a safety-relevant signal."""
    try:
        promql = f'sum by (pod) (rate(demo_app_requests_total{{namespace="{namespace}",route="{route}"}}[1m]))'
        import urllib.parse

        url = f"{prom_base_url}/api/v1/query?query=" + urllib.parse.quote(promql)
        with urllib.request.urlopen(url, timeout=5.0) as resp:  # noqa: S310
            data = json.loads(resp.read())
        result = data.get("data", {}).get("result", [])
        return {row["metric"].get("pod", "unknown"): float(row["value"][1]) for row in result}
    except Exception:  # noqa: BLE001
        return None


# -- scenario result + runner -------------------------------------------------


@dataclass
class ScenarioResult:
    name: str
    started_utc: str
    ended_utc: Optional[str] = None
    aborted: bool = False
    abort_reason: Optional[str] = None
    n_samples: int = 0
    restart_deltas: Dict[str, int] = field(default_factory=dict)


class ValidationRunner:
    def __init__(
        self,
        output_dir: Path,
        kubectl_run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        scenarios: Optional[List[Dict[str, Any]]] = None,
        sample_interval_s: float = SAMPLE_INTERVAL_S,
        idle_settle_s: float = IDLE_SETTLE_S,
        backend: Optional[LiveKubernetesBackend] = None,
        orchestrator: Optional[to.TrafficOrchestrator] = None,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.kubectl_run = kubectl_run
        self.sleep_fn = sleep_fn
        self.clock = clock
        self.now_fn = now_fn
        self.scenarios = scenarios if scenarios is not None else SCENARIOS
        self.sample_interval_s = sample_interval_s
        self.idle_settle_s = idle_settle_s

        self.samples_path = self.output_dir / "samples.jsonl"
        self.status_path = self.output_dir / "run_status.json"
        self.summary_path = self.output_dir / "scenario_summary.json"

        self.env_config = load_env_config_v2_from_yaml()
        self.backend = backend if backend is not None else LiveKubernetesBackend(
            hpa_name=V2_HPA_NAME, hpa_manifest_path=V2_HPA_MANIFEST_PATH,
            min_replicas=self.env_config.gating.min_replicas, initial_replicas=self.env_config.initial_replicas,
        )
        self.orchestrator = orchestrator if orchestrator is not None else ValidationTrafficOrchestrator(
            http_get=_default_http_get, traffic_base_dir=self.output_dir / "traffic",
            # V2 Phase 4D-D3 (Fix D): explicit, own-run-directory ledger path --
            # the base `to.TrafficOrchestrator`'s own default
            # (`TRAFFIC_PROVENANCE_LEDGER_PATH`) points at V1's SHARED
            # `persistent-data/rl_training/traffic_provenance_ledger.jsonl`,
            # which this validation runner must never write into. The
            # orchestrator's own default behaviour for training callers is
            # completely unchanged -- only this runner's own construction
            # passes an explicit override.
            ledger_path=self.output_dir / "traffic_provenance_ledger.jsonl",
        )

    # -- small I/O helpers ---------------------------------------------------

    def _write_status(self, **fields: Any) -> None:
        payload = {"updated_utc": self.now_fn().isoformat(), **fields}
        tmp = self.status_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        tmp.replace(self.status_path)

    def _append_sample(self, row: Dict[str, Any]) -> None:
        with open(self.samples_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def _read_hpa_v2_json(self) -> Optional[dict]:
        proc = self.kubectl_run(
            ["kubectl", "get", "hpa", V2_HPA_NAME, "-n", APP_NAMESPACE, "-o", "json"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            return None
        try:
            return json.loads(proc.stdout or "{}")
        except ValueError:
            return None

    def _ensure_2_ready(self) -> None:
        current = self.backend._kubectl.get_replica_state()
        target = self.env_config.initial_replicas
        if (current.desired, current.current, current.ready, current.available) != (target,) * 4:
            self.backend._kubectl.scale(target)
            self.backend._kubectl.wait_for_replica_state(target, DEPLOYMENT_READY_WAIT_S)

    # -- one scenario ---------------------------------------------------------

    def run_scenario(self, scenario: Dict[str, Any]) -> ScenarioResult:
        name, kind, duration_s = scenario["name"], scenario["kind"], scenario["duration_s"]
        profile_name = scenario.get("profile")
        result = ScenarioResult(name=name, started_utc=self.now_fn().isoformat())
        self._write_status(status="RUNNING", current_scenario=name)

        baseline_pods = get_pod_health(APP_NAMESPACE, POD_LABEL, run=self.kubectl_run)
        baseline_restarts = {p.name: p.restart_count for p in baseline_pods}

        lock_acquired: Optional[str] = None
        hpa_removed_for_scenario = False
        handle: Optional[to.GeneratorHandle] = None
        history = HistoryState.new()
        previous_normalized: Optional[List[float]] = None
        ready_zero_since: Optional[float] = None
        n_samples = 0

        try:
            self._ensure_2_ready()

            if kind == "fixed2":
                self.backend._kubectl.delete_hpa()
                hpa_removed_for_scenario = True
                acquire_controller(CONTROLLER_PPO, self.backend._kubectl)
                lock_acquired = CONTROLLER_PPO
            elif kind == "hpa":
                self.backend._kubectl.restore_hpa()
                acquire_controller(CONTROLLER_CPU_MEMORY_HPA, self.backend._kubectl)
                lock_acquired = CONTROLLER_CPU_MEMORY_HPA
            # kind == "idle": no lock change; HPA stays whatever it already is (canonical: present).

            if profile_name:
                handle = self.orchestrator.start_episode_traffic(profile_name, f"v2validation_{name}")

            start_mono = self.clock()
            while (self.clock() - start_mono) < duration_s:
                elapsed = self.clock() - start_mono
                snapshot = self.backend.current_snapshot()
                obs_result = build_observation(
                    snapshot.metrics, snapshot.replicas,
                    seconds_since_last_effective_action=elapsed,
                    history=history, previous_normalized=previous_normalized,
                    config=self.env_config.observation,
                )
                previous_normalized = obs_result.normalized

                hpa_status = parse_hpa_v2_status(self._read_hpa_v2_json()) if kind != "fixed2" else None
                pods = get_pod_health(APP_NAMESPACE, POD_LABEL, run=self.kubectl_run)
                restart_counts = {p.name: p.restart_count for p in pods}
                max_mem_mib = get_max_per_pod_memory_mib(PROMETHEUS_URL, APP_NAMESPACE, POD_REGEX)
                request_rates = query_per_pod_request_rate(PROMETHEUS_URL, APP_NAMESPACE)

                row = build_sample_row(
                    scenario=name, timestamp_utc=self.now_fn().isoformat(), elapsed_in_scenario_s=elapsed,
                    obs_result=obs_result, replicas_desired=snapshot.replicas.desired,
                    replicas_ready=snapshot.replicas.ready, hpa_status=hpa_status,
                    pod_restart_counts=restart_counts, pod_request_rates=request_rates,
                    max_per_pod_memory_mib=max_mem_mib,
                )
                self._append_sample(row)
                n_samples += 1

                ready_zero_since, abort_reason = evaluate_safety_abort(
                    snapshot.replicas.ready, ready_zero_since, self.clock(), max_mem_mib,
                )
                if abort_reason is not None:
                    result.aborted = True
                    result.abort_reason = abort_reason
                    break

                remaining_in_cycle = self.sample_interval_s - ((self.clock() - start_mono) - elapsed)
                remaining_in_scenario = duration_s - (self.clock() - start_mono)
                sleep_for = max(0.0, min(remaining_in_cycle, remaining_in_scenario))
                if sleep_for > 0:
                    self.sleep_fn(sleep_for)

            result.n_samples = n_samples
            final_pods = get_pod_health(APP_NAMESPACE, POD_LABEL, run=self.kubectl_run)
            result.restart_deltas = {
                p.name: p.restart_count - baseline_restarts.get(p.name, p.restart_count) for p in final_pods
            }
        finally:
            if handle is not None:
                try:
                    self.orchestrator.stop_episode_traffic(handle, f"v2validation_{name}")
                except Exception:  # noqa: BLE001
                    pass
                try:
                    self.orchestrator.force_cleanup()
                except Exception:  # noqa: BLE001
                    pass
            if lock_acquired is not None:
                try:
                    release_controller(lock_acquired)
                except Exception:  # noqa: BLE001
                    pass
            if hpa_removed_for_scenario:
                try:
                    self.backend._kubectl.restore_hpa()
                except Exception:  # noqa: BLE001
                    pass
            result.ended_utc = self.now_fn().isoformat()

        return result

    # -- final cleanup (always) ------------------------------------------------

    def _final_cleanup(self) -> None:
        for controller in (CONTROLLER_PPO, CONTROLLER_CPU_MEMORY_HPA):
            try:
                release_controller(controller)
            except Exception:  # noqa: BLE001
                pass
        try:
            self.orchestrator.force_cleanup()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.backend._kubectl.restore_hpa()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.backend.set_desired_replicas(self.env_config.initial_replicas)
            self.backend._kubectl.wait_for_replica_state(self.env_config.initial_replicas, DEPLOYMENT_READY_WAIT_S)
        except Exception:  # noqa: BLE001
            pass

    # -- full run ---------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        self._write_status(
            status="RUNNING", scenarios_completed=[], current_scenario=None,
            started_utc=self.now_fn().isoformat(),
        )
        results: List[ScenarioResult] = []
        final_status, reason = "COMPLETE", None
        try:
            for i, scenario in enumerate(self.scenarios):
                if i > 0:
                    self._write_status(
                        status="RUNNING", current_scenario=f"idle_settle_before_{scenario['name']}",
                        scenarios_completed=[r.name for r in results],
                    )
                    self.sleep_fn(self.idle_settle_s)
                res = self.run_scenario(scenario)
                results.append(res)
                self._write_status(
                    status="RUNNING", scenarios_completed=[r.name for r in results], current_scenario=None,
                )
        except Exception as exc:  # noqa: BLE001
            final_status, reason = "FAILED", f"{type(exc).__name__}: {exc}"
        finally:
            self._final_cleanup()
            summary = {"results": [asdict(r) for r in results]}
            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            self._write_status(
                status=final_status, reason=reason, scenarios_completed=[r.name for r in results],
                ended_utc=self.now_fn().isoformat(),
            )
        return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args(argv)

    run_dir = Path(args.output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runner = ValidationRunner(output_dir=run_dir)
    try:
        runner.run()
    except ControllerConflictError as exc:
        print(f"Refusing to start validation: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
