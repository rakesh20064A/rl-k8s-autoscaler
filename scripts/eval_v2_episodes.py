#!/usr/bin/env python3
"""scripts/eval_v2_episodes.py

V2 Phase 5C-A: the REAL, unified per-arm episode functions for the V2
evaluation campaign (CORR-60/CORR-61) -- replaces the fake episode
functions `scripts/evaluate_v2_campaign.py`'s `build_dry_run_context()`
used in Phase 5B. A NEW file -- imports and reuses, unmodified, every one
of: `autoscaler.src.rl_env.environment_v2.KubernetesAutoscalerEnvV2`,
`autoscaler.src.rl_env.safe_ppo_controller.run_safe_ppo_step`,
`autoscaler.src.rl_env.safe_inference.load_ppo_model_safe`,
`autoscaler.src.rl_env.observation.build_observation`/`HistoryState`,
`autoscaler.src.rl_env.live_backend.LiveKubernetesBackend`,
`autoscaler.src.rl_env.controller_exclusivity` (read-only, including via
`getattr` for the not-yet-existing Phase 5C cpu-only constant),
`autoscaler.src.rl_env.config_v2.load_env_config_v2_from_yaml`, and
`scripts.v2_pilot_safety`'s memory/pod-health helpers. None of these 43
FREEZE_MANIFEST.json files is edited.

Every arm produces the SAME `EpisodeSample` schema, one per decision
interval (30s, `experiment.yaml`'s `rl.control_interval_seconds`), built
from the SAME `build_observation()` call training itself uses (the PPO
arm gets this for free from `KubernetesAutoscalerEnvV2.step()`'s own
`info["raw_observation"]`/normalized-obs; the two HPA arms call
`build_observation()` directly against `LiveKubernetesBackend.reset()`/
`.advance()`'s own `BackendSnapshot`, in a loop that NEVER calls
`backend.set_desired_replicas()` -- `advance()` only sleeps and re-reads
cluster state, exactly the same observe-only mechanism
`evaluate_ppo_vs_hpa_v2.run_hpa_v2_episode()` already relies on).

CORR-61 (this phase): the safety-termination thresholds used here are the
SAME ones training already uses (`scripts.v2_pilot_safety.
MEMORY_SAFETY_THRESHOLD_MIB`/`SUSTAINED_BREACH_SAMPLES`, and the PPO
environment's own `max_consecutive_invalid_steps`) -- stated here as an
explicit evaluation-time clarification of the pre-registration, decided
BEFORE any evaluation (or further training) result exists.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
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
    acquire_controller,
    release_controller,
)
from autoscaler.src.rl_env.environment_v2 import KubernetesAutoscalerEnvV2  # noqa: E402
from autoscaler.src.rl_env.live_backend import LiveKubernetesBackend  # noqa: E402
from autoscaler.src.rl_env.observation import FEATURE_NAMES, HistoryState, build_observation  # noqa: E402
from autoscaler.src.rl_env.safe_inference import load_ppo_model_safe  # noqa: E402
from autoscaler.src.rl_env.safe_ppo_controller import run_safe_ppo_step  # noqa: E402

import scripts.v2_pilot_safety as pilot_safety  # noqa: E402

ACTION_NAMES = {0: "SCALE_DOWN", 1: "HOLD", 2: "SCALE_UP"}


class ToolingFailureError(RuntimeError):
    """A tooling/infrastructure failure (route/Prometheus/generator/K8s
    API) -- no scientific outcome; the campaign runner retries exactly
    once per the pre-registration's validity rules."""


class SafetyTerminationError(RuntimeError):
    """A safety-triggered episode termination -- a COUNTED outcome, never
    retried."""


class CpuOnlyArmNotYetSupportedError(RuntimeError):
    """V2 Phase 5B/5C-A: the hpa_v2_cpu_only arm has no registered
    controller identity in controller_exclusivity.py yet. The exact
    one-line fix Phase 5C needs is named in this message."""

    def __init__(self) -> None:
        super().__init__(
            "hpa_v2_cpu_only arm requires a Phase 5C controller_exclusivity.py "
            "amendment before it can run against a live cluster: "
            "autoscaler/src/rl_env/controller_exclusivity.py: add "
            "CONTROLLER_CPU_ONLY_HPA_V2 = 'cpu_only_hpa_v2' to ALL_CONTROLLERS and "
            "HPA_NAME_BY_CONTROLLER[CONTROLLER_CPU_ONLY_HPA_V2] = 'demo-app-hpa-v2-cpu' "
            "(mirroring the existing CONTROLLER_CPU_MEMORY_HPA entry)."
        )


def resolve_cpu_only_controller_identity() -> str:
    """`getattr` against `controller_exclusivity` for the not-yet-existing
    Phase 5C constant -- fails closed (raises
    `CpuOnlyArmNotYetSupportedError`) if absent; once Phase 5C adds
    `CONTROLLER_CPU_ONLY_HPA_V2`, this starts succeeding with NO code
    change here."""
    import autoscaler.src.rl_env.controller_exclusivity as ce

    controller_id = getattr(ce, "CONTROLLER_CPU_ONLY_HPA_V2", None)
    if controller_id is None:
        raise CpuOnlyArmNotYetSupportedError()
    return controller_id


@dataclass
class EpisodeSample:
    """One decision interval's worth of telemetry, identical in shape for
    every arm (fields that don't apply to a given arm are simply None /
    empty)."""

    decision_index: int
    t_s: float
    timestamp_utc: str
    profile: str
    arm_id: str

    normalized_observation: List[float]
    raw_observation: Dict[str, Optional[float]]
    sample_valid: bool

    p95_latency_ms: Optional[float]
    error_rate: Optional[float]
    request_rate_rps: Optional[float]

    desired_replicas: Optional[int]
    ready_replicas: Optional[int]

    max_per_pod_memory_mib: Optional[float]
    pod_restart_counts: Dict[str, int] = field(default_factory=dict)

    hpa_status: Optional[Dict[str, Any]] = None
    ppo_decision: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_index": self.decision_index, "t_s": self.t_s, "timestamp_utc": self.timestamp_utc,
            "profile": self.profile, "arm_id": self.arm_id, "normalized_observation": self.normalized_observation,
            "raw_observation": self.raw_observation, "sample_valid": self.sample_valid,
            "p95_latency_ms": self.p95_latency_ms, "error_rate": self.error_rate,
            "request_rate_rps": self.request_rate_rps, "desired_replicas": self.desired_replicas,
            "ready_replicas": self.ready_replicas, "max_per_pod_memory_mib": self.max_per_pod_memory_mib,
            "pod_restart_counts": self.pod_restart_counts, "hpa_status": self.hpa_status,
            "ppo_decision": self.ppo_decision,
        }


def _check_restart_and_zero_ready(pods, baseline_restart_counts: Dict[str, int]) -> Optional[str]:
    for pod in pods:
        baseline = baseline_restart_counts.get(pod.name, pod.restart_count)
        if pod.restart_count > baseline:
            return f"pod_restart_detected:{pod.name}"
    if pods and not any(p.ready for p in pods):
        return "zero_ready_replicas"
    return None


def _check_memory_breach(
    max_pod_mib: Optional[float], consecutive_breaches: int,
) -> "tuple[int, Optional[str]]":
    if max_pod_mib is None:
        return consecutive_breaches, None
    if max_pod_mib >= pilot_safety.MEMORY_SAFETY_THRESHOLD_MIB:
        consecutive_breaches += 1
    else:
        consecutive_breaches = 0
    if consecutive_breaches >= pilot_safety.SUSTAINED_BREACH_SAMPLES:
        return consecutive_breaches, f"memory_safety_threshold_breached:{max_pod_mib:.1f}MiB_per_pod"
    return consecutive_breaches, None


# ---------------------------------------------------------------------------
# Arm: ppo_v2
# ---------------------------------------------------------------------------


def run_ppo_v2_eval_episode(
    model_path: Path,
    epsilon: float,
    profile_name: str,
    episode_duration_s: float,
    episode_id: str,
    namespace: str = "autoscaler-app",
    pod_label_selector: str = "app=demo-app",
    pod_regex: str = "demo-app-.*",
    prometheus_url: Optional[str] = None,
    model_loader: Callable[[Path], Any] = load_ppo_model_safe,
    backend_factory: Optional[Callable[..., Any]] = None,
    pod_health_fn: Callable[..., Any] = pilot_safety.get_pod_health,
    max_pod_memory_fn: Callable[..., Any] = None,
    on_sample: Optional[Callable[["EpisodeSample"], None]] = None,
) -> Dict[str, Any]:
    """The ppo_v2 arm: `KubernetesAutoscalerEnvV2` + `run_safe_ppo_step`
    per decision, requires BOTH HPAs absent (enforced by
    `acquire_controller(CONTROLLER_PPO, ...)`, unchanged from
    `evaluate_ppo_vs_hpa_v2.run_live_v2_episode()`). Raises
    `SafetyTerminationError` on a memory/restart/zero-ready breach or on
    the env's own 4-consecutive-invalid-step termination; raises
    `ToolingFailureError` on any other exception (route/Prometheus/K8s API
    failure).

    `on_sample` (V2 Phase 5E, additive, default None -> no behavior change
    for any existing caller): invoked with each `EpisodeSample` the moment
    it is built, BEFORE any safety check for that decision -- lets a
    streaming caller (e.g. `scripts/run_demo_episode_v2.py`) write/display
    a sample even on the decision that ultimately trips a safety
    termination."""
    env_config = load_env_config_v2_from_yaml()
    if prometheus_url is None:
        from scripts.v2_pilot_safety import PROMETHEUS_URL

        prometheus_url = PROMETHEUS_URL
    if max_pod_memory_fn is None:
        max_pod_memory_fn = pilot_safety.get_max_per_pod_memory_mib

    backend_factory = backend_factory or LiveKubernetesBackend
    backend = backend_factory(
        prometheus_url=prometheus_url, hpa_name=V2_HPA_NAME, hpa_manifest_path=V2_HPA_MANIFEST_PATH,
        min_replicas=env_config.gating.min_replicas, initial_replicas=env_config.initial_replicas,
    )

    try:
        baseline_pods = pod_health_fn(namespace, pod_label_selector)
        baseline_restart_counts = {p.name: p.restart_count for p in baseline_pods}
    except Exception as exc:  # noqa: BLE001
        raise ToolingFailureError(f"failed to read baseline pod health: {exc}") from exc

    acquire_controller(CONTROLLER_PPO, backend._kubectl)
    try:
        model = model_loader(model_path)
        if model is None:
            raise ToolingFailureError(f"failed to load PPO model at {model_path}")

        env = KubernetesAutoscalerEnvV2(backend=backend, episode_duration_s=episode_duration_s, config=env_config)
        try:
            obs, info = env.reset()
        except Exception as exc:  # noqa: BLE001
            raise ToolingFailureError(f"env.reset() failed: {exc}") from exc

        decision_time_raw_observation = {
            "p95_latency_ms": (info.get("raw_observation") or {}).get("p95_latency_norm"),
            "error_rate": (info.get("raw_observation") or {}).get("error_rate"),
            "cpu_per_replica": (info.get("raw_observation") or {}).get("cpu_per_replica"),
            "memory_per_replica": (info.get("raw_observation") or {}).get("memory_per_replica"),
        }
        reward_config = env_config.reward

        n_steps = int(round(episode_duration_s / env_config.decision_interval_s))
        samples: List[EpisodeSample] = []
        consecutive_breaches = 0
        pod_restarts_total = 0

        for decision_index in range(n_steps):
            try:
                obs, _reward, terminated, truncated, info, decision_record = run_safe_ppo_step(
                    env, model, obs, model_identifier=str(model_path), epsilon=epsilon,
                    decision_time_raw_observation=decision_time_raw_observation, reward_config=reward_config,
                )
            except Exception as exc:  # noqa: BLE001
                raise ToolingFailureError(f"run_safe_ppo_step failed at decision {decision_index}: {exc}") from exc
            decision_time_raw_observation = {
                "p95_latency_ms": (info.get("raw_observation") or {}).get("p95_latency_norm"),
                "error_rate": (info.get("raw_observation") or {}).get("error_rate"),
                "cpu_per_replica": (info.get("raw_observation") or {}).get("cpu_per_replica"),
                "memory_per_replica": (info.get("raw_observation") or {}).get("memory_per_replica"),
            }

            raw = info.get("raw_observation") or {}
            try:
                pods = pod_health_fn(namespace, pod_label_selector)
            except Exception as exc:  # noqa: BLE001
                raise ToolingFailureError(f"pod health read failed at decision {decision_index}: {exc}") from exc
            restart_reason = _check_restart_and_zero_ready(pods, baseline_restart_counts)
            pod_restarts_total += sum(max(0, p.restart_count - baseline_restart_counts.get(p.name, p.restart_count)) for p in pods)

            try:
                max_pod_mib = max_pod_memory_fn(prometheus_url, namespace, pod_regex)
            except Exception as exc:  # noqa: BLE001
                raise ToolingFailureError(f"memory query failed at decision {decision_index}: {exc}") from exc
            consecutive_breaches, memory_reason = _check_memory_breach(max_pod_mib, consecutive_breaches)

            sample = EpisodeSample(
                decision_index=decision_index, t_s=decision_index * env_config.decision_interval_s,
                timestamp_utc=datetime.now(timezone.utc).isoformat(), profile=profile_name, arm_id="ppo_v2",
                normalized_observation=list(obs), raw_observation={name: raw.get(name) for name in FEATURE_NAMES},
                sample_valid=bool(info.get("sample_valid", True)), p95_latency_ms=raw.get("p95_latency_norm"),
                error_rate=raw.get("error_rate"), request_rate_rps=raw.get("request_rate_norm"),
                desired_replicas=info.get("deployment_current_replicas_after"),
                ready_replicas=info.get("ready_replicas_after"), max_per_pod_memory_mib=max_pod_mib,
                pod_restart_counts={p.name: p.restart_count for p in pods},
                ppo_decision=decision_record.to_dict(),
            )
            samples.append(sample)
            if on_sample is not None:
                on_sample(sample)

            if restart_reason is not None:
                raise SafetyTerminationError(restart_reason)
            if memory_reason is not None:
                raise SafetyTerminationError(memory_reason)
            if terminated or truncated:
                reason = info.get("termination_reason") or "env_terminated_or_truncated"
                if terminated:
                    raise SafetyTerminationError(f"ppo_env_terminated:{reason}")
                break  # truncated (natural episode-duration end) -- not a safety event

        return {
            "episode_id": episode_id, "arm_id": "ppo_v2", "profile": profile_name,
            "samples": [s.to_dict() for s in samples], "pod_restarts_total": pod_restarts_total,
        }
    finally:
        release_controller(CONTROLLER_PPO)


# ---------------------------------------------------------------------------
# Arm: hpa_v2_cpu_memory / hpa_v2_cpu_only (shared implementation)
# ---------------------------------------------------------------------------


def run_hpa_v2_eval_episode(
    arm_id: str,
    hpa_name: str,
    hpa_manifest_path: str,
    controller_id: str,
    profile_name: str,
    episode_duration_s: float,
    episode_id: str,
    namespace: str = "autoscaler-app",
    pod_label_selector: str = "app=demo-app",
    pod_regex: str = "demo-app-.*",
    prometheus_url: Optional[str] = None,
    backend_factory: Optional[Callable[..., Any]] = None,
    pod_health_fn: Callable[..., Any] = pilot_safety.get_pod_health,
    max_pod_memory_fn: Callable[..., Any] = None,
    on_sample: Optional[Callable[["EpisodeSample"], None]] = None,
) -> Dict[str, Any]:
    """Shared implementation for BOTH HPA arms -- the only per-arm
    difference is the HPA identity (`hpa_name`/`hpa_manifest_path`/
    `controller_id`). Uses `LiveKubernetesBackend.reset()` (one, normal
    episode-start convergence to the floor replica count -- the SAME
    convergence point the HPA's own floor would also settle to, not an
    ongoing actuation) then `advance()` in a loop, which NEVER calls
    `backend.set_desired_replicas()` -- observe-only, exactly the pattern
    `evaluate_ppo_vs_hpa_v2.run_hpa_v2_episode()` already relies on.
    `build_observation()` (the SAME training-contract function) is called
    directly against each `BackendSnapshot` so this arm's samples are
    observation-comparable to the ppo_v2 arm's own."""
    env_config = load_env_config_v2_from_yaml()
    if prometheus_url is None:
        from scripts.v2_pilot_safety import PROMETHEUS_URL

        prometheus_url = PROMETHEUS_URL
    if max_pod_memory_fn is None:
        max_pod_memory_fn = pilot_safety.get_max_per_pod_memory_mib

    backend_factory = backend_factory or LiveKubernetesBackend
    backend = backend_factory(
        prometheus_url=prometheus_url, hpa_name=hpa_name, hpa_manifest_path=hpa_manifest_path,
        min_replicas=env_config.gating.min_replicas, initial_replicas=env_config.initial_replicas,
    )

    try:
        baseline_pods = pod_health_fn(namespace, pod_label_selector)
        baseline_restart_counts = {p.name: p.restart_count for p in baseline_pods}
    except Exception as exc:  # noqa: BLE001
        raise ToolingFailureError(f"failed to read baseline pod health: {exc}") from exc

    acquire_controller(controller_id, backend._kubectl)
    try:
        try:
            backend._kubectl.restore_hpa()  # kubectl apply -f hpa_manifest_path (idempotent)
            applied_names = backend._kubectl.list_hpa_names()
            if hpa_name not in applied_names:
                raise ToolingFailureError(f"failed to apply {hpa_name!r}: live HPA names are {applied_names}")
        except ToolingFailureError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolingFailureError(f"failed to apply/verify {hpa_name!r}: {exc}") from exc

        try:
            snapshot = backend.reset()
        except Exception as exc:  # noqa: BLE001
            raise ToolingFailureError(f"backend.reset() failed: {exc}") from exc

        history = HistoryState.new()
        previous_normalized: Optional[List[float]] = None
        n_steps = int(round(episode_duration_s / env_config.decision_interval_s))
        samples: List[EpisodeSample] = []
        consecutive_breaches = 0
        pod_restarts_total = 0

        for decision_index in range(n_steps):
            build_result = build_observation(
                metrics=snapshot.metrics, replicas=snapshot.replicas,
                seconds_since_last_effective_action=decision_index * env_config.decision_interval_s,
                history=history, previous_normalized=previous_normalized, config=env_config.observation,
            )
            previous_normalized = build_result.normalized

            try:
                pods = pod_health_fn(namespace, pod_label_selector)
            except Exception as exc:  # noqa: BLE001
                raise ToolingFailureError(f"pod health read failed at decision {decision_index}: {exc}") from exc
            restart_reason = _check_restart_and_zero_ready(pods, baseline_restart_counts)
            pod_restarts_total += sum(max(0, p.restart_count - baseline_restart_counts.get(p.name, p.restart_count)) for p in pods)

            try:
                max_pod_mib = max_pod_memory_fn(prometheus_url, namespace, pod_regex)
            except Exception as exc:  # noqa: BLE001
                raise ToolingFailureError(f"memory query failed at decision {decision_index}: {exc}") from exc
            consecutive_breaches, memory_reason = _check_memory_breach(max_pod_mib, consecutive_breaches)

            sample = EpisodeSample(
                decision_index=decision_index, t_s=snapshot.time_s,
                timestamp_utc=datetime.now(timezone.utc).isoformat(), profile=profile_name, arm_id=arm_id,
                normalized_observation=build_result.normalized, raw_observation=build_result.raw,
                sample_valid=build_result.raw.get("request_rate_norm") is not None or snapshot.metrics.target_up,
                p95_latency_ms=snapshot.metrics.latency_p95_ms, error_rate=snapshot.metrics.error_rate_rps,
                request_rate_rps=snapshot.metrics.request_rate_rps, desired_replicas=snapshot.replicas.desired,
                ready_replicas=snapshot.replicas.ready, max_per_pod_memory_mib=max_pod_mib,
                pod_restart_counts={p.name: p.restart_count for p in pods},
                hpa_status={"hpa_name": hpa_name, "applied": True},
            )
            samples.append(sample)
            if on_sample is not None:
                on_sample(sample)

            if restart_reason is not None:
                raise SafetyTerminationError(restart_reason)
            if memory_reason is not None:
                raise SafetyTerminationError(memory_reason)

            if decision_index < n_steps - 1:
                try:
                    snapshot = backend.advance(env_config.decision_interval_s)
                except Exception as exc:  # noqa: BLE001
                    raise ToolingFailureError(f"backend.advance() failed at decision {decision_index}: {exc}") from exc

        return {
            "episode_id": episode_id, "arm_id": arm_id, "profile": profile_name,
            "samples": [s.to_dict() for s in samples], "pod_restarts_total": pod_restarts_total,
        }
    finally:
        try:
            backend._kubectl.restore_hpa()  # re-apply this arm's own HPA on the way out, never deletes it
        finally:
            release_controller(controller_id)


def run_hpa_v2_cpu_memory_eval_episode(profile_name: str, episode_duration_s: float, episode_id: str, **kwargs: Any) -> Dict[str, Any]:
    return run_hpa_v2_eval_episode(
        "hpa_v2_cpu_memory", V2_HPA_NAME, V2_HPA_MANIFEST_PATH, CONTROLLER_CPU_MEMORY_HPA,
        profile_name, episode_duration_s, episode_id, **kwargs,
    )


def run_hpa_v2_cpu_only_eval_episode(profile_name: str, episode_duration_s: float, episode_id: str, **kwargs: Any) -> Dict[str, Any]:
    """Fails closed via `resolve_cpu_only_controller_identity()` BEFORE
    touching the cluster at all, if the Phase 5C controller_exclusivity.py
    amendment has not landed yet."""
    controller_id = resolve_cpu_only_controller_identity()
    cpu_only_hpa_name = "demo-app-hpa-v2-cpu"
    cpu_only_manifest_path = "k8s/hpa/hpa_v2_cpu_only_min2.yaml"
    return run_hpa_v2_eval_episode(
        "hpa_v2_cpu_only", cpu_only_hpa_name, cpu_only_manifest_path, controller_id,
        profile_name, episode_duration_s, episode_id, **kwargs,
    )


# ---------------------------------------------------------------------------
# Adapter: episode result -> analyze_v2_evaluation.RunSeries dict
# ---------------------------------------------------------------------------


def episode_result_to_run_series_dict(
    result: Dict[str, Any], run_id: str, repeat_index: int, decision_interval_s: float = 30.0,
    load_change_times_s: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Converts EITHER arm's `run_*_eval_episode()` result dict directly
    into `analyze_v2_evaluation.normalize_run_series_from_dict()`'s
    expected schema -- the ONE adapter this phase's own report flagged as
    a Phase 5C gap in Phase 5B, now closed."""
    run_samples = []
    for s in result["samples"]:
        ppo_decision = s.get("ppo_decision")
        action = None
        reversal = False
        if ppo_decision is not None:
            action = ppo_decision.get("effective_action_name")
            reversal = bool(ppo_decision.get("extra", {}).get("is_reversal", False))
        run_samples.append({
            "t_s": s["t_s"], "p95_latency_ms": s["p95_latency_ms"], "error_rate": s["error_rate"],
            "memory_per_replica_mib": s["max_per_pod_memory_mib"], "replicas": s["desired_replicas"],
            "action": action, "reversal": reversal, "invalid": not s["sample_valid"],
        })

    return {
        "run_id": run_id, "profile": result["profile"], "arm_id": result["arm_id"], "repeat_index": repeat_index,
        "decision_interval_s": decision_interval_s, "samples": run_samples,
        "pod_restarts": result.get("pod_restarts_total", 0), "load_change_times_s": load_change_times_s or [],
    }
