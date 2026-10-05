"""autoscaler/src/rl_env/types.py

Shared, backend-agnostic dataclasses for the RL environment (Phase 15B.1).
No Kubernetes/Prometheus dependency -- these types are the contract between
`environment.KubernetesAutoscalerEnv` and any `backend.DynamicsBackend`
implementation (scripted, future empirical simulator, or future live
Kubernetes/Prometheus backend).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Action encoding -- FROZEN, immutable (docs/rl_mdp_spec.md sec.9). Never
# rely on Python enum declaration ordering for this; it is a plain dict.
ACTION_MAP = {0: -1, 1: 0, 2: 1}
ACTION_NAMES = {0: "SCALE_DOWN_ONE", 1: "HOLD", 2: "SCALE_UP_ONE"}

# action_override_reason enum -- FROZEN (docs/rl_mdp_spec.md sec.13, CORR-27).
OVERRIDE_NONE = "none"
OVERRIDE_REPLICA_BOUND_CLAMP = "replica_bound_clamp"
OVERRIDE_PENDING_SCALE_GATE = "pending_scale_gate"
OVERRIDE_SCALE_DOWN_COOLDOWN = "scale_down_cooldown"
OVERRIDE_STALE_OR_MISSING_METRIC = "stale_or_missing_metric"
OVERRIDE_PROMETHEUS_UNAVAILABLE = "prometheus_unavailable"
OVERRIDE_POLICY_INFERENCE_FAILURE = "policy_inference_failure"
OVERRIDE_K8S_API_FAILURE = "k8s_api_failure"

ALL_OVERRIDE_REASONS = (
    OVERRIDE_NONE,
    OVERRIDE_REPLICA_BOUND_CLAMP,
    OVERRIDE_PENDING_SCALE_GATE,
    OVERRIDE_SCALE_DOWN_COOLDOWN,
    OVERRIDE_STALE_OR_MISSING_METRIC,
    OVERRIDE_PROMETHEUS_UNAVAILABLE,
    OVERRIDE_POLICY_INFERENCE_FAILURE,
    OVERRIDE_K8S_API_FAILURE,
)

# latency_observation_state enum -- FROZEN (docs/rl_mdp_spec.md sec.6, CORR-27).
LATENCY_STATE_OBSERVED = "observed"
LATENCY_STATE_NO_TRAFFIC = "no_traffic"
LATENCY_STATE_INVALID = "invalid"


@dataclass(frozen=True)
class RawMetricSnapshot:
    """One decision-step's worth of raw, backend-supplied metrics -- already
    aggregated to a single per-decision-step value (docs/rl_mdp_spec.md
    sec.2: a live backend would average Prometheus samples in (t-30,t] into
    this same shape; a scripted/test backend supplies it directly).

    `None` for any field means that metric was not obtained this step
    (query failure / absent series) -- NEVER fabricated as 0.0 by a
    backend. `target_up=False` models the whole scrape target being down
    (CORR-9); `query_error=True` models an individual query failure while
    the target itself is reachable.
    """

    request_rate_rps: Optional[float]
    error_rate_rps: Optional[float]
    latency_p95_ms: Optional[float]
    requests_in_progress: Optional[float]
    cpu_cores: Optional[float]
    memory_bytes: Optional[float]
    target_up: bool = True
    query_error: bool = False


@dataclass(frozen=True)
class ReplicaState:
    """The four distinct replica-count concepts (docs/rl_mdp_spec.md sec.10,
    CORR-25's evidence hierarchy) -- never conflated."""

    desired: int
    current: int
    ready: int
    available: int


@dataclass(frozen=True)
class BackendSnapshot:
    """Everything the environment needs from a DynamicsBackend at one point
    in logical decision time."""

    time_s: float
    metrics: RawMetricSnapshot
    replicas: ReplicaState


@dataclass(frozen=True)
class GatingResult:
    """Output of the V2 action-gating decision (gating.py)."""

    effective_delta: int
    reason: str  # one of ALL_OVERRIDE_REASONS
    bound_clamped: bool
    pending_scale: bool


@dataclass(frozen=True)
class ValidityResult:
    """Output of observation-validity determination (observation.py)."""

    sample_valid: bool
    override_reason: Optional[str]  # None if sample_valid, else a telemetry-failure reason
    latency_observation_state: str  # one of LATENCY_STATE_*
