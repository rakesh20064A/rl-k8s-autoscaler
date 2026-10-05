"""autoscaler/src/rl_env/observation.py

Frozen V2 observation construction (docs/rl_mdp_spec.md sec.4-8, CORR-27).
Pure functions/small stateful trackers -- no Kubernetes/Prometheus/backend
dependency beyond the RawMetricSnapshot/ReplicaState types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .types import (
    LATENCY_STATE_INVALID,
    LATENCY_STATE_NO_TRAFFIC,
    LATENCY_STATE_OBSERVED,
    OVERRIDE_PROMETHEUS_UNAVAILABLE,
    OVERRIDE_STALE_OR_MISSING_METRIC,
    RawMetricSnapshot,
    ReplicaState,
    ValidityResult,
)

OBSERVATION_SIZE = 11
FEATURE_NAMES = (
    "cpu_per_replica",
    "cpu_per_replica_delta",
    "memory_per_replica",
    "in_progress_per_replica",
    "request_rate_norm",
    "request_rate_delta",
    "p95_latency_norm",
    "p95_latency_delta",
    "error_rate",
    "ready_replica_norm",
    "seconds_since_last_scale_norm",
)

CPU_REQUEST_CORES = 0.100
MEMORY_REQUEST_BYTES = 67108864  # 64 MiB


@dataclass(frozen=True)
class ObservationConfig:
    """Raw clip ranges -- FROZEN v2 values (docs/rl_mdp_spec.md sec.7-8).
    Constructed with these exact defaults; `config.py`'s
    `load_env_config_from_yaml()` cross-checks them against
    configs/rl.yaml at load time so this module never silently drifts from
    the authoritative YAML contract."""

    cpu_raw_min: float = 0.0
    cpu_raw_max: float = 5.0
    cpu_delta_raw_min: float = -5.0
    cpu_delta_raw_max: float = 5.0
    memory_raw_min: float = 0.0
    memory_raw_max: float = 4.0
    in_progress_raw_min: float = 0.0
    in_progress_raw_max: float = 20.0
    request_rate_raw_min: float = 0.0
    request_rate_raw_max: float = 100.0
    request_rate_delta_raw_min: float = -100.0
    request_rate_delta_raw_max: float = 100.0
    latency_raw_min_ms: float = 0.0
    latency_raw_max_ms: float = 2000.0
    latency_delta_raw_min_ms: float = -2000.0
    latency_delta_raw_max_ms: float = 2000.0
    min_replicas: int = 1
    max_replicas: int = 8
    seconds_since_scale_cap_s: float = 300.0
    ewma_span_steps: int = 3


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _norm(value: float, lo: float, hi: float) -> float:
    """Clip then map [lo,hi] -> [0,1]."""
    return (_clip(value, lo, hi) - lo) / (hi - lo)


def _norm_signed(value: float, abs_max: float) -> float:
    """Clip to [-abs_max, abs_max] then map to [-1,1]."""
    return _clip(value, -abs_max, abs_max) / abs_max


class EwmaTracker:
    """Backward-looking-only EWMA/delta tracker (docs/rl_mdp_spec.md sec.5).

    alpha = 2 / (span_steps + 1). `update(step_value)`:
      - step_value is None (invalid step): the EWMA is NOT updated; returns
        None (caller must hold the previous normalized delta).
      - first valid step_value seen: ewma initialized to it; delta = 0.0.
      - subsequent valid step_value: delta = step_value - ewma_(t-1),
        THEN ewma is updated to alpha*step_value + (1-alpha)*ewma_(t-1).
    """

    def __init__(self, span_steps: int = 3) -> None:
        self.alpha = 2.0 / (span_steps + 1)
        self._ewma: Optional[float] = None

    @property
    def initialized(self) -> bool:
        return self._ewma is not None

    def update(self, step_value: Optional[float]) -> Optional[float]:
        if step_value is None:
            return None
        if self._ewma is None:
            self._ewma = step_value
            return 0.0
        delta = step_value - self._ewma
        self._ewma = self.alpha * step_value + (1.0 - self.alpha) * self._ewma
        return delta


def determine_validity(metrics: RawMetricSnapshot) -> ValidityResult:
    """Frozen V2 validity rule (docs/rl_mdp_spec.md sec.3/6, CORR-9, CORR-27).

    Priority: target-down > query-error > missing infra telemetry >
    missing request-rate > (request_rate==0 -> no_traffic, valid) >
    (request_rate>0 and latency missing -> invalid) > observed/valid.
    `error_rate_rps` being None while request_rate>0 is NOT itself
    invalidating -- per the project's existing null-vs-zero convention
    (CORR-9 sec.7), it is treated as a genuine 0.0 by the caller, not
    checked here.
    """
    if not metrics.target_up:
        return ValidityResult(False, OVERRIDE_PROMETHEUS_UNAVAILABLE, LATENCY_STATE_INVALID)
    if metrics.query_error:
        return ValidityResult(False, OVERRIDE_STALE_OR_MISSING_METRIC, LATENCY_STATE_INVALID)
    if metrics.cpu_cores is None or metrics.memory_bytes is None or metrics.requests_in_progress is None:
        return ValidityResult(False, OVERRIDE_STALE_OR_MISSING_METRIC, LATENCY_STATE_INVALID)
    if metrics.request_rate_rps is None:
        return ValidityResult(False, OVERRIDE_STALE_OR_MISSING_METRIC, LATENCY_STATE_INVALID)
    if metrics.request_rate_rps == 0.0:
        if metrics.latency_p95_ms is None:
            return ValidityResult(True, None, LATENCY_STATE_NO_TRAFFIC)
        return ValidityResult(True, None, LATENCY_STATE_OBSERVED)
    # request_rate_rps > 0
    if metrics.latency_p95_ms is None:
        return ValidityResult(False, OVERRIDE_STALE_OR_MISSING_METRIC, LATENCY_STATE_INVALID)
    return ValidityResult(True, None, LATENCY_STATE_OBSERVED)


def effective_latency_ms(metrics: RawMetricSnapshot, latency_state: str) -> Optional[float]:
    """The latency value to feed into normalization/EWMA: a genuine 0.0 for
    no_traffic, the real reading for observed, or None for invalid (caller
    must not use None -- an invalid step never reaches normalization)."""
    if latency_state == LATENCY_STATE_NO_TRAFFIC:
        return 0.0
    if latency_state == LATENCY_STATE_OBSERVED:
        return metrics.latency_p95_ms
    return None


def per_replica_ratios(
    metrics: RawMetricSnapshot, ready_replicas: int
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(cpu_per_replica_frac_request, memory_per_replica_frac_request,
    in_progress_per_replica) -- all None if ready_replicas==0 (sec.4's
    documented fallback: never divide by zero, never fabricate)."""
    if ready_replicas <= 0:
        return None, None, None
    cpu = None if metrics.cpu_cores is None else metrics.cpu_cores / ready_replicas / CPU_REQUEST_CORES
    mem = None if metrics.memory_bytes is None else metrics.memory_bytes / ready_replicas / MEMORY_REQUEST_BYTES
    inprog = None if metrics.requests_in_progress is None else metrics.requests_in_progress / ready_replicas
    return cpu, mem, inprog


@dataclass
class HistoryState:
    """The three EWMA trackers required by V2 (docs/rl_mdp_spec.md sec.5)."""

    cpu: EwmaTracker
    request_rate: EwmaTracker
    latency: EwmaTracker

    @classmethod
    def new(cls, span_steps: int = 3) -> "HistoryState":
        return cls(EwmaTracker(span_steps), EwmaTracker(span_steps), EwmaTracker(span_steps))


@dataclass(frozen=True)
class ObservationBuildResult:
    normalized: List[float]  # length 11, matches FEATURE_NAMES order
    raw: dict  # feature_name -> raw (pre-normalization) value or None
    validity: ValidityResult
    ready_replicas_used: int


def build_observation(
    metrics: RawMetricSnapshot,
    replicas: ReplicaState,
    seconds_since_last_effective_action: float,
    history: HistoryState,
    previous_normalized: Optional[List[float]],
    config: ObservationConfig = ObservationConfig(),
) -> ObservationBuildResult:
    """Build the frozen 11-element V2 observation for one decision step.

    `previous_normalized` (the last successfully-returned normalized
    vector, or None at episode start) is used to hold indices whose raw
    input is missing this step, per sec.7/8's "never fabricate, hold
    previous normalized value" rule. EWMA trackers are updated (or
    deliberately skipped) as a side effect.
    """
    validity = determine_validity(metrics)
    ready = replicas.ready

    if previous_normalized is None:
        # Episode-start fallback: raw-range midpoint (0.5 after normalization
        # for symmetric-around-midpoint ranges) per sec.7.
        previous_normalized = [0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5]

    out = list(previous_normalized)
    raw: dict = {name: None for name in FEATURE_NAMES}

    if not validity.sample_valid:
        # Structurally invalid step: hold every level/trend feature except
        # the two controller-tracked ones (ready_replica_norm always comes
        # straight from the Deployment's own replica-count fields, which
        # are a SEPARATE Kubernetes/kube-state-metrics query from whichever
        # application-level metric failed, and seconds_since_last_scale is
        # purely controller-internal state) -- neither depends on the
        # failed application telemetry, so both are still safe to report.
        # EWMA trackers are NOT updated (per sec.5: never update from an
        # invalid step). error_rate (index 8) is likewise held at its
        # previous value -- it is only well-defined relative to a valid
        # request_rate reading, which this step does not have.
        out[9] = _norm(float(ready), config.min_replicas, config.max_replicas)
        out[10] = _norm(seconds_since_last_effective_action, 0.0, config.seconds_since_scale_cap_s)
        return ObservationBuildResult(out, raw, validity, ready)

    cpu_raw, mem_raw, inprog_raw = per_replica_ratios(metrics, ready)
    latency_ms = effective_latency_ms(metrics, validity.latency_observation_state)
    request_rate = metrics.request_rate_rps
    error_rate = 0.0 if metrics.error_rate_rps is None else (
        metrics.error_rate_rps / request_rate if request_rate else 0.0
    )

    raw.update(
        cpu_per_replica=cpu_raw,
        memory_per_replica=mem_raw,
        in_progress_per_replica=inprog_raw,
        request_rate_norm=request_rate,
        p95_latency_norm=latency_ms,
        error_rate=error_rate,
    )

    # Index 0/1: cpu_per_replica (+ delta). None if ready==0 (sec.4).
    if cpu_raw is not None:
        out[0] = _norm(cpu_raw, config.cpu_raw_min, config.cpu_raw_max)
        delta = history.cpu.update(cpu_raw)
        if delta is not None:
            out[1] = _norm_signed(delta, config.cpu_delta_raw_max)
    # else: hold previous out[0]/out[1], EWMA untouched.

    # Index 2: memory_per_replica.
    if mem_raw is not None:
        out[2] = _norm(mem_raw, config.memory_raw_min, config.memory_raw_max)

    # Index 3: in_progress_per_replica.
    if inprog_raw is not None:
        out[3] = _norm(inprog_raw, config.in_progress_raw_min, config.in_progress_raw_max)

    # Index 4/5: request_rate_norm (+ delta). request_rate is only None if
    # sample invalid, which is already excluded by the branch above.
    out[4] = _norm(request_rate, config.request_rate_raw_min, config.request_rate_raw_max)
    delta = history.request_rate.update(request_rate)
    if delta is not None:
        out[5] = _norm_signed(delta, config.request_rate_delta_raw_max)

    # Index 6/7: p95_latency_norm (+ delta). latency_ms is a real value
    # (0.0 for no_traffic, or the genuine reading) whenever sample_valid.
    out[6] = _norm(latency_ms, config.latency_raw_min_ms, config.latency_raw_max_ms)
    delta = history.latency.update(latency_ms)
    if delta is not None:
        out[7] = _norm_signed(delta, config.latency_delta_raw_max_ms)

    # Index 8: error_rate (already a fraction in [0,1]).
    out[8] = _clip(error_rate, 0.0, 1.0)

    # Index 9: ready_replica_norm.
    out[9] = _norm(float(ready), config.min_replicas, config.max_replicas)

    # Index 10: seconds_since_last_scale_norm.
    out[10] = _norm(seconds_since_last_effective_action, 0.0, config.seconds_since_scale_cap_s)

    return ObservationBuildResult(out, raw, validity, ready)
