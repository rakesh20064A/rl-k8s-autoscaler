"""autoscaler/src/rl_env/backend.py

The pluggable dynamics-backend interface (Phase 15B.1). The environment
(`environment.KubernetesAutoscalerEnv`) is written entirely against this
Protocol and never hard-codes Kubernetes or Prometheus operations directly.

Three intended implementations, only the first of which exists today:
  1. `scripted_backend.ScriptedBackend` -- deterministic, TEST_ONLY_SYNTHETIC_METRICS,
     built in this phase to validate environment mechanics.
  2. A future empirical calibrated simulator backend (NOT built in this
     phase -- requires a separate, evidence-based dynamics-calibration
     phase per docs/experimental_design_corrections.md CORR-29).
  3. A future live Kubernetes/Prometheus backend (NOT built in this phase).

Replacing the backend must never require changing observation.py, reward.py,
gating.py, or environment.py.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import BackendSnapshot


@runtime_checkable
class DynamicsBackend(Protocol):
    """Minimal interface a dynamics source must implement.

    Time is logical decision time in seconds, starting at 0.0 at
    `reset()`. Implementations are free to represent wall-clock time,
    simulated time, or purely synthetic scripted time -- the environment
    never assumes anything about how time is produced beyond the ordering
    and duration semantics of `advance()`.
    """

    def reset(self) -> BackendSnapshot:
        """Reset to the standardized episode-start state: Deployment at
        `initial_replicas`, Ready confirmed, post-stabilization metrics,
        logical time = 0.0. Must be idempotent-safe to call multiple times
        (e.g. once per Gymnasium episode)."""
        ...

    def current_snapshot(self) -> BackendSnapshot:
        """The backend's current state without advancing time. Provided
        for introspection/testing; the environment itself relies on the
        return values of `reset()`/`advance()` rather than calling this
        mid-step, to avoid any risk of reading a stale/inconsistent view."""
        ...

    def set_desired_replicas(self, desired: int) -> None:
        """Apply an effective (already gated/bound-clamped) replica-count
        change, effective immediately at the backend's current logical
        time. Must implement the CORR-25/V2-frozen asymmetry: on scale-up,
        `desired` and `current` update immediately but `ready`/`available`
        only catch up after the backend's own configured readiness delay
        (resolved during a later `advance()` call); on scale-down, all four
        fields update immediately. Must clamp to the backend's own
        configured [min,max] as a defensive measure, though the caller
        (the environment's gating layer) is expected to never request an
        out-of-bounds value in the first place."""
        ...

    def advance(self, dt_seconds: float) -> BackendSnapshot:
        """Advance logical time by exactly `dt_seconds`, resolving any
        scheduled replica transitions that fall within `(t, t+dt_seconds]`,
        and return the resulting snapshot at `t+dt_seconds`. Must never
        expose information from beyond the new time -- no future leakage."""
        ...
