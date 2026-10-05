"""autoscaler/src/rl_env/gating.py

Frozen V2 action-gating rule (docs/rl_mdp_spec.md sec.12-13, CORR-27).
Pure function -- no state, no I/O. Precedence (matches the task's own
recommended ordering, applied deterministically):

  1. structural/safety invalidity      -- handled by the CALLER (environment.py)
     before this function is even invoked (an invalid step never reaches
     replica-bound/pending/cooldown gating -- it is unconditionally HOLD).
  2. replica-bound handling            -- checked first here: an
     out-of-bounds request is a `replica_bound_clamp` regardless of any
     pending transition or cooldown state.
  3. pending-scale gating              -- checked next: an in-bounds
     request is still deferred while a prior transition has not realized.
  4. scale-down cooldown               -- checked only for scale-down
     requests, only once bound/pending checks have passed.
  5. action executes                   -- scale-up has NO additional
     cooldown beyond pending-scale gating (CORR-27); scale-down executes
     once its cooldown has elapsed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import (
    OVERRIDE_NONE,
    OVERRIDE_PENDING_SCALE_GATE,
    OVERRIDE_REPLICA_BOUND_CLAMP,
    OVERRIDE_SCALE_DOWN_COOLDOWN,
    GatingResult,
)


@dataclass(frozen=True)
class GatingConfig:
    min_replicas: int = 1
    max_replicas: int = 8
    scale_down_cooldown_seconds: float = 60.0


def compute_effective_action(
    requested_delta: int,
    current_desired: int,
    current_ready: int,
    seconds_since_last_effective_action: Optional[float],
    config: GatingConfig = GatingConfig(),
) -> GatingResult:
    """Assumes the caller has already established the current step's
    telemetry is valid (an invalid step must never call this function --
    it is unconditionally an effective HOLD with a telemetry-failure
    override reason, decided entirely outside this function)."""
    if requested_delta == 0:
        return GatingResult(0, OVERRIDE_NONE, bound_clamped=False, pending_scale=False)

    candidate = current_desired + requested_delta
    if candidate < config.min_replicas or candidate > config.max_replicas:
        return GatingResult(0, OVERRIDE_REPLICA_BOUND_CLAMP, bound_clamped=True, pending_scale=False)

    if current_desired != current_ready:
        return GatingResult(0, OVERRIDE_PENDING_SCALE_GATE, bound_clamped=False, pending_scale=True)

    if requested_delta > 0:
        # SCALE_UP: no additional cooldown beyond pending-scale gating
        # (CORR-27) -- effective at the very next decision point.
        return GatingResult(requested_delta, OVERRIDE_NONE, bound_clamped=False, pending_scale=False)

    # requested_delta < 0: SCALE_DOWN, subject to the 60s cooldown.
    if (
        seconds_since_last_effective_action is not None
        and seconds_since_last_effective_action < config.scale_down_cooldown_seconds
    ):
        return GatingResult(0, OVERRIDE_SCALE_DOWN_COOLDOWN, bound_clamped=False, pending_scale=False)

    return GatingResult(requested_delta, OVERRIDE_NONE, bound_clamped=False, pending_scale=False)
