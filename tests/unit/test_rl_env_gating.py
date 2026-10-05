"""tests/unit/test_rl_env_gating.py

Phase 15B.1: behavioral tests for autoscaler/src/rl_env/gating.py --
the frozen V2 asymmetric action-gating rule (CORR-27).
"""

from autoscaler.src.rl_env.gating import GatingConfig, compute_effective_action
from autoscaler.src.rl_env.types import (
    OVERRIDE_NONE,
    OVERRIDE_PENDING_SCALE_GATE,
    OVERRIDE_REPLICA_BOUND_CLAMP,
    OVERRIDE_SCALE_DOWN_COOLDOWN,
)

CFG = GatingConfig(min_replicas=1, max_replicas=8, scale_down_cooldown_seconds=60.0)


class TestHold:
    def test_hold_is_always_effective_none(self):
        result = compute_effective_action(0, current_desired=3, current_ready=3, seconds_since_last_effective_action=None, config=CFG)
        assert result.effective_delta == 0
        assert result.reason == OVERRIDE_NONE


class TestScaleUp:
    def test_valid_scale_up_executes(self):
        result = compute_effective_action(+1, current_desired=2, current_ready=2, seconds_since_last_effective_action=1000.0, config=CFG)
        assert result.effective_delta == 1
        assert result.reason == OVERRIDE_NONE

    def test_scale_up_has_no_cooldown_even_immediately_after_prior_action(self):
        """CORR-27: unlike scale-down, scale-up is free of any minimum
        spacing beyond pending-scale gating."""
        result = compute_effective_action(+1, current_desired=2, current_ready=2, seconds_since_last_effective_action=0.0, config=CFG)
        assert result.effective_delta == 1
        assert result.reason == OVERRIDE_NONE

    def test_scale_up_blocked_at_max_replicas(self):
        result = compute_effective_action(+1, current_desired=8, current_ready=8, seconds_since_last_effective_action=1000.0, config=CFG)
        assert result.effective_delta == 0
        assert result.reason == OVERRIDE_REPLICA_BOUND_CLAMP
        assert result.bound_clamped is True


class TestScaleDown:
    def test_scale_down_blocked_before_60s(self):
        result = compute_effective_action(-1, current_desired=3, current_ready=3, seconds_since_last_effective_action=59.9, config=CFG)
        assert result.effective_delta == 0
        assert result.reason == OVERRIDE_SCALE_DOWN_COOLDOWN

    def test_scale_down_allowed_at_exactly_60s(self):
        result = compute_effective_action(-1, current_desired=3, current_ready=3, seconds_since_last_effective_action=60.0, config=CFG)
        assert result.effective_delta == -1
        assert result.reason == OVERRIDE_NONE

    def test_scale_down_allowed_when_no_prior_effective_action(self):
        result = compute_effective_action(-1, current_desired=3, current_ready=3, seconds_since_last_effective_action=None, config=CFG)
        assert result.effective_delta == -1
        assert result.reason == OVERRIDE_NONE

    def test_scale_down_blocked_at_min_replicas(self):
        result = compute_effective_action(-1, current_desired=1, current_ready=1, seconds_since_last_effective_action=1000.0, config=CFG)
        assert result.effective_delta == 0
        assert result.reason == OVERRIDE_REPLICA_BOUND_CLAMP
        assert result.bound_clamped is True


class TestPendingScaleGate:
    def test_pending_transition_blocks_scale_up(self):
        result = compute_effective_action(+1, current_desired=2, current_ready=1, seconds_since_last_effective_action=1000.0, config=CFG)
        assert result.effective_delta == 0
        assert result.reason == OVERRIDE_PENDING_SCALE_GATE
        assert result.pending_scale is True

    def test_pending_transition_blocks_scale_down(self):
        result = compute_effective_action(-1, current_desired=2, current_ready=1, seconds_since_last_effective_action=1000.0, config=CFG)
        assert result.effective_delta == 0
        assert result.reason == OVERRIDE_PENDING_SCALE_GATE

    def test_bound_clamp_takes_precedence_over_pending_scale(self):
        """Ambiguous case (task sec.13): a request that is BOTH out-of-bounds
        AND would otherwise be pending-gated -- bound clamp wins, since the
        request is fundamentally impossible regardless of pending state."""
        result = compute_effective_action(+1, current_desired=8, current_ready=7, seconds_since_last_effective_action=1000.0, config=CFG)
        assert result.reason == OVERRIDE_REPLICA_BOUND_CLAMP


class TestEmergencyReversal:
    def test_scale_up_not_blocked_30s_after_scale_down(self):
        """CORR-27 emergency-reversal rule: the 60s scale-down cooldown
        must never block a needed scale-up."""
        result = compute_effective_action(+1, current_desired=1, current_ready=1, seconds_since_last_effective_action=30.0, config=CFG)
        assert result.effective_delta == 1
        assert result.reason == OVERRIDE_NONE

    def test_another_scale_down_still_blocked_30s_after_a_scale_down(self):
        result = compute_effective_action(-1, current_desired=2, current_ready=2, seconds_since_last_effective_action=30.0, config=CFG)
        assert result.effective_delta == 0
        assert result.reason == OVERRIDE_SCALE_DOWN_COOLDOWN
