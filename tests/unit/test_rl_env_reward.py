"""tests/unit/test_rl_env_reward.py

Phase 15B.1 / 15B.1A: behavioral tests for autoscaler/src/rl_env/reward.py
-- the frozen V2 reward formula (CORR-27) and the CORR-30 decision-state
vs. outcome-state validity decomposition.
"""

import math

from autoscaler.src.rl_env.reward import RewardConfig, compute_reward
from autoscaler.src.rl_env.types import (
    OVERRIDE_NONE,
    OVERRIDE_PENDING_SCALE_GATE,
    OVERRIDE_REPLICA_BOUND_CLAMP,
    OVERRIDE_SCALE_DOWN_COOLDOWN,
    OVERRIDE_STALE_OR_MISSING_METRIC,
)

CFG = RewardConfig()


def _reward(**overrides):
    base = dict(
        decision_state_valid=True,
        outcome_state_valid=True,
        p95_latency_ms=100.0,
        error_rate=0.0,
        deployment_current_replicas=1,
        cpu_per_replica=0.5,
        requested_delta=0,
        effective_delta=0,
        override_reason=OVERRIDE_NONE,
        is_reversal=False,
        config=CFG,
    )
    base.update(overrides)
    return compute_reward(**base)


class TestHealthyStep:
    def test_healthy_efficient_hold_step_gets_full_bonus(self):
        result = _reward(cpu_per_replica=2.0, p95_latency_ms=100.0, error_rate=0.0)
        assert math.isclose(result.total, 0.3)
        assert result.components["efficiency_bonus"] == 0.3
        assert result.reward_validity == "full"

    def test_outside_efficiency_band_no_bonus(self):
        result = _reward(cpu_per_replica=0.5)  # below 1.0 low bound
        assert result.components["efficiency_bonus"] == 0.0


class TestResourceUsesCurrentReplicas:
    def test_resource_scales_with_current_replicas_not_ready(self):
        result = _reward(deployment_current_replicas=8, cpu_per_replica=None)
        assert math.isclose(result.components["resource"], 1.0)  # (8-1)/7 = 1.0

    def test_resource_zero_at_one_replica(self):
        result = _reward(deployment_current_replicas=1, cpu_per_replica=None)
        assert result.components["resource"] == 0.0


class TestSlaSeverityQuadratic:
    def test_sla_zero_at_or_below_target(self):
        result = _reward(p95_latency_ms=250.0, cpu_per_replica=None)
        assert result.components["sla"] == 0.0

    def test_sla_grows_quadratically(self):
        # excess at 450ms = (450-300)/300 = 0.5 -> sla = 1.0 * 0.5^2 = 0.25
        result = _reward(p95_latency_ms=450.0, cpu_per_replica=None)
        assert math.isclose(result.components["sla"], 0.25)

    def test_sla_clipped_at_severe_excess(self):
        # 3000ms excess = (3000-300)/300 = 9.0 -> 9^2=81, clipped to 4.0
        result = _reward(p95_latency_ms=3000.0, cpu_per_replica=None)
        assert result.components["sla"] == 4.0

    def test_301ms_and_3000ms_not_treated_equally(self):
        near_threshold = _reward(p95_latency_ms=301.0, cpu_per_replica=None).components["sla"]
        severe = _reward(p95_latency_ms=3000.0, cpu_per_replica=None).components["sla"]
        assert near_threshold < severe


class TestReliabilityPenalty:
    def test_reliability_scales_with_error_rate(self):
        result = _reward(error_rate=0.2, cpu_per_replica=None)
        assert math.isclose(result.components["reliability"], 1.0)  # 5.0 * 0.2

    def test_reliability_zero_when_no_errors(self):
        result = _reward(error_rate=0.0, cpu_per_replica=None)
        assert result.components["reliability"] == 0.0


class TestActionAndBoundaryPenalty:
    def test_action_penalty_on_valid_requested_scale(self):
        result = _reward(requested_delta=1, effective_delta=1, override_reason=OVERRIDE_NONE, cpu_per_replica=None)
        assert math.isclose(result.components["action"], 0.1)

    def test_no_action_penalty_on_hold(self):
        result = _reward(requested_delta=0, effective_delta=0, override_reason=OVERRIDE_NONE, cpu_per_replica=None)
        assert result.components["action"] == 0.0

    def test_action_penalty_fires_for_pending_scale_gate(self):
        result = _reward(requested_delta=1, effective_delta=0, override_reason=OVERRIDE_PENDING_SCALE_GATE, cpu_per_replica=None)
        assert math.isclose(result.components["action"], 0.1)
        assert result.components["boundary"] == 0.0

    def test_action_penalty_fires_for_scale_down_cooldown(self):
        result = _reward(requested_delta=-1, effective_delta=0, override_reason=OVERRIDE_SCALE_DOWN_COOLDOWN, cpu_per_replica=None)
        assert math.isclose(result.components["action"], 0.1)

    def test_boundary_only_no_action_double_count(self):
        """CORR-27: replica_bound_clamp incurs boundary penalty ONLY, never
        the action penalty too, for the same event."""
        result = _reward(requested_delta=1, effective_delta=0, override_reason=OVERRIDE_REPLICA_BOUND_CLAMP, cpu_per_replica=None)
        assert math.isclose(result.components["boundary"], 0.2)
        assert result.components["action"] == 0.0

    def test_no_action_or_boundary_penalty_when_decision_state_invalid(self):
        result = _reward(
            decision_state_valid=False, requested_delta=1, effective_delta=0,
            override_reason=OVERRIDE_STALE_OR_MISSING_METRIC, cpu_per_replica=None,
            deployment_current_replicas=2,
        )
        assert result.components["action"] == 0.0
        assert result.components["boundary"] == 0.0


class TestReversalPenalty:
    def test_reversal_penalty_fires_only_for_effective_reversal(self):
        result = _reward(requested_delta=1, effective_delta=1, override_reason=OVERRIDE_NONE, is_reversal=True, cpu_per_replica=None)
        assert math.isclose(result.components["reversal"], 0.5)

    def test_no_reversal_penalty_when_effective_delta_zero(self):
        # is_reversal flagged True but effective_delta==0 -- must not apply.
        result = _reward(requested_delta=1, effective_delta=0, override_reason=OVERRIDE_PENDING_SCALE_GATE, is_reversal=True, cpu_per_replica=None)
        assert result.components["reversal"] == 0.0


class TestValidityMatrixCasesBAndD:
    """decision_state_valid=False (CORR-30 Cases B/D) -- safety-HOLD
    semantics, identical regardless of outcome_state_valid."""

    def test_case_b_decision_invalid_outcome_valid_zeros_policy_terms(self):
        result = _reward(
            decision_state_valid=False, outcome_state_valid=True,
            p95_latency_ms=None, error_rate=None,
            deployment_current_replicas=None, cpu_per_replica=None,
            requested_delta=1, effective_delta=0, override_reason=OVERRIDE_STALE_OR_MISSING_METRIC,
        )
        assert result.reward_validity == "partial"
        for key in ("sla", "action", "reversal", "boundary", "efficiency_bonus", "resource", "reliability"):
            assert result.components[key] == 0.0

    def test_case_d_both_invalid_same_as_case_b(self):
        result = _reward(
            decision_state_valid=False, outcome_state_valid=False,
            p95_latency_ms=None, error_rate=None,
            deployment_current_replicas=None, cpu_per_replica=None,
            requested_delta=1, effective_delta=0, override_reason=OVERRIDE_STALE_OR_MISSING_METRIC,
        )
        assert result.reward_validity == "partial"
        for key in ("sla", "action", "reversal", "boundary", "efficiency_bonus"):
            assert result.components[key] == 0.0

    def test_resource_computed_if_current_replicas_independently_known(self):
        result = _reward(
            decision_state_valid=False, p95_latency_ms=None, error_rate=None,
            deployment_current_replicas=4, cpu_per_replica=None,
        )
        assert math.isclose(result.components["resource"], 3.0 / 7.0)

    def test_reliability_computed_if_error_rate_independently_valid(self):
        result = _reward(
            decision_state_valid=False, p95_latency_ms=None, error_rate=0.4,
            deployment_current_replicas=None, cpu_per_replica=None,
        )
        assert math.isclose(result.components["reliability"], 2.0)


class TestValidityMatrixCaseC:
    """decision_state_valid=True, outcome_state_valid=False (CORR-30 Case C):
    the policy's action/reversal/boundary consequences were already fully
    determined at decision time and must NOT be erased by a later telemetry
    failure. This is the exact defect Phase 15B.1A corrects."""

    def test_c1_valid_scale_up_then_outcome_telemetry_fails(self):
        result = _reward(
            decision_state_valid=True, outcome_state_valid=False,
            p95_latency_ms=None, error_rate=None, cpu_per_replica=None,
            deployment_current_replicas=2,  # independently known (Kubernetes-sourced)
            requested_delta=1, effective_delta=1, override_reason=OVERRIDE_NONE,
        )
        assert result.reward_validity == "partial"
        assert math.isclose(result.components["action"], 0.1)  # PRESERVED
        assert result.components["sla"] == 0.0
        assert result.components["efficiency_bonus"] == 0.0
        assert math.isclose(result.components["resource"], 1.0 / 7.0)  # still computed

    def test_c2_valid_emergency_reversal_then_outcome_telemetry_fails(self):
        result = _reward(
            decision_state_valid=True, outcome_state_valid=False,
            p95_latency_ms=None, error_rate=None, cpu_per_replica=None,
            deployment_current_replicas=1,
            requested_delta=1, effective_delta=1, override_reason=OVERRIDE_NONE, is_reversal=True,
        )
        assert math.isclose(result.components["action"], 0.1)  # PRESERVED
        assert math.isclose(result.components["reversal"], 0.5)  # PRESERVED -- not erased
        assert result.reward_validity == "partial"

    def test_c3_valid_replica_bound_request_then_outcome_telemetry_fails(self):
        result = _reward(
            decision_state_valid=True, outcome_state_valid=False,
            p95_latency_ms=None, error_rate=None, cpu_per_replica=None,
            deployment_current_replicas=8,
            requested_delta=1, effective_delta=0, override_reason=OVERRIDE_REPLICA_BOUND_CLAMP,
        )
        assert math.isclose(result.components["boundary"], 0.2)  # PRESERVED
        assert result.components["action"] == 0.0  # still no double-count
        assert result.reward_validity == "partial"

    def test_sla_never_computed_from_unavailable_outcome_latency(self):
        result = _reward(
            decision_state_valid=True, outcome_state_valid=False,
            p95_latency_ms=None, error_rate=None, cpu_per_replica=2.0,
            deployment_current_replicas=1,
        )
        assert result.components["sla"] == 0.0
        assert result.components["efficiency_bonus"] == 0.0

    def test_reliability_and_resource_still_computed_if_independently_valid_in_case_c(self):
        result = _reward(
            decision_state_valid=True, outcome_state_valid=False,
            p95_latency_ms=None, error_rate=0.1, cpu_per_replica=None,
            deployment_current_replicas=3,
        )
        assert math.isclose(result.components["reliability"], 0.5)
        assert math.isclose(result.components["resource"], 2.0 / 7.0)


class TestRewardCannotIncentivizeTelemetryFailure:
    """Regression guard: for the SAME valid action event, a subsequent
    telemetry failure must never make an already-determined action/
    reversal/boundary cost disappear (it would otherwise create a perverse
    incentive to prefer telemetry failure over a clean observation)."""

    def test_action_and_boundary_costs_identical_whether_outcome_valid_or_not(self):
        shared_kwargs = dict(
            decision_state_valid=True, requested_delta=1, effective_delta=0,
            override_reason=OVERRIDE_REPLICA_BOUND_CLAMP, deployment_current_replicas=8,
        )
        with_valid_outcome = _reward(outcome_state_valid=True, p95_latency_ms=100.0, error_rate=0.0, cpu_per_replica=1.0, **shared_kwargs)
        with_invalid_outcome = _reward(outcome_state_valid=False, p95_latency_ms=None, error_rate=None, cpu_per_replica=None, **shared_kwargs)
        assert with_valid_outcome.components["boundary"] == with_invalid_outcome.components["boundary"]
        assert with_valid_outcome.components["action"] == with_invalid_outcome.components["action"]
        # Outcome-dependent terms legitimately differ because the outcome data is unavailable.
        assert with_invalid_outcome.components["sla"] == 0.0
        assert with_invalid_outcome.components["efficiency_bonus"] == 0.0
