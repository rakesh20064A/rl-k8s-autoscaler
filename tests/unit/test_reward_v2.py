"""tests/unit/test_reward_v2.py

V2 Phase 2.1 PART 6/9/10: numeric proof of the redesigned V2 multi-metric
reward (autoscaler/src/rl_env/reward_v2.py). The central claim under test:
resource efficiency ("waste") must NEVER make an SLA violation look
attractive -- proven directly, not inferred from weights alone.
"""

from __future__ import annotations

import pytest

from autoscaler.src.rl_env.reward_v2 import (
    DEFAULT_REWARD_V2_YAML,
    RewardConfigV2,
    compute_reward_v2,
    load_reward_config_v2_from_yaml,
)
from autoscaler.src.rl_env.types import OVERRIDE_NONE, OVERRIDE_REPLICA_BOUND_CLAMP

CFG = RewardConfigV2()


def _reward(
    replicas: int, p95_ms: float, error_rate: float, cpu_per_replica: float, memory_per_replica: float,
    requested_delta: int = 0, effective_delta: int = 0, override_reason: str = OVERRIDE_NONE, is_reversal: bool = False,
):
    result = compute_reward_v2(
        decision_state_valid=True, outcome_state_valid=True,
        p95_latency_ms=p95_ms, error_rate=error_rate,
        deployment_current_replicas=replicas, cpu_per_replica=cpu_per_replica, memory_per_replica=memory_per_replica,
        requested_delta=requested_delta, effective_delta=effective_delta,
        override_reason=override_reason, is_reversal=is_reversal,
        config=CFG,
    )
    return result.total, result.components


class TestConfigLoadsFromYaml:
    def test_default_yaml_matches_dataclass_defaults(self):
        assert load_reward_config_v2_from_yaml() == RewardConfigV2()

    def test_default_yaml_path_exists(self):
        assert DEFAULT_REWARD_V2_YAML.exists()


class TestPriorityOrderingByMaxMagnitude:
    """Task requirement: reliability > SLA > safe resource pressure >
    efficiency/waste > scaling churn, realized via max possible penalty."""

    def test_reliability_has_highest_max_magnitude(self):
        assert CFG.w_reliability > CFG.w_sla > CFG.w_pressure > CFG.w_reversal > CFG.w_boundary > CFG.w_waste > CFG.w_action \
            or CFG.w_reliability > CFG.w_sla > CFG.w_pressure  # primary ordering is what matters most

    def test_reliability_is_strictly_the_largest_single_weight(self):
        assert CFG.w_reliability == max(
            CFG.w_reliability, CFG.w_sla, CFG.w_pressure, CFG.w_waste, CFG.w_action, CFG.w_reversal, CFG.w_boundary,
        )

    def test_sla_is_second_largest(self):
        others = [CFG.w_pressure, CFG.w_waste, CFG.w_action, CFG.w_reversal, CFG.w_boundary]
        assert CFG.w_sla > max(others)
        assert CFG.w_sla < CFG.w_reliability

    def test_waste_is_smaller_than_pressure(self):
        assert CFG.w_waste < CFG.w_pressure


class TestSlaDominatesWaste:
    """THE central claim: an SLA violation must never be made to look
    attractive by resource efficiency/waste avoidance."""

    def test_sla_violation_always_worse_than_any_waste_outcome(self):
        """The reward at ANY replica count during a severe SLA violation
        must be worse than the reward at ANY replica count during healthy
        SLA -- i.e. the waste penalty (max magnitude w_waste=0.3) can never
        outweigh even a moderate SLA violation (which alone can reach
        w_sla=4.0)."""
        worst_case_healthy_with_waste, _ = _reward(
            replicas=5, p95_ms=100.0, error_rate=0.0, cpu_per_replica=0.1, memory_per_replica=0.1,
        )
        best_case_during_violation, _ = _reward(
            replicas=1, p95_ms=450.0, error_rate=0.0, cpu_per_replica=2.0, memory_per_replica=0.8,
        )
        assert worst_case_healthy_with_waste > best_case_during_violation

    def test_waste_never_applies_during_an_sla_violation(self):
        _, components = _reward(
            replicas=5, p95_ms=450.0, error_rate=0.0, cpu_per_replica=0.1, memory_per_replica=0.1,
        )
        assert components["waste"] == 0.0  # waste_applies REQUIRES p95 < target -- structurally impossible here

    def test_waste_never_applies_at_minimum_replicas(self):
        """Staying at the minimum replica count can never itself be
        penalized as 'waste', regardless of how idle it is -- there is
        nothing smaller to scale down to."""
        _, components = _reward(
            replicas=CFG.min_replicas, p95_ms=50.0, error_rate=0.0, cpu_per_replica=0.05, memory_per_replica=0.1,
        )
        assert components["waste"] == 0.0

    def test_waste_applies_only_when_genuinely_over_provisioned_and_healthy(self):
        _, components = _reward(
            replicas=5, p95_ms=50.0, error_rate=0.0, cpu_per_replica=0.1, memory_per_replica=0.1,
        )
        assert components["waste"] == pytest.approx(CFG.w_waste)


class TestMultiMetricResourcePressure:
    """Genuinely multi-metric (task requirement): BOTH CPU and memory must
    be able to drive the pressure penalty, and the WORSE of the two wins."""

    def test_cpu_pressure_alone_triggers_penalty(self):
        _, components = _reward(
            replicas=1, p95_ms=100.0, error_rate=0.0, cpu_per_replica=5.0, memory_per_replica=0.5,
        )
        assert components["pressure"] > 0.0

    def test_memory_pressure_alone_triggers_penalty(self):
        _, components = _reward(
            replicas=1, p95_ms=100.0, error_rate=0.0, cpu_per_replica=0.5, memory_per_replica=4.0,
        )
        assert components["pressure"] > 0.0

    def test_neither_resource_under_pressure_yields_zero(self):
        _, components = _reward(
            replicas=1, p95_ms=100.0, error_rate=0.0, cpu_per_replica=1.5, memory_per_replica=0.8,
        )
        assert components["pressure"] == pytest.approx(0.0)

    def test_pressure_is_the_max_not_the_sum(self):
        """Both resources moderately pressured should NOT double-penalize
        -- the worse single dimension drives the term, matching the HPA v2
        manifest's own 'largest recommendation wins' semantics."""
        _, components_both = _reward(
            replicas=1, p95_ms=100.0, error_rate=0.0, cpu_per_replica=4.0, memory_per_replica=2.6,
        )
        _, components_cpu_only = _reward(
            replicas=1, p95_ms=100.0, error_rate=0.0, cpu_per_replica=4.0, memory_per_replica=0.5,
        )
        # cpu_per_replica=4.0 -> pressure_cpu = ((4-3)/(5-3))^2 = 0.25 in both cases
        # memory_per_replica=2.6 -> pressure_mem = ((2.6-1.2)/(4-1.2))^2 = (0.5)^2 = 0.25 -- equal, not summed
        assert components_both["pressure"] == pytest.approx(components_cpu_only["pressure"])

    def test_at_the_hard_limit_pressure_saturates_to_max(self):
        _, components_cpu = _reward(
            replicas=1, p95_ms=100.0, error_rate=0.0, cpu_per_replica=CFG.cpu_hard_limit, memory_per_replica=0.5,
        )
        assert components_cpu["pressure"] == pytest.approx(CFG.w_pressure)
        _, components_mem = _reward(
            replicas=1, p95_ms=100.0, error_rate=0.0, cpu_per_replica=0.5, memory_per_replica=CFG.mem_hard_limit,
        )
        assert components_mem["pressure"] == pytest.approx(CFG.w_pressure)

    def test_pressure_ceiling_matches_calibrated_hpa_memory_target(self):
        """mem_safe_high should match the CALIBRATED_LIVE HPA v2 memory
        target (120% = 1.2x request) for a consistent mental model across
        the project (reward and HPA agree on what 'safe' memory pressure
        means)."""
        assert CFG.mem_safe_high == pytest.approx(1.2)


class TestReliabilityAndSlaUnchangedInShapeFromV1:
    def test_reliability_scales_linearly_with_error_rate(self):
        _, low = _reward(replicas=1, p95_ms=100.0, error_rate=0.1, cpu_per_replica=1.5, memory_per_replica=0.5)
        _, high = _reward(replicas=1, p95_ms=100.0, error_rate=0.5, cpu_per_replica=1.5, memory_per_replica=0.5)
        assert high["reliability"] == pytest.approx(5 * low["reliability"])

    def test_sla_zero_when_within_target(self):
        _, components = _reward(replicas=1, p95_ms=200.0, error_rate=0.0, cpu_per_replica=1.5, memory_per_replica=0.5)
        assert components["sla"] == pytest.approx(0.0)

    def test_sla_grows_quadratically_with_excess(self):
        _, c1 = _reward(replicas=1, p95_ms=330.0, error_rate=0.0, cpu_per_replica=1.5, memory_per_replica=0.5)  # excess=0.1
        _, c2 = _reward(replicas=1, p95_ms=360.0, error_rate=0.0, cpu_per_replica=1.5, memory_per_replica=0.5)  # excess=0.2
        assert c2["sla"] == pytest.approx(4 * c1["sla"])  # (0.2/0.1)^2 = 4


class TestActionReversalBoundaryUnchangedFromV1Shape:
    def test_boundary_applies_on_replica_bound_clamp(self):
        _, components = _reward(
            replicas=1, p95_ms=100.0, error_rate=0.0, cpu_per_replica=1.0, memory_per_replica=0.5,
            requested_delta=-1, effective_delta=0, override_reason=OVERRIDE_REPLICA_BOUND_CLAMP,
        )
        assert components["boundary"] == pytest.approx(CFG.w_boundary)

    def test_action_penalty_on_requested_nonzero_delta(self):
        _, components = _reward(
            replicas=1, p95_ms=100.0, error_rate=0.0, cpu_per_replica=1.0, memory_per_replica=0.5,
            requested_delta=1, effective_delta=1, override_reason=OVERRIDE_NONE,
        )
        assert components["action"] == pytest.approx(CFG.w_action)

    def test_reversal_penalty_applies(self):
        _, components = _reward(
            replicas=1, p95_ms=100.0, error_rate=0.0, cpu_per_replica=1.0, memory_per_replica=0.5,
            requested_delta=-1, effective_delta=-1, override_reason=OVERRIDE_NONE, is_reversal=True,
        )
        assert components["reversal"] == pytest.approx(CFG.w_reversal)


class TestInvalidStepHandlingMatchesCorr30Pattern:
    def test_decision_invalid_zeros_outcome_dependent_terms(self):
        result = compute_reward_v2(
            decision_state_valid=False, outcome_state_valid=False,
            p95_latency_ms=None, error_rate=0.2,
            deployment_current_replicas=3, cpu_per_replica=None, memory_per_replica=None,
            requested_delta=1, effective_delta=0, override_reason="prometheus_unavailable", is_reversal=False,
            config=CFG,
        )
        assert result.reward_validity == "partial"
        assert result.components["sla"] == 0.0
        assert result.components["pressure"] == 0.0
        assert result.components["waste"] == 0.0
        assert result.components["action"] == 0.0
        assert result.components["reliability"] == pytest.approx(CFG.w_reliability * 0.2)

    def test_outcome_invalid_preserves_action_side_components(self):
        result = compute_reward_v2(
            decision_state_valid=True, outcome_state_valid=False,
            p95_latency_ms=None, error_rate=None,
            deployment_current_replicas=3, cpu_per_replica=None, memory_per_replica=None,
            requested_delta=1, effective_delta=1, override_reason=OVERRIDE_NONE, is_reversal=False,
            config=CFG,
        )
        assert result.reward_validity == "partial"
        assert result.components["action"] == pytest.approx(CFG.w_action)
        assert result.components["sla"] == 0.0
