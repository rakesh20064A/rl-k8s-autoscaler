"""tests/unit/test_reward_v1_audit.py

V2 Phase 2.1 PART 5: audits the FROZEN V1 reward (autoscaler/src/rl_env/
reward.py, configs/reward.yaml, unchanged by this phase) against the
observed V1 problem ("PPO often stayed near low replica counts while
latency/SLA degraded") using actual computed component totals for
representative synthetic states -- not just an inference from the raw
weights. All action/reversal/boundary components are held at their
"no override, no reversal" baseline (requested_delta=0,
override_reason=OVERRIDE_NONE, is_reversal=False) in every case below so
the SLA/resource/efficiency trade-off is isolated and comparable across
replica counts, per the task's explicit instruction to show real totals.

This test file does NOT modify reward.py/configs/reward.yaml -- it is a
read-only audit of frozen V1 behavior, reproduced here as regression-
protected evidence for the V2 reward design decisions in
test_reward_v2.py.
"""

from __future__ import annotations

import pytest

from autoscaler.src.rl_env.reward import RewardConfig, compute_reward
from autoscaler.src.rl_env.types import OVERRIDE_NONE

CFG = RewardConfig()  # frozen V1 defaults, matching configs/reward.yaml exactly


def _reward(replicas: int, p95_ms: float, error_rate: float, cpu_per_replica: float) -> float:
    result = compute_reward(
        decision_state_valid=True, outcome_state_valid=True,
        p95_latency_ms=p95_ms, error_rate=error_rate,
        deployment_current_replicas=replicas, cpu_per_replica=cpu_per_replica,
        requested_delta=0, effective_delta=0, override_reason=OVERRIDE_NONE, is_reversal=False,
        config=CFG,
    )
    return result.total, result.components


class TestV1RewardNumericCases:
    """Case numbers match the task's PART 5 list exactly."""

    def test_case_1_low_replicas_healthy_sla(self):
        total, components = _reward(replicas=1, p95_ms=150.0, error_rate=0.0, cpu_per_replica=1.5)
        assert components["sla"] == pytest.approx(0.0)
        assert components["resource"] == pytest.approx(0.0)
        assert components["efficiency_bonus"] == pytest.approx(0.3)  # in-band, healthy, no errors
        assert total == pytest.approx(0.3)

    def test_case_2_low_replicas_moderate_sla_violation(self):
        # excess = (450-300)/300 = 0.5 -> excess^2 = 0.25
        total, components = _reward(replicas=1, p95_ms=450.0, error_rate=0.0, cpu_per_replica=4.0)
        assert components["sla"] == pytest.approx(0.25)
        assert components["resource"] == pytest.approx(0.0)
        assert components["efficiency_bonus"] == pytest.approx(0.0)  # cpu_per_replica=4.0 exceeds the 3.0 band
        assert total == pytest.approx(-0.25)

    def test_case_3_low_replicas_severe_sla_violation(self):
        # excess = (1500-300)/300 = 4.0 -> excess^2 = 16.0, clipped to sla_excess_clip=4.0
        total, components = _reward(replicas=1, p95_ms=1500.0, error_rate=0.05, cpu_per_replica=5.0)
        assert components["sla"] == pytest.approx(4.0)  # clipped
        assert components["reliability"] == pytest.approx(0.25)  # 5.0 * 0.05
        assert total == pytest.approx(-4.25)

    def test_case_4_higher_replicas_healthy_sla(self):
        # resource = (5-1)/(8-1) = 0.5714..., but cpu_per_replica below the
        # efficiency band (over-provisioned relative to load) -- no bonus.
        total, components = _reward(replicas=5, p95_ms=120.0, error_rate=0.0, cpu_per_replica=0.5)
        assert components["resource"] == pytest.approx(4 / 7)
        assert components["efficiency_bonus"] == pytest.approx(0.0)
        assert total == pytest.approx(-4 / 7)

    def test_case_5_higher_replicas_moderate_sla_violation(self):
        total, components = _reward(replicas=5, p95_ms=450.0, error_rate=0.0, cpu_per_replica=3.5)
        assert components["sla"] == pytest.approx(0.25)
        assert components["resource"] == pytest.approx(4 / 7)
        assert components["efficiency_bonus"] == pytest.approx(0.0)  # 3.5 exceeds the 3.0 band
        assert total == pytest.approx(-(0.25 + 4 / 7))

    def test_case_6_memory_heavy_high_pressure_healthy_latency(self):
        """V1's reward.py has NO memory-related input at all (confirmed by
        reading reward.py's full parameter list: p95_latency_ms, error_rate,
        deployment_current_replicas, cpu_per_replica, requested_delta,
        effective_delta, override_reason, is_reversal -- memory_per_replica
        is never passed in). A memory-mode request also performs near-zero
        CPU work (app/src/workload.py's perform_work(): resource_mode=
        "memory" skips CPU work entirely), so cpu_per_replica is low here
        too -- below the efficiency band."""
        total, components = _reward(replicas=1, p95_ms=200.0, error_rate=0.0, cpu_per_replica=0.5)
        assert components["sla"] == pytest.approx(0.0)
        assert total == pytest.approx(0.0)

    def test_case_7_memory_pressure_approaching_unsafe_level_identical_reward(self):
        """THE key finding: with SLA/CPU/error held at the same healthy
        values as case 6, the reward is IDENTICAL regardless of how close
        memory usage is to the pod's OOM limit -- because reward.py simply
        never looks at memory. This is not a close call; it is a structural
        blind spot, confirmed by identical totals."""
        total, components = _reward(replicas=1, p95_ms=200.0, error_rate=0.0, cpu_per_replica=0.5)
        assert total == pytest.approx(0.0)  # same as case 6 -- memory pressure is invisible to this formula


class TestV1ReplicaSavingIncentiveConfirmed:
    """Directly answers the task's question: 'Determine whether the current
    V1 reward can rationally prefer saving replicas during sustained SLA
    degradation.' Answer: YES, demonstrated numerically below."""

    def test_low_replica_moderate_violation_beats_higher_replica_same_violation(self):
        """For the SAME SLA-violation severity, fewer replicas is a BETTER
        (higher/less-negative) reward purely due to the resource term --
        this is case 2 vs case 5 from above."""
        low_total, _ = _reward(replicas=1, p95_ms=450.0, error_rate=0.0, cpu_per_replica=4.0)
        high_total, _ = _reward(replicas=5, p95_ms=450.0, error_rate=0.0, cpu_per_replica=3.5)
        assert low_total > high_total
        assert low_total == pytest.approx(-0.25)
        assert high_total == pytest.approx(-(0.25 + 4 / 7))

    def test_resource_penalty_is_incurred_every_step_regardless_of_need(self):
        """The resource term is purely a function of replica count -- it
        applies identically whether or not those extra replicas are
        currently needed, unlike the SLA term which only fires during an
        actual violation. This asymmetry (constant cost vs. intermittent
        cost) is the structural mechanism behind the observed V1 behavior:
        a policy that stays lean pays nothing extra until a violation is
        ALREADY happening (reactive, not preventive)."""
        resource_only_5_replicas, _ = _reward(replicas=5, p95_ms=100.0, error_rate=0.0, cpu_per_replica=0.3)
        resource_only_1_replica, _ = _reward(replicas=1, p95_ms=100.0, error_rate=0.0, cpu_per_replica=0.3)
        # Both healthy SLA, both out-of-band CPU (no efficiency bonus) --
        # the ONLY difference is the resource term, and it is substantial.
        assert resource_only_1_replica - resource_only_5_replicas == pytest.approx(4 / 7)

    def test_efficiency_bonus_is_easier_to_earn_at_low_replica_count(self):
        """The efficiency band (1.0x-3.0x the CPU request) is naturally
        easier to land in at LOW replica count under light/moderate load
        (CPU concentrated on fewer pods); spreading the same load across
        more replicas dilutes cpu_per_replica below the band, losing the
        bonus entirely -- this is case 1 vs case 4's cpu_per_replica choice,
        which reflects a realistic per-replica CPU dilution effect, not an
        arbitrary difference."""
        low_replica_bonus, _ = _reward(replicas=1, p95_ms=150.0, error_rate=0.0, cpu_per_replica=1.5)
        high_replica_bonus, _ = _reward(replicas=5, p95_ms=150.0, error_rate=0.0, cpu_per_replica=0.5)
        assert low_replica_bonus > high_replica_bonus


class TestV1RewardMemoryBlindSpotConfirmed:
    def test_compute_reward_signature_has_no_memory_parameter(self):
        import inspect

        sig = inspect.signature(compute_reward)
        assert "memory_per_replica" not in sig.parameters
        assert "memory_bytes" not in sig.parameters

    def test_reward_identical_across_wide_range_of_hypothetical_memory_pressure(self):
        """Since reward.py has no memory parameter at all, this test cannot
        even pass a different memory value to compute_reward() to prove the
        point directly -- which IS the point. The only way to demonstrate
        the blind spot is to show that every other case 6/7-shaped state
        (healthy SLA, low CPU, no errors) produces the SAME reward
        regardless of the real-world memory pressure that state actually
        represents (idle vs. near-OOM)."""
        idle_like, _ = _reward(replicas=1, p95_ms=200.0, error_rate=0.0, cpu_per_replica=0.5)
        near_oom_like, _ = _reward(replicas=1, p95_ms=200.0, error_rate=0.0, cpu_per_replica=0.5)
        assert idle_like == near_oom_like
