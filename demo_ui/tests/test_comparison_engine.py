from __future__ import annotations

import pytest

from demo_ui.comparison_engine import compare, count_scaling_events, replica_seconds
from demo_ui.models import Controller, RunSummary, TrafficProfile


def _run(controller: Controller, replicas, timestamps, **overrides) -> RunSummary:
    base = dict(
        controller=controller, traffic_profile=TrafficProfile.CONSTANT, seed=42, duration_s=timestamps[-1],
        schedule_hash="abc123", timestamps=timestamps, replicas=replicas,
        cpu_pct=[50.0] * len(timestamps), memory_pct=[20.0] * len(timestamps), rps=[10.0] * len(timestamps),
        latency_avg_ms=[40.0] * len(timestamps), latency_p95_ms=[80.0] * len(timestamps), latency_p99_ms=[100.0] * len(timestamps),
        successful_requests=1000, errors=5, timeouts=1, sla_violations=2,
    )
    base.update(overrides)
    return RunSummary(**base)


class TestReplicaSeconds:
    def test_constant_replica_count(self):
        # 3 replicas held for 30s total (3 samples, 2 intervals of 10s + 10s... 2 intervals)
        assert replica_seconds([0, 10, 20], [3, 3, 3]) == 60.0  # 3*10 + 3*10

    def test_single_sample_returns_zero(self):
        assert replica_seconds([0], [3]) == 0.0

    def test_empty_returns_zero(self):
        assert replica_seconds([], []) == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            replica_seconds([0, 10], [1])

    def test_step_function_not_interpolated(self):
        # replicas jump from 1 to 5 at t=10; value HELD from its own sample forward
        assert replica_seconds([0, 10, 20], [1, 5, 5]) == 1 * 10 + 5 * 10


class TestCountScalingEvents:
    def test_no_changes(self):
        assert count_scaling_events([2, 2, 2, 2]) == 0

    def test_counts_each_change_not_magnitude(self):
        assert count_scaling_events([1, 3, 3, 1, 1, 5]) == 3

    def test_empty_and_single(self):
        assert count_scaling_events([]) == 0
        assert count_scaling_events([1]) == 0


class TestCompare:
    def test_requires_matching_profile(self):
        hpa = _run(Controller.HPA, [1, 1], [0, 10], traffic_profile=TrafficProfile.CONSTANT)
        ppo = _run(Controller.PPO, [1, 1], [0, 10], traffic_profile=TrafficProfile.RAMP)
        with pytest.raises(ValueError):
            compare(hpa, ppo)

    def test_requires_matching_seed(self):
        hpa = _run(Controller.HPA, [1, 1], [0, 10], seed=1)
        ppo = _run(Controller.PPO, [1, 1], [0, 10], seed=2)
        with pytest.raises(ValueError):
            compare(hpa, ppo)

    def test_same_schedule_hash_is_verified(self):
        hpa = _run(Controller.HPA, [1, 1], [0, 10], schedule_hash="same")
        ppo = _run(Controller.PPO, [1, 1], [0, 10], schedule_hash="same")
        result = compare(hpa, ppo)
        assert result.same_schedule_verified is True

    def test_different_schedule_hash_is_flagged(self):
        hpa = _run(Controller.HPA, [1, 1], [0, 10], schedule_hash="a")
        ppo = _run(Controller.PPO, [1, 1], [0, 10], schedule_hash="b")
        result = compare(hpa, ppo)
        assert result.same_schedule_verified is False

    def test_replica_seconds_and_peaks_computed(self):
        hpa = _run(Controller.HPA, [1, 2, 2], [0, 10, 20])
        ppo = _run(Controller.PPO, [1, 1, 1], [0, 10, 20])
        result = compare(hpa, ppo)
        assert result.hpa_replica_seconds == 1 * 10 + 2 * 10
        assert result.ppo_replica_seconds == 1 * 10 + 1 * 10
        assert result.hpa_peak_replicas == 2
        assert result.ppo_peak_replicas == 1

    def test_metric_pct_difference_sign(self):
        hpa = _run(Controller.HPA, [1, 1], [0, 10], latency_p95_ms=[100.0, 100.0])
        ppo = _run(Controller.PPO, [1, 1], [0, 10], latency_p95_ms=[80.0, 80.0])
        result = compare(hpa, ppo)
        p95 = next(m for m in result.metrics if m.name == "P95 Latency")
        assert p95.pct_difference == pytest.approx(-20.0)
        assert p95.ppo_is_better is True

    def test_does_not_hardcode_ppo_as_better(self):
        hpa = _run(Controller.HPA, [1, 1], [0, 10], latency_p95_ms=[50.0, 50.0])
        ppo = _run(Controller.PPO, [1, 1], [0, 10], latency_p95_ms=[90.0, 90.0])
        result = compare(hpa, ppo)
        p95 = next(m for m in result.metrics if m.name == "P95 Latency")
        assert p95.ppo_is_better is False
