"""tests/unit/test_rl_env_observation.py

Phase 15B.1: behavioral tests for autoscaler/src/rl_env/observation.py --
exact numeric normalization, clipping, EWMA, and zero-traffic/invalid
latency semantics. No Kubernetes/Prometheus/network access.
"""

import math

from autoscaler.src.rl_env.observation import (
    EwmaTracker,
    HistoryState,
    ObservationConfig,
    build_observation,
    determine_validity,
)
from autoscaler.src.rl_env.types import (
    LATENCY_STATE_INVALID,
    LATENCY_STATE_NO_TRAFFIC,
    OVERRIDE_PROMETHEUS_UNAVAILABLE,
    OVERRIDE_STALE_OR_MISSING_METRIC,
    RawMetricSnapshot,
    ReplicaState,
)

CFG = ObservationConfig()


def _metrics(**overrides):
    base = dict(
        request_rate_rps=5.0, error_rate_rps=0.0, latency_p95_ms=100.0,
        requests_in_progress=2.0, cpu_cores=0.2, memory_bytes=64_000_000.0,
        target_up=True, query_error=False,
    )
    base.update(overrides)
    return RawMetricSnapshot(**base)


class TestExactShapeAndClipping:
    def test_observation_has_exactly_11_elements(self):
        history = HistoryState.new()
        result = build_observation(_metrics(), ReplicaState(2, 2, 2, 2), 300.0, history, None, CFG)
        assert len(result.normalized) == 11

    def test_cpu_clipped_at_5x_request(self):
        history = HistoryState.new()
        # cpu_cores = 2.0 cores / 2 ready replicas / 0.1 request = 10x -- must clip to 5.0 -> normalized 1.0
        result = build_observation(_metrics(cpu_cores=2.0), ReplicaState(2, 2, 2, 2), 300.0, history, None, CFG)
        assert result.normalized[0] == 1.0

    def test_cpu_delta_clipped_symmetric_5x(self):
        history = HistoryState.new()
        build_observation(_metrics(cpu_cores=0.1), ReplicaState(1, 1, 1, 1), 300.0, history, None, CFG)
        # Huge jump: cpu_cores=5.0/1 replica/0.1 request = 50x -- delta clipped to +-5 -> normalized +1.0
        result = build_observation(_metrics(cpu_cores=5.0), ReplicaState(1, 1, 1, 1), 300.0, history, [0.5] * 11, CFG)
        assert result.normalized[1] == 1.0

    def test_request_rate_normalization_exact(self):
        history = HistoryState.new()
        result = build_observation(_metrics(request_rate_rps=25.0), ReplicaState(1, 1, 1, 1), 300.0, history, None, CFG)
        assert math.isclose(result.normalized[4], 0.25)

    def test_request_rate_clipped_above_100(self):
        history = HistoryState.new()
        result = build_observation(_metrics(request_rate_rps=500.0), ReplicaState(1, 1, 1, 1), 300.0, history, None, CFG)
        assert result.normalized[4] == 1.0

    def test_latency_normalization_exact(self):
        history = HistoryState.new()
        result = build_observation(_metrics(latency_p95_ms=600.0), ReplicaState(1, 1, 1, 1), 300.0, history, None, CFG)
        assert math.isclose(result.normalized[6], 0.3)

    def test_replica_normalization_exact(self):
        history = HistoryState.new()
        # ready=4 -> (4-1)/(8-1) = 3/7
        result = build_observation(_metrics(), ReplicaState(4, 4, 4, 4), 300.0, history, None, CFG)
        assert math.isclose(result.normalized[9], 3.0 / 7.0)

    def test_seconds_since_scale_cap_exact(self):
        history = HistoryState.new()
        result = build_observation(_metrics(), ReplicaState(1, 1, 1, 1), 150.0, history, None, CFG)
        assert math.isclose(result.normalized[10], 0.5)


class TestZeroTrafficAndInvalidLatency:
    def test_zero_traffic_no_histogram_is_valid_semantic_zero(self):
        v = determine_validity(_metrics(request_rate_rps=0.0, latency_p95_ms=None))
        assert v.sample_valid is True
        assert v.latency_observation_state == LATENCY_STATE_NO_TRAFFIC

    def test_zero_traffic_observation_latency_is_zero_ms(self):
        history = HistoryState.new()
        result = build_observation(
            _metrics(request_rate_rps=0.0, latency_p95_ms=None), ReplicaState(1, 1, 1, 1), 300.0, history, None, CFG
        )
        assert result.normalized[6] == 0.0
        assert result.validity.latency_observation_state == LATENCY_STATE_NO_TRAFFIC
        assert result.validity.sample_valid is True

    def test_positive_traffic_missing_latency_is_invalid(self):
        v = determine_validity(_metrics(request_rate_rps=5.0, latency_p95_ms=None))
        assert v.sample_valid is False
        assert v.latency_observation_state == LATENCY_STATE_INVALID
        assert v.override_reason == OVERRIDE_STALE_OR_MISSING_METRIC

    def test_positive_traffic_missing_latency_holds_previous_observation(self):
        history = HistoryState.new()
        previous = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0]
        result = build_observation(
            _metrics(request_rate_rps=5.0, latency_p95_ms=None), ReplicaState(1, 1, 1, 1), 300.0, history, previous, CFG
        )
        # Every application-derived feature is held at its previous value.
        assert result.normalized[:9] == previous[:9]
        assert result.validity.sample_valid is False

    def test_target_down_is_prometheus_unavailable(self):
        v = determine_validity(_metrics(target_up=False))
        assert v.sample_valid is False
        assert v.override_reason == OVERRIDE_PROMETHEUS_UNAVAILABLE

    def test_ready_replicas_zero_yields_missing_per_replica_ratios(self):
        history = HistoryState.new()
        previous = [0.42] * 11
        result = build_observation(_metrics(), ReplicaState(1, 1, 0, 0), 300.0, history, previous, CFG)
        # cpu/memory/in_progress (indices 0,2,3) held; request_rate/latency/error/ready still compute.
        assert result.normalized[0] == previous[0]
        assert result.normalized[2] == previous[2]
        assert result.normalized[3] == previous[3]
        assert result.normalized[9] == 0.0  # ready=0 -> (0-1)/7 clipped to 0 -> normalized 0.0


class TestEwmaTracker:
    def test_alpha_is_exactly_half_for_span_3(self):
        tracker = EwmaTracker(span_steps=3)
        assert math.isclose(tracker.alpha, 0.5)

    def test_first_value_initializes_ewma_delta_zero(self):
        tracker = EwmaTracker(span_steps=3)
        delta = tracker.update(10.0)
        assert delta == 0.0

    def test_second_value_delta_and_ewma_update_exact(self):
        tracker = EwmaTracker(span_steps=3)
        tracker.update(10.0)  # ewma = 10.0
        delta = tracker.update(20.0)  # delta = 20 - 10 = 10; ewma = 0.5*20 + 0.5*10 = 15
        assert math.isclose(delta, 10.0)
        assert math.isclose(tracker._ewma, 15.0)

    def test_positive_trend_produces_positive_deltas(self):
        tracker = EwmaTracker(span_steps=3)
        tracker.update(10.0)
        d1 = tracker.update(20.0)
        d2 = tracker.update(30.0)
        assert d1 > 0
        assert d2 > 0

    def test_negative_trend_produces_negative_deltas(self):
        tracker = EwmaTracker(span_steps=3)
        tracker.update(30.0)
        d1 = tracker.update(20.0)
        d2 = tracker.update(10.0)
        assert d1 < 0
        assert d2 < 0

    def test_invalid_step_does_not_update_ewma(self):
        tracker = EwmaTracker(span_steps=3)
        tracker.update(10.0)
        tracker.update(20.0)
        ewma_before = tracker._ewma
        delta = tracker.update(None)
        assert delta is None
        assert tracker._ewma == ewma_before

    def test_history_state_ewma_integration_across_build_observation_calls(self):
        history = HistoryState.new()
        # Step 1: cpu_per_replica raw = 0.1/1/0.1 = 1.0
        r1 = build_observation(_metrics(cpu_cores=0.1), ReplicaState(1, 1, 1, 1), 300.0, history, None, CFG)
        assert r1.normalized[1] == 0.0  # first step: delta 0
        # Step 2: cpu_cores = 0.2 -> raw = 2.0; delta = 2.0 - 1.0 = 1.0 -> normalized 1.0/5=0.2
        r2 = build_observation(_metrics(cpu_cores=0.2), ReplicaState(1, 1, 1, 1), 300.0, history, r1.normalized, CFG)
        assert math.isclose(r2.normalized[1], 0.2)
