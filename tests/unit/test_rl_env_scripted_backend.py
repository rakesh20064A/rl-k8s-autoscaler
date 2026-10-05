"""tests/unit/test_rl_env_scripted_backend.py

Phase 15B.1: behavioral tests for autoscaler/src/rl_env/scripted_backend.py
-- replica-transition mechanics (delayed scale-up readiness, immediate
scale-down) and TEST_ONLY_SYNTHETIC_METRICS timeline playback. No wall-clock
sleeping -- all "time" is logical/deterministic.
"""

from autoscaler.src.rl_env.scripted_backend import TEST_ONLY_SYNTHETIC_METRICS, ScriptedBackend
from autoscaler.src.rl_env.types import RawMetricSnapshot


def _snap(**overrides):
    base = dict(
        request_rate_rps=5.0, error_rate_rps=0.0, latency_p95_ms=100.0,
        requests_in_progress=1.0, cpu_cores=0.15, memory_bytes=30_000_000.0,
    )
    base.update(overrides)
    return RawMetricSnapshot(**base)


def test_module_is_explicitly_marked_test_only():
    assert TEST_ONLY_SYNTHETIC_METRICS is True


class TestReset:
    def test_reset_starts_at_time_zero_and_initial_replicas(self):
        backend = ScriptedBackend(timeline={0.0: _snap()}, initial_replicas=1)
        snapshot = backend.reset()
        assert snapshot.time_s == 0.0
        assert snapshot.replicas.desired == snapshot.replicas.current == snapshot.replicas.ready == snapshot.replicas.available == 1

    def test_reset_is_idempotent(self):
        backend = ScriptedBackend(timeline={0.0: _snap(), 30.0: _snap()}, initial_replicas=1)
        backend.reset()
        backend.set_desired_replicas(2)
        backend.advance(30.0)
        snapshot = backend.reset()
        assert snapshot.time_s == 0.0
        assert snapshot.replicas.desired == 1


class TestScaleUpMechanics:
    def test_desired_and_current_update_immediately_ready_does_not(self):
        backend = ScriptedBackend(timeline={0.0: _snap()}, initial_replicas=1, scale_up_ready_delay_seconds=10.0)
        backend.reset()
        backend.set_desired_replicas(2)
        snapshot = backend.current_snapshot()
        assert snapshot.replicas.desired == 2
        assert snapshot.replicas.current == 2
        assert snapshot.replicas.ready == 1
        assert snapshot.replicas.available == 1

    def test_ready_realized_after_delay_within_one_advance(self):
        timeline = {0.0: _snap(), 30.0: _snap()}
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1, scale_up_ready_delay_seconds=10.0)
        backend.reset()
        backend.set_desired_replicas(2)
        snapshot = backend.advance(30.0)  # readiness scheduled at t=10, resolved within (0,30]
        assert snapshot.time_s == 30.0
        assert snapshot.replicas.ready == 2
        assert snapshot.replicas.available == 2

    def test_ready_not_yet_realized_before_delay_elapses(self):
        timeline = {0.0: _snap(), 5.0: _snap()}
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1, scale_up_ready_delay_seconds=10.0)
        backend.reset()
        backend.set_desired_replicas(2)
        snapshot = backend.advance(5.0)  # readiness scheduled at t=10, not yet reached at t=5
        assert snapshot.replicas.ready == 1
        assert snapshot.replicas.current == 2

    def test_readiness_delay_is_configurable(self):
        timeline = {0.0: _snap(), 3.0: _snap()}
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1, scale_up_ready_delay_seconds=2.0)
        backend.reset()
        backend.set_desired_replicas(2)
        snapshot = backend.advance(3.0)
        assert snapshot.replicas.ready == 2


class TestScaleDownMechanics:
    def test_scale_down_all_fields_update_immediately(self):
        backend = ScriptedBackend(timeline={0.0: _snap()}, initial_replicas=2)
        backend.reset()
        backend.set_desired_replicas(1)
        snapshot = backend.current_snapshot()
        assert snapshot.replicas.desired == 1
        assert snapshot.replicas.current == 1
        assert snapshot.replicas.ready == 1
        assert snapshot.replicas.available == 1


class TestBoundsDefensiveClamp:
    def test_set_desired_replicas_clamps_to_max(self):
        backend = ScriptedBackend(timeline={0.0: _snap()}, initial_replicas=1, max_replicas=8)
        backend.reset()
        backend.set_desired_replicas(999)
        assert backend.current_snapshot().replicas.desired == 8

    def test_set_desired_replicas_clamps_to_min(self):
        backend = ScriptedBackend(timeline={0.0: _snap()}, initial_replicas=1, min_replicas=1)
        backend.reset()
        backend.set_desired_replicas(-5)
        assert backend.current_snapshot().replicas.desired == 1


class TestTimelinePlayback:
    def test_missing_timeline_entry_yields_target_down_and_query_error(self):
        backend = ScriptedBackend(timeline={0.0: _snap()}, initial_replicas=1)
        backend.reset()
        snapshot = backend.advance(30.0)  # no entry scripted at t=30
        assert snapshot.metrics.target_up is False
        assert snapshot.metrics.query_error is True
        assert snapshot.metrics.cpu_cores is None

    def test_exact_time_lookup_returns_scripted_values(self):
        distinctive = _snap(cpu_cores=0.42)
        backend = ScriptedBackend(timeline={0.0: _snap(), 30.0: distinctive}, initial_replicas=1)
        backend.reset()
        snapshot = backend.advance(30.0)
        assert snapshot.metrics.cpu_cores == 0.42

    def test_empty_timeline_rejected(self):
        import pytest
        with pytest.raises(ValueError):
            ScriptedBackend(timeline={})
