"""tests/unit/test_v2_drain_retry.py

V2 Phase 3D PART 5: proves the bounded-retry drain wrapper -- transient
Prometheus failures are tolerated with backoff, a genuine sustained outage
still fails closed, and the frozen `drain_in_flight_requests()` itself is
never modified (V1's frozen scripts/evaluate_ppo_vs_hpa.py calls it
directly and must be byte-for-byte unaffected).
"""

from __future__ import annotations

import pytest

from scripts import traffic_orchestrator as to
from scripts.v2_traffic_orchestrator import (
    DEFAULT_DRAIN_QUERY_RETRY_BACKOFF_SECONDS,
    DEFAULT_DRAIN_QUERY_RETRY_BUDGET_SECONDS,
    V2TrafficOrchestrator,
    drain_in_flight_requests_with_retry,
)


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class TestSuccessfulDrainUnchanged:
    def test_immediate_success_no_retry_needed(self):
        calls = []

        def query_fn():
            calls.append(1)
            return 0.0, None

        clock = _FakeClock()
        result = drain_in_flight_requests_with_retry(
            query_fn, timeout_seconds=5.0, poll_interval_seconds=1.0,
            sleep_fn=clock.sleep, clock=clock,
        )
        assert result == 0.0
        assert len(calls) == 1  # no retry overhead on the happy path

    def test_drains_after_normal_poll_within_single_attempt(self):
        """Normal (non-error) polling behavior -- values decreasing to zero
        -- must behave identically to the unwrapped function; the retry
        wrapper must never interfere with a successful-but-slow drain."""
        values = iter([3.0, 1.0, 0.0])

        def query_fn():
            return next(values), None

        clock = _FakeClock()
        result = drain_in_flight_requests_with_retry(
            query_fn, timeout_seconds=30.0, poll_interval_seconds=1.0,
            sleep_fn=clock.sleep, clock=clock,
        )
        assert result == 0.0


class TestTransientFailureRecovery:
    def test_one_query_error_then_recovery_succeeds(self):
        calls = {"n": 0}

        def query_fn():
            calls["n"] += 1
            if calls["n"] == 1:
                return None, "connection refused"
            return 0.0, None

        clock = _FakeClock()
        result = drain_in_flight_requests_with_retry(
            query_fn, timeout_seconds=5.0, poll_interval_seconds=1.0,
            sleep_fn=clock.sleep, clock=clock, retry_budget_seconds=30.0, retry_backoff_seconds=2.0,
        )
        assert result == 0.0
        assert calls["n"] == 2

    def test_http_5xx_style_error_retries(self):
        calls = {"n": 0}

        def query_fn():
            calls["n"] += 1
            if calls["n"] <= 2:
                return None, "HTTP 503"
            return 0.0, None

        clock = _FakeClock()
        result = drain_in_flight_requests_with_retry(
            query_fn, timeout_seconds=5.0, poll_interval_seconds=1.0,
            sleep_fn=clock.sleep, clock=clock, retry_budget_seconds=30.0, retry_backoff_seconds=2.0,
        )
        assert result == 0.0
        assert calls["n"] == 3

    def test_multiple_failures_then_successful_zero_in_flight(self):
        calls = {"n": 0}

        def query_fn():
            calls["n"] += 1
            if calls["n"] <= 4:
                return None, "network error"
            return 0.0, None

        clock = _FakeClock()
        result = drain_in_flight_requests_with_retry(
            query_fn, timeout_seconds=5.0, poll_interval_seconds=1.0,
            sleep_fn=clock.sleep, clock=clock, retry_budget_seconds=60.0, retry_backoff_seconds=5.0,
        )
        assert result == 0.0
        assert calls["n"] == 5

    def test_backoff_interval_is_honored(self):
        """Confirms sleep_fn is actually called with the configured backoff
        between retries (not busy-looping)."""
        sleeps = []

        def fake_sleep(seconds):
            sleeps.append(seconds)
            clock.t += seconds

        clock = _FakeClock()
        calls = {"n": 0}

        def query_fn():
            calls["n"] += 1
            if calls["n"] == 1:
                return None, "connection refused"
            return 0.0, None

        drain_in_flight_requests_with_retry(
            query_fn, timeout_seconds=5.0, poll_interval_seconds=1.0,
            sleep_fn=fake_sleep, clock=clock, retry_budget_seconds=30.0, retry_backoff_seconds=7.5,
        )
        assert sleeps == [7.5]


class TestSustainedFailureFailsClosed:
    def test_prometheus_never_recovers_raises_drain_query_error(self):
        def query_fn():
            return None, "connection refused"

        clock = _FakeClock()
        with pytest.raises(to.DrainQueryError):
            drain_in_flight_requests_with_retry(
                query_fn, timeout_seconds=5.0, poll_interval_seconds=1.0,
                sleep_fn=clock.sleep, clock=clock, retry_budget_seconds=20.0, retry_backoff_seconds=5.0,
            )

    def test_unavailable_prometheus_never_interpreted_as_zero_in_flight(self):
        """The central safety property: an error must NEVER be silently
        treated as 'drained'."""
        calls = {"n": 0}

        def query_fn():
            calls["n"] += 1
            return None, "connection refused"

        clock = _FakeClock()
        with pytest.raises(to.DrainQueryError):
            drain_in_flight_requests_with_retry(
                query_fn, timeout_seconds=5.0, poll_interval_seconds=1.0,
                sleep_fn=clock.sleep, clock=clock, retry_budget_seconds=15.0, retry_backoff_seconds=5.0,
            )
        assert calls["n"] >= 2  # genuinely retried, not given up after one try

    def test_drain_timeout_error_never_retried(self):
        """A genuine 'requests never drained' signal (DrainTimeoutError,
        NOT a query failure) must propagate immediately, unchanged -- this
        is a different, non-transient condition."""
        calls = {"n": 0}

        def query_fn():
            calls["n"] += 1
            return 999.0, None  # always "still in flight", never an error

        clock = _FakeClock()
        with pytest.raises(to.DrainTimeoutError):
            drain_in_flight_requests_with_retry(
                query_fn, timeout_seconds=3.0, poll_interval_seconds=1.0,
                sleep_fn=clock.sleep, clock=clock, retry_budget_seconds=60.0, retry_backoff_seconds=5.0,
            )
        # Only the ONE underlying drain_in_flight_requests call's own
        # internal polling happened -- no retry-wrapper-level re-invocation.
        assert calls["n"] <= 4  # bounded by the 3s timeout / 1s poll interval, not by retry_budget_seconds=60


class TestDefaultsAndConstants:
    def test_default_retry_budget_is_75_seconds(self):
        assert DEFAULT_DRAIN_QUERY_RETRY_BUDGET_SECONDS == 75.0

    def test_default_backoff_is_5_seconds(self):
        assert DEFAULT_DRAIN_QUERY_RETRY_BACKOFF_SECONDS == 5.0

    def test_retry_budget_within_60_to_90_second_guidance(self):
        assert 60.0 <= DEFAULT_DRAIN_QUERY_RETRY_BUDGET_SECONDS <= 90.0


class TestFrozenDrainFunctionUntouched:
    def test_drain_in_flight_requests_still_raises_immediately_on_error(self):
        """The ORIGINAL, frozen-adjacent function (used directly by V1's
        frozen scripts/evaluate_ppo_vs_hpa.py) must still fail on the VERY
        FIRST error, with zero retry behavior -- proves it was not modified
        to add retries itself."""
        def query_fn():
            return None, "connection refused"

        with pytest.raises(to.DrainQueryError):
            to.drain_in_flight_requests(query_fn, timeout_seconds=5.0, poll_interval_seconds=1.0, sleep_fn=lambda s: None)

    def test_v2_orchestrator_uses_retry_wrapper_not_frozen_function_directly(self):
        orchestrator = V2TrafficOrchestrator()
        calls = {"n": 0}

        def fake_query():
            calls["n"] += 1
            if calls["n"] == 1:
                return None, "connection refused"
            return 0.0, None

        orchestrator.drain_query_fn = fake_query
        orchestrator.sleep_fn = lambda s: None
        orchestrator.drain_timeout_seconds = 5.0
        orchestrator.drain_poll_interval_seconds = 1.0
        result = orchestrator.drain_server_side_requests()
        assert result == 0.0
        assert calls["n"] == 2  # retried -- would have raised on the first call via the unwrapped function


class TestTelemetryUnaffected:
    def test_retry_wrapper_has_no_reward_or_observation_dependency(self):
        """Structural proof: the retry module touches only traffic/drain
        plumbing, never autoscaler.src.rl_env.reward_v2 or .observation."""
        import inspect

        import scripts.v2_traffic_orchestrator as mod

        source = inspect.getsource(mod)
        assert "reward_v2" not in source
        assert "compute_reward" not in source
        assert "observation" not in source.lower()
