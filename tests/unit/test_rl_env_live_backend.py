"""tests/unit/test_rl_env_live_backend.py

Phase 16V: offline tests for autoscaler/src/rl_env/live_backend.py. NO real
Kubernetes/Prometheus access -- every kubectl/HTTP call is replaced with a
deterministic fake via dependency injection (KubectlRunner.run,
PrometheusClient monkeypatch). Proves the backend satisfies the
DynamicsBackend Protocol and its documented failure-handling contract
without ever touching a live cluster.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Dict, List

import pytest

from autoscaler.src.rl_env.backend import DynamicsBackend
from autoscaler.src.rl_env.live_backend import (
    KubectlApiError,
    KubectlRunner,
    LiveKubernetesBackend,
    RestorationTimeoutError,
)
from autoscaler.src.rl_env.types import BackendSnapshot


def _proc(rc: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


@dataclass
class FakeKubectl:
    """Scriptable fake standing in for the real `kubectl` subprocess.
    `replica_state` is mutated by `scale()` to emulate a real cluster's
    eventual convergence (immediately, unless `converge_after_calls` says
    otherwise)."""

    desired: int = 1
    current: int = 1
    ready: int = 1
    available: int = 1
    scale_should_fail: bool = False
    hpa_present: bool = False
    calls: List[List[str]] = field(default_factory=list)

    def run(self, cmd: List[str], timeout: float) -> subprocess.CompletedProcess:
        self.calls.append(cmd)
        if cmd[:2] == ["kubectl", "scale"]:
            if self.scale_should_fail:
                return _proc(rc=1, stderr="connection refused")
            replicas = int(cmd[-1].split("=")[1])
            self.desired = self.current = self.ready = self.available = replicas
            return _proc(rc=0)
        if cmd[:3] == ["kubectl", "get", "deployment"]:
            field_map = {
                "jsonpath={.spec.replicas}": self.desired,
                "jsonpath={.status.replicas}": self.current,
                "jsonpath={.status.readyReplicas}": self.ready,
                "jsonpath={.status.availableReplicas}": self.available,
            }
            return _proc(rc=0, stdout=str(field_map[cmd[-1]]))
        if cmd[:3] == ["kubectl", "get", "hpa"]:
            return _proc(rc=0, stdout="demo-app-hpa   ..." if self.hpa_present else "")
        if cmd[:3] == ["kubectl", "delete", "hpa"]:
            self.hpa_present = False
            return _proc(rc=0)
        if cmd[:2] == ["kubectl", "apply"]:
            self.hpa_present = True
            return _proc(rc=0)
        raise AssertionError(f"unexpected kubectl invocation: {cmd}")


@dataclass
class FakePrometheusClient:
    """Stands in for collectors.metrics_collector.PrometheusClient."""

    target_up: bool = True
    values: Dict[str, float] = field(default_factory=lambda: {
        "request_rate_rps": 5.0, "error_rate_rps": 0.0, "latency_p95_seconds": 0.05,
        "requests_in_progress": 1.0, "cpu_cores": 0.2, "memory_bytes": 40_000_000.0,
    })
    error_on: str = ""  # name of a query to fail, or "" for none
    raise_on_query: bool = False

    # Distinguishing substrings of the ACTUAL PromQL strings live_backend.py
    # generates (see _query_dict) -- the metric names themselves ("cpu_cores")
    # never literally appear in the query text, so matching must use the
    # real Prometheus metric/label fragments.
    _SIGNATURES = {
        "error_rate_rps": 'status_code=~"5..',
        "request_rate_rps": "demo_app_requests_total",  # checked AFTER error_rate_rps (more specific)
        "latency_p95_seconds": "histogram_quantile(0.95",
        "requests_in_progress": "demo_app_requests_in_progress",
        "cpu_cores": "container_cpu_usage_seconds_total",
        "memory_bytes": "container_memory_working_set_bytes",
    }

    def check_target_health(self, namespace: str, app_name: str):
        return self.target_up, "up" if self.target_up else "down"

    def query(self, promql: str):
        if self.raise_on_query:
            raise ConnectionError("prometheus unreachable")
        # error_rate_rps's signature is a strict superset match of
        # request_rate_rps's (both contain "demo_app_requests_total"), so
        # check it first.
        for name in ("error_rate_rps", "request_rate_rps", "latency_p95_seconds",
                     "requests_in_progress", "cpu_cores", "memory_bytes"):
            if self._SIGNATURES[name] in promql:
                if name == self.error_on:
                    return None, "boom"
                if name not in self.values:
                    return None, None
                return self.values[name], None
        raise AssertionError(f"unrecognized PromQL in test fake: {promql}")


def _no_sleep(_seconds: float) -> None:
    return None


def _backend(fake_kubectl: FakeKubectl, fake_prom: FakePrometheusClient, **kwargs) -> LiveKubernetesBackend:
    kubectl = KubectlRunner(run=fake_kubectl.run)
    return LiveKubernetesBackend(kubectl=kubectl, prom_client=fake_prom, sleep_fn=_no_sleep, **kwargs)


class TestProtocolConformance:
    def test_implements_dynamics_backend_protocol(self):
        backend = _backend(FakeKubectl(), FakePrometheusClient())
        assert isinstance(backend, DynamicsBackend)


class TestReset:
    def test_reset_returns_snapshot_at_time_zero(self):
        backend = _backend(FakeKubectl(desired=1, current=1, ready=1, available=1), FakePrometheusClient())
        snap = backend.reset()
        assert isinstance(snap, BackendSnapshot)
        assert snap.time_s == 0.0
        assert snap.replicas.ready == 1
        assert snap.metrics.request_rate_rps == 5.0

    def test_reset_scales_to_initial_replicas_if_not_already_there(self):
        fk = FakeKubectl(desired=3, current=3, ready=3, available=3)
        backend = _backend(fk, FakePrometheusClient(), initial_replicas=1)
        backend.reset()
        assert fk.desired == 1 and fk.ready == 1

    def test_reset_raises_restoration_timeout_if_never_converges(self):
        class NeverConverges(FakeKubectl):
            def run(self, cmd, timeout):
                if cmd[:2] == ["kubectl", "scale"]:
                    return _proc(rc=0)  # accepted, but status never updates
                return super().run(cmd, timeout)

        fk = NeverConverges(desired=5, current=5, ready=5, available=5)
        backend = _backend(fk, FakePrometheusClient(), initial_replicas=1, restoration_max_wait_s=0.01)
        with pytest.raises(RestorationTimeoutError):
            backend.reset()

    def test_reset_never_raises_on_prometheus_failure_reports_target_down_instead(self):
        backend = _backend(FakeKubectl(), FakePrometheusClient(raise_on_query=True))
        snap = backend.reset()  # must not raise
        assert snap.metrics.target_up is True or snap.metrics.request_rate_rps is None
        # raise_on_query means query() itself raises -> caught -> target_up forced False
        assert snap.metrics.target_up is False


class TestAdvance:
    def test_advance_increments_logical_time_and_sleeps_real_seconds(self):
        slept = []
        backend = _backend(FakeKubectl(), FakePrometheusClient())
        backend.reset()
        backend._sleep = slept.append  # type: ignore[attr-defined]  -- swapped AFTER reset's own 60s stabilization sleep
        snap = backend.advance(30.0)
        assert snap.time_s == 30.0
        assert slept == [30.0]

    def test_advance_never_raises_on_target_down(self):
        fake_prom = FakePrometheusClient(target_up=True)
        backend = _backend(FakeKubectl(), fake_prom)
        backend.reset()
        fake_prom.target_up = False
        snap = backend.advance(30.0)
        assert snap.metrics.target_up is False
        assert snap.metrics.request_rate_rps is None

    def test_advance_sets_query_error_on_any_metric_failure(self):
        fake_prom = FakePrometheusClient(error_on="cpu_cores")
        backend = _backend(FakeKubectl(), fake_prom)
        backend.reset()
        snap = backend.advance(30.0)
        assert snap.metrics.query_error is True

    def test_zero_traffic_produces_genuine_zero_error_rate_not_none(self):
        fake_prom = FakePrometheusClient(values={
            "request_rate_rps": 0.0, "latency_p95_seconds": 0.0,
            "requests_in_progress": 0.0, "cpu_cores": 0.01, "memory_bytes": 30_000_000.0,
            # error_rate_rps deliberately absent from the dict -> query() returns (None, None)
        })
        backend = _backend(FakeKubectl(), fake_prom)
        backend.reset()
        snap = backend.advance(30.0)
        assert snap.metrics.request_rate_rps == 0.0
        assert snap.metrics.error_rate_rps == 0.0  # genuine zero, not None

    def test_missing_request_rate_forces_error_rate_none_too(self):
        fake_prom = FakePrometheusClient(error_on="request_rate_rps")
        backend = _backend(FakeKubectl(), fake_prom)
        backend.reset()
        snap = backend.advance(30.0)
        assert snap.metrics.request_rate_rps is None
        assert snap.metrics.error_rate_rps is None

    def test_latency_converted_seconds_to_milliseconds(self):
        fake_prom = FakePrometheusClient(values={
            "request_rate_rps": 5.0, "error_rate_rps": 0.0, "latency_p95_seconds": 0.123,
            "requests_in_progress": 1.0, "cpu_cores": 0.2, "memory_bytes": 40_000_000.0,
        })
        backend = _backend(FakeKubectl(), fake_prom)
        backend.reset()
        snap = backend.advance(30.0)
        assert snap.metrics.latency_p95_ms == pytest.approx(123.0)


class TestSetDesiredReplicas:
    def test_clamps_to_configured_bounds(self):
        fk = FakeKubectl(desired=1, current=1, ready=1, available=1)
        backend = _backend(fk, FakePrometheusClient(), min_replicas=1, max_replicas=8)
        backend.reset()
        backend.set_desired_replicas(99)
        assert fk.desired == 8
        backend.set_desired_replicas(-5)
        assert fk.desired == 1

    def test_raises_kubectl_api_error_on_scale_failure(self):
        fk = FakeKubectl(desired=1, current=1, ready=1, available=1, scale_should_fail=False)
        backend = _backend(fk, FakePrometheusClient())
        backend.reset()
        fk.scale_should_fail = True
        with pytest.raises(KubectlApiError):
            backend.set_desired_replicas(2)

    def test_noop_when_already_at_requested_value(self):
        fk = FakeKubectl(desired=1, current=1, ready=1, available=1)
        backend = _backend(fk, FakePrometheusClient())
        backend.reset()
        n_calls_before = len(fk.calls)
        backend.set_desired_replicas(1)  # already at 1 -- no kubectl scale issued
        scale_calls = [c for c in fk.calls[n_calls_before:] if c[:2] == ["kubectl", "scale"]]
        assert scale_calls == []


class TestKubectlRunnerHpaLifecycle:
    def test_hpa_exists_true_when_present(self):
        fk = FakeKubectl(hpa_present=True)
        runner = KubectlRunner(run=fk.run)
        assert runner.hpa_exists() is True

    def test_hpa_exists_false_when_absent(self):
        fk = FakeKubectl(hpa_present=False)
        runner = KubectlRunner(run=fk.run)
        assert runner.hpa_exists() is False

    def test_delete_then_restore_round_trip(self):
        fk = FakeKubectl(hpa_present=True)
        runner = KubectlRunner(run=fk.run)
        runner.delete_hpa()
        assert fk.hpa_present is False
        runner.restore_hpa()
        assert fk.hpa_present is True
