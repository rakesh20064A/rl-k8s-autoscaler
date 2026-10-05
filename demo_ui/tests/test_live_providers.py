"""Offline tests for demo_ui/live_providers.py -- every kubectl/Prometheus
call is mocked; no real subprocess or network call happens here."""

from __future__ import annotations

from unittest.mock import patch

from demo_ui.live_providers import (
    KubernetesClusterProvider,
    PrometheusMetricsProvider,
)
from demo_ui.models import PodState


class _FakePrometheusClient:
    def __init__(self, target_up=True, health_msg="up", query_values=None):
        self._target_up = target_up
        self._health_msg = health_msg
        self._query_values = query_values or {}

    def check_target_health(self, namespace, app_name):
        return self._target_up, self._health_msg

    def query(self, promql):
        for key, val in self._query_values.items():
            if key in promql:
                return val, None
        return None, None


class TestPrometheusMetricsProvider:
    def test_target_down_returns_all_none_not_zero(self):
        client = _FakePrometheusClient(target_up=False, health_msg="down")
        provider = PrometheusMetricsProvider(client=client)
        snap = provider.get_live_snapshot()
        assert snap.target_up is False
        assert snap.cpu_cores is None
        assert snap.request_rate_rps is None

    def test_target_up_returns_real_values(self):
        client = _FakePrometheusClient(
            target_up=True,
            query_values={
                "demo_app_requests_total": 12.5,
                "container_cpu_usage_seconds_total": 0.3,
                "container_memory_working_set_bytes": 1024.0,
            },
        )
        provider = PrometheusMetricsProvider(client=client)
        snap = provider.get_live_snapshot()
        assert snap.target_up is True
        assert snap.cpu_cores == 0.3
        assert snap.memory_bytes == 1024.0

    def test_absent_metric_is_none_not_fabricated(self):
        client = _FakePrometheusClient(target_up=True, query_values={})
        provider = PrometheusMetricsProvider(client=client)
        snap = provider.get_live_snapshot()
        assert snap.request_rate_rps is None
        assert snap.latency_p95_ms is None

    def test_never_starts_or_stops_a_port_forward(self):
        # Structural guarantee: the class has no method whose name suggests
        # port-forward lifecycle management.
        forbidden = {"start_port_forward", "stop_port_forward", "establish_route"}
        assert forbidden.isdisjoint(dir(PrometheusMetricsProvider))


def _fake_pods_json(names_states):
    items = []
    for name, ready in names_states:
        items.append({
            "metadata": {"name": name},
            "status": {
                "phase": "Running",
                "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
                "containerStatuses": [{"restartCount": 2}],
            },
        })
    return {"items": items}


class TestKubernetesClusterProvider:
    def test_get_pods_maps_ready_condition(self):
        payload = _fake_pods_json([("pod-a", True), ("pod-b", False)])
        with patch("demo_ui.live_providers._kubectl_json", return_value=payload):
            provider = KubernetesClusterProvider()
            pods = provider.get_pods()
        assert pods[0].state == PodState.READY
        assert pods[1].state == PodState.STARTING

    def test_get_pods_returns_empty_on_kubectl_failure(self):
        with patch("demo_ui.live_providers._kubectl_json", return_value=None):
            provider = KubernetesClusterProvider()
            assert provider.get_pods() == []

    def test_restart_counts_extracted(self):
        payload = _fake_pods_json([("pod-a", True)])
        with patch("demo_ui.live_providers._kubectl_json", return_value=payload):
            provider = KubernetesClusterProvider()
            counts = provider.get_pod_restart_counts()
        assert counts["pod-a"] == 2

    def test_hpa_snapshot_parses_real_shape(self):
        payload = {
            "spec": {"minReplicas": 1, "maxReplicas": 8, "metrics": [{"resource": {"target": {"averageUtilization": 260}}}]},
            "status": {
                "currentReplicas": 2, "desiredReplicas": 2,
                "currentMetrics": [{"resource": {"current": {"averageUtilization": 40}}}],
            },
        }
        with patch("demo_ui.live_providers._kubectl_json", return_value=payload):
            provider = KubernetesClusterProvider()
            hpa = provider.get_hpa_snapshot()
        assert hpa.cpu_target_pct == 260.0
        assert hpa.current_cpu_pct == 40.0
        assert hpa.desired_replicas == 2

    def test_hpa_snapshot_none_on_failure(self):
        with patch("demo_ui.live_providers._kubectl_json", return_value=None):
            provider = KubernetesClusterProvider()
            assert provider.get_hpa_snapshot() is None

    def test_never_issues_a_mutating_kubectl_verb(self):
        import inspect

        import demo_ui.live_providers as mod
        source = inspect.getsource(mod)
        for verb in ['"scale"', '"delete"', '"apply"', '"patch"', '"exec"']:
            assert verb not in source, f"mutating verb {verb} must never appear in live_providers.py"
