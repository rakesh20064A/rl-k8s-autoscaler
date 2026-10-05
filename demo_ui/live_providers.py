"""live_providers.py -- REAL, READ-ONLY data providers (sec.8-9 of
Index32A).

Both classes here issue only read operations:
  - PrometheusMetricsProvider: instant PromQL queries via the project's own
    `collectors.metrics_collector.PrometheusClient` (imported, not
    reimplemented) -- the exact metric names already used by
    `collectors/metrics_collector.py` / `autoscaler/src/rl_env/live_backend.py`,
    never invented. It does NOT start/stop/manage the Prometheus
    port-forward itself (mirrors LiveKubernetesBackend's own documented
    boundary) -- the caller must have one already running.
  - KubernetesClusterProvider: `kubectl get ...` (read verbs only --
    `get`/no mutating subcommand ever appears in this file) for pods,
    deployment replica state, HPA status, and recent events.

Neither class can remove the HPA, scale the Deployment, start traffic, or
launch PPO -- no such method exists here.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from collectors.metrics_collector import PrometheusClient
from demo_ui.models import HpaSnapshot, PodState, PodStatus

NOT_AVAILABLE = "Not available"

DEFAULT_APP_NAMESPACE = "autoscaler-app"
DEFAULT_DEPLOYMENT = "demo-app"
DEFAULT_HPA_NAME = "demo-app-hpa"
DEFAULT_ROUTE = "/work"

_POD_STATE_MAP = {
    # "Running" is handled explicitly in get_pods() using the Ready
    # condition, not via this table.
    "Pending": PodState.STARTING,
    "ContainerCreating": PodState.STARTING,
    "Terminating": PodState.TERMINATING,
    "Failed": PodState.FAILED,
    "CrashLoopBackOff": PodState.FAILED,
    "Error": PodState.FAILED,
}


@dataclass
class LiveMetricsSnapshot:
    """One instant read from Prometheus. Any field Prometheus cannot answer
    right now is left as None -- the UI renders `Not available`, never a
    fabricated number. This is NOT a RunSummary: it has no time series,
    because building real historical series requires a running sampler,
    out of scope for this read-only provider."""

    target_up: bool
    target_health_message: str
    request_rate_rps: Optional[float]
    success_rate_rps: Optional[float]
    error_rate_rps: Optional[float]
    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]
    latency_p99_ms: Optional[float]
    requests_in_progress: Optional[float]
    cpu_cores: Optional[float]
    memory_bytes: Optional[float]


class PrometheusMetricsProvider:
    """Real, read-only Prometheus reader. Requires an externally-managed
    port-forward to already be running (never started/stopped by this
    class) -- same documented boundary as LiveKubernetesBackend."""

    def __init__(
        self,
        base_url: str = "http://localhost:9090",
        app_namespace: str = DEFAULT_APP_NAMESPACE,
        deployment: str = DEFAULT_DEPLOYMENT,
        route: str = DEFAULT_ROUTE,
        client: Optional[PrometheusClient] = None,
    ) -> None:
        self.app_namespace = app_namespace
        self.deployment = deployment
        self.route = route
        self._client = client or PrometheusClient(base_url=base_url)

    def get_live_snapshot(self) -> LiveMetricsSnapshot:
        target_up, message = self._client.check_target_health(
            namespace=self.app_namespace, app_name=self.deployment
        )
        if not target_up:
            return LiveMetricsSnapshot(
                target_up=False, target_health_message=message,
                request_rate_rps=None, success_rate_rps=None, error_rate_rps=None,
                latency_p50_ms=None, latency_p95_ms=None, latency_p99_ms=None,
                requests_in_progress=None, cpu_cores=None, memory_bytes=None,
            )

        ns, dep, route = self.app_namespace, self.deployment, self.route
        # Exact query strings, reused verbatim from
        # collectors/metrics_collector.py:368-381 (never invented here).
        queries = {
            "request_rate_rps": f'sum(rate(demo_app_requests_total{{namespace="{ns}",route="{route}"}}[1m]))',
            "success_rate_rps": f'sum(rate(demo_app_requests_total{{namespace="{ns}",route="{route}",status_code=~"2.."}}[1m]))',
            "error_rate_rps": f'sum(rate(demo_app_requests_total{{namespace="{ns}",route="{route}",status_code=~"5.."}}[1m]))',
            "latency_p50_seconds": f'histogram_quantile(0.50, sum(rate(demo_app_request_duration_seconds_bucket{{namespace="{ns}",route="{route}"}}[1m])) by (le))',
            "latency_p95_seconds": f'histogram_quantile(0.95, sum(rate(demo_app_request_duration_seconds_bucket{{namespace="{ns}",route="{route}"}}[1m])) by (le))',
            "latency_p99_seconds": f'histogram_quantile(0.99, sum(rate(demo_app_request_duration_seconds_bucket{{namespace="{ns}",route="{route}"}}[1m])) by (le))',
            "requests_in_progress": f'sum(demo_app_requests_in_progress{{namespace="{ns}",route="{route}"}})',
            "cpu_cores": f'sum(rate(container_cpu_usage_seconds_total{{namespace="{ns}",pod=~"{dep}-.*"}}[1m]))',
            "memory_bytes": f'sum(container_memory_working_set_bytes{{namespace="{ns}",pod=~"{dep}-.*"}})',
        }
        values = {}
        for name, promql in queries.items():
            val, _err = self._client.query(promql)
            values[name] = val

        def _ms(seconds: Optional[float]) -> Optional[float]:
            return None if seconds is None else seconds * 1000.0

        return LiveMetricsSnapshot(
            target_up=True, target_health_message=message,
            request_rate_rps=values["request_rate_rps"],
            success_rate_rps=values["success_rate_rps"],
            error_rate_rps=values["error_rate_rps"],
            latency_p50_ms=_ms(values["latency_p50_seconds"]),
            latency_p95_ms=_ms(values["latency_p95_seconds"]),
            latency_p99_ms=_ms(values["latency_p99_seconds"]),
            requests_in_progress=values["requests_in_progress"],
            cpu_cores=values["cpu_cores"],
            memory_bytes=values["memory_bytes"],
        )


def _kubectl_json(args: List[str], timeout_s: float = 10.0) -> Optional[dict]:
    """Runs a READ-ONLY `kubectl get ... -o json` and parses it. Returns
    None on any failure (process error, non-JSON, timeout) -- callers must
    render `Not available`, never fabricate a value."""
    try:
        proc = subprocess.run(
            ["kubectl"] + args, capture_output=True, text=True, timeout=timeout_s,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return None


class KubernetesClusterProvider:
    """Real, read-only Kubernetes reader. Every method here issues exactly
    one `kubectl get ... -o json` (a read verb) and nothing else -- no
    `scale`, `delete`, `apply`, `patch`, or `exec` appears anywhere in this
    class."""

    def __init__(self, app_namespace: str = DEFAULT_APP_NAMESPACE, deployment: str = DEFAULT_DEPLOYMENT, hpa_name: str = DEFAULT_HPA_NAME) -> None:
        self.app_namespace = app_namespace
        self.deployment = deployment
        self.hpa_name = hpa_name

    def get_pods(self) -> List[PodStatus]:
        data = _kubectl_json([
            "get", "pods", "-n", self.app_namespace,
            "-l", f"app={self.deployment}", "-o", "json",
        ])
        if data is None:
            return []
        pods: List[PodStatus] = []
        for item in data.get("items", []):
            name = item.get("metadata", {}).get("name", "unknown")
            phase = item.get("status", {}).get("phase", "Unknown")
            conditions = item.get("status", {}).get("conditions", [])
            ready_cond = next((c for c in conditions if c.get("type") == "Ready"), None)
            if item.get("metadata", {}).get("deletionTimestamp"):
                state = PodState.TERMINATING
            elif phase in ("Failed",):
                state = PodState.FAILED
            elif ready_cond is not None and ready_cond.get("status") == "True":
                state = PodState.READY
            elif phase == "Running":
                # Running but NOT Ready (e.g. failing readiness probe) is
                # reported as STARTING, never silently upgraded to READY.
                state = PodState.STARTING
            else:
                state = _POD_STATE_MAP.get(phase, PodState.STARTING)
            pods.append(PodStatus(name=name, state=state))
        return pods

    def get_pod_restart_counts(self) -> dict:
        """pod name -> total container restart count, or {} if unavailable."""
        data = _kubectl_json([
            "get", "pods", "-n", self.app_namespace,
            "-l", f"app={self.deployment}", "-o", "json",
        ])
        if data is None:
            return {}
        counts = {}
        for item in data.get("items", []):
            name = item.get("metadata", {}).get("name", "unknown")
            statuses = item.get("status", {}).get("containerStatuses", [])
            counts[name] = sum(cs.get("restartCount", 0) for cs in statuses)
        return counts

    def get_deployment_replica_state(self) -> Optional[dict]:
        data = _kubectl_json([
            "get", "deployment", self.deployment, "-n", self.app_namespace, "-o", "json",
        ])
        if data is None:
            return None
        spec = data.get("spec", {})
        status = data.get("status", {})
        return {
            "desired": spec.get("replicas"),
            "ready": status.get("readyReplicas"),
            "available": status.get("availableReplicas"),
        }

    def get_hpa_snapshot(self) -> Optional[HpaSnapshot]:
        data = _kubectl_json([
            "get", "hpa", self.hpa_name, "-n", self.app_namespace, "-o", "json",
        ])
        if data is None:
            return None
        spec = data.get("spec", {})
        status = data.get("status", {})
        metrics = spec.get("metrics", [{}])
        cpu_target = metrics[0].get("resource", {}).get("target", {}).get("averageUtilization") if metrics else None
        current_metrics = status.get("currentMetrics", [{}])
        current_cpu = (
            current_metrics[0].get("resource", {}).get("current", {}).get("averageUtilization")
            if current_metrics else None
        )
        return HpaSnapshot(
            cpu_target_pct=float(cpu_target) if cpu_target is not None else float("nan"),
            current_cpu_pct=float(current_cpu) if current_cpu is not None else float("nan"),
            current_replicas=status.get("currentReplicas", 0),
            desired_replicas=status.get("desiredReplicas", 0),
            min_replicas=spec.get("minReplicas", 0),
            max_replicas=spec.get("maxReplicas", 0),
            latest_scaling_event=None,  # sec.10: not available from HPA status alone, never fabricated
        )

    def get_recent_events(self, limit: int = 20) -> List[dict]:
        data = _kubectl_json([
            "get", "events", "-n", self.app_namespace,
            "--field-selector", f"involvedObject.name={self.deployment}",
            "--sort-by", ".lastTimestamp", "-o", "json",
        ])
        if data is None:
            return []
        items = data.get("items", [])[-limit:]
        return [
            {
                "reason": e.get("reason", ""),
                "message": e.get("message", ""),
                "lastTimestamp": e.get("lastTimestamp", ""),
                "type": e.get("type", ""),
            }
            for e in items
        ]
