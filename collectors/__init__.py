"""Metrics collector package for Kubernetes autoscaling experiments."""

__all__ = ["MetricsCollector", "PrometheusClient"]


def __getattr__(name: str):
    if name in ("MetricsCollector", "PrometheusClient"):
        from collectors.metrics_collector import MetricsCollector, PrometheusClient
        return {"MetricsCollector": MetricsCollector, "PrometheusClient": PrometheusClient}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

