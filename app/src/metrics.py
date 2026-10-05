# app/src/metrics.py
# Prometheus instrumentation for the demo application.
# Exposes request counters, latency histograms, and an in-progress gauge
# so that kube-prometheus-stack can scrape them from the /metrics endpoint.

from prometheus_client import Counter, Histogram, Gauge

# Total HTTP request count by route and status code
REQUEST_COUNT = Counter(
    "demo_app_requests_total",
    "Total number of HTTP requests",
    ["route", "status_code"],
)

# Request latency histogram (buckets in seconds)
REQUEST_LATENCY = Histogram(
    "demo_app_request_duration_seconds",
    "HTTP request duration in seconds",
    ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Requests currently being processed
REQUESTS_IN_PROGRESS = Gauge(
    "demo_app_requests_in_progress",
    "Number of HTTP requests currently being processed",
    ["route"],
)

# Work intensity of the last /work request — useful for correlating CPU with load
LAST_WORK_INTENSITY = Gauge(
    "demo_app_last_work_intensity",
    "CPU work intensity parameter of the most recent /work request",
)
