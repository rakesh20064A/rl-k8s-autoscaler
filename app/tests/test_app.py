# app/tests/test_app.py
# Automated test suite for the RL autoscaler demo FastAPI application.
#
# Design constraints:
#   - No timing assertions (execution time is machine-dependent).
#   - No heavy workloads in unit tests (intensity >= 7 used at most once, intensity=10 never).
#   - No live Kubernetes, Prometheus, or Docker required.
#   - Uses FastAPI TestClient (in-process, no network port opened).
#
# Run from project root:
#   .venv/bin/pytest app/tests/ -v

import inspect
import re

import pytest
from fastapi.testclient import TestClient

try:
    from app.src.main import app
except ImportError:
    import sys
    sys.path.insert(0, "app/src")
    from main import app  # type: ignore[import]


@pytest.fixture(scope="module")
def client():
    """Module-scoped client — app lifespan runs once for the entire module."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _work_count(client) -> float:
    """Return current value of demo_app_requests_total for /work status=200."""
    r = client.get("/metrics")
    m = re.search(
        r'demo_app_requests_total\{route="/work",status_code="200"\}\s+([\d.]+)',
        r.text,
    )
    return float(m.group(1)) if m else 0.0


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_status_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_body(self, client):
        r = client.get("/health")
        assert r.json() == {"status": "ok"}

    def test_health_handler_is_a_coroutine_function(self, client):
        """Index33 correction regression test: `/health` must be `async def`,
        not `def`, so FastAPI/Starlette serves it directly on the event loop
        and it never requests a token from the shared AnyIO sync-worker
        thread pool that `/work` can saturate (Index32 diagnosis). This is a
        deterministic structural check -- not a timing/performance
        benchmark -- so it cannot be flaky under machine load."""
        route = next(r for r in client.app.routes if getattr(r, "path", None) == "/health")
        assert inspect.iscoroutinefunction(route.endpoint), (
            "/health must be `async def` so it is never routed through the "
            "shared sync-endpoint worker-thread pool that /work can saturate"
        )

    def test_work_handler_remains_a_sync_function(self, client):
        """Regression guard: Correction 1 must change ONLY /health. /work's
        execution model (sync def, offloaded to the thread pool) is
        unchanged -- the scientific CPU-bound workload itself is untouched."""
        route = next(r for r in client.app.routes if getattr(r, "path", None) == "/work")
        assert not inspect.iscoroutinefunction(route.endpoint)


# ---------------------------------------------------------------------------
# / (root)
# ---------------------------------------------------------------------------

class TestRoot:
    def test_status_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_response_body(self, client):
        body = client.get("/").json()
        assert body.get("service") == "demo-app"
        assert body.get("status") == "running"


# ---------------------------------------------------------------------------
# /work — valid requests
# ---------------------------------------------------------------------------

class TestWork:
    def test_low_intensity_returns_200(self, client):
        r = client.get("/work?intensity=1")
        assert r.status_code == 200

    def test_response_contains_required_fields(self, client):
        body = client.get("/work?intensity=2").json()
        for field in ("intensity", "elapsed_ms", "iterations_completed", "result"):
            assert field in body, f"Missing field: {field}"

    def test_intensity_echoed_in_response(self, client):
        for intensity in (1, 3, 5):
            body = client.get(f"/work?intensity={intensity}").json()
            assert body["intensity"] == intensity

    def test_elapsed_ms_positive(self, client):
        body = client.get("/work?intensity=1").json()
        assert body["elapsed_ms"] > 0

    def test_iterations_positive(self, client):
        body = client.get("/work?intensity=1").json()
        assert body["iterations_completed"] > 0

    def test_result_is_float(self, client):
        body = client.get("/work?intensity=2").json()
        assert isinstance(body["result"], float)

    def test_default_intensity_is_3(self, client):
        body = client.get("/work").json()
        assert body["intensity"] == 3

    def test_boundary_intensity_1_accepted(self, client):
        assert client.get("/work?intensity=1").status_code == 200

    def test_boundary_intensity_10_accepted(self, client):
        assert client.get("/work?intensity=10").status_code == 200


# ---------------------------------------------------------------------------
# /work — input validation (must reject with 422)
# ---------------------------------------------------------------------------

class TestWorkInputValidation:
    def test_intensity_below_1_rejected(self, client):
        assert client.get("/work?intensity=0").status_code == 422

    def test_negative_intensity_rejected(self, client):
        assert client.get("/work?intensity=-5").status_code == 422

    def test_intensity_above_10_rejected(self, client):
        assert client.get("/work?intensity=11").status_code == 422

    def test_non_integer_rejected(self, client):
        assert client.get("/work?intensity=abc").status_code == 422

    def test_float_rejected(self, client):
        # FastAPI int parameter rejects decimal strings
        assert client.get("/work?intensity=3.5").status_code == 422

    def test_error_body_is_structured(self, client):
        r = client.get("/work?intensity=0")
        body = r.json()
        assert "detail" in body
        assert isinstance(body["detail"], list)
        assert len(body["detail"]) > 0


# ---------------------------------------------------------------------------
# /metrics — Prometheus endpoint format
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_status_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_content_type_is_prometheus_text(self, client):
        ct = client.get("/metrics").headers.get("content-type", "")
        assert "text/plain" in ct
        assert "version=0.0.4" in ct

    def test_contains_all_app_metric_names(self, client):
        client.get("/work?intensity=2")  # ensure counters exist
        text = client.get("/metrics").text
        assert "demo_app_requests_total" in text
        assert "demo_app_request_duration_seconds" in text
        assert "demo_app_requests_in_progress" in text
        assert "demo_app_last_work_intensity" in text


# ---------------------------------------------------------------------------
# Metric correctness — counter increments
# ---------------------------------------------------------------------------

class TestMetricCorrectness:
    def test_work_request_increments_counter(self, client):
        before = _work_count(client)
        client.get("/work?intensity=1")
        after = _work_count(client)
        assert after == before + 1

    def test_multiple_work_requests_increment_counter(self, client):
        before = _work_count(client)
        for _ in range(3):
            client.get("/work?intensity=2")
        after = _work_count(client)
        assert after == before + 3

    def test_latency_histogram_sum_increases_after_work(self, client):
        def latency_sum():
            r = client.get("/metrics")
            m = re.search(
                r'demo_app_request_duration_seconds_sum\{route="/work"\}\s+([\d.e+]+)',
                r.text,
            )
            return float(m.group(1)) if m else 0.0

        before = latency_sum()
        client.get("/work?intensity=3")
        after = latency_sum()
        assert after > before

    def test_last_work_intensity_gauge_updated(self, client):
        client.get("/work?intensity=4")
        r = client.get("/metrics")
        m = re.search(r"demo_app_last_work_intensity\s+([\d.]+)", r.text)
        assert m is not None
        assert float(m.group(1)) == 4.0

    def test_health_does_not_increment_work_counter(self, client):
        before = _work_count(client)
        for _ in range(10):
            client.get("/health")
        after = _work_count(client)
        assert after == before, (
            f"/health incremented work counter: before={before}, after={after}"
        )
