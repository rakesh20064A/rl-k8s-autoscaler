"""demo_ui/tests/test_comparison_live_pairing.py

Regression coverage for `demo_ui/views/comparison.py`'s live (store-backed)
HPA/PPO pairing section -- found broken two ways while verifying the demo
pipeline fixes end-to-end against a real cluster (task sec.6/sec.7):

1. `_latest_current_system_run()` must never choose an `invalid_no_traffic`
   run as "the workload" this section leads with (task sec.3).
2. The Replicas/P95 overlay charts read `experiment_store.extract_series()`
   with field names that must match `EpisodeSample`/
   `scripts/run_demo_episode_v2.py`'s real per-decision sample schema
   (`desired_replicas`/`p95_latency_ms`) -- a PRE-EXISTING, unrelated bug
   (`"replicas_desired"`/`"latency_p95_ms"`, neither a real key) silently
   made those charts always empty even for a correctly-paired run, fixed
   in passing while verifying comparison pairing works end-to-end on the
   two real smoke runs sec.7 produced.
"""

from __future__ import annotations

import inspect

import pytest

from demo_ui import experiment_store as es
from demo_ui.views import comparison


@pytest.fixture()
def store_path(tmp_path):
    return tmp_path / "runs.json"


def _completed_with_samples(store_path, controller, profile, h="h1", seed=42, dur=120.0):
    """One completed run with the REAL EpisodeSample-shaped sample schema
    (`desired_replicas`/`p95_latency_ms`/`time_s`), the same shape
    `scripts/run_demo_episode_v2.py`'s own `_to_status_sample()` produces."""
    rid = es.start_run(controller, profile, path=store_path)
    samples = [
        {"time_s": float(i) * 30.0, "desired_replicas": 4 + i, "p95_latency_ms": 100.0 + i * 10,
         "request_rate_rps": 5.0 + i}
        for i in range(3)
    ]
    es.update_run(
        rid, status="completed", workload_hash=h, workload_seed=seed, workload_duration_s=dur,
        summary={"samples": samples}, path=store_path,
    )
    return es.get_run(rid, store_path)


class TestOverlayChartFieldNamesMatchTheRealSampleSchema:
    """Source-level guard: `_render_live_comparison()` must read
    `extract_series()` with the field names `EpisodeSample`'s own schema
    actually uses -- never the old, silently-always-empty names."""

    def test_uses_desired_replicas_never_the_old_wrong_key(self):
        src = inspect.getsource(comparison._render_live_comparison)
        assert '"desired_replicas"' in src
        assert '"replicas_desired"' not in src

    def test_uses_p95_latency_ms_never_the_old_wrong_key(self):
        src = inspect.getsource(comparison._render_live_comparison)
        assert '"p95_latency_ms"' in src
        assert '"latency_p95_ms"' not in src

    def test_extract_series_returns_real_data_for_a_completed_run(self, store_path):
        hpa = _completed_with_samples(store_path, "hpa_v2_cpu_memory", "demo_smoke_bursts_cpu")
        ts, vals = es.extract_series(hpa, "desired_replicas")
        assert ts == [0.0, 30.0, 60.0]
        assert vals == [4, 5, 6]
        ts_p95, p95 = es.extract_series(hpa, "p95_latency_ms")
        assert p95 == [100.0, 110.0, 120.0]


class TestLatestCurrentSystemRunExcludesInvalidNoTraffic:
    def test_skips_an_invalid_no_traffic_run_in_favor_of_the_next_valid_one(self, monkeypatch, store_path):
        monkeypatch.setattr(es, "STORE_PATH", store_path)
        valid = _completed_with_samples(store_path, "ppo_v2", "demo_memory_short")
        invalid = _completed_with_samples(store_path, "hpa_v2_cpu_memory", "demo_cpu_short")
        es.update_run(valid["run_id"], generation="v2", path=store_path)
        es.update_run(invalid["run_id"], generation="v2", status=es.INVALID_NO_TRAFFIC, path=store_path)

        latest = comparison._latest_current_system_run()
        assert latest is not None
        assert latest["run_id"] == valid["run_id"]

    def test_returns_none_when_every_run_is_invalid_no_traffic(self, monkeypatch, store_path):
        monkeypatch.setattr(es, "STORE_PATH", store_path)
        rec = _completed_with_samples(store_path, "ppo_v2", "demo_cpu_short")
        es.update_run(rec["run_id"], status=es.INVALID_NO_TRAFFIC, generation="v2", path=store_path)

        assert comparison._latest_current_system_run() is None
