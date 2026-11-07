"""Coverage for incremental (progressive) LIVE telemetry: samples must
reach `experiment_store` as the real backend/evaluation loop produces them,
not only once the episode completes, with no duplication across repeated
polls and no divergence between HPA and PPO.
"""

from __future__ import annotations

import json

import pytest

from demo_ui import experiment_store as es
from demo_ui.live_execution_adapter import RealExperimentController


def _write_status(path, payload):
    path.write_text(json.dumps(payload))


@pytest.fixture()
def ctrl(tmp_path, monkeypatch):
    store_path = tmp_path / "runs.json"
    monkeypatch.setattr(es, "STORE_PATH", store_path)
    monkeypatch.setattr("demo_ui.live_execution_adapter.experiment_store", es)
    c = RealExperimentController()
    c.run_id = es.start_run("HPA", "demo_smoke_test", path=store_path)
    c.is_running = True
    status_path = tmp_path / "status.json"
    c._status_path = status_path
    return c, status_path, store_path


class TestIncrementalSync:
    def test_active_run_with_zero_samples_has_empty_series(self, ctrl):
        _, status_path, store_path = ctrl
        _write_status(status_path, {"status": "running", "stage": "EPISODE_RUNNING"})
        rec = es.active_run(store_path)
        assert es.extract_series(rec, "replicas_desired") == ([], [])

    def test_first_sample_syncs_exactly_one(self, ctrl):
        c, status_path, store_path = ctrl
        sample1 = {"time_s": 0.0, "replicas_desired": 1, "latency_p95_ms": 90.0}
        _write_status(status_path, {
            "status": "running", "stage": "EPISODE_RUNNING", "controller": "HPA", "profile": "demo_smoke_test",
            "workload_hash": "h1", "workload_seed": 42, "workload_duration_seconds": 40.0,
            "samples": [sample1],
        })
        c.status()
        rec = es.get_run(c.run_id, store_path)
        assert rec["summary"]["samples"] == [sample1]
        assert rec["workload_hash"] == "h1"

    def test_repeated_poll_of_unchanged_payload_does_not_duplicate(self, ctrl):
        c, status_path, store_path = ctrl
        sample1 = {"time_s": 0.0, "replicas_desired": 1}
        payload = {"status": "running", "stage": "EPISODE_RUNNING", "samples": [sample1]}
        _write_status(status_path, payload)
        c.status()
        c.status()
        c.status()
        rec = es.get_run(c.run_id, store_path)
        assert rec["summary"]["samples"] == [sample1]

    def test_second_sample_appends_in_order(self, ctrl):
        c, status_path, store_path = ctrl
        s1 = {"time_s": 0.0, "replicas_desired": 1}
        s2 = {"time_s": 30.0, "replicas_desired": 2}
        _write_status(status_path, {"status": "running", "samples": [s1]})
        c.status()
        _write_status(status_path, {"status": "running", "samples": [s1, s2]})
        c.status()
        rec = es.get_run(c.run_id, store_path)
        assert rec["summary"]["samples"] == [s1, s2]
        ts, replicas = es.extract_series(rec, "replicas_desired")
        assert ts == [0.0, 30.0]
        assert replicas == [1, 2]

    def test_navigation_away_and_back_preserves_samples(self, ctrl):
        """Simulates leaving/returning to Live Details: a FRESH read of the
        store (no shared Python object) must still see the samples."""
        c, status_path, store_path = ctrl
        _write_status(status_path, {"status": "running", "samples": [{"time_s": 0.0, "replicas_desired": 1}]})
        c.status()
        reloaded = es.get_run(c.run_id, store_path)  # independent re-read
        assert reloaded["summary"]["samples"] == [{"time_s": 0.0, "replicas_desired": 1}]

    def test_latest_metric_uses_last_real_sample(self, ctrl):
        c, status_path, store_path = ctrl
        samples = [{"time_s": 0.0, "replicas_desired": 1}, {"time_s": 30.0, "replicas_desired": 4}]
        _write_status(status_path, {"status": "running", "samples": samples})
        c.status()
        rec = es.get_run(c.run_id, store_path)
        _, replicas = es.extract_series(rec, "replicas_desired")
        assert replicas[-1] == 4  # latest, not first/aggregated

    def test_completion_does_not_duplicate_incremental_samples(self, ctrl):
        c, status_path, store_path = ctrl
        s1 = {"time_s": 0.0, "replicas_desired": 1}
        s2 = {"time_s": 30.0, "replicas_desired": 2}
        _write_status(status_path, {"status": "running", "samples": [s1]})
        c.status()
        _write_status(status_path, {"status": "running", "samples": [s1, s2]})
        c.status()
        # Final completion write: the SAME cumulative list, wrapped in the
        # full final summary -- must replace, not append to, the partial one.
        final_summary = {
            "workload_hash": "h1", "workload_seed": 42, "workload_duration_seconds": 40.0,
            "samples": [s1, s2], "mean_replicas": 1.5,
        }
        _write_status(status_path, {"status": "completed", "summary": final_summary})
        c.status()
        rec = es.get_run(c.run_id, store_path)
        assert rec["summary"]["samples"] == [s1, s2]
        assert rec["status"] == "completed"

    @pytest.mark.parametrize("controller", ["HPA", "PPO"])
    def test_hpa_and_ppo_use_same_persistence_path(self, tmp_path, monkeypatch, controller):
        store_path = tmp_path / "runs.json"
        monkeypatch.setattr(es, "STORE_PATH", store_path)
        monkeypatch.setattr("demo_ui.live_execution_adapter.experiment_store", es)
        c = RealExperimentController()
        c.run_id = es.start_run(controller, "demo_smoke_test", path=store_path)
        c.is_running = True
        status_path = tmp_path / "status.json"
        c._status_path = status_path
        _write_status(status_path, {"status": "running", "controller": controller, "samples": [{"time_s": 0.0, "replicas_desired": 1}]})
        c.status()
        rec = es.get_run(c.run_id, store_path)
        assert rec["summary"]["samples"] == [{"time_s": 0.0, "replicas_desired": 1}]
        assert rec["controller"] == controller


class TestNoFixtureLeak:
    def test_running_record_never_contains_fixture_markers(self, ctrl):
        c, status_path, store_path = ctrl
        _write_status(status_path, {"status": "running", "samples": [{"time_s": 0.0, "replicas_desired": 1}]})
        c.status()
        rec = es.get_run(c.run_id, store_path)
        assert rec["data_source"] == "LIVE"
        assert "fixture" not in json.dumps(rec).lower()
