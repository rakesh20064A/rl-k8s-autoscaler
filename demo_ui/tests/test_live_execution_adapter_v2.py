"""demo_ui/tests/test_live_execution_adapter_v2.py

V2 Phase 5E: `RealExperimentControllerV2` has no Streamlit dependency
(same as `live_execution_adapter.py` itself), so these tests run under
the scientific .venv directly, not just the UI venv.
"""

from __future__ import annotations

import json

from demo_ui import experiment_store
from demo_ui.live_execution_adapter import RealExperimentControllerV2


class TestRealExperimentControllerV2Status:
    def test_status_is_unknown_before_start(self):
        ctrl = RealExperimentControllerV2()
        assert ctrl.status() == {"status": "unknown"}

    def test_status_syncs_running_stage_into_experiment_store(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")
        ctrl = RealExperimentControllerV2()
        ctrl.run_id = experiment_store.start_run("ppo_v2", "demo_cpu_short")
        experiment_store.update_run(ctrl.run_id, generation="v2")

        status_path = tmp_path / "status.json"
        status_path.write_text(json.dumps({"status": "running", "stage": "EPISODE_RUNNING", "samples": [{"time_s": 0.0, "p95_latency_ms": 100.0}]}))
        ctrl._status_path = status_path

        raw = ctrl.status()
        assert raw["status"] == "running"

        rec = experiment_store.get_run(ctrl.run_id)
        assert rec["stage"] == "EPISODE_RUNNING"
        assert rec["generation"] == "v2"
        assert rec["summary"]["samples"][0]["time_s"] == 0.0

    def test_refused_status_surfaces_the_error_message(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")
        ctrl = RealExperimentControllerV2()
        ctrl.run_id = experiment_store.start_run("ppo_v2", "demo_cpu_short")

        status_path = tmp_path / "status.json"
        status_path.write_text(json.dumps({"status": "refused", "stage": "REFUSED", "error": "refusing to start: controller lock is 'ppo', not 'none'"}))
        ctrl._status_path = status_path

        raw = ctrl.status()
        assert raw["status"] == "refused"
        rec = experiment_store.get_run(ctrl.run_id)
        assert "controller lock" in rec["error"]

    def test_completed_status_carries_full_summary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")
        ctrl = RealExperimentControllerV2()
        ctrl.run_id = experiment_store.start_run("hpa_v2_cpu_memory", "demo_mixed_short")

        status_path = tmp_path / "status.json"
        summary = {"episode_id": "e1", "arm": "hpa_v2_cpu_memory", "profile": "demo_mixed_short", "n_samples": 2, "samples": [{"time_s": 0.0}, {"time_s": 30.0}]}
        status_path.write_text(json.dumps({"status": "completed", "stage": "COMPLETED", "summary": summary}))
        ctrl._status_path = status_path

        ctrl.status()
        rec = experiment_store.get_run(ctrl.run_id)
        assert rec["status"] == "completed"
        assert rec["summary"]["n_samples"] == 2


class TestRealExperimentControllerV2StartGuard:
    def test_start_is_a_noop_while_already_running(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")

        class _FakeProc:
            def poll(self):
                return None

        ctrl = RealExperimentControllerV2()
        ctrl.is_running = True
        ctrl._proc = _FakeProc()
        ctrl.run_id = "sentinel"
        ctrl.start("ppo_v2", "demo_cpu_short")
        assert ctrl.run_id == "sentinel"  # start() never ran again
