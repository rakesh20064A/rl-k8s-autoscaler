"""demo_ui/tests/test_live_execution_adapter_v2.py

V2 Phase 5E: `RealExperimentControllerV2` has no Streamlit dependency
(same as `live_execution_adapter.py` itself), so these tests run under
the scientific .venv directly, not just the UI venv.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time

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


class TestReattach:
    """V2 Phase 9F: the actual root cause of the recurring "LIVE run
    status: UNKNOWN, controls locked forever" bug -- a `RealExperimentControllerV2`
    created fresh in a NEW Streamlit session (e.g. after a UI restart) had
    no way to resume showing a run it did not itself launch. `attach()` is
    the fix: re-hydrate a fresh instance from the persisted
    `experiment_store` record alone, no `_proc` required."""

    def test_attach_hydrates_run_id_status_path_log_path_and_marks_running(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")
        run_id = experiment_store.start_run("ppo_v2", "demo_cpu_short")
        status_path = tmp_path / "status.json"
        log_path = tmp_path / "log.log"
        status_path.write_text("{}")
        log_path.write_text("hello\n")
        experiment_store.update_run(
            run_id, pid=12345, pgid=12345, status_path=str(status_path), log_path=str(log_path),
        )
        rec = experiment_store.get_run(run_id)

        ctrl = RealExperimentControllerV2()  # a FRESH instance, as after a Streamlit restart
        assert ctrl.run_id is None
        assert ctrl.is_running is False

        ctrl.attach(rec)

        assert ctrl.run_id == run_id
        assert ctrl.is_running is True
        assert ctrl._status_path == status_path
        assert ctrl._log_path == log_path
        assert ctrl.recent_log_lines() == ["hello"]

    def test_stop_signals_the_stored_pgid_even_without_a_proc_handle(self, tmp_path, monkeypatch):
        """STOP must work for a re-attached run -- one this particular
        `RealExperimentControllerV2` instance never itself launched, so it
        has no `subprocess.Popen` handle for it. The pid/pgid to signal
        come entirely from the persisted record."""
        monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")
        proc = subprocess.Popen(["sleep", "100"], start_new_session=True)
        try:
            run_id = experiment_store.start_run("ppo_v2", "demo_cpu_short")
            status_path = tmp_path / "status.json"
            status_path.write_text("{}")  # never updated -- stop() must fall back to pid-death detection
            experiment_store.update_run(
                run_id, pid=proc.pid, pgid=proc.pid, status_path=str(status_path),
            )
            rec = experiment_store.get_run(run_id)

            ctrl = RealExperimentControllerV2()
            ctrl.attach(rec)
            assert ctrl._proc is None  # never launched by this instance

            # `ctrl` has no `_proc` handle (a re-attached run, by design),
            # so it never reaps the child itself -- but THIS TEST's own
            # process is still its real OS parent, so without something
            # reaping it, it stays a zombie whose pid keeps answering
            # `kill(pid, 0)` even after it has actually been terminated
            # (unlike a true orphaned re-attach in production, where the
            # original parent is gone and the OS/init reaps it
            # immediately). A background thread blocked in `proc.wait()`
            # stands in for that reaper so `experiment_store._pid_alive()`
            # can see the real outcome promptly, exactly as it would for a
            # genuinely orphaned process.
            reaper = threading.Thread(target=proc.wait, daemon=True)
            reaper.start()

            ctrl.stop()

            reaper.join(timeout=5.0)
            assert not reaper.is_alive()
            assert proc.poll() is not None
            assert ctrl.is_running is False
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def test_stop_with_own_proc_handle_returns_promptly_not_after_the_full_timeout(self, tmp_path, monkeypatch):
        """Regression: when THIS instance holds the real `Popen` handle
        (the common case -- the same session stopping a run it started
        itself), the wait loop must detect termination via `.poll()`
        (which also reaps the child), never via
        `experiment_store._pid_alive()` alone -- a terminated-but-not-yet-
        reaped child is a zombie whose pid still answers `kill(pid, 0)`,
        which used to make every STOP spin for the full
        `_TERMINATE_WAIT_SECONDS` (30s) even though the process had
        already exited instantly on SIGTERM."""
        monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")
        proc = subprocess.Popen(["sleep", "100"], start_new_session=True)
        try:
            run_id = experiment_store.start_run("ppo_v2", "demo_cpu_short")
            status_path = tmp_path / "status.json"
            status_path.write_text("{}")
            experiment_store.update_run(
                run_id, pid=proc.pid, pgid=proc.pid, status_path=str(status_path),
            )
            ctrl = RealExperimentControllerV2()
            ctrl.run_id = run_id
            ctrl._proc = proc  # THIS instance launched it -- has the real handle

            started = time.monotonic()
            ctrl.stop()
            elapsed = time.monotonic() - started

            assert elapsed < 5.0  # must not have spun the full 30s wait loop
            assert proc.poll() is not None
            assert ctrl.is_running is False
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
