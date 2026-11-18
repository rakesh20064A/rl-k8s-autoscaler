"""demo_ui/tests/test_ui_reattach.py

V2 Phase 9F: end-to-end (Streamlit `AppTest`) coverage for the actual
production incident this phase fixes -- run `4d4edc8274e6` (PID 20102,
profile `demo_constant_cpu_normal`) left the Experiment page showing
"LIVE run status: UNKNOWN" with every control (Arm/Pattern/Resource/
Load/START) disabled and no STOP button, after a Streamlit restart
orphaned the `RealExperimentControllerV2` instance that had been tracking
it.

Each `AppTest.from_file(...)` call below is a FRESH Streamlit session --
no `st.session_state` carries over between them -- which is exactly what
a UI restart (or simply a different browser tab) looks like. These tests
seed `experiment_store` directly (never via a real `.start()` click) to
simulate a run that some OTHER session already launched, then load the
Experiment page fresh and check it re-attaches correctly.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from demo_ui import experiment_store  # noqa: E402
from demo_ui.views import experiment as experiment_view  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")
    # These tests exercise reconciliation paths that, in the real app,
    # trigger a live-cluster cleanup check (`ensure_canonical_idle_state`,
    # which shells out to `kubectl`) whenever a run is freshly found
    # stale. Faked here so this ENTIRE test file never touches a real
    # cluster regardless of which scenario a given test seeds -- these
    # are UI re-attach tests, not cluster-cleanup tests (those live in
    # `tests/unit/test_v2_live_cluster_state.py`).
    monkeypatch.setattr(experiment_view, "ensure_canonical_idle_state", lambda: {"ran": False, "reason": "faked for this test"})
    monkeypatch.setattr(experiment_view, "no_runner_process_alive", lambda: True)
    return tmp_path


def _all_markdown_html(at: AppTest) -> str:
    return "".join(m.value for m in at.markdown)


def _run_experiment_page() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value("Experiment").run()
    return at


class TestReattachToAGenuinelyActiveRun:
    """A run some OTHER session started is still genuinely alive (real
    pid + fresh heartbeat) when this fresh session loads the page."""

    def _seed_active_run(self, tmp_path):
        proc = subprocess.Popen(["sleep", "100"], start_new_session=True)
        status_path = tmp_path / "status.json"
        status_path.write_text(json.dumps({
            "status": "running", "stage": "EPISODE_RUNNING", "arm": "ppo_v2",
            "profile": "demo_constant_cpu_normal", "decision_index": 3,
            "samples": [{"time_s": 0.0, "desired_replicas": 2}],
        }))
        run_id = experiment_store.start_run("ppo_v2", "demo_constant_cpu_normal")
        experiment_store.update_run(
            run_id, generation="v2", pid=proc.pid, pgid=proc.pid, status_path=str(status_path),
        )
        return proc, run_id

    def test_fresh_session_shows_real_stage_not_unknown(self, tmp_path):
        proc, run_id = self._seed_active_run(tmp_path)
        try:
            at = _run_experiment_page()
            assert not at.exception

            html = _all_markdown_html(at)
            assert "UNKNOWN" not in html
            assert "EPISODE_RUNNING" in html
            assert run_id in html
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def test_fresh_session_shows_stop_button_not_start_and_controls_locked(self, tmp_path):
        proc, _run_id = self._seed_active_run(tmp_path)
        try:
            at = _run_experiment_page()
            assert not at.exception

            button_keys = {b.key for b in at.button}
            assert "du_v2_stop_button" in button_keys
            assert "du_v2_start_button" not in button_keys

            arm_select = at.selectbox(key="du_v2_arm_widget")
            assert arm_select.disabled is True
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def test_stop_click_on_a_reattached_run_actually_stops_it(self, tmp_path):
        proc, run_id = self._seed_active_run(tmp_path)
        reaper = threading.Thread(target=proc.wait, daemon=True)
        reaper.start()
        try:
            at = _run_experiment_page()
            at.button(key="du_v2_stop_button").click().run()
            assert not at.exception

            reaper.join(timeout=10.0)
            assert not reaper.is_alive()  # the real subprocess was actually signaled and died

            rec = experiment_store.get_run(run_id)
            assert rec["status"] != "running"
        finally:
            if proc.poll() is None:
                proc.kill()
            if reaper.is_alive():
                reaper.join(timeout=5.0)


class TestReattachAfterRunFinishedWhileUiWasClosed:
    """The EXACT `4d4edc8274e6` regression: the run's process (PID 20102)
    is dead by the time anyone looks again, AND the store record's
    status-file mtime reads stale -- but the status file's OWN content
    says `"completed"` with a full summary. The fresh session must show
    COMPLETED and unlock the controls, never a fabricated "interrupted"
    verdict and never "UNKNOWN"."""

    def _seed_finished_while_closed(self, tmp_path):
        status_path = tmp_path / "status.json"
        summary = {
            "episode_id": "demo_v2_ppo_v2_demo_constant_cpu_normal_1", "arm": "ppo_v2",
            "profile": "demo_constant_cpu_normal", "n_samples": 16, "demo_only": True, "partial": False,
            "samples": [{"time_s": float(i) * 30.0, "desired_replicas": 2} for i in range(16)],
        }
        status_path.write_text(json.dumps({
            "status": "completed", "stage": "COMPLETED", "arm": "ppo_v2",
            "profile": "demo_constant_cpu_normal", "summary": summary,
        }))
        old = time.time() - (experiment_store.STALE_STATUS_FILE_AGE_S + 600)
        os.utime(status_path, (old, old))
        dead_pid = 2**31 - 1
        run_id = experiment_store.start_run("ppo_v2", "demo_constant_cpu_normal")
        experiment_store.update_run(
            run_id, generation="v2", pid=dead_pid, status_path=str(status_path),
        )
        return run_id

    def test_fresh_session_shows_completed_not_interrupted_or_unknown(self, tmp_path):
        run_id = self._seed_finished_while_closed(tmp_path)

        at = _run_experiment_page()

        assert not at.exception
        html = _all_markdown_html(at)
        assert "UNKNOWN" not in html
        assert "INTERRUPTED" not in html
        assert "COMPLETED" in html

        rec = experiment_store.get_run(run_id)
        assert rec["status"] == "completed"
        assert rec["summary"]["n_samples"] == 16

    def test_fresh_session_unlocks_controls(self, tmp_path):
        self._seed_finished_while_closed(tmp_path)

        at = _run_experiment_page()

        arm_select = at.selectbox(key="du_v2_arm_widget")
        assert arm_select.disabled is False
        button_keys = {b.key for b in at.button}
        assert "du_v2_start_button" in button_keys


class TestLiveLogTailingDuringAnActiveRun:
    """Task sec.2/7: the "Recent log output" panel must show log content
    WHILE a run is still active -- read incrementally from the log file,
    never waiting for the run to finish. Reproduces this by seeding a
    genuinely active run whose log file already has several lines (as
    `scripts/run_demo_episode_v2.py` now writes incrementally, task
    sec.1) and checking those lines are visible on a single fresh page
    load, with the status still non-terminal."""

    def test_in_progress_log_lines_are_shown_while_run_is_still_running(self, tmp_path):
        proc = subprocess.Popen(["sleep", "100"], start_new_session=True)
        status_path = tmp_path / "status.json"
        log_path = tmp_path / "run.log"
        status_path.write_text(json.dumps({
            "status": "running", "stage": "EPISODE_RUNNING", "arm": "ppo_v2",
            "profile": "demo_constant_cpu_normal", "decision_index": 2,
            "samples": [{"time_s": 0.0, "desired_replicas": 2}],
        }))
        log_path.write_text(
            "[12:40:05] Preflight OK: lock none, no conflicting process or unexpected HPA detected\n"
            "[12:40:06] Removed idle HPA ['demo-app-hpa-v2']; controller lock → ppo\n"
            "[12:40:08] Traffic started: Bursts / CPU / High (480 s)\n"
            "[12:40:38] Decision 1/16: HOLD | replicas 2→2 | 41 req/s | p95 612 ms | CPU/replica 2.3 | mem/replica 0.9 | tie rule: none\n"
        )
        run_id = experiment_store.start_run("ppo_v2", "demo_bursts_cpu_high")
        experiment_store.update_run(
            run_id, generation="v2", pid=proc.pid, pgid=proc.pid,
            status_path=str(status_path), log_path=str(log_path),
        )
        try:
            at = _run_experiment_page()
            assert not at.exception

            html = _all_markdown_html(at)
            assert "Traffic started: Bursts / CPU / High" in html
            assert "Decision 1/16: HOLD" in html
            # The run is still genuinely in progress -- this is NOT a
            # post-completion log dump; the status shown is still the
            # running stage, never a terminal one.
            assert "EPISODE_RUNNING" in html
            assert "COMPLETED" not in html
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


class TestButtonFlipsToStartAfterCompletionWithoutManualRerun:
    """Task sec.3/7's key regression test: the ROOT CAUSE was
    `render_controls_v2()`'s `already_running = controls_disabled or
    st.session_state["du_v2_run_in_progress"]` -- that session_state flag
    was set once, at START-click time, and only ever reset by an explicit
    STOP click, so it stayed `True` forever once a LIVE run finished on
    its own. `controls_disabled` (derived from the store's `active_run()`)
    correctly flipped back to `False`, but the `or` kept `already_running`
    `True` regardless, so STOP (never START) kept showing even after a
    full-page rerun that had already re-synced everything else. This test
    reproduces that exact sequence WITHOUT ever starting a fresh `AppTest`
    session (a manual browser refresh) -- the SAME session must flip on
    its own the next time it reruns."""

    def _seed_active_run(self, tmp_path):
        proc = subprocess.Popen(["sleep", "100"], start_new_session=True)
        status_path = tmp_path / "status.json"
        status_path.write_text(json.dumps({
            "status": "running", "stage": "EPISODE_RUNNING", "arm": "ppo_v2",
            "profile": "demo_constant_cpu_normal", "decision_index": 1,
            "samples": [{"time_s": 0.0, "desired_replicas": 2}],
        }))
        run_id = experiment_store.start_run("ppo_v2", "demo_constant_cpu_normal")
        experiment_store.update_run(
            run_id, generation="v2", pid=proc.pid, pgid=proc.pid, status_path=str(status_path),
        )
        return proc, run_id, status_path

    def test_same_session_shows_start_after_run_completes_no_manual_refresh(self, tmp_path):
        proc, run_id, status_path = self._seed_active_run(tmp_path)
        try:
            # 1) Fresh load of a session that itself clicked START earlier
            #    (never via a real click here -- that would launch a real
            #    subprocess against the live cluster; `du_v2_run_in_progress`
            #    is set directly, exactly as `views/experiment.py`'s own
            #    START-click handler sets it). The run is genuinely active:
            #    STOP is shown either way, so this alone cannot distinguish
            #    the fix from the bug -- step 3 below is what does.
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.session_state["du_v2_run_in_progress"] = True
            at.run()
            at.sidebar.radio[0].set_value("Experiment").run()
            assert not at.exception
            button_keys = {b.key for b in at.button}
            assert "du_v2_stop_button" in button_keys
            assert "du_v2_start_button" not in button_keys

            # 2) The run finishes on its own (never via a STOP click) --
            #    its OWN status file now records a real terminal outcome,
            #    and the backing process is gone, exactly as
            #    `scripts/run_demo_episode_v2.py`'s own unconditional
            #    cleanup leaves things.
            proc.kill()
            proc.wait()
            summary = {
                "episode_id": "demo_v2_ppo_v2_demo_constant_cpu_normal_1", "arm": "ppo_v2",
                "profile": "demo_constant_cpu_normal", "n_samples": 16, "demo_only": True, "partial": False,
                "samples": [{"time_s": float(i) * 30.0, "desired_replicas": 2} for i in range(16)],
            }
            status_path.write_text(json.dumps({
                "status": "completed", "stage": "COMPLETED", "arm": "ppo_v2",
                "profile": "demo_constant_cpu_normal", "summary": summary,
            }))

            # 3) The SAME `AppTest` session reruns (what the fragment's own
            #    periodic tick -- or any other widget interaction -- would
            #    trigger) -- NOT a new `AppTest.from_file(...)` call, which
            #    would be a manual browser refresh and could trivially
            #    "fix" a stale in-memory flag just by recreating it.
            at.run()
            assert not at.exception

            html = _all_markdown_html(at)
            assert "COMPLETED" in html

            button_keys = {b.key for b in at.button}
            assert "du_v2_start_button" in button_keys, "must flip to START without a manual refresh"
            assert "du_v2_stop_button" not in button_keys, "must not keep showing STOP after completion"
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


class TestHostRestartWipesTmpNoLock:
    """A record survives (it lives under `persistent-data/`, not `/tmp`),
    but its pid is dead AND its status file is simply gone (the OLD
    `/tmp`-based design -- this models what a host restart used to do to
    it). No terminal outcome can be recovered; the record must still end
    up unlocking the UI (`STALE_STATUS`, never stuck `"running"`
    forever) rather than requiring manual intervention."""

    def test_missing_status_file_unlocks_controls_not_stuck_forever(self, tmp_path):
        missing_status_path = tmp_path / "wiped_tmp_status.json"  # never written
        dead_pid = 2**31 - 1
        experiment_store.start_run("ppo_v2", "demo_constant_cpu_normal")
        run_id = experiment_store.latest_run()["run_id"]
        experiment_store.update_run(
            run_id, generation="v2", pid=dead_pid, status_path=str(missing_status_path),
        )

        at = _run_experiment_page()

        assert not at.exception
        arm_select = at.selectbox(key="du_v2_arm_widget")
        assert arm_select.disabled is False
        rec = experiment_store.get_run(run_id)
        assert rec["status"] == experiment_store.STALE_STATUS
