"""live_execution_adapter.py -- REAL experiment execution for the demo UI.

This module is the one place in demo_ui allowed to launch a real
subprocess. `start()` launches `scripts/run_demo_episode_v2.py` -- a
small, isolated, non-scientific adapter -- in the project's scientific
`.venv` (NOT the demo UI's own `requirements-ui.txt` venv, which lacks
stable_baselines3/kubernetes client libs). The subprocess is started in
its own process group (`start_new_session=True`) so STOP can terminate it
and everything it spawned (the traffic generator child process) in one
`killpg` call, without touching any other process on the machine.

RE-ATTACH: a `RealExperimentControllerV2` instance lives in
`st.session_state`, which is per-browser-session and is wiped by a
Streamlit restart. A fresh instance (no `_proc`, no `run_id`) used to show
"UNKNOWN" status with no STOP button for a run it did not itself launch,
even though `experiment_store` (on disk) knew the run was active -- the
actual root cause of the recurring "LIVE run status: UNKNOWN, controls
locked forever" bug. `attach()` re-hydrates an instance from a persisted
`experiment_store` record (run_id/status_path/log_path) so the Experiment
page can call it unconditionally on every render, whether this session
started the run or not. `stop()` likewise never assumes `self._proc` is
set -- it resolves the pid/pgid to signal from the STORE record first,
falling back to `self._proc` only if the store has nothing (e.g. a run
started and immediately stopped within the same call before `start()`
persisted it, which cannot actually happen given `start()`'s ordering, but
costs nothing to guard)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from demo_ui import experiment_store

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
RUNNER_SCRIPT_V2 = PROJECT_ROOT / "scripts" / "run_demo_episode_v2.py"

# Deliberately NOT the system temp dir (`/tmp` on macOS): that is wiped on
# every host restart, which used to orphan a run's status/log files while
# `experiment_store` still had it recorded as `"running"` -- locking the
# Experiment page's controls forever with no process left to ever finish
# or be stopped. This directory persists across restarts like the rest of
# `persistent-data/`.
RUN_FILES_DIR = PROJECT_ROOT / "persistent-data" / "demo_ui"

# Spec: "SIGTERM to the run's process group; wait up to 30s for the
# terminal status; then SIGKILL and let reconciliation clean up."
_TERMINATE_WAIT_SECONDS = 30.0


@dataclass
class RealExperimentControllerV2:
    """Launches `scripts/run_demo_episode_v2.py` (an `--arm`/`--profile`
    contract) and marks every run it starts with `generation="v2"` in
    `experiment_store`. This class does NOT duplicate
    `run_demo_episode_v2.py`'s own training/evaluation
    protection check -- that check runs INSIDE the subprocess (so it
    refuses even if launched outside this UI) and its refusal surfaces
    here via the normal `status()` polling path (`status: "refused"`)."""

    is_running: bool = field(default=False, init=False)
    run_id: Optional[str] = field(default=None, init=False)
    _proc: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    _status_path: Optional[Path] = field(default=None, init=False, repr=False)
    _log_path: Optional[Path] = field(default=None, init=False, repr=False)

    def start(self, arm: str, profile_name: str, model_path: Optional[Path] = None, epsilon: float = 0.05) -> None:
        if self.is_running and self._proc is not None and self._proc.poll() is None:
            return  # already running -- Streamlit rerun/double-click guard

        if not VENV_PYTHON.exists():
            raise RuntimeError(f"project venv python not found at {VENV_PYTHON}; run scripts/bootstrap.sh first")

        RUN_FILES_DIR.mkdir(parents=True, exist_ok=True)
        status_fd, status_name = tempfile.mkstemp(prefix="du_demo_v2_status_", suffix=".json", dir=str(RUN_FILES_DIR))
        os.close(status_fd)
        log_fd, log_name = tempfile.mkstemp(prefix="du_demo_v2_log_", suffix=".log", dir=str(RUN_FILES_DIR))
        os.close(log_fd)
        self._status_path = Path(status_name)
        self._log_path = Path(log_name)

        self.run_id = experiment_store.start_run(arm, profile_name)
        experiment_store.update_run(self.run_id, generation="v2")

        cmd = [
            str(VENV_PYTHON), "-u", str(RUNNER_SCRIPT_V2),
            "--arm", arm, "--profile", profile_name, "--status-file", str(self._status_path),
            "--epsilon", str(epsilon),
        ]
        if model_path is not None:
            cmd += ["--model", str(model_path)]
        child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        log_handle = open(self._log_path, "w")
        self._proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT), stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True, env=child_env,
        )
        self.is_running = True
        # `start_new_session=True` makes the child call `setsid()` as part
        # of its own post-fork setup, which DETERMINISTICALLY makes it its
        # own session AND process group leader (pgid == pid) -- but that
        # call happens in the child, independently of this parent code, so
        # querying `os.getpgid(self._proc.pid)` here immediately after
        # `Popen()` returns would race it: if asked before the child's own
        # `setsid()` has run, it would (briefly, incorrectly) report the
        # PARENT's (this process's) pgid instead -- signaling that via
        # `killpg` would hit this Streamlit process's whole group, not the
        # child. `self._proc.pid` is used directly instead: given
        # `start_new_session=True`, it IS the eventual pgid by
        # construction, with no query/race involved.
        pgid = self._proc.pid
        expected_duration_s = self._resolve_expected_duration_s(profile_name)
        # Persisted (not just held on this in-memory instance) so
        # `experiment_store.active_run()`/`attach()` can tell a genuinely
        # running process apart from a stale record across a Streamlit
        # restart -- a fresh `RealExperimentControllerV2` has no
        # `_proc`/`_status_path` of its own to check against.
        experiment_store.update_run(
            self.run_id, pid=self._proc.pid, pgid=pgid, status_path=str(self._status_path),
            log_path=str(self._log_path), expected_duration_s=expected_duration_s,
        )

    @staticmethod
    def _resolve_expected_duration_s(profile_name: str) -> Optional[float]:
        """Best-effort: the demo profile's own declared duration, read the
        SAME way the Traffic Preview chart does, so `is_run_alive()`'s
        "elapsed <= expected duration + 5 min" liveness check has a real
        number to compare against. Returns None (liveness simply skips
        that check) rather than ever blocking a run from starting over a
        profile file it cannot resolve -- that firewall already lives in
        `run_demo_episode_v2.py` itself."""
        try:
            from demo_ui.profile_preview import resolve_demo_profile_path
            from loadgen.traffic_generator import parse_traffic_profile

            profile = parse_traffic_profile(resolve_demo_profile_path(profile_name))
            return float(profile.total_duration)
        except Exception:  # noqa: BLE001 -- never blocks start() over a preview-only lookup
            return None

    def attach(self, rec: Dict[str, Any]) -> None:
        """Re-hydrates this instance from a persisted `experiment_store`
        record so a controller created fresh in THIS Streamlit session
        (e.g. right after a UI restart) can resume showing the real
        stage/progress/log output for a run it did not itself launch, and
        accept a STOP click for it. Never recreates `_proc`: there is no
        real `subprocess.Popen` handle for a process this instance did not
        spawn -- `stop()` falls back to the record's own pid/pgid, and
        `status()` reads the status file directly, so neither needs one."""
        self.run_id = rec["run_id"]
        self.is_running = True
        if rec.get("status_path"):
            self._status_path = Path(rec["status_path"])
        if rec.get("log_path"):
            self._log_path = Path(rec["log_path"])

    def stop(self) -> None:
        """SIGTERM to the run's whole process group (so the traffic
        generator child dies too), wait up to `_TERMINATE_WAIT_SECONDS`
        for the runner's OWN terminal status write (reflecting its
        unconditional cleanup -- see `run_demo_episode_v2.py`), then
        SIGKILL as a last resort and let reconciliation (`active_run()`)
        finish the job. Resolves the pid/pgid to signal from the
        PERSISTED record first (works for a re-attached run this instance
        never itself launched), falling back to `self._proc` only if the
        store has nothing recorded."""
        rec = experiment_store.get_run(self.run_id) if self.run_id else None
        pid = (rec or {}).get("pid")
        pgid = (rec or {}).get("pgid")
        status_path = (rec or {}).get("status_path") or (str(self._status_path) if self._status_path else None)
        if pid is None and self._proc is not None:
            pid = self._proc.pid
        if pid is None:
            self.is_running = False
            return
        if pgid is None:
            try:
                pgid = os.getpgid(pid)
            except (ProcessLookupError, OSError):
                pgid = pid

        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        else:
            deadline = time.monotonic() + _TERMINATE_WAIT_SECONDS
            terminated = False
            while time.monotonic() < deadline:
                # When this instance holds the real `Popen` handle (the
                # common case: the same session that started the run is
                # the one stopping it), `.poll()` is the ONLY correct
                # liveness check -- it also reaps the child the moment it
                # exits. `experiment_store._pid_alive()` alone would see a
                # TERMINATED-BUT-NOT-YET-REAPED child as still "alive"
                # (a zombie's pid remains valid until its parent reaps
                # it), which used to make every STOP spin for the full
                # `_TERMINATE_WAIT_SECONDS` even though the process had
                # already exited instantly. For a re-attached run with no
                # `_proc` (this instance is not the process's parent), the
                # OS reaps it elsewhere, so the pid check alone is correct.
                if self._proc is not None:
                    if self._proc.poll() is not None:
                        terminated = True
                        break
                elif not experiment_store._pid_alive(pid):
                    terminated = True
                    break
                raw = experiment_store._read_status_file(status_path) if status_path else None
                if raw is not None and raw.get("status") in experiment_store._STATUS_FILE_TERMINAL:
                    terminated = True
                    break
                time.sleep(0.5)
            if not terminated:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass

        if self._proc is not None:
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
        self.is_running = False
        if self.run_id is not None:
            experiment_store.active_run()  # reconciliation: picks up the runner's own terminal status if it managed to write one before SIGKILL
            rec = experiment_store.get_run(self.run_id)
            if rec is not None and rec.get("status") == "running":
                # the runner never got to write its own terminal status
                # (e.g. it had to be SIGKILLed) -- record the stop
                # explicitly rather than leaving the UI to guess.
                experiment_store.update_run(self.run_id, status="exited", stage="STOPPED_BY_USER")

    def reset(self) -> None:
        self.stop()
        self._proc = None
        self._status_path = None
        self._log_path = None
        self.run_id = None

    def status(self) -> dict:
        """Mirrors `RealExperimentController.status()` exactly, with one
        addition: a `"refused"` status (the training/evaluation-protection
        check inside `run_demo_episode_v2.py` failing closed) is treated
        the same way `"error"` is -- surfaced, never silently dropped."""
        if self._status_path is None or not self._status_path.exists():
            if self._proc is not None and self._proc.poll() is not None:
                raw = {"status": "exited", "returncode": self._proc.returncode}
            else:
                raw = {"status": "unknown"}
        else:
            try:
                raw = json.loads(self._status_path.read_text())
            except (json.JSONDecodeError, OSError):
                raw = {"status": "unknown"}

        if self.run_id is not None and raw.get("status") != "unknown":
            # Shared with `experiment_store`'s own reconciliation path
            # (which reads this exact file when no UI was around to poll
            # it) so the two can never map status-file fields differently.
            experiment_store.update_run(self.run_id, **experiment_store.status_file_fields_for_store(raw))

        return raw

    def recent_log_lines(self, n: int = 20) -> list:
        if self._log_path is None or not self._log_path.exists():
            return []
        try:
            lines = self._log_path.read_text(errors="replace").splitlines()
        except OSError:
            return []
        return lines[-n:]
