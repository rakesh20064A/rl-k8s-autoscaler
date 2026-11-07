"""live_execution_adapter.py -- REAL experiment execution for the demo UI.

Deliberately kept separate from `live_adapter.py` (whose own test suite,
`demo_ui/tests/test_live_adapter.py::test_no_real_mutation_path_exists_in_source`,
asserts that module stays subprocess-free/dry-run-only). This module is the
one place in demo_ui allowed to launch a real subprocess.

`start()` launches `scripts/run_demo_episode.py` -- a small, isolated,
non-scientific adapter that reuses the exact tested live mechanics from
`scripts/evaluate_ppo_vs_hpa.py` (`run_ppo_episode`/`run_hpa_episode`) --
in the project's scientific `.venv` (NOT the demo UI's own
`requirements-ui.txt` venv, which lacks stable_baselines3/kubernetes
client libs). The subprocess is started in its own process group
(`start_new_session=True`) so STOP can terminate it and everything it
spawned (the traffic generator child process) in one `killpg` call,
without touching any other process on the machine.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from demo_ui import experiment_store
from demo_ui.models import Controller, TrafficProfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
RUNNER_SCRIPT = PROJECT_ROOT / "scripts" / "run_demo_episode.py"

_TERMINATE_WAIT_SECONDS = 10.0


@dataclass
class RealExperimentController:
    """Implements the `ExperimentController` Protocol (providers.py) with
    real subprocess execution. One instance per Streamlit session
    (`st.session_state`), so `is_running`/the process handle survive
    reruns -- a second `start()` call while already running is a no-op,
    mirroring `LiveExperimentController`'s own duplicate-launch guard."""

    is_running: bool = field(default=False, init=False)
    profile_override: Optional[str] = field(default=None, init=False, repr=False)
    run_id: Optional[str] = field(default=None, init=False)
    _proc: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    _status_path: Optional[Path] = field(default=None, init=False, repr=False)
    _log_path: Optional[Path] = field(default=None, init=False, repr=False)

    def set_profile_override(self, profile_name: Optional[str]) -> None:
        """Lets the UI request a demo-only profile name (e.g.
        "demo_smoke_test") not present in the `TrafficProfile` enum, without
        changing that enum or the `ExperimentController` Protocol. `None`
        clears the override, falling back to the selected `TrafficProfile`."""
        self.profile_override = profile_name

    def start(self, controller: Controller, traffic_profile: TrafficProfile) -> None:
        if self.is_running and self._proc is not None and self._proc.poll() is None:
            return  # already running -- Streamlit rerun/double-click guard

        if not VENV_PYTHON.exists():
            raise RuntimeError(f"project venv python not found at {VENV_PYTHON}; run scripts/bootstrap.sh first")

        status_fd, status_name = tempfile.mkstemp(prefix="du_demo_status_", suffix=".json")
        os.close(status_fd)
        log_fd, log_name = tempfile.mkstemp(prefix="du_demo_log_", suffix=".log")
        os.close(log_fd)
        self._status_path = Path(status_name)
        self._log_path = Path(log_name)

        profile_name = self.profile_override or traffic_profile.value
        # The ONE place a new run_id is minted (guarded by the is_running
        # check above, same guard that already prevents a duplicate
        # subprocess on a Streamlit rerun) -- this record is what
        # Live Details/History/Comparison read, via experiment_store, to
        # follow this exact run from any page.
        self.run_id = experiment_store.start_run(controller.value, profile_name)
        cmd = [
            str(VENV_PYTHON), "-u", str(RUNNER_SCRIPT),
            "--controller", controller.value.lower(),
            "--profile", profile_name,
            "--status-file", str(self._status_path),
        ]
        # -u (and PYTHONUNBUFFERED, belt-and-suspenders) matter because
        # stdout is redirected to a file below: Python block-buffers
        # non-TTY stdout by default, so without this the log file would
        # stay empty for the whole episode (flushed only at process exit),
        # even though the subprocess is actively running and producing
        # output the whole time.
        child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        log_handle = open(self._log_path, "w")
        self._proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT), stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True, env=child_env,
        )
        self.is_running = True

    def stop(self) -> None:
        if self._proc is None:
            self.is_running = False
            return
        if self._proc.poll() is None:
            try:
                pgid = os.getpgid(self._proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                self._proc.wait(timeout=_TERMINATE_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self._proc.wait(timeout=_TERMINATE_WAIT_SECONDS)
            except ProcessLookupError:
                pass
        self.is_running = False
        if self.run_id is not None:
            rec = experiment_store.get_run(self.run_id)
            if rec is not None and rec.get("status") == "running":
                experiment_store.update_run(self.run_id, status="exited", stage="STOPPED_BY_USER")

    def reset(self) -> None:
        self.stop()
        self._proc = None
        self._status_path = None
        self._log_path = None
        self.run_id = None

    def status(self) -> dict:
        """Best-effort read of the runner's status file. Never raises --
        returns {"status": "unknown"} if nothing has been written yet or
        the file is mid-write. Also syncs whatever it learns into
        `experiment_store` (the cross-page source of truth), since this is
        already the one place polled regularly (by the Experiment page's
        status fragment) while a run is active."""
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
            fields = {"status": raw["status"]}
            if raw.get("stage"):
                fields["stage"] = raw["stage"]
            summary = raw.get("summary")
            if summary:
                # Final write (run_demo_episode.py's `main()` on completion):
                # the full stats dict, itself carrying "samples" too -- this
                # naturally supersedes whatever partial `samples` list was
                # synced below while the run was still in progress; it's
                # that SAME cumulative list (same collection, just at its
                # final length), never a second, independently-built one.
                fields.update({
                    "workload_hash": summary.get("workload_hash"),
                    "workload_seed": summary.get("workload_seed"),
                    "workload_duration_s": summary.get("workload_duration_seconds"),
                    "summary": summary,
                })
            else:
                # Still running: `raw["samples"]`, when present, is the
                # subprocess's own CUMULATIVE list as of this exact poll
                # (not a delta) -- storing it is a plain idempotent
                # overwrite, so repeated polling of an unchanged payload
                # never duplicates anything, and a longer list on the next
                # poll simply replaces the shorter one.
                partial_summary = {"samples": raw["samples"]} if "samples" in raw else None
                if partial_summary is not None:
                    fields["summary"] = partial_summary
                if raw.get("workload_hash"):
                    fields.update({
                        "workload_hash": raw.get("workload_hash"),
                        "workload_seed": raw.get("workload_seed"),
                        "workload_duration_s": raw.get("workload_duration_seconds"),
                    })
            experiment_store.update_run(self.run_id, **fields)

        return raw

    def recent_log_lines(self, n: int = 20) -> list:
        if self._log_path is None or not self._log_path.exists():
            return []
        try:
            lines = self._log_path.read_text(errors="replace").splitlines()
        except OSError:
            return []
        return lines[-n:]
