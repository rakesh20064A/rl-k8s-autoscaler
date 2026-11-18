"""live_execution_adapter.py -- REAL experiment execution for the demo UI.

This module is the one place in demo_ui allowed to launch a real
subprocess. `start()` launches `scripts/run_demo_episode_v2.py` -- a
small, isolated, non-scientific adapter -- in the project's scientific
`.venv` (NOT the demo UI's own `requirements-ui.txt` venv, which lacks
stable_baselines3/kubernetes client libs). The subprocess is started in
its own process group (`start_new_session=True`) so STOP can terminate it
and everything it spawned (the traffic generator child process) in one
`killpg` call, without touching any other process on the machine.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
RUNNER_SCRIPT_V2 = PROJECT_ROOT / "scripts" / "run_demo_episode_v2.py"

_TERMINATE_WAIT_SECONDS = 10.0


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

        status_fd, status_name = tempfile.mkstemp(prefix="du_demo_v2_status_", suffix=".json")
        os.close(status_fd)
        log_fd, log_name = tempfile.mkstemp(prefix="du_demo_v2_log_", suffix=".log")
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
            fields = {"status": raw["status"]}
            if raw.get("stage"):
                fields["stage"] = raw["stage"]
            if raw.get("error"):
                fields["error"] = raw["error"]
            summary = raw.get("summary")
            if summary:
                fields.update({
                    "workload_hash": summary.get("workload_hash"), "workload_seed": summary.get("workload_seed"),
                    "workload_duration_s": summary.get("workload_duration_seconds"), "summary": summary,
                })
            elif "samples" in raw:
                fields["summary"] = {"samples": raw["samples"]}
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
