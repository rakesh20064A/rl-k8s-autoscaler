"""experiment_store.py -- the single source of truth for REAL (LIVE)
experiment runs, shared by Experiment/Live Details/History/Comparison.

Deliberately NOT `st.session_state`: session_state is per-browser-session
and is cleared on a Streamlit restart (and, for widget-bound keys, even on
ordinary page navigation -- see live_execution_adapter.py's own note on
that). A run record must survive page navigation, reruns, and normal
Streamlit restarts, so it is persisted to a small JSON file instead.

Deliberately NOT a scientific ledger/checkpoint path: this lives entirely
under `demo_ui/.run_store/` (gitignored, demo-only), is never read by
`scripts/evaluate_ppo_vs_hpa.py` or any training/evaluation code, and has
no bearing on governing hashes, checkpoints, or final-holdout state.

Pure Python + stdlib only (no Streamlit import) so it stays importable and
unit-testable in isolation, same discipline as `models.py`.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

STORE_DIR = Path(__file__).resolve().parent / ".run_store"
STORE_PATH = STORE_DIR / "runs.json"

STALE_STATUS = "INTERRUPTED (stale)"
TERMINAL_STATUSES = {"completed", "error", "exited", STALE_STATUS}

# How long a "running" record may go without a liveness signal before it is
# treated as orphaned (e.g. the host machine was restarted and the process
# that was writing its status file is simply gone). Decision steps land on
# the status file roughly every 30s (see scripts/run_demo_episode_v2.py),
# so this gives a generous multiple of that before calling it stale.
STALE_STATUS_FILE_AGE_S = 120.0

# Fields that identify "the same workload" for HPA/PPO compatibility
# (Comparison pairing) -- same profile, same hash, same seed, same
# configured duration. Never includes `controller` (that's the axis being
# compared) or timing fields.
_WORKLOAD_IDENTITY_FIELDS = ("profile_name", "workload_hash", "workload_seed", "workload_duration_s")


def _load(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = path if path is not None else STORE_PATH
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def _save(records: List[Dict[str, Any]], path: Optional[Path] = None) -> None:
    path = path if path is not None else STORE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(records, default=str, indent=None))
    tmp.replace(path)


def start_run(controller: str, profile_name: str, path: Optional[Path] = None) -> str:
    """Creates a new `status="running"` record and returns its `run_id`.
    Called exactly once per real click of START (RealExperimentControllerV2
    guards against a second call while already running), so this is the
    only place a new run_id is minted -- a Streamlit rerun never calls this
    again for the same click.

    `pid`/`status_path` start out `None` and are filled in afterwards (via
    `update_run`) once the subprocess actually exists -- see
    `live_execution_adapter.py`. They are what lets `active_run()` tell a
    genuinely running process apart from a stale record left behind by a
    host restart (which wipes any status/log files kept in a non-persistent
    location -- this is why those files now live under
    `persistent-data/demo_ui/`, not `/tmp`)."""
    path = path if path is not None else STORE_PATH
    run_id = uuid.uuid4().hex[:12]
    records = _load(path)
    records.append({
        "run_id": run_id,
        "started_at": time.time(),
        "completed_at": None,
        "data_source": "LIVE",
        "controller": controller,
        "profile_name": profile_name,
        "workload_hash": None,
        "workload_seed": None,
        "workload_duration_s": None,
        "status": "running",
        "stage": "STARTING",
        "summary": None,
        "pid": None,
        "status_path": None,
    })
    _save(records, path)
    return run_id


def update_run(run_id: str, path: Optional[Path] = None, **fields: Any) -> None:
    """Merges `fields` into the record for `run_id`. Setting `status` to a
    terminal value auto-stamps `completed_at` (once) unless the caller
    already supplied one. No-op if `run_id` is unknown (e.g. called after
    the store was manually cleared)."""
    path = path if path is not None else STORE_PATH
    records = _load(path)
    changed = False
    for rec in records:
        if rec["run_id"] != run_id:
            continue
        rec.update(fields)
        if rec.get("status") in TERMINAL_STATUSES and not rec.get("completed_at"):
            rec["completed_at"] = time.time()
        changed = True
        break
    if changed:
        _save(records, path)


def get_run(run_id: str, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = path if path is not None else STORE_PATH
    for rec in _load(path):
        if rec["run_id"] == run_id:
            return rec
    return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists (owned by someone/something else) -- still alive
    except OSError:
        return False
    return True


def _status_file_fresh(status_path: str, max_age_s: float = STALE_STATUS_FILE_AGE_S) -> bool:
    try:
        age = time.time() - Path(status_path).stat().st_mtime
    except OSError:
        return False
    return age < max_age_s


def is_run_alive(rec: Dict[str, Any]) -> bool:
    """A `"running"` record is only actually alive if every liveness signal
    it carries confirms it -- a stored pid must still be a live process, AND
    (when a status-file path is also stored) that file must have been
    touched recently. A record with neither signal recorded (e.g. hand-built
    in a test, or predating this field) cannot be verified and counts as not
    alive, so it gets marked stale rather than locking the UI forever."""
    pid = rec.get("pid")
    status_path = rec.get("status_path")
    if pid is None and not status_path:
        return False
    if pid is not None and not _pid_alive(pid):
        return False
    if status_path and not _status_file_fresh(status_path):
        return False
    return True


def active_run(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The most recently started run still genuinely in progress, or None.

    Any `"running"` record that fails the liveness check (dead PID and/or a
    missing/stale status file -- e.g. the host machine was restarted and
    the process writing it is simply gone) is updated in place to
    `STALE_STATUS` rather than left to lock the UI's controls forever. This
    never deletes run history -- it only corrects a status field that no
    longer reflects reality, the same way a real process ending would."""
    path = path if path is not None else STORE_PATH
    running = [r for r in _load(path) if r.get("status") == "running"]
    alive = []
    for rec in running:
        if is_run_alive(rec):
            alive.append(rec)
        else:
            update_run(rec["run_id"], path=path, status=STALE_STATUS, stage="INTERRUPTED")
    if not alive:
        return None
    return max(alive, key=lambda r: r["started_at"])


def latest_run(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The most recently started run of any status, or None."""
    path = path if path is not None else STORE_PATH
    records = _load(path)
    if not records:
        return None
    return max(records, key=lambda r: r["started_at"])


def completed_runs(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """All terminal-status runs, newest first."""
    path = path if path is not None else STORE_PATH
    records = [r for r in _load(path) if r.get("status") in TERMINAL_STATUSES]
    records.sort(key=lambda r: r.get("completed_at") or r["started_at"], reverse=True)
    return records


def _same_workload(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return all(a.get(f) is not None and a.get(f) == b.get(f) for f in _WORKLOAD_IDENTITY_FIELDS)


def _controller_family(controller: Optional[str]) -> Optional[str]:
    """Classifies a controller id into "hpa" or "ppo" by substring, not
    exact match -- the real V2 controller ids are `ppo_v2` and
    `hpa_v2_cpu_memory` (see components/controls_v2.py / config_v2.py),
    not the bare `"ppo"`/`"hpa"` this used to require, and V1 fixtures use
    `"HPA"`/`"PPO"`. Any future controller id containing "hpa" or "ppo"
    keeps working without another change here."""
    c = (controller or "").lower()
    if "ppo" in c:
        return "ppo"
    if "hpa" in c:
        return "hpa"
    return None


def latest_compatible_pair(run: Dict[str, Any], path: Optional[Path] = None) -> Dict[str, Optional[Dict[str, Any]]]:
    """Given a completed run, returns {"hpa": <run-or-None>, "ppo": <run-or-None>}
    -- the latest completed runs (including `run` itself) sharing the exact
    same workload identity (profile/hash/seed/duration), one per
    controller family. Never pairs runs from different workloads."""
    path = path if path is not None else STORE_PATH
    candidates = [r for r in completed_runs(path) if _same_workload(r, run)]
    result: Dict[str, Optional[Dict[str, Any]]] = {"hpa": None, "ppo": None}
    for r in candidates:  # already newest-first; first match per controller family wins
        family = _controller_family(r.get("controller"))
        if family is not None and result[family] is None:
            result[family] = r
    return result


def extract_series(run: Optional[Dict[str, Any]], field: str) -> "tuple[List[float], List[Any]]":
    """Pulls one metric's time series out of a run record's persisted
    per-decision-step samples (`run["summary"]["samples"]`, written by
    `scripts/run_demo_episode.py` only once the real episode has produced
    them). Returns ([], []) -- never fabricated points -- if `run` is None
    or has no samples yet (e.g. still running, or ended before any
    decision step completed)."""
    if run is None:
        return [], []
    samples = (run.get("summary") or {}).get("samples") or []
    timestamps = [s["time_s"] for s in samples if s.get(field) is not None]
    values = [s[field] for s in samples if s.get(field) is not None]
    return timestamps, values


def clear(path: Optional[Path] = None) -> None:
    """Test/debug helper: wipes the store. Never called from app code."""
    path = path if path is not None else STORE_PATH
    _save([], path)
