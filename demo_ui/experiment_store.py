"""experiment_store.py -- the single source of truth for REAL (LIVE)
experiment runs, shared by Experiment/Live Details/History/Comparison.

Deliberately NOT `st.session_state`: session_state is per-browser-session
and is cleared on a Streamlit restart (and, for widget-bound keys, even on
ordinary page navigation -- see live_execution_adapter.py's own note on
that). A run record must survive page navigation, reruns, and normal
Streamlit restarts, so it is persisted to a small JSON file instead.

Deliberately NOT a scientific ledger/checkpoint path: this lives entirely
under `persistent-data/demo_ui/` (gitignored, demo-only -- same directory
as the status/log files `live_execution_adapter.py` already writes there,
and for the SAME reason: the system temp dir is wiped on every host
restart, which used to orphan this very store's "running" records with no
process left to ever finish or be stopped), is never read by
`scripts/evaluate_ppo_vs_hpa.py` or any training/evaluation code, and has
no bearing on governing hashes, checkpoints, or final-holdout state.

Pure Python + stdlib only (no Streamlit import, no live-cluster/kubectl
calls) so it stays importable and unit-testable in isolation, same
discipline as `models.py`. Live-cluster cleanup triggered by reconciliation
lives one layer up (`scripts/v2_live_cluster_state.py`, called from the UI
and from `scripts/demo_recover.py`), never here.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = PROJECT_ROOT / "persistent-data" / "demo_ui"
STORE_PATH = STORE_DIR / "runs.json"

# Pre-move location (V2 Phase 9-era): ad hoc, gitignored, and NOT under
# `persistent-data/`, so a record created there had no persistence
# guarantee across a host restart either (the directory itself survived,
# but nothing else about this module's design assumed that). Migrated
# in-place, once, the first time this module is imported in an
# environment that still has the old file and not yet the new one -- never
# duplicated, never silently dropped.
_LEGACY_STORE_PATH = Path(__file__).resolve().parent / ".run_store" / "runs.json"

STALE_STATUS = "INTERRUPTED (stale)"
TERMINAL_STATUSES = {"completed", "error", "exited", "stopped", "refused", STALE_STATUS}

# Terminal outcomes a run's OWN status file (written by
# `scripts/run_demo_episode_v2.py`, surviving under `persistent-data/`
# across a host restart) can record about itself. Checked BEFORE any
# pid/heartbeat liveness check -- it is ground truth about how the run
# actually ended, and must never be overwritten by a fabricated
# `STALE_STATUS` verdict just because the store record hasn't caught up
# yet (e.g. the UI was closed when the run finished).
_STATUS_FILE_TERMINAL = {"completed", "error", "stopped", "refused"}


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


def _migrate_legacy_store_once() -> None:
    """Best-effort, one-time: copies the pre-move `.run_store/runs.json`
    (if it exists and the new `persistent-data/demo_ui/runs.json` does
    not yet) to the new canonical location. Never runs again once the new
    file exists, and never raises -- a failed migration must not crash
    import; it just means that one environment starts its history fresh."""
    try:
        if STORE_PATH.exists() or not _LEGACY_STORE_PATH.exists():
            return
        data = json.loads(_LEGACY_STORE_PATH.read_text())
        if isinstance(data, list):
            _save(data, STORE_PATH)
    except (OSError, json.JSONDecodeError):
        pass


_migrate_legacy_store_once()


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
        "pgid": None,
        "status_path": None,
        "log_path": None,
        "expected_duration_s": None,
        "error": None,
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
    touched recently, AND (when an expected duration was recorded at
    start) elapsed time has not blown past it by more than a 5-minute
    grace period. A record with neither a pid nor a status-file signal
    (e.g. hand-built in a test, or predating those fields) cannot be
    verified and counts as not alive, so it gets marked stale rather than
    locking the UI forever."""
    pid = rec.get("pid")
    status_path = rec.get("status_path")
    if pid is None and not status_path:
        return False
    if pid is not None and not _pid_alive(pid):
        return False
    if status_path and not _status_file_fresh(status_path):
        return False
    expected = rec.get("expected_duration_s")
    if expected is not None:
        elapsed = time.time() - rec.get("started_at", time.time())
        if elapsed > expected + 300.0:
            return False
    return True


def _stale_reason(rec: Dict[str, Any]) -> str:
    """Human-readable reason recorded on a record marked `STALE_STATUS`,
    so RECOVER/History shows WHY a run was declared interrupted, not just
    that it was."""
    pid = rec.get("pid")
    if pid is not None and not _pid_alive(pid):
        return f"process (pid {pid}) is no longer running"
    status_path = rec.get("status_path")
    if status_path and not _status_file_fresh(status_path):
        return "heartbeat (status file) went stale -- no update in over 2 minutes"
    expected = rec.get("expected_duration_s")
    if expected is not None:
        elapsed = time.time() - rec.get("started_at", time.time())
        if elapsed > expected + 300.0:
            return "run exceeded its expected duration plus a 5 minute grace period"
    return "liveness could not be verified (no pid or status file recorded)"


def _read_status_file(status_path: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(Path(status_path).read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def status_file_fields_for_store(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Maps a status-file payload (as `scripts/run_demo_episode_v2.py`
    writes it) onto the `update_run()` fields that should be copied into
    the persisted store record. The ONE place this mapping happens, so
    `RealExperimentControllerV2.status()`'s normal polling path and this
    module's own reconciliation path (which reads the same file when the
    UI was not around to poll it) can never drift apart."""
    fields: Dict[str, Any] = {"status": raw["status"]}
    if raw.get("stage"):
        fields["stage"] = raw["stage"]
    if raw.get("error"):
        fields["error"] = raw["error"]
    summary = raw.get("summary")
    if summary:
        fields["summary"] = summary
        fields["workload_hash"] = summary.get("workload_hash")
        fields["workload_seed"] = summary.get("workload_seed")
        fields["workload_duration_s"] = summary.get("workload_duration_seconds")
    elif "samples" in raw:
        fields["summary"] = {"samples": raw["samples"]}
    return fields


def _terminal_outcome_from_status_file(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """If this record's OWN status file already recorded a terminal
    outcome (completed/error/stopped/refused) that the store record
    hasn't caught up to yet -- e.g. the run finished (or was refused)
    while no UI instance was polling it -- returns that raw payload so the
    caller can copy the REAL outcome into the store instead of fabricating
    `STALE_STATUS`. Returns None if there is no status file, it is
    unreadable, or it does not (yet) show a terminal status."""
    status_path = rec.get("status_path")
    if not status_path:
        return None
    raw = _read_status_file(status_path)
    if raw is None or raw.get("status") not in _STATUS_FILE_TERMINAL:
        return None
    return raw


def _reconcile_running(path: Path) -> Dict[str, Any]:
    """Shared implementation behind `active_run()` and the public
    `reconcile()`: for every `"running"` record, in order --
    (1) does its OWN status file already show a real terminal outcome the
        store hasn't caught up to? Copy that in. This is ground truth and
        is checked BEFORE any liveness check, so a run that completed (or
        was refused, errored, or stopped) while no UI was watching is
        never mistaken for merely "stale".
    (2) otherwise, is it still genuinely alive (pid + fresh heartbeat +
        within its expected duration)? Keep it `"running"`.
    (3) otherwise, mark it `STALE_STATUS` with a reason -- never left to
        lock the UI's controls forever, and never deleted from history."""
    running = [r for r in _load(path) if r.get("status") == "running"]
    alive: List[Dict[str, Any]] = []
    newly_terminal: List[Dict[str, Any]] = []
    newly_stale: List[Dict[str, Any]] = []
    for rec in running:
        raw = _terminal_outcome_from_status_file(rec)
        if raw is not None:
            update_run(rec["run_id"], path=path, **status_file_fields_for_store(raw))
            newly_terminal.append(get_run(rec["run_id"], path))
            continue
        if is_run_alive(rec):
            alive.append(rec)
            continue
        update_run(rec["run_id"], path=path, status=STALE_STATUS, stage="INTERRUPTED", error=_stale_reason(rec))
        newly_stale.append(get_run(rec["run_id"], path))
    active = max(alive, key=lambda r: r["started_at"]) if alive else None
    return {"active": active, "newly_terminal": newly_terminal, "newly_stale": newly_stale}


def active_run(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The most recently started run still genuinely in progress, or None.

    Any `"running"` record that fails the liveness check (dead PID and/or a
    missing/stale status file -- e.g. the host machine was restarted and
    the process writing it is simply gone) is updated in place to
    `STALE_STATUS` rather than left to lock the UI's controls forever --
    UNLESS its own status file already recorded a real terminal outcome,
    which is copied in instead (see `_reconcile_running`). This never
    deletes run history -- it only corrects a status field that no longer
    reflects reality, the same way a real process ending would."""
    path = path if path is not None else STORE_PATH
    return _reconcile_running(path)["active"]


def reconcile(path: Optional[Path] = None) -> Dict[str, Any]:
    """Public entry point for callers that need to know not just the
    active run but WHAT reconciliation just changed: `{"active":
    <rec-or-None>, "newly_terminal": [...], "newly_stale": [...]}`. Used
    by the Experiment page (to trigger a live-cluster idle-state cleanup
    check exactly when a run is freshly discovered to be stale/dead, never
    on every cheap page poll) and by `scripts/demo_recover.py`."""
    path = path if path is not None else STORE_PATH
    return _reconcile_running(path)


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
    -- the latest GENUINELY completed runs (including `run` itself, if it
    qualifies) sharing the exact same workload identity (profile/hash/
    seed/duration), one per controller family. Never pairs runs from
    different workloads, and -- task sec.7 -- never offers an interrupted,
    stopped, refused, or errored run as "the" result for its controller
    family: `completed_runs()` (used by History) deliberately includes
    every TERMINAL status so a run's real outcome is always visible there,
    but only `status == "completed"` ever qualifies as a pairing
    candidate here."""
    path = path if path is not None else STORE_PATH
    candidates = [r for r in completed_runs(path) if r.get("status") == "completed" and _same_workload(r, run)]
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
