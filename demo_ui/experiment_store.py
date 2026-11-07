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
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

STORE_DIR = Path(__file__).resolve().parent / ".run_store"
STORE_PATH = STORE_DIR / "runs.json"

TERMINAL_STATUSES = {"completed", "error", "exited"}

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
    Called exactly once per real click of START (RealExperimentController
    guards against a second call while already running), so this is the
    only place a new run_id is minted -- a Streamlit rerun never calls this
    again for the same click."""
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


def active_run(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The most recently started run still in progress, or None."""
    path = path if path is not None else STORE_PATH
    running = [r for r in _load(path) if r.get("status") == "running"]
    if not running:
        return None
    return max(running, key=lambda r: r["started_at"])


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


def latest_compatible_pair(run: Dict[str, Any], path: Optional[Path] = None) -> Dict[str, Optional[Dict[str, Any]]]:
    """Given a completed run, returns {"hpa": <run-or-None>, "ppo": <run-or-None>}
    -- the latest completed runs (including `run` itself) sharing the exact
    same workload identity (profile/hash/seed/duration), one per
    controller. Never pairs runs from different workloads."""
    path = path if path is not None else STORE_PATH
    candidates = [r for r in completed_runs(path) if _same_workload(r, run)]
    result: Dict[str, Optional[Dict[str, Any]]] = {"hpa": None, "ppo": None}
    for r in candidates:  # already newest-first; first match per controller wins
        key = r.get("controller", "").lower()
        if key in result and result[key] is None:
            result[key] = r
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
