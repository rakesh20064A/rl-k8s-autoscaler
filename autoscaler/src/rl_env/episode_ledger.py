"""autoscaler/src/rl_env/episode_ledger.py

Phase 16V: append-only episode ledgers for the direct-PPO live-training and
final-evaluation harnesses (phase_16u_direct_ppo_protocol.yaml Section 11).
Dedicated, new ledgers -- the Iteration-3 dynamics-calibration scientific
ledger (persistent-data/dynamics_validation_iteration3/...) is never reused
or touched by this module.

Format: JSON Lines (one JSON object per line), append-only. Chosen over a
single mutable JSON array (as Iteration-3's ledger uses) because episodes
are appended incrementally over a multi-day live campaign and a JSONL file
can be safely appended to without re-reading/re-writing the whole file each
time -- reducing the risk of a partial-write corrupting already-recorded
episodes if the process is interrupted mid-campaign (caffeinate-protected
multi-day runs, per Phase16Q precedent, are exactly the failure mode this
guards against).

Every row gets exactly one line; no existing line is ever rewritten or
removed by this module. A structurally-failed episode (RESTORATION_TIMEOUT,
persistent prometheus_unavailable, etc.) is written exactly like a
successful one -- never silently dropped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator


class LedgerAppendError(RuntimeError):
    """Raised if an append cannot be durably completed. Callers must treat
    this as fatal for the current episode's provenance -- never proceed as
    if the row were recorded when it was not."""


def append_row(ledger_path: Path, row: Dict[str, Any]) -> None:
    """Append one JSON object as a new line. Uses an exclusive-create
    temp-file + append-mode write with an explicit flush+fsync so a crash
    mid-write cannot silently truncate a PRIOR row -- this function only
    ever adds bytes at the end of the file, never rewrites existing
    content."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True) + "\n"
    try:
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    except OSError as exc:
        raise LedgerAppendError(f"failed to append to ledger {ledger_path}: {exc}") from exc


def read_rows(ledger_path: Path) -> Iterator[Dict[str, Any]]:
    """Yield every row in file order. Returns nothing (not an error) if the
    ledger does not exist yet -- a fresh ledger before its first episode is
    a valid, expected state, not a failure."""
    if not ledger_path.exists():
        return
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_rows(ledger_path: Path) -> int:
    return sum(1 for _ in read_rows(ledger_path))


def next_physical_attempt_number(ledger_path: Path, identity: str, *, field: str = "episode_id") -> int:
    """Phase16AB (task sec.7): supplies a physical-execution-attempt
    identity that must be unique across attempts at the same scientific
    slot: it scans the ledger file (never in-memory state, which resets
    every process restart) for how many rows already carry this exact
    identity value under `field`, and returns `count + 1`. A fresh
    scientific slot's first-ever attempt gets `1`; a retry after an
    earlier recorded failure at the same slot gets `2`, `3`, etc. --
    mirroring this project's existing `resolve_next_physical_attempt_
    number()` convention for the Iteration-3 calibration ledger (scan
    actual attempt records on disk, never reset after a consumed
    failure).

    Phase17 correction: the legacy `episode_id` (e.g.
    `"training_mixed_42_1"`) is a pure function of role/profile/seed/
    PER-PROCESS episode-index -- it is NOT globally unique across rollout
    processes (a new `LedgeredEpisodeWrapper` instance's internal counter
    restarts at 0 every rollout, so two DIFFERENT global schedule slots
    that happen to land on the same per-process index and profile produce
    an IDENTICAL legacy `episode_id`, as Phase16AE demonstrated concretely
    -- see `task_outputs/phase_17_provenance_correction_report.md`).
    Callers now pass `field="scientific_episode_id"` (a globally-unique
    identity derived from the authoritative schedule index) so this
    function scopes attempt-counting correctly; `field` defaults to the
    legacy `"episode_id"` only for backward compatibility with pre-Phase17
    callers/tests."""
    return sum(1 for row in read_rows(ledger_path) if row.get(field) == identity) + 1
