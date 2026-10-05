"""scripts/hpa_phase_entry_snapshot.py

Tiny, pure-function helper for Phase 11C tuning analysis -- NOT the general
results-analysis framework (that remains deliberately unbuilt for now).

Defines the authoritative "phase-entry snapshot" concept from Phase 11C.3:
for every phase in a tuning profile, the first `hpa_status.csv` sample at or
after that phase's planned start time, giving the ACTUAL observed replica/CPU
state a candidate entered that phase with -- never assumed from single-replica
Phase-10 calibration data (see CORR-19 / task_outputs/phase_11c3_*.md).

This module does not run traffic, does not talk to Kubernetes/Prometheus, and
does not choose a winner -- it only reads two already-persisted files
(a traffic profile YAML and one run's hpa_status.csv) and reports what was
actually observed at each phase boundary.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loadgen.traffic_generator import parse_traffic_profile  # noqa: E402


PHASE_ENTRY_FIELDS = [
    "hpa_status_current_replicas",
    "hpa_status_desired_replicas",
    "deployment_spec_desired_replicas",
    "deployment_status_replicas",
    "deployment_status_ready_replicas",
    "deployment_status_available_replicas",
    "hpa_current_cpu_utilization_percent",
    "last_scale_time",
]


def compute_phase_boundaries(profile_path: Path) -> Dict[str, float]:
    """Return {phase_name: planned_start_elapsed_seconds} for a traffic profile."""
    profile = parse_traffic_profile(profile_path)
    boundaries: Dict[str, float] = {}
    acc = 0.0
    for phase in profile.phases:
        boundaries[phase.name] = acc
        acc += phase.duration_seconds
    return boundaries


def find_phase_entry_row(
    hpa_status_rows: List[Dict[str, Any]], phase_start_seconds: float
) -> Optional[Dict[str, Any]]:
    """The first hpa_status.csv row observed at or after a phase's planned start.

    Returns None if no sample exists at or after that time (e.g. the collector
    stopped before the phase began -- a run that should likely be flagged
    invalid, not silently backfilled)."""
    candidates = [
        r for r in hpa_status_rows
        if r.get("elapsed_seconds") not in ("", None)
        and float(r["elapsed_seconds"]) >= phase_start_seconds
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: float(r["elapsed_seconds"]))


def phase_entry_snapshots(
    hpa_status_csv_path: Path, profile_path: Path
) -> Dict[str, Optional[Dict[str, Any]]]:
    """For every phase in the profile, the actual observed state entering it.

    Each snapshot (when found) contains: phase_name, planned_start_seconds,
    observed_elapsed_seconds, and every field in PHASE_ENTRY_FIELDS -- taken
    verbatim from the first hpa_status.csv sample at/after that phase's start.
    A missing snapshot (None) means no sample existed at or after that phase
    began; this should be treated as a data-completeness question for the
    run's validity, not silently ignored.
    """
    boundaries = compute_phase_boundaries(profile_path)

    with open(hpa_status_csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    snapshots: Dict[str, Optional[Dict[str, Any]]] = {}
    for phase_name, start_seconds in boundaries.items():
        row = find_phase_entry_row(rows, start_seconds)
        if row is None:
            snapshots[phase_name] = None
            continue
        snapshot = {
            "phase_name": phase_name,
            "planned_start_seconds": start_seconds,
            "observed_elapsed_seconds": float(row["elapsed_seconds"]),
        }
        for field in PHASE_ENTRY_FIELDS:
            snapshot[field] = row.get(field, "")
        snapshots[phase_name] = snapshot

    return snapshots
