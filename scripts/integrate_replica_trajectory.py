#!/usr/bin/env python3
"""scripts/integrate_replica_trajectory.py

Phase 11C.4I: ONE shared, offline, read-only integration method for
replica-trajectory efficiency metrics (replica-seconds/hours, average, peak),
used to recompute all Block-1 official trials with a single consistent
bookkeeping convention instead of each trial report's own ad-hoc windowing.

Operates ONLY on an already-persisted `hpa_status.csv` file. No Kubernetes
API access, no Prometheus access, no network access of any kind.

Integration semantics (frozen for this and all future official trials):
- Integrate over the exact half-open interval [0, SCORED_END) seconds
  (SCORED_END defaults to 1230.0, the frozen hpa_tuning_v2.yaml duration).
- Each sampled value is treated as a RIGHT-CONTINUOUS step function: the
  value observed at timestamp t_i applies from t_i until the next observed
  timestamp (exclusive), and the final observed value (at or before
  SCORED_END) is carried forward to exactly SCORED_END.
- If the first sample's elapsed_seconds is greater than 0, the interval
  [0, first_sample_elapsed) is assigned `start_value` (the standardized
  known starting replica count for this project's official trials, 1) --
  never left uncounted and never assumed to equal the first sample's value
  by coincidence.
- Samples at or after SCORED_END are ignored entirely (never extrapolated
  into or used to influence the scored window).
- A row whose target field is missing/empty/unparseable is skipped as a
  transition point (it does not start a new segment) -- the previous
  segment's value simply continues through that row's timestamp. This
  mirrors how a single transient collector error (e.g. a brief
  target-health blip while a new pod is not yet Ready) must not silently
  corrupt the integrated trajectory for the field it failed to report.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

SCORED_END_DEFAULT = 1230.0
STANDARDIZED_START_VALUE = 1.0


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_hpa_status_rows(path: Path) -> List[Dict[str, Any]]:
    """Read and sort hpa_status.csv rows by elapsed_seconds. Rows with a
    missing/unparseable elapsed_seconds are dropped entirely (they cannot be
    placed on the timeline at all)."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            elapsed = _parse_float(raw.get("elapsed_seconds"))
            if elapsed is None:
                continue
            raw["_elapsed_seconds"] = elapsed
            rows.append(raw)
    rows.sort(key=lambda r: r["_elapsed_seconds"])
    return rows


def integrate_step_function(
    rows: List[Dict[str, Any]],
    field: str,
    scored_end: float = SCORED_END_DEFAULT,
    start_value: float = STANDARDIZED_START_VALUE,
) -> Dict[str, float]:
    """Integrate one field of an hpa_status.csv row list as a right-
    continuous step function over [0, scored_end).

    Returns replica_seconds, replica_hours, average, peak, sample_count
    (the number of distinct valid samples actually used within the window).
    Raises ValueError if there is no usable data at all within the window.
    """
    # Only rows within the scored window, with a parseable value for this field.
    usable = [
        (r["_elapsed_seconds"], _parse_float(r.get(field)))
        for r in rows
        if r["_elapsed_seconds"] < scored_end
    ]
    usable = [(t, v) for (t, v) in usable if v is not None]

    if not usable:
        raise ValueError(f"No usable samples for field '{field}' within [0, {scored_end})")

    # Prepend the standardized start value at t=0 if the first usable
    # sample is not already at t=0.
    if usable[0][0] > 0.0:
        usable = [(0.0, start_value)] + usable

    replica_seconds = 0.0
    peak = usable[0][1]
    for i in range(len(usable)):
        t0, v0 = usable[i]
        t1 = usable[i + 1][0] if i + 1 < len(usable) else scored_end
        if t1 > scored_end:
            t1 = scored_end
        duration = t1 - t0
        if duration < 0:
            duration = 0.0
        replica_seconds += v0 * duration
        if v0 > peak:
            peak = v0

    average = replica_seconds / scored_end if scored_end > 0 else 0.0

    return {
        "replica_seconds": replica_seconds,
        "replica_hours": replica_seconds / 3600.0,
        "average": average,
        "peak": peak,
        "sample_count": len(usable),
    }


FIELDS = {
    "hpa_desired": "hpa_status_desired_replicas",
    "deployment_desired": "deployment_spec_desired_replicas",
    "current": "deployment_status_replicas",
    "ready": "deployment_status_ready_replicas",
    "available": "deployment_status_available_replicas",
}


def integrate_all_fields(
    hpa_status_csv_path: Path,
    scored_end: float = SCORED_END_DEFAULT,
    start_value: float = STANDARDIZED_START_VALUE,
) -> Dict[str, Dict[str, float]]:
    rows = read_hpa_status_rows(hpa_status_csv_path)
    return {
        label: integrate_step_function(rows, csv_field, scored_end, start_value)
        for label, csv_field in FIELDS.items()
    }


def find_transitions(
    rows: List[Dict[str, Any]], field: str, scored_end: float = SCORED_END_DEFAULT
) -> List[Dict[str, Any]]:
    """Return every row (within the scored window) where the given field's
    value differs from the immediately preceding valid value for that
    field -- i.e. every observed transition, in chronological order."""
    transitions = []
    prev: Optional[float] = None
    for r in rows:
        if r["_elapsed_seconds"] >= scored_end:
            continue
        v = _parse_float(r.get(field))
        if v is None:
            continue
        if prev is None or v != prev:
            transitions.append({"elapsed_seconds": r["_elapsed_seconds"], "value": v, "row": r})
        prev = v
    return transitions


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hpa_status_csv", type=Path)
    parser.add_argument("--scored-end", type=float, default=SCORED_END_DEFAULT)
    args = parser.parse_args()

    result = integrate_all_fields(args.hpa_status_csv, args.scored_end)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
