#!/usr/bin/env python3
"""scripts/generate_iteration2b_r7_remediation_schedule.py

Phase 16J.1: generates and hashes, OFFLINE, the single canonical
Iteration-2B r7-remediation traffic schedule frozen by this phase
(replicas=7, offered_rps=24.5 -- the deterministic midpoint of the known
usable r7 anchor at 14.0 RPS and the failed r7 development boundary at 35.0
RPS). Mirrors scripts/generate_iteration2_dynamics_calibration_schedules.py's
own pattern exactly -- same `generate_and_persist()` pure path, same
deterministic-arrivals policy, same steady-state 180s traffic-duration
convention -- but writes into a completely separate, isolated Iteration-2B
config tree. This script never opens an HTTP client and never sends a single
request to demo-app. It never reads or writes anything under the historical
or Iteration-2 config trees.

Per the established convention: "the same schedule is replayed for every
repeat of a given cell" -- exactly ONE schedule is generated for this one
remediation cell, reused by all 3 planned repeats.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
I2B_CAL_ROOT = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration_iteration2b"
I2B_DIR = I2B_CAL_ROOT / "r7_remediation"

import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.generate_tuning_schedule import generate_and_persist  # noqa: E402

INTENSITY = 3
ENDPOINT = "/work"
SEED = 42  # provenance only -- deterministic arrivals never consult it (matches the historical/I2 convention exactly)

# Frozen Iteration-2B remediation cell -- a deterministic midpoint
# contraction fixed BEFORE any new outcome exists, per this phase's own
# design decision. Must never diverge from
# phase_16j1_iteration2b_r7_remediation_design.yaml.
REMEDIATION_CELLS: List[Tuple[str, int, float]] = [
    ("i2b_r7_mid", 7, 24.5),
]

# Unchanged historical/I2 steady-state timing convention (60s warmup + 120s
# scored = 180s traffic phase) -- reused exactly, never redefined.
STEADY_STATE_WARMUP_SECONDS = 60.0
STEADY_STATE_SCORED_SECONDS = 120.0
STEADY_STATE_TRAFFIC_DURATION = STEADY_STATE_WARMUP_SECONDS + STEADY_STATE_SCORED_SECONDS  # 180s


def _write_profile(path: Path, profile_name: str, phases: List[dict]) -> None:
    doc = {
        "profile": profile_name,
        "description": f"Phase 16J.1 Iteration-2B r7-remediation calibration schedule: {profile_name}",
        "seed": SEED,
        "endpoint": ENDPOINT,
        "intensity": INTENSITY,
        "phases": phases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def _generate_one(profile_path: Path, schedule_path: Path) -> str:
    if schedule_path.exists():
        digest = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
        return digest
    return generate_and_persist(profile_path, "deterministic", SEED, schedule_path)


def main() -> None:
    hashes: Dict[str, dict] = {}

    for cell_id, replicas, total_rps in REMEDIATION_CELLS:
        profile_path = I2B_DIR / f"{cell_id}.yaml"
        schedule_path = I2B_DIR / f"{cell_id}_schedule.csv"
        _write_profile(
            profile_path, cell_id,
            [{"type": "constant", "name": "steady_state", "duration_seconds": STEADY_STATE_TRAFFIC_DURATION, "target_rps": total_rps}],
        )
        digest = _generate_one(profile_path, schedule_path)
        hashes[cell_id] = {
            "kind": "iteration2b_r7_remediation", "replicas": replicas, "total_offered_rps": total_rps,
            "schedule_sha256": digest,
            "schedule_path": str(schedule_path.relative_to(PROJECT_ROOT)),
        }

    registry_path = I2B_CAL_ROOT / "iteration2b_r7_remediation_schedule_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)

    print(f"Generated/verified {len(hashes)} canonical Iteration-2B remediation schedule(s).")
    print(f"Registry written: {registry_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
