#!/usr/bin/env python3
"""scripts/generate_iteration2_dynamics_calibration_schedules.py

Phase 16I.1: generates and hashes, OFFLINE, the 8 canonical Iteration-2
DEVELOPMENT_FITTING traffic schedules frozen by Phase16H
(task_outputs/phase_16h_iteration2_development_matrix.csv). Mirrors
scripts/generate_dynamics_calibration_schedules.py's own pattern exactly --
same `generate_and_persist()` pure path, same deterministic-arrivals policy,
same steady-state 180s traffic-duration convention -- but writes into a
completely separate, isolated Iteration-2 config tree. This script never
opens an HTTP client and never sends a single request to demo-app. It never
reads or writes anything under the historical
configs/traffic_profiles/rl_dynamics_calibration/ tree.

Per docs/experimental_design_corrections.md and
scripts/generate_dynamics_calibration_schedules.py's own note: "the same
schedule is replayed for every repeat of a given cell" -- exactly ONE
schedule is generated per Iteration-2 development CELL (8 total), reused by
all 3 planned repeats of that cell, matching the historical convention
exactly rather than generating one schedule per repeat.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
I2_CAL_ROOT = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration_iteration2"
I2_DEV_DIR = I2_CAL_ROOT / "development"

import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.generate_tuning_schedule import generate_and_persist  # noqa: E402

INTENSITY = 3
ENDPOINT = "/work"
SEED = 42  # provenance only -- deterministic arrivals never consult it (matches the historical convention exactly)

# ---------------------------------------------------------------------------
# Frozen Iteration-2 development cells -- a deterministic, mechanical
# projection of the authoritative Phase16H design
# (task_outputs/phase_16h_iteration2_development_matrix.csv). Must never
# diverge from that file; this script does not invent any coordinate.
# ---------------------------------------------------------------------------
DEVELOPMENT_CELLS: List[Tuple[str, int, float]] = [
    ("i2dev_r3_L1", 3, 6.0),
    ("i2dev_r3_L2", 3, 15.0),
    ("i2dev_r3_L3", 3, 24.0),
    ("i2dev_r3_L4", 3, 30.0),
    ("i2dev_r5_L1", 5, 10.0),
    ("i2dev_r5_L2", 5, 25.0),
    ("i2dev_r7_L1", 7, 14.0),
    ("i2dev_r7_L2", 7, 35.0),
]

# Unchanged historical steady-state timing convention (60s warmup + 120s
# scored = 180s traffic phase) -- reused exactly, never redefined.
STEADY_STATE_WARMUP_SECONDS = 60.0
STEADY_STATE_SCORED_SECONDS = 120.0
STEADY_STATE_TRAFFIC_DURATION = STEADY_STATE_WARMUP_SECONDS + STEADY_STATE_SCORED_SECONDS  # 180s


def _write_profile(path: Path, profile_name: str, phases: List[dict]) -> None:
    doc = {
        "profile": profile_name,
        "description": f"Phase 16I.1 Iteration-2 development-fitting calibration schedule: {profile_name}",
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

    for cell_id, replicas, total_rps in DEVELOPMENT_CELLS:
        profile_path = I2_DEV_DIR / f"{cell_id}.yaml"
        schedule_path = I2_DEV_DIR / f"{cell_id}_schedule.csv"
        _write_profile(
            profile_path, cell_id,
            [{"type": "constant", "name": "steady_state", "duration_seconds": STEADY_STATE_TRAFFIC_DURATION, "target_rps": total_rps}],
        )
        digest = _generate_one(profile_path, schedule_path)
        hashes[cell_id] = {
            "kind": "iteration2_development_fitting", "replicas": replicas, "total_offered_rps": total_rps,
            "schedule_sha256": digest,
            "schedule_path": str(schedule_path.relative_to(PROJECT_ROOT)),
        }

    registry_path = I2_CAL_ROOT / "iteration2_development_schedule_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)

    print(f"Generated/verified {len(hashes)} canonical Iteration-2 development schedules.")
    print(f"Registry written: {registry_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
