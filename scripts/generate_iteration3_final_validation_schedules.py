#!/usr/bin/env python3
"""scripts/generate_iteration3_final_validation_schedules.py

Phase 16N: generates and hashes, OFFLINE, the 16 canonical Iteration-3
I3_FINAL_UNSEEN_VALIDATION traffic schedules frozen by Phase16N's fresh
validation matrix (configs are a deterministic, mechanical projection of
task_outputs/phase_16n_iteration3_fresh_validation_matrix.csv). Mirrors
scripts/generate_iteration2b_final_validation_schedules.py's own pattern
exactly -- same generate_and_persist() pure path, same deterministic-arrivals
policy, same steady-state 180s traffic-duration convention -- but writes into
a completely separate, isolated Iteration-3 final-validation config tree.
This script never opens an HTTP client and never sends a single request to
demo-app, and never reads or writes anything under the historical, Iteration-2
development, Iteration-2B remediation, or Iteration-2B final-validation config
trees.

Per the established convention: exactly ONE schedule is generated per
validation CELL (16 total, replicas 1..8 x {V1,V2}), reused by both of that
cell's 2 planned repeats -- never one schedule per repeat. No traffic is
generated or dispatched by running this script; it only writes profile YAML,
schedule CSV, and a registry JSON to disk.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
I3VAL_CAL_ROOT = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration_iteration3"
I3VAL_DIR = I3VAL_CAL_ROOT / "final_unseen"
MATRIX_PATH = PROJECT_ROOT / "task_outputs" / "phase_16n_iteration3_fresh_validation_matrix.csv"
TASK_OUTPUTS_ALT = Path("/Users/rakeshdandamudi/Capstone/task_outputs/phase_16n_iteration3_fresh_validation_matrix.csv")

import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.generate_tuning_schedule import generate_and_persist  # noqa: E402

INTENSITY = 3
ENDPOINT = "/work"
SEED = 42  # provenance only -- deterministic arrivals never consult it (matches every prior iteration's convention exactly)

STEADY_STATE_WARMUP_SECONDS = 60.0
STEADY_STATE_SCORED_SECONDS = 120.0
STEADY_STATE_TRAFFIC_DURATION = STEADY_STATE_WARMUP_SECONDS + STEADY_STATE_SCORED_SECONDS  # 180s


def _write_profile(path: Path, profile_name: str, phases: List[dict]) -> None:
    doc = {
        "profile": profile_name,
        "description": f"Phase 16N Iteration-3 final-validation calibration schedule: {profile_name}",
        "seed": SEED, "endpoint": ENDPOINT, "intensity": INTENSITY, "phases": phases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def _generate_one(profile_path: Path, schedule_path: Path) -> str:
    if schedule_path.exists():
        return hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    return generate_and_persist(profile_path, "deterministic", SEED, schedule_path)


def main() -> None:
    matrix_path = MATRIX_PATH if MATRIX_PATH.exists() else TASK_OUTPUTS_ALT
    matrix = list(csv.DictReader(open(matrix_path)))
    assert len(matrix) == 16

    hashes: Dict[str, dict] = {}
    for row in matrix:
        replicas = int(row["replicas"])
        rps = float(row["offered_rps"])
        cell_id = row["cell_id"]
        assert cell_id == f"i3val_r{replicas}_{row['label']}"
        profile_path = I3VAL_DIR / f"{cell_id}.yaml"
        schedule_path = I3VAL_DIR / f"{cell_id}_schedule.csv"
        _write_profile(
            profile_path, cell_id,
            [{"type": "constant", "name": "steady_state", "duration_seconds": STEADY_STATE_TRAFFIC_DURATION, "target_rps": rps}],
        )
        digest = _generate_one(profile_path, schedule_path)
        hashes[cell_id] = {
            "kind": "iteration3_final_unseen_validation", "replicas": replicas, "total_offered_rps": rps,
            "schedule_sha256": digest, "schedule_path": str(schedule_path.relative_to(PROJECT_ROOT)),
        }

    registry_path = I3VAL_CAL_ROOT / "iteration3_final_validation_schedule_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)

    print(f"Generated/verified {len(hashes)} canonical Iteration-3 final-validation schedules.")
    print(f"Registry written: {registry_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
