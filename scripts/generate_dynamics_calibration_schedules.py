#!/usr/bin/env python3
"""scripts/generate_dynamics_calibration_schedules.py

Phase 15B.2A (CORR-32): generates and hashes, OFFLINE, every canonical
traffic schedule the frozen empirical dynamics calibration protocol
(experiments/manifests/rl_dynamics_calibration_v1.yaml) requires. This
script never opens an HTTP client and never sends a single request to
demo-app -- it only calls scripts/generate_tuning_schedule.py's existing,
tested, pure `generate_and_persist()` path (the same one used to freeze the
HPA-tuning canonical schedule, CORR-24), once per calibration cell.

Deterministic arrivals are used throughout (arrival_process="deterministic"),
per the protocol's frozen arrival-process decision -- the same schedule is
replayed for every repeat of a given cell, and (for replica-transition
pairs) for both directions of a pair, since the offered workload during a
replica-count transition is defined independent of transition direction.

Running this script is idempotent-safe: scripts.generate_tuning_schedule's
generate_and_persist() refuses to overwrite an existing schedule file, so a
second run against the same --out-dir with all files already present will
raise FileExistsError per-file (this script itself checks and skips
existing files instead of calling generate_and_persist redundantly, since
canonical schedules must never be silently regenerated once frozen).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAL_ROOT = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration"
STEADY_DIR = CAL_ROOT / "steady_state"
REPLICA_TRANS_DIR = CAL_ROOT / "replica_transitions"
WORKLOAD_TRANS_DIR = CAL_ROOT / "workload_transitions"

import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.generate_tuning_schedule import generate_and_persist  # noqa: E402

INTENSITY = 3
ENDPOINT = "/work"
SEED = 42  # provenance only -- deterministic arrivals never consult it

# ---------------------------------------------------------------------------
# Frozen steady-state cells (CORR-32 corrected matrix -- see
# experiments/manifests/rl_dynamics_calibration_v1.yaml for the authoritative
# copy of this exact table; this script and that manifest must never diverge).
# Each entry: (cell_id, replicas, total_offered_rps, roles)
# ---------------------------------------------------------------------------
STEADY_STATE_CELLS: List[Tuple[str, int, float, List[str]]] = [
    # replicas=1
    ("dyncal_r1_L1", 1, 2.0, ["idle_per_replica_anchor"]),
    ("dyncal_r1_L2", 1, 5.0, ["sustainable_per_replica_anchor"]),
    ("dyncal_r1_L3", 1, 8.0, ["near_capacity_per_replica_anchor"]),
    ("dyncal_r1_L4", 1, 12.0, ["overload_per_replica_anchor"]),
    ("dyncal_r1_L5", 1, 40.0, ["fixed_total_contention_anchor"]),
    # replicas=2
    ("dyncal_r2_L1", 2, 4.0, ["idle_per_replica_anchor"]),
    ("dyncal_r2_L2", 2, 10.0, ["sustainable_per_replica_anchor"]),
    ("dyncal_r2_L3", 2, 16.0, ["near_capacity_per_replica_anchor"]),
    ("dyncal_r2_L4", 2, 24.0, ["overload_per_replica_anchor"]),
    # replicas=4
    ("dyncal_r4_L1", 4, 8.0, ["idle_per_replica_anchor"]),
    ("dyncal_r4_L2", 4, 20.0, ["sustainable_per_replica_anchor"]),
    ("dyncal_r4_L3", 4, 32.0, ["near_capacity_per_replica_anchor"]),
    ("dyncal_r4_L4", 4, 48.0, ["overload_per_replica_anchor"]),
    ("dyncal_r4_L5", 4, 40.0, ["fixed_total_contention_anchor"]),
    # replicas=6
    ("dyncal_r6_L1", 6, 12.0, ["idle_per_replica_anchor"]),
    ("dyncal_r6_L2", 6, 30.0, ["sustainable_per_replica_anchor"]),
    ("dyncal_r6_L3", 6, 48.0, ["near_capacity_per_replica_anchor"]),
    # replicas=8 -- L2 and L5 coincide at exactly 40 RPS total (CORR-32 dedup)
    ("dyncal_r8_L1", 8, 16.0, ["idle_per_replica_anchor"]),
    ("dyncal_r8_L2_L5", 8, 40.0, ["sustainable_per_replica_anchor", "fixed_total_contention_anchor"]),
    ("dyncal_r8_L3", 8, 64.0, ["near_capacity_per_replica_anchor"]),
]

# Interpolation-validation cells (replicas 3/5/7, levels L1-L3 only)
VALIDATION_CELLS: List[Tuple[str, int, float, List[str]]] = [
    ("dyncal_r3_L1", 3, 6.0, ["interpolation_validation"]),
    ("dyncal_r3_L2", 3, 15.0, ["interpolation_validation"]),
    ("dyncal_r3_L3", 3, 24.0, ["interpolation_validation"]),
    ("dyncal_r5_L1", 5, 10.0, ["interpolation_validation"]),
    ("dyncal_r5_L2", 5, 25.0, ["interpolation_validation"]),
    ("dyncal_r5_L3", 5, 40.0, ["interpolation_validation"]),
    ("dyncal_r7_L1", 7, 14.0, ["interpolation_validation"]),
    ("dyncal_r7_L2", 7, 35.0, ["interpolation_validation"]),
    ("dyncal_r7_L3", 7, 56.0, ["interpolation_validation"]),
]

STEADY_STATE_WARMUP_SECONDS = 60.0
STEADY_STATE_SCORED_SECONDS = 120.0
STEADY_STATE_TRAFFIC_DURATION = STEADY_STATE_WARMUP_SECONDS + STEADY_STATE_SCORED_SECONDS  # 180s

# ---------------------------------------------------------------------------
# Frozen replica-count transition pairs (adjacent +-1 only, CORR-32 sec.10).
# Workload = 5.0 RPS x lower_replica_count of the pair, held CONSTANT across
# the transition (only replica count changes) -- so ONE schedule per pair
# serves BOTH directions.
# ---------------------------------------------------------------------------
REPLICA_TRANSITION_PAIRS: List[Tuple[str, int, int]] = [
    ("dyncal_transrep_1-2", 1, 2),
    ("dyncal_transrep_3-4", 3, 4),
    ("dyncal_transrep_5-6", 5, 6),
    ("dyncal_transrep_7-8", 7, 8),
]
REPLICA_TRANSITION_PRE_TRAFFIC_SECONDS = 60.0
REPLICA_TRANSITION_POST_TRIGGER_SECONDS = 180.0
REPLICA_TRANSITION_TRAFFIC_DURATION = (
    REPLICA_TRANSITION_PRE_TRAFFIC_SECONDS + REPLICA_TRANSITION_POST_TRIGGER_SECONDS
)  # 240s

# ---------------------------------------------------------------------------
# Frozen workload-step transitions (fixed replica count, RPS step change).
# ---------------------------------------------------------------------------
WORKLOAD_TRANSITIONS: List[Tuple[str, int, float, float]] = [
    ("dyncal_transwork_r1_2to8", 1, 2.0, 8.0),
    ("dyncal_transwork_r1_8to2", 1, 8.0, 2.0),
    ("dyncal_transwork_r4_8to32", 4, 8.0, 32.0),
    ("dyncal_transwork_r4_32to8", 4, 32.0, 8.0),
]
WORKLOAD_TRANSITION_PRE_STEP_SECONDS = 60.0
WORKLOAD_TRANSITION_POST_STEP_SECONDS = 180.0


def _write_profile(path: Path, profile_name: str, phases: List[dict]) -> None:
    doc = {
        "profile": profile_name,
        "description": f"Phase 15B.2A (CORR-32) empirical dynamics calibration schedule: {profile_name}",
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
        # Already frozen -- never silently regenerate. Return its existing hash.
        digest = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
        return digest
    return generate_and_persist(profile_path, "deterministic", SEED, schedule_path)


def main() -> None:
    hashes: Dict[str, dict] = {}

    for cell_id, replicas, total_rps, roles in STEADY_STATE_CELLS + VALIDATION_CELLS:
        profile_path = STEADY_DIR / f"{cell_id}.yaml"
        schedule_path = STEADY_DIR / f"{cell_id}_schedule.csv"
        _write_profile(
            profile_path, cell_id,
            [{"type": "constant", "name": "steady_state", "duration_seconds": STEADY_STATE_TRAFFIC_DURATION, "target_rps": total_rps}],
        )
        digest = _generate_one(profile_path, schedule_path)
        hashes[cell_id] = {
            "kind": "steady_state", "replicas": replicas, "total_offered_rps": total_rps,
            "roles": roles, "schedule_sha256": digest,
            "schedule_path": str(schedule_path.relative_to(PROJECT_ROOT)),
        }

    for cell_id, replicas_low, replicas_high in REPLICA_TRANSITION_PAIRS:
        rps = 5.0 * replicas_low
        profile_path = REPLICA_TRANS_DIR / f"{cell_id}.yaml"
        schedule_path = REPLICA_TRANS_DIR / f"{cell_id}_schedule.csv"
        _write_profile(
            profile_path, cell_id,
            [{"type": "constant", "name": "replica_transition", "duration_seconds": REPLICA_TRANSITION_TRAFFIC_DURATION, "target_rps": rps}],
        )
        digest = _generate_one(profile_path, schedule_path)
        hashes[cell_id] = {
            "kind": "replica_transition", "replicas_low": replicas_low, "replicas_high": replicas_high,
            "total_offered_rps": rps, "schedule_sha256": digest,
            "schedule_path": str(schedule_path.relative_to(PROJECT_ROOT)),
            "trigger_at_traffic_elapsed_seconds": REPLICA_TRANSITION_PRE_TRAFFIC_SECONDS,
        }

    for cell_id, replicas, rps_before, rps_after in WORKLOAD_TRANSITIONS:
        profile_path = WORKLOAD_TRANS_DIR / f"{cell_id}.yaml"
        schedule_path = WORKLOAD_TRANS_DIR / f"{cell_id}_schedule.csv"
        _write_profile(
            profile_path, cell_id,
            [
                {"type": "constant", "name": "pre_step", "duration_seconds": WORKLOAD_TRANSITION_PRE_STEP_SECONDS, "target_rps": rps_before},
                {"type": "constant", "name": "post_step", "duration_seconds": WORKLOAD_TRANSITION_POST_STEP_SECONDS, "target_rps": rps_after},
            ],
        )
        digest = _generate_one(profile_path, schedule_path)
        hashes[cell_id] = {
            "kind": "workload_transition", "replicas": replicas,
            "rps_before": rps_before, "rps_after": rps_after, "schedule_sha256": digest,
            "schedule_path": str(schedule_path.relative_to(PROJECT_ROOT)),
            "trigger_at_traffic_elapsed_seconds": WORKLOAD_TRANSITION_PRE_STEP_SECONDS,
        }

    registry_path = CAL_ROOT / "schedule_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)

    print(f"Generated/verified {len(hashes)} canonical calibration schedules.")
    print(f"Registry written: {registry_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
