"""scripts/generate_global_campaign_sequence.py

Phase 15B.2C (CORR-34): freezes the ONE authoritative, global, 102-entry
execution order for the empirical dynamics-calibration campaign, so the
future execution harness never has to decide which scientific run comes
next.

This is purely a DERIVATION script over already-frozen data -- it does not
introduce any new randomization. It reads:
  - experiments/manifests/rl_dynamics_calibration_v1.yaml's `randomization`
    block (the already-frozen within-block shuffled orders from CORR-32),
  - the cell/transition definitions (fitting_cells, validation_cells,
    replica_transitions, workload_transitions),
  - configs/traffic_profiles/rl_dynamics_calibration/schedule_registry.json
    (already-frozen, SHA-256-hashed canonical schedules),
and deterministically concatenates the within-block orders into one global
sequence using this precommitted block order (CORR-34 sec.3):

    1. fitting repeat 1        (20)
    2. interpolation-validation repeat 1  (9)
    3. replica-transition repeat 1        (8)
    4. workload-transition repeat 1       (4)
    5. fitting repeat 2        (20)
    6. interpolation-validation repeat 2  (9)
    7. replica-transition repeat 2        (8)
    8. workload-transition repeat 2       (4)
    9. fitting repeat 3        (20)
                                          ---
                                          102

No traffic is sent. No cell, repeat count, or schedule is modified. Running
this script is idempotent -- re-running it reproduces byte-identical output,
since every input it reads is already frozen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "experiments/manifests/rl_dynamics_calibration_v1.yaml"
SCHEDULE_REGISTRY_PATH = (
    PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration/schedule_registry.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration/global_campaign_sequence.json"
)


def _load_manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_registry() -> dict:
    with open(SCHEDULE_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _fitting_cell_lookup(manifest: dict) -> dict:
    """(replicas, load_level_short) -> fitting cell dict.

    The randomization.fitting_order entries use short load_level tags
    (L1..L5, or "L2_L5" for the deduplicated cell) that must be resolved
    against the cell_id's own suffix.
    """
    lookup = {}
    for cell in manifest["fitting_cells"]:
        cell_id = cell["cell_id"]  # e.g. dyncal_r2_L4, dyncal_r8_L2_L5
        suffix = cell_id.split("_", 2)[-1]  # "L4" or "L2_L5"
        lookup[(cell["replicas"], suffix)] = cell
    return lookup


def _validation_cell_lookup(manifest: dict) -> dict:
    lookup = {}
    for cell in manifest["validation_cells"]:
        cell_id = cell["cell_id"]
        suffix = cell_id.split("_", 2)[-1]
        lookup[(cell["replicas"], suffix)] = cell
    return lookup


def _build_fitting_entries(manifest: dict, registry: dict, repeat_number: int) -> list[dict]:
    lookup = _fitting_cell_lookup(manifest)
    order = manifest["randomization"]["fitting_order"][f"repeat_{repeat_number}"]
    entries = []
    for item in order:
        cell = lookup[(item["replicas"], item["load_level"])]
        cell_id = cell["cell_id"]
        reg = registry[cell_id]
        entries.append(
            {
                "planned_slot_id": f"fit_{cell_id}_repeat{repeat_number}",
                "run_kind": "fitting",
                "cell_or_transition_id": cell_id,
                "repeat_number": repeat_number,
                "schedule_id": cell_id,
                "schedule_path": reg["schedule_path"],
                "schedule_sha256": reg["schedule_sha256"],
                "replicas": cell["replicas"],
                "offered_rps": cell["total_offered_rps"],
            }
        )
    return entries


def _build_validation_entries(manifest: dict, registry: dict, repeat_number: int) -> list[dict]:
    lookup = _validation_cell_lookup(manifest)
    order = manifest["randomization"]["validation_order"][f"repeat_{repeat_number}"]
    entries = []
    for item in order:
        cell = lookup[(item["replicas"], item["load_level"])]
        cell_id = cell["cell_id"]
        reg = registry[cell_id]
        entries.append(
            {
                "planned_slot_id": f"val_{cell_id}_repeat{repeat_number}",
                "run_kind": "interpolation_validation",
                "cell_or_transition_id": cell_id,
                "repeat_number": repeat_number,
                "schedule_id": cell_id,
                "schedule_path": reg["schedule_path"],
                "schedule_sha256": reg["schedule_sha256"],
                "replicas": cell["replicas"],
                "offered_rps": cell["total_offered_rps"],
            }
        )
    return entries


def _build_replica_transition_entries(manifest: dict, registry: dict, repeat_number: int) -> list[dict]:
    rt = manifest["replica_transitions"]
    workload_by_pair = rt["workload_by_pair"]
    order = manifest["randomization"]["replica_transition_order"][f"repeat_{repeat_number}"]
    entries = []
    for pair in order:
        before, after = pair[0], pair[1]
        pair_key = ",".join(str(x) for x in sorted((before, after)))
        transition_id = f"dyncal_transrep_{min(before, after)}-{max(before, after)}"
        direction = f"{before}to{after}"
        reg = registry[transition_id]
        entries.append(
            {
                "planned_slot_id": f"transrep_{min(before, after)}-{max(before, after)}_{direction}_repeat{repeat_number}",
                "run_kind": "replica_transition",
                "cell_or_transition_id": transition_id,
                "repeat_number": repeat_number,
                "schedule_id": transition_id,
                "schedule_path": reg["schedule_path"],
                "schedule_sha256": reg["schedule_sha256"],
                "replicas_before": before,
                "replicas_after": after,
                "offered_rps": workload_by_pair[pair_key],
            }
        )
    return entries


def _build_workload_transition_entries(manifest: dict, registry: dict, repeat_number: int) -> list[dict]:
    wt = manifest["workload_transitions"]
    types_by_id = {t["cell_id"]: t for t in wt["types"]}
    order = manifest["randomization"]["workload_transition_order"][f"repeat_{repeat_number}"]
    entries = []
    for item in order:
        replicas = item["replicas"]
        transition = item["transition"]  # e.g. "8to32"
        transition_id = next(
            t["cell_id"]
            for t in wt["types"]
            if t["replicas"] == replicas and t["cell_id"].endswith(transition)
        )
        t = types_by_id[transition_id]
        reg = registry[transition_id]
        entries.append(
            {
                "planned_slot_id": f"transwork_{transition_id.replace('dyncal_transwork_', '')}_repeat{repeat_number}",
                "run_kind": "workload_transition",
                "cell_or_transition_id": transition_id,
                "repeat_number": repeat_number,
                "schedule_id": transition_id,
                "schedule_path": reg["schedule_path"],
                "schedule_sha256": reg["schedule_sha256"],
                "replicas": t["replicas"],
                "rps_before": t["rps_before"],
                "rps_after": t["rps_after"],
            }
        )
    return entries


def build_global_sequence() -> list[dict]:
    manifest = _load_manifest()
    registry = _load_registry()

    blocks = [
        _build_fitting_entries(manifest, registry, 1),
        _build_validation_entries(manifest, registry, 1),
        _build_replica_transition_entries(manifest, registry, 1),
        _build_workload_transition_entries(manifest, registry, 1),
        _build_fitting_entries(manifest, registry, 2),
        _build_validation_entries(manifest, registry, 2),
        _build_replica_transition_entries(manifest, registry, 2),
        _build_workload_transition_entries(manifest, registry, 2),
        _build_fitting_entries(manifest, registry, 3),
    ]

    sequence = []
    index = 1
    for block in blocks:
        for entry in block:
            entry_with_index = {"campaign_sequence_index": index, **entry}
            sequence.append(entry_with_index)
            index += 1
    return sequence


def generate_and_persist() -> tuple[Path, str]:
    sequence = build_global_sequence()
    assert len(sequence) == 102, f"expected 102 entries, got {len(sequence)}"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sequence, f, indent=2)
        f.write("\n")
    digest = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    return OUTPUT_PATH, digest


if __name__ == "__main__":
    path, digest = generate_and_persist()
    print(f"Generated global campaign sequence: {path}")
    print(f"SHA-256: {digest}")
