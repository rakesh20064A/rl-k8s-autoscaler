#!/usr/bin/env python3
"""scripts/run_iteration2b_r7_remediation_slot.py

Phase 16J.1: Iteration-2B r7-remediation execution entry point.

This module does NOT duplicate any execution/collection/V8-structural-
validity/route-integrity/dataplane-acquisition/HPA-isolation/restoration
logic. It reuses `scripts.run_dynamics_calibration_slot.main()` -- the exact
same shared, historically-hardened pipeline the official 102-slot campaign
and the completed 24-slot Iteration-2 campaign both use -- supplying only a
separately-frozen Iteration-2B remediation campaign context (sequence,
schedule registry, persistent root, ledger path) via that function's
code-level-only override parameters. It never touches the historical
`global_campaign_sequence.json`, the historical `schedule_registry.json`,
the historical `persistent-data/dynamics_calibration/` tree, the historical
campaign ledger, the completed Iteration-2 sequence/registry, the completed
Iteration-2 `persistent-data/dynamics_calibration_iteration2/development/`
tree, or the completed Iteration-2 ledger.

Like the historical and Iteration-2 CLIs, this CLI accepts NO override of
replicas, RPS, schedule, or any other scientific parameter -- every
scientific parameter is resolved solely from the frozen Iteration-2B
remediation sequence
(configs/traffic_profiles/rl_dynamics_calibration_iteration2b/iteration2b_r7_remediation_sequence.json),
itself a deterministic, mechanical, hash-verified projection of the
Phase16J.1 design freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Optional

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_dynamics_calibration_slot as historical  # noqa: E402 -- reuse, never duplicate
from scripts import run_iteration2_dynamics_calibration_slot as iteration2  # noqa: E402 -- for the I2 firewall check
from scripts.dynamics_calibration_lib import GLOBAL_SEQUENCE_PATH as HISTORICAL_GLOBAL_SEQUENCE_PATH  # noqa: E402
from scripts.dynamics_calibration_lib import SCHEDULE_REGISTRY_PATH as HISTORICAL_SCHEDULE_REGISTRY_PATH  # noqa: E402

I2B_CAL_ROOT = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration_iteration2b"
I2B_SEQUENCE_PATH = I2B_CAL_ROOT / "iteration2b_r7_remediation_sequence.json"
I2B_SCHEDULE_REGISTRY_PATH = I2B_CAL_ROOT / "iteration2b_r7_remediation_schedule_registry.json"
I2B_PERSISTENT_ROOT = PROJECT_ROOT / "persistent-data/dynamics_calibration_iteration2b/r7_remediation"
I2B_LEDGER_PATH = I2B_PERSISTENT_ROOT / "iteration2b_r7_remediation_ledger.json"

# Frozen BEFORE any Iteration-2B outcome exists (Phase16J.1). Any drift in
# either file must fail closed, never silently proceed.
EXPECTED_PHASE16J1_DESIGN_YAML_SHA256 = "0f74b3d69f43eb6f11b08f31150dae48144d2ba989dc66df8d0c38d2c674aaa4"
EXPECTED_I2B_SEQUENCE_PROJECTION_SHA256 = "77523671325d790f65d1a23f0e5b112df471f9c752e0504b77090f4c9c3a6a13"

# Hard write-firewall: an Iteration-2B context must never coincide with the
# historical campaign context NOR with the completed Iteration-2 campaign
# context. Checked once at import time (fails the whole process immediately,
# before any CLI parsing) and again defensively in main().
assert I2B_PERSISTENT_ROOT != historical.CALIBRATION_ROOT, "Iteration-2B root must not equal the historical calibration root"
assert I2B_PERSISTENT_ROOT != iteration2.I2_PERSISTENT_ROOT, "Iteration-2B root must not equal the Iteration-2 persistent root"
assert I2B_LEDGER_PATH != historical.LEDGER_PATH, "Iteration-2B ledger must not equal the historical ledger"
assert I2B_LEDGER_PATH != iteration2.I2_LEDGER_PATH, "Iteration-2B ledger must not equal the Iteration-2 ledger"
assert I2B_SEQUENCE_PATH != HISTORICAL_GLOBAL_SEQUENCE_PATH, "Iteration-2B sequence must not equal the historical global sequence"
assert I2B_SEQUENCE_PATH != iteration2.I2_SEQUENCE_PATH, "Iteration-2B sequence must not equal the Iteration-2 sequence"
assert I2B_SCHEDULE_REGISTRY_PATH != HISTORICAL_SCHEDULE_REGISTRY_PATH, "Iteration-2B schedule registry must not equal the historical schedule registry"
assert I2B_SCHEDULE_REGISTRY_PATH != iteration2.I2_SCHEDULE_REGISTRY_PATH, "Iteration-2B schedule registry must not equal the Iteration-2 schedule registry"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute exactly one frozen Iteration-2B r7-remediation campaign slot "
            "(Phase16J.1 design freeze). All scientific parameters are resolved from "
            "the frozen Iteration-2B remediation sequence -- this CLI accepts no "
            "override of replicas, RPS, schedule, or any other scientific parameter."
        )
    )
    parser.add_argument("--remediation-sequence-index", required=True, type=int, help="1..3, resolved against the frozen Iteration-2B remediation sequence")
    parser.add_argument("--attempt-number", default=1, type=int, help="Execution attempt number for this planned slot (default 1)")
    parser.add_argument("--session-id", default=None, help="Optional session identifier; auto-generated if omitted")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan and exit; no traffic, no Kubernetes mutation")
    return parser


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_i2b_sequence() -> List[dict]:
    """Fail-closed hash verification before returning the Iteration-2B
    sequence: both the frozen Phase16J.1 design YAML and this module's own
    executable JSON projection must match their recorded hashes exactly."""
    design_yaml = PROJECT_ROOT / "task_outputs" / "phase_16j1_iteration2b_r7_remediation_design.yaml"
    task_outputs_alt = Path("/Users/rakeshdandamudi/Capstone/task_outputs/phase_16j1_iteration2b_r7_remediation_design.yaml")
    source_path = design_yaml if design_yaml.exists() else task_outputs_alt
    if not source_path.exists():
        raise RuntimeError("STOP (fail-closed): phase_16j1_iteration2b_r7_remediation_design.yaml not found -- cannot verify design provenance")
    actual_source_hash = _sha256_of(source_path)
    if actual_source_hash != EXPECTED_PHASE16J1_DESIGN_YAML_SHA256:
        raise RuntimeError(
            f"STOP (fail-closed): Phase16J.1 remediation design hash mismatch: "
            f"{actual_source_hash} != {EXPECTED_PHASE16J1_DESIGN_YAML_SHA256}"
        )

    if not I2B_SEQUENCE_PATH.exists():
        raise RuntimeError(f"STOP (fail-closed): Iteration-2B sequence projection not found: {I2B_SEQUENCE_PATH}")
    actual_projection_hash = _sha256_of(I2B_SEQUENCE_PATH)
    if actual_projection_hash != EXPECTED_I2B_SEQUENCE_PROJECTION_SHA256:
        raise RuntimeError(
            f"STOP (fail-closed): Iteration-2B sequence projection hash mismatch: "
            f"{actual_projection_hash} != {EXPECTED_I2B_SEQUENCE_PROJECTION_SHA256}"
        )

    with open(I2B_SEQUENCE_PATH, "r", encoding="utf-8") as f:
        sequence = json.load(f)
    if len(sequence) != 3:
        raise RuntimeError(f"STOP (fail-closed): expected exactly 3 Iteration-2B remediation slots, found {len(sequence)}")
    # historical.resolve_planned_slot() keys off "campaign_sequence_index" --
    # the I2B sequence rows use "remediation_sequence_index" for clarity, so
    # project it onto the field the shared resolver expects.
    for entry in sequence:
        entry["campaign_sequence_index"] = entry["remediation_sequence_index"]
    return sequence


def load_i2b_schedule_registry() -> dict:
    if not I2B_SCHEDULE_REGISTRY_PATH.exists():
        raise RuntimeError(f"STOP (fail-closed): Iteration-2B schedule registry not found: {I2B_SCHEDULE_REGISTRY_PATH}")
    with open(I2B_SCHEDULE_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not (1 <= args.remediation_sequence_index <= 3):
        raise SystemExit(
            f"STOP (fail-closed): --remediation-sequence-index={args.remediation_sequence_index} "
            f"outside the frozen Iteration-2B remediation range [1, 3]"
        )

    i2b_sequence = load_verified_i2b_sequence()
    i2b_registry = load_i2b_schedule_registry()

    # Defensive, runtime re-check of the write firewall (import-time asserts
    # above already guarantee this structurally; repeated here so a future
    # edit to either constant cannot silently regress it).
    if I2B_PERSISTENT_ROOT in (historical.CALIBRATION_ROOT, iteration2.I2_PERSISTENT_ROOT):
        raise RuntimeError("STOP (fail-closed): Iteration-2B persistent root resolves to the historical or Iteration-2 root")
    if I2B_LEDGER_PATH in (historical.LEDGER_PATH, iteration2.I2_LEDGER_PATH):
        raise RuntimeError("STOP (fail-closed): Iteration-2B ledger path resolves to the historical or Iteration-2 ledger path")

    hist_argv = ["--campaign-sequence-index", str(args.remediation_sequence_index), "--attempt-number", str(args.attempt_number)]
    if args.session_id:
        hist_argv += ["--session-id", args.session_id]
    if args.dry_run:
        hist_argv += ["--dry-run"]

    return historical.main(
        hist_argv,
        sequence_override=i2b_sequence,
        schedule_registry_override=i2b_registry,
        calibration_root=I2B_PERSISTENT_ROOT,
        ledger_path=I2B_LEDGER_PATH,
    )


if __name__ == "__main__":
    raise SystemExit(main())
