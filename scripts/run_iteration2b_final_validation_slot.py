#!/usr/bin/env python3
"""scripts/run_iteration2b_final_validation_slot.py

Phase 16K: Iteration-2B I2B_FINAL_UNSEEN_VALIDATION execution entry point.

This module does NOT duplicate any execution/collection/V8-structural-
validity/route-integrity/dataplane-acquisition/HPA-isolation/restoration
logic. It reuses `scripts.run_dynamics_calibration_slot.main()` -- the exact
same shared, historically-hardened pipeline the historical 102-slot
campaign, the completed Iteration-2 24-slot campaign, and the completed
Iteration-2B 3-slot remediation campaign all use -- supplying only a
separately-frozen Iteration-2B final-validation campaign context (sequence,
schedule registry, persistent root, ledger path) via that function's
code-level-only override parameters. It never touches the historical,
Iteration-2 development, or Iteration-2B remediation sequences/registries/
persistent roots/ledgers.

Like every other CLI in this family, this CLI accepts NO override of
replicas, RPS, schedule, or any other scientific parameter -- every
scientific parameter is resolved solely from the frozen Iteration-2B
final-validation sequence, itself a deterministic, mechanical, hash-verified
projection of the Phase16J.3 design freeze (activated r7 coordinates) and
the Phase16H validation-seed/shuffle algorithm (reused unchanged).

Every I2B_FINAL_UNSEEN_VALIDATION run permanently contributes
fitting_weight = 0 -- no future phase may fold a validation observation back
into model fitting after results are seen.
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
from scripts import run_iteration2_dynamics_calibration_slot as iteration2  # noqa: E402
from scripts import run_iteration2b_r7_remediation_slot as iteration2b_remediation  # noqa: E402
from scripts.dynamics_calibration_lib import GLOBAL_SEQUENCE_PATH as HISTORICAL_GLOBAL_SEQUENCE_PATH  # noqa: E402
from scripts.dynamics_calibration_lib import SCHEDULE_REGISTRY_PATH as HISTORICAL_SCHEDULE_REGISTRY_PATH  # noqa: E402

I2BVAL_CAL_ROOT = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration_iteration2b"
I2BVAL_SEQUENCE_PATH = I2BVAL_CAL_ROOT / "iteration2b_final_validation_sequence.json"
I2BVAL_SCHEDULE_REGISTRY_PATH = I2BVAL_CAL_ROOT / "iteration2b_final_validation_schedule_registry.json"
I2BVAL_PERSISTENT_ROOT = PROJECT_ROOT / "persistent-data/dynamics_validation_iteration2b/final_unseen"
I2BVAL_LEDGER_PATH = I2BVAL_PERSISTENT_ROOT / "iteration2b_final_validation_ledger.json"

# Frozen BEFORE any I2B_FINAL_UNSEEN_VALIDATION outcome exists (Phase16K).
EXPECTED_MATRIX_SHA256 = "e1b2cd02e602bda855760b02062f2b721297e1b5c945860a2517f13575569661"
EXPECTED_SEQUENCE_PROJECTION_SHA256 = "654213e5671ec89d0a592fd906ba78602497af633fd18585c57251417eb4db57"

# Hard write-firewall: this context must never coincide with the historical,
# Iteration-2 development, or Iteration-2B remediation contexts.
_OTHER_ROOTS = (historical.CALIBRATION_ROOT, iteration2.I2_PERSISTENT_ROOT, iteration2b_remediation.I2B_PERSISTENT_ROOT)
_OTHER_LEDGERS = (historical.LEDGER_PATH, iteration2.I2_LEDGER_PATH, iteration2b_remediation.I2B_LEDGER_PATH)
_OTHER_SEQUENCES = (HISTORICAL_GLOBAL_SEQUENCE_PATH, iteration2.I2_SEQUENCE_PATH, iteration2b_remediation.I2B_SEQUENCE_PATH)
_OTHER_REGISTRIES = (HISTORICAL_SCHEDULE_REGISTRY_PATH, iteration2.I2_SCHEDULE_REGISTRY_PATH, iteration2b_remediation.I2B_SCHEDULE_REGISTRY_PATH)

assert I2BVAL_PERSISTENT_ROOT not in _OTHER_ROOTS, "I2B final-validation root must not equal any prior campaign root"
assert I2BVAL_LEDGER_PATH not in _OTHER_LEDGERS, "I2B final-validation ledger must not equal any prior campaign ledger"
assert I2BVAL_SEQUENCE_PATH not in _OTHER_SEQUENCES, "I2B final-validation sequence must not equal any prior campaign sequence"
assert I2BVAL_SCHEDULE_REGISTRY_PATH not in _OTHER_REGISTRIES, "I2B final-validation registry must not equal any prior campaign registry"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute exactly one frozen Iteration-2B I2B_FINAL_UNSEEN_VALIDATION campaign slot "
            "(Phase16K design freeze). All scientific parameters are resolved from the frozen "
            "Iteration-2B final-validation sequence -- this CLI accepts no override of replicas, "
            "RPS, schedule, or any other scientific parameter."
        )
    )
    parser.add_argument("--validation-sequence-index", required=True, type=int, help="1..32, resolved against the frozen Iteration-2B final-validation sequence")
    parser.add_argument("--attempt-number", default=1, type=int, help="Execution attempt number for this planned slot (default 1)")
    parser.add_argument("--session-id", default=None, help="Optional session identifier; auto-generated if omitted")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan and exit; no traffic, no Kubernetes mutation")
    return parser


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_i2bval_sequence() -> List[dict]:
    """Fail-closed hash verification before returning the Iteration-2B
    final-validation sequence: both the frozen Phase16J.3 matrix and this
    module's own executable JSON projection must match their recorded
    hashes exactly."""
    matrix_path = PROJECT_ROOT / "task_outputs" / "phase_16j3_iteration2b_final_validation_matrix.csv"
    task_outputs_alt = Path("/Users/rakeshdandamudi/Capstone/task_outputs/phase_16j3_iteration2b_final_validation_matrix.csv")
    source_path = matrix_path if matrix_path.exists() else task_outputs_alt
    if not source_path.exists():
        raise RuntimeError("STOP (fail-closed): phase_16j3_iteration2b_final_validation_matrix.csv not found -- cannot verify design provenance")
    actual_matrix_hash = _sha256_of(source_path)
    if actual_matrix_hash != EXPECTED_MATRIX_SHA256:
        raise RuntimeError(
            f"STOP (fail-closed): Phase16J.3 final-validation matrix hash mismatch: "
            f"{actual_matrix_hash} != {EXPECTED_MATRIX_SHA256}"
        )

    if not I2BVAL_SEQUENCE_PATH.exists():
        raise RuntimeError(f"STOP (fail-closed): Iteration-2B final-validation sequence projection not found: {I2BVAL_SEQUENCE_PATH}")
    actual_projection_hash = _sha256_of(I2BVAL_SEQUENCE_PATH)
    if actual_projection_hash != EXPECTED_SEQUENCE_PROJECTION_SHA256:
        raise RuntimeError(
            f"STOP (fail-closed): Iteration-2B final-validation sequence projection hash mismatch: "
            f"{actual_projection_hash} != {EXPECTED_SEQUENCE_PROJECTION_SHA256}"
        )

    with open(I2BVAL_SEQUENCE_PATH, "r", encoding="utf-8") as f:
        sequence = json.load(f)
    if len(sequence) != 32:
        raise RuntimeError(f"STOP (fail-closed): expected exactly 32 Iteration-2B final-validation slots, found {len(sequence)}")
    return sequence


def load_i2bval_schedule_registry() -> dict:
    if not I2BVAL_SCHEDULE_REGISTRY_PATH.exists():
        raise RuntimeError(f"STOP (fail-closed): Iteration-2B final-validation schedule registry not found: {I2BVAL_SCHEDULE_REGISTRY_PATH}")
    with open(I2BVAL_SCHEDULE_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not (1 <= args.validation_sequence_index <= 32):
        raise SystemExit(
            f"STOP (fail-closed): --validation-sequence-index={args.validation_sequence_index} "
            f"outside the frozen Iteration-2B final-validation range [1, 32]"
        )

    i2bval_sequence = load_verified_i2bval_sequence()
    i2bval_registry = load_i2bval_schedule_registry()

    if I2BVAL_PERSISTENT_ROOT in _OTHER_ROOTS:
        raise RuntimeError("STOP (fail-closed): Iteration-2B final-validation root resolves to a prior campaign root")
    if I2BVAL_LEDGER_PATH in _OTHER_LEDGERS:
        raise RuntimeError("STOP (fail-closed): Iteration-2B final-validation ledger path resolves to a prior campaign ledger")

    hist_argv = ["--campaign-sequence-index", str(args.validation_sequence_index), "--attempt-number", str(args.attempt_number)]
    if args.session_id:
        hist_argv += ["--session-id", args.session_id]
    if args.dry_run:
        hist_argv += ["--dry-run"]

    return historical.main(
        hist_argv,
        sequence_override=i2bval_sequence,
        schedule_registry_override=i2bval_registry,
        calibration_root=I2BVAL_PERSISTENT_ROOT,
        ledger_path=I2BVAL_LEDGER_PATH,
    )


if __name__ == "__main__":
    raise SystemExit(main())
