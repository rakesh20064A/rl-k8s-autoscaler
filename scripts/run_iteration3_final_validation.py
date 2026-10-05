#!/usr/bin/env python3
"""scripts/run_iteration3_final_validation.py

Phase 16P: Iteration-3 ITERATION3_FINAL_UNSEEN_VALIDATION execution entry
point (pre-live closure only -- this module is written and verified in
Phase16P, but Phase16P itself never invokes its live-execution path).

This module does NOT duplicate any execution/collection/V8-structural-
validity/route-integrity/dataplane-acquisition/HPA-isolation/restoration
logic, does NOT implement a new traffic generator, does NOT implement a new
route classifier, does NOT change Axis-A/Axis-B eligibility, does NOT
contain model-fitting logic, and does NOT score predictions or read
final-validation outcome metrics for scientific interpretation. It reuses
`scripts.run_dynamics_calibration_slot.main()` -- the exact same shared,
historically-hardened (CORR-51 through CORR-54) pipeline the historical
102-slot campaign, the Iteration-2 24-slot campaign, the Iteration-2B
3-slot remediation campaign, and the completed Iteration-2B 32-slot final-
validation campaign all use -- supplying only a separately-frozen
Iteration-3 final-validation campaign context (sequence, schedule registry,
persistent root, ledger path) via that function's code-level-only override
parameters. It never touches the historical, Iteration-2 development,
Iteration-2B remediation, or Iteration-2B final-validation sequences/
registries/persistent roots/ledgers.

Like every other CLI in this family, this CLI accepts NO override of
replicas, RPS, schedule, or any other scientific parameter -- every
scientific parameter is resolved solely from the frozen Iteration-3 final-
validation sequence (Phase16N Section 16, Phase16N.1 Section 7's strict-
support reconfirmation).

Every ITERATION3_FINAL_UNSEEN_VALIDATION run permanently contributes
fitting_weight = 0 -- no future phase may fold a validation observation
back into model fitting after results are seen. There are exactly 16
cells, 2 independent scientific repeats per cell (never a third), and 32
sequence slots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_dynamics_calibration_slot as historical  # noqa: E402 -- reuse, never duplicate
from scripts import run_iteration2_dynamics_calibration_slot as iteration2  # noqa: E402
from scripts import run_iteration2b_r7_remediation_slot as iteration2b_remediation  # noqa: E402
from scripts import run_iteration2b_final_validation_slot as iteration2b_final_validation  # noqa: E402
from scripts.dynamics_calibration_lib import GLOBAL_SEQUENCE_PATH as HISTORICAL_GLOBAL_SEQUENCE_PATH  # noqa: E402
from scripts.dynamics_calibration_lib import SCHEDULE_REGISTRY_PATH as HISTORICAL_SCHEDULE_REGISTRY_PATH  # noqa: E402

I3VAL_CAL_ROOT = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration_iteration3"
I3VAL_SEQUENCE_PROJECTION_PATH = I3VAL_CAL_ROOT / "iteration3_final_validation_sequence_projection.json"
I3VAL_SCHEDULE_REGISTRY_PATH = I3VAL_CAL_ROOT / "iteration3_final_validation_schedule_registry.json"
I3VAL_PERSISTENT_ROOT = PROJECT_ROOT / "persistent-data/dynamics_validation_iteration3/final_unseen"
I3VAL_LEDGER_PATH = I3VAL_PERSISTENT_ROOT / "iteration3_final_validation_ledger.json"

# Frozen BEFORE any ITERATION3_FINAL_UNSEEN_VALIDATION outcome exists (Phase16N/16N.1/16P).
EXPECTED_FRESH_MATRIX_SHA256 = "dfdc9f8698c131c7096c3e779dc20077fa1c10f9cab21561f646bba8d9bb637b"
EXPECTED_SEQUENCE_CSV_SHA256 = "d97b2b3be8e182e2c299d4abaec91e54fdbbc9639c1ab070f2def3084a01d7de"
EXPECTED_SEQUENCE_PROJECTION_SHA256 = "e3fc17a166049c28eb8dcc252e29062c7052a2dda91f73118cc7c27046417743"
EXPECTED_SCHEDULE_REGISTRY_SHA256 = "e3a952e9be006702f01b8a698987224fa204ec5c1ef135ebb8bc5d3222a70c4b"
EXPECTED_PROTOCOL_SHA256 = "6c140ec705afb797a4683689f126ed9de9064067b91c026142c3aa0f5db8c794"
EXPECTED_PREDICTION_PREREGISTRATION_SHA256 = "525bd55ad0d9c609329607a7fb8c8522a20a2a04bb91a0ccc775814952d57d8f"

# Hard write-firewall: this context must never coincide with any prior campaign context.
_OTHER_ROOTS = (historical.CALIBRATION_ROOT, iteration2.I2_PERSISTENT_ROOT,
                iteration2b_remediation.I2B_PERSISTENT_ROOT, iteration2b_final_validation.I2BVAL_PERSISTENT_ROOT)
_OTHER_LEDGERS = (historical.LEDGER_PATH, iteration2.I2_LEDGER_PATH,
                  iteration2b_remediation.I2B_LEDGER_PATH, iteration2b_final_validation.I2BVAL_LEDGER_PATH)
_OTHER_SEQUENCES = (HISTORICAL_GLOBAL_SEQUENCE_PATH, iteration2.I2_SEQUENCE_PATH,
                    iteration2b_remediation.I2B_SEQUENCE_PATH, iteration2b_final_validation.I2BVAL_SEQUENCE_PATH)
_OTHER_REGISTRIES = (HISTORICAL_SCHEDULE_REGISTRY_PATH, iteration2.I2_SCHEDULE_REGISTRY_PATH,
                     iteration2b_remediation.I2B_SCHEDULE_REGISTRY_PATH, iteration2b_final_validation.I2BVAL_SCHEDULE_REGISTRY_PATH)

assert I3VAL_PERSISTENT_ROOT not in _OTHER_ROOTS, "I3 final-validation root must not equal any prior campaign root"
assert I3VAL_LEDGER_PATH not in _OTHER_LEDGERS, "I3 final-validation ledger must not equal any prior campaign ledger"
assert I3VAL_SEQUENCE_PROJECTION_PATH not in _OTHER_SEQUENCES, "I3 final-validation sequence must not equal any prior campaign sequence"
assert I3VAL_SCHEDULE_REGISTRY_PATH not in _OTHER_REGISTRIES, "I3 final-validation registry must not equal any prior campaign registry"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute (or resolve/dry-run) exactly one frozen Iteration-3 "
            "ITERATION3_FINAL_UNSEEN_VALIDATION campaign slot (Phase16N/16N.1 "
            "design freeze). All scientific parameters are resolved from the "
            "frozen Iteration-3 final-validation sequence -- this CLI accepts "
            "no override of replicas, RPS, schedule, or any other scientific "
            "parameter."
        )
    )
    parser.add_argument("--validation-sequence-index", type=int, help="1..32, resolved against the frozen Iteration-3 final-validation sequence")
    parser.add_argument("--attempt-number", default=None, type=int,
                         help="Execution attempt number for this planned slot; if omitted, auto-resolved as the next physical attempt from disk")
    parser.add_argument("--session-id", default=None, help="Optional session identifier; auto-generated if omitted")
    parser.add_argument("--resolve", action="store_true", help="Print the resolved slot identity only (no historical.main() call at all)")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan and exit; no traffic, no Kubernetes mutation")
    parser.add_argument("--dry-run-all", action="store_true", help="Dry-run all 32 frozen sequence slots, in strict order, in one invocation")
    parser.add_argument("--execute", action="store_true",
                         help="LIVE scientific execution. Never invoked by Phase16P. Requires all earlier sequence indices already VALID.")
    return parser


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_all_governing_hashes() -> Dict[str, Any]:
    """Fail-closed hash verification of every governing input before this
    wrapper trusts any of them. Returns a report dict; raises RuntimeError
    (STOP, fail-closed) on the first mismatch."""
    checks = [
        (PROJECT_ROOT.parent / "task_outputs/phase_16n_iteration3_fresh_validation_matrix.csv",
         EXPECTED_FRESH_MATRIX_SHA256, "fresh_validation_matrix"),
        (PROJECT_ROOT.parent / "task_outputs/phase_16n_iteration3_final_validation_sequence.csv",
         EXPECTED_SEQUENCE_CSV_SHA256, "sequence_csv"),
        (I3VAL_SEQUENCE_PROJECTION_PATH, EXPECTED_SEQUENCE_PROJECTION_SHA256, "sequence_projection"),
        (I3VAL_SCHEDULE_REGISTRY_PATH, EXPECTED_SCHEDULE_REGISTRY_SHA256, "schedule_registry"),
        (PROJECT_ROOT.parent / "task_outputs/phase_16n_iteration3_final_validation_protocol.yaml",
         EXPECTED_PROTOCOL_SHA256, "execution_protocol"),
        (PROJECT_ROOT.parent / "task_outputs/phase_16o_iteration3_final_validation_prediction_preregistration.csv",
         EXPECTED_PREDICTION_PREREGISTRATION_SHA256, "prediction_preregistration"),
    ]
    report = {}
    for path, expected, name in checks:
        if not path.exists():
            raise RuntimeError(f"STOP (fail-closed): {name} not found: {path}")
        actual = _sha256_of(path)
        if actual != expected:
            raise RuntimeError(f"STOP (fail-closed): {name} hash mismatch: {actual} != {expected}")
        report[name] = {"path": str(path), "sha256": actual, "match": True}
    return report


def load_verified_i3val_sequence() -> List[dict]:
    """Fail-closed hash verification before returning the Iteration-3
    final-validation sequence PROJECTION (the executable, harness-schema
    form -- campaign_sequence_index/cell_or_transition_id/schedule_path/
    schedule_sha256 -- mechanically and deterministically derived, once,
    from the already-frozen design sequence CSV + schedule registry; never
    regenerated by this wrapper)."""
    verify_all_governing_hashes()
    with open(I3VAL_SEQUENCE_PROJECTION_PATH, "r", encoding="utf-8") as f:
        sequence = json.load(f)
    if len(sequence) != 32:
        raise RuntimeError(f"STOP (fail-closed): expected exactly 32 Iteration-3 final-validation slots, found {len(sequence)}")
    return sequence


def load_i3val_schedule_registry() -> dict:
    if not I3VAL_SCHEDULE_REGISTRY_PATH.exists():
        raise RuntimeError(f"STOP (fail-closed): Iteration-3 final-validation schedule registry not found: {I3VAL_SCHEDULE_REGISTRY_PATH}")
    with open(I3VAL_SCHEDULE_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_next_physical_attempt_number(calibration_root: Path, planned_slot_id: str) -> int:
    """Physical attempt numbering is separate from scientific-ledger
    membership: every attempt directory that was ever physically created
    for this planned_slot_id consumes its number, including a preserved
    pre-traffic-tooling-failure directory or pre-ledger crash directory
    with no corresponding ledger row. Never resets merely because no
    scientific ledger entry exists."""
    if not calibration_root.exists():
        return 1
    pattern = re.compile(rf"^{re.escape(planned_slot_id)}_attempt(\d+)_")
    max_n = 0
    for p in calibration_root.iterdir():
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def resolve_slot(index: int) -> Dict[str, Any]:
    """Resolves and returns exactly the required display fields for a
    requested sequence index: sequence index, planned slot ID, cell ID,
    repeat number, replicas, offered RPS, schedule path, schedule SHA256,
    current physical attempt number. Read-only; no execution, no
    historical.main() call."""
    sequence = load_verified_i3val_sequence()
    matches = [e for e in sequence if e["campaign_sequence_index"] == index]
    if not matches:
        raise ValueError(f"validation_sequence_index={index} not found (valid range: 1..{len(sequence)})")
    slot = matches[0]
    next_attempt = resolve_next_physical_attempt_number(I3VAL_PERSISTENT_ROOT, slot["planned_slot_id"])
    return {
        "sequence_index": slot["campaign_sequence_index"],
        "planned_slot_id": slot["planned_slot_id"],
        "cell_id": slot["cell_or_transition_id"],
        "repeat_number": slot["repeat_number"],
        "replicas": slot["replicas"],
        "offered_rps": slot["offered_rps"],
        "schedule_path": slot["schedule_path"],
        "schedule_sha256": slot["schedule_sha256"],
        "current_physical_attempt_number": next_attempt,
    }


def _earlier_indices_all_valid(index: int) -> bool:
    """Strict frozen sequence order: no later sequence index may execute
    while an earlier slot remains scientifically unresolved (VALID)."""
    from scripts.dynamics_calibration_lib import load_ledger, LEDGER_STATE_VALID
    ledger = load_ledger(I3VAL_LEDGER_PATH)
    if ledger is None:
        return index == 1
    for i in range(1, index):
        slot_state = ledger["slots"].get(str(i), {}).get("state")
        if slot_state != LEDGER_STATE_VALID:
            return False
    return True


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.dry_run_all:
        report = verify_all_governing_hashes()
        print(json.dumps({"governing_hash_verification": report}, indent=2))
        sequence = load_verified_i3val_sequence()
        registry = load_i3val_schedule_registry()
        results = []
        for index in range(1, 33):
            hist_argv = ["--campaign-sequence-index", str(index), "--attempt-number", "1", "--dry-run"]
            rc = historical.main(
                hist_argv, sequence_override=sequence, schedule_registry_override=registry,
                calibration_root=I3VAL_PERSISTENT_ROOT, ledger_path=I3VAL_LEDGER_PATH,
            )
            results.append({"sequence_index": index, "dry_run_exit_code": rc})
        print(json.dumps({"all_32_dry_run_results": results}, indent=2))
        return 0 if all(r["dry_run_exit_code"] == 0 for r in results) else 1

    if args.validation_sequence_index is None:
        raise SystemExit("STOP: --validation-sequence-index is required unless --dry-run-all is given")

    if not (1 <= args.validation_sequence_index <= 32):
        raise SystemExit(
            f"STOP (fail-closed): --validation-sequence-index={args.validation_sequence_index} "
            f"outside the frozen Iteration-3 final-validation range [1, 32]"
        )

    if args.resolve:
        info = resolve_slot(args.validation_sequence_index)
        print(json.dumps(info, indent=2))
        return 0

    i3val_sequence = load_verified_i3val_sequence()
    i3val_registry = load_i3val_schedule_registry()

    if I3VAL_PERSISTENT_ROOT in _OTHER_ROOTS:
        raise RuntimeError("STOP (fail-closed): Iteration-3 final-validation root resolves to a prior campaign root")
    if I3VAL_LEDGER_PATH in _OTHER_LEDGERS:
        raise RuntimeError("STOP (fail-closed): Iteration-3 final-validation ledger path resolves to a prior campaign ledger")

    resolved = resolve_slot(args.validation_sequence_index)
    attempt_number = args.attempt_number if args.attempt_number is not None else resolved["current_physical_attempt_number"]

    if args.execute:
        if not _earlier_indices_all_valid(args.validation_sequence_index):
            raise SystemExit(
                f"STOP (fail-closed): sequence index {args.validation_sequence_index} requested for live "
                f"execution, but not every earlier index (1..{args.validation_sequence_index - 1}) is "
                f"already scientifically VALID in the Iteration-3 ledger. Strict frozen sequence order "
                f"prohibits executing a later slot while an earlier one remains unresolved."
            )

    hist_argv = ["--campaign-sequence-index", str(args.validation_sequence_index), "--attempt-number", str(attempt_number)]
    if args.session_id:
        hist_argv += ["--session-id", args.session_id]
    if args.dry_run:
        hist_argv += ["--dry-run"]
    elif not args.execute:
        # Neither --dry-run, --dry-run-all, --resolve, nor --execute was
        # given: refuse to guess. This wrapper never executes live traffic
        # by accident.
        raise SystemExit("STOP: pass exactly one of --resolve, --dry-run, --dry-run-all, or --execute")

    return historical.main(
        hist_argv,
        sequence_override=i3val_sequence,
        schedule_registry_override=i3val_registry,
        calibration_root=I3VAL_PERSISTENT_ROOT,
        ledger_path=I3VAL_LEDGER_PATH,
    )


if __name__ == "__main__":
    raise SystemExit(main())
