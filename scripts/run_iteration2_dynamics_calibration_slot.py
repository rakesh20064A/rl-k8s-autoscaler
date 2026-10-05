#!/usr/bin/env python3
"""scripts/run_iteration2_dynamics_calibration_slot.py

Phase 16I.1: Iteration-2 DEVELOPMENT_FITTING execution entry point.

This module does NOT duplicate any execution/collection/V8-structural-
validity/route-integrity/dataplane-acquisition/HPA-isolation/restoration
logic. It reuses `scripts.run_dynamics_calibration_slot.main()` -- the exact
same shared, historically-hardened pipeline the official 102-slot campaign
uses -- supplying only a separately-frozen Iteration-2 campaign context
(sequence, schedule registry, persistent root, ledger path) via that
function's code-level-only override parameters. It never touches the
historical `global_campaign_sequence.json`, historical
`schedule_registry.json`, historical `persistent-data/dynamics_calibration/`
tree, or the historical campaign ledger.

Like the historical CLI, this CLI accepts NO override of replicas, RPS,
schedule, or any other scientific parameter -- every scientific parameter is
resolved solely from the frozen Iteration-2 development sequence
(configs/traffic_profiles/rl_dynamics_calibration_iteration2/iteration2_development_sequence.json),
itself a deterministic, mechanical, hash-verified projection of the
Phase16H design freeze.
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
from scripts.dynamics_calibration_lib import GLOBAL_SEQUENCE_PATH as HISTORICAL_GLOBAL_SEQUENCE_PATH  # noqa: E402
from scripts.dynamics_calibration_lib import SCHEDULE_REGISTRY_PATH as HISTORICAL_SCHEDULE_REGISTRY_PATH  # noqa: E402

I2_CAL_ROOT = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration_iteration2"
I2_SEQUENCE_PATH = I2_CAL_ROOT / "iteration2_development_sequence.json"
I2_SCHEDULE_REGISTRY_PATH = I2_CAL_ROOT / "iteration2_development_schedule_registry.json"
I2_PERSISTENT_ROOT = PROJECT_ROOT / "persistent-data/dynamics_calibration_iteration2/development"
I2_LEDGER_PATH = I2_PERSISTENT_ROOT / "iteration2_development_ledger.json"

# Frozen BEFORE any Iteration-2 outcome exists (Phase16I.1). Any drift in
# either file must fail closed, never silently proceed.
EXPECTED_PHASE16H_DEVELOPMENT_SEQUENCE_SHA256 = "4c99a9b0c283105c1c6790ff8265195be2d11a01f356d3d514c825297923259b"
EXPECTED_I2_SEQUENCE_PROJECTION_SHA256 = "9caa37cb94b6229fbb2d841f3ebb13fa89ca2d1bacc6df420c672dcdc43b0f4a"

# Hard write-firewall: an Iteration-2 context must never coincide with the
# historical campaign context. Checked once at import time (fails the whole
# process immediately, before any CLI parsing) and again defensively in main().
assert I2_PERSISTENT_ROOT != historical.CALIBRATION_ROOT, "Iteration-2 root must not equal the historical calibration root"
assert I2_LEDGER_PATH != historical.LEDGER_PATH, "Iteration-2 ledger must not equal the historical ledger"
assert I2_SEQUENCE_PATH != HISTORICAL_GLOBAL_SEQUENCE_PATH, "Iteration-2 sequence must not equal the historical global sequence"
assert I2_SCHEDULE_REGISTRY_PATH != HISTORICAL_SCHEDULE_REGISTRY_PATH, "Iteration-2 schedule registry must not equal the historical schedule registry"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute exactly one frozen Iteration-2 DEVELOPMENT_FITTING campaign slot "
            "(Phase16H design freeze, Phase16I.1 tooling). All scientific parameters are "
            "resolved from the frozen Iteration-2 development sequence -- this CLI accepts "
            "no override of replicas, RPS, schedule, or any other scientific parameter."
        )
    )
    parser.add_argument("--iteration2-sequence-index", required=True, type=int, help="1..24, resolved against the frozen Iteration-2 development sequence")
    parser.add_argument("--attempt-number", default=1, type=int, help="Execution attempt number for this planned slot (default 1)")
    parser.add_argument("--session-id", default=None, help="Optional session identifier; auto-generated if omitted")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan and exit; no traffic, no Kubernetes mutation")
    return parser


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_i2_sequence() -> List[dict]:
    """Fail-closed hash verification before returning the Iteration-2
    sequence: both the frozen Phase16H CSV design source and this module's
    own executable JSON projection must match their recorded hashes exactly."""
    phase16h_csv = PROJECT_ROOT / "task_outputs" / "phase_16h_iteration2_development_sequence.csv"
    task_outputs_alt = Path("/Users/rakeshdandamudi/Capstone/task_outputs/phase_16h_iteration2_development_sequence.csv")
    source_path = phase16h_csv if phase16h_csv.exists() else task_outputs_alt
    if not source_path.exists():
        raise RuntimeError("STOP (fail-closed): phase_16h_iteration2_development_sequence.csv not found -- cannot verify design provenance")
    actual_source_hash = _sha256_of(source_path)
    if actual_source_hash != EXPECTED_PHASE16H_DEVELOPMENT_SEQUENCE_SHA256:
        raise RuntimeError(
            f"STOP (fail-closed): Phase16H development sequence hash mismatch: "
            f"{actual_source_hash} != {EXPECTED_PHASE16H_DEVELOPMENT_SEQUENCE_SHA256}"
        )

    if not I2_SEQUENCE_PATH.exists():
        raise RuntimeError(f"STOP (fail-closed): Iteration-2 sequence projection not found: {I2_SEQUENCE_PATH}")
    actual_projection_hash = _sha256_of(I2_SEQUENCE_PATH)
    if actual_projection_hash != EXPECTED_I2_SEQUENCE_PROJECTION_SHA256:
        raise RuntimeError(
            f"STOP (fail-closed): Iteration-2 sequence projection hash mismatch: "
            f"{actual_projection_hash} != {EXPECTED_I2_SEQUENCE_PROJECTION_SHA256}"
        )

    with open(I2_SEQUENCE_PATH, "r", encoding="utf-8") as f:
        sequence = json.load(f)
    if len(sequence) != 24:
        raise RuntimeError(f"STOP (fail-closed): expected exactly 24 Iteration-2 development slots, found {len(sequence)}")
    return sequence


def load_i2_schedule_registry() -> dict:
    if not I2_SCHEDULE_REGISTRY_PATH.exists():
        raise RuntimeError(f"STOP (fail-closed): Iteration-2 schedule registry not found: {I2_SCHEDULE_REGISTRY_PATH}")
    with open(I2_SCHEDULE_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not (1 <= args.iteration2_sequence_index <= 24):
        raise SystemExit(
            f"STOP (fail-closed): --iteration2-sequence-index={args.iteration2_sequence_index} "
            f"outside the frozen Iteration-2 development range [1, 24]"
        )

    i2_sequence = load_verified_i2_sequence()
    i2_registry = load_i2_schedule_registry()

    # Defensive, runtime re-check of the write firewall (import-time asserts
    # above already guarantee this structurally; repeated here so a future
    # edit to either constant cannot silently regress it).
    if I2_PERSISTENT_ROOT == historical.CALIBRATION_ROOT:
        raise RuntimeError("STOP (fail-closed): Iteration-2 persistent root resolves to the historical calibration root")
    if I2_LEDGER_PATH == historical.LEDGER_PATH:
        raise RuntimeError("STOP (fail-closed): Iteration-2 ledger path resolves to the historical ledger path")

    hist_argv = ["--campaign-sequence-index", str(args.iteration2_sequence_index), "--attempt-number", str(args.attempt_number)]
    if args.session_id:
        hist_argv += ["--session-id", args.session_id]
    if args.dry_run:
        hist_argv += ["--dry-run"]

    return historical.main(
        hist_argv,
        sequence_override=i2_sequence,
        schedule_registry_override=i2_registry,
        calibration_root=I2_PERSISTENT_ROOT,
        ledger_path=I2_LEDGER_PATH,
    )


if __name__ == "__main__":
    raise SystemExit(main())
