#!/usr/bin/env python3
"""scripts/generate_tuning_schedule.py

Generate and persist the canonical Phase-11C HPA-tuning arrival schedule ONCE,
offline -- this script never opens an HTTP client and never sends a single
request to the demo-app. It only parses a traffic profile and computes planned
arrival timestamps (loadgen.traffic_generator.generate_arrival_schedule()),
then writes them to a fixed, version-controlled location together with a
SHA-256 digest sidecar file.

Why this exists (Phase 11C.1, see task_outputs/phase_11c1_hpa_tuning_preparation.md):
loadgen/traffic_generator.py couples "generate a schedule" and "dispatch traffic"
inside one run (TrafficGenerator.run_async()), but schedule GENERATION itself
(generate_arrival_schedule()) is a pure, non-networked function. This script
calls only that pure path, so the canonical schedule for all three HPA-target
candidates (220% / 230% / 250%) can be produced exactly once, hashed, and then
exact-replayed via `traffic_generator.py --replay-schedule <path>` for each
candidate run -- never independently regenerated per candidate.

Usage:
    python3 -m scripts.generate_tuning_schedule \\
        --profile configs/traffic_profiles/hpa_tuning/hpa_tuning_v1.yaml \\
        --arrival-process deterministic \\
        --out configs/traffic_profiles/hpa_tuning/canonical_schedule.csv

Running this script twice with the same inputs against a fresh --out path is
expected to (and, for deterministic arrivals, verifiably does) produce a
byte-identical file and hash -- see tests/unit/test_generate_tuning_schedule.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loadgen.traffic_generator import (  # noqa: E402
    SCHEDULE_CSV_FIELDNAMES,
    generate_arrival_schedule,
    parse_traffic_profile,
)


def generate_and_persist(
    profile_path: Path,
    arrival_process: str,
    seed: int,
    out_path: Path,
) -> str:
    """Generate a schedule offline (no dispatch) and write it + its SHA-256 hash.

    Refuses to overwrite an existing schedule file, consistent with this
    project's existing preserve-don't-overwrite convention for experiment
    artifacts (metrics_collector.py, traffic_generator.py both refuse to
    overwrite existing run directories for the same reason).

    Returns the SHA-256 hex digest of the written file.
    """
    if out_path.exists():
        raise FileExistsError(
            f"Canonical schedule already exists: {out_path}. "
            "Refusing to overwrite an existing tuning schedule -- if a new "
            "schedule is genuinely needed, choose a new --out path so the "
            "old one remains available for audit."
        )

    profile = parse_traffic_profile(profile_path)
    schedule = generate_arrival_schedule(profile, seed=seed, arrival_process=arrival_process)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEDULE_CSV_FIELDNAMES)
        writer.writeheader()
        for req in schedule:
            writer.writerow(req.to_dict())

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    hash_path = out_path.with_suffix(out_path.suffix + ".sha256")
    hash_path.write_text(f"{digest}  {out_path.name}\n")

    return digest


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Path to the traffic profile YAML")
    parser.add_argument(
        "--arrival-process",
        choices=["poisson", "deterministic", "even"],
        default="deterministic",
        help="Arrival process (default: deterministic, per Phase 11C.1's tuning rationale)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed (unused for deterministic arrivals)")
    parser.add_argument("--out", required=True, help="Output path for the canonical schedule CSV")
    return parser.parse_args(args)


def main() -> None:
    parsed = parse_args()
    profile_path = Path(parsed.profile)
    if not profile_path.is_absolute():
        profile_path = PROJECT_ROOT / profile_path
    out_path = Path(parsed.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    try:
        digest = generate_and_persist(profile_path, parsed.arrival_process, parsed.seed, out_path)
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        sys.exit(1)

    print(f"Canonical schedule written: {out_path}")
    print(f"SHA-256: {digest}")
    print(f"Hash sidecar: {out_path}.sha256")


if __name__ == "__main__":
    main()
