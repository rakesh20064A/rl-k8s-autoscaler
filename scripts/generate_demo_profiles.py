#!/usr/bin/env python3
"""scripts/generate_demo_profiles.py

Deterministic generator for the demo UI's `pattern x resource x load`
traffic-profile catalog.

DEMO ONLY -- every file this script writes lives under
`configs/traffic_profiles/v2_demo/`, is firewalled from both the training
schedule (`autoscaler.src.rl_env.training_contract_v2.
V2_TRAINING_PROFILE_ALLOWLIST`) and the evaluation/stress profile sets
(`scripts.evaluate_v2_campaign.EVAL_PROFILE_ALLOWLIST` /
`STRESS_PROFILE_ALLOWLIST`), by construction (different directory,
different allowlist, distinct seed range, no import relationship -- see
`tests/unit/test_v2_demo_profile_catalog.py`). None of this is scientific
evidence.

Catalog
-------
6 traffic patterns x 3 resource types x 2 load levels = 36 profiles,
named `demo_<pattern>_<resource>_<load>.yaml`:

  patterns  : constant, step, ramp, bursts, periodic, ramp_bursts
  resources : cpu, memory, mixed
  loads     : normal, high

Every profile totals 480s (= 16 x the project's 30s decision interval --
see `scripts/eval_v2_episodes.py`'s `decision_interval_s=30.0` default and
`demo_ui/fixtures_v2.py`'s `N_DECISIONS=16`, matching the existing 3
`demo_*_short` profiles' own 480s/16-decision convention).

Load-level envelope (confirmed against existing profiles rather than
invented):
  - normal: peak RPS ~20 (<= 20), intensity 3 -- matches the existing
    `demo_*_short` profiles' own top tier (18 RPS) and
    `configs/traffic_profiles/v2_eval/`'s intensity=3 convention.
  - high, cpu/mixed: peak RPS <= 70, intensity 4 -- the SAME peak already
    validated live by `configs/traffic_profiles/v2/burst_mixed.yaml`
    (training) and `configs/traffic_profiles/v2_stress/stress_cpu_burst.
    yaml` / `stress_mixed_burst.yaml` (stress), never a harder regime.
  - high, memory: peak RPS <= 45, intensity 4 -- the SAME peak already
    validated live by `configs/traffic_profiles/v2_stress/
    stress_memory_ramp.yaml` (see `docs/tables/
    stress_explore_descriptive_metrics.*` for that number's provenance).
  - memory/mixed profiles (both load levels) set `memory_mb: 16` /
    `memory_safe_max_in_flight: 8`, reusing the same validated safety
    ceiling every other V2 memory/mixed profile (training, eval, stress,
    and the original 3 `demo_*_short` profiles) already uses.

Seed assignment (deterministic, documented): seeds 7101-7136, one per
file, assigned in a single fixed iteration order -- patterns outer,
resources middle, loads inner:

    index = pattern_index * (3 resources * 2 loads)
          + resource_index * (2 loads)
          + load_index
    seed  = 7101 + index

`seed_for()` implements this exactly; `ALL_PROFILE_NAMES` lists every
name in that same order.

The original 3 demo profiles (`demo_cpu_short`, `demo_memory_short`,
`demo_mixed_short`) are left in place, UNTOUCHED, as a separate legacy
set (closest matches: `demo_ramp_cpu_normal`-ish for cpu [it is a 2-step
constant, closest to this catalog's `step` pattern],
`demo_ramp_memory_normal` for memory, and `demo_bursts_mixed_normal` for
mixed) -- not deleted, not aliased, not re-pointed; they keep working via
their own existing allowlist entry in `scripts/run_demo_episode_v2.py`,
and the new 36-profile catalog is additive.

Every file this script writes is validated FIRST (parses via
`loadgen.traffic_generator.parse_traffic_profile`, has exactly the
intended 480s total duration, is not degenerate, and its peak RPS is
within its load level's limit) -- `write_profile()` refuses to write a
file that fails validation. See `tests/unit/test_v2_demo_profile_catalog.
py` for the equivalent pytest coverage.

Usage:
    .venv/bin/python scripts/generate_demo_profiles.py [--check]

`--check` validates the already-written files on disk without
regenerating them (used by CI/tests to detect drift).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loadgen.traffic_generator import TrafficProfile, parse_traffic_profile  # noqa: E402

DEMO_DIR = PROJECT_ROOT / "configs" / "traffic_profiles" / "v2_demo"

PATTERNS: Tuple[str, ...] = ("constant", "step", "ramp", "bursts", "periodic", "ramp_bursts")
RESOURCES: Tuple[str, ...] = ("cpu", "memory", "mixed")
LOADS: Tuple[str, ...] = ("normal", "high")

PATTERN_LABELS: Dict[str, str] = {
    "constant": "Constant",
    "step": "Step",
    "ramp": "Ramp",
    "bursts": "Bursts",
    "periodic": "Periodic (wave)",
    "ramp_bursts": "Ramp + bursts",
}
RESOURCE_LABELS: Dict[str, str] = {"cpu": "CPU", "memory": "Memory", "mixed": "Mixed"}
LOAD_LABELS: Dict[str, str] = {"normal": "Normal", "high": "High"}

DEFAULT_PATTERN = "constant"
DEFAULT_RESOURCE = "cpu"
DEFAULT_LOAD = "normal"

TOTAL_DURATION_S = 480.0
DECISION_INTERVAL_S = 30.0
N_DECISIONS = int(TOTAL_DURATION_S // DECISION_INTERVAL_S)  # 16

SEED_BASE = 7101
SEED_MAX = 7136  # SEED_BASE + (6 * 3 * 2) - 1

NORMAL_PEAK_LIMIT = 20.0
HIGH_PEAK_LIMIT_CPU_MIXED = 70.0
HIGH_PEAK_LIMIT_MEMORY = 45.0

# Actual target peaks used when building phases -- comfortably under each
# limit above (never equal to it), so legitimate floating-point phase math
# (ramps/periodic sine peaks/step edges) can never tip a generated file
# over its own load level's ceiling.
NORMAL_PEAK = 18.0
HIGH_PEAK_CPU_MIXED = 65.0
HIGH_PEAK_MEMORY = 40.0

INTENSITY_NORMAL = 3
INTENSITY_HIGH = 4


def profile_slug(pattern: str, resource: str, load: str) -> str:
    return f"demo_{pattern}_{resource}_{load}"


def profile_filename(pattern: str, resource: str, load: str) -> str:
    return f"{profile_slug(pattern, resource, load)}.yaml"


def seed_for(pattern: str, resource: str, load: str) -> int:
    """Deterministic seed assignment: patterns outer, resources middle,
    loads inner (see module docstring's `index`/`seed` formula)."""
    p_idx = PATTERNS.index(pattern)
    r_idx = RESOURCES.index(resource)
    l_idx = LOADS.index(load)
    index = p_idx * (len(RESOURCES) * len(LOADS)) + r_idx * len(LOADS) + l_idx
    return SEED_BASE + index


ALL_COMBOS: Tuple[Tuple[str, str, str], ...] = tuple(
    (pattern, resource, load) for pattern in PATTERNS for resource in RESOURCES for load in LOADS
)
ALL_PROFILE_NAMES: Tuple[str, ...] = tuple(profile_slug(*combo) for combo in ALL_COMBOS)

LEGACY_DEMO_PROFILE_NAMES: Tuple[str, ...] = ("demo_cpu_short", "demo_memory_short", "demo_mixed_short")


def parse_profile_name(name: str) -> Optional[Tuple[str, str, str]]:
    """Reverse of `profile_slug()`: recovers (pattern, resource, load) from
    a generated name, or None if `name` isn't one of this catalog's 36
    names (e.g. one of the 3 legacy `demo_*_short` profiles)."""
    if not name.startswith("demo_"):
        return None
    rest = name[len("demo_"):]
    for load in LOADS:
        suffix = f"_{load}"
        if not rest.endswith(suffix):
            continue
        rest2 = rest[: -len(suffix)]
        for resource in RESOURCES:
            mid_suffix = f"_{resource}"
            if not rest2.endswith(mid_suffix):
                continue
            pattern = rest2[: -len(mid_suffix)]
            if pattern in PATTERNS:
                return pattern, resource, load
    return None


def _peak_for(resource: str, load: str) -> float:
    if load == "normal":
        return NORMAL_PEAK
    return HIGH_PEAK_MEMORY if resource == "memory" else HIGH_PEAK_CPU_MIXED


def _peak_limit_for(resource: str, load: str) -> float:
    if load == "normal":
        return NORMAL_PEAK_LIMIT
    return HIGH_PEAK_LIMIT_MEMORY if resource == "memory" else HIGH_PEAK_LIMIT_CPU_MIXED


def _resource_fields(resource: str) -> Dict[str, Any]:
    if resource == "cpu":
        return {"resource_mode": "cpu", "memory_mb": 0}
    return {"resource_mode": resource, "memory_mb": 16, "memory_safe_max_in_flight": 8}


def _phases_constant(peak: float) -> List[Dict[str, Any]]:
    return [{"type": "constant", "name": "level", "duration_seconds": TOTAL_DURATION_S, "target_rps": peak}]


def _phases_step(peak: float) -> List[Dict[str, Any]]:
    low = round(peak * 0.25, 1)
    mid = round(peak * 0.6, 1)
    return [{
        "type": "step", "name": "step", "duration_seconds": TOTAL_DURATION_S,
        "steps": [
            {"at_second": 0, "rps": low},
            {"at_second": 160, "rps": mid},
            {"at_second": 320, "rps": peak},
        ],
    }]


def _phases_ramp(peak: float) -> List[Dict[str, Any]]:
    start = round(peak * 0.15, 1)
    return [
        {"type": "ramp", "name": "ramp_up", "duration_seconds": 200, "start_rps": start, "end_rps": peak},
        {"type": "constant", "name": "hold", "duration_seconds": 80, "target_rps": peak},
        {"type": "ramp", "name": "ramp_down", "duration_seconds": 200, "start_rps": peak, "end_rps": start},
    ]


def _phases_bursts(peak: float) -> List[Dict[str, Any]]:
    baseline = round(peak * 0.22, 1)
    return [{
        "type": "burst", "name": "burst", "duration_seconds": TOTAL_DURATION_S,
        "baseline_rps": baseline, "burst_rps": peak,
        "burst_duration_seconds": 30, "inter_burst_interval_seconds": 90,
    }]


def _phases_periodic(peak: float) -> List[Dict[str, Any]]:
    mean = round(peak * 0.58, 1)
    amplitude = round(peak - mean, 1)
    return [{
        "type": "periodic", "name": "wave", "duration_seconds": TOTAL_DURATION_S,
        "mean_rps": mean, "amplitude_rps": amplitude, "period_seconds": 120, "noise_std_rps": 0.0,
    }]


def _phases_ramp_bursts(peak: float) -> List[Dict[str, Any]]:
    low = round(peak * 0.15, 1)
    mid = round(peak * 0.5, 1)
    return [
        {"type": "ramp", "name": "ramp_up", "duration_seconds": 200, "start_rps": low, "end_rps": mid},
        {
            "type": "burst", "name": "burst", "duration_seconds": 280,
            "baseline_rps": mid, "burst_rps": peak,
            "burst_duration_seconds": 30, "inter_burst_interval_seconds": 90,
        },
    ]


_PHASE_BUILDERS = {
    "constant": _phases_constant,
    "step": _phases_step,
    "ramp": _phases_ramp,
    "bursts": _phases_bursts,
    "periodic": _phases_periodic,
    "ramp_bursts": _phases_ramp_bursts,
}


def build_profile_dict(pattern: str, resource: str, load: str) -> Dict[str, Any]:
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown pattern {pattern!r}; must be one of {PATTERNS}")
    if resource not in RESOURCES:
        raise ValueError(f"Unknown resource {resource!r}; must be one of {RESOURCES}")
    if load not in LOADS:
        raise ValueError(f"Unknown load {load!r}; must be one of {LOADS}")

    name = profile_slug(pattern, resource, load)
    seed = seed_for(pattern, resource, load)
    peak = _peak_for(resource, load)
    phases = _PHASE_BUILDERS[pattern](peak)
    intensity = INTENSITY_NORMAL if load == "normal" else INTENSITY_HIGH

    description = (
        f"DEMO ONLY (not evaluation evidence): {PATTERN_LABELS[pattern].lower()} traffic, "
        f"{RESOURCE_LABELS[resource].lower()} resource pressure, {LOAD_LABELS[load].lower()} load "
        f"(peak ~{peak:g} req/s), 8 min"
    )

    data: Dict[str, Any] = {
        "profile": name,
        "description": description,
        "seed": seed,
        "intensity": intensity,
        "endpoint": "/work",
        **_resource_fields(resource),
        "phases": phases,
    }
    return data


def _dump_yaml(data: Dict[str, Any], name: str, pattern: str, resource: str, load: str) -> str:
    import yaml

    header = (
        f"# configs/traffic_profiles/v2_demo/{name}.yaml\n"
        "# Generated by scripts/generate_demo_profiles.py -- DEMO ONLY, NOT evaluation\n"
        "# evidence. Firewalled from autoscaler.src.rl_env.training_contract_v2.\n"
        "# V2_TRAINING_PROFILE_ALLOWLIST and scripts.evaluate_v2_campaign.\n"
        "# EVAL_PROFILE_ALLOWLIST / STRESS_PROFILE_ALLOWLIST (different directory,\n"
        "# different allowlist; see tests/unit/test_v2_demo_profile_catalog.py).\n"
        f"# pattern={pattern} resource={resource} load={load} seed={data['seed']}\n"
    )
    body = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    return header + "\n" + body


class ProfileValidationError(RuntimeError):
    pass


def _peak_rps(profile: TrafficProfile, step_s: float = 1.0) -> float:
    total = profile.total_duration
    peak = 0.0
    t = 0.0
    while t < total:
        rate, _, _ = profile.get_phase_and_target_rps(t)
        peak = max(peak, rate)
        t += step_s
    return peak


def validate_profile_text(yaml_text: str, name: str, load: str) -> TrafficProfile:
    """Parses `yaml_text` exactly as `parse_traffic_profile()` would (via a
    temp file) and checks: parses cleanly, total duration is exactly
    480s, is not degenerate (peak RPS > 0), and peak RPS is within the
    load level's limit. Raises ProfileValidationError on any failure --
    never silently written."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_text)
        tmp_path = Path(f.name)
    try:
        profile = parse_traffic_profile(tmp_path)
    except Exception as exc:  # noqa: BLE001
        raise ProfileValidationError(f"{name}: failed to parse: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    if profile.total_duration != TOTAL_DURATION_S:
        raise ProfileValidationError(
            f"{name}: total_duration {profile.total_duration} != {TOTAL_DURATION_S}"
        )

    peak = _peak_rps(profile)
    if peak <= 0.0:
        raise ProfileValidationError(f"{name}: degenerate profile, peak RPS is {peak}")

    limit = _peak_limit_for(profile.resource_mode, load)
    if peak > limit:
        raise ProfileValidationError(f"{name}: peak RPS {peak} exceeds {load} limit {limit}")

    return profile


def write_profile(pattern: str, resource: str, load: str, out_dir: Path = DEMO_DIR) -> Path:
    """Builds, validates, and writes ONE profile file. Refuses to write
    (raises ProfileValidationError, no file touched) if validation fails."""
    data = build_profile_dict(pattern, resource, load)
    name = data["profile"]
    yaml_text = _dump_yaml(data, name, pattern, resource, load)

    validate_profile_text(yaml_text, name, load)  # raises before any write on failure

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.yaml"
    out_path.write_text(yaml_text, encoding="utf-8")
    return out_path


def generate_all(out_dir: Path = DEMO_DIR) -> List[Path]:
    written = []
    for pattern, resource, load in ALL_COMBOS:
        written.append(write_profile(pattern, resource, load, out_dir))
    return written


def check_all(out_dir: Path = DEMO_DIR) -> List[str]:
    """Validates the 36 files already on disk (no writes). Returns a list
    of problem descriptions (empty means everything is valid)."""
    problems: List[str] = []
    for pattern, resource, load in ALL_COMBOS:
        name = profile_slug(pattern, resource, load)
        path = out_dir / f"{name}.yaml"
        if not path.is_file():
            problems.append(f"{name}: missing file {path}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
            validate_profile_text(text, name, load)
        except ProfileValidationError as exc:
            problems.append(str(exc))
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate existing files on disk, do not write")
    parser.add_argument("--out-dir", type=Path, default=DEMO_DIR)
    args = parser.parse_args(argv)

    if args.check:
        problems = check_all(args.out_dir)
        if problems:
            for p in problems:
                print(f"INVALID: {p}", file=sys.stderr)
            return 1
        print(f"OK: all {len(ALL_COMBOS)} demo profiles valid under {args.out_dir}")
        return 0

    written = generate_all(args.out_dir)
    print(f"Wrote {len(written)} demo profiles under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
