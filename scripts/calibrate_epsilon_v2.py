#!/usr/bin/env python3
"""scripts/calibrate_epsilon_v2.py

V2 Phase 5B PART B: computes the pre-registered (CORR-60) near-tie epsilon
for PPO-V2's deterministic, safety-aware evaluation inference. A NEW file
-- imports `action_selection.compute_margin_stats()` /
`extract_action_probabilities()` and `safe_inference.load_ppo_model_safe()`
unmodified; does not edit `action_selection.py` or any of the 43
FREEZE_MANIFEST.json files.

Per `EVALUATION_PREREGISTRATION.json`'s `epsilon_rule`:
    epsilon = 10th percentile of the final model's top1-top2 action-
              probability margin, computed over `normalized_observation`
              values from the LAST 5 completed training sessions'
              `step_telemetry.jsonl`, capped at 0.05, floored at 0.0.

Read-only against `persistent-data/rl_training_v2/`: loads a given model
checkpoint and reads completed sessions' telemetry; never writes anything
under that tree. The output path is always a caller-supplied argument --
this script never decides for itself where the OFFICIAL calibration
record lives (that decision belongs to whoever invokes it with
`--official`, once session_047 exists).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoscaler.src.rl_env.action_selection import compute_margin_stats, extract_action_probabilities  # noqa: E402

DEFAULT_SESSIONS_DIR = PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final_min2" / "sessions"
DEFAULT_N_SESSIONS = 5
DEFAULT_PERCENTILE = 10.0
DEFAULT_CAP = 0.05
DEFAULT_FLOOR = 0.0


def find_completed_sessions(sessions_dir: Path) -> List[Path]:
    """A session counts as completed/verified for calibration purposes
    only if it has BOTH `ppo_model.zip` and `metadata.json` (the same pair
    `train_ppo_final_v2.py` writes only once a session has been fully
    verified) -- an in-progress session (telemetry only, no checkpoint
    yet) is never included. Sorted ascending by session index."""
    if not sessions_dir.exists():
        return []
    candidates = []
    for child in sorted(sessions_dir.iterdir()):
        if child.is_dir() and (child / "ppo_model.zip").is_file() and (child / "metadata.json").is_file():
            candidates.append(child)
    return sorted(candidates, key=lambda p: p.name)


def load_last_n_sessions_observations(sessions_dir: Path, n: int) -> "tuple[List[List[float]], List[str]]":
    """Returns (normalized_observation vectors, session names used) from
    the LAST `n` completed sessions' `step_telemetry.jsonl` (fewer than
    `n` if fewer completed sessions exist yet -- never an error; the
    caller/output record discloses the shortfall explicitly)."""
    completed = find_completed_sessions(sessions_dir)
    used = completed[-n:] if n > 0 else []
    observations: List[List[float]] = []
    session_names: List[str] = []
    for session_dir in used:
        telemetry_path = session_dir / "step_telemetry.jsonl"
        if not telemetry_path.exists():
            continue
        session_names.append(session_dir.name)
        with open(telemetry_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                obs = row.get("normalized_observation")
                if obs is not None:
                    observations.append(obs)
    return observations, session_names


def _percentile(values: List[float], percentile: float) -> float:
    """A dependency-free percentile (linear interpolation, numpy's default
    method) -- avoids an unnecessary numpy import for a single statistic."""
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    frac = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac


def _clamp(value: float, floor: float, cap: float) -> float:
    return min(cap, max(floor, value))


def compute_epsilon(
    model: Any,
    observations: List[List[float]],
    percentile: float = DEFAULT_PERCENTILE,
    cap: float = DEFAULT_CAP,
    floor: float = DEFAULT_FLOOR,
    probability_extractor: Callable[[Any, List[float]], Any] = extract_action_probabilities,
) -> Dict[str, Any]:
    """Pure computation, model/extractor both injectable (tests never need
    a real SB3 checkpoint)."""
    if not observations:
        raise ValueError("no observations to calibrate against -- at least one completed session is required")

    margins = [compute_margin_stats(probability_extractor(model, obs)).margin for obs in observations]
    raw_percentile_value = _percentile(margins, percentile)
    epsilon = _clamp(raw_percentile_value, floor, cap)
    tie_fraction = sum(1 for m in margins if m <= epsilon) / len(margins)

    return {
        "n_observations": len(margins),
        "percentile": percentile,
        "raw_percentile_value": raw_percentile_value,
        "cap": cap,
        "floor": floor,
        "epsilon": epsilon,
        "tie_fraction_at_epsilon": tie_fraction,
        "margin_min": min(margins),
        "margin_max": max(margins),
        "margin_mean": sum(margins) / len(margins),
    }


def run_calibration(
    model_path: Path,
    sessions_dir: Path = DEFAULT_SESSIONS_DIR,
    n_sessions: int = DEFAULT_N_SESSIONS,
    percentile: float = DEFAULT_PERCENTILE,
    cap: float = DEFAULT_CAP,
    floor: float = DEFAULT_FLOOR,
    official: bool = False,
    model_loader: Optional[Callable[[Path], Any]] = None,
    probability_extractor: Callable[[Any, List[float]], Any] = extract_action_probabilities,
) -> Dict[str, Any]:
    if model_loader is None:
        from autoscaler.src.rl_env.safe_inference import load_ppo_model_safe

        model_loader = load_ppo_model_safe

    model = model_loader(model_path)
    if model is None:
        raise RuntimeError(f"failed to load model at {model_path}")

    observations, sessions_used = load_last_n_sessions_observations(sessions_dir, n_sessions)
    stats = compute_epsilon(model, observations, percentile, cap, floor, probability_extractor)

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official": official,
        "model_path": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "sessions_dir": str(sessions_dir),
        "sessions_requested": n_sessions,
        "sessions_used": sessions_used,
        "sessions_used_count": len(sessions_used),
        "fewer_than_requested_sessions_disclosed": len(sessions_used) < n_sessions,
        **stats,
    }
    return record


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sessions-dir", default=str(DEFAULT_SESSIONS_DIR))
    parser.add_argument("--n-sessions", type=int, default=DEFAULT_N_SESSIONS)
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument("--cap", type=float, default=DEFAULT_CAP)
    parser.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--official", action="store_true",
        help="Mark this record as the OFFICIAL evaluation calibration. Omit for a proof/test run.",
    )
    args = parser.parse_args(argv)

    record = run_calibration(
        Path(args.model_path), Path(args.sessions_dir), args.n_sessions, args.percentile,
        args.cap, args.floor, args.official,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
