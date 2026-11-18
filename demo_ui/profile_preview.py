"""demo_ui/profile_preview.py -- request-rate-over-time preview for a demo
traffic profile, computed from the ACTUAL generated profile YAML (never a
hand-authored approximation), via the same parser the real demo episode
runner and loadgen use.

Used by the Experiment page's live preview chart AND (through
`demo_ui.fixtures_v2.rps_curve_from_profile`) to derive Sample data
(offline) fixtures for the same 36+ profiles -- one computation, two
callers, so the preview and the eventual run always agree on shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from loadgen.traffic_generator import parse_traffic_profile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_PROFILE_DIR = PROJECT_ROOT / "configs" / "traffic_profiles" / "v2_demo"


def resolve_demo_profile_path(profile_name: str) -> Path:
    return DEMO_PROFILE_DIR / f"{profile_name}.yaml"


def compute_rps_curve(profile_name: str, step_seconds: float = 5.0) -> Tuple[List[float], List[float]]:
    """Returns (timestamps, request_rate_rps) sampled every `step_seconds`
    across the profile's full declared duration. Raises whatever
    `parse_traffic_profile` raises for a missing/malformed profile --
    never silently substitutes an empty or fabricated curve."""
    profile = parse_traffic_profile(resolve_demo_profile_path(profile_name))
    total = profile.total_duration
    timestamps: List[float] = []
    rps: List[float] = []
    t = 0.0
    while t < total:
        rate, _, _ = profile.get_phase_and_target_rps(t)
        timestamps.append(round(t, 1))
        rps.append(round(rate, 2))
        t += step_seconds
    # Always include the final instant so the chart's last visible point
    # reflects the profile's true end-of-run rate, not a step short of it.
    final_rate, _, _ = profile.get_phase_and_target_rps(max(0.0, total - 0.01))
    timestamps.append(round(total, 1))
    rps.append(round(final_rate, 2))
    return timestamps, rps
