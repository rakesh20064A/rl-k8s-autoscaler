"""conclusion_engine.py -- Deterministic, evidence-traceable conclusion
generator (sec.15). Every sentence it produces is derived from the
ComparisonResult passed in; nothing here hard-codes PPO (or HPA) as the
winner. Thresholds below are the only tunable "opinion" in this module and
are intentionally named/explicit rather than magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from demo_ui.models import (
    Conclusion,
    ConclusionCategory,
    ConclusionDimension,
    ComparisonResult,
    TrafficProfile,
)


@dataclass(frozen=True)
class ConclusionThresholds:
    """A difference smaller than `material_pct` is treated as "no material
    difference" for that dimension, not as a win for either side."""
    material_pct: float = 5.0


def _pct(a: float, b: float) -> float:
    """Signed percent change of b relative to a; 0.0 if a == 0."""
    if a == 0:
        return 0.0
    return ((b - a) / abs(a)) * 100.0


def _latency_dimension(result: ComparisonResult, thresholds: ConclusionThresholds) -> ConclusionDimension:
    p95 = next(m for m in result.metrics if m.name == "P95 Latency")
    diff = p95.pct_difference
    if abs(diff) < thresholds.material_pct:
        return ConclusionDimension("Latency", "No material P95 latency difference observed.", None)
    direction = "lower" if diff < 0 else "higher"
    return ConclusionDimension(
        "Latency", f"PPO P95 latency is {abs(diff):.1f}% {direction} than HPA.", diff < 0,
    )


def _sla_dimension(result: ComparisonResult, thresholds: ConclusionThresholds) -> ConclusionDimension:
    sla = next(m for m in result.metrics if m.name == "SLA Violations")
    if sla.hpa_value == 0 and sla.ppo_value == 0:
        return ConclusionDimension("SLA", "Neither controller incurred SLA violations.", None)
    diff = sla.pct_difference
    if abs(diff) < thresholds.material_pct:
        return ConclusionDimension("SLA", "No material difference in SLA violations observed.", None)
    direction = "fewer" if diff < 0 else "more"
    return ConclusionDimension(
        "SLA", f"PPO incurred {abs(diff):.1f}% {direction} SLA violations than HPA.", diff < 0,
    )


def _resource_dimension(result: ComparisonResult, thresholds: ConclusionThresholds) -> ConclusionDimension:
    if result.hpa_replica_seconds == 0:
        return ConclusionDimension("Resource efficiency", "No replica-second baseline available from HPA.", None)
    diff = _pct(result.hpa_replica_seconds, result.ppo_replica_seconds)
    if abs(diff) < thresholds.material_pct:
        return ConclusionDimension("Resource efficiency", "No material difference in replica-seconds observed.", None)
    direction = "fewer" if diff < 0 else "more"
    return ConclusionDimension(
        "Resource efficiency", f"PPO used {abs(diff):.1f}% {direction} replica-seconds than HPA.", diff < 0,
    )


def _scaling_behaviour_dimension(result: ComparisonResult) -> ConclusionDimension:
    hpa_ev, ppo_ev = result.hpa_scaling_events, result.ppo_scaling_events
    if hpa_ev == ppo_ev:
        return ConclusionDimension("Scaling behaviour", f"Both controllers issued {hpa_ev} scaling events.", None)
    fewer = "PPO" if ppo_ev < hpa_ev else "HPA"
    return ConclusionDimension(
        "Scaling behaviour",
        f"HPA issued {hpa_ev} scaling events vs PPO's {ppo_ev} ({fewer} scaled less often).",
        ppo_ev < hpa_ev,
    )


def generate_conclusion(
    result: ComparisonResult,
    traffic_profile: TrafficProfile,
    thresholds: ConclusionThresholds = ConclusionThresholds(),
    is_demo_data: bool = True,
) -> Conclusion:
    dimensions = [
        _latency_dimension(result, thresholds),
        _sla_dimension(result, thresholds),
        _resource_dimension(result, thresholds),
        _scaling_behaviour_dimension(result),
    ]

    favor_flags = [d.ppo_favored for d in dimensions if d.ppo_favored is not None]
    ppo_favored_count = sum(1 for f in favor_flags if f)
    hpa_favored_count = sum(1 for f in favor_flags if not f)

    if not favor_flags:
        category = ConclusionCategory.SIMILAR
        headline = (
            f"For this {traffic_profile.value.capitalize()} workload, no material difference was observed "
            f"between HPA and PPO according to the configured thresholds."
        )
    elif ppo_favored_count > 0 and hpa_favored_count > 0:
        category = ConclusionCategory.TRADE_OFF
        ppo_wins = [d.name for d in dimensions if d.ppo_favored]
        hpa_wins = [d.name for d in dimensions if d.ppo_favored is False]
        headline = (
            f"For this {traffic_profile.value.capitalize()} workload, PPO performed better on "
            f"{', '.join(ppo_wins)} while HPA performed better on {', '.join(hpa_wins)}, "
            f"indicating a trade-off rather than a uniform improvement."
        )
    elif ppo_favored_count > 0:
        category = ConclusionCategory.PPO_BETTER_OVERALL
        p95 = next(m for m in result.metrics if m.name == "P95 Latency")
        sla = next(m for m in result.metrics if m.name == "SLA Violations")
        res_diff = _pct(result.hpa_replica_seconds, result.ppo_replica_seconds) if result.hpa_replica_seconds else 0.0
        headline = (
            f"For this {traffic_profile.value.capitalize()} workload, PPO achieved "
            f"{abs(p95.pct_difference):.1f}% {'lower' if p95.pct_difference < 0 else 'higher'} P95 latency, "
            f"{abs(sla.pct_difference):.1f}% {'fewer' if sla.pct_difference < 0 else 'more'} SLA violations "
            f"and {abs(res_diff):.1f}% {'fewer' if res_diff < 0 else 'more'} replica-seconds than HPA."
        )
    else:
        category = ConclusionCategory.HPA_BETTER_OVERALL
        headline = (
            f"For this {traffic_profile.value.capitalize()} workload, HPA performed better than PPO "
            f"across the measured dimensions in this run."
        )

    return Conclusion(category=category, headline=headline, dimensions=dimensions, is_demo_data=is_demo_data)
