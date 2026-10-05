"""comparison_engine.py -- Pure, offline comparison calculations (sec.14).

Consumes two RunSummary objects (HPA, PPO) and derives metrics. No
preference for either controller is hard-coded: every comparison is a
computed value from the inputs.
"""

from __future__ import annotations

from typing import List

from demo_ui.models import ComparisonMetric, ComparisonResult, RunSummary


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def replica_seconds(timestamps: List[float], replicas: List[int]) -> float:
    """Integrates replica count over time via the trapezoid-free
    "hold value until next sample" rule (replicas is a step function, not
    interpolated) -- i.e. sum(replicas[i] * (t[i+1] - t[i])) for all but the
    last sample. Requires len(timestamps) == len(replicas); returns 0.0 for
    fewer than 2 samples (no interval to integrate over)."""
    if len(timestamps) != len(replicas):
        raise ValueError("timestamps and replicas must be the same length")
    if len(timestamps) < 2:
        return 0.0
    total = 0.0
    for i in range(len(timestamps) - 1):
        dt = timestamps[i + 1] - timestamps[i]
        total += replicas[i] * dt
    return total


def count_scaling_events(replicas: List[int]) -> int:
    """Number of times the replica count actually changed between
    consecutive samples (not the number of samples)."""
    return sum(1 for a, b in zip(replicas, replicas[1:]) if a != b)


def compare(hpa: RunSummary, ppo: RunSummary) -> ComparisonResult:
    if hpa.traffic_profile != ppo.traffic_profile:
        raise ValueError("compare() requires both runs to use the same traffic_profile")
    if hpa.seed != ppo.seed:
        raise ValueError("compare() requires both runs to use the same seed")

    same_schedule_verified = hpa.schedule_hash == ppo.schedule_hash

    metrics = [
        ComparisonMetric("Average Latency", _avg(hpa.latency_avg_ms), _avg(ppo.latency_avg_ms), "ms", lower_is_better=True),
        ComparisonMetric("P95 Latency", _avg(hpa.latency_p95_ms), _avg(ppo.latency_p95_ms), "ms", lower_is_better=True),
        ComparisonMetric("P99 Latency", _avg(hpa.latency_p99_ms), _avg(ppo.latency_p99_ms), "ms", lower_is_better=True),
        ComparisonMetric("Successful Requests", float(hpa.successful_requests), float(ppo.successful_requests), "", lower_is_better=False),
        ComparisonMetric("Errors", float(hpa.errors), float(ppo.errors), "", lower_is_better=True),
        ComparisonMetric("Timeouts", float(hpa.timeouts), float(ppo.timeouts), "", lower_is_better=True),
        ComparisonMetric("SLA Violations", float(hpa.sla_violations), float(ppo.sla_violations), "", lower_is_better=True),
        ComparisonMetric("Average CPU", _avg(hpa.cpu_pct), _avg(ppo.cpu_pct), "%", lower_is_better=True),
    ]

    hpa_rs = replica_seconds(hpa.timestamps, hpa.replicas)
    ppo_rs = replica_seconds(ppo.timestamps, ppo.replicas)

    return ComparisonResult(
        metrics=metrics,
        hpa_replica_seconds=hpa_rs,
        ppo_replica_seconds=ppo_rs,
        hpa_avg_replicas=_avg([float(r) for r in hpa.replicas]),
        ppo_avg_replicas=_avg([float(r) for r in ppo.replicas]),
        hpa_peak_replicas=max(hpa.replicas) if hpa.replicas else 0,
        ppo_peak_replicas=max(ppo.replicas) if ppo.replicas else 0,
        hpa_scaling_events=count_scaling_events(hpa.replicas),
        ppo_scaling_events=count_scaling_events(ppo.replicas),
        same_schedule_verified=same_schedule_verified,
    )
