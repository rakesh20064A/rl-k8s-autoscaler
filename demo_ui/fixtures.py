"""fixtures.py -- Deterministic offline demo data (sec.3, 16-17).

Everything here is clearly DEMO DATA (RunSummary.is_demo_data=True) and must
never be mistaken for real scientific evidence. Generation is seeded and
deterministic so the same inputs always reproduce the same fixture, matching
the "fair traffic replay" concept the real experiment will eventually use
(sec.13) -- here it's just a stand-in hash, not a real schedule.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Dict, List, Tuple

from demo_ui.models import (
    Controller,
    EventMarker,
    ExperimentRecord,
    HpaSnapshot,
    LogEvent,
    LogLevel,
    LogSource,
    PodState,
    PodStatus,
    PpoAction,
    PpoDecision,
    PpoSnapshot,
    RunSummary,
    TrafficProfile,
)

DEMO_SEED = 42
DEMO_DURATION_S = 300.0
SAMPLE_INTERVAL_S = 10.0

TRAFFIC_PROFILE_DESCRIPTIONS = {
    TrafficProfile.CONSTANT: "Holds offered workload steady for the full duration -- a baseline against which scaling reactions can be isolated from demand change.",
    TrafficProfile.RAMP: "Gradually increases offered workload, making the scaling response easy to observe as it unfolds.",
    TrafficProfile.BURST: "Alternates short high-intensity bursts with quiet periods, testing reaction speed to sudden demand spikes.",
    TrafficProfile.STEP: "Jumps once from a low to a high plateau partway through, isolating the controller's response to a single step change.",
    TrafficProfile.PERIODIC: "Oscillates smoothly on a fixed cycle, testing whether the controller tracks a repeating pattern without over-scaling.",
    TrafficProfile.MIXED: "Combines ramp and burst dynamics in one run, exercising both gradual and sudden demand changes together.",
}


def schedule_hash(traffic_profile: TrafficProfile, seed: int, duration_s: float) -> str:
    """Stand-in for the real deterministic-traffic-schedule hash (sec.13).
    Deliberately simple/reproducible -- NOT a cryptographic provenance
    artifact, just enough to demonstrate "same schedule = VERIFIED" in the
    UI."""
    payload = f"{traffic_profile.value}:{seed}:{duration_s}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _rps_curve(profile: TrafficProfile, t: float, duration: float) -> float:
    if profile == TrafficProfile.CONSTANT:
        return 25.0
    if profile == TrafficProfile.RAMP:
        return 5.0 + (t / duration) * 45.0
    if profile == TrafficProfile.BURST:
        return 60.0 if (t % 90.0) < 20.0 else 10.0
    if profile == TrafficProfile.STEP:
        return 10.0 if t < duration / 2 else 40.0
    if profile == TrafficProfile.PERIODIC:
        return 25.0 + 20.0 * math.sin(2 * math.pi * t / 60.0)
    if profile == TrafficProfile.MIXED:
        return _rps_curve(TrafficProfile.RAMP, t, duration) * 0.5 + _rps_curve(TrafficProfile.BURST, t, duration) * 0.5
    return 20.0


def _build_run(
    controller: Controller,
    profile: TrafficProfile,
    seed: int = DEMO_SEED,
    duration_s: float = DEMO_DURATION_S,
) -> RunSummary:
    rng = random.Random(seed if controller == Controller.HPA else seed + 1)
    n = int(duration_s // SAMPLE_INTERVAL_S) + 1
    timestamps = [i * SAMPLE_INTERVAL_S for i in range(n)]

    rps = [_rps_curve(profile, t, duration_s) for t in timestamps]
    cpu = [min(260.0, r * (7.5 if controller == Controller.HPA else 6.5)) for r in rps]

    # Reactive HPA: replicas lag demand by one sample and overshoot slightly.
    # Proactive PPO (fixture only): tracks demand more closely, fewer flaps.
    replicas: List[int] = []
    current = 1
    for i, c in enumerate(cpu):
        target_util = 60.0
        if controller == Controller.HPA:
            desired = max(1, min(8, math.ceil((c / target_util) * current))) if current else 1
            desired = max(1, min(8, round(c / 65.0) + 1))
        else:
            desired = max(1, min(8, round(c / 80.0) + 1))
        current = desired
        replicas.append(current)

    latency_avg = [40.0 + max(0.0, cpu[i] - 100.0) * 0.6 + rng.uniform(-2, 2) for i in range(n)]
    latency_p95 = [v * 1.8 for v in latency_avg]
    latency_p99 = [v * 2.4 for v in latency_avg]

    total_requests = int(sum(rps) * SAMPLE_INTERVAL_S)
    sla_violations = sum(1 for v in latency_p95 if v > 300.0)
    errors = max(0, int(sla_violations * 0.4))
    timeouts = max(0, int(sla_violations * 0.15))
    successful = max(0, total_requests - errors - timeouts)

    events: List[EventMarker] = []
    for i in range(1, n):
        if replicas[i] != replicas[i - 1]:
            label = "SCALE_UP" if replicas[i] > replicas[i - 1] else "SCALE_DOWN"
            category = f"{controller.value}_{label}"
            events.append(EventMarker(timestamp=timestamps[i], label=label, category=category))
            events.append(EventMarker(timestamp=timestamps[i] + 1.0, label="replica change", category="DEPLOYMENT_REPLICA_CHANGE"))
            events.append(EventMarker(timestamp=timestamps[i] + 4.0, label="pod Ready", category="POD_READY"))

    logs: List[LogEvent] = []
    for i, t in enumerate(timestamps):
        logs.append(LogEvent(t, LogSource.TRAFFIC, LogLevel.INFO, "RPS_SAMPLE", f"RPS -> {rps[i]:.1f}", {"rps": rps[i]}))
        logs.append(LogEvent(t + 0.2, LogSource.PROMETHEUS, LogLevel.INFO, "CPU_SAMPLE", f"CPU -> {cpu[i]:.0f}%", {"cpu_pct": cpu[i]}))
    for ev in events:
        src = LogSource.PPO if controller == Controller.PPO and "SCALE" in ev.category else LogSource.HPA
        if ev.category == "DEPLOYMENT_REPLICA_CHANGE":
            src = LogSource.KUBERNETES
        elif ev.category == "POD_READY":
            src = LogSource.KUBERNETES
        logs.append(LogEvent(ev.timestamp, src, LogLevel.INFO, ev.category, ev.label, {}))
    logs.sort(key=lambda e: e.timestamp)

    pods = [
        PodStatus(name=f"demo-app-{i}", state=PodState.READY if i < replicas[-1] else PodState.TERMINATING)
        for i in range(max(replicas[-1], 1))
    ]

    hpa_snapshot = None
    ppo_snapshot = None
    if controller == Controller.HPA:
        hpa_snapshot = HpaSnapshot(
            cpu_target_pct=260.0, current_cpu_pct=cpu[-1], current_replicas=replicas[-1],
            desired_replicas=replicas[-1], min_replicas=1, max_replicas=8,
            latest_scaling_event=events[-1].label if events else None,
        )
    else:
        decisions = []
        prev = 1
        for i, t in enumerate(timestamps):
            if i == 0 or replicas[i] == prev:
                continue
            action = PpoAction.SCALE_UP if replicas[i] > prev else PpoAction.SCALE_DOWN
            decisions.append(PpoDecision(
                timestamp=t, observation_summary=f"cpu={cpu[i]:.0f}% rps={rps[i]:.1f}",
                replicas_before=prev, action=action, replicas_after=replicas[i],
                reward=round(rng.uniform(-1.0, 1.0), 3),
            ))
            prev = replicas[i]
        ppo_snapshot = PpoSnapshot(
            latest_action=decisions[-1].action if decisions else PpoAction.KEEP,
            latest_reward=decisions[-1].reward if decisions else 0.0,
            observation_timestamp=timestamps[-1],
            decisions=decisions,
        )

    return RunSummary(
        controller=controller, traffic_profile=profile, seed=seed, duration_s=duration_s,
        schedule_hash=schedule_hash(profile, seed, duration_s),
        timestamps=timestamps, replicas=replicas, cpu_pct=cpu, memory_pct=[c * 0.4 for c in cpu],
        rps=rps, latency_avg_ms=latency_avg, latency_p95_ms=latency_p95, latency_p99_ms=latency_p99,
        successful_requests=successful, errors=errors, timeouts=timeouts, sla_violations=sla_violations,
        events=events, logs=logs, pods=pods, hpa=hpa_snapshot, ppo=ppo_snapshot, is_demo_data=True,
    )


def build_fixture_pair(profile: TrafficProfile, seed: int = DEMO_SEED, duration_s: float = DEMO_DURATION_S) -> Tuple[RunSummary, RunSummary]:
    """Returns (hpa_run, ppo_run) sharing the same seed/duration/profile, so
    schedule_hash matches -- demonstrating 'same traffic verified' (sec.13)."""
    return _build_run(Controller.HPA, profile, seed, duration_s), _build_run(Controller.PPO, profile, seed, duration_s)


def build_all_fixture_summaries() -> Dict[str, RunSummary]:
    summaries: Dict[str, RunSummary] = {}
    for profile in TrafficProfile:
        hpa_run, ppo_run = build_fixture_pair(profile)
        summaries[f"{profile.value}_hpa"] = hpa_run
        summaries[f"{profile.value}_ppo"] = ppo_run
    return summaries


def build_experiment_history() -> List[ExperimentRecord]:
    records = []
    for i, profile in enumerate(TrafficProfile):
        records.append(ExperimentRecord(
            experiment_id=f"comparison_{i + 1:03d}",
            traffic_profile=profile,
            controllers=[Controller.HPA, Controller.PPO],
            timestamp_label=f"2026-10-0{(i % 9) + 1} 14:{(i * 7) % 60:02d}",
            status="COMPLETE",
            valid=True,
        ))
    return records
