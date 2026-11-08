"""demo_ui/fixtures_v2.py -- deterministic offline DEMO data for the V2
(multi-metric autoscaler) UI.

Produces run records in EXACTLY the same dict shape
`scripts/run_demo_episode_v2.py` writes into `experiment_store` for a real
LIVE run (see that script's `_to_status_sample()`/`run_demo_episode_v2()`)
-- so `views/live_details_v2.py` has ONE rendering code path for both
fixture and live data, never two parallel implementations. Every record
here is clearly DEMO DATA (`data_source: "DEMO"`) and must never be
mistaken for real scientific evidence (CORR-60/61's own evaluation
evidence discipline).

Deterministic and seeded (DEMO_SEED_V2=4242, distinct from V1's
DEMO_SEED=42) so the same inputs always reproduce the same fixture.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List

DEMO_SEED_V2 = 4242
DECISION_INTERVAL_S = 30.0
N_DECISIONS = 16  # matches the ~8 min v2_demo profiles (480s / 30s)

ARMS_V2 = ("ppo_v2", "hpa_v2_cpu_memory")
PROFILES_V2 = ("demo_cpu_short", "demo_memory_short", "demo_mixed_short")

ACTION_NAMES = ("SCALE_DOWN", "HOLD", "SCALE_UP")


def workload_hash_v2(profile_name: str, seed: int) -> str:
    payload = f"v2:{profile_name}:{seed}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _rps_curve(profile_name: str, i: int, n: int) -> float:
    frac = i / max(n - 1, 1)
    if profile_name == "demo_cpu_short":
        return 5.0 if frac < 0.5 else 18.0
    if profile_name == "demo_memory_short":
        return 3.0 + frac * 15.0 if frac < 0.5 else 18.0 - (frac - 0.5) * 30.0
    return 18.0 if (i % 5) == 0 else 4.0  # demo_mixed_short: short bursts


def _build_ppo_decision(rng: random.Random, replicas: int, action_name: str) -> Dict[str, Any]:
    probs = {"SCALE_DOWN": 0.1, "HOLD": 0.7, "SCALE_UP": 0.2}
    if action_name == "SCALE_UP":
        probs = {"SCALE_DOWN": 0.05, "HOLD": 0.25, "SCALE_UP": 0.70}
    elif action_name == "SCALE_DOWN":
        probs = {"SCALE_DOWN": 0.65, "HOLD": 0.30, "SCALE_UP": 0.05}
    jitter = rng.uniform(-0.03, 0.03)
    probs = {k: max(0.0, v + jitter) for k, v in probs.items()}
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}

    stressed = action_name == "SCALE_UP"
    return {
        "action_probabilities": probs,
        "requested_action_name": action_name,
        "raw_argmax_action_name": action_name,
        "safe_selected_action_name": action_name,
        "effective_action_name": action_name,
        "override_reason": "none",
        "tie_set": None,
        "tie_rule_applied": None,
        "stressed": stressed,
        "stress_reasons": ("p95_latency_ms >= sla.p95_target_ms",) if stressed else (),
        "epsilon": 0.03,
    }


def build_fixture_v2_run(arm: str, profile_name: str, seed: int = DEMO_SEED_V2) -> Dict[str, Any]:
    """One full run record, in the SAME shape experiment_store stores for a
    real V2 LIVE run (see run_demo_episode_v2.py's status-file contract)."""
    if arm not in ARMS_V2:
        raise ValueError(f"arm must be one of {ARMS_V2}, got {arm!r}")
    if profile_name not in PROFILES_V2:
        raise ValueError(f"profile_name must be one of {PROFILES_V2}, got {profile_name!r}")

    rng = random.Random(f"{seed}:{arm}:{profile_name}")
    samples: List[Dict[str, Any]] = []
    replicas = 2
    for i in range(N_DECISIONS):
        rps = _rps_curve(profile_name, i, N_DECISIONS)
        p95 = 150.0 + rps * 12.0 + rng.uniform(-20, 20)
        error_rate = 0.0 if p95 < 900 else min(0.05, (p95 - 900) / 5000.0)
        cpu_per_replica = min(1.0, 0.15 + rps / replicas * 0.03)
        memory_mib = 60.0 + rps * 1.2 + rng.uniform(-5, 5)

        if p95 >= 1000 and replicas < 8:
            action_name = "SCALE_UP"
            replicas = min(8, replicas + 1)
        elif p95 < 300 and replicas > 2 and rng.random() < 0.3:
            action_name = "SCALE_DOWN"
            replicas = max(2, replicas - 1)
        else:
            action_name = "HOLD"

        sample: Dict[str, Any] = {
            "decision_index": i, "time_s": i * DECISION_INTERVAL_S, "t_s": i * DECISION_INTERVAL_S,
            "timestamp_utc": "demo-fixture", "profile": profile_name, "arm_id": arm,
            "normalized_observation": [round(rng.random(), 3) for _ in range(11)],
            "raw_observation": {
                "cpu_per_replica": round(cpu_per_replica, 3), "memory_per_replica": round(memory_mib * 1_000_000, 0),
                "request_rate_norm": round(rps, 2), "p95_latency_norm": round(p95, 1), "error_rate": round(error_rate, 4),
            },
            "sample_valid": True,
            "p95_latency_ms": round(p95, 1), "error_rate": round(error_rate, 4), "request_rate_rps": round(rps, 2),
            "desired_replicas": replicas, "ready_replicas": replicas, "max_per_pod_memory_mib": round(memory_mib, 1),
            "pod_restart_counts": {f"demo-app-{j}": 0 for j in range(replicas)},
            "hpa_status": None if arm == "ppo_v2" else {
                "hpa_name": "demo-app-hpa-v2", "cpu_target_pct": 260.0, "current_cpu_pct": round(cpu_per_replica * 100, 1),
                "memory_target_pct": 120.0, "current_memory_pct": round(memory_mib / 64.0 * 100, 1),
            },
            "ppo_decision": _build_ppo_decision(rng, replicas, action_name) if arm == "ppo_v2" else None,
        }
        samples.append(sample)

    workload_hash = workload_hash_v2(profile_name, seed)
    return {
        "run_id": f"fixture_v2_{arm}_{profile_name}",
        "started_at": 0.0, "completed_at": N_DECISIONS * DECISION_INTERVAL_S,
        "data_source": "DEMO", "generation": "v2",
        "controller": arm, "profile_name": profile_name,
        "workload_hash": workload_hash, "workload_seed": seed, "workload_duration_s": N_DECISIONS * DECISION_INTERVAL_S,
        "status": "completed", "stage": "COMPLETED",
        "summary": {
            "episode_id": f"demo_v2_fixture_{arm}_{profile_name}", "arm": arm, "profile": profile_name,
            "n_samples": len(samples), "demo_only": True, "samples": samples,
        },
    }


def build_all_fixture_v2_runs() -> Dict[str, Dict[str, Any]]:
    """One fixture run per (arm, profile) combination, keyed
    `f"{profile_name}_{arm}"` (same key style as `fixtures.build_all_fixture_summaries()`)."""
    return {
        f"{profile_name}_{arm}": build_fixture_v2_run(arm, profile_name)
        for arm in ARMS_V2 for profile_name in PROFILES_V2
    }
