#!/usr/bin/env python3
"""scripts/smoke_test_ppo_sb3_compat.py

Phase 16V, task sec.9: an OFFLINE SB3/Gymnasium API-compatibility smoke
test. Uses `ScriptedBackend` (TEST_ONLY_SYNTHETIC_METRICS) exclusively --
never a live backend, never real Kubernetes/Prometheus.

*** THIS SCRIPT HAS ZERO SCIENTIFIC WEIGHT. ***
It proves only that `KubernetesAutoscalerEnv` satisfies the Gymnasium API
that `stable_baselines3.PPO` expects, and that `PPO.learn()` can execute a
handful of timesteps against it without raising. Any resulting "trained"
policy, reward curve, or loss value is a synthetic-data artifact and MUST
NEVER be cited as evidence about real autoscaling performance, PPO's
eventual live-trained behavior, or anything resembling a result from
phase_16u_direct_ppo_protocol.yaml's actual training budget. Running this
does not consume, advance, or count toward that frozen 12,288-step budget
in any way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SYNTHETIC_SMOKE_TEST_TIMESTEPS = 64  # deliberately tiny; API-compat only

ZERO_SCIENTIFIC_WEIGHT_BANNER = (
    "*** SYNTHETIC SMOKE TEST -- ZERO SCIENTIFIC WEIGHT. "
    "NOT PPO TRAINING EVIDENCE. DOES NOT COUNT TOWARD ANY FROZEN "
    "TRAINING BUDGET. ScriptedBackend only, no live Kubernetes. ***"
)


def run_smoke_test() -> Dict[str, Any]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    from autoscaler.src.rl_env.config import load_env_config_from_yaml
    from autoscaler.src.rl_env.environment import KubernetesAutoscalerEnv
    from autoscaler.src.rl_env.ppo_config import build_ppo_kwargs
    from autoscaler.src.rl_env.scripted_backend import ScriptedBackend
    from autoscaler.src.rl_env.types import RawMetricSnapshot

    report: Dict[str, Any] = {"banner": ZERO_SCIENTIFIC_WEIGHT_BANNER, "scientific_weight": "ZERO"}

    def synthetic_snapshot(**overrides):
        base = dict(
            request_rate_rps=5.0, error_rate_rps=0.0, latency_p95_ms=100.0,
            requests_in_progress=1.0, cpu_cores=0.2, memory_bytes=40_000_000.0,
        )
        base.update(overrides)
        return RawMetricSnapshot(**base)

    timeline = {i * 30.0: synthetic_snapshot() for i in range(200)}
    backend = ScriptedBackend(timeline=timeline, initial_replicas=1)
    env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=6000.0, config=load_env_config_from_yaml())

    # 1. Gymnasium environment checker.
    check_env(env, warn=True)  # raises on failure
    report["gymnasium_check_env_passed"] = True

    # 2. SB3 PPO construction using the FROZEN configs/rl.yaml hyperparameters
    #    (unchanged -- this smoke test does not need or use Phase16U's
    #    reduced live-training n_steps override; it is not live training).
    ppo_kwargs = build_ppo_kwargs(seed=1234)
    model = PPO(env=env, **ppo_kwargs)
    report["ppo_constructed"] = True
    report["policy_net_arch_wired"] = [
        layer for layer in model.policy.mlp_extractor.policy_net if hasattr(layer, "out_features")
    ][0].out_features == ppo_kwargs["policy_kwargs"]["net_arch"][0]

    # 3. Tiny synthetic rollout/"training" call -- proves the API surface
    #    works end to end (rollout collection, GAE, PPO update), nothing more.
    model.learn(total_timesteps=SYNTHETIC_SMOKE_TEST_TIMESTEPS)
    report["synthetic_learn_completed"] = True
    report["synthetic_timesteps"] = SYNTHETIC_SMOKE_TEST_TIMESTEPS

    # 4. Prove a trained-policy prediction call works (still synthetic).
    obs, _info = env.reset(seed=1)
    action, _state = model.predict(obs, deterministic=True)
    report["predict_call_succeeded"] = True
    report["predict_returned_valid_action"] = int(action) in (0, 1, 2)

    report["backend_used"] = "ScriptedBackend (TEST_ONLY_SYNTHETIC_METRICS)"
    report["live_kubernetes_touched"] = False
    report["traffic_generated"] = False
    report["counts_toward_frozen_training_budget"] = False
    report["ok"] = all([
        report["gymnasium_check_env_passed"],
        report["ppo_constructed"],
        report["synthetic_learn_completed"],
        report["predict_call_succeeded"],
        report["predict_returned_valid_action"],
    ])
    return report


def main() -> int:
    print(ZERO_SCIENTIFIC_WEIGHT_BANNER, file=sys.stderr)
    report = run_smoke_test()
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
