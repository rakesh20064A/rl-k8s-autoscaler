#!/usr/bin/env python3
"""scripts/evaluate_ppo_vs_hpa_v2.py

V2 Phase 2 PART 5: a SEPARATE V2 inference path. Does NOT modify or import
from the frozen `scripts/evaluate_ppo_vs_hpa.py` -- this is a new, parallel
script reusing the same already-tested building blocks
(`LiveKubernetesBackend`, `KubernetesAutoscalerEnv`, `collectors.
metrics_collector.PrometheusClient`, `scripts.traffic_orchestrator`) but
replacing V1's direct, unguarded `model.predict(obs, deterministic=True)`
call with the V2 safe path: `autoscaler.src.rl_env.safe_ppo_controller.
run_safe_ppo_step()`, which validates the observation, extracts the full
action-probability distribution, applies the near-tie-safe selector
(provisional, configurable `epsilon` -- NOT finalized), and degrades any
inference failure to HOLD with `policy_inference_failure`, never a silent
SCALE_DOWN.

CLASSIFICATION: like the frozen V1 script, this has a `--dry-run` path
(OFFLINE_VALIDATION_ONLY, the default) and a `--execute` path gated behind
an explicit confirmation flag. `--execute` is NOT invoked by this phase's
own work -- V2 Phase 2 performs no PPO training and no formal evaluation;
this script exists so a LATER, separately authorized phase has a real V2
inference path to run, with safety wiring already in place and tested.

Also enforces controller exclusivity (PART 4,
`autoscaler.src.rl_env.controller_exclusivity`): refuses to start a PPO
episode while any HPA object (CPU-only `demo-app-hpa` or the new CPU+memory
`demo-app-hpa-v2`) is present in the cluster, and records/releases the PPO
controller lock around the episode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoscaler.src.rl_env.action_selection import DEFAULT_NEAR_TIE_EPSILON  # noqa: E402
from autoscaler.src.rl_env.config import load_env_config_from_yaml  # noqa: E402
from autoscaler.src.rl_env.controller_exclusivity import (  # noqa: E402
    CONTROLLER_PPO,
    ControllerConflictError,
    acquire_controller,
    release_controller,
)
from autoscaler.src.rl_env.environment import KubernetesAutoscalerEnv  # noqa: E402
from autoscaler.src.rl_env.live_backend import LiveKubernetesBackend  # noqa: E402
from autoscaler.src.rl_env.safe_inference import load_ppo_model_safe  # noqa: E402
from autoscaler.src.rl_env.safe_ppo_controller import run_safe_ppo_step  # noqa: E402

V2_RESULTS_ROOT = PROJECT_ROOT / "results" / "evaluation_v2" / "raw"


def run_dry_run_validation(epsilon: float = DEFAULT_NEAR_TIE_EPSILON) -> Dict[str, Any]:
    """Offline-safe: no Kubernetes/Prometheus access, no traffic generated.
    Confirms the V2 safe-inference wiring is importable/constructible and
    reports the configuration this script WOULD run with -- mirrors the
    frozen V1 script's own `run_dry_run_validation()` in spirit (an
    always-safe default path), without duplicating any of its final-
    evaluation-specific plan/ledger logic (this is not a replacement for
    that frozen script)."""
    report: Dict[str, Any] = {"classification": "OFFLINE_DRY_RUN_VALIDATION_ONLY"}
    env_config = load_env_config_from_yaml()
    report["decision_interval_s"] = env_config.decision_interval_s
    report["near_tie_epsilon"] = epsilon
    report["epsilon_status"] = "provisional_unvalidated"
    report["inference_path"] = "autoscaler.src.rl_env.safe_ppo_controller.run_safe_ppo_step"
    report["controller_exclusivity_enforced"] = True
    report["traffic_generated"] = False
    report["kubernetes_mutation"] = False
    report["ok"] = True
    return report


def run_live_v2_episode(
    model_path: Path, episode_duration_s: float, episode_id: str, epsilon: float = DEFAULT_NEAR_TIE_EPSILON,
) -> Dict[str, Any]:
    """Runs one V2-safe PPO inference episode against a real cluster.
    Requires `acquire_controller(CONTROLLER_PPO, ...)` to succeed (no HPA
    object present) before doing anything else -- fail-closed, per PART 4.
    Always releases the lock in a `finally` block, even on failure."""
    backend = LiveKubernetesBackend()
    acquire_controller(CONTROLLER_PPO, backend._kubectl)
    try:
        model = load_ppo_model_safe(model_path)
        env_config = load_env_config_from_yaml()
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=episode_duration_s, config=env_config)

        obs, _info = env.reset()
        n_steps = int(round(episode_duration_s / env_config.decision_interval_s))
        decisions: List[Dict[str, Any]] = []

        for _ in range(n_steps):
            obs, _reward, terminated, truncated, _info, decision_record = run_safe_ppo_step(
                env, model, obs, model_identifier=str(model_path), epsilon=epsilon,
            )
            decisions.append(decision_record.to_dict())
            if terminated or truncated:
                break

        V2_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        out_path = V2_RESULTS_ROOT / f"{episode_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"episode_id": episode_id, "model_path": str(model_path), "decisions": decisions}, f, indent=2, default=str)

        return {"episode_id": episode_id, "n_decisions": len(decisions), "result_path": str(out_path)}
    finally:
        release_controller(CONTROLLER_PPO)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--i-confirm-live-cluster-interaction", action="store_true")
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "ppo" / "ppo_model.zip"))
    parser.add_argument("--episode-duration-s", type=float, default=600.0)
    parser.add_argument("--episode-id", default="v2_inference_episode")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_NEAR_TIE_EPSILON)
    args = parser.parse_args(argv)

    if args.execute:
        if not args.i_confirm_live_cluster_interaction:
            print("Refusing --execute without --i-confirm-live-cluster-interaction.", file=sys.stderr)
            return 2
        try:
            result = run_live_v2_episode(
                Path(args.model_path), args.episode_duration_s, args.episode_id, args.epsilon,
            )
        except ControllerConflictError as exc:
            print(f"Refusing to start PPO: {exc}", file=sys.stderr)
            return 3
        print(json.dumps(result, indent=2, default=str))
        return 0

    report = run_dry_run_validation(args.epsilon)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
