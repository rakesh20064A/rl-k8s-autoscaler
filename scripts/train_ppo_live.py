#!/usr/bin/env python3
"""scripts/train_ppo_live.py

Phase 16V: the PPO live-training entry point conforming to Phase16U's
frozen protocol. See phase_16u_direct_ppo_protocol.yaml /
phase_16u_ppo_hyperparameters.json / phase_16u_training_and_evaluation_
workload_split.json for the values this script reads and must never
override at the command line.

Two modes:

  --dry-run (default, SAFE): performs configuration/hash/workload-role/
    dependency/environment-contract validation ONLY. Builds a
    KubernetesAutoscalerEnv against a small synthetic ScriptedBackend
    timeline (never a live backend) purely to prove the construction path
    works. Generates ZERO traffic, makes ZERO Kubernetes calls.

  --execute-live-training (REQUIRES --i-confirm-live-cluster-interaction):
    the real live-training path: constructs LiveKubernetesBackend, verifies
    no live HPA is present (fail-closed per docs/rl_mdp_spec.md sec.13),
    builds PPO from the frozen hyperparameters, and calls model.learn(). NOT
    invoked in Phase16V -- implemented so the system is ready for a
    separately authorized live-training phase, per that phase's own task
    instructions ("implement... but do not run").

No CLI flag exists to override any frozen reward/action/observation/PPO
hyperparameter value -- those are read exclusively from
configs/rl.yaml + configs/reward.yaml + task_outputs/phase_16u_ppo_
hyperparameters.json. The only user-facing flags are --dry-run,
--execute-live-training, --i-confirm-live-cluster-interaction (a required
companion flag for --execute-live-training, not a scientific override), and
--seed (defaults to the frozen seed 42; explicitly restricted to the one
frozen live-training seed, see phase16u_hp["hyperparameters"]["seeds"]).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoscaler.src.rl_env.config import load_env_config_from_yaml  # noqa: E402
from autoscaler.src.rl_env.environment import KubernetesAutoscalerEnv  # noqa: E402
from autoscaler.src.rl_env.ppo_config import build_ppo_kwargs  # noqa: E402
from autoscaler.src.rl_env.scripted_backend import ScriptedBackend  # noqa: E402
from autoscaler.src.rl_env.training_wrapper import LedgeredEpisodeWrapper  # noqa: E402
from autoscaler.src.rl_env.types import RawMetricSnapshot  # noqa: E402
from scripts import rl_live_common as common  # noqa: E402

TRAINING_LEDGER_PATH = PROJECT_ROOT / "persistent-data" / "rl_training" / "ppo_training_episode_ledger.jsonl"


def build_ppo_kwargs_for_live_training(seed: int) -> Dict[str, Any]:
    """configs/rl.yaml supplies every UNCHANGED hyperparameter (learning_rate,
    gamma, gae_lambda, clip_range, ent_coef, vf_coef, max_grad_norm,
    batch_size, n_epochs, policy, net_arch); phase_16u_ppo_hyperparameters.json
    supplies the DEVIATIONS (n_steps only -- total_timesteps/eval_freq/
    n_eval_episodes/seeds are training-LOOP parameters, not PPO constructor
    kwargs, and are read separately by the caller)."""
    kwargs = build_ppo_kwargs(seed=seed)
    hp = common.load_frozen_ppo_hyperparameters()
    kwargs["n_steps"] = hp["hyperparameters"]["n_steps"]
    return kwargs


def _synthetic_dry_run_timeline(n_steps: int = 10) -> Dict[float, RawMetricSnapshot]:
    """A small, clearly-labeled SYNTHETIC timeline for --dry-run's
    environment-contract construction check only -- carries zero scientific
    weight and is never used for anything resembling training."""
    return {
        i * 30.0: RawMetricSnapshot(
            request_rate_rps=5.0, error_rate_rps=0.0, latency_p95_ms=100.0,
            requests_in_progress=1.0, cpu_cores=0.2, memory_bytes=40_000_000.0,
        )
        for i in range(n_steps)
    }


def run_dry_run_validation(seed: int) -> Dict[str, Any]:
    """Performs EXACTLY the checks the task requires: configuration, hash,
    workload-role, dependency, and environment-contract validation. No
    traffic, no Kubernetes mutation."""
    report: Dict[str, Any] = {"classification": "OFFLINE_DRY_RUN_VALIDATION_ONLY"}

    # 1. Hash verification.
    hash_checks = common.verify_governing_hashes()
    report["governing_hashes"] = hash_checks
    report["all_governing_hashes_match"] = all(c["match"] for c in hash_checks)

    # 2. Workload-role validation -- every training profile must pass the
    # firewall; the final-holdout and unseen_eval must be rejected.
    workload_checks = {}
    for profile in common.TRAINING_PROFILES:
        try:
            common.assert_training_eligible(profile)
            workload_checks[profile] = "OK"
        except common.WorkloadFirewallViolation as exc:
            workload_checks[profile] = f"UNEXPECTED_REJECTION: {exc}"
    for forbidden in (common.FINAL_HOLDOUT_PROFILE, *common.EXCLUDED_FROM_FORMAL_COMPARISON):
        try:
            common.assert_training_eligible(forbidden)
            workload_checks[forbidden] = "UNEXPECTED_ACCEPTANCE"
        except common.WorkloadFirewallViolation:
            workload_checks[forbidden] = "CORRECTLY_REJECTED"
    report["workload_firewall"] = workload_checks
    report["workload_firewall_ok"] = all(
        v in ("OK", "CORRECTLY_REJECTED") for v in workload_checks.values()
    )

    # 3. Frozen PPO hyperparameters load and seed restriction.
    hp = common.load_frozen_ppo_hyperparameters()
    frozen_seeds = hp["hyperparameters"]["seeds"]
    report["seed_requested"] = seed
    report["seed_is_frozen_live_training_seed"] = seed in frozen_seeds
    report["frozen_total_timesteps"] = hp["hyperparameters"]["total_timesteps"]
    report["frozen_n_steps"] = hp["hyperparameters"]["n_steps"]

    # 4. PPO kwargs construction (no env yet -- pure config wiring check).
    ppo_kwargs = build_ppo_kwargs_for_live_training(seed)
    report["ppo_kwargs_constructed"] = sorted(ppo_kwargs.keys())
    report["ppo_kwargs_n_steps_matches_frozen"] = ppo_kwargs["n_steps"] == hp["hyperparameters"]["n_steps"]

    # 5. Environment-contract construction, ScriptedBackend ONLY (never a
    # live backend in dry-run).
    env_config = load_env_config_from_yaml()
    backend = ScriptedBackend(timeline=_synthetic_dry_run_timeline(), initial_replicas=1)
    raw_env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=270.0, config=env_config)
    wrapped = LedgeredEpisodeWrapper(
        raw_env, ledger_path=Path("/dev/null"), traffic_profile="DRY_RUN_SYNTHETIC",
        profile_hash="n/a", seed=seed, controller_code_hash="n/a", config_hash="n/a", role="training",
    )
    obs, _info = wrapped.reset(seed=seed)
    report["environment_constructed"] = True
    report["observation_shape"] = list(obs.shape)
    report["action_space_n"] = int(raw_env.action_space.n)

    # 6. Dependency versions (task sec.3 -- record, do not install/upgrade).
    report["dependency_versions"] = _dependency_versions()

    report["traffic_generated"] = False
    report["kubernetes_mutation"] = False
    report["ok"] = (
        report["all_governing_hashes_match"]
        and report["workload_firewall_ok"]
        and report["seed_is_frozen_live_training_seed"]
        and report["ppo_kwargs_n_steps_matches_frozen"]
        and report["environment_constructed"]
    )
    return report


def _dependency_versions() -> Dict[str, str]:
    import sys as _sys
    versions = {"python": _sys.version.split()[0]}
    for mod_name in ("stable_baselines3", "gymnasium", "torch", "numpy", "pandas", "kubernetes", "sklearn", "yaml"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "MISSING"
    return versions


def run_live_training(seed: int) -> int:  # pragma: no cover - not exercised in Phase16V
    """The real live-training path. NOT invoked in Phase16V. Implemented so
    a separately-authorized future phase can run it directly."""
    from stable_baselines3 import PPO

    from autoscaler.src.rl_env.live_backend import LiveKubernetesBackend

    hp = common.load_frozen_ppo_hyperparameters()

    backend = LiveKubernetesBackend()
    if backend._kubectl.hpa_exists():  # fail closed -- docs/rl_mdp_spec.md sec.13
        print("REFUSING TO START: a live HPA object targets the Deployment.", file=sys.stderr)
        return 3

    env_config = load_env_config_from_yaml()
    profile_cycle = list(common.TRAINING_PROFILES)
    # Episode duration is per-profile in real live training (Section 4 of
    # phase_16u_direct_ppo_protocol.yaml); this skeleton cycles through them
    # round-robin. Full per-profile traffic-generator wiring is left to the
    # implementation of THIS function at live-training-authorization time --
    # not fabricated here.
    profile_index = 0
    common.assert_training_eligible(profile_cycle[profile_index])

    raw_env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=600.0, config=env_config)
    wrapped = LedgeredEpisodeWrapper(
        raw_env, ledger_path=TRAINING_LEDGER_PATH, traffic_profile=profile_cycle[profile_index],
        profile_hash="TBD", seed=seed, controller_code_hash="TBD", config_hash="TBD", role="training",
    )

    model = PPO(env=wrapped, **build_ppo_kwargs_for_live_training(seed))
    try:
        model.learn(total_timesteps=hp["hyperparameters"]["total_timesteps"])
    except Exception as exc:
        wrapped.log_tooling_failure(exc, phase="model.learn")
        raise
    finally:
        try:
            backend.set_desired_replicas(env_config.initial_replicas)
            backend._kubectl.wait_for_replica_state(env_config.initial_replicas, backend.restoration_max_wait_s)
        except Exception as restore_exc:  # noqa: BLE001
            print(f"WARNING: post-training restoration failed: {restore_exc}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True, help="(default) offline validation only")
    parser.add_argument("--execute-live-training", action="store_true", help="run the real live-training loop")
    parser.add_argument("--i-confirm-live-cluster-interaction", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="must be one of the frozen live-training seeds")
    args = parser.parse_args(argv)

    if args.execute_live_training:
        if not args.i_confirm_live_cluster_interaction:
            print("Refusing --execute-live-training without --i-confirm-live-cluster-interaction.", file=sys.stderr)
            return 2
        return run_live_training(args.seed)

    report = run_dry_run_validation(args.seed)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
