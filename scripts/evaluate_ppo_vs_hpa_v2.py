#!/usr/bin/env python3
"""scripts/evaluate_ppo_vs_hpa_v2.py

V2 Phase 2 PART 5: a SEPARATE V2 inference path. Does NOT modify or import
from the frozen `scripts/evaluate_ppo_vs_hpa.py` -- this is a new, parallel
script reusing the same already-tested building blocks
(`LiveKubernetesBackend`, `KubernetesAutoscalerEnvV2`, `collectors.
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

V2 Phase 4C-1b (P1-2): also provides the V2 HPA evaluation arm
(`run_hpa_v2_episode()`, selected via `--arm hpa`), which acquires
`CONTROLLER_CPU_MEMORY_HPA`, applies/verifies `demo-app-hpa-v2` from
`k8s/hpa/hpa_v2_cpu_memory.yaml`, and fails closed (never auto-deletes)
if V1's `demo-app-hpa` is present instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoscaler.src.rl_env.config_v2 import (  # noqa: E402
    load_env_config_v2_from_yaml,
    load_safe_action_epsilon_from_v2_config,
)
from autoscaler.src.rl_env.controller_exclusivity import (  # noqa: E402
    CONTROLLER_CPU_MEMORY_HPA,
    CONTROLLER_PPO,
    V2_HPA_MANIFEST_PATH,
    V2_HPA_NAME,
    ControllerConflictError,
    acquire_controller,
    release_controller,
)
from autoscaler.src.rl_env.environment_v2 import KubernetesAutoscalerEnvV2  # noqa: E402
from autoscaler.src.rl_env.live_backend import LiveKubernetesBackend  # noqa: E402
from autoscaler.src.rl_env.safe_inference import load_ppo_model_safe  # noqa: E402
from autoscaler.src.rl_env.safe_ppo_controller import run_safe_ppo_step  # noqa: E402

V2_RESULTS_ROOT = PROJECT_ROOT / "results" / "evaluation_v2" / "raw"


def _map_raw_observation_to_stress_keys(raw_observation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """V2 Phase 4D-E / CORR-58: `environment_v2.py`'s own `info["raw_observation"]`
    uses `p95_latency_norm` (the observation-pipeline's own feature name);
    `safe_inference.infer_safe_action()`'s `decision_time_raw_observation`
    expects `p95_latency_ms` (the physical unit name, matching
    `RewardConfigV2.sla_p95_target_ms`'s own units). This is the ONE place
    that mapping happens -- a pure, defensive function (returns an
    all-missing dict, never raises, if `raw_observation` is None)."""
    raw = raw_observation or {}
    return {
        "p95_latency_ms": raw.get("p95_latency_norm"),
        "error_rate": raw.get("error_rate"),
        "cpu_per_replica": raw.get("cpu_per_replica"),
        "memory_per_replica": raw.get("memory_per_replica"),
    }


def run_dry_run_validation(epsilon: Optional[float] = None) -> Dict[str, Any]:
    """Offline-safe: no Kubernetes/Prometheus access, no traffic generated.
    Confirms the V2 safe-inference wiring is importable/constructible and
    reports the configuration this script WOULD run with -- mirrors the
    frozen V1 script's own `run_dry_run_validation()` in spirit (an
    always-safe default path), without duplicating any of its final-
    evaluation-specific plan/ledger logic (this is not a replacement for
    that frozen script).

    `epsilon=None` (the default) sources near_tie_epsilon/epsilon_status
    directly from configs/rl_v2.yaml's `safe_action` block (V2 Phase 4C-1
    P1-1 fix) rather than only action_selection.py's own Python-level
    default; an explicit value (e.g. from --epsilon) overrides the YAML
    value but keeps the YAML's status label, since only the value itself
    is caller-overridable."""
    report: Dict[str, Any] = {"classification": "OFFLINE_DRY_RUN_VALIDATION_ONLY"}
    env_config = load_env_config_v2_from_yaml()
    yaml_epsilon, epsilon_status = load_safe_action_epsilon_from_v2_config()
    report["decision_interval_s"] = env_config.decision_interval_s
    report["min_replicas"] = env_config.gating.min_replicas
    report["max_replicas"] = env_config.gating.max_replicas
    report["initial_replicas"] = env_config.initial_replicas
    report["near_tie_epsilon"] = yaml_epsilon if epsilon is None else epsilon
    report["epsilon_status"] = epsilon_status
    report["inference_path"] = "autoscaler.src.rl_env.safe_ppo_controller.run_safe_ppo_step"
    report["env_class"] = "KubernetesAutoscalerEnvV2"
    report["config_source"] = "configs/rl_v2.yaml + configs/reward_v2.yaml"
    report["controller_exclusivity_enforced"] = True
    report["traffic_generated"] = False
    report["kubernetes_mutation"] = False
    report["ok"] = True
    return report


def run_live_v2_episode(
    model_path: Path, episode_duration_s: float, episode_id: str, epsilon: Optional[float] = None,
) -> Dict[str, Any]:
    """Runs one V2-safe PPO inference episode against a real cluster.
    Requires `acquire_controller(CONTROLLER_PPO, ...)` to succeed (no HPA
    object present) before doing anything else -- fail-closed, per PART 4.
    Always releases the lock in a `finally` block, even on failure.

    `epsilon=None` sources near_tie_epsilon from configs/rl_v2.yaml (see
    `run_dry_run_validation`'s docstring); an explicit value overrides it."""
    if epsilon is None:
        epsilon, _epsilon_status = load_safe_action_epsilon_from_v2_config()
    env_config = load_env_config_v2_from_yaml()
    # V2 Phase 4C-1 (P1-2): this backend's HPA identity is explicitly the
    # V2 CPU+memory HPA (from the ONE canonical constant location,
    # controller_exclusivity.py), never the V1 default -- closes the R4F
    # "HPA identity is implicit" gap for the V2 inference path. The PPO arm
    # itself still requires BOTH HPAs absent (acquire_controller(
    # CONTROLLER_PPO, ...) below rejects ANY HPA object, V1 or V2, per
    # controller_exclusivity.detect_conflicting_hpas); this identity only
    # matters for this backend's own (currently unused in this script)
    # delete_hpa()/restore_hpa() calls, so they would act on demo-app-hpa-v2
    # if ever invoked, never silently fall back to demo-app-hpa.
    # V2 Phase 4D-B / CORR-55: explicit V2 replica floor (2), never
    # LiveKubernetesBackend's own V1-default min_replicas=1/initial_replicas=1.
    backend = LiveKubernetesBackend(
        hpa_name=V2_HPA_NAME, hpa_manifest_path=V2_HPA_MANIFEST_PATH,
        min_replicas=env_config.gating.min_replicas, initial_replicas=env_config.initial_replicas,
    )
    acquire_controller(CONTROLLER_PPO, backend._kubectl)
    try:
        model = load_ppo_model_safe(model_path)
        env = KubernetesAutoscalerEnvV2(backend=backend, episode_duration_s=episode_duration_s, config=env_config)

        obs, _info = env.reset()
        n_steps = int(round(episode_duration_s / env_config.decision_interval_s))
        decisions: List[Dict[str, Any]] = []

        # V2 Phase 4D-E / CORR-58: the safety-aware tie-break's STRESSED
        # determination needs the RAW state that produced `obs` -- i.e.
        # the PREVIOUS step's (or reset()'s) own `info["raw_observation"]`,
        # never this step's own post-action raw observation (which isn't
        # known until AFTER this step runs). Carried forward across
        # iterations; `reward_config` loaded once, outside the loop, so
        # every step reuses the same instance instead of re-reading YAML.
        decision_time_raw_observation = _map_raw_observation_to_stress_keys(_info.get("raw_observation"))
        reward_config = env_config.reward

        for _ in range(n_steps):
            obs, _reward, terminated, truncated, _info, decision_record = run_safe_ppo_step(
                env, model, obs, model_identifier=str(model_path), epsilon=epsilon,
                decision_time_raw_observation=decision_time_raw_observation, reward_config=reward_config,
            )
            decision_time_raw_observation = _map_raw_observation_to_stress_keys(_info.get("raw_observation"))
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


def run_hpa_v2_episode(episode_duration_s: float, episode_id: str) -> Dict[str, Any]:
    """V2 Phase 4C-1b (P1-2): the HPA arm of the V2 evaluation path --
    mirrors the frozen V1 script's own `run_hpa_episode()` in shape (reset,
    then advance the decision interval in a loop, no model/safe-inference
    involved since the HPA itself governs scaling), but for the V2
    CPU+memory HPA (`demo-app-hpa-v2`), never V1's `demo-app-hpa`.

    Acquires `CONTROLLER_CPU_MEMORY_HPA` BEFORE applying anything -- per
    `controller_exclusivity.detect_conflicting_hpas`, this fails closed
    (raises `ControllerConflictError`) if V1's `demo-app-hpa` (or any other
    unexpected HPA) is already present, and NEVER auto-deletes it. Only
    once acquisition succeeds does this apply/verify `demo-app-hpa-v2` from
    `V2_HPA_MANIFEST_PATH`. Cleanup (`finally`) re-applies the same V2
    manifest (idempotent `kubectl apply`) so the canonical V2 HPA is left
    in place regardless of how the episode ends, and always releases the
    lock."""
    env_config = load_env_config_v2_from_yaml()
    # V2 Phase 4D-B / CORR-55: explicit V2 replica floor (2), never
    # LiveKubernetesBackend's own V1-default min_replicas=1/initial_replicas=1.
    backend = LiveKubernetesBackend(
        hpa_name=V2_HPA_NAME, hpa_manifest_path=V2_HPA_MANIFEST_PATH,
        min_replicas=env_config.gating.min_replicas, initial_replicas=env_config.initial_replicas,
    )
    acquire_controller(CONTROLLER_CPU_MEMORY_HPA, backend._kubectl)
    try:
        backend._kubectl.restore_hpa()  # kubectl apply -f V2_HPA_MANIFEST_PATH (idempotent)
        applied_names = backend._kubectl.list_hpa_names()
        if V2_HPA_NAME not in applied_names:
            raise RuntimeError(f"failed to apply {V2_HPA_NAME!r}: live HPA names are {applied_names}")

        snapshot = backend.reset()  # canonical reset + stabilization; HPA governs scaling independently
        n_steps = int(round(episode_duration_s / env_config.decision_interval_s))
        samples: List[Dict[str, Any]] = [
            {"time_s": snapshot.time_s, "replicas_desired": snapshot.replicas.desired, "replicas_ready": snapshot.replicas.ready},
        ]
        for _ in range(n_steps):
            snapshot = backend.advance(env_config.decision_interval_s)
            samples.append(
                {"time_s": snapshot.time_s, "replicas_desired": snapshot.replicas.desired, "replicas_ready": snapshot.replicas.ready},
            )

        V2_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        out_path = V2_RESULTS_ROOT / f"{episode_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"episode_id": episode_id, "controller": "demo-app-hpa-v2", "samples": samples}, f, indent=2, default=str)

        return {"episode_id": episode_id, "n_samples": len(samples), "result_path": str(out_path)}
    finally:
        try:
            backend._kubectl.restore_hpa()  # re-apply demo-app-hpa-v2 on the way out, never deletes it
        finally:
            release_controller(CONTROLLER_CPU_MEMORY_HPA)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--i-confirm-live-cluster-interaction", action="store_true")
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "ppo" / "ppo_model.zip"))
    parser.add_argument("--episode-duration-s", type=float, default=600.0)
    parser.add_argument("--episode-id", default="v2_inference_episode")
    parser.add_argument(
        "--epsilon", type=float, default=None,
        help="Override near_tie_epsilon; defaults to configs/rl_v2.yaml's safe_action.near_tie_epsilon.",
    )
    parser.add_argument(
        "--arm", choices=("ppo", "hpa"), default="ppo",
        help="Which V2 evaluation arm to run under --execute: the PPO safe-inference arm (requires BOTH HPAs "
             "absent), or the demo-app-hpa-v2 HPA arm (applies/verifies the V2 manifest).",
    )
    args = parser.parse_args(argv)

    if args.execute:
        if not args.i_confirm_live_cluster_interaction:
            print("Refusing --execute without --i-confirm-live-cluster-interaction.", file=sys.stderr)
            return 2
        try:
            if args.arm == "hpa":
                result = run_hpa_v2_episode(args.episode_duration_s, args.episode_id)
            else:
                result = run_live_v2_episode(
                    Path(args.model_path), args.episode_duration_s, args.episode_id, args.epsilon,
                )
        except ControllerConflictError as exc:
            print(f"Refusing to start {args.arm.upper()} arm: {exc}", file=sys.stderr)
            return 3
        print(json.dumps(result, indent=2, default=str))
        return 0

    report = run_dry_run_validation(args.epsilon)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
