#!/usr/bin/env python3
"""scripts/train_ppo_live_v2.py

V2 Phase 2.2 PART 7/10: the explicit V2 PPO training entry point. Mirrors
`scripts/train_ppo_live.py`'s two-mode shape (--dry-run default SAFE,
--execute-live-training gated behind an explicit confirmation flag) but
wires EVERYTHING through the V2 namespace -- never the V1 one:

    V1 (frozen, untouched)              V2 (this script, exclusively)
    configs/rl.yaml                     configs/rl_v2.yaml
    configs/reward.yaml                 configs/reward_v2.yaml
    autoscaler.src.rl_env.environment   autoscaler.src.rl_env.environment_v2
    .KubernetesAutoscalerEnv            .KubernetesAutoscalerEnvV2
    reward.py                           reward_v2.py
    models/ppo/                         models/ppo_v2/
    persistent-data/rl_training/        persistent-data/rl_training_v2/

`--dry-run` (the default) performs ONLY: V2 config/contract validation, PPO
kwarg construction from rl_v2.yaml, V2-namespace path/destination
validation (fail-closed against any V1 path), training-profile-allowlist
validation, controller-exclusivity wiring validation (against a FAKE
kubectl, never a real cluster), and a no-learning integration check:
construct the V2 environment against a synthetic ScriptedBackend timeline,
reset, and step through a handful of SCRIPTED actions -- `PPO.learn()` is
never imported or called anywhere in this path (see
tests/unit/test_train_ppo_live_v2.py for an explicit proof of this).

V2 Phase 3A `run_pilot_training()`: the real, ONE-ROLLOUT live PPO pilot
path (`--execute-live-training`). Writes ONLY to
`persistent-data/rl_training_v2/pilot/` -- deliberately NEVER to
`models/ppo_v2/ppo_model_v2.zip` (that name is reserved for a later,
separately-accepted final V2 model; see `resolve_next_pilot_session_dir()`
vs `V2_MODEL_PATH`). Removes the canonical HPA for the duration of the pilot
and restores it (and the deployment's replica count, and the controller
lock) in `finally` blocks regardless of success, an SB3-level safety
termination (`scripts.v2_pilot_safety.PilotSafetyCallback`), or an
exception.

Fresh policy only: this script contains no code path that loads an
existing checkpoint (V1 or V2) before training -- `training_contract_v2.
assert_fresh_policy_or_justified()` is called explicitly, and would raise
on an attempted (not-yet-implemented) resume without an explicit, non-empty
justification string.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoscaler.src.rl_env.config_v2 import DEFAULT_RL_V2_YAML, load_env_config_v2_from_yaml  # noqa: E402
from autoscaler.src.rl_env.controller_exclusivity import (  # noqa: E402
    CONTROLLER_PPO,
    V2_HPA_MANIFEST_PATH,
    V2_HPA_NAME,
    ControllerConflictError,
    acquire_controller,
    release_controller,
)
from autoscaler.src.rl_env.environment_v2 import KubernetesAutoscalerEnvV2  # noqa: E402
from autoscaler.src.rl_env.ppo_config import build_ppo_kwargs  # noqa: E402
from autoscaler.src.rl_env.scripted_backend import ScriptedBackend  # noqa: E402
from autoscaler.src.rl_env.training_contract_v2 import (  # noqa: E402
    V1_MODEL_PATH,
    V1_TRAINING_LEDGER,
    V2_CHECKPOINT_DIR,
    V2_MODEL_PATH,
    V2_TRAINING_LEDGER,
    V2_TRAINING_PROFILE_ALLOWLIST,
    V1ArtifactProtectionError,
    assert_fresh_policy_or_justified,
    assert_v1_artifacts_untouched,
    build_v2_model_metadata,
    ensure_v2_training_directories,
)
from autoscaler.src.rl_env.types import RawMetricSnapshot  # noqa: E402


def _synthetic_dry_run_timeline(n_steps: int = 10) -> Dict[float, RawMetricSnapshot]:
    """SYNTHETIC-only timeline, mirrors train_ppo_live.py's own dry-run
    fixture in spirit -- carries zero scientific weight, used purely to
    prove the V2 environment construction/step path works end-to-end."""
    return {
        i * 30.0: RawMetricSnapshot(
            request_rate_rps=5.0, error_rate_rps=0.0, latency_p95_ms=100.0,
            requests_in_progress=1.0, cpu_cores=0.2, memory_bytes=40_000_000.0,
        )
        for i in range(n_steps)
    }


def validate_profile_allowlist(profiles: List[str]) -> Dict[str, str]:
    """Fail-closed: every requested profile must already be in
    V2_TRAINING_PROFILE_ALLOWLIST. Returns a per-profile OK/REJECTED report
    rather than raising on the first bad entry, so a dry run can see the
    full picture."""
    return {p: ("OK" if p in V2_TRAINING_PROFILE_ALLOWLIST else "REJECTED_NOT_ALLOWLISTED") for p in profiles}


def run_dry_run_validation(seed: int = 42) -> Dict[str, Any]:
    report: Dict[str, Any] = {"classification": "OFFLINE_DRY_RUN_VALIDATION_ONLY"}

    # 1. V2 config/contract validation.
    env_config_v2 = load_env_config_v2_from_yaml()
    report["controller_generation"] = env_config_v2.controller_generation
    report["workload_contract"] = env_config_v2.workload_contract
    report["reward_status"] = env_config_v2.reward_status
    report["v2_identity_ok"] = (
        env_config_v2.controller_generation == "v2"
        and env_config_v2.workload_contract == "multi_resource_v2"
    )

    # 2. PPO kwargs constructed from rl_v2.yaml (reused unmodified from
    # ppo_config.py -- its `ppo:` block schema is identical to rl.yaml's).
    ppo_kwargs = build_ppo_kwargs(rl_yaml_path=DEFAULT_RL_V2_YAML, seed=seed)
    report["ppo_kwargs_constructed"] = sorted(ppo_kwargs.keys())

    # 3. Fresh-policy enforcement (default path -- no justification given).
    assert_fresh_policy_or_justified(None)
    report["fresh_policy_enforced"] = True

    # 4. Namespace/destination fail-closed checks: V1 paths must be refused.
    v1_refused = {}
    for name, path in (("v1_model", V1_MODEL_PATH), ("v1_ledger", V1_TRAINING_LEDGER)):
        try:
            assert_v1_artifacts_untouched([path])
            v1_refused[name] = "UNEXPECTED_ACCEPTANCE"
        except V1ArtifactProtectionError:
            v1_refused[name] = "CORRECTLY_REJECTED"
    report["v1_destination_checks"] = v1_refused
    report["v1_destinations_fail_closed"] = all(v == "CORRECTLY_REJECTED" for v in v1_refused.values())

    assert_v1_artifacts_untouched([V2_MODEL_PATH, V2_TRAINING_LEDGER])  # must not raise
    report["v2_destinations_accepted"] = True

    # 5. Training profile allowlist.
    allowlist_report = validate_profile_allowlist(
        list(V2_TRAINING_PROFILE_ALLOWLIST) + ["some_unsupported_v1_profile"],
    )
    report["profile_allowlist_checks"] = allowlist_report
    report["profile_allowlist_ok"] = (
        all(allowlist_report[p] == "OK" for p in V2_TRAINING_PROFILE_ALLOWLIST)
        and allowlist_report["some_unsupported_v1_profile"] == "REJECTED_NOT_ALLOWLISTED"
    )

    # 6. Controller-exclusivity wiring (FAKE kubectl only -- no real cluster).
    class _FakeKubectlNoHpa:
        def list_hpa_names(self):
            return []

    class _FakeKubectlWithHpa:
        def list_hpa_names(self):
            return ["demo-app-hpa"]

    conflict_report = {}
    try:
        acquire_controller(CONTROLLER_PPO, _FakeKubectlNoHpa(), lock_path=Path("/tmp/.v2_dry_run_lock_unused.json"))
        conflict_report["no_hpa_present"] = "CORRECTLY_ACQUIRED"
        release_controller(CONTROLLER_PPO, lock_path=Path("/tmp/.v2_dry_run_lock_unused.json"))
    except ControllerConflictError:
        conflict_report["no_hpa_present"] = "UNEXPECTED_REJECTION"
    try:
        acquire_controller(CONTROLLER_PPO, _FakeKubectlWithHpa(), lock_path=Path("/tmp/.v2_dry_run_lock_unused2.json"))
        conflict_report["hpa_present"] = "UNEXPECTED_ACCEPTANCE"
    except ControllerConflictError:
        conflict_report["hpa_present"] = "CORRECTLY_REJECTED"
    report["controller_exclusivity_checks"] = conflict_report
    report["controller_exclusivity_ok"] = (
        conflict_report["no_hpa_present"] == "CORRECTLY_ACQUIRED" and conflict_report["hpa_present"] == "CORRECTLY_REJECTED"
    )

    # 7. No-learning integration check: construct V2 env (ScriptedBackend
    # only), reset, step through a few SCRIPTED actions, verify shapes.
    backend = ScriptedBackend(timeline=_synthetic_dry_run_timeline(), initial_replicas=1)
    env = KubernetesAutoscalerEnvV2(backend=backend, episode_duration_s=270.0, config=env_config_v2)
    obs, _info = env.reset(seed=seed)
    report["observation_shape"] = list(obs.shape)
    report["action_space_n"] = int(env.action_space.n)

    scripted_actions = [1, 1, 2, 1, 0]  # HOLD, HOLD, SCALE_UP, HOLD, SCALE_DOWN -- fixed, not model-sampled
    rewards = []
    for action in scripted_actions:
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    report["dry_run_rewards"] = rewards
    report["dry_run_rewards_computed"] = len(rewards) == len(scripted_actions)
    report["dry_run_used_reward_v2"] = info["controller_generation"] == "v2"

    # 8. Prepare (not populate) the V2 directory namespace.
    ensure_v2_training_directories()
    report["v2_directories_prepared"] = {
        "model_dir": str(V2_MODEL_PATH.parent), "checkpoint_dir": str(V2_CHECKPOINT_DIR),
        "ledger_path": str(V2_TRAINING_LEDGER),
    }

    # 9. Model metadata schema (PART 9) -- built, not written (no training occurred).
    metadata = build_v2_model_metadata(training_seed=seed, ppo_hyperparameters=ppo_kwargs)
    report["model_metadata_preview"] = metadata.to_dict()

    report["ppo_learn_called"] = False
    report["traffic_generated"] = False
    report["kubernetes_mutation"] = False
    report["ok"] = (
        report["v2_identity_ok"]
        and report["v1_destinations_fail_closed"]
        and report["v2_destinations_accepted"]
        and report["profile_allowlist_ok"]
        and report["controller_exclusivity_ok"]
        and report["dry_run_rewards_computed"]
        and report["dry_run_used_reward_v2"]
        and report["observation_shape"] == [11]
    )
    return report


PILOT_ROOT = PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "pilot"
PILOT_LEDGER_PATH = PILOT_ROOT / "ppo_pilot_episode_ledger_v2.jsonl"
PILOT_TRAFFIC_PROVENANCE_LEDGER = PILOT_ROOT / "traffic_provenance_ledger_v2.jsonl"

PILOT_N_STEPS = 256
PILOT_TOTAL_TIMESTEPS = 256


def resolve_next_pilot_session_dir(pilot_root: Path = PILOT_ROOT) -> Path:
    """V2 Phase 3B PART 3/5: retry-safe session allocation -- mirrors the
    project's own established `resolve_next_available_run_id()` pattern
    (scripts/traffic_orchestrator.py). Scans PERSISTED on-disk evidence for
    the first `session_NNN` directory that does not already exist, so a
    failed attempt's evidence (e.g. session_000 from V2 Phase 3A's
    physical attempt 1) is NEVER reused or overwritten -- each physical
    attempt gets its own, permanently distinct directory."""
    n = 0
    while (pilot_root / f"session_{n:03d}").exists():
        n += 1
    return pilot_root / f"session_{n:03d}"


def run_pilot_training(seed: int = 42) -> Dict[str, Any]:
    """V2 Phase 3B: the ONE-ROLLOUT live PPO pilot retry. Fails closed on
    an active HPA/controller conflict OR an unreachable Prometheus endpoint
    before doing anything else; ALL cleanup/restoration happens in
    `finally` blocks regardless of how the pilot ends (success, safety
    termination, or exception). Writes to a FRESH, retry-safe session
    directory every call -- never reuses or overwrites a prior attempt's
    evidence (see `resolve_next_pilot_session_dir`)."""
    import hashlib

    from scripts.v2_pilot_safety import PROMETHEUS_URL, verify_prometheus_internal_paths

    pilot_session_dir = resolve_next_pilot_session_dir()
    pilot_model_path = pilot_session_dir / "ppo_model.zip"
    pilot_metadata_path = pilot_session_dir / "metadata.json"
    pilot_metrics_path = pilot_session_dir / "training_metrics.json"
    pilot_telemetry_path = pilot_session_dir / "step_telemetry.jsonl"

    # V2 Phase 3B PART 2: test the ACTUAL internal Prometheus path (the
    # same PrometheusClient type LiveKubernetesBackend constructs, and the
    # exact drain_in_flight_requests() PromQL) BEFORE any HPA removal or
    # controller-lock acquisition. An external-only curl check would have
    # missed exactly the failure mode that invalidated physical attempt 1
    # (the EXTERNAL monitoring port-forward was healthy on 19090 while the
    # INTERNAL clients, still pointed at the unmodified 9090 default,
    # could not connect at all).
    prom_preflight = verify_prometheus_internal_paths(PROMETHEUS_URL)
    if not prom_preflight["all_ok"]:
        raise RuntimeError(
            f"Prometheus internal-path preflight failed -- refusing to start before any cluster "
            f"mutation: {prom_preflight}"
        )

    from stable_baselines3 import PPO

    from autoscaler.src.rl_env.live_backend import LiveKubernetesBackend
    from autoscaler.src.rl_env.training_wrapper import LedgeredEpisodeWrapper
    from scripts.v2_pilot_safety import PilotSafetyCallback, get_pod_health
    from scripts.v2_traffic_orchestrator import V2TrafficOrchestrator, V2TrafficSyncedWrapper

    assert_fresh_policy_or_justified(None)  # fresh by default; see module docstring
    if pilot_model_path == V2_MODEL_PATH:
        raise V1ArtifactProtectionError("pilot must never be written to the reserved final-model path")
    assert_v1_artifacts_untouched([pilot_model_path, PILOT_LEDGER_PATH])

    env_config_v2 = load_env_config_v2_from_yaml()
    pilot_session_dir.mkdir(parents=True, exist_ok=False)  # must not already exist -- see resolve_next_pilot_session_dir

    # V2 Phase 4C-1 (P1-2): this pilot's own canonical HPA is
    # demo-app-hpa-v2 (V2_HPA_NAME), never V1's demo-app-hpa -- the backend
    # is constructed with that explicit identity so delete_hpa()/
    # restore_hpa() below act on demo-app-hpa-v2.
    backend = LiveKubernetesBackend(
        prometheus_url=PROMETHEUS_URL, hpa_name=V2_HPA_NAME, hpa_manifest_path=V2_HPA_MANIFEST_PATH,
    )

    # Step 2 (pre-removal): refuse closed on any UNEXPECTED conflict before
    # touching anything -- a stale non-"none" controller lock, or any HPA
    # object other than the canonical demo-app-hpa-v2 this pilot is about to
    # remove itself (if V1's demo-app-hpa is present instead, that is an
    # unexpected conflict too -- never silently deleted).
    # `acquire_controller(CONTROLLER_PPO, ...)` cannot be
    # called yet at this point: by design (autoscaler.src.rl_env.
    # controller_exclusivity) it unconditionally refuses PPO while ANY HPA
    # is present -- including the canonical one THIS pilot is authorized to
    # remove as its own startup action (task PART 2 step 3). Acquiring
    # exclusivity therefore happens AFTER removal (below), once the cluster
    # genuinely has no HPA left; this explicit pre-check still fails closed
    # on anything this pilot did NOT itself expect to see.
    from autoscaler.src.rl_env.controller_exclusivity import CONTROLLER_NONE, read_lock

    existing_lock = read_lock()
    if existing_lock != CONTROLLER_NONE:
        raise ControllerConflictError(f"refusing to start: controller lock already held by {existing_lock!r}")
    existing_hpas = backend._kubectl.list_hpa_names()
    unexpected_hpas = [h for h in existing_hpas if h != V2_HPA_NAME]
    if unexpected_hpas:
        raise ControllerConflictError(f"refusing to start: unexpected HPA object(s) present: {unexpected_hpas}")

    baseline_pods = get_pod_health("autoscaler-app", "app=demo-app")
    baseline_restart_counts = {p.name: p.restart_count for p in baseline_pods}

    hpa_removed = False
    synced_env = None
    ledgered_env = None
    result: Dict[str, Any] = {
        "pilot_classification": "V2_LIVE_TRAINING_PILOT",
        "prometheus_preflight": prom_preflight,
    }

    try:
        # Step 3: safely remove the canonical HPA for PPO control.
        backend._kubectl.delete_hpa()
        hpa_removed = True

        # Step 1 (post-removal): NOW acquire PPO controller exclusivity --
        # the cluster genuinely has no HPA left, so this call both records
        # the lock AND re-confirms (via its own live kubectl read) that
        # nothing re-appeared between the check above and this point.
        acquire_controller(CONTROLLER_PPO, backend._kubectl)

        # Step 6: construct the V2 environment against the live backend.
        raw_env = KubernetesAutoscalerEnvV2(
            backend=backend, episode_duration_s=600.0, config=env_config_v2,  # overwritten per-episode by V2TrafficSyncedWrapper
        )
        controller_code_hash = hashlib.sha256(
            (PROJECT_ROOT / "autoscaler/src/rl_env/environment_v2.py").read_bytes()
            + (PROJECT_ROOT / "autoscaler/src/rl_env/reward_v2.py").read_bytes()
        ).hexdigest()
        config_hash = hashlib.sha256(
            (PROJECT_ROOT / "configs/rl_v2.yaml").read_bytes() + (PROJECT_ROOT / "configs/reward_v2.yaml").read_bytes()
        ).hexdigest()
        ledgered_env = LedgeredEpisodeWrapper(
            raw_env, ledger_path=PILOT_LEDGER_PATH, traffic_profile="PENDING_FIRST_RESET",
            profile_hash="PENDING_FIRST_RESET", seed=seed, controller_code_hash=controller_code_hash,
            config_hash=config_hash, role="v2_pilot",
        )

        episode_counter = {"n": -1}

        def _next_episode_index() -> int:
            episode_counter["n"] += 1
            return episode_counter["n"]

        # Step 7: V2 training traffic/profile orchestration -- real TrafficGenerator, never curl.
        from collectors.metrics_collector import PrometheusClient

        orchestrator = V2TrafficOrchestrator(
            ledger_path=PILOT_TRAFFIC_PROVENANCE_LEDGER, prom_client=PrometheusClient(base_url=PROMETHEUS_URL),
        )
        import time as _time

        def _http_get(url: str, timeout: float):
            import httpx
            try:
                resp = httpx.get(url, timeout=timeout)
                return resp.status_code == 200, f"HTTP {resp.status_code}"
            except Exception as exc:  # noqa: BLE001
                return False, str(exc)

        orchestrator.http_get = _http_get
        synced_env = V2TrafficSyncedWrapper(
            ledgered_env, orchestrator=orchestrator, episode_index_provider=_next_episode_index, training_seed=seed,
        )

        # Step 8: fresh PPO model (no checkpoint loaded, no resume).
        ppo_kwargs = build_ppo_kwargs(rl_yaml_path=DEFAULT_RL_V2_YAML, seed=seed)
        ppo_kwargs["n_steps"] = PILOT_N_STEPS
        model = PPO(env=synced_env, **ppo_kwargs)

        safety_callback = PilotSafetyCallback(
            namespace="autoscaler-app", label_selector="app=demo-app",
            prom_base_url=PROMETHEUS_URL, pod_regex="demo-app-.*",
            baseline_restart_counts=baseline_restart_counts, telemetry_path=pilot_telemetry_path,
            synced_wrapper=synced_env,
        )

        # Step 9: exactly one rollout/update.
        start_utc = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
        model.learn(total_timesteps=PILOT_TOTAL_TIMESTEPS, callback=safety_callback)
        end_utc = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())

        # Step 10: save pilot checkpoint + provenance (never the reserved final-model path).
        model.save(str(pilot_model_path))
        model_sha256 = hashlib.sha256(pilot_model_path.read_bytes()).hexdigest()
        metadata = build_v2_model_metadata(training_seed=seed, ppo_hyperparameters=ppo_kwargs)
        metadata_dict = metadata.to_dict()
        metadata_dict["classification"] = "V2_LIVE_TRAINING_PILOT"
        metadata_dict["model_sha256"] = model_sha256
        metadata_dict["start_utc"] = start_utc
        metadata_dict["end_utc"] = end_utc
        with open(pilot_metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, default=str)

        training_metrics = dict(safety_callback.summary)
        training_metrics["num_timesteps"] = model.num_timesteps
        with open(pilot_metrics_path, "w", encoding="utf-8") as f:
            json.dump(training_metrics, f, indent=2, default=str)

        result.update({
            "ok": safety_callback.termination_reason is None and model.num_timesteps == PILOT_TOTAL_TIMESTEPS,
            "termination_reason": safety_callback.termination_reason,
            "num_timesteps": model.num_timesteps,
            "training_metrics": training_metrics,
            "model_sha256": model_sha256,
            "pilot_session_dir": str(pilot_session_dir),
            "pilot_model_path": str(pilot_model_path),
            "pilot_metadata_path": str(pilot_metadata_path),
            "pilot_metrics_path": str(pilot_metrics_path),
            "pilot_ledger_path": str(PILOT_LEDGER_PATH),
            "prometheus_url": PROMETHEUS_URL,
        })
    except Exception as exc:
        # V2 Phase 3C PART 4: ensure a genuine TOOLING failure (e.g. V2
        # Phase 3A physical attempt 1's Prometheus-port DrainQueryError,
        # which never received its own ledger row) is append-only logged
        # going forward, using the EXISTING, unmodified
        # log_tooling_failure() mechanism -- idempotent-safe via the same
        # `_episode_completed` guard `log_pilot_safety_interruption()` uses,
        # so this never double-writes alongside a safety-callback row for
        # the same step.
        if ledgered_env is not None and hasattr(ledgered_env, "log_tooling_failure"):
            try:
                ledgered_env.log_tooling_failure(exc, phase="run_pilot_training")
            except Exception as log_exc:  # noqa: BLE001 -- logging the failure must never mask the original one
                result.setdefault("cleanup_warnings", []).append(f"log_tooling_failure itself failed: {log_exc}")
        raise
    finally:
        # Step 11: stop traffic (orchestrator.close() on the synced wrapper).
        if synced_env is not None:
            try:
                synced_env.close()
            except Exception as exc:  # noqa: BLE001
                result.setdefault("cleanup_warnings", []).append(f"synced_env.close failed: {exc}")
        # Step 12: restore deployment to canonical replica count.
        try:
            backend.set_desired_replicas(env_config_v2.initial_replicas)
            backend._kubectl.wait_for_replica_state(env_config_v2.initial_replicas, backend.restoration_max_wait_s)
        except Exception as exc:  # noqa: BLE001
            result.setdefault("cleanup_warnings", []).append(f"deployment restoration failed: {exc}")
        # Step 13: restore the canonical frozen HPA (never alter the manifest itself).
        if hpa_removed:
            try:
                backend._kubectl.restore_hpa()
            except Exception as exc:  # noqa: BLE001
                result.setdefault("cleanup_warnings", []).append(f"HPA restoration failed: {exc}")
        # Step 14: release the controller lock.
        release_controller(CONTROLLER_PPO)

    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute-live-training", action="store_true")
    parser.add_argument("--i-confirm-live-cluster-interaction", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.execute_live_training:
        if not args.i_confirm_live_cluster_interaction:
            print("Refusing --execute-live-training without --i-confirm-live-cluster-interaction.", file=sys.stderr)
            return 2
        result = run_pilot_training(args.seed)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") else 1

    report = run_dry_run_validation(args.seed)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
