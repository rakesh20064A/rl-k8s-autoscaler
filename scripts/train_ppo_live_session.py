#!/usr/bin/env python3
"""scripts/train_ppo_live_session.py

Phase 16W: the session-based, checkpoint/resume-capable live-training
runner that supersedes `scripts/train_ppo_live.py::run_live_training()`'s
single-monolithic-call design as the AUTHORIZED way to execute the frozen
12,288-step budget once live training itself is separately authorized.
See `task_outputs/phase_16w_checkpoint_resume_protocol.yaml` for the full
design and the exact SB3-internals verification this implementation relies
on (`autoscaler/src/rl_env/checkpoint_state.py`'s module docstring
duplicates the same citations for convenience).

Two modes, mirroring `train_ppo_live.py`'s existing convention:

  --dry-run (default, SAFE): validates the checkpoint/resume machinery
    end-to-end using ScriptedBackend/mocks only -- zero live Kubernetes
    interaction, zero real PPO training. Exercises BOTH the fresh-start
    path and (if a prior dry-run checkpoint exists) the resume-provenance
    path.

  --execute-session (REQUIRES --i-confirm-live-cluster-interaction): runs
    exactly one real live-training session (a fixed number of complete
    PPO rollouts against LiveKubernetesBackend), then checkpoints and
    exits. NOT invoked in Phase16W.

No CLI flag can override any frozen PPO/reward/action hyperparameter --
only `--dry-run`, `--execute-session`, `--i-confirm-live-cluster-interaction`,
and `--rollouts-per-session` (an OPERATIONAL parameter, not a scientific
one -- it changes how many of the SAME frozen 256-step rollouts happen
per session, never the rollout size, total budget, or any PPO/reward value).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoscaler.src.rl_env.checkpoint_state import (  # noqa: E402
    CheckpointProvenanceError,
    SessionCheckpointMetadata,
    capture_rng_state,
    load_rng_state,
    resolve_profile_for_episode,
    restore_rng_state,
    save_rng_state,
    verify_resume_provenance,
)
from autoscaler.src.rl_env.config import load_env_config_from_yaml  # noqa: E402
from autoscaler.src.rl_env.environment import KubernetesAutoscalerEnv  # noqa: E402
from autoscaler.src.rl_env.scripted_backend import ScriptedBackend  # noqa: E402
from autoscaler.src.rl_env.training_wrapper import LedgeredEpisodeWrapper  # noqa: E402
from autoscaler.src.rl_env.types import RawMetricSnapshot  # noqa: E402
from scripts import rl_live_common as common  # noqa: E402
from scripts.train_ppo_live import build_ppo_kwargs_for_live_training  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / "persistent-data" / "rl" / "checkpoints" / "phase16w_sessions"
SB3_CHECKPOINT_NAME = "ppo_model.zip"
RNG_STATE_NAME = "rng_state.pkl"
METADATA_NAME = "session_metadata.json"
TRAINING_LEDGER_PATH = PROJECT_ROOT / "persistent-data" / "rl_training" / "ppo_training_episode_ledger.jsonl"

FROZEN_N_STEPS = 256
FROZEN_TOTAL_TIMESTEPS = 12288
FROZEN_TRAINING_SEED = 42
DEFAULT_ROLLOUTS_PER_SESSION = 2  # ~2.35h/rollout (Phase16V recompute) x 2 ~= 4.7-4.9h, safely under the ~5.5h Phase16Q precedent


def _session_dir(session_index: int) -> Path:
    return CHECKPOINT_DIR / f"session_{session_index:03d}"


def _next_rollout_index(prior_index: Optional[int]) -> int:
    """Phase16AD fix: `prior_index` (`find_latest_checkpoint()`'s result)
    legitimately holds `0` for the first completed checkpoint. The
    original inline expression `(find_latest_checkpoint() or -1) + 1`
    miscomputed the next rollout index as `0` (not `1`) in exactly that
    case, because Python's `or` treats the int `0` as falsy --
    indistinguishable from `None`. This explicit `is None` check is the
    correct, general form: `None -> 0`, `0 -> 1`, `1 -> 2`, ..."""
    return 0 if prior_index is None else prior_index + 1


def _current_governing_hashes() -> Dict[str, str]:
    return {c["path"]: c["actual"] for c in common.verify_governing_hashes()}


def _synthetic_dry_run_timeline(n_steps: int = 600) -> Dict[float, RawMetricSnapshot]:
    return {
        i * 30.0: RawMetricSnapshot(
            request_rate_rps=5.0, error_rate_rps=0.0, latency_p95_ms=100.0,
            requests_in_progress=1.0, cpu_cores=0.2, memory_bytes=40_000_000.0,
        )
        for i in range(n_steps)
    }


def find_latest_checkpoint() -> Optional[int]:
    """Returns the highest session_index with a complete checkpoint on
    disk, or None if no checkpoint exists yet (fresh start)."""
    if not CHECKPOINT_DIR.exists():
        return None
    indices = []
    for child in CHECKPOINT_DIR.iterdir():
        if child.is_dir() and child.name.startswith("session_"):
            meta_path = child / METADATA_NAME
            zip_path = child / SB3_CHECKPOINT_NAME
            rng_path = child / RNG_STATE_NAME
            if meta_path.exists() and zip_path.exists() and rng_path.exists():
                try:
                    indices.append(int(child.name.split("_")[1]))
                except ValueError:
                    continue
    return max(indices) if indices else None


def run_dry_run_validation() -> Dict[str, Any]:
    """Exercises fresh-start AND resume-provenance end-to-end, entirely
    offline (ScriptedBackend + a throwaway `tempfile.TemporaryDirectory()`
    for checkpoint/RNG/metadata files). Deliberately NEVER writes under
    `CHECKPOINT_DIR` / `persistent-data/rl/checkpoints` -- that exact path
    is asserted to stay empty/nonexistent until real PPO training occurs by
    a pre-existing project invariant, and a dry-run validation script must
    never trip that invariant."""
    import tempfile

    from stable_baselines3 import PPO

    with tempfile.TemporaryDirectory(prefix="phase16w_dry_run_") as tmp_dir_str:
        return _run_dry_run_validation_in(Path(tmp_dir_str), PPO)


def _run_dry_run_validation_in(dry_run_root: Path, PPO) -> Dict[str, Any]:  # noqa: N803
    report: Dict[str, Any] = {"classification": "OFFLINE_DRY_RUN_VALIDATION_ONLY"}
    session0_dir = dry_run_root / "session_000"
    session1_dir = dry_run_root / "session_001"

    def build_env():
        env_config = load_env_config_from_yaml()
        backend = ScriptedBackend(timeline=_synthetic_dry_run_timeline(), initial_replicas=1)
        raw_env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=270.0, config=env_config)
        return LedgeredEpisodeWrapper(
            raw_env, ledger_path=Path("/dev/null"), traffic_profile=resolve_profile_for_episode(0, FROZEN_TRAINING_SEED),
            profile_hash="n/a", seed=FROZEN_TRAINING_SEED, controller_code_hash="n/a", config_hash="n/a", role="training",
        )

    # --- Fresh-start "session 0": construct, run one tiny rollout, checkpoint.
    env0 = build_env()
    ppo_kwargs = build_ppo_kwargs_for_live_training(FROZEN_TRAINING_SEED)
    ppo_kwargs["n_steps"] = 32  # tiny for a fast dry run; production uses the frozen 256
    ppo_kwargs["batch_size"] = 16
    model0 = PPO(env=env0, **ppo_kwargs)
    model0.learn(total_timesteps=32, reset_num_timesteps=False)
    session0_dir.mkdir(parents=True, exist_ok=True)
    model0.save(session0_dir / SB3_CHECKPOINT_NAME)
    rng_hash0 = save_rng_state(capture_rng_state(), session0_dir / RNG_STATE_NAME)
    sb3_hash0 = common.sha256_file(session0_dir / SB3_CHECKPOINT_NAME)
    meta0 = SessionCheckpointMetadata(
        session_index=0, cumulative_timesteps=model0.num_timesteps, cumulative_ppo_updates=1,
        cumulative_episodes_completed=env0._episode_index, n_steps=32, training_seed=FROZEN_TRAINING_SEED,
        sb3_checkpoint_sha256=sb3_hash0, rng_state_sha256=rng_hash0,
        governing_hashes_snapshot=_current_governing_hashes(),
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    meta0.to_json(session0_dir / METADATA_NAME)
    report["session0_num_timesteps"] = model0.num_timesteps
    report["session0_checkpoint_written"] = True

    # --- Resume "session 1": load checkpoint, verify provenance, restore RNG, continue.
    loaded_meta = SessionCheckpointMetadata.from_json(session0_dir / METADATA_NAME)
    provenance_ok = True
    provenance_error = None
    try:
        verify_resume_provenance(
            loaded_meta, session0_dir / SB3_CHECKPOINT_NAME, session0_dir / RNG_STATE_NAME,
            expected_next_session_index=1, expected_n_steps=32, expected_training_seed=FROZEN_TRAINING_SEED,
            current_governing_hashes=_current_governing_hashes(),
        )
    except CheckpointProvenanceError as exc:
        provenance_ok = False
        provenance_error = str(exc)
    report["resume_provenance_verified"] = provenance_ok
    report["resume_provenance_error"] = provenance_error

    env1 = build_env()
    model1 = PPO.load(session0_dir / SB3_CHECKPOINT_NAME, env=env1)
    restore_rng_state(load_rng_state(session0_dir / RNG_STATE_NAME))
    num_timesteps_before_resume = model1.num_timesteps
    model1.learn(total_timesteps=32, reset_num_timesteps=False)
    report["num_timesteps_before_resume"] = num_timesteps_before_resume
    report["num_timesteps_after_resumed_session"] = model1.num_timesteps
    report["timesteps_correctly_appended_not_duplicated"] = (
        model1.num_timesteps == num_timesteps_before_resume + 32
    )

    # --- Fail-closed check: a deliberately WRONG expected_next_session_index must be rejected.
    rejected_bad_resume = False
    try:
        verify_resume_provenance(
            loaded_meta, session0_dir / SB3_CHECKPOINT_NAME, session0_dir / RNG_STATE_NAME,
            expected_next_session_index=5,  # wrong on purpose
            expected_n_steps=32, expected_training_seed=FROZEN_TRAINING_SEED,
            current_governing_hashes=_current_governing_hashes(),
        )
    except CheckpointProvenanceError:
        rejected_bad_resume = True
    report["fail_closed_on_session_index_mismatch"] = rejected_bad_resume

    session1_dir.mkdir(parents=True, exist_ok=True)
    model1.save(session1_dir / SB3_CHECKPOINT_NAME)

    report["frozen_n_steps"] = FROZEN_N_STEPS
    report["frozen_total_timesteps"] = FROZEN_TOTAL_TIMESTEPS
    report["frozen_training_seed"] = FROZEN_TRAINING_SEED
    report["default_rollouts_per_session"] = DEFAULT_ROLLOUTS_PER_SESSION
    report["final_holdout_untouched"] = _final_holdout_untouched_check()

    report["traffic_orchestration"] = _run_traffic_orchestration_dry_run(dry_run_root)

    report["traffic_generated"] = False
    report["kubernetes_mutation"] = False
    report["ok"] = (
        report["session0_checkpoint_written"]
        and report["resume_provenance_verified"]
        and report["timesteps_correctly_appended_not_duplicated"]
        and report["fail_closed_on_session_index_mismatch"]
        and report["final_holdout_untouched"]
        and report["traffic_orchestration"]["ok"]
    )
    return report


def _run_traffic_orchestration_dry_run(dry_run_root: Path) -> Dict[str, Any]:
    """Phase16Y task sec.11: exercises profile resolution, the workload
    firewall, generator command construction, episode-lifecycle hooks, and
    provenance recording via the traffic orchestrator -- entirely with
    fakes (no real subprocess, no real HTTP, no real docker/kubectl). All
    provenance is written under `dry_run_root` (a `tempfile.
    TemporaryDirectory()`), never the canonical
    `persistent-data/rl_training/` path."""
    import subprocess as _subprocess

    from scripts import traffic_orchestrator as to

    report: Dict[str, Any] = {}

    # 1. Profile resolution + firewall (task sec.7/12, Phase16Z): all SIX
    # authorized training profiles now parse correctly and pass the
    # shape-integrity check (loadgen.traffic_generator.parse_traffic_
    # profile()'s flat-schema fix); final-holdout and unseen_eval remain
    # correctly rejected.
    accepted = {}
    for name in ("constant", "step", "burst", "ramp", "periodic", "mixed"):
        try:
            to.resolve_profile_metadata(name)
            accepted[name] = "PARSE_OK"
        except Exception as exc:  # noqa: BLE001
            accepted[name] = f"UNEXPECTED_REJECTION: {exc}"
    for name in ("iteration_rl_final_holdout_v1", "unseen_eval"):
        try:
            to.resolve_profile_metadata(name)
            accepted[name] = "UNEXPECTED_ACCEPTANCE"
        except Exception:  # noqa: BLE001
            accepted[name] = "CORRECTLY_REJECTED"
    report["profile_firewall"] = accepted
    report["profile_firewall_ok"] = all(
        v in ("PARSE_OK", "CORRECTLY_REJECTED") for v in accepted.values()
    )

    # 2. Fake command runner / Popen / HTTP -- zero real subprocess/network.
    # Phase16AB: docker port only reports a mapping for a container whose
    # docker run command actually published a port (models real Docker
    # behavior -- the Phase16X-Retry defect was a dry-run fake that always
    # "succeeded" regardless of whether the real command would have).
    class _FakeRun:
        def __init__(self):
            self._published_containers = {}

        def __call__(self, cmd, timeout):
            if cmd[:2] == ["docker", "inspect"] and "minikube" in cmd:
                return _subprocess.CompletedProcess([], 0, stdout='{"minikube": {}}')
            if cmd[:1] == ["minikube"]:
                return _subprocess.CompletedProcess([], 0, stdout="192.168.49.2")
            if cmd[:2] == ["kubectl", "get"]:
                return _subprocess.CompletedProcess([], 0, stdout="31234")
            if cmd[:2] == ["docker", "run"]:
                container_name = cmd[cmd.index("--name") + 1]
                self._published_containers[container_name] = "-p" in cmd or "-P" in cmd
                return _subprocess.CompletedProcess([], 0, stdout="containerid")
            if cmd[:2] == ["docker", "port"]:
                container_name = cmd[2]
                if self._published_containers.get(container_name, False):
                    return _subprocess.CompletedProcess([], 0, stdout="0.0.0.0:55555\n[::]:55555")
                return _subprocess.CompletedProcess([], 0, stdout="")
            if cmd[:2] == ["docker", "inspect"]:
                return _subprocess.CompletedProcess([], 0, stdout="true|0")
            if cmd[:2] == ["docker", "rm"]:
                return _subprocess.CompletedProcess([], 0)
            raise AssertionError(f"dry-run must never issue a real command: {cmd}")

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            self.args = cmd
            self._rc = None

        def poll(self):
            return self._rc

        def wait(self, timeout=None):
            self._rc = 0
            return 0

        def terminate(self):
            self._rc = -15

        def kill(self):
            self._rc = -9

        @property
        def returncode(self):
            return self._rc

    def _fake_popen_factory(cmd, **kwargs):
        return _FakePopen(cmd)

    def _fake_http_get(url, timeout):
        return True, "ok"

    orchestrator = to.TrafficOrchestrator(
        transport=to.V5TransportManager(run=_FakeRun()),
        popen_factory=_fake_popen_factory,
        http_get=_fake_http_get,
        sleep_fn=lambda s: None,
        ledger_path=dry_run_root / "traffic_provenance_ledger.jsonl",
        # Index33 correction: episode-end now drains server-side in-flight
        # requests via a real Prometheus query by default. This dry-run
        # validation has no live Prometheus (by design, fully offline), so
        # inject an immediate-zero fake -- the drain mechanism itself is
        # covered separately in tests/unit/test_traffic_orchestrator.py.
        drain_query_fn=lambda: (0.0, None),
    )

    # 3. Command construction correctness.
    profile = to.resolve_profile_metadata("constant")
    route = orchestrator.ensure_route_established("dry_run_seed")
    cmd = to.build_generator_command(profile, route, "dry_run_1")
    report["generator_command_uses_frozen_seed"] = str(profile.seed) in cmd
    report["generator_command_uses_fresh_connection_mode"] = "fresh" in cmd
    report["generator_command_omits_replay_schedule"] = "--replay-schedule" not in cmd

    # 4. Episode-lifecycle hooks + provenance, via a synthetic ScriptedBackend
    # environment -- no live Kubernetes, no real traffic.
    env_config = load_env_config_from_yaml()
    backend = ScriptedBackend(timeline=_synthetic_dry_run_timeline(60), initial_replicas=1)
    raw_env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=30.0, config=env_config)
    ledgered = LedgeredEpisodeWrapper(
        raw_env, ledger_path=dry_run_root / "training_ledger.jsonl",
        traffic_profile="placeholder", profile_hash="placeholder",
        seed=FROZEN_TRAINING_SEED, controller_code_hash="n/a", config_hash="n/a", role="training",
    )
    episode_counter = {"n": 0}
    traffic_synced = to.TrafficSyncedWrapper(
        ledgered, orchestrator, lambda: episode_counter.__setitem__("n", episode_counter["n"] + 1) or episode_counter["n"] - 1,
        FROZEN_TRAINING_SEED,
    )
    traffic_synced.reset(seed=1)
    report["episode_lifecycle_traffic_started_after_reset"] = traffic_synced._current_handle is not None
    _, _, terminated, truncated, _ = traffic_synced.step(1)
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = traffic_synced.step(1)
    report["episode_lifecycle_traffic_stopped_at_episode_end"] = traffic_synced._current_handle is None
    traffic_synced.close()

    from autoscaler.src.rl_env.episode_ledger import count_rows
    report["provenance_rows_recorded"] = count_rows(orchestrator.ledger_path)
    report["provenance_schema_ok"] = report["provenance_rows_recorded"] >= 1

    report["ok"] = (
        report["profile_firewall_ok"]
        and report["generator_command_uses_frozen_seed"]
        and report["generator_command_uses_fresh_connection_mode"]
        and report["generator_command_omits_replay_schedule"]
        and report["episode_lifecycle_traffic_started_after_reset"]
        and report["episode_lifecycle_traffic_stopped_at_episode_end"]
        and report["provenance_schema_ok"]
    )
    return report


def _final_holdout_untouched_check() -> bool:
    """Phase16W task sec.9: hash/existence-only re-verification -- never
    inspects the workload's traffic-shape characteristics beyond what was
    already known at Phase16U design time."""
    spec_path = common.PHASE16U_WORKLOAD_SPLIT_JSON
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)["final_evaluation_workload"]
    target = PROJECT_ROOT / spec["new_file_name_reserved"]
    if not target.exists():
        return True
    from autoscaler.src.rl_env.episode_ledger import count_rows, read_rows
    return (
        count_rows(TRAINING_LEDGER_PATH) == 0
        or all(
            "iteration_rl_final_holdout_v1" not in str(row.get("traffic_profile", ""))
            for row in read_rows(TRAINING_LEDGER_PATH)
        )
    )


def _make_live_traffic_orchestrator(session_index: int):  # pragma: no cover - not exercised in Phase16Y
    """Real-dependency TrafficOrchestrator (real subprocess.Popen, real V5
    docker/kubectl transport, real HTTP health check) -- constructed only
    on the run_live_session() path, never on --dry-run."""
    import httpx

    from scripts import traffic_orchestrator as to

    def http_get(url: str, timeout: float):
        try:
            resp = httpx.get(url, timeout=timeout)
            return resp.status_code == 200, f"HTTP {resp.status_code}"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    return to.TrafficOrchestrator(session_index=session_index, http_get=http_get)


def run_live_session(rollouts_per_session: int) -> int:  # pragma: no cover - not exercised in Phase16Y
    """The real live-training session path. NOT invoked in Phase16Y.

    Checkpoint granularity is ALWAYS one completed PPO rollout (256 steps),
    regardless of `rollouts_per_session` -- a crash or interruption at any
    point loses at most the ONE rollout currently in flight (~2.35h of live
    decision time at Phase16V's recomputed per-step overhead), never more,
    no matter how many rollouts this process happens to attempt before
    exiting or being interrupted. `rollouts_per_session` is purely an
    OPERATIONAL bound on how many of these individually-checkpointed
    rollouts this one process invocation will attempt before voluntarily
    stopping -- it is not the unit of risk.

    Phase16Y correction: each episode's traffic is now orchestrated via
    `scripts.traffic_orchestrator.TrafficSyncedWrapper`, synchronized with
    the frozen episode lifecycle (docs/rl_mdp_spec.md sec.20) -- traffic
    starts only after canonical reset + stabilization complete, and stops
    at episode end, never merely "somewhere in parallel" for the whole
    session (Phase16X's finding)."""
    from stable_baselines3 import PPO

    from autoscaler.src.rl_env.live_backend import LiveKubernetesBackend
    from scripts.traffic_orchestrator import TrafficSyncedWrapper

    backend = LiveKubernetesBackend()
    if backend._kubectl.hpa_exists():
        print("REFUSING TO START: a live HPA object targets the Deployment.", file=sys.stderr)
        return 3

    env_config = load_env_config_from_yaml()

    for _ in range(rollouts_per_session):
        # Phase16AD fix: see _next_rollout_index()'s docstring. One call to
        # find_latest_checkpoint() now feeds both prior_index and
        # rollout_index, removing the previous redundant second scan too.
        prior_index = find_latest_checkpoint()
        rollout_index = _next_rollout_index(prior_index)
        orchestrator = _make_live_traffic_orchestrator(rollout_index)

        if prior_index is None:
            cumulative_episodes_start = 0
        else:
            prior_dir = _session_dir(prior_index)
            prior_meta = SessionCheckpointMetadata.from_json(prior_dir / METADATA_NAME)
            verify_resume_provenance(
                prior_meta, prior_dir / SB3_CHECKPOINT_NAME, prior_dir / RNG_STATE_NAME,
                expected_next_session_index=rollout_index, expected_n_steps=FROZEN_N_STEPS,
                expected_training_seed=FROZEN_TRAINING_SEED, current_governing_hashes=_current_governing_hashes(),
            )  # raises CheckpointProvenanceError -> fail closed, never proceeds silently
            cumulative_episodes_start = prior_meta.cumulative_episodes_completed

        episode_counter = {"n": cumulative_episodes_start}

        def next_episode_index() -> int:
            idx = episode_counter["n"]
            episode_counter["n"] += 1
            return idx

        # episode_duration_s is a harmless placeholder here -- TrafficSyncedWrapper
        # overwrites it to the assigned profile's own duration before every reset().
        # Wrapping order: raw_env -> LedgeredEpisodeWrapper -> TrafficSyncedWrapper
        # (outermost) -- see TrafficSyncedWrapper's class docstring for why this
        # order (not the reverse) is required.
        raw_env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=600.0, config=env_config)
        ledgered = LedgeredEpisodeWrapper(
            raw_env, ledger_path=TRAINING_LEDGER_PATH,
            traffic_profile="resolved_per_episode", profile_hash="resolved_per_episode",
            seed=FROZEN_TRAINING_SEED, controller_code_hash="TBD", config_hash="TBD", role="training",
        )
        wrapped = TrafficSyncedWrapper(ledgered, orchestrator, next_episode_index, FROZEN_TRAINING_SEED)

        if prior_index is None:
            model = PPO(env=wrapped, **build_ppo_kwargs_for_live_training(FROZEN_TRAINING_SEED))
        else:
            model = PPO.load(prior_dir / SB3_CHECKPOINT_NAME, env=wrapped)
            restore_rng_state(load_rng_state(prior_dir / RNG_STATE_NAME))

        try:
            model.learn(total_timesteps=FROZEN_N_STEPS, reset_num_timesteps=False)
        except Exception as exc:
            ledgered.log_tooling_failure(exc, phase=f"rollout_{rollout_index}_learn")
            raise
        finally:
            ledgered.log_session_boundary_if_episode_in_progress()
            wrapped.close()  # stops any still-active generator, tears down the relay route
            try:
                backend.set_desired_replicas(env_config.initial_replicas)
                backend._kubectl.wait_for_replica_state(env_config.initial_replicas, backend.restoration_max_wait_s)
            except Exception as restore_exc:  # noqa: BLE001
                print(f"WARNING: post-rollout restoration failed: {restore_exc}", file=sys.stderr)

        _checkpoint_after_rollout(model, wrapped, rollout_index, episode_counter["n"])

    return 0


def _checkpoint_after_rollout(model, wrapped, rollout_index: int, cumulative_episodes: int) -> None:  # pragma: no cover
    this_dir = _session_dir(rollout_index)
    this_dir.mkdir(parents=True, exist_ok=True)
    model.save(this_dir / SB3_CHECKPOINT_NAME)
    rng_hash = save_rng_state(capture_rng_state(), this_dir / RNG_STATE_NAME)
    sb3_hash = common.sha256_file(this_dir / SB3_CHECKPOINT_NAME)
    SessionCheckpointMetadata(
        session_index=rollout_index, cumulative_timesteps=model.num_timesteps,
        cumulative_ppo_updates=model.num_timesteps // FROZEN_N_STEPS,
        cumulative_episodes_completed=cumulative_episodes,
        n_steps=FROZEN_N_STEPS, training_seed=FROZEN_TRAINING_SEED,
        sb3_checkpoint_sha256=sb3_hash, rng_state_sha256=rng_hash,
        governing_hashes_snapshot=_current_governing_hashes(),
        created_utc=datetime.now(timezone.utc).isoformat(),
    ).to_json(this_dir / METADATA_NAME)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute-session", action="store_true")
    parser.add_argument("--i-confirm-live-cluster-interaction", action="store_true")
    parser.add_argument("--rollouts-per-session", type=int, default=DEFAULT_ROLLOUTS_PER_SESSION)
    args = parser.parse_args(argv)

    if args.execute_session:
        if not args.i_confirm_live_cluster_interaction:
            print("Refusing --execute-session without --i-confirm-live-cluster-interaction.", file=sys.stderr)
            return 2
        return run_live_session(args.rollouts_per_session)

    report = run_dry_run_validation()
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
