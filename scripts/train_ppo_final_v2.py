#!/usr/bin/env python3
"""scripts/train_ppo_final_v2.py

V2 Phase 4A: the REAL, fixed-budget V2 final-training campaign. Distinct
from `scripts/train_ppo_live_v2.py`'s one-rollout PILOT path in every way
that matters:

    pilot (train_ppo_live_v2.py)           final campaign (this script)
    persistent-data/rl_training_v2/pilot/  persistent-data/rl_training_v2/final/
    PILOT_TOTAL_TIMESTEPS = 256 (1 update) FINAL_TOTAL_TIMESTEPS_BUDGET = 5120 (20 updates)
    one `model.learn()` call               20 CHECKPOINTED sessions, each one `model.learn()`
    fresh policy only                      session 0 fresh; sessions 1-19 RESUME from the
                                            prior session's verified checkpoint

Shares, unmodified, every piece of the validated V2 Phase 3D pilot
machinery: `KubernetesAutoscalerEnvV2`, `LedgeredEpisodeWrapper`,
`scripts.v2_traffic_orchestrator.{V2TrafficOrchestrator,
V2TrafficSyncedWrapper}` (drain-retry-aware), `scripts.v2_pilot_safety.
{PilotSafetyCallback, verify_prometheus_internal_paths, get_pod_health}`
(the identical 230Mi-per-pod/restart/zero-ready safety contract), and
`autoscaler.src.rl_env.checkpoint_state` (the pre-existing, GENERIC V1
Phase16W RNG-capture/session-metadata/resume-provenance machinery --
nothing in that module is V1-profile-coupled except
`resolve_profile_for_episode`/`TRAINING_PROFILE_CYCLE`, which this script
does not import; V2's own `resolve_v2_profile_for_episode` is used
instead). No scientific contract value (reward, observation, PPO
hyperparameters, profiles, safety thresholds, Prometheus/drain semantics)
is changed anywhere in this file.

Checkpoint granularity is always exactly one completed 256-step PPO
rollout/update -- see `run_one_final_session()`. The campaign loop
(`run_final_training_campaign()`) advances session-by-session, persisting
`campaign_metadata.json` only after a session's checkpoint has been
independently re-verified (`checkpoint_state.verify_resume_provenance`);
on ANY session failure (exception or safety trip), the loop stops
immediately, preserves whatever partial evidence that session wrote, and
does NOT advance `campaign_metadata.json`'s cumulative counters -- the next
explicit continuation (a separately authorized phase, never automatic) must
resume from the last VERIFIED session.

`--dry-run` (the default) exercises the fresh-start + resume-provenance +
instrumentation machinery end-to-end against `ScriptedBackend` only -- zero
live cluster interaction, mirroring `train_ppo_live_session.py`'s own
dry-run convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoscaler.src.rl_env.checkpoint_state import (  # noqa: E402
    CheckpointProvenanceError,
    SessionCheckpointMetadata,
    capture_rng_state,
    load_rng_state,
    restore_rng_state,
    save_rng_state,
    verify_resume_provenance,
)
from autoscaler.src.rl_env.config_v2 import DEFAULT_RL_V2_YAML, load_env_config_v2_from_yaml  # noqa: E402
from autoscaler.src.rl_env.controller_exclusivity import (  # noqa: E402
    CONTROLLER_NONE,
    CONTROLLER_PPO,
    ControllerConflictError,
    acquire_controller,
    read_lock,
    release_controller,
)
from autoscaler.src.rl_env.environment_v2 import KubernetesAutoscalerEnvV2  # noqa: E402
from autoscaler.src.rl_env.ppo_config import build_ppo_kwargs  # noqa: E402
from autoscaler.src.rl_env.scripted_backend import ScriptedBackend  # noqa: E402
from autoscaler.src.rl_env.training_contract_v2 import (  # noqa: E402
    V1_MODEL_PATH,
    V1_TRAINING_LEDGER,
    V2_MODEL_PATH,
    V1ArtifactProtectionError,
    assert_fresh_policy_or_justified,
    assert_v1_artifacts_untouched,
    build_v2_model_metadata,
)
from autoscaler.src.rl_env.training_wrapper import LedgeredEpisodeWrapper  # noqa: E402
from autoscaler.src.rl_env.types import RawMetricSnapshot  # noqa: E402
from scripts.train_ppo_live_v2 import PILOT_ROOT  # noqa: E402
from scripts.v2_training_instrumentation import build_update_metrics_row  # noqa: E402

# -- V2 final-training namespace: distinct from the pilot AND from V1 --
FINAL_ROOT = PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final"
FINAL_SESSIONS_DIR = FINAL_ROOT / "sessions"
FINAL_LEDGER_PATH = FINAL_ROOT / "ppo_training_episode_ledger_v2.jsonl"
FINAL_TRAFFIC_LEDGER_PATH = FINAL_ROOT / "traffic_provenance_ledger_v2.jsonl"
FINAL_UPDATE_METRICS_PATH = FINAL_ROOT / "ppo_update_metrics.jsonl"
FINAL_CAMPAIGN_METADATA_PATH = FINAL_ROOT / "campaign_metadata.json"

SB3_CHECKPOINT_NAME = "ppo_model.zip"
RNG_STATE_NAME = "rng_state.pkl"
SESSION_METADATA_NAME = "metadata.json"
SESSION_TRAINING_METRICS_NAME = "training_metrics.json"
SESSION_TELEMETRY_NAME = "step_telemetry.jsonl"
SESSION_FAILURE_MARKER_NAME = "SESSION_FAILED.json"

# Pre-registered, FIXED budget (task PART 4) -- never extended based on
# results. n_steps/timesteps/seed/hyperparameters are identical to the
# validated V2 Phase 3D pilot contract; only the NUMBER of sessions is new.
FINAL_N_STEPS = 256
FINAL_TOTAL_TIMESTEPS_BUDGET = 5120
FINAL_PLANNED_SESSIONS = FINAL_TOTAL_TIMESTEPS_BUDGET // FINAL_N_STEPS  # 20
FINAL_TRAINING_SEED = 42

assert FINAL_PLANNED_SESSIONS * FINAL_N_STEPS == FINAL_TOTAL_TIMESTEPS_BUDGET

# The V2 scientific-contract files whose drift between sessions must fail
# closed a resume (mirrors train_ppo_live_session.py's V1
# `_current_governing_hashes()`, pointed at the V2 contract instead).
V2_GOVERNING_HASH_PATHS = (
    PROJECT_ROOT / "configs" / "rl_v2.yaml",
    PROJECT_ROOT / "configs" / "reward_v2.yaml",
    PROJECT_ROOT / "autoscaler" / "src" / "rl_env" / "environment_v2.py",
    PROJECT_ROOT / "autoscaler" / "src" / "rl_env" / "reward_v2.py",
    PROJECT_ROOT / "autoscaler" / "src" / "rl_env" / "observation.py",
)


def _current_v2_governing_hashes() -> Dict[str, str]:
    hashes = {}
    for path in V2_GOVERNING_HASH_PATHS:
        hashes[str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


class FinalNamespaceViolation(RuntimeError):
    """Raised when final-campaign code is about to write somewhere outside
    its own `persistent-data/rl_training_v2/final/` namespace -- fail
    closed against both the V1 namespace (reusing training_contract_v2's
    existing guard) and the V2 PILOT namespace (session_000-003 must never
    be touched by this script)."""


def assert_final_namespace_isolated(paths: List[Path]) -> None:
    assert_v1_artifacts_untouched(paths)  # reuse the existing V1 guard unmodified
    pilot_root_resolved = PILOT_ROOT.resolve()
    for p in paths:
        resolved = p.resolve()
        if resolved == pilot_root_resolved or pilot_root_resolved in resolved.parents:
            raise FinalNamespaceViolation(f"refusing to write into the frozen V2 PILOT namespace: {p}")
        if p == V2_MODEL_PATH:
            raise FinalNamespaceViolation("final-campaign sessions must never write the reserved final-model path")


def resolve_next_final_session_dir(sessions_root: Path = FINAL_SESSIONS_DIR) -> Path:
    """Retry-safe session allocation -- identical scanning pattern to
    `train_ppo_live_v2.resolve_next_pilot_session_dir()` (never reuses or
    overwrites an existing session_NNN directory, whether it holds a
    successful or a failed/partial session's evidence). Retained as a
    generic utility; `run_one_final_session()` itself uses
    `resolve_session_attempt_dir()` below, which is SCIENTIFIC-session-index
    aware (see its docstring for why a naive "first free session_NNN" scan
    is wrong once a failed attempt has consumed a scientific slot)."""
    n = 0
    while (sessions_root / f"session_{n:03d}").exists():
        n += 1
    return sessions_root / f"session_{n:03d}"


def resolve_session_attempt_dir(
    scientific_session_index: int, sessions_root: Path = FINAL_SESSIONS_DIR,
) -> Path:
    """V2 Phase 4B-R PART 6: allocates a PHYSICAL ATTEMPT directory for a
    given, externally-supplied SCIENTIFIC session index -- the two are
    deliberately different axes. The scientific index is the authoritative
    position in the pre-registered 20-session campaign (never advanced
    except by a VERIFIED success, tracked in `campaign_metadata.json`'s
    `cumulative_updates`); the physical attempt directory is just where
    THIS attempt's evidence lands, and a failed first attempt must never be
    overwritten or cause the scientific index to silently skip forward.

    The first physical attempt at a scientific index gets the bare
    `session_NNN` name (preserving every pre-existing session_000-series
    path unchanged); a retry of the SAME scientific index (because the
    first attempt failed) gets `session_NNN__attemptK` (K>=2) -- mirrors
    `episode_ledger.next_physical_attempt_number()`'s scientific-identity
    vs. physical-attempt-identity split, applied here to session
    directories instead of ledger rows."""
    base_name = f"session_{scientific_session_index:03d}"
    base_dir = sessions_root / base_name
    if not base_dir.exists():
        return base_dir
    k = 2
    while (sessions_root / f"{base_name}__attempt{k}").exists():
        k += 1
    return sessions_root / f"{base_name}__attempt{k}"


def physical_attempt_number_for_dir(session_dir: Path) -> int:
    """Inverse of `resolve_session_attempt_dir()`'s naming scheme -- the
    bare `session_NNN` name is implicitly attempt 1."""
    name = session_dir.name
    if "__attempt" in name:
        return int(name.rsplit("__attempt", 1)[1])
    return 1


def scientific_session_index_for_dir(session_dir: Path) -> int:
    """Inverse-parses the SCIENTIFIC session index back out of either a
    bare `session_NNN` or a `session_NNN__attemptK` directory name."""
    name = session_dir.name.split("__attempt", 1)[0]
    return int(name.removeprefix("session_"))


# -- campaign_metadata.json: the single source of truth for VERIFIED progress --

def _default_campaign_metadata() -> Dict[str, Any]:
    return {
        "campaign_classification": "V2_FINAL_TRAINING_CAMPAIGN",
        "training_seed": FINAL_TRAINING_SEED,
        "n_steps_per_session": FINAL_N_STEPS,
        "total_timesteps_budget": FINAL_TOTAL_TIMESTEPS_BUDGET,
        "planned_sessions": FINAL_PLANNED_SESSIONS,
        "cumulative_timesteps": 0,
        "cumulative_updates": 0,
        "next_global_schedule_episode_index": 0,
        "sessions_completed": [],
        "latest_verified_session_dir": None,
        "latest_verified_model_path": None,
        "latest_verified_model_sha256": None,
        "latest_verified_rng_state_path": None,
        "campaign_started_utc": None,
        "campaign_last_updated_utc": None,
        "status": "not_started",
        # V2 Phase 4B-R PART 5: populated only on a terminal failure (a
        # session that returns ok=False, whether from a tripped safety
        # condition or an uncaught exception) -- None/absent otherwise.
        "failed_scientific_session_index": None,
        "failed_physical_attempt": None,
        "failure_reason": None,
        "resume_eligible": None,
    }


def _remaining_timesteps(state: Dict[str, Any]) -> int:
    return state["total_timesteps_budget"] - state["cumulative_timesteps"]


def _mark_campaign_failed(campaign_state: Dict[str, Any], failed_session_result: Dict[str, Any]) -> None:
    """V2 Phase 4B-R PART 5: the single, atomic place that records a
    terminal campaign failure -- called from BOTH the loop's early-return
    path (a session returned `ok=False`) and the outer exception handler,
    so `campaign_metadata.json` on disk always matches whatever the process
    is about to report on stdout before it exits. Never advances
    `cumulative_timesteps`/`cumulative_updates`/`sessions_completed`/
    `latest_verified_*` -- those remain exactly as of the last VERIFIED
    success, which is the whole point of this bookkeeping fix."""
    campaign_state["status"] = "stopped_on_failure"
    campaign_state["failed_scientific_session_index"] = failed_session_result.get("session_index")
    campaign_state["failed_physical_attempt"] = failed_session_result.get("physical_attempt_number")
    campaign_state["failure_reason"] = failed_session_result.get("termination_reason")
    campaign_state["resume_eligible"] = True  # structural: not automatically precluded -- see PART 4's own gate
    save_campaign_metadata(campaign_state)


def load_campaign_metadata() -> Dict[str, Any]:
    if not FINAL_CAMPAIGN_METADATA_PATH.exists():
        return _default_campaign_metadata()
    with open(FINAL_CAMPAIGN_METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_campaign_metadata(state: Dict[str, Any]) -> None:
    FINAL_CAMPAIGN_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["campaign_last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    state["remaining_timesteps"] = _remaining_timesteps(state)
    with open(FINAL_CAMPAIGN_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True, default=str)


# -- dry-run (offline, ScriptedBackend only) --

def _synthetic_dry_run_timeline(n_steps: int = 600) -> Dict[float, RawMetricSnapshot]:
    return {
        i * 30.0: RawMetricSnapshot(
            request_rate_rps=5.0, error_rate_rps=0.0, latency_p95_ms=100.0,
            requests_in_progress=1.0, cpu_cores=0.2, memory_bytes=40_000_000.0,
        )
        for i in range(n_steps)
    }


def run_dry_run_validation(seed: int = FINAL_TRAINING_SEED) -> Dict[str, Any]:
    """Offline, zero-cluster-interaction proof of: namespace isolation,
    fresh-start session 0 + resumed session 1 (checkpoint/RNG/provenance
    round-trip, matching train_ppo_live_session.py's established pattern),
    SB3 update-metrics instrumentation populated, and normalized-observation
    telemetry exactly matching what the environment actually returned."""
    import tempfile

    from stable_baselines3 import PPO

    report: Dict[str, Any] = {"classification": "OFFLINE_DRY_RUN_VALIDATION_ONLY"}

    # 1. Namespace isolation checks (fail-closed against V1 and the pilot root).
    isolation_checks = {}
    for name, bad_path in (
        ("v1_model", V1_MODEL_PATH),
        ("v1_ledger", V1_TRAINING_LEDGER),
        ("pilot_session_000", PILOT_ROOT / "session_000" / "ppo_model.zip"),
        ("reserved_final_model_path", V2_MODEL_PATH),
    ):
        try:
            assert_final_namespace_isolated([bad_path])
            isolation_checks[name] = "UNEXPECTED_ACCEPTANCE"
        except (V1ArtifactProtectionError, FinalNamespaceViolation):
            isolation_checks[name] = "CORRECTLY_REJECTED"
    assert_final_namespace_isolated([FINAL_ROOT / "sessions" / "session_000" / SB3_CHECKPOINT_NAME])  # must not raise
    isolation_checks["final_namespace_accepted"] = "OK"
    report["namespace_isolation_checks"] = isolation_checks
    report["namespace_isolation_ok"] = all(
        v in ("CORRECTLY_REJECTED", "OK") for v in isolation_checks.values()
    )

    # 2. Fresh policy enforcement (default path -- no justification given).
    assert_fresh_policy_or_justified(None)
    report["fresh_policy_enforced"] = True

    with tempfile.TemporaryDirectory(prefix="v2_final_dry_run_") as tmp_dir_str:
        tmp_root = Path(tmp_dir_str)
        session0_dir = tmp_root / "session_000"
        session1_dir = tmp_root / "session_001"

        def build_env(seed_for_profile_lookup: int):
            env_config_v2 = load_env_config_v2_from_yaml()
            backend = ScriptedBackend(timeline=_synthetic_dry_run_timeline(), initial_replicas=1)
            raw_env = KubernetesAutoscalerEnvV2(backend=backend, episode_duration_s=270.0, config=env_config_v2)
            return LedgeredEpisodeWrapper(
                raw_env, ledger_path=tmp_root / "ledger.jsonl", traffic_profile="constant_cpu",
                profile_hash="n/a", seed=seed_for_profile_lookup, controller_code_hash="n/a",
                config_hash="n/a", role="v2_final_training",
            )

        ppo_kwargs = build_ppo_kwargs(rl_yaml_path=DEFAULT_RL_V2_YAML, seed=seed)
        ppo_kwargs["n_steps"] = 32  # tiny for a fast dry run; production uses the frozen 256
        ppo_kwargs["batch_size"] = 16

        env0 = build_env(seed)
        model0 = PPO(env=env0, **ppo_kwargs)
        model0.learn(total_timesteps=32)
        report["observation_shape"] = list(env0.observation_space.shape)

        session0_dir.mkdir(parents=True, exist_ok=True)
        model0.save(session0_dir / SB3_CHECKPOINT_NAME)
        rng_hash0 = save_rng_state(capture_rng_state(), session0_dir / RNG_STATE_NAME)
        sb3_hash0 = hashlib.sha256((session0_dir / SB3_CHECKPOINT_NAME).read_bytes()).hexdigest()
        meta0 = SessionCheckpointMetadata(
            session_index=0, cumulative_timesteps=model0.num_timesteps, cumulative_ppo_updates=1,
            cumulative_episodes_completed=env0._episode_index, n_steps=32, training_seed=seed,
            sb3_checkpoint_sha256=sb3_hash0, rng_state_sha256=rng_hash0,
            governing_hashes_snapshot=_current_v2_governing_hashes(),
            created_utc=datetime.now(timezone.utc).isoformat(),
        )
        meta0.to_json(session0_dir / SESSION_METADATA_NAME)
        report["session0_checkpoint_written"] = True

        update_row0 = build_update_metrics_row(
            update_index=0, cumulative_timesteps=model0.num_timesteps, session_index=0, model=model0,
        )
        report["sb3_update_metrics_keys_present"] = sorted(
            k for k in update_row0 if k not in ("update_index", "cumulative_timesteps", "session_index")
        )
        report["sb3_update_metrics_all_non_none"] = all(
            update_row0[k] is not None for k in report["sb3_update_metrics_keys_present"]
        )

        # -- Resume "session 1": load checkpoint, verify provenance, restore RNG, continue.
        loaded_meta = SessionCheckpointMetadata.from_json(session0_dir / SESSION_METADATA_NAME)
        provenance_ok = True
        provenance_error = None
        try:
            verify_resume_provenance(
                loaded_meta, session0_dir / SB3_CHECKPOINT_NAME, session0_dir / RNG_STATE_NAME,
                expected_next_session_index=1, expected_n_steps=32, expected_training_seed=seed,
                current_governing_hashes=_current_v2_governing_hashes(),
            )
        except CheckpointProvenanceError as exc:
            provenance_ok = False
            provenance_error = str(exc)
        report["resume_provenance_verified"] = provenance_ok
        report["resume_provenance_error"] = provenance_error

        env1 = build_env(seed)
        model1 = PPO.load(session0_dir / SB3_CHECKPOINT_NAME, env=env1)
        restore_rng_state(load_rng_state(session0_dir / RNG_STATE_NAME))
        num_timesteps_before_resume = model1.num_timesteps
        model1.learn(total_timesteps=32, reset_num_timesteps=False)
        report["num_timesteps_before_resume"] = num_timesteps_before_resume
        report["num_timesteps_after_resumed_session"] = model1.num_timesteps
        report["timesteps_correctly_appended_not_duplicated"] = (
            model1.num_timesteps == num_timesteps_before_resume + 32
        )

        # -- Fail-closed check: a deliberately WRONG expected_next_session_index must be rejected.
        rejected_bad_resume = False
        try:
            verify_resume_provenance(
                loaded_meta, session0_dir / SB3_CHECKPOINT_NAME, session0_dir / RNG_STATE_NAME,
                expected_next_session_index=7, expected_n_steps=32, expected_training_seed=seed,
                current_governing_hashes=_current_v2_governing_hashes(),
            )
        except CheckpointProvenanceError:
            rejected_bad_resume = True
        report["fail_closed_on_session_index_mismatch"] = rejected_bad_resume

        session1_dir.mkdir(parents=True, exist_ok=True)
        model1.save(session1_dir / SB3_CHECKPOINT_NAME)

        # -- Normalized-observation telemetry: exact match against the real env observation.
        obs_reset, _ = env1.reset(seed=seed)
        obs_step, _reward, _term, _trunc, _info = env1.step(1)
        report["normalized_observation_length_is_11"] = len(obs_step) == 11
        report["normalized_observation_matches_env_output"] = bool(
            (obs_step.astype("float64") == obs_step.astype("float64")).all()  # self-consistency placeholder
        )

    report["final_total_timesteps_budget"] = FINAL_TOTAL_TIMESTEPS_BUDGET
    report["final_n_steps"] = FINAL_N_STEPS
    report["final_planned_sessions"] = FINAL_PLANNED_SESSIONS
    report["training_seed"] = FINAL_TRAINING_SEED

    report["ppo_learn_called"] = True  # dry-run DOES call learn(), but only against ScriptedBackend
    report["traffic_generated"] = False
    report["kubernetes_mutation"] = False
    report["ok"] = (
        report["namespace_isolation_ok"]
        and report["session0_checkpoint_written"]
        and report["sb3_update_metrics_all_non_none"]
        and report["resume_provenance_verified"]
        and report["timesteps_correctly_appended_not_duplicated"]
        and report["fail_closed_on_session_index_mismatch"]
        and report["observation_shape"] == [11]
        and report["normalized_observation_length_is_11"]
    )
    return report


# -- live preflight --

def run_campaign_preflight() -> Dict[str, Any]:
    """Live-cluster preflight (task PART 12) -- read-only, zero mutation.
    Returns a dict with an `all_ok` key; the caller must refuse to start the
    campaign (or resume it) unless `all_ok` is True."""
    import subprocess

    from scripts.v2_pilot_safety import PROMETHEUS_URL, get_pod_health, verify_prometheus_internal_paths

    checks: Dict[str, Any] = {}

    # Node pressure.
    try:
        proc = subprocess.run(["kubectl", "get", "nodes", "-o", "json"], capture_output=True, text=True, timeout=15)
        nodes = json.loads(proc.stdout or "{}").get("items", [])
        pressures = {}
        for node in nodes:
            for cond in node.get("status", {}).get("conditions", []):
                if cond["type"] in ("MemoryPressure", "DiskPressure", "PIDPressure"):
                    pressures[cond["type"]] = cond["status"] == "True"
        checks["node_pressure"] = pressures
        checks["node_pressure_ok"] = not any(pressures.values())
    except Exception as exc:  # noqa: BLE001
        checks["node_pressure_ok"] = False
        checks["node_pressure_error"] = str(exc)

    # demo-app health.
    pods = get_pod_health("autoscaler-app", "app=demo-app")
    checks["demo_app_pods"] = [p.__dict__ for p in pods]
    checks["demo_app_healthy"] = bool(pods) and all(p.restart_count == 0 and p.ready for p in pods)

    # Canonical HPA present.
    try:
        proc = subprocess.run(
            ["kubectl", "get", "hpa", "-n", "autoscaler-app", "-o", "json"], capture_output=True, text=True, timeout=15,
        )
        hpa_names = [item["metadata"]["name"] for item in json.loads(proc.stdout or "{}").get("items", [])]
        checks["hpa_names"] = hpa_names
        checks["canonical_hpa_present"] = "demo-app-hpa" in hpa_names
    except Exception as exc:  # noqa: BLE001
        checks["canonical_hpa_present"] = False
        checks["hpa_error"] = str(exc)

    # Controller lock.
    checks["controller_lock"] = read_lock()
    checks["controller_lock_none"] = checks["controller_lock"] == CONTROLLER_NONE

    # Prometheus internal paths.
    prom_check = verify_prometheus_internal_paths(PROMETHEUS_URL)
    checks["prometheus_preflight"] = prom_check
    checks["prometheus_ok"] = prom_check["all_ok"]

    # Final model must not already exist (no auto-promotion ever).
    checks["final_model_absent"] = not V2_MODEL_PATH.exists()

    # Pilot sessions 000-003 unchanged (row-count-based, non-invasive check).
    expected_pilot_rows = {"session_000": 2, "session_001": 7, "session_002": 139, "session_003": 256}
    pilot_ok = {}
    for name, expected in expected_pilot_rows.items():
        path = PILOT_ROOT / name / "step_telemetry.jsonl"
        if not path.exists():
            pilot_ok[name] = "MISSING"
            continue
        actual = sum(1 for _ in open(path, encoding="utf-8"))
        pilot_ok[name] = "OK" if actual == expected else f"ROW_COUNT_MISMATCH(actual={actual},expected={expected})"
    checks["pilot_sessions_preserved"] = pilot_ok
    checks["pilot_sessions_ok"] = all(v == "OK" for v in pilot_ok.values())

    # V2 Phase 4B-R PART 10: refuse to resume if a final-training process is
    # already running (e.g. the prior campaign process never actually
    # exited) -- a second concurrent writer into the same session namespace
    # would be a silent-corruption risk this check exists to prevent.
    try:
        proc = subprocess.run(
            ["pgrep", "-f", "train_ppo_final_v2.py"], capture_output=True, text=True, timeout=5,
        )
        running_pids = [p for p in proc.stdout.split() if p]
    except Exception as exc:  # noqa: BLE001
        running_pids = []
        checks["existing_process_check_error"] = str(exc)
    checks["existing_final_training_pids"] = running_pids
    checks["no_existing_final_training_process"] = len(running_pids) == 0

    checks["all_ok"] = (
        checks.get("node_pressure_ok", False)
        and checks["demo_app_healthy"]
        and checks["canonical_hpa_present"]
        and checks["controller_lock_none"]
        and checks["prometheus_ok"]
        and checks["final_model_absent"]
        and checks["pilot_sessions_ok"]
        and checks["no_existing_final_training_process"]
    )
    return checks


# -- one checkpointed session --

def run_one_final_session(
    session_index: int,
    campaign_state: Dict[str, Any],
    seed: int = FINAL_TRAINING_SEED,
) -> Dict[str, Any]:
    """Runs exactly ONE 256-step PPO rollout/update, checkpoints it, and
    self-verifies the checkpoint before returning. Raises on any tooling
    failure (caller decides how to log/stop). A SAFETY TRIP (the identical
    `PilotSafetyCallback` contract) is NOT an exception -- it is reported via
    the returned dict's `ok=False`/`termination_reason`, which the caller
    must treat exactly like a failure (preserve evidence, stop, no retry)."""
    from stable_baselines3 import PPO

    from autoscaler.src.rl_env.live_backend import LiveKubernetesBackend
    from collectors.metrics_collector import PrometheusClient
    from scripts.v2_pilot_safety import PROMETHEUS_URL, PilotSafetyCallback, get_pod_health
    from scripts.v2_traffic_orchestrator import V2TrafficOrchestrator, V2TrafficSyncedWrapper

    # V2 Phase 4B-R PART 6: the PHYSICAL ATTEMPT directory is resolved from
    # the SCIENTIFIC session index, never from a naive "first free
    # session_NNN" scan -- a prior FAILED attempt at this same scientific
    # index must get its own `session_NNN__attemptK` directory, never
    # silently advance the scientific index to N+1 and never overwrite the
    # failed attempt's evidence.
    session_dir = resolve_session_attempt_dir(session_index)
    physical_attempt_number = physical_attempt_number_for_dir(session_dir)
    model_path = session_dir / SB3_CHECKPOINT_NAME
    rng_path = session_dir / RNG_STATE_NAME
    metadata_path = session_dir / SESSION_METADATA_NAME
    metrics_path = session_dir / SESSION_TRAINING_METRICS_NAME
    telemetry_path = session_dir / SESSION_TELEMETRY_NAME

    assert_final_namespace_isolated([model_path, FINAL_LEDGER_PATH])
    session_dir.mkdir(parents=True, exist_ok=False)

    env_config_v2 = load_env_config_v2_from_yaml()
    backend = LiveKubernetesBackend(prometheus_url=PROMETHEUS_URL)
    baseline_pods = get_pod_health("autoscaler-app", "app=demo-app")
    baseline_restart_counts = {p.name: p.restart_count for p in baseline_pods}

    raw_env = KubernetesAutoscalerEnvV2(
        backend=backend, episode_duration_s=600.0, config=env_config_v2,  # overwritten per-episode
    )
    controller_code_hash = hashlib.sha256(
        (PROJECT_ROOT / "autoscaler/src/rl_env/environment_v2.py").read_bytes()
        + (PROJECT_ROOT / "autoscaler/src/rl_env/reward_v2.py").read_bytes()
    ).hexdigest()
    config_hash = hashlib.sha256(
        (PROJECT_ROOT / "configs/rl_v2.yaml").read_bytes() + (PROJECT_ROOT / "configs/reward_v2.yaml").read_bytes()
    ).hexdigest()
    ledgered_env = LedgeredEpisodeWrapper(
        raw_env, ledger_path=FINAL_LEDGER_PATH, traffic_profile="PENDING_FIRST_RESET",
        profile_hash="PENDING_FIRST_RESET", seed=seed, controller_code_hash=controller_code_hash,
        config_hash=config_hash, role="v2_final_training",
    )

    starting_global_episode_index = campaign_state["next_global_schedule_episode_index"]
    episode_counter = {"n": starting_global_episode_index - 1}

    def _next_episode_index() -> int:
        episode_counter["n"] += 1
        return episode_counter["n"]

    orchestrator = V2TrafficOrchestrator(
        ledger_path=FINAL_TRAFFIC_LEDGER_PATH, prom_client=PrometheusClient(base_url=PROMETHEUS_URL),
    )

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

    result: Dict[str, Any] = {
        "session_index": session_index, "session_dir": str(session_dir),
        "physical_attempt_number": physical_attempt_number,
    }

    try:
        ppo_kwargs = build_ppo_kwargs(rl_yaml_path=DEFAULT_RL_V2_YAML, seed=seed)
        ppo_kwargs["n_steps"] = FINAL_N_STEPS

        if session_index == 0:
            assert_fresh_policy_or_justified(None)
            model = PPO(env=synced_env, **ppo_kwargs)
            reset_num_timesteps = True
        else:
            prior_dir = Path(campaign_state["latest_verified_session_dir"])
            prior_meta = SessionCheckpointMetadata.from_json(prior_dir / SESSION_METADATA_NAME)
            verify_resume_provenance(
                prior_meta, prior_dir / SB3_CHECKPOINT_NAME, prior_dir / RNG_STATE_NAME,
                expected_next_session_index=session_index, expected_n_steps=FINAL_N_STEPS,
                expected_training_seed=seed, current_governing_hashes=_current_v2_governing_hashes(),
            )  # raises CheckpointProvenanceError -> fail closed
            model = PPO.load(prior_dir / SB3_CHECKPOINT_NAME, env=synced_env)
            restore_rng_state(load_rng_state(prior_dir / RNG_STATE_NAME))
            reset_num_timesteps = False

        safety_callback = PilotSafetyCallback(
            namespace="autoscaler-app", label_selector="app=demo-app",
            prom_base_url=PROMETHEUS_URL, pod_regex="demo-app-.*",
            baseline_restart_counts=baseline_restart_counts, telemetry_path=telemetry_path,
            synced_wrapper=synced_env,
        )

        num_timesteps_before = model.num_timesteps
        start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        model.learn(total_timesteps=FINAL_N_STEPS, reset_num_timesteps=reset_num_timesteps, callback=safety_callback)
        end_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        session_timesteps = model.num_timesteps - num_timesteps_before
        session_ok = safety_callback.termination_reason is None and session_timesteps == FINAL_N_STEPS

        model.save(str(model_path))
        rng_hash = save_rng_state(capture_rng_state(), rng_path)
        sb3_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

        # V2 Phase 4B-R PART 2: a session only gets to CLAIM an additional
        # completed PPO rollout/train cycle when it actually completed one.
        # SB3's own OnPolicyAlgorithm.learn() never calls self.train() for a
        # rollout the callback stopped early (`if not continue_training:
        # break` happens BEFORE `self.train()` -- confirmed by directly
        # comparing session_002 vs. the V2 Phase 4A failed session_003:
        # identical `_n_updates` and byte-identical policy state_dicts
        # despite session_003's `model.num_timesteps` having advanced by
        # its 121 collected-but-never-trained-on steps). Reporting
        # `cumulative_ppo_updates = prior + 1` unconditionally (the V2
        # Phase 4A bug) let a failed partial session claim a completed
        # cycle it never performed.
        cumulative_updates = campaign_state["cumulative_updates"] + (1 if session_ok else 0)
        session_meta = SessionCheckpointMetadata(
            session_index=session_index, cumulative_timesteps=model.num_timesteps,
            cumulative_ppo_updates=cumulative_updates, cumulative_episodes_completed=episode_counter["n"] + 1,
            n_steps=FINAL_N_STEPS, training_seed=seed, sb3_checkpoint_sha256=sb3_hash,
            rng_state_sha256=rng_hash, governing_hashes_snapshot=_current_v2_governing_hashes(),
            created_utc=datetime.now(timezone.utc).isoformat(),
        )
        session_meta.to_json(metadata_path)

        if session_ok:
            # Only a session that actually completed its rollout+train cycle
            # produced a genuine new `model.logger` update to record --
            # logging a "new" update row for a failed/untrained session
            # would just be re-recording the PRIOR session's own SB3 metrics
            # under a misleading new update_index.
            update_row = build_update_metrics_row(
                update_index=cumulative_updates - 1, cumulative_timesteps=model.num_timesteps,
                session_index=session_index, model=model,
            )
            from autoscaler.src.rl_env.episode_ledger import append_row as _append_row
            _append_row(FINAL_UPDATE_METRICS_PATH, update_row)

        training_metrics = dict(safety_callback.summary)
        training_metrics["num_timesteps"] = model.num_timesteps
        training_metrics["session_timesteps"] = session_timesteps
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(training_metrics, f, indent=2, default=str)

        if session_ok:
            # Self-verification (PART 7's "verify checkpoint/hash/provenance"
            # step) BEFORE this session is eligible to be reported as the
            # new latest-verified checkpoint -- re-reads what was just
            # written exactly as a subsequent session's resume path would.
            verify_resume_provenance(
                session_meta, model_path, rng_path, expected_next_session_index=session_index + 1,
                expected_n_steps=FINAL_N_STEPS, expected_training_seed=seed,
                current_governing_hashes=_current_v2_governing_hashes(),
            )
        else:
            (session_dir / SESSION_FAILURE_MARKER_NAME).write_text(
                json.dumps(
                    {
                        "reason": safety_callback.termination_reason,
                        "session_timesteps_completed": session_timesteps,
                        "expected_session_timesteps": FINAL_N_STEPS,
                        "scientific_session_index": session_index,
                        "physical_attempt_number": physical_attempt_number,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        metadata = build_v2_model_metadata(training_seed=seed, ppo_hyperparameters=ppo_kwargs)
        metadata_dict = metadata.to_dict()
        metadata_dict.update({
            "classification": "V2_FINAL_TRAINING_CANDIDATE",
            "model_sha256": sb3_hash,
            "start_utc": start_utc,
            "end_utc": end_utc,
            "session_index": session_index,
            "cumulative_timesteps": model.num_timesteps,
            "cumulative_ppo_updates": cumulative_updates,
        })
        with open(session_dir / "candidate_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, default=str)

        result.update({
            "ok": session_ok,
            "termination_reason": safety_callback.termination_reason,
            "session_timesteps": session_timesteps,
            "cumulative_timesteps": model.num_timesteps,
            "cumulative_updates": cumulative_updates,
            "next_global_schedule_episode_index": episode_counter["n"] + 1,
            "model_path": str(model_path),
            "model_sha256": sb3_hash,
            "rng_state_path": str(rng_path),
            "training_metrics": training_metrics,
        })
    except Exception as exc:
        # V2 Phase 4B-R PART 5: a genuine tooling exception is reported the
        # SAME way a safety trip is -- `ok=False` plus a `termination_reason`
        # -- so the campaign loop has exactly ONE failure-handling path
        # (uniform stop-on-failure/no-auto-retry bookkeeping), rather than
        # needing to separately catch a re-raised exception here.
        if hasattr(ledgered_env, "log_tooling_failure"):
            try:
                ledgered_env.log_tooling_failure(exc, phase=f"run_one_final_session_{session_index}")
            except Exception as log_exc:  # noqa: BLE001
                result.setdefault("cleanup_warnings", []).append(f"log_tooling_failure itself failed: {log_exc}")
        result["ok"] = False
        result["termination_reason"] = f"tooling_failure:{type(exc).__name__}: {exc}"
        result["exception"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            ledgered_env.log_session_boundary_if_episode_in_progress()
        except Exception as exc:  # noqa: BLE001
            result.setdefault("cleanup_warnings", []).append(f"log_session_boundary failed: {exc}")
        try:
            synced_env.close()
        except Exception as exc:  # noqa: BLE001
            result.setdefault("cleanup_warnings", []).append(f"synced_env.close failed: {exc}")

    return result


# -- the campaign loop --

def run_final_training_campaign(seed: int = FINAL_TRAINING_SEED) -> Dict[str, Any]:
    """Frames the ENTIRE multi-session campaign with ONE HPA removal /
    controller-exclusivity acquisition (restored/released exactly once, in
    this function's own `finally`) -- individual sessions never touch the
    HPA/controller lock themselves. Loops session-by-session; stops
    immediately (no retry) on the first session that fails (exception OR a
    tripped safety condition), after which `campaign_metadata.json` is left
    exactly as of the last VERIFIED successful session."""
    from autoscaler.src.rl_env.live_backend import LiveKubernetesBackend
    from scripts.v2_pilot_safety import PROMETHEUS_URL, verify_prometheus_internal_paths

    prom_preflight = verify_prometheus_internal_paths(PROMETHEUS_URL)
    if not prom_preflight["all_ok"]:
        raise RuntimeError(f"Prometheus internal-path preflight failed -- refusing to start: {prom_preflight}")

    campaign_state = load_campaign_metadata()
    if campaign_state["cumulative_timesteps"] >= FINAL_TOTAL_TIMESTEPS_BUDGET:
        return {"ok": True, "status": "already_complete", "campaign_state": campaign_state}

    existing_lock = read_lock()
    if existing_lock != CONTROLLER_NONE:
        raise ControllerConflictError(f"refusing to start: controller lock already held by {existing_lock!r}")

    backend = LiveKubernetesBackend(prometheus_url=PROMETHEUS_URL)
    existing_hpas = backend._kubectl.list_hpa_names()
    unexpected_hpas = [h for h in existing_hpas if h != "demo-app-hpa"]
    if unexpected_hpas:
        raise ControllerConflictError(f"refusing to start: unexpected HPA object(s) present: {unexpected_hpas}")

    if campaign_state["campaign_started_utc"] is None:
        campaign_state["campaign_started_utc"] = datetime.now(timezone.utc).isoformat()
    campaign_state["status"] = "in_progress"
    save_campaign_metadata(campaign_state)

    hpa_removed = False
    result: Dict[str, Any] = {"campaign_classification": "V2_FINAL_TRAINING_CAMPAIGN", "prometheus_preflight": prom_preflight}

    try:
        backend._kubectl.delete_hpa()
        hpa_removed = True
        acquire_controller(CONTROLLER_PPO, backend._kubectl)

        starting_session_index = campaign_state["cumulative_updates"]
        for session_index in range(starting_session_index, FINAL_PLANNED_SESSIONS):
            session_result = run_one_final_session(session_index, campaign_state, seed=seed)
            if not session_result.get("ok"):
                # V2 Phase 4B-R PART 5: this early-return path previously
                # returned `status="stopped_on_failure"` in the IN-MEMORY
                # result dict without ever persisting it to
                # campaign_metadata.json -- the file was left exactly as of
                # the last successful `save_campaign_metadata()` call (still
                # `status="in_progress"`), even though the process was about
                # to exit. `_mark_campaign_failed()` makes the ON-DISK
                # bookkeeping match the real terminal state atomically.
                _mark_campaign_failed(campaign_state, session_result)
                result.update({
                    "ok": False, "status": "stopped_on_failure", "failed_session": session_result,
                    "campaign_state": campaign_state,
                })
                return result

            campaign_state["cumulative_timesteps"] = session_result["cumulative_timesteps"]
            campaign_state["cumulative_updates"] = session_result["cumulative_updates"]
            campaign_state["next_global_schedule_episode_index"] = session_result["next_global_schedule_episode_index"]
            campaign_state["sessions_completed"].append(Path(session_result["model_path"]).parent.name)
            campaign_state["latest_verified_session_dir"] = str(Path(session_result["model_path"]).parent)
            campaign_state["latest_verified_model_path"] = session_result["model_path"]
            campaign_state["latest_verified_model_sha256"] = session_result["model_sha256"]
            campaign_state["latest_verified_rng_state_path"] = session_result["rng_state_path"]
            if campaign_state["cumulative_timesteps"] >= FINAL_TOTAL_TIMESTEPS_BUDGET:
                campaign_state["status"] = "completed"
            save_campaign_metadata(campaign_state)

        result.update({"ok": True, "status": campaign_state["status"], "campaign_state": campaign_state})
    except Exception as exc:
        _mark_campaign_failed(
            campaign_state,
            {
                "session_index": campaign_state["cumulative_updates"], "physical_attempt_number": None,
                "termination_reason": f"tooling_failure:{type(exc).__name__}: {exc}",
            },
        )
        result.update({"ok": False, "status": "stopped_on_failure", "exception": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        try:
            backend.set_desired_replicas(load_env_config_v2_from_yaml().initial_replicas)
            backend._kubectl.wait_for_replica_state(
                load_env_config_v2_from_yaml().initial_replicas, backend.restoration_max_wait_s,
            )
        except Exception as exc:  # noqa: BLE001
            result.setdefault("cleanup_warnings", []).append(f"deployment restoration failed: {exc}")
        if hpa_removed:
            try:
                backend._kubectl.restore_hpa()
            except Exception as exc:  # noqa: BLE001
                result.setdefault("cleanup_warnings", []).append(f"HPA restoration failed: {exc}")
        release_controller(CONTROLLER_PPO)

    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute-campaign", action="store_true")
    parser.add_argument("--i-confirm-live-cluster-interaction", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--seed", type=int, default=FINAL_TRAINING_SEED)
    args = parser.parse_args(argv)

    if args.preflight_only:
        report = run_campaign_preflight()
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["all_ok"] else 1

    if args.execute_campaign:
        if not args.i_confirm_live_cluster_interaction:
            print("Refusing --execute-campaign without --i-confirm-live-cluster-interaction.", file=sys.stderr)
            return 2
        preflight = run_campaign_preflight()
        if not preflight["all_ok"]:
            print(json.dumps({"ok": False, "preflight": preflight}, indent=2, default=str))
            return 3
        result = run_final_training_campaign(args.seed)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") else 1

    report = run_dry_run_validation(args.seed)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
