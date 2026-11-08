#!/usr/bin/env python3
"""scripts/evaluate_v2_campaign.py

V2 Phase 5B PART A: the evaluation CAMPAIGN runner implementing the
pre-registered design in
`persistent-data/rl_training_v2/final_min2/EVALUATION_PREREGISTRATION.json`
(CORR-60). A NEW file -- does not modify `scripts/evaluate_ppo_vs_hpa_v2.py`,
`autoscaler/src/rl_env/controller_exclusivity.py`, or any of the 43 files
listed in `FREEZE_MANIFEST.json` (the live training contract); it IMPORTS
and reuses `run_live_v2_episode()`/`run_hpa_v2_episode()` unmodified.

Runs the pre-registered 3 arms x 3 profiles x 3 repeats = 27 evaluation
episodes in a deterministic cyclic-Latin-square order (so no arm is
systematically first/last within a (profile, repeat) block), against the
3 held-out `configs/traffic_profiles/v2_eval/` profiles ONLY -- resolved
through `resolve_eval_profile_metadata()` below, which is firewalled
against `EVAL_PROFILE_ALLOWLIST`, completely separate from (and unable to
select) any training profile.

Classification: like every other V2 execution script, `--dry-run` (the
default) never touches the cluster -- it runs the full planning/resume/
retry/cleanup control flow against INJECTED FAKES (see
`build_dry_run_context()`), proving the logic offline. `--execute` wires
the SAME control flow to the real, already-tested building blocks
(`run_live_v2_episode`, `run_hpa_v2_episode`,
`autoscaler.src.rl_env.live_backend.LiveKubernetesBackend`,
`scripts.traffic_orchestrator`'s generic generator/route machinery) --
this phase prepares that wiring but does not invoke `--execute` (training
is live; see the phase's own protection constraints).

The `hpa_v2_cpu_only` arm (the third, CPU-only HPA arm added by Phase 5A)
has NO registered controller identity in `controller_exclusivity.py` yet
(`CONTROLLER_CPU_HPA`/`CONTROLLER_CPU_MEMORY_HPA` exist; no
`CONTROLLER_CPU_ONLY_HPA_V2`). Rather than edit that frozen-for-training
file, every attempt at this arm fails closed with
`CpuOnlyArmNotYetSupportedError`, naming the EXACT one-line change Phase
5C needs to make (see that exception's message and this phase's report).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.traffic_orchestrator as to  # noqa: E402
from autoscaler.src.rl_env.controller_exclusivity import (  # noqa: E402
    CONTROLLER_NONE,
    V2_HPA_MANIFEST_PATH,
    V2_HPA_NAME,
    read_lock,
)
from loadgen.traffic_generator import parse_traffic_profile  # noqa: E402

# V2 Phase 5C-A: the error classes and the real per-arm episode functions
# now live in scripts/eval_v2_episodes.py (a new file) -- re-exported here
# unchanged so every existing caller/test of this module (`campaign_mod.
# ToolingFailureError`, etc.) keeps working without modification. This
# module's own OWN responsibility stays campaign orchestration (plan,
# ledger, resume, retry, cleanup); scripts/eval_v2_episodes.py owns what
# one arm's one episode actually does against a live cluster.
from scripts.eval_v2_episodes import (  # noqa: E402
    CpuOnlyArmNotYetSupportedError,
    SafetyTerminationError,
    ToolingFailureError,
    episode_result_to_run_series_dict,
    resolve_cpu_only_controller_identity,
    run_hpa_v2_cpu_memory_eval_episode,
    run_hpa_v2_cpu_only_eval_episode,
    run_ppo_v2_eval_episode,
)

# -- firewall: the 3 held-out evaluation profiles ONLY, never the training
# profiles in configs/traffic_profiles/v2/ (V2_TRAINING_PROFILE_ALLOWLIST) --
EVAL_PROFILE_DIR = PROJECT_ROOT / "configs" / "traffic_profiles" / "v2_eval"
EVAL_PROFILE_ALLOWLIST = ("eval_cpu_step", "eval_memory_ramp", "eval_mixed_burst")

PREREGISTRATION_PATH = (
    PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final_min2" / "EVALUATION_PREREGISTRATION.json"
)
EVAL_OUTPUT_ROOT = PROJECT_ROOT / "persistent-data" / "evaluation_v2"

ARM_ID_MAP = {"PPO_V2": "ppo_v2", "HPA_V2_CPU_MEMORY": "hpa_v2_cpu_memory", "HPA_CPU_ONLY": "hpa_v2_cpu_only"}
ARM_IDS = ("ppo_v2", "hpa_v2_cpu_memory", "hpa_v2_cpu_only")


class EvalProfileFirewallViolation(RuntimeError):
    pass


def resolve_eval_profile_metadata(profile_name: str) -> "to.ProfileMetadata":
    """V2 Phase 5B: the evaluation-only equivalent of
    `v2_traffic_orchestrator.resolve_v2_profile_metadata()` -- fails closed
    against `EVAL_PROFILE_ALLOWLIST`/`EVAL_PROFILE_DIR`, structurally
    unable to select any training profile (different allowlist, different
    directory, no import of `V2_TRAINING_PROFILE_ALLOWLIST` at all)."""
    if profile_name not in EVAL_PROFILE_ALLOWLIST:
        raise EvalProfileFirewallViolation(f"{profile_name!r} is not in EVAL_PROFILE_ALLOWLIST {EVAL_PROFILE_ALLOWLIST}")
    path = EVAL_PROFILE_DIR / f"{profile_name}.yaml"
    if not path.exists():
        raise EvalProfileFirewallViolation(f"authorized eval profile file missing on disk: {path}")
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    profile = parse_traffic_profile(path)
    return to.ProfileMetadata(
        name=profile_name, path=path, sha256=actual_sha256, seed=profile.seed,
        duration_seconds=profile.total_duration, n_phases=len(profile.phases),
    )


@dataclass
class V2EvalTrafficOrchestrator(to.TrafficOrchestrator):
    """V2 Phase 5B: overrides ONLY `start_episode_traffic` to resolve
    against `resolve_eval_profile_metadata()` instead of the base class's
    frozen V1 six-name registry -- every other method (route lifecycle,
    drain, provenance, cleanup) is inherited unmodified, exactly the same
    pattern `v2_traffic_orchestrator.V2TrafficOrchestrator` already uses
    for V2 training."""

    def start_episode_traffic(self, profile_name: str, episode_id: str) -> "to.GeneratorHandle":
        profile = resolve_eval_profile_metadata(profile_name)
        run_id = to.resolve_next_available_run_id(episode_id, self.traffic_base_dir)
        route = self.ensure_route_established(run_id)
        if not to.verify_route_alive(route, self.http_get):
            raise to.RouteEstablishmentFailure(f"route {route.target_url} not alive before episode {episode_id}")
        handle = to.start_generator(profile, route, run_id, popen_factory=self.popen_factory, clock=self.clock)
        if not to.verify_generator_started(handle, sleep_fn=self.sleep_fn):
            to.stop_generator(handle)
            to.record_traffic_provenance(
                handle, episode_id=episode_id, session_index=self.session_index,
                exit_code=handle.process.poll(), reason="failed_to_start", ledger_path=self.ledger_path,
            )
            raise to.TrafficGeneratorFailure(f"generator failed to start for eval profile {profile_name!r}, episode {episode_id}")
        return handle


# ---------------------------------------------------------------------------
# Run plan (Latin-square order)
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    run_index: int
    block_index: int
    position_in_block: int
    profile: str
    repeat_index: int
    arm_id: str

    @property
    def run_id(self) -> str:
        return f"{self.profile}__repeat{self.repeat_index}__{self.arm_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_index": self.run_index, "run_id": self.run_id, "block_index": self.block_index,
            "position_in_block": self.position_in_block, "profile": self.profile,
            "repeat_index": self.repeat_index, "arm_id": self.arm_id,
        }


def load_preregistration(path: Path = PREREGISTRATION_PATH) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_run_plan(preregistration: Dict[str, Any]) -> List[RunRecord]:
    """9 blocks (3 profiles x 3 repeats) x 3 arms = 27 runs. Within block
    `b`, the arm order is `ARM_IDS` cyclically rotated left by `b % 3` --
    a cyclic Latin square of order 3: across any 3 consecutive blocks,
    each arm occupies every position (first/second/third) exactly once,
    so no arm is systematically advantaged/disadvantaged by always
    running first (e.g. against a warmer/cooler cluster) or last."""
    profiles = [Path(p).stem for p in preregistration["workloads"]["profiles"]]
    repeats = preregistration["workloads"]["repeats_per_profile_per_arm"]
    arms = list(ARM_IDS)
    n_arms = len(arms)

    blocks = [(profile, repeat) for profile in profiles for repeat in range(repeats)]

    runs: List[RunRecord] = []
    run_index = 0
    for block_index, (profile, repeat) in enumerate(blocks):
        shift = block_index % n_arms
        rotated = arms[shift:] + arms[:shift]
        for position_in_block, arm_id in enumerate(rotated):
            runs.append(RunRecord(run_index, block_index, position_in_block, profile, repeat, arm_id))
            run_index += 1
    return runs


def write_run_plan(
    runs: List[RunRecord], campaign_id: str, preregistration: Dict[str, Any], output_path: Path,
) -> Dict[str, Any]:
    prereg_sha256 = hashlib.sha256(PREREGISTRATION_PATH.read_bytes()).hexdigest()
    plan = {
        "campaign_id": campaign_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_path": str(PREREGISTRATION_PATH.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": prereg_sha256,
        "arms": list(ARM_IDS),
        "profiles": sorted({r.profile for r in runs}),
        "repeats_per_profile_per_arm": preregistration["workloads"]["repeats_per_profile_per_arm"],
        "n_blocks": len({r.block_index for r in runs}),
        "n_runs": len(runs),
        "order_rule": "cyclic Latin square: block b's arm order = ARM_IDS rotated left by b % len(ARM_IDS)",
        "runs": [r.to_dict() for r in runs],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    return plan


# ---------------------------------------------------------------------------
# Ledger / run_status (resumable, append-only)
# ---------------------------------------------------------------------------

TERMINAL_STATUSES = ("VALID", "SAFETY_TERMINATION", "FAILED_CLOSED", "FAILED_AFTER_RETRY")


def load_ledger(ledger_path: Path) -> List[Dict[str, Any]]:
    if not ledger_path.exists():
        return []
    rows = []
    with open(ledger_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_ledger(ledger_path: Path, entry: Dict[str, Any]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def completed_run_ids(ledger: List[Dict[str, Any]]) -> set:
    return {row["run_id"] for row in ledger if row.get("status") in TERMINAL_STATUSES}


def write_run_status(path: Path, status: str, current_run: Optional[Dict[str, Any]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "current_run": current_run},
            f, indent=2,
        )


# ---------------------------------------------------------------------------
# Execution context (every cluster-touching dependency injectable)
# ---------------------------------------------------------------------------


@dataclass
class CampaignContext:
    """`orchestrator` needs only `start_episode_traffic`/
    `stop_episode_traffic`/`drain_server_side_requests` -- satisfied by
    both the real `V2EvalTrafficOrchestrator` and `_FakeOrchestrator`
    below (structural typing, not inheritance)."""

    orchestrator: Any
    preflight_fn: Callable[[], Dict[str, Any]]
    ppo_episode_fn: Callable[["to.ProfileMetadata", str], Dict[str, Any]]
    hpa_episode_fn: Callable[["to.ProfileMetadata", str], Dict[str, Any]]
    cleanup_fn: Callable[[], Dict[str, Any]]
    settle_fn: Callable[[float], None]
    settle_seconds: float = 120.0
    runs_dir: Path = field(default_factory=lambda: EVAL_OUTPUT_ROOT / "_unset" / "runs")
    cpu_only_episode_fn: Optional[Callable[["to.ProfileMetadata", str], Dict[str, Any]]] = None


def run_one_attempt(run_record: RunRecord, ctx: CampaignContext, attempt_number: int) -> Dict[str, Any]:
    episode_id = f"{run_record.run_id}__attempt{attempt_number}"
    started_utc = datetime.now(timezone.utc).isoformat()
    entry: Dict[str, Any] = {
        "attempt_id": str(uuid.uuid4()), "run_id": run_record.run_id, "run_index": run_record.run_index,
        "profile": run_record.profile, "repeat_index": run_record.repeat_index, "arm_id": run_record.arm_id,
        "attempt_number": attempt_number, "episode_id": episode_id, "started_utc": started_utc,
    }

    handle = None
    try:
        preflight = ctx.preflight_fn()
        if not preflight.get("ok"):
            raise ToolingFailureError(f"preflight failed: {preflight}")

        if run_record.arm_id == "hpa_v2_cpu_only":
            # V2 Phase 5C-A: the REAL, getattr-based readiness check --
            # never cluster-touching -- checked BEFORE any profile
            # resolution or traffic start. Raises CpuOnlyArmNotYetSupportedError
            # (fails closed) until controller_exclusivity.py gains
            # CONTROLLER_CPU_ONLY_HPA_V2 (Phase 5C); a test that monkeypatches
            # that constant in makes this call succeed with NO code change.
            resolve_cpu_only_controller_identity()

        profile_meta = resolve_eval_profile_metadata(run_record.profile)
        handle = ctx.orchestrator.start_episode_traffic(run_record.profile, episode_id)
        try:
            if run_record.arm_id == "ppo_v2":
                result = ctx.ppo_episode_fn(profile_meta, episode_id)
            elif run_record.arm_id == "hpa_v2_cpu_only":
                result = ctx.cpu_only_episode_fn(profile_meta, episode_id)
            else:
                result = ctx.hpa_episode_fn(profile_meta, episode_id)
        finally:
            ctx.orchestrator.stop_episode_traffic(handle, episode_id)
            ctx.orchestrator.drain_server_side_requests()

        cleanup_result = ctx.cleanup_fn()
        ctx.settle_fn(ctx.settle_seconds)
        entry.update({"status": "VALID", "result": result, "cleanup": cleanup_result})

    except CpuOnlyArmNotYetSupportedError as exc:
        cleanup_result = ctx.cleanup_fn()
        entry.update({"status": "FAILED_CLOSED", "error": str(exc), "cleanup": cleanup_result})
    except SafetyTerminationError as exc:
        cleanup_result = ctx.cleanup_fn()
        entry.update({"status": "SAFETY_TERMINATION", "error": str(exc), "cleanup": cleanup_result})
    except ToolingFailureError as exc:
        cleanup_result = ctx.cleanup_fn()
        entry.update({"status": "TOOLING_FAILURE", "error": str(exc), "cleanup": cleanup_result})
    except Exception as exc:  # noqa: BLE001 -- never silently drop an unclassified failure
        cleanup_result = ctx.cleanup_fn()
        entry.update({"status": "TOOLING_FAILURE", "error": f"unclassified: {exc}", "cleanup": cleanup_result})

    entry["ended_utc"] = datetime.now(timezone.utc).isoformat()

    run_dir = ctx.runs_dir / f"{run_record.run_id}__attempt{attempt_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "attempt.json", "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, default=str)

    # V2 Phase 5C-A: a VALID run whose result carries the real per-decision
    # `samples` shape (the unified eval_v2_episodes.EpisodeSample schema --
    # dry-run fakes do not produce this) is normalized into
    # analyze_v2_evaluation's RunSeries schema immediately, so a later
    # analysis pass needs no manual conversion step.
    if entry["status"] == "VALID" and isinstance(entry.get("result"), dict) and "samples" in entry["result"]:
        series_dict = episode_result_to_run_series_dict(entry["result"], run_record.run_id, run_record.repeat_index)
        with open(run_dir / "normalized_series.json", "w", encoding="utf-8") as f:
            json.dump(series_dict, f, indent=2, default=str)

    return entry


def run_campaign(
    campaign_id: str, ctx: CampaignContext, output_root: Path = EVAL_OUTPUT_ROOT,
    preregistration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    preregistration = preregistration or load_preregistration()
    campaign_dir = output_root / campaign_id
    plan_path = campaign_dir / "RUN_PLAN.json"
    ledger_path = campaign_dir / "ledger.jsonl"
    status_path = campaign_dir / "run_status.json"
    ctx.runs_dir = campaign_dir / "runs"

    if plan_path.exists():
        with open(plan_path, encoding="utf-8") as f:
            plan = json.load(f)
        runs = [RunRecord(**{k: v for k, v in r.items() if k != "run_id"}) for r in plan["runs"]]
    else:
        runs = build_run_plan(preregistration)
        plan = write_run_plan(runs, campaign_id, preregistration, plan_path)

    ledger = load_ledger(ledger_path)
    done = completed_run_ids(ledger)

    write_run_status(status_path, "RUNNING", None)

    for run_record in runs:
        if run_record.run_id in done:
            continue

        write_run_status(status_path, "RUNNING", run_record.to_dict())
        entry = run_one_attempt(run_record, ctx, attempt_number=1)
        append_ledger(ledger_path, entry)

        if entry["status"] == "TOOLING_FAILURE":
            retry_entry = run_one_attempt(run_record, ctx, attempt_number=2)
            if retry_entry["status"] == "TOOLING_FAILURE":
                retry_entry["status"] = "FAILED_AFTER_RETRY"
            append_ledger(ledger_path, retry_entry)
            if retry_entry["status"] == "FAILED_AFTER_RETRY":
                write_run_status(status_path, "FAILED", run_record.to_dict())
                return {"ok": False, "status": "FAILED", "campaign_dir": str(campaign_dir), "failed_run": run_record.to_dict()}

    write_run_status(status_path, "COMPLETE", None)
    return {"ok": True, "status": "COMPLETE", "campaign_dir": str(campaign_dir), "n_runs": len(runs)}


# ---------------------------------------------------------------------------
# Dry-run context (fakes -- no cluster, no subprocess, no sleep)
# ---------------------------------------------------------------------------


class _NoopSleepRecorder:
    def __init__(self) -> None:
        self.calls: List[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class _FakeOrchestrator:
    """V2 Phase 5B dry-run double: implements the 3 methods
    `run_one_attempt()` actually calls on an orchestrator
    (`start_episode_traffic`/`stop_episode_traffic`/
    `drain_server_side_requests`) with no subprocess, no real HTTP, and no
    Kubernetes/Prometheus access -- a true offline fake, rather than trying
    to fake every layer of `V5TransportManager`'s own `docker exec`-level
    command runner (which `to.TrafficOrchestrator.ensure_route_established()`
    needs for real). Records every call for test assertions."""

    def __init__(self) -> None:
        self.started: List[tuple] = []
        self.stopped: List[tuple] = []
        self.drained = 0

    def start_episode_traffic(self, profile_name: str, episode_id: str):
        resolve_eval_profile_metadata(profile_name)  # still exercises the real firewall/parser
        self.started.append((profile_name, episode_id))
        return {"profile_name": profile_name, "episode_id": episode_id, "fake_handle": True}

    def stop_episode_traffic(self, handle: Any, episode_id: str) -> None:
        self.stopped.append((episode_id,))

    def drain_server_side_requests(self) -> float:
        self.drained += 1
        return 0.0


def _fake_episode_samples(arm_id: str, profile_name: str, n_decisions: int, base_latency_ms: float) -> List[Dict[str, Any]]:
    """V2 Phase 5C-A: synthetic samples in the SAME shape
    `eval_v2_episodes.EpisodeSample.to_dict()` produces -- lets a dry-run
    campaign's ledger feed directly into `analyze_v2_evaluation.py` (via
    `eval_v2_episodes.episode_result_to_run_series_dict()`) without a real
    cluster, proving the runner-to-analysis pipeline end-to-end offline."""
    samples = []
    for i in range(n_decisions):
        ppo_decision = None
        if arm_id == "ppo_v2":
            ppo_decision = {"effective_action_name": "HOLD" if i % 3 else "SCALE_UP", "extra": {"is_reversal": False}}
        samples.append({
            "decision_index": i, "t_s": i * 30.0, "timestamp_utc": "fake", "profile": profile_name, "arm_id": arm_id,
            "normalized_observation": [0.5] * 11, "raw_observation": {}, "sample_valid": True,
            "p95_latency_ms": base_latency_ms + i, "error_rate": 0.0, "request_rate_rps": 10.0,
            "desired_replicas": 2, "ready_replicas": 2, "max_per_pod_memory_mib": 80.0,
            "pod_restart_counts": {}, "hpa_status": None if arm_id == "ppo_v2" else {"hpa_name": arm_id, "applied": True},
            "ppo_decision": ppo_decision,
        })
    return samples


def build_dry_run_context(campaign_id: str) -> CampaignContext:
    """Every cluster-touching dependency is a fake: no subprocess, no real
    HTTP, no real sleep, no Kubernetes/Prometheus access -- proves the
    planning/resume/retry/cleanup/fail-closed control flow entirely
    offline."""
    sleep_recorder = _NoopSleepRecorder()
    orchestrator = _FakeOrchestrator()

    def fake_ppo_episode(profile_meta: "to.ProfileMetadata", episode_id: str) -> Dict[str, Any]:
        n = int(profile_meta.duration_seconds // 30)
        # Deterministically the lowest latency of the 3 arms, every decision,
        # every repeat -- so the "better in all 3 repeats" comparison has a
        # decidable winner to prove end-to-end (see
        # TestEndToEndDryCampaignToAnalysis in tests/unit/test_evaluate_v2_campaign.py).
        return {
            "episode_id": episode_id, "arm_id": "ppo_v2", "profile": profile_meta.name, "pod_restarts_total": 0,
            "samples": _fake_episode_samples("ppo_v2", profile_meta.name, n, base_latency_ms=200.0),
        }

    def fake_hpa_episode(profile_meta: "to.ProfileMetadata", episode_id: str) -> Dict[str, Any]:
        n = int(profile_meta.duration_seconds // 30)
        return {
            "episode_id": episode_id, "arm_id": "hpa_v2_cpu_memory", "profile": profile_meta.name, "pod_restarts_total": 0,
            "samples": _fake_episode_samples("hpa_v2_cpu_memory", profile_meta.name, n, base_latency_ms=600.0),
        }

    def fake_cpu_only_episode(profile_meta: "to.ProfileMetadata", episode_id: str) -> Dict[str, Any]:
        # Only ever reached once resolve_cpu_only_controller_identity() (called
        # by run_one_attempt BEFORE this fn) has already succeeded -- i.e. a
        # test has monkeypatched the future Phase 5C constant in.
        n = int(profile_meta.duration_seconds // 30)
        return {
            "episode_id": episode_id, "arm_id": "hpa_v2_cpu_only", "profile": profile_meta.name, "pod_restarts_total": 0,
            "samples": _fake_episode_samples("hpa_v2_cpu_only", profile_meta.name, n, base_latency_ms=650.0),
        }

    return CampaignContext(
        orchestrator=orchestrator,
        preflight_fn=lambda: {"ok": True, "fake": True},
        ppo_episode_fn=fake_ppo_episode,
        hpa_episode_fn=fake_hpa_episode,
        cpu_only_episode_fn=fake_cpu_only_episode,
        cleanup_fn=lambda: {"restored": True, "fake": True},
        settle_fn=sleep_recorder,
        settle_seconds=120.0,
    )


def restore_canonical_idle_state() -> Dict[str, Any]:
    """V2 Phase 5B: the campaign-level cleanup every run needs (task sec.3)
    -- neither `run_live_v2_episode()` nor `run_hpa_v2_episode()` guarantee
    this on their own: the PPO arm's own `finally` releases its lock but
    applies no HPA at all (PPO requires zero HPAs present to even start),
    and the HPA arm's own `finally` re-applies `demo-app-hpa-v2` but this
    campaign must do the same after a PPO run too, so every run leaves the
    SAME canonical idle state behind it regardless of which arm just ran.
    Idempotent (`kubectl apply`); never deletes/scales down below the
    floor. NOT called by this phase (dry-run only)."""
    from autoscaler.src.rl_env.live_backend import LiveKubernetesBackend

    backend = LiveKubernetesBackend(
        hpa_name=V2_HPA_NAME, hpa_manifest_path=V2_HPA_MANIFEST_PATH, min_replicas=2, initial_replicas=2,
    )
    backend._kubectl.restore_hpa()
    applied = backend._kubectl.list_hpa_names()
    backend.set_desired_replicas(2)
    backend._kubectl.wait_for_replica_state(2, timeout=backend.restoration_max_wait_s)
    return {"restored": True, "hpa_names": applied, "replicas": 2}


def live_preflight_check() -> Dict[str, Any]:
    """V2 Phase 5B: a real (but minimal) preflight -- controller lock must
    read `none`, and the only HPA object present (if any) must be
    `demo-app-hpa-v2` (the canonical idle state `restore_canonical_idle_state()`
    leaves behind). NOT called by this phase (dry-run only; see the phase's
    cluster-interaction protection constraints)."""
    from autoscaler.src.rl_env.live_backend import LiveKubernetesBackend

    backend = LiveKubernetesBackend(
        hpa_name=V2_HPA_NAME, hpa_manifest_path=V2_HPA_MANIFEST_PATH, min_replicas=2, initial_replicas=2,
    )
    lock = read_lock()
    hpas = backend._kubectl.list_hpa_names()
    ok = lock == CONTROLLER_NONE and hpas in ([], [V2_HPA_NAME])
    return {"ok": ok, "lock": lock, "hpa_names": hpas}


def build_live_context(model_path: Path, epsilon: float, settle_seconds: float = 120.0) -> CampaignContext:
    """V2 Phase 5C-A: wires the SAME control flow to the REAL, unified
    per-arm episode functions in scripts/eval_v2_episodes.py (all three
    arms now measured identically). NOT invoked by this phase (training is
    live) -- see the module docstring."""
    orchestrator = V2EvalTrafficOrchestrator()

    def ppo_episode(profile_meta: "to.ProfileMetadata", episode_id: str) -> Dict[str, Any]:
        return run_ppo_v2_eval_episode(model_path, epsilon, profile_meta.name, profile_meta.duration_seconds, episode_id)

    def hpa_episode(profile_meta: "to.ProfileMetadata", episode_id: str) -> Dict[str, Any]:
        return run_hpa_v2_cpu_memory_eval_episode(profile_meta.name, profile_meta.duration_seconds, episode_id)

    def cpu_only_episode(profile_meta: "to.ProfileMetadata", episode_id: str) -> Dict[str, Any]:
        return run_hpa_v2_cpu_only_eval_episode(profile_meta.name, profile_meta.duration_seconds, episode_id)

    return CampaignContext(
        orchestrator=orchestrator, preflight_fn=live_preflight_check, ppo_episode_fn=ppo_episode,
        hpa_episode_fn=hpa_episode, cpu_only_episode_fn=cpu_only_episode, cleanup_fn=restore_canonical_idle_state,
        settle_fn=__import__("time").sleep, settle_seconds=settle_seconds,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--i-confirm-live-cluster-interaction", action="store_true")
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "ppo_v2" / "ppo_model_v2_min2.zip"))
    parser.add_argument("--epsilon-calibration-path", default=None)
    parser.add_argument("--output-root", default=str(EVAL_OUTPUT_ROOT))
    args = parser.parse_args(argv)

    campaign_id = args.campaign_id or f"campaign_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_root = Path(args.output_root)

    if args.execute:
        if not args.i_confirm_live_cluster_interaction:
            print("Refusing --execute without --i-confirm-live-cluster-interaction.", file=sys.stderr)
            return 2
        if args.epsilon_calibration_path is None:
            print("Refusing --execute without --epsilon-calibration-path (the ppo_v2 arm requires a calibrated epsilon).", file=sys.stderr)
            return 2
        with open(args.epsilon_calibration_path, encoding="utf-8") as f:
            calibration = json.load(f)
        epsilon = calibration["epsilon"]
        ctx = build_live_context(Path(args.model_path), epsilon)
    else:
        ctx = build_dry_run_context(campaign_id)

    result = run_campaign(campaign_id, ctx, output_root=output_root)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
