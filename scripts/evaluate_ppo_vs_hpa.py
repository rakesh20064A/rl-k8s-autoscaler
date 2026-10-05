#!/usr/bin/env python3
"""scripts/evaluate_ppo_vs_hpa.py

Phase 16V: the final PPO-vs-HPA evaluation harness conforming to
phase_16u_direct_ppo_protocol.yaml Sections 9-10 (baselines, metrics,
decision rule) and phase_16u_training_and_evaluation_workload_split.json
(final_evaluation_design: 3 repeats per controller, identical workload).

CLASSIFICATION: this script's --dry-run path is OFFLINE_VALIDATION_ONLY.
Its --execute path (NOT invoked in Phase16V) would open/run the final
held-out workload for real -- Phase16V's task instructions explicitly
require this to remain unexecuted until a separately authorized phase.

Guarantees implemented (task sec.7):
  - identical workload schedule for the PPO and HPA repeat of each pair
    (both read the SAME materialized/hash-pinned YAML;
    `build_evaluation_plan()` below assigns identical `workload_hash` to
    every planned episode).
  - no simultaneous controllers: `plan` is a strictly ordered list of
    single-controller episodes; the executor (not implemented/run here)
    must process it sequentially, never interleaved.
  - controller-order provenance: each planned episode records its position
    in the sequence and which controller it is.
  - workload hash verification: `verify_final_holdout_untouched()` checks
    the materialized file's hash against phase_16u_training_and_evaluation_
    workload_split.json's frozen spec, and that it has never appeared in
    any existing training/evaluation ledger.
  - separate PPO/HPA attempt IDs: `episode_id` embeds the controller name.
  - canonical restoration between runs: delegated to the same
    live_backend.LiveKubernetesBackend.reset() / KubectlRunner.restore_hpa()
    mechanisms already implemented and tested for training.
  - no tuning from final results: this script contains no code path that
    reads back evaluation results to adjust any config/hyperparameter --
    by construction, it only ever APPENDS to the final-evaluation ledger.
  - final evaluation ledger distinct from the training ledger: separate
    path constants below.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoscaler.src.rl_env.config import load_env_config_from_yaml  # noqa: E402
from autoscaler.src.rl_env.episode_ledger import append_row, count_rows, read_rows  # noqa: E402
from autoscaler.src.rl_env.environment import KubernetesAutoscalerEnv  # noqa: E402
from autoscaler.src.rl_env.live_backend import LiveKubernetesBackend  # noqa: E402
from collectors.metrics_collector import PrometheusClient  # noqa: E402
from scripts import rl_live_common as common  # noqa: E402
from scripts import traffic_orchestrator as to  # noqa: E402
from scripts.materialize_rl_final_holdout_workload import _load_spec as _load_holdout_spec  # noqa: E402

EVALUATION_ROOT = PROJECT_ROOT / "results" / "evaluation" / "raw"
EVAL_APP_NS = "autoscaler-app"
EVAL_DEPLOYMENT = "demo-app"
EVAL_ROUTE = "/work"

PPO_FINAL_EVAL_LEDGER = PROJECT_ROOT / "results" / "evaluation" / "ledgers" / "ppo_final_evaluation_ledger.jsonl"
HPA_FINAL_EVAL_LEDGER = PROJECT_ROOT / "results" / "evaluation" / "ledgers" / "hpa_final_evaluation_ledger.jsonl"
TRAINING_LEDGER = PROJECT_ROOT / "persistent-data" / "rl_training" / "ppo_training_episode_ledger.jsonl"

N_REPEATS = 3
CONTROLLERS = ("PPO", "HPA")


def verify_final_holdout_untouched() -> Dict[str, Any]:
    """Confirms the materialized final-holdout workload file's hash matches
    the frozen Phase16U spec, and that its profile name has never appeared
    in the training ledger or either evaluation ledger (i.e. it has never
    actually been executed)."""
    spec = _load_holdout_spec()
    target = PROJECT_ROOT / spec["new_file_name_reserved"]
    result: Dict[str, Any] = {"file": str(target.relative_to(PROJECT_ROOT))}

    if not target.exists():
        result["exists"] = False
        result["untouched"] = True  # cannot have been executed if it was never even materialized
        return result

    result["exists"] = True
    actual_hash = common.sha256_file(target)
    result["sha256"] = actual_hash

    appeared_in_training = any(
        "iteration_rl_final_holdout_v1" in str(row.get("traffic_profile", ""))
        for row in read_rows(TRAINING_LEDGER)
    )
    appeared_in_ppo_eval = count_rows(PPO_FINAL_EVAL_LEDGER) > 0
    appeared_in_hpa_eval = count_rows(HPA_FINAL_EVAL_LEDGER) > 0

    result["appeared_in_training_ledger"] = appeared_in_training
    result["appeared_in_ppo_final_eval_ledger"] = appeared_in_ppo_eval
    result["appeared_in_hpa_final_eval_ledger"] = appeared_in_hpa_eval
    # Index38B: once attempt1 has genuinely executed, this correctly
    # becomes False and MUST NEVER be forced back to True -- a resume
    # after a partial attempt uses verify_resumable_from_attempt1() below
    # instead, never this function pretending nothing happened.
    result["untouched"] = not (appeared_in_training or appeared_in_ppo_eval or appeared_in_hpa_eval)
    return result


def verify_resumable_from_attempt1(expected_prior_rows: Dict[str, int]) -> Dict[str, Any]:
    """Index38B STEP5: resume-aware provenance gate, DISTINCT from
    `verify_final_holdout_untouched()` (which correctly reports
    untouched=False forever after attempt1 -- this function is what
    actually authorizes a resume). Verifies:
      - the holdout file/hash still matches the frozen spec (never
        silently accepts a changed workload)
      - every existing final-eval ledger row matches an EXACT expected
        prior-attempt identity (episode_id, repeat_index, workload_hash)
        -- never silently accepts arbitrary pre-existing evaluation data
        as a basis for resuming
      - no unexpected extra rows exist in either ledger
    Never deletes, edits, or reclassifies any existing row."""
    expected_hash = "308f4f90ea3cea97f1927cec97efeaa652c4424e9fe5bcea421358f3425e3409"
    spec = _load_holdout_spec()
    target = PROJECT_ROOT / spec["new_file_name_reserved"]
    actual_hash = common.sha256_file(target) if target.exists() else None
    result: Dict[str, Any] = {"holdout_hash": actual_hash, "holdout_hash_matches": actual_hash == expected_hash}

    ppo_rows = list(read_rows(PPO_FINAL_EVAL_LEDGER))
    hpa_rows = list(read_rows(HPA_FINAL_EVAL_LEDGER))
    result["ppo_rows_found"] = len(ppo_rows)
    result["hpa_rows_found"] = len(hpa_rows)

    expected_ppo_ids = {f"final_eval_ppo_repeat{i}" for i in expected_prior_rows.get("ppo_repeats", [])}
    expected_hpa_ids = {f"final_eval_hpa_repeat{i}" for i in expected_prior_rows.get("hpa_repeats", [])}

    def _rows_match(rows: List[Dict[str, Any]], expected_ids: set) -> bool:
        actual_ids = {r.get("episode_id") for r in rows}
        if actual_ids != expected_ids:
            return False
        return all(r.get("workload_hash") == expected_hash for r in rows)

    ppo_ok = _rows_match(ppo_rows, expected_ppo_ids)
    hpa_ok = _rows_match(hpa_rows, expected_hpa_ids)
    result["prior_ppo_evidence_matches_expected_identity"] = ppo_ok
    result["prior_hpa_evidence_matches_expected_identity"] = hpa_ok
    result["resumable"] = bool(result["holdout_hash_matches"] and ppo_ok and hpa_ok)
    return result


def build_evaluation_plan() -> List[Dict[str, Any]]:
    """The frozen, ordered list of episodes to run -- PPO and HPA
    interleaved pair-by-pair (PPO repeat 1, HPA repeat 1, PPO repeat 2, ...)
    so that infrastructure conditions drift as little as possible between a
    matched pair, while guaranteeing the two controllers are never
    simultaneous (this is a PLAN, not a live schedule -- no episode in it
    has been executed)."""
    spec = _load_holdout_spec()
    target = PROJECT_ROOT / spec["new_file_name_reserved"]
    workload_hash = common.sha256_file(target) if target.exists() else "NOT_YET_MATERIALIZED"
    plan = []
    for repeat_index in range(1, N_REPEATS + 1):
        for controller in CONTROLLERS:
            plan.append({
                "sequence_index": len(plan) + 1,
                "controller": controller,
                "repeat_index": repeat_index,
                "workload_profile": spec["new_file_name_reserved"],
                "workload_seed": spec["seed"],
                "workload_hash": workload_hash,
                "episode_id": f"final_eval_{controller.lower()}_repeat{repeat_index}",
                "executed": False,
            })
    return plan


def run_dry_run_validation() -> Dict[str, Any]:
    report: Dict[str, Any] = {"classification": "OFFLINE_DRY_RUN_VALIDATION_ONLY"}

    hash_checks = common.verify_governing_hashes()
    report["all_governing_hashes_match"] = all(c["match"] for c in hash_checks)

    holdout_status = verify_final_holdout_untouched()
    report["final_holdout_status"] = holdout_status

    plan = build_evaluation_plan()
    report["evaluation_plan"] = plan
    report["n_planned_episodes"] = len(plan)
    report["plan_alternates_controllers_never_simultaneous"] = all(
        plan[i]["controller"] != plan[i + 1]["controller"] for i in range(0, len(plan) - 1, 2)
    )
    report["plan_uses_identical_workload_for_every_episode"] = len({p["workload_profile"] for p in plan}) == 1
    report["distinct_ledgers_configured"] = str(PPO_FINAL_EVAL_LEDGER) != str(HPA_FINAL_EVAL_LEDGER) != str(TRAINING_LEDGER)

    try:
        common.assert_final_evaluation_eligible(common.FINAL_HOLDOUT_PROFILE)
        report["final_holdout_profile_name_accepted_by_firewall"] = True
    except common.WorkloadFirewallViolation:
        report["final_holdout_profile_name_accepted_by_firewall"] = False

    firewall_rejects_training_profiles = True
    for profile in common.TRAINING_PROFILES:
        try:
            common.assert_final_evaluation_eligible(profile)
            firewall_rejects_training_profiles = False
        except common.WorkloadFirewallViolation:
            pass
    report["firewall_rejects_training_profiles_for_final_eval"] = firewall_rejects_training_profiles

    report["n_ppo_final_eval_episodes_executed_so_far"] = count_rows(PPO_FINAL_EVAL_LEDGER)
    report["n_hpa_final_eval_episodes_executed_so_far"] = count_rows(HPA_FINAL_EVAL_LEDGER)

    report["traffic_generated"] = False
    report["kubernetes_mutation"] = False
    report["ok"] = (
        report["all_governing_hashes_match"]
        and holdout_status["untouched"]
        and report["plan_alternates_controllers_never_simultaneous"]
        and report["plan_uses_identical_workload_for_every_episode"]
        and report["distinct_ledgers_configured"]
        and report["final_holdout_profile_name_accepted_by_firewall"]
        and report["firewall_rejects_training_profiles_for_final_eval"]
        and report["n_ppo_final_eval_episodes_executed_so_far"] == 0
        and report["n_hpa_final_eval_episodes_executed_so_far"] == 0
    )
    return report


def resolve_holdout_profile_metadata() -> "to.ProfileMetadata":
    """Index38A: the eval-side equivalent of traffic_orchestrator.py's
    `resolve_profile_metadata()` -- that function is intentionally
    training-only (it calls `common.assert_training_eligible()`, which
    REJECTS the final-holdout name by design). Rather than weaken that
    firewall, this builds the same `ProfileMetadata` shape directly, gated
    by the correct eval-side firewall function
    (`common.assert_final_evaluation_eligible`) and an explicit hash-pin
    check against the frozen spec -- mirrors, never bypasses, the existing
    fail-closed pattern."""
    common.assert_final_evaluation_eligible(common.FINAL_HOLDOUT_PROFILE)
    spec = _load_holdout_spec()
    path = PROJECT_ROOT / spec["new_file_name_reserved"]
    if not path.exists():
        raise FileNotFoundError(f"final holdout workload not materialized: {path}")
    actual_sha256 = common.sha256_file(path)
    profile = to.parse_traffic_profile(path)
    to.verify_profile_shape_not_degenerate(common.FINAL_HOLDOUT_PROFILE, profile)
    return to.ProfileMetadata(
        name=common.FINAL_HOLDOUT_PROFILE, path=path, sha256=actual_sha256,
        seed=profile.seed, duration_seconds=profile.total_duration, n_phases=len(profile.phases),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _http_get(url: str, timeout: float):
    """Real HTTP GET -- identical implementation to
    train_ppo_live_session.py's `_make_live_traffic_orchestrator()` inner
    `http_get`, reused verbatim rather than reimplemented."""
    import httpx
    try:
        resp = httpx.get(url, timeout=timeout)
        return resp.status_code == 200, f"HTTP {resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _start_holdout_traffic(episode_id: str, profile_meta: "to.ProfileMetadata", transport: "to.V5TransportManager"):
    run_id = to.resolve_next_available_run_id(episode_id, to.TRAFFIC_OUTPUT_BASE_DIR)
    route = transport.establish_route(run_id)
    if not to.verify_route_alive(route, _http_get):
        raise to.RouteEstablishmentFailure(f"route {route.target_url} not alive before {episode_id}")
    handle = to.start_generator(profile_meta, route, run_id)
    if not to.verify_generator_started(handle):
        to.stop_generator(handle)
        raise to.TrafficGeneratorFailure(f"generator failed to start for final-eval episode {episode_id}")
    return handle, route


def _stop_holdout_traffic(handle, route, transport: "to.V5TransportManager") -> None:
    to.stop_generator(handle)
    query = (
        f'sum(demo_app_requests_in_progress{{namespace="{EVAL_APP_NS}",route="{EVAL_ROUTE}"}})'
    )
    prom = PrometheusClient()
    to.drain_in_flight_requests(query_fn=lambda: prom.query(query))
    transport.teardown_route(route)


def _snapshot_dict(snapshot, extra_point_metrics: Optional[Dict[str, Optional[float]]] = None) -> Dict[str, Any]:
    m = snapshot.metrics
    r = snapshot.replicas
    d = {
        "time_s": snapshot.time_s,
        "request_rate_rps": m.request_rate_rps,
        # NOTE: RawMetricSnapshot's own `error_rate_rps` IS the real 5xx
        # rate (live_backend.py's six-query subset), already correct --
        # `_prom_point_query()` below adds ONLY the fields RawMetricSnapshot
        # does not carry (p50/p99/success_rate_rps), never a duplicate.
        "error_rate_rps": m.error_rate_rps,
        "latency_p95_ms": m.latency_p95_ms,
        "requests_in_progress": m.requests_in_progress,
        "cpu_cores": m.cpu_cores,
        "memory_bytes": m.memory_bytes,
        "target_up": m.target_up,
        "replicas_desired": r.desired,
        "replicas_current": r.current,
        "replicas_ready": r.ready,
        "replicas_available": r.available,
    }
    if extra_point_metrics:
        d.update(extra_point_metrics)
    return d


TERMINATION_CONSECUTIVE_INVALID_LIMIT = 4  # frozen, docs/rl_mdp_spec.md sec.13 / phase_16u_safety_and_fallback_protocol.yaml


def _classify_termination(samples: List[Dict[str, Any]], nominal_steps: int, env_terminated: Optional[bool]) -> Dict[str, Any]:
    """Index38B STEP3B: classify using the frozen protocol's own named
    outcomes (phase_16u_safety_and_fallback_protocol.yaml
    episode_termination_rules) -- never silently calls a premature stop a
    normal full-duration completion."""
    observed_steps = max(0, len(samples) - 1)  # samples includes the t=0 snapshot
    full_completed = observed_steps >= nominal_steps

    # Trailing run of target_up=False, mirroring consecutive_invalid_steps
    # (the env's own counter is per-sample-validity, not exactly target_up,
    # but target_up=False is always invalid -- this reconstructs the
    # equivalent trailing-run signal from the raw samples we recorded).
    trailing_down = 0
    for s in reversed(samples):
        if s.get("target_up") is False:
            trailing_down += 1
        else:
            break

    if full_completed:
        status, reason = "completed_full_duration", "none"
    elif env_terminated or trailing_down >= TERMINATION_CONSECUTIVE_INVALID_LIMIT:
        status = "terminated_safety"
        reason = "consecutive_invalid_steps_limit_reached"
    else:
        status, reason = "truncated_other", "episode_ended_before_nominal_duration_unclassified"

    return {
        "episode_status": status,
        "termination_reason": reason,
        "full_holdout_completed": full_completed,
        "nominal_decision_steps": nominal_steps,
        "observed_decision_steps": observed_steps,
        # Per phase_16u_direct_ppo_protocol.yaml Section 10: the 3-repeat
        # decision rule compares episodes on the IDENTICAL workload: a
        # repeat that observed materially less of that workload than its
        # matched counterpart is not comparable for the primary aggregate,
        # even though it is a legitimate, fully-recorded outcome that must
        # never be discarded (phase_16u_safety_and_fallback_protocol.yaml:
        # "never silently discarded, never excluded from any later report").
        "valid_for_primary_comparison": full_completed,
    }


def _summarize_arm(controller: str, repeat_index: int, episode_id: str, samples: List[Dict[str, Any]],
                    decisions: List[Dict[str, Any]], profile_meta: "to.ProfileMetadata",
                    nominal_steps: int, env_terminated: Optional[bool] = None) -> Dict[str, Any]:
    cpu_pcts = [s["cpu_cores"] for s in samples if s["cpu_cores"] is not None]
    # Index38B STEP3B fix: scaling events must count EFFECTIVE/actuated
    # replica changes (deployment desired-replica count), never the
    # Ready-replica count, which flickers under pod restarts unrelated to
    # any scaling decision.
    desired_replicas = [s["replicas_desired"] for s in samples]
    ready_replicas = [s["replicas_ready"] for s in samples]
    latencies = [s["latency_p95_ms"] for s in samples if s["latency_p95_ms"] is not None]
    p50s = [s["latency_p50_ms"] for s in samples if s.get("latency_p50_ms") is not None]
    p99s = [s["latency_p99_ms"] for s in samples if s.get("latency_p99_ms") is not None]
    success_rates = [s["success_rate_rps"] for s in samples if s.get("success_rate_rps") is not None]
    error_rates = [s["error_rate_rps"] for s in samples if s.get("error_rate_rps") is not None]
    scaling_events = sum(1 for a, b in zip(desired_replicas, desired_replicas[1:]) if a != b)
    mean_replicas = sum(ready_replicas) / len(ready_replicas) if ready_replicas else 0.0
    max_replicas = max(ready_replicas) if ready_replicas else 0

    classification = _classify_termination(samples, nominal_steps, env_terminated)

    return {
        "timestamp_utc": _utc_now(),
        "controller": controller,
        "repeat_index": repeat_index,
        "episode_id": episode_id,
        "workload_profile": profile_meta.name,
        "workload_seed": profile_meta.seed,
        "workload_hash": profile_meta.sha256,
        "workload_duration_seconds": profile_meta.duration_seconds,
        "n_decision_steps": len(samples),
        "mean_replicas": mean_replicas,
        "max_replicas": max_replicas,
        "scaling_events": scaling_events,
        "mean_cpu_cores": (sum(cpu_pcts) / len(cpu_pcts)) if cpu_pcts else None,
        "mean_latency_p95_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "latency_p50_ms": (sum(p50s) / len(p50s)) if p50s else None,
        "latency_p99_ms": (sum(p99s) / len(p99s)) if p99s else None,
        "success_rate_rps": (sum(success_rates) / len(success_rates)) if success_rates else None,
        "error_rate_rps": (sum(error_rates) / len(error_rates)) if error_rates else None,
        **classification,
        "samples": samples,
        "decisions": decisions,
    }


def _prom_point_query(prom: PrometheusClient) -> Dict[str, Optional[float]]:
    """Index38B STEP4 fix: these metrics are NOT part of RawMetricSnapshot,
    so they must be sampled PER DECISION STEP (same cadence as the rest of
    the snapshot), not once at episode end after traffic has already
    stopped and the target may be down -- that timing bug, not a missing
    metric definition, is why these were previously always null. Uses the
    exact PromQL conventions already defined in
    collectors/metrics_collector.py -- never an invented metric name."""
    ns, route = EVAL_APP_NS, EVAL_ROUTE
    queries = {
        "latency_p50_ms": f'histogram_quantile(0.50, sum(rate(demo_app_request_duration_seconds_bucket{{namespace="{ns}",route="{route}"}}[1m])) by (le)) * 1000',
        "latency_p99_ms": f'histogram_quantile(0.99, sum(rate(demo_app_request_duration_seconds_bucket{{namespace="{ns}",route="{route}"}}[1m])) by (le)) * 1000',
        "success_rate_rps": f'sum(rate(demo_app_requests_total{{namespace="{ns}",route="{route}",status_code=~"2.."}}[1m]))',
    }
    out: Dict[str, Optional[float]] = {}
    for name, q in queries.items():
        val, _err = prom.query(q)
        out[name] = val
    return out


def run_ppo_episode(repeat_index: int, sequence_index: int, model, env_config, profile_meta: "to.ProfileMetadata") -> Dict[str, Any]:
    backend = LiveKubernetesBackend()
    if backend._kubectl.hpa_exists():
        raise RuntimeError("PPO evaluation episode refused: HPA is present (must be absent for PPO arm)")

    episode_id = f"final_eval_ppo_repeat{repeat_index}"
    transport = to.V5TransportManager()
    env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=profile_meta.duration_seconds, config=env_config)

    obs, info = env.reset()
    handle, route = _start_holdout_traffic(episode_id, profile_meta, transport)
    prom = PrometheusClient()
    n_steps = int(round(profile_meta.duration_seconds / env_config.decision_interval_s))

    samples: List[Dict[str, Any]] = [_snapshot_dict(backend.current_snapshot(), _prom_point_query(prom))]
    decisions: List[Dict[str, Any]] = []
    env_terminated = False
    try:
        for _ in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action)
            replicas_before = backend._safe_replica_state().desired
            obs, reward, terminated, truncated, info = env.step(action_int)
            snap = backend.current_snapshot()
            samples.append(_snapshot_dict(snap, _prom_point_query(prom)))
            decisions.append({
                "decision_time_s": info.get("decision_time_s"),
                "action_index": action_int,
                "semantic_action": info.get("semantic_action"),
                "replicas_before": replicas_before,
                "replicas_after": snap.replicas.desired,
                "reward_recorded_not_used_for_training": info.get("reward_total"),
            })
            if terminated or truncated:
                env_terminated = bool(terminated)
                break
    finally:
        _stop_holdout_traffic(handle, route, transport)

    return _summarize_arm("PPO", repeat_index, episode_id, samples, decisions, profile_meta, n_steps, env_terminated)


def run_hpa_episode(repeat_index: int, sequence_index: int, env_config, profile_meta: "to.ProfileMetadata") -> Dict[str, Any]:
    backend = LiveKubernetesBackend()
    if not backend._kubectl.hpa_exists():
        raise RuntimeError("HPA evaluation episode refused: HPA is absent (must be present for HPA arm)")

    episode_id = f"final_eval_hpa_repeat{repeat_index}"
    transport = to.V5TransportManager()

    snapshot = backend.reset()  # canonical reset + stabilization, HPA governs scaling independently
    handle, route = _start_holdout_traffic(episode_id, profile_meta, transport)
    prom = PrometheusClient()
    n_steps = int(round(profile_meta.duration_seconds / env_config.decision_interval_s))

    samples: List[Dict[str, Any]] = [_snapshot_dict(snapshot, _prom_point_query(prom))]
    try:
        for _ in range(n_steps):
            snapshot = backend.advance(env_config.decision_interval_s)
            samples.append(_snapshot_dict(snapshot, _prom_point_query(prom)))
    finally:
        _stop_holdout_traffic(handle, route, transport)

    return _summarize_arm("HPA", repeat_index, episode_id, samples, [], profile_meta, n_steps, env_terminated=None)


def run_live_evaluation() -> int:
    """Index38A: the real final-evaluation execution path. Runs the frozen
    6-episode plan (PPO1,HPA1,PPO2,HPA2,PPO3,HPA3) sequentially, inference
    only -- never calls model.learn(), never writes a training checkpoint.
    HPA presence/absence around each episode is the CALLER's (supervisor's)
    responsibility, verified (not mutated) at the top of each run_*_episode
    call above. Appends one summary row per episode to the correct
    final-evaluation ledger and writes full per-episode evidence under
    EVALUATION_ROOT. Raises on any failure -- never silently continues."""
    from stable_baselines3 import PPO

    session_019 = PROJECT_ROOT / "models" / "ppo" / "ppo_model.zip"
    model = PPO.load(session_019)
    if model.num_timesteps != 5120:
        raise RuntimeError(f"frozen checkpoint num_timesteps mismatch: {model.num_timesteps} != 5120")

    env_config = load_env_config_from_yaml()
    profile_meta = resolve_holdout_profile_metadata()
    expected_hash = "308f4f90ea3cea97f1927cec97efeaa652c4424e9fe5bcea421358f3425e3409"
    if profile_meta.sha256 != expected_hash:
        raise RuntimeError(f"holdout hash mismatch: {profile_meta.sha256} != {expected_hash}")

    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    plan = build_evaluation_plan()

    for episode in plan:
        seq, controller, repeat = episode["sequence_index"], episode["controller"], episode["repeat_index"]
        if controller == "PPO":
            result = run_ppo_episode(repeat, seq, model, env_config, profile_meta)
            append_row(PPO_FINAL_EVAL_LEDGER, {k: v for k, v in result.items() if k not in ("samples", "decisions")})
        else:
            result = run_hpa_episode(repeat, seq, env_config, profile_meta)
            append_row(HPA_FINAL_EVAL_LEDGER, {k: v for k, v in result.items() if k not in ("samples", "decisions")})

        out_path = EVALUATION_ROOT / f"episode_{seq:02d}_{controller.lower()}_repeat{repeat}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        episode["executed"] = True
        episode["result_path"] = str(out_path.relative_to(PROJECT_ROOT))

    plan_path = EVALUATION_ROOT / "evaluation_plan_executed.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2, default=str)

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--i-confirm-live-cluster-interaction", action="store_true")
    args = parser.parse_args(argv)

    if args.execute:
        if not args.i_confirm_live_cluster_interaction:
            print("Refusing --execute without --i-confirm-live-cluster-interaction.", file=sys.stderr)
            return 2
        return run_live_evaluation()

    report = run_dry_run_validation()
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
