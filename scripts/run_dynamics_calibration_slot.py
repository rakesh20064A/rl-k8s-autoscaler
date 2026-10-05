"""scripts/run_dynamics_calibration_slot.py

Phase 15B.3A: the reusable execution harness for ONE frozen empirical
dynamics-calibration campaign slot. Resolves ALL scientific parameters
(cell, replicas, offered RPS, schedule, repeat) exclusively from the frozen
global campaign sequence / manifest / schedule registry -- this CLI accepts
no scientific override flag of any kind (no --rps, --replicas, --intensity).
The only inputs are `--campaign-sequence-index` (which slot) and pure
operational flags (`--dry-run`, `--attempt-number`, `--session-id`).

Reuses existing, already-tested tooling wherever it exists:
  - loadgen.traffic_generator (schedule REPLAY only, connection_mode=fresh)
  - collectors.metrics_collector (mode="manual" -- see NOTE below)
  - the exact HPA delete/restore and kubectl scale/rollout-status commands
    already used by scripts/run_rl_experiment.sh and
    scripts/run_hpa_tuning_trial.sh

New, narrow additions for this harness only:
  - scripts/dynamics_calibration_lib.py (pure, unit-tested: slot resolution,
    schedule-hash verification, ledger, CORR-35 routing math,
    decision-aligned sample selection, structural checks)
  - scripts/pod_readiness_collector.py / scripts/per_pod_counter_collector.py
    (new -- nothing in the repo previously captured per-pod Ready history or
    per-pod raw request counters; CORR-35's routing diagnostic requires both)

NOTE on metrics_collector mode: MetricsCollector's `mode` parameter only
supports {"hpa","rl","manual","test"} and both "manual" and "test" produce
`validity_status="test_non_benchmark"` in its own metadata.json -- there is
no "calibration" mode. This is a known, documented SCOPE LIMITATION of that
collector (it predates the calibration protocol), not a defect requiring a
CORR: this harness's OWN structural-validity audit (per CORR-32/33/34/35)
is independently authoritative for calibration purposes and does not read
or depend on metrics_collector's internal validity_status field.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.dynamics_calibration_lib import (  # noqa: E402
    CALIBRATION_ROOT,
    LEDGER_PATH,
    DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8,
    DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7,
    DATAPLANE_INGRESS_TRANSITION_RESAMPLE_DELAY_SECONDS,
    EXECUTION_INVALID,
    EXECUTION_VALID,
    LEDGER_STATE_VALID,
    MIN_EVIDENCE_WINDOW_SECONDS_V8,
    PROJECT_ROOT,
    REPLICA_TRANSITION_MATERIALITY_FRACTION,
    ROUTE_INTEGRITY_EVIDENCE_WINDOW_SECONDS,
    RELAY_IMAGE,
    TRANSITION_SCIENTIFIC_ROLE,
    TRANSPORT_TYPE_DOCKER_NETWORK_TCP_RELAY,
    TRANSPORT_VALIDATION_ROOT,
    aggregate_observer_evidence_for_window,
    assess_decision_point_eligibility,
    atomic_write_ledger,
    build_protocol_provenance,
    build_relay_docker_run_command,
    build_relay_target_url,
    build_transition_metrics_summary,
    check_no_dropped_overload,
    check_no_external_contamination_during_run,
    check_offered_equals_sent,
    classify_external_container_contamination,
    classify_route_integrity,
    classify_route_integrity_window,
    classify_routing,
    classify_tunnel_termination_action,
    compute_dataplane_ingress_evidence_windows,
    build_workload_transition_trajectory_rows,
    compute_delivered_request_evidence_windows,
    compute_expected_and_actual,
    count_schedule_data_rows,
    cpu_normalization,
    ensure_ledger,
    evaluate_fixed_replica_invariant,
    find_all_zero_success_runs,
    first_post_step_request_elapsed_seconds,
    initial_replica_count_for_restoration,
    is_interval_material,
    is_pod_ready_row,
    is_replica_transition_slot,
    is_workload_transition_slot,
    ledger_state_for_execution_validity,
    load_global_sequence,
    load_manifest,
    make_attempt_id,
    partition_into_evidence_windows,
    normalize_kubernetes_events,
    parse_dataplane_ingress_rules,
    parse_endpoints_ready_count,
    parse_pod_readiness_snapshot,
    replica_transition_materiality_threshold_seconds,
    replica_transition_route_scored_window,
    resolve_planned_slot,
    resolve_transition_timing_plan,
    resolve_workload_transition_timing_plan,
    restarted_pods_between,
    restoration_wait_seconds,
    run_level_fit_classification,
    select_decision_sample,
    should_trigger_immediate_resample,
    transition_direction,
    update_slot_state,
    utc_timestamp_id,
    verify_schedule_hash,
    WORKLOAD_TRANSITION_DERIVED_METRICS_STATUS,
    WORKLOAD_TRANSITION_DERIVED_METRICS_STATUS_REASON,
    WORKLOAD_TRANSITION_MATERIALITY_FRACTION,
    WORKLOAD_TRANSITION_SCIENTIFIC_ROLE,
    workload_transition_materiality_threshold_seconds,
    workload_transition_route_scored_window,
)

APP_NS = "autoscaler-app"
MONITORING_NS = "monitoring"
HPA_NAME = "demo-app-hpa"
LOADTEST_SVC = "demo-app-loadtest"
DEPLOYMENT = "demo-app"
HPA_MANIFEST_PATH = PROJECT_ROOT / "k8s/hpa/hpa.yaml"
VENV_PYTHON = str(PROJECT_ROOT / ".venv/bin/python")

# Phase 15B.3K regression: normalize_kubernetes_events()'s output keys and
# this CSV's fieldnames drifted out of sync (Phase 15B.3J.1 added
# `involved_object_uid` to the former without updating the latter), which
# crashed an official attempt (index10) partway through, after traffic had
# already run, on `csv.DictWriter.writerow()`. Named as a single constant,
# checked directly against `normalize_kubernetes_events`'s own output by a
# unit test, so the two can never silently drift apart again.
KUBERNETES_EVENTS_NORMALIZED_CSV_FIELDNAMES = [
    "first_timestamp", "last_timestamp", "count", "type", "reason",
    "involved_object_kind", "involved_object_name", "involved_object_uid", "message",
]


# ---------------------------------------------------------------------------
# CLI -- deliberately narrow: no scientific override flag exists anywhere.
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute exactly one frozen empirical dynamics-calibration campaign slot. "
            "All scientific parameters are resolved from the frozen global sequence -- "
            "this CLI accepts no override of replicas, RPS, schedule, or any other "
            "scientific parameter."
        )
    )
    parser.add_argument("--campaign-sequence-index", required=True, type=int, help="1..102, resolved against the frozen global sequence")
    parser.add_argument("--attempt-number", default=1, type=int, help="Execution attempt number for this planned slot (default 1)")
    parser.add_argument("--session-id", default=None, help="Optional session identifier; auto-generated if omitted")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan and exit; no traffic, no Kubernetes mutation")
    return parser


# ---------------------------------------------------------------------------
# Dry-run report (pure -- no I/O beyond what's passed in)
# ---------------------------------------------------------------------------


def dry_run_report(
    slot: dict,
    schedule_hash: str,
    manifest: dict,
    ledger: dict,
    transport_discovery: Optional[Dict[str, Any]] = None,
    calibration_root: Path = CALIBRATION_ROOT,
) -> Dict[str, Any]:
    """Pure formatting (no I/O of its own): `transport_discovery` is
    resolved by the CALLER (`main()`, via live `docker`/`kubectl`/
    `minikube` reads) before this function is invoked, so this function
    itself remains unit-testable without a live cluster -- exactly like
    every other field here, which is already resolved by the caller.

    Phase 15B.5B: branches on `run_kind` -- prior to this phase, EVERY
    slot (including `replica_transition`) was reported using the
    steady-state timing block unconditionally, which produced a
    factually wrong report for transition slots (`replicas: null`, a
    180s/60s duration instead of the transition's own real 240s/60s-
    trigger/300s-total plan; Phase 15B.5A sec.6(c)). The steady-state
    branch below is BYTE-IDENTICAL to the pre-15B.5B implementation.

    Phase 16I.1: `calibration_root` is a new, CODE-LEVEL-ONLY keyword
    parameter (default reproduces the exact historical value) so a trusted
    caller executing under a separately-frozen campaign context reports its
    own isolated artifact destination, never the historical one."""
    ledger_slot = ledger["slots"][str(slot["campaign_sequence_index"])]
    transport_discovery = transport_discovery or {}
    transport_fields = {
        # Phase 15B.3N.3 (CORR-40, Protocol V5): structural, read-only
        # resolution of the planned transport's inputs (resolved by the
        # caller, see `transport_discovery` -- proves the NodePort/Docker-
        # network/node-IP/relay-image can all be resolved BEFORE any
        # mutation, without creating the relay container, deleting the
        # HPA, scaling the Deployment, or sending traffic.
        "planned_transport_type": TRANSPORT_TYPE_DOCKER_NETWORK_TCP_RELAY,
        "planned_transport_docker_network": transport_discovery.get("docker_network"),
        "planned_transport_node_ip": transport_discovery.get("node_ip"),
        "planned_transport_nodeport": transport_discovery.get("nodeport"),
        "planned_transport_relay_image_identity": transport_discovery.get("relay_image_identity"),
    }

    if is_replica_transition_slot(slot):
        timing = resolve_transition_timing_plan(manifest)
        schedule_full_path = PROJECT_ROOT / slot["schedule_path"]
        expected_requests = count_schedule_data_rows(schedule_full_path) if schedule_full_path.exists() else None
        replicas_before = slot.get("replicas_before")
        replicas_after = slot.get("replicas_after")
        return {
            "campaign_sequence_index": slot["campaign_sequence_index"],
            "planned_slot_id": slot["planned_slot_id"],
            "run_kind": slot["run_kind"],
            "repeat_number": slot["repeat_number"],
            "replicas_before": replicas_before,
            "replicas_after": replicas_after,
            "transition_direction": (
                transition_direction(int(replicas_before), int(replicas_after))
                if replicas_before is not None and replicas_after is not None
                else None
            ),
            "offered_rps": slot.get("offered_rps"),
            "intensity": manifest["scope"]["application_intensity"],
            "schedule_path": slot["schedule_path"],
            "computed_schedule_sha256": schedule_hash,
            "expected_schedule_sha256": slot["schedule_sha256"],
            "expected_requests": expected_requests,
            "pre_run_replica_stabilization_seconds": timing["pre_run_replica_stabilization_seconds"],
            "pre_transition_traffic_steady_seconds": timing["pre_transition_traffic_steady_seconds"],
            "trigger_at_traffic_elapsed_seconds": timing["trigger_at_traffic_elapsed_seconds"],
            "post_transition_observation_seconds": timing["post_transition_observation_seconds"],
            "traffic_duration_seconds": timing["total_traffic_seconds"],
            "total_run_seconds": timing["total_seconds_per_run"],
            "scientific_role": TRANSITION_SCIENTIFIC_ROLE,
            "connection_mode": manifest["arrival_process"]["connection_mode"],
            "timeout_seconds": manifest["generator_capacity"]["default_timeout_seconds"],
            "max_in_flight": manifest["generator_capacity"]["frozen_calibration_max_in_flight"],
            "expected_workload_endpoint": manifest["scope"]["endpoint"],
            "artifact_destination_root": str(CALIBRATION_ROOT),
            "non_official_artifact_destination_root": str(TRANSPORT_VALIDATION_ROOT / "replica_transition_smoke"),
            "current_ledger_state_for_this_slot": ledger_slot["state"],
            **transport_fields,
        }

    if is_workload_transition_slot(slot):
        timing = resolve_workload_transition_timing_plan(manifest)
        schedule_full_path = PROJECT_ROOT / slot["schedule_path"]
        expected_requests = count_schedule_data_rows(schedule_full_path) if schedule_full_path.exists() else None
        return {
            "campaign_sequence_index": slot["campaign_sequence_index"],
            "planned_slot_id": slot["planned_slot_id"],
            "run_kind": slot["run_kind"],
            "repeat_number": slot["repeat_number"],
            "replicas": slot.get("replicas"),
            "rps_before": slot.get("rps_before"),
            "rps_after": slot.get("rps_after"),
            "intensity": manifest["scope"]["application_intensity"],
            "schedule_path": slot["schedule_path"],
            "computed_schedule_sha256": schedule_hash,
            "expected_schedule_sha256": slot["schedule_sha256"],
            "expected_requests": expected_requests,
            "pre_run_replica_stabilization_seconds": timing["pre_run_replica_stabilization_seconds"],
            "pre_step_traffic_seconds": timing["pre_step_traffic_seconds"],
            "workload_step_elapsed_seconds": timing["trigger_at_traffic_elapsed_seconds"],
            "post_step_observation_seconds": timing["post_step_observation_seconds"],
            "traffic_duration_seconds": timing["total_traffic_seconds"],
            "total_run_seconds": timing["total_seconds_per_run"],
            "workload_step_owner": "schedule",
            "scientific_role": WORKLOAD_TRANSITION_SCIENTIFIC_ROLE,
            "connection_mode": manifest["arrival_process"]["connection_mode"],
            "timeout_seconds": manifest["generator_capacity"]["default_timeout_seconds"],
            "max_in_flight": manifest["generator_capacity"]["frozen_calibration_max_in_flight"],
            "expected_workload_endpoint": manifest["scope"]["endpoint"],
            "artifact_destination_root": str(CALIBRATION_ROOT),
            "non_official_artifact_destination_root": str(TRANSPORT_VALIDATION_ROOT / "workload_transition_smoke"),
            "current_ledger_state_for_this_slot": ledger_slot["state"],
            **transport_fields,
        }

    ss_timing = manifest["steady_state_timing"]
    arrival = manifest["arrival_process"]
    gen_cap = manifest["generator_capacity"]
    return {
        "campaign_sequence_index": slot["campaign_sequence_index"],
        "planned_slot_id": slot["planned_slot_id"],
        "run_kind": slot["run_kind"],
        "repeat_number": slot["repeat_number"],
        "replicas": slot.get("replicas"),
        "offered_rps": slot.get("offered_rps"),
        "intensity": manifest["scope"]["application_intensity"],
        "schedule_path": slot["schedule_path"],
        "computed_schedule_sha256": schedule_hash,
        "expected_schedule_sha256": slot["schedule_sha256"],
        "traffic_duration_seconds": ss_timing["traffic_duration_seconds"],
        "stabilization_duration_seconds": ss_timing["fixed_replica_stabilization_seconds"],
        "warmup_window_seconds": ss_timing["warmup_window_seconds"],
        "scored_window_seconds": ss_timing["scored_window_seconds"],
        "decision_aligned_extraction_timestamps": ss_timing["decision_aligned_extraction_timestamps_relative_to_traffic_onset"],
        "connection_mode": arrival["connection_mode"],
        "timeout_seconds": gen_cap["default_timeout_seconds"],
        "max_in_flight": gen_cap["frozen_calibration_max_in_flight"],
        "expected_workload_endpoint": manifest["scope"]["endpoint"],
        "artifact_destination_root": str(calibration_root),
        "current_ledger_state_for_this_slot": ledger_slot["state"],
        **transport_fields,
    }


# ---------------------------------------------------------------------------
# CORR-54 (Phase 16L.5.1): pre-traffic tooling-failure control-layer signal.
#
# A REQUIRED tooling/resolution step (e.g. establishing the V5 relay
# transport) can fail before any workload traffic has been generated.
# Phase16L.5 (CORR-53) already stopped this from crashing the process
# uncaught, but the failure then fell through into the SAME
# `invalidity_reasons` list that also drives scientific
# `execution_structural_validity` -- meaning a pre-traffic tooling failure
# and a genuinely-executed structurally-invalid traffic run were
# indistinguishable in the ledger, both recorded as scientific
# `EXECUTION_INVALID` / `LEDGER_STATE_INVALID_RETRY_REQUIRED`.
#
# `PreTrafficToolingFailure` is raised at the exact two call sites where a
# transport-establishment failure is detected BEFORE any traffic duration
# wait, metrics collection, or route-integrity/structural-validity
# computation has occurred. It is caught at the attempt's own outer
# boundary in `main()`, deliberately BEFORE the scientific
# classification/ledger-write code is ever reached: no
# `execution_structural_validity` value is computed, no `route_integrity`
# value is computed, and `atomic_write_ledger()` is never called for this
# attempt. The physical attempt directory (and its evidence up to the
# failure point) is preserved, and a dedicated
# `pre_traffic_tooling_failure.json` artifact records the exact failure --
# distinct from, and never confused with, a scientific attempt record.
#
# This does NOT change the scientific meaning of `wrong_traffic_path_used`
# for the one OTHER, unrelated call site (route-integrity classification
# `FAILED` after traffic was actually generated and then the transport
# broke mid-run) -- that remains an executed, scientifically classified
# run exactly as before.
class PreTrafficToolingFailure(Exception):
    """Raised when a required pre-traffic tooling/resolution step fails
    before any traffic has been generated. Never allowed to reach the
    scientific structural-classification or ledger-write code."""

    def __init__(self, reason: str, detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


EXIT_PRE_TRAFFIC_TOOLING_FAILURE = 3


# ---------------------------------------------------------------------------
# Cleanup registry -- guarantees HPA restoration on success, invalid run,
# exception, SIGINT, SIGTERM, or ordinary harness failure.
# ---------------------------------------------------------------------------


class CleanupRegistry:
    def __init__(self) -> None:
        self._callbacks: List[Tuple[str, Any]] = []
        self._executed = False

    def register(self, name: str, callback) -> None:
        self._callbacks.append((name, callback))

    def run_all(self) -> List[Tuple[str, bool, str]]:
        if self._executed:
            return []
        self._executed = True
        results = []
        for name, callback in reversed(self._callbacks):
            try:
                callback()
                results.append((name, True, "ok"))
            except Exception as exc:  # noqa: BLE001 -- cleanup must never raise past this point
                results.append((name, False, str(exc)))
        return results

    @property
    def executed(self) -> bool:
        return self._executed

    def install_signal_and_exit_handlers(self) -> None:
        atexit.register(self.run_all)

        def _handler(signum, _frame):
            self.run_all()
            sys.exit(128 + signum)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# kubectl/minikube/docker helpers (thin subprocess wrappers)
# ---------------------------------------------------------------------------


def _run(cmd: List[str], timeout: Optional[float] = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def kubectl_jsonpath(args: List[str]) -> str:
    proc = _run(["kubectl"] + args)
    return proc.stdout.strip()


def preflight_checks() -> List[Tuple[bool, str]]:
    """Read-only checks. Every result is (ok, message); the harness STOPS
    before any mutation if any check fails."""
    results: List[Tuple[bool, str]] = []

    docker_ok = _run(["docker", "info"], timeout=15).returncode == 0
    results.append((docker_ok, "Docker daemon reachable" if docker_ok else "Docker daemon unreachable"))

    mk = _run(["minikube", "status", "-p", "minikube"], timeout=15)
    mk_ok = "host: Running" in mk.stdout
    results.append((mk_ok, "Minikube host Running" if mk_ok else f"Minikube not running: {mk.stdout}"))

    kind_ps = _run(["docker", "ps", "--format", "{{.Names}}"], timeout=15)
    running_kind = classify_external_container_contamination(kind_ps.stdout.splitlines())
    kind_ok = len(running_kind) == 0
    results.append((kind_ok, "No unrelated Kind containers running" if kind_ok else f"Unrelated Kind containers running: {running_kind}"))

    node_ready = "True" in kubectl_jsonpath(["get", "nodes", "minikube", "-o", "jsonpath={.status.conditions[?(@.type==\"Ready\")].status}"])
    results.append((node_ready, "Node minikube Ready" if node_ready else "Node minikube not Ready"))

    ms_args = kubectl_jsonpath(["get", "deployment", "-n", "kube-system", "metrics-server", "-o", "jsonpath={.spec.template.spec.containers[0].args}"])
    ms_res_ok = "metric-resolution=15s" in ms_args
    results.append((ms_res_ok, "Metrics Server at 15s resolution" if ms_res_ok else f"Metrics Server resolution unexpected: {ms_args}"))

    prom_phase = kubectl_jsonpath(["get", "pods", "-n", MONITORING_NS, "-l", "app.kubernetes.io/name=prometheus", "-o", "jsonpath={.items[0].status.phase}"])
    prom_ok = prom_phase == "Running"
    results.append((prom_ok, "Prometheus pod Running" if prom_ok else f"Prometheus pod not Running: {prom_phase!r}"))

    rl_ps = _run(["ps", "aux"], timeout=15)
    # Phase 15B.3D: match on actual RL-controller/PPO-training SCRIPT
    # filenames only (not a bare substring anywhere in the whole ps aux
    # line). The original substring check ("autoscal" among the tokens)
    # produced a systematic false positive on this project's OWN
    # directory name (rl-k8s-autosc-AL-er contains "autoscal"), meaning
    # it could never pass while this harness itself -- or literally any
    # script under this project's path -- was running via a shell
    # wrapper whose own ps aux entry echoes the invoking command line.
    RL_CONTROLLER_SCRIPT_TOKENS = ("rl_agent.py", "rl_controller.py", "train_ppo.py", "ppo_train.py")
    rl_proc = [
        line
        for line in rl_ps.stdout.splitlines()
        if "grep" not in line and any(tok in line for tok in RL_CONTROLLER_SCRIPT_TOKENS)
    ]
    rl_ok = len(rl_proc) == 0
    results.append((rl_ok, "No RL/custom-controller process detected" if rl_ok else f"RL-like process detected: {rl_proc}"))

    hpa_target = kubectl_jsonpath(["get", "hpa", HPA_NAME, "-n", APP_NS, "-o", "jsonpath={.spec.metrics[0].resource.target.averageUtilization}"])
    hpa_target_ok = hpa_target == "260"
    results.append((hpa_target_ok, "HPA target=260% verified before removal" if hpa_target_ok else f"HPA target unexpected: {hpa_target!r}"))

    min_r = kubectl_jsonpath(["get", "hpa", HPA_NAME, "-n", APP_NS, "-o", "jsonpath={.spec.minReplicas}"])
    max_r = kubectl_jsonpath(["get", "hpa", HPA_NAME, "-n", APP_NS, "-o", "jsonpath={.spec.maxReplicas}"])
    bounds_ok = min_r == "1" and max_r == "8"
    results.append((bounds_ok, "HPA min/max=1/8" if bounds_ok else f"HPA bounds unexpected: min={min_r} max={max_r}"))

    cpu_req = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.spec.template.spec.containers[0].resources.requests.cpu}"])
    cpu_lim = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.spec.template.spec.containers[0].resources.limits.cpu}"])
    mem_req = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.spec.template.spec.containers[0].resources.requests.memory}"])
    mem_lim = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.spec.template.spec.containers[0].resources.limits.memory}"])
    res_ok = (cpu_req, cpu_lim, mem_req, mem_lim) == ("100m", "500m", "64Mi", "256Mi")
    results.append((res_ok, "demo-app resources exact (100m/500m/64Mi/256Mi)" if res_ok else f"demo-app resources unexpected: {cpu_req}/{cpu_lim} {mem_req}/{mem_lim}"))

    loadtest_sel = kubectl_jsonpath(["get", "svc", LOADTEST_SVC, "-n", APP_NS, "-o", "jsonpath={.spec.selector}"])
    demo_sel = kubectl_jsonpath(["get", "svc", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.spec.selector}"])
    svc_ok = bool(loadtest_sel) and loadtest_sel == demo_sel
    results.append((svc_ok, "demo-app-loadtest selector matches demo-app" if svc_ok else f"Service selector mismatch: {loadtest_sel!r} vs {demo_sel!r}"))

    affinity = kubectl_jsonpath(["get", "svc", LOADTEST_SVC, "-n", APP_NS, "-o", "jsonpath={.spec.sessionAffinity}"])
    affinity_ok = affinity in ("", "None")
    results.append((affinity_ok, "demo-app-loadtest sessionAffinity=None" if affinity_ok else f"sessionAffinity unexpected: {affinity!r}"))

    stale = _run(["ps", "aux"], timeout=15)
    stale_procs = [
        line for line in stale.stdout.splitlines()
        if any(tok in line for tok in ("traffic_generator", "metrics_collector", "pod_readiness_collector", "per_pod_counter_collector"))
        and "grep" not in line
    ]
    stale_ok = len(stale_procs) == 0
    results.append((stale_ok, "No stale calibration processes" if stale_ok else f"Stale calibration process(es) detected: {stale_procs}"))

    return results


def delete_hpa() -> None:
    _run(["kubectl", "delete", "hpa", HPA_NAME, "-n", APP_NS, "--ignore-not-found"])


def restore_hpa() -> None:
    _run(["kubectl", "apply", "-f", str(HPA_MANIFEST_PATH)])


def verify_no_hpa_targets_deployment() -> bool:
    count = kubectl_jsonpath(["get", "hpa", "-n", APP_NS, "--no-headers"])
    return count == ""


def scale_and_wait(replicas: int, timeout_seconds: int = 90) -> Dict[str, int]:
    _run(["kubectl", "scale", "deployment", DEPLOYMENT, "-n", APP_NS, f"--replicas={replicas}"])
    _run(["kubectl", "rollout", "status", f"deployment/{DEPLOYMENT}", "-n", APP_NS, "--timeout=60s"], timeout=70)

    deadline = time.monotonic() + timeout_seconds
    state = {"desired": -1, "current": -1, "ready": -1, "available": -1}
    while time.monotonic() < deadline:
        desired = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.spec.replicas}"])
        current = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.status.replicas}"])
        ready = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.status.readyReplicas}"])
        available = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.status.availableReplicas}"])
        try:
            state = {"desired": int(desired or -1), "current": int(current or -1), "ready": int(ready or -1), "available": int(available or -1)}
        except ValueError:
            state = {"desired": -1, "current": -1, "ready": -1, "available": -1}
        if all(v == replicas for v in state.values()):
            return state
        time.sleep(2)
    return state


def wait_for_idle_one_replica(timeout_seconds: int = 120) -> Dict[str, int]:
    deadline = time.monotonic() + timeout_seconds
    state = {"desired": -1, "current": -1, "ready": -1, "available": -1}
    while time.monotonic() < deadline:
        desired = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.spec.replicas}"])
        current = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.status.replicas}"])
        ready = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.status.readyReplicas}"])
        available = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.status.availableReplicas}"])
        try:
            state = {"desired": int(desired or -1), "current": int(current or -1), "ready": int(ready or -1), "available": int(available or -1)}
        except ValueError:
            state = {"desired": -1, "current": -1, "ready": -1, "available": -1}
        if all(v == 1 for v in state.values()):
            return state
        time.sleep(3)
    return state


def get_pod_names(label_selector: str = "app=demo-app") -> List[str]:
    out = kubectl_jsonpath(["get", "pods", "-n", APP_NS, "-l", label_selector, "-o", "jsonpath={.items[*].metadata.name}"])
    return out.split() if out else []


def _extract_port_from_url(url: Optional[str]) -> Optional[int]:
    """Phase 15B.3M.3: pure parse of the tunnel's own reported URL (e.g.
    `http://127.0.0.1:50408`) into its local port number, for the
    read-only listener check below. Returns None if `url` is falsy or has
    no parseable port -- never guesses a port."""
    if not url:
        return None
    try:
        return urllib.parse.urlparse(url).port
    except ValueError:
        return None


def check_local_listener_present(port: int) -> Optional[bool]:
    """Phase 15B.3M.3 (CORR-39 proof-sufficiency hardening): read-only local
    TCP listener check (`lsof -nP -iTCP:<port> -sTCP:LISTEN`). Proves only
    that SOME process is bound to 127.0.0.1:<port> in LISTEN state -- it
    does not identify which process, nor whether it is actually forwarding
    correctly. This is deliberately SUPPORTING evidence only (see
    `classify_route_integrity`'s own docstring): a present listener never
    by itself confirms route integrity, and an absent one is direct
    evidence of a harness-side transport fault. Sends no traffic. Returns
    `None` (never fabricated True/False) if the check itself could not be
    performed (e.g. `lsof` unavailable in this environment)."""
    try:
        proc = _run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], timeout=5)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    return bool(proc.stdout.strip())


def resolve_local_listener_owner(port: int) -> Dict[str, Optional[str]]:
    """Phase 15B.3N.1 (workload-transport root-cause hardening): PURE
    DIAGNOSTIC enrichment only -- never consulted by `classify_route_
    integrity_window()`/`classify_route_integrity()`, which continue to
    rely solely on `check_local_listener_present()`'s presence/absence
    boolean. Root-cause investigation (Phase 15B.3N.1) established, by
    direct process-tree inspection on this Docker-driver/macOS
    environment, that `minikube service --url` (and `minikube tunnel`,
    which uses the identical mechanism) spawns a CHILD `ssh` process that
    actually owns the local listening socket -- the parent `minikube`
    CLI process (whose PID the harness already tracks via `tunnel_proc`)
    can remain alive (`Popen.poll() is None`) for the entire run even
    after this child dies and the listener disappears with it. This
    function parses the same read-only `lsof -nP -iTCP:<port>
    -sTCP:LISTEN` output already used by `check_local_listener_present()`
    to additionally recover the ACTUAL listener-owning PID and command
    name, purely for forensic/provenance purposes -- so a future
    diagnostic review can directly see whether the listener-owning PID
    equals, or differs from, the tracked parent tunnel PID, without
    changing route-integrity classification in any way. Returns
    `{"listener_owner_pid": None, "listener_owner_command": None}`
    (never fabricated) if the check cannot be performed or no listener
    is present."""
    try:
        proc = _run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], timeout=5)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return {"listener_owner_pid": None, "listener_owner_command": None}
    lines = [line for line in proc.stdout.splitlines() if line and not line.startswith("COMMAND")]
    if not lines:
        return {"listener_owner_pid": None, "listener_owner_command": None}
    fields = lines[0].split()
    command = fields[0] if len(fields) > 0 else None
    pid = fields[1] if len(fields) > 1 else None
    return {"listener_owner_pid": pid, "listener_owner_command": command}


def fetch_loadtest_endpoints_json() -> Dict[str, Any]:
    """Phase 15B.3M.2 (CORR-39 resolution): read-only snapshot of the
    `demo-app-loadtest` Service's own Endpoints object (the Service the
    external load generator actually routes through -- distinct from
    `demo-app`, which Prometheus scrapes). A Kubernetes API read only; sends
    no traffic through the loadtest path itself."""
    proc = _run(["kubectl", "get", "endpoints", LOADTEST_SVC, "-n", APP_NS, "-o", "json"], timeout=15)
    if proc.returncode != 0:
        return {"subsets": [], "_error": proc.stderr.strip()}
    try:
        return json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return {"subsets": [], "_error": "failed to parse kubectl output as JSON"}


def fetch_events_json(namespace: str = APP_NS) -> Dict[str, Any]:
    """Phase 15B.3C hardening: capture Kubernetes Events for supplemental
    restart/health-instability evidence. Read-only. Attempt1's restart
    cause was only recoverable because live Events happened to still be
    within their retention window at forensic-review time -- capturing
    them DURING this run removes that dependency for future runs.

    Phase 16I.6 hardening: forensic-evidence-only, like the rest of this
    file's post-run evidence captures -- a transient subprocess failure here
    must be recorded (via the pre-existing `_error` convention this function
    already uses for a non-zero exit / JSON-decode failure), not allowed to
    crash the attempt before scientific classification runs."""
    try:
        proc = _run(["kubectl", "get", "events", "-n", namespace, "-o", "json"], timeout=15)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        return {"items": [], "_error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {"items": [], "_error": proc.stderr.strip()}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"items": [], "_error": f"JSON decode error: {exc}"}


def fetch_previous_container_logs(pod_name: str, namespace: str = APP_NS) -> Tuple[str, bool]:
    """Best-effort: fetch the crashed container's previous log. Returns
    (log_text, available). Failure to obtain previous logs is NOT itself a
    structural-invalidity condition -- only whether it was attempted and
    whether it succeeded is recorded.

    Phase 16I.6 hardening: a transient subprocess failure is recorded via
    the same (log_text, available=False) shape this function already uses
    for a non-zero exit, rather than allowed to crash the attempt."""
    try:
        proc = _run(["kubectl", "logs", pod_name, "-n", namespace, "--previous", "--all-containers=true"], timeout=15)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        return f"{type(exc).__name__}: {exc}", False
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout, True
    return proc.stderr.strip(), False


def safe_fetch_current_pod_snapshot_json(namespace: str = APP_NS, label_selector: str = "app=demo-app") -> Dict[str, Any]:
    """Forensic-evidence-only: obtains a current pod snapshot used solely to
    detect container restarts and trigger `fetch_previous_container_logs`
    capture -- never consulted for execution_structural_validity,
    route_integrity classification, or steady_state_fit_eligibility.

    Phase 16I.6 hardening: this exact call (`kubectl get pods ...`) crashed
    a Phase16I.5 attempt with an uncaught `subprocess.TimeoutExpired` after
    traffic and metrics collection had already completed. A transient
    failure here must be recorded, not allowed to crash the attempt."""
    try:
        proc = _run(["kubectl", "get", "pods", "-n", namespace, "-l", label_selector, "-o", "json"], timeout=15)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        return {"items": [], "_error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {"items": [], "_error": proc.stderr.strip()}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"items": [], "_error": f"JSON decode error: {exc}"}


def capture_minikube_resource_context() -> Dict[str, Any]:
    """Phase 16I.6 hardening: diagnostic-only host-resource context (used
    solely for the `minikube_outer_resource_limits` metadata field, never
    consulted for scientific classification). A transient subprocess
    failure is recorded via the same `observation_error` convention this
    file's pre-existing dataplane-ingress observer already uses, rather
    than allowed to crash the attempt."""
    try:
        inspect = _run(["docker", "inspect", "minikube", "--format", "{{.HostConfig.NanoCpus}} {{.HostConfig.Memory}}"], timeout=15)
        nano_cpus, mem_bytes = (inspect.stdout.strip().split() + ["", ""])[:2]
        cpu_stat = _run(["minikube", "ssh", "--", "cat /sys/fs/cgroup/cpu.stat"], timeout=15)
        usage_usec = None
        for line in cpu_stat.stdout.splitlines():
            if line.startswith("usage_usec"):
                usage_usec = int(line.split()[1])
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        return {
            "outer_cpu_nano_cpus": None,
            "outer_cpu_cores": None,
            "outer_memory_bytes": None,
            "outer_memory_mib": None,
            "cgroup_cpu_usage_usec": None,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "observation_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "outer_cpu_nano_cpus": int(nano_cpus) if nano_cpus.isdigit() else None,
        "outer_cpu_cores": (int(nano_cpus) / 1e9) if nano_cpus.isdigit() else None,
        "outer_memory_bytes": int(mem_bytes) if mem_bytes.isdigit() else None,
        "outer_memory_mib": (int(mem_bytes) / (1024 * 1024)) if mem_bytes.isdigit() else None,
        "cgroup_cpu_usage_usec": usage_usec,
        "observation_error": None,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def start_loadtest_tunnel(experiment_id: str) -> Tuple[subprocess.Popen, str]:
    """Pre-V5 (SSH-tunnel-to-ClusterIP) transport. Retained UNCHANGED for
    historical-artifact compatibility (attempt1/attempt2 and all pre-V5
    slots were executed through this path) -- no longer called from the
    live V5 official-attempt path, which uses `start_loadtest_transport()`
    instead (Phase 15B.3N.3, CORR-40)."""
    log_path = Path(f"/tmp/mk_svc_{experiment_id}.log")
    proc = subprocess.Popen(
        ["minikube", "service", LOADTEST_SVC, "-n", APP_NS, "--url"],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )
    url = ""
    for _ in range(30):
        if log_path.exists():
            text = log_path.read_text()
            import re

            match = re.search(r"http://[0-9.]+:[0-9]+", text)
            if match:
                url = match.group(0)
                break
        time.sleep(1)
    return proc, url


MINIKUBE_CONTAINER_NAME = "minikube"


def resolve_minikube_docker_network() -> Optional[str]:
    """Phase 15B.3N.3 (CORR-40, Protocol V5): structural discovery (never a
    guess) of the Docker network the Minikube node container is attached
    to -- the relay container must join this SAME network to reach the
    node's NodePort (Phase 15B.3N.1/15B.3N.2 proved the Mac host itself
    cannot route there directly under the Docker driver).

    Phase 16L.5 (CORR-53): this function already returns None (never
    raises) for a non-zero exit or unparseable output -- callers
    (`start_loadtest_transport()`) already treat None as "could not
    resolve," surfacing it as a transport-establishment error rather than
    a scientific outcome. It did not previously extend that same
    already-established "return None, do not raise" contract to a hung
    `docker inspect` itself; a slow Docker daemon could therefore crash
    the whole attempt with an uncaught `TimeoutExpired` before any
    scientific classification logic ever ran. Sibling of
    `resolve_minikube_node_ip()`/`resolve_loadtest_nodeport()`, hardened
    together (same file, same caller, same latent defect class)."""
    try:
        proc = _run(["docker", "inspect", MINIKUBE_CONTAINER_NAME, "--format", "{{json .NetworkSettings.Networks}}"], timeout=15)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        return None
    if proc.returncode != 0:
        return None
    try:
        networks = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    names = list(networks.keys())
    return names[0] if names else None


def resolve_minikube_node_ip() -> Optional[str]:
    """Structural discovery of the Minikube node's own IP on its Docker
    network (`minikube ip`, the stable CLI subcommand -- never parsed from
    transient console text elsewhere).

    Phase 16L.5 (CORR-53): hardened alongside `resolve_minikube_docker_network()`
    and `resolve_loadtest_nodeport()` -- same latent unguarded-subprocess
    defect class, same "return None on any failure" contract already
    relied on by `start_loadtest_transport()`."""
    try:
        proc = _run(["minikube", "-p", "minikube", "ip"], timeout=15)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        return None
    ip = proc.stdout.strip()
    return ip or None


def resolve_loadtest_nodeport() -> Optional[int]:
    """Structural discovery of `demo-app-loadtest`'s own NodePort via the
    Kubernetes API -- never assumed or hardcoded.

    Phase 16L.5 (CORR-53): this function already returned None (never
    raised) for non-digit output, but a hung `kubectl get svc` itself
    raised an uncaught `subprocess.TimeoutExpired` that crashed the whole
    attempt process before any ledger entry could be written (Phase16L.4,
    validation_sequence_index=30, attempt1). `start_loadtest_transport()`
    already treats a None return here as a resolvable, non-crashing
    transport-establishment failure (`{"error": ...}`, surfaced by its
    caller as the pre-existing `wrong_traffic_path_used` invalidity
    reason -- never a fabricated VALID/CONFIRMED result, never traffic
    generation, never an official attempt); this hardening only extends
    that same already-established contract to the exception case,
    exactly as `check_prometheus_diagnostic_health()` (Phase 16I.4) and
    `safe_fetch_current_pod_snapshot_json()` (Phase 16I.6) already did for
    their own respective unguarded call sites."""
    try:
        proc = _run(["kubectl", "get", "svc", LOADTEST_SVC, "-n", APP_NS, "-o", "jsonpath={.spec.ports[0].nodePort}"], timeout=15)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        return None
    val = proc.stdout.strip()
    return int(val) if val.isdigit() else None


def resolve_relay_image_identity() -> Dict[str, Optional[str]]:
    """Read-only, no-pull resolution of the already-local relay image's
    identity (Phase 15B.3N.3 sec.6: 'do not silently pull/update to
    another relay implementation merely for convenience'). Used by
    `dry_run_report()` to prove the frozen relay image is actually present
    before any mutation, and by `start_loadtest_transport()` to record
    reproducibility provenance on every attempt."""
    id_proc = _run(["docker", "image", "inspect", RELAY_IMAGE, "--format", "{{.Id}}"], timeout=10)
    image_id = id_proc.stdout.strip() if id_proc.returncode == 0 and id_proc.stdout.strip() else None
    digest_proc = _run(["docker", "image", "inspect", RELAY_IMAGE, "--format", "{{index .RepoDigests 0}}"], timeout=10)
    repo_digest = digest_proc.stdout.strip() if digest_proc.returncode == 0 and digest_proc.stdout.strip() else None
    return {"image": RELAY_IMAGE, "image_id": image_id, "repo_digest": repo_digest}


def check_relay_container_state(container_name: str) -> Tuple[Optional[bool], Optional[str]]:
    """Read-only `docker inspect` of the relay container's own running
    state and exit code -- the V5 (container-based) analogue of
    `Popen.poll()` for the pre-V5 (process-based) SSH tunnel. Returns
    `(None, None)` (never fabricated) if the container cannot be
    inspected at all (e.g. already removed).

    CORR-46 (Phase 15B.5H): this is called both from the periodic
    route-integrity sampler thread (already independently guarded by its
    own caller-side try/except) AND directly from the main thread during
    STOP_TRAFFIC/FINAL CAPTURE -- an unguarded `TimeoutExpired` here could
    therefore crash the entire attempt, not just kill a background
    collector thread. A timeout is exactly the "cannot be inspected"
    case this function's own docstring already promises `(None, None)`
    for; it was simply not previously implemented for this specific
    failure mode."""
    try:
        proc = _run(["docker", "inspect", "--format", "{{.State.Running}}|{{.State.ExitCode}}", container_name], timeout=10)
    except subprocess.TimeoutExpired:
        return None, None
    if proc.returncode != 0:
        return None, None
    parts = proc.stdout.strip().split("|")
    running = (parts[0] == "true") if len(parts) > 0 and parts[0] in ("true", "false") else None
    exit_code = parts[1] if len(parts) > 1 else None
    return running, exit_code


def sample_dataplane_ingress_rules(nodeport: Optional[int]) -> Dict[str, Any]:
    """Phase 15B.3O.1 (CORR-41): read-only sample of the Minikube node's
    kube-proxy dataplane (`docker exec minikube iptables-save -c` for the
    `nat` and `filter` tables), delegated to the pure
    `dynamics_calibration_lib.parse_dataplane_ingress_rules()` for parsing.
    Sends no traffic through the loadtest path; performs no mutation.
    `nodeport` is recorded for provenance only (rule matching itself is by
    Service comment, not by port number). On any command failure, returns
    `regime=None`/`observation_error=<message>` rather than fabricating a
    regime -- an observer failure must never be misread as route failure
    (sec.24).

    Phase 15B.3P.3 (implementation hardening, pre-first-use amendment to
    CORR-43/Protocol V8 -- V8 has not yet been used by any OFFICIAL
    attempt): each `_run(..., timeout=5)` call can itself raise
    `subprocess.TimeoutExpired` on a genuinely hung `docker exec` -- this
    is NOT an `OSError` subclass and was NOT previously caught here, so a
    single slow command could silently kill the entire dataplane sampler
    thread (an uncaught exception in a daemon thread terminates that
    thread with no further samples for the rest of the attempt, and no
    visible error in the main process). This is now caught exactly like
    any other observer failure -- an observer TIMEOUT must never be
    misread as route failure any more than an observer command-exit
    failure is (sec.24), and it must never silently end all future
    sampling for the attempt."""
    try:
        nat_proc = _run(["docker", "exec", MINIKUBE_CONTAINER_NAME, "iptables-save", "-t", "nat", "-c"], timeout=5)
        filter_proc = _run(["docker", "exec", MINIKUBE_CONTAINER_NAME, "iptables-save", "-t", "filter", "-c"], timeout=5)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"regime": None, "rule_identity": None, "total_packet_count": None, "total_byte_count": None, "observation_error": str(exc)}
    if nat_proc.returncode != 0 or filter_proc.returncode != 0:
        return {
            "regime": None, "rule_identity": None, "total_packet_count": None, "total_byte_count": None,
            "observation_error": f"iptables-save non-zero exit (nat={nat_proc.returncode}, filter={filter_proc.returncode}): {(nat_proc.stderr or filter_proc.stderr).strip()}",
        }
    parsed = parse_dataplane_ingress_rules(nat_proc.stdout, filter_proc.stdout)
    parsed["observation_error"] = None
    parsed["nodeport"] = nodeport
    return parsed


def start_loadtest_transport(experiment_id: str) -> Dict[str, Any]:
    """Phase 15B.3N.3 (CORR-40, Protocol V5): launches the frozen official
    V5 workload transport -- a raw TCP relay container (`alpine/socat`,
    `--restart=no`) attached to the Minikube Docker network, forwarding a
    Docker-published `127.0.0.1`-only loopback host port to the Minikube
    node's `demo-app-loadtest` NodePort. Every input (Docker network, node
    IP, NodePort, the actual assigned host port) is resolved structurally,
    never assumed or manually guessed. No HTTP traffic is sent to
    establish or verify this -- only `docker run`/`docker inspect`/
    `docker port` reads and one `docker run` mutation. On any resolution
    or launch failure, returns a dict carrying `"error"` and no traffic is
    attempted through it; the caller is responsible for recording this as
    an invalidity reason (mirroring the pre-V5 tunnel's own
    "could not resolve URL" handling) rather than silently retrying or
    falling back to any other transport."""
    docker_network = resolve_minikube_docker_network()
    node_ip = resolve_minikube_node_ip()
    nodeport = resolve_loadtest_nodeport()
    if not docker_network or not node_ip or not nodeport:
        return {
            "error": "failed to resolve docker_network/node_ip/nodeport for V5 relay transport",
            "docker_network": docker_network,
            "node_ip": node_ip,
            "nodeport": nodeport,
        }

    container_name = f"loadtest-relay-{experiment_id}"
    container_port = 8080
    launch_command = build_relay_docker_run_command(
        container_name=container_name,
        docker_network=docker_network,
        host_port=None,  # Docker-assigned ephemeral port -- never manually guessed
        container_port=container_port,
        node_ip=node_ip,
        nodeport=nodeport,
    )
    launch_time = datetime.now(timezone.utc).isoformat()
    run_result = _run(launch_command, timeout=30)
    if run_result.returncode != 0:
        return {"error": f"docker run failed: {run_result.stderr.strip()}", "launch_command": launch_command}
    container_id = run_result.stdout.strip()

    port_proc = _run(["docker", "port", container_name, str(container_port)], timeout=10)
    host_ip: Optional[str] = None
    host_port: Optional[int] = None
    if port_proc.returncode == 0 and port_proc.stdout.strip():
        addr = port_proc.stdout.strip().splitlines()[0]
        host_ip, _, host_port_str = addr.rpartition(":")
        host_port = int(host_port_str) if host_port_str.isdigit() else None

    image_id_proc = _run(["docker", "inspect", container_name, "--format", "{{.Image}}"], timeout=10)
    image_id = image_id_proc.stdout.strip() if image_id_proc.returncode == 0 else None
    digest_proc = _run(["docker", "inspect", RELAY_IMAGE, "--format", "{{index .RepoDigests 0}}"], timeout=10)
    repo_digest = digest_proc.stdout.strip() if digest_proc.returncode == 0 and digest_proc.stdout.strip() else None

    return {
        "transport_type": TRANSPORT_TYPE_DOCKER_NETWORK_TCP_RELAY,
        "container_id": container_id,
        "container_name": container_name,
        "image": RELAY_IMAGE,
        "image_id": image_id or None,
        "repo_digest": repo_digest,
        "docker_network": docker_network,
        "node_ip": node_ip,
        "nodeport": nodeport,
        "host_ip": host_ip,
        "host_port": host_port,
        "container_port": container_port,
        "target_url": build_relay_target_url(host_port) if host_port else None,
        "launch_command": launch_command,
        "launch_time": launch_time,
    }


def start_prometheus_port_forward() -> Tuple[Optional[subprocess.Popen], bool]:
    check = subprocess.run(["curl", "-s", "-f", "http://localhost:9090/-/ready"], capture_output=True, timeout=5)
    if check.returncode == 0:
        return None, False
    proc = subprocess.Popen(
        ["kubectl", "port-forward", "-n", MONITORING_NS, "svc/kube-prom-stack-kube-prome-prometheus", "9090:9090"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)
    return proc, True


def check_prometheus_diagnostic_health() -> Dict[str, Any]:
    """Phase 16I.4: the post-run `prometheus_health` metadata field is
    diagnostic-only -- never consulted for execution_structural_validity,
    route_integrity classification, or steady_state_fit_eligibility (see the
    call sites below). A transient failure of this specific check must be
    recorded as diagnostic state, not allowed to raise and terminate the
    attempt before the actual scientific classification logic runs."""
    try:
        result = subprocess.run(
            ["curl", "-s", "-f", "http://localhost:9090/-/ready"],
            capture_output=True,
            timeout=5,
        )
        return {"healthy": result.returncode == 0, "error_type": None, "error_message": None}
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        return {"healthy": False, "error_type": type(exc).__name__, "error_message": str(exc)}


# ---------------------------------------------------------------------------
# Attempt metadata (pure assembly)
# ---------------------------------------------------------------------------

REQUIRED_ATTEMPT_METADATA_FIELDS = [
    "calibration_protocol_version",
    "campaign_sequence_index",
    "planned_slot_id",
    "execution_attempt_number",
    "attempt_id",
    "session_id",
    "wall_clock_start_utc",
    "wall_clock_end_utc",
    "run_kind",
    "cell_or_transition_id",
    "repeat_number",
    "replicas",
    "offered_rps",
    "intensity",
    "schedule_path",
    "schedule_sha256",
    "connection_mode",
    "timeout_seconds",
    "max_in_flight",
    "target_url",
    "hpa_absence_verified",
    "rl_absence_verified",
    "kind_state",
    "app_resource_spec",
    "minikube_outer_resource_limits",
    "prometheus_health",
    "metrics_server_resolution",
    "protocol_version_applied",
    "execution_structural_validity",  # Axis A (CORR-37)
    "invalidity_reasons",
    "steady_state_fit_eligibility",  # Axis B (CORR-37)
    "decision_points_observable_count",
    "decision_points_fit_eligible_count",
]


def build_attempt_metadata(**fields) -> Dict[str, Any]:
    missing = [f for f in REQUIRED_ATTEMPT_METADATA_FIELDS if f not in fields]
    if missing:
        raise ValueError(f"attempt metadata missing required fields: {missing}")
    return dict(fields)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Post-processing: decision-aligned extraction + CORR-35 routing analysis
# ---------------------------------------------------------------------------


def load_metrics_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["elapsed_seconds"] = float(row["elapsed_seconds"])
            rows.append(row)
    return rows


def load_traffic_summary_csv(path: Path) -> List[Dict[str, Any]]:
    """Phase 15B.3M.2 (CORR-39 resolution): the traffic generator's own
    per-second outcome summary, used to detect a material zero-success
    delivery interval for the route-integrity check -- never assumed
    equivalent to the 10s-cadence metrics-collector rows."""
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["elapsed_seconds"] = float(row["elapsed_seconds"])
            rows.append(row)
    return rows


def load_requests_csv(path: Path) -> List[Dict[str, Any]]:
    """Phase 15B.3M.4 (CORR-39 temporal-coverage hardening): the traffic
    generator's own per-request `requests.csv` (hardened Phase 15B.3M.2
    with `error_type`/`error_message`), the SOURCE of per-window HTTP-
    response delivery evidence -- `completion_elapsed_seconds` is on the
    same traffic-elapsed clock as evidence-window bounds, so no epoch
    conversion is needed for this file specifically. A dropped-overload or
    still-in-flight-at-shutdown row has no `completion_elapsed_seconds`
    (left as `None`, never fabricated as 0.0)."""
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row.get("completion_elapsed_seconds")
            row["completion_elapsed_seconds"] = float(raw) if raw not in (None, "") else None
            rows.append(row)
    return rows


def authoritative_ready_count_at_target(readiness_rows: List[Dict[str, Any]], target: float, tolerance: float = 5.0) -> int:
    """CORR-37: Axis B always consults the authoritative per-pod Kubernetes
    Ready-condition history (never a Prometheus/kube-state-metrics
    aggregate) for the true Ready count at a decision timestamp."""
    at_target = [r for r in readiness_rows if abs(r["elapsed_seconds"] - target) <= tolerance]
    return sum(1 for r in at_target if is_pod_ready_row(r))


def extract_decision_aligned_v2(
    metrics_rows: List[Dict[str, Any]],
    readiness_rows: List[Dict[str, Any]],
    decision_timestamps: List[int],
    staleness_threshold: float,
    intended_replicas: int,
) -> Dict[str, Any]:
    """Protocol V2 (CORR-37) decision-aligned extraction: for every frozen
    target, combine the unchanged v1 "latest valid sample" selection rule
    with the new Axis-B fit-eligibility assessment. Never imputes a
    missing/ineligible point; all four targets are always represented."""
    rows = []
    for target in decision_timestamps:
        sample = select_decision_sample(float(target), metrics_rows, staleness_threshold_seconds=staleness_threshold)
        ready_count = authoritative_ready_count_at_target(readiness_rows, float(target))
        fit_eligible, exclusion_reason = assess_decision_point_eligibility(sample, ready_count, intended_replicas)

        # Phase 15B.3F.1: derived CPU-normalization diagnostics, computed
        # only when both a valid CPU sample and a valid positive
        # current-replica count exist for this decision point -- never
        # fabricated for an unobservable/health-unstable point. This is
        # REPORTING metadata only; `cpu_cores` itself (the frozen empirical
        # dynamics target) is never altered.
        cpu_normalization_fields: Dict[str, Any] = {
            "mean_cpu_cores_per_current_replica": None,
            "hpa_comparable_cpu_percent": None,
            "aggregate_cpu_limit_percent": None,
        }
        if sample is not None:
            try:
                cpu_val = float(sample.get("cpu_cores"))
                current_val = int(float(sample.get("replicas_current")))
                norm = cpu_normalization(cpu_val, current_val)
                cpu_normalization_fields = {
                    "mean_cpu_cores_per_current_replica": norm["mean_cpu_cores_per_current_replica"],
                    "hpa_comparable_cpu_percent": norm["hpa_comparable_cpu_percent"],
                    "aggregate_cpu_limit_percent": norm["aggregate_cpu_limit_percent"],
                }
            except (TypeError, ValueError):
                pass  # missing/invalid cpu_cores or replicas_current -- leave as None, never fabricate

        rows.append(
            {
                "target_decision_timestamp_s": target,
                "decision_point_observable": sample is not None,
                "selected_source_sample_elapsed_s": sample["elapsed_seconds"] if sample else None,
                "observation_age_s": (target - sample["elapsed_seconds"]) if sample else None,
                "request_rate_rps": sample.get("request_rate_rps") if sample else None,
                "cpu_cores": sample.get("cpu_cores") if sample else None,
                "latency_p95_seconds": sample.get("latency_p95_seconds") if sample else None,
                "memory_bytes": sample.get("memory_bytes") if sample else None,
                "requests_in_progress": sample.get("requests_in_progress") if sample else None,
                "error_rate_rps": sample.get("error_rate_rps") if sample else None,
                "replicas_current_prometheus_sourced": sample.get("replicas_current") if sample else None,
                "replicas_ready_prometheus_sourced": sample.get("replicas_ready") if sample else None,
                "ready_count_authoritative_k8s_api": ready_count,
                "intended_replicas": intended_replicas,
                "steady_state_fit_eligible": fit_eligible,
                "exclusion_reason": exclusion_reason,
                **cpu_normalization_fields,
            }
        )
    observable_count = sum(1 for r in rows if r["decision_point_observable"])
    fit_eligible_count = sum(1 for r in rows if r["steady_state_fit_eligible"])
    return {
        "protocol_version_applied": 2,
        "corr_reference": "CORR-37",
        "decision_points": rows,
        "decision_points_observable_count": observable_count,
        "decision_points_fit_eligible_count": fit_eligible_count,
        "run_level_classification": run_level_fit_classification(fit_eligible_count, total_points=len(decision_timestamps)),
    }


def load_pod_readiness_history(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["timestamp_epoch"] = float(row["timestamp_epoch"])
            row["elapsed_seconds"] = float(row["elapsed_seconds"])
            rows.append(row)
    return rows


def load_per_pod_counters(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["timestamp_epoch"] = float(row["timestamp_epoch"])
            row["elapsed_seconds"] = float(row["elapsed_seconds"])
            row["raw_counter_value"] = float(row["raw_counter_value"])
            rows.append(row)
    return rows


def gate_and_classify_routing(result: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 15B.3C.1 (CORR-37 conformance fix): classification is
    permitted ONLY when the allocatable-domain reconstruction conserves
    for BOTH `expected` and `actual` (they must describe the identical
    interval domain -- see `compute_expected_and_actual`'s own docstring).
    Extracted as a pure function so the gate itself is directly unit
    testable, independent of the full live-cluster pipeline.

    `routing_evidence_sufficient` (Phase 15B.3J.1, index9 evidence closure)
    is a read-only derived summary of whether the FINAL classification
    reached a substantive routing verdict at all
    (`classification != "INSUFFICIENT_ROUTING_SAMPLE"`) -- distinct from
    `conservation_assessable` (whether the conservation-error arithmetic
    itself had a non-empty allocatable domain to sum over). A run can have
    `conservation_assessable=True` (a real, non-empty allocatable domain
    that genuinely conserved) yet still end up `routing_evidence_sufficient
    =False` (e.g. too few pods crossed `classify_routing`'s minimum
    expected-traffic threshold) -- these two fields intentionally answer
    different questions and neither substitutes for the other.
    """
    if abs(result["expected_conservation_error"]) > 0 or abs(result["actual_conservation_error"]) > 0:
        return {
            "classification": "INSUFFICIENT_ROUTING_SAMPLE",
            "reason": (
                f"allocatable-domain conservation failed "
                f"(expected_conservation_error={result['expected_conservation_error']}, "
                f"actual_conservation_error={result['actual_conservation_error']}) -- "
                f"classification withheld per CORR-37"
            ),
            **result,
            "routing_evidence_sufficient": False,
        }
    classification = classify_routing(result["expected"], result["actual"])
    return {
        **result,
        **classification,
        "routing_evidence_sufficient": classification["classification"] != "INSUFFICIENT_ROUTING_SAMPLE",
    }


def build_routing_inputs(
    readiness_rows: List[Dict[str, Any]],
    counter_rows: List[Dict[str, Any]],
    scored_window: Tuple[float, float],
    alignment_tolerance_seconds: float = 5.0,
) -> Tuple[List[float], Dict[float, List[str]], Dict[str, List[Tuple[float, float]]], Dict[float, List[str]]]:
    """Restrict to the scored window and build the (boundaries, ready_set,
    pod_raw_samples, relevant_pod_set) inputs `compute_expected_and_actual`
    needs.

    IMPORTANT: the readiness and per-pod-counter collectors are two
    independently-launched subprocesses, each with its OWN monotonic clock
    reference -- their `elapsed_seconds` values are never exactly equal
    even at "the same" nominal 10s tick (a bug found and fixed during
    Phase 15B.3A's first live run: naive exact-float matching on
    `elapsed_seconds` silently produced zero matches, so every expected
    count computed as 0.0). The correct shared time axis across processes
    is wall-clock `timestamp_epoch`. Boundaries are anchored to the
    readiness collector's own tick grid (elapsed_seconds relative to ITS
    start); each pod's counter sample nearest in EPOCH time to a given
    boundary (within `alignment_tolerance_seconds`, half the nominal
    cadence) is treated as that boundary's sample for that pod.

    `relevant_pod_set_at` (Phase 15B.3C.2, CORR-37 conformance tightening):
    every demo-app pod name present in the readiness snapshot at a given
    boundary, REGARDLESS of Ready status (`parse_pod_readiness_snapshot`
    writes one row per pod per tick unconditionally -- this is NOT a
    Ready-filtered set, unlike `ready_pod_set_at`). A NotReady-but-observed
    pod can still legitimately carry real counter traffic (e.g. endpoint/
    readiness-propagation lag), so it must remain eligible to force
    `compute_expected_and_actual`'s complete-joint-observability gate even
    though it never receives expected-routing credit.
    """
    lo, hi = scored_window
    readiness_in_window = [r for r in readiness_rows if lo <= r["elapsed_seconds"] <= hi]
    if not readiness_in_window:
        return [], {}, {}, {}

    boundaries = sorted({r["elapsed_seconds"] for r in readiness_in_window})
    boundary_epoch = {t: next(r["timestamp_epoch"] for r in readiness_in_window if r["elapsed_seconds"] == t) for t in boundaries}

    ready_pod_set_at: Dict[float, List[str]] = {}
    relevant_pod_set_at: Dict[float, List[str]] = {}
    for t in boundaries:
        rows_at_t = [r for r in readiness_in_window if r["elapsed_seconds"] == t]
        ready_pod_set_at[t] = [r["pod_name"] for r in rows_at_t if is_pod_ready_row(r)]
        relevant_pod_set_at[t] = sorted({r["pod_name"] for r in rows_at_t})

    lo_epoch = min(boundary_epoch.values()) - alignment_tolerance_seconds
    hi_epoch = max(boundary_epoch.values()) + alignment_tolerance_seconds
    counters_in_window = [r for r in counter_rows if lo_epoch <= r["timestamp_epoch"] <= hi_epoch]

    pods = sorted({r["pod"] for r in counters_in_window})
    pod_raw_samples: Dict[str, List[Tuple[float, float]]] = {p: [] for p in pods}
    for p in pods:
        rows_for_pod = [r for r in counters_in_window if r["pod"] == p]
        for t in boundaries:
            target_epoch = boundary_epoch[t]
            nearest = min(rows_for_pod, key=lambda r: abs(r["timestamp_epoch"] - target_epoch), default=None)
            if nearest is not None and abs(nearest["timestamp_epoch"] - target_epoch) <= alignment_tolerance_seconds:
                pod_raw_samples[p].append((t, nearest["raw_counter_value"]))

    return boundaries, ready_pod_set_at, pod_raw_samples, relevant_pod_set_at


# ---------------------------------------------------------------------------
# Phase 15B.5B: replica-transition execution.
#
# A NEW, self-contained execution path -- deliberately kept SEPARATE from
# the steady-state body of `main()` below rather than interleaved into it,
# so the steady-state flow (already exercised by 29 official attempts) is
# provably untouched by this addition. Reuses every existing shared helper
# (preflight_checks, delete_hpa, scale_and_wait, start_loadtest_transport,
# the dataplane/route-integrity pure functions, CleanupRegistry, etc.)
# exactly as the steady-state flow does.
# ---------------------------------------------------------------------------


def trigger_replica_transition(target_replicas: int) -> Dict[str, Any]:
    """Issues exactly the requested replica-count change and returns
    immediately -- does NOT wait for rollout completion (traffic must
    remain active and uninterrupted across the transition; see governing
    Phase 15B.5B sec.11). Supports both scale-up and scale-down directions
    identically (a `kubectl scale` command has no notion of direction)."""
    triggered_at_utc = datetime.now(timezone.utc).isoformat()
    triggered_at_monotonic = time.monotonic()
    proc = _run(["kubectl", "scale", "deployment", DEPLOYMENT, "-n", APP_NS, f"--replicas={target_replicas}"], timeout=15)
    return {
        "target_replicas": target_replicas,
        "command_returncode": proc.returncode,
        "command_stdout": proc.stdout.strip(),
        "command_stderr": proc.stderr.strip(),
        "triggered_at_utc": triggered_at_utc,
        "triggered_at_monotonic": triggered_at_monotonic,
    }


def sample_replica_transition_state(traffic_start_monotonic: float, trigger_monotonic: Optional[float]) -> Dict[str, Any]:
    """One read-only Kubernetes-API poll of the Deployment's own
    desired/current/ready/available replica counts plus the current Ready
    pod set and the loadtest Service's own endpoint count -- the building
    block for the high-resolution `replica_transition_timeline.csv`
    trajectory artifact (governing Phase 15B.5B sec.12). `trigger_monotonic`
    is `None` until the transition has actually been triggered; samples
    taken before it carry a negative `elapsed_from_trigger`.

    CORR-46 (Phase 15B.5H): this function is called from THREE places --
    the one-off pre-traffic baseline sample, the periodic background
    trajectory-sampler thread, and the one-off final-capture sample -- so a
    single transient `kubectl` timeout escaping unhandled here would not
    just silently kill a background thread (as it did for index33
    attempt1's `_sample_replica_trajectory_periodically`), it could also
    crash the MAIN thread outright if hit during the baseline or final
    capture. All four `kubectl_jsonpath()` calls plus `get_pod_names()`/
    `fetch_loadtest_endpoints_json()` are therefore now wrapped in a single
    atomic try/except, mirroring the already-established
    `sample_dataplane_ingress_rules()` observer-failure convention
    (sec.24: an observer timeout must never be misread as route failure,
    and must never silently end all future sampling). On timeout, every
    replica/endpoint field is left `None`/empty -- missing means missing,
    never fabricated from a stale prior value -- and the new
    `replica_state_observation_error` field records the failure for audit."""
    now_mono = time.monotonic()
    sampled_at_utc = datetime.now(timezone.utc).isoformat()
    elapsed_from_traffic_start = round(now_mono - traffic_start_monotonic, 3)
    elapsed_from_trigger = round(now_mono - trigger_monotonic, 3) if trigger_monotonic is not None else None

    def _int_or_none(raw: Optional[str]) -> Optional[int]:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    try:
        desired = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.spec.replicas}"])
        current = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.status.replicas}"])
        ready = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.status.readyReplicas}"])
        available = kubectl_jsonpath(["get", "deployment", DEPLOYMENT, "-n", APP_NS, "-o", "jsonpath={.status.availableReplicas}"])
        ready_pod_names = sorted(get_pod_names())
        endpoints = fetch_loadtest_endpoints_json()
    except subprocess.TimeoutExpired as exc:
        return {
            "sample_monotonic": now_mono,
            "sampled_at_utc": sampled_at_utc,
            "elapsed_from_traffic_start": elapsed_from_traffic_start,
            "elapsed_from_trigger": elapsed_from_trigger,
            "desired_replicas": None,
            "current_replicas": None,
            "ready_replicas": None,
            "available_replicas": None,
            "ready_pod_names": [],
            "ready_pod_count": 0,
            "loadtest_endpoint_ready_count": None,
            "endpoint_query_error": None,
            "replica_state_observation_error": str(exc),
        }

    endpoint_summary = parse_endpoints_ready_count(endpoints) if "_error" not in endpoints else None
    return {
        "sample_monotonic": now_mono,
        "sampled_at_utc": sampled_at_utc,
        "elapsed_from_traffic_start": elapsed_from_traffic_start,
        "elapsed_from_trigger": elapsed_from_trigger,
        "desired_replicas": _int_or_none(desired),
        "current_replicas": _int_or_none(current),
        "ready_replicas": _int_or_none(ready),
        "available_replicas": _int_or_none(available),
        "ready_pod_names": ready_pod_names,
        "ready_pod_count": len(ready_pod_names),
        "loadtest_endpoint_ready_count": endpoint_summary.get("ready_count") if endpoint_summary else None,
        "endpoint_query_error": endpoints.get("_error"),
        "replica_state_observation_error": None,
    }


REPLICA_TRANSITION_TIMELINE_CSV_FIELDNAMES = [
    "sample_index", "sampled_at_utc", "elapsed_from_traffic_start", "elapsed_from_trigger",
    "desired_replicas", "current_replicas", "ready_replicas", "available_replicas",
    "ready_pod_names", "ready_pod_count", "loadtest_endpoint_ready_count", "endpoint_query_error",
    "replica_state_observation_error",
]


def write_replica_transition_timeline_csv(path: Path, samples: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPLICA_TRANSITION_TIMELINE_CSV_FIELDNAMES)
        writer.writeheader()
        for i, sample in enumerate(samples):
            row = {k: sample.get(k) for k in REPLICA_TRANSITION_TIMELINE_CSV_FIELDNAMES if k != "sample_index"}
            row["sample_index"] = i
            row["ready_pod_names"] = json.dumps(sample.get("ready_pod_names", []))
            writer.writerow(row)


def execute_replica_transition_slot(
    args: argparse.Namespace,
    manifest: dict,
    sequence: List[dict],
    slot: dict,
    schedule_hash: str,
    ledger: dict,
    official: bool = True,
) -> int:
    """Phase 15B.5B: the replica-transition analogue of the steady-state
    body inside `main()`. Implements the state machine required by the
    governing task's sec.8:

        PRECHECK -> ISOLATE_HPA -> SET_INITIAL_REPLICA_COUNT ->
        WAIT_FOR_INITIAL_READY/STABILIZED -> START_COLLECTORS ->
        START_V5_RELAY -> START_TRAFFIC -> (pre-trigger traffic) ->
        TRIGGER REPLICA CHANGE -> (post-transition observation) ->
        STOP TRAFFIC -> FINAL CAPTURE -> CLEANUP -> RESTORE_HPA ->
        CANONICAL RESTORATION

    `official=False` (NON-OFFICIAL smoke validation only, governing
    sec.24): artifacts are written under
    `TRANSPORT_VALIDATION_ROOT/replica_transition_smoke/`, never under
    `CALIBRATION_ROOT` (the official campaign attempt namespace), and the
    official campaign ledger is never read for a state transition nor
    written to at all -- `ledger` is accepted only so official runs can
    reuse the identical call signature; it is never touched when
    `official=False`.
    """
    replicas_before = int(slot["replicas_before"])
    replicas_after = int(slot["replicas_after"])
    direction = transition_direction(replicas_before, replicas_after)
    timing = resolve_transition_timing_plan(manifest)

    if official:
        attempt_id = make_attempt_id(slot["planned_slot_id"], args.attempt_number)
        attempt_root = CALIBRATION_ROOT
    else:
        attempt_id = f"NONOFFICIAL_{slot['planned_slot_id']}_smoke_{utc_timestamp_id()}"
        attempt_root = TRANSPORT_VALIDATION_ROOT / "replica_transition_smoke"

    print(f"=== run_dynamics_calibration_slot [REPLICA_TRANSITION{'':s}{'' if official else ' NON-OFFICIAL'}]: "
          f"index={args.campaign_sequence_index} attempt_id={attempt_id} "
          f"({replicas_before}->{replicas_after}, {direction}) (pending preflight) ===")

    print("=== PRECHECK (preflight) ===")
    checks = preflight_checks()
    for ok, msg in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {msg}")
    failed = [msg for ok, msg in checks if not ok]
    if failed:
        print("Preflight FAILED -- stopping before any mutation. No HPA change, no traffic. No attempt directory created.")
        return 2

    attempt_dir = attempt_root / attempt_id
    if attempt_dir.exists():
        raise SystemExit(f"Attempt directory already exists, refusing to overwrite: {attempt_dir}")
    (attempt_dir / "cluster").mkdir(parents=True)
    (attempt_dir / "analysis").mkdir(parents=True)

    pre_run_context = capture_minikube_resource_context()
    write_json_atomic(attempt_dir / "cluster" / "minikube_resource_context_pre_run.json", pre_run_context)

    cleanup = CleanupRegistry()
    invalidity_reasons: List[str] = []
    procs_to_stop: List[Tuple[str, subprocess.Popen]] = []
    transport_lifecycle: Dict[str, Any] = {}
    relay_container_name: Dict[str, Optional[str]] = {"name": None}

    def _stop_background_procs():
        for name, p in procs_to_stop:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()

    def _stop_relay_container():
        name = relay_container_name["name"]
        if not name:
            return
        running, exit_code = check_relay_container_state(name)
        already_dead_before_cleanup = running is False
        transport_lifecycle["already_dead_before_cleanup"] = already_dead_before_cleanup
        transport_lifecycle["exit_code_before_cleanup"] = exit_code if already_dead_before_cleanup else None
        logs_proc = _run(["docker", "logs", name], timeout=15)
        docker_logs_path = Path(f"/tmp/relay_{attempt_id}.log")
        docker_logs_path.write_text((logs_proc.stdout or "") + (logs_proc.stderr or ""))
        transport_lifecycle["docker_logs_path"] = str(docker_logs_path)
        _run(["docker", "rm", "-f", name], timeout=15)
        transport_lifecycle["exit_code_after_cleanup"] = exit_code
        transport_lifecycle["cleanup_action"] = classify_tunnel_termination_action(bool(already_dead_before_cleanup))

    def _restore_hpa_and_idle():
        restore_hpa()
        time.sleep(2)
        wait_timeout = restoration_wait_seconds(initial_replica_count_for_restoration(slot))
        state = wait_for_idle_one_replica(timeout_seconds=wait_timeout)
        converged = all(v == 1 for v in state.values())
        if converged:
            print(f"  Restored HPA; idle state converged within {wait_timeout:.0f}s wait: {state}")
        else:
            print(
                f"  Restored HPA (confirmed applied), but natural scale-down convergence to 1/1/1/1 "
                f"did NOT complete within the {wait_timeout:.0f}s wait. Last observed state: {state}. "
                f"HPA itself IS restored and will continue converging on its own. No new official run "
                f"should start until 1/1/1/1 is independently confirmed."
            )

    cleanup.register("stop_background_processes", _stop_background_procs)
    cleanup.register("stop_relay_container", _stop_relay_container)
    cleanup.register("restore_hpa_and_wait_idle", _restore_hpa_and_idle)
    cleanup.install_signal_and_exit_handlers()

    start_utc = datetime.now(timezone.utc).isoformat()
    tunnel_url: Optional[str] = None
    hpa_absent: Optional[bool] = None
    trigger_result: Dict[str, Any] = {}
    replica_timeline_samples: List[Dict[str, Any]] = []
    traffic_start_monotonic: Optional[float] = None
    traffic_start_epoch: Optional[float] = None
    traffic_end_epoch: Optional[float] = None
    dataplane_periodic_samples: List[Dict[str, Any]] = []
    dataplane_reactive_resample_count = 0
    dataplane_anchor_sample_taken = False
    dataplane_acquisition_gap_exceeded_count = 0
    tunnel_periodic_samples: List[Dict[str, Any]] = []

    try:
        print("=== ISOLATE_HPA ===")
        delete_hpa()
        time.sleep(2)
        hpa_absent = verify_no_hpa_targets_deployment()
        if not hpa_absent:
            invalidity_reasons.append("hpa_unexpectedly_active: HPA still present after delete")

        print(f"=== SET_INITIAL_REPLICA_COUNT ({replicas_before}) ===")
        replica_state = scale_and_wait(replicas_before)
        replicas_ok = all(v == replicas_before for v in replica_state.values())
        if not replicas_ok:
            invalidity_reasons.append(f"wrong_initial_replica_count_achieved: {replica_state}")
        pod_names_pre = sorted(get_pod_names())
        write_json_atomic(attempt_dir / "cluster" / "pod_identities_pre_stabilization.json", {"pod_names": pod_names_pre})

        print(f"=== WAIT_FOR_INITIAL_READY/STABILIZED ({timing['pre_run_replica_stabilization_seconds']:.0f}s, no traffic) ===")
        time.sleep(timing["pre_run_replica_stabilization_seconds"])

        events_pre = fetch_events_json()
        write_json_atomic(attempt_dir / "cluster" / "kubernetes_events_pre.json", events_pre)
        loadtest_endpoints_pre = fetch_loadtest_endpoints_json()
        write_json_atomic(attempt_dir / "cluster" / "loadtest_endpoints_pre_traffic.json", loadtest_endpoints_pre)

        print("=== START_V5_RELAY ===")
        transport_result = start_loadtest_transport(attempt_id)
        tunnel_url = transport_result.get("target_url")
        if "error" in transport_result:
            invalidity_reasons.append(f"wrong_traffic_path_used: failed to establish V5 relay transport before traffic: {transport_result['error']}")
            transport_lifecycle.update(transport_result)
        else:
            relay_container_name["name"] = transport_result["container_name"]
            transport_lifecycle.update(transport_result)
            if not tunnel_url:
                invalidity_reasons.append("wrong_traffic_path_used: V5 relay launched but its host port could not be resolved")

        prom_proc, started_prom = start_prometheus_port_forward()
        if started_prom and prom_proc is not None:
            procs_to_stop.append(("prometheus_port_forward", prom_proc))

        gen_cap = manifest["generator_capacity"]
        collector_duration = int(timing["total_traffic_seconds"]) + 30

        print("=== START_COLLECTORS ===")
        metrics_proc = subprocess.Popen(
            [VENV_PYTHON, "-m", "collectors.metrics_collector",
             "--experiment-id", attempt_id, "--mode", "manual",
             "--traffic-profile", slot["cell_or_transition_id"],
             "--duration-seconds", str(collector_duration), "--seed", "42"],
        )
        procs_to_stop.append(("metrics_collector", metrics_proc))

        readiness_proc = subprocess.Popen(
            [VENV_PYTHON, "-m", "scripts.pod_readiness_collector",
             "--experiment-dir", str(attempt_dir / "cluster"),
             "--duration-seconds", str(collector_duration)],
        )
        procs_to_stop.append(("pod_readiness_collector", readiness_proc))

        time.sleep(1)

        schedule_full_path = PROJECT_ROOT / slot["schedule_path"]
        profile_path = schedule_full_path.parent / f"{slot['cell_or_transition_id']}.yaml"
        if not profile_path.exists():
            invalidity_reasons.append(f"wrong_or_unhashed_schedule: traffic profile YAML not found: {profile_path}")

        tunnel_port = _extract_port_from_url(tunnel_url)
        relay_name_for_sampling = relay_container_name["name"]
        _stop_periodic_sampling = threading.Event()

        def _sample_route_integrity_periodically(sample_start_mono: float) -> None:
            # CORR-46 (Phase 15B.5H): index33 attempt1's `TimeoutExpired`
            # from `fetch_loadtest_endpoints_json()` (and, latently,
            # `check_relay_container_state()`'s own unguarded `docker
            # inspect` call) escaped this loop unhandled and permanently
            # killed this thread (Thread-1) for the rest of that attempt.
            # The whole per-iteration body is now one atomic try/except,
            # mirroring `sample_dataplane_ingress_rules()`'s own observer-
            # failure convention: a single transient timeout becomes one
            # recorded `endpoint_query_error` sample (no field fabricated),
            # and the loop continues to its next scheduled iteration.
            while not _stop_periodic_sampling.wait(10):
                try:
                    endpoints = fetch_loadtest_endpoints_json()
                    container_running, container_exit_code = (
                        check_relay_container_state(relay_name_for_sampling) if relay_name_for_sampling else (None, None)
                    )
                    listener_present = check_local_listener_present(tunnel_port) if tunnel_port else None
                except subprocess.TimeoutExpired as exc:
                    tunnel_periodic_samples.append(
                        {
                            "elapsed_seconds_since_traffic_start": round(time.monotonic() - sample_start_mono, 1),
                            "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                            "tunnel_process_alive": None,
                            "tunnel_process_exit_code": None,
                            "local_listener_present": None,
                            "listener_port": tunnel_port,
                            "relay_container_running": None,
                            "relay_container_exit_code": None,
                            "host_listener_present": None,
                            "loadtest_endpoints": None,
                            "endpoint_query_error": str(exc),
                        }
                    )
                    continue
                tunnel_periodic_samples.append(
                    {
                        "elapsed_seconds_since_traffic_start": round(time.monotonic() - sample_start_mono, 1),
                        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                        "tunnel_process_alive": container_running,
                        "tunnel_process_exit_code": container_exit_code,
                        "local_listener_present": listener_present,
                        "listener_port": tunnel_port,
                        "relay_container_running": container_running,
                        "relay_container_exit_code": container_exit_code,
                        "host_listener_present": listener_present,
                        "loadtest_endpoints": parse_endpoints_ready_count(endpoints) if "_error" not in endpoints else None,
                        "endpoint_query_error": endpoints.get("_error"),
                    }
                )

        dataplane_nodeport = transport_lifecycle.get("nodeport")
        _stop_dataplane_sampling = threading.Event()
        _dataplane_samples_lock = threading.Lock()
        _last_dataplane_sample_taken_mono: List[Optional[float]] = [None]

        def _take_and_record_dataplane_sample(sample_start_mono: float) -> Dict[str, Any]:
            nonlocal dataplane_acquisition_gap_exceeded_count
            t_start = time.monotonic()
            sample = sample_dataplane_ingress_rules(dataplane_nodeport)
            with _dataplane_samples_lock:
                previous_taken_mono = _last_dataplane_sample_taken_mono[0]
                gap = None if previous_taken_mono is None else round(t_start - previous_taken_mono, 4)
                _last_dataplane_sample_taken_mono[0] = t_start
                gap_exceeded = gap is not None and gap > DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8
                if gap_exceeded:
                    dataplane_acquisition_gap_exceeded_count += 1
                sample["elapsed_seconds_since_traffic_start"] = round(time.monotonic() - sample_start_mono, 2)
                sample["sampled_at_utc"] = datetime.now(timezone.utc).isoformat()
                sample["dataplane_mode"] = "kube_proxy_iptables_nf_tables"
                sample["service_namespace"] = APP_NS
                sample["service_name"] = LOADTEST_SVC
                sample["actual_gap_seconds_from_previous_sample"] = gap
                sample["acquisition_gap_exceeded_assumed_bound"] = gap_exceeded
                dataplane_periodic_samples.append(sample)
            return sample

        def _sample_dataplane_ingress_periodically(sample_start_mono: float) -> None:
            nonlocal dataplane_reactive_resample_count
            previous_sample: Optional[Dict[str, Any]] = None
            while not _stop_dataplane_sampling.wait(DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7):
                sample = _take_and_record_dataplane_sample(sample_start_mono)
                if should_trigger_immediate_resample(previous_sample, sample):
                    if _stop_dataplane_sampling.wait(DATAPLANE_INGRESS_TRANSITION_RESAMPLE_DELAY_SECONDS):
                        previous_sample = sample
                        break
                    dataplane_reactive_resample_count += 1
                    sample = _take_and_record_dataplane_sample(sample_start_mono)
                previous_sample = sample

        # Phase 15B.5B sec.12: a THIRD sampler thread, distinct from the two
        # above (route-integrity/dataplane, both reused verbatim from the
        # steady-state flow) -- polls the Deployment's own desired/current/
        # ready/available replica counts plus the Ready pod set and loadtest
        # endpoint count, at the same 2.0s cadence already established for
        # dataplane acquisition (DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7).
        _stop_replica_sampling = threading.Event()
        _trigger_monotonic_box: List[Optional[float]] = [None]

        def _sample_replica_trajectory_periodically(sample_start_mono: float) -> None:
            while not _stop_replica_sampling.wait(DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7):
                replica_timeline_samples.append(
                    sample_replica_transition_state(sample_start_mono, _trigger_monotonic_box[0])
                )

        print("=== START_TRAFFIC (background; trigger issued mid-traffic while it continues) ===")
        traffic_start_epoch = datetime.now(timezone.utc).timestamp()
        traffic_start_monotonic = time.monotonic()

        sampler_thread = threading.Thread(target=_sample_route_integrity_periodically, args=(traffic_start_monotonic,), daemon=True)
        sampler_thread.start()
        dataplane_sampler_thread = threading.Thread(target=_sample_dataplane_ingress_periodically, args=(traffic_start_monotonic,), daemon=True)
        dataplane_sampler_thread.start()
        replica_sampler_thread = threading.Thread(target=_sample_replica_trajectory_periodically, args=(traffic_start_monotonic,), daemon=True)
        replica_sampler_thread.start()
        # One immediate sample before the pre-trigger wait even starts, so
        # the trajectory artifact has a genuine t=0 baseline row.
        replica_timeline_samples.append(sample_replica_transition_state(traffic_start_monotonic, None))

        gen_proc = subprocess.Popen(
            [VENV_PYTHON, "-m", "loadgen.traffic_generator",
             "--profile", str(profile_path),
             "--experiment-id", attempt_id,
             "--target-url", tunnel_url or "http://localhost:1",
             "--replay-schedule", str(schedule_full_path),
             "--arrival-process", "deterministic",
             "--connection-mode", manifest["arrival_process"]["connection_mode"],
             "--timeout", str(gen_cap["default_timeout_seconds"]),
             "--max-in-flight", str(gen_cap["frozen_calibration_max_in_flight"])],
        )
        procs_to_stop.append(("traffic_generator", gen_proc))

        # Phase 15B.5B sec.10: precise monotonic-clock trigger timing --
        # no scheduler/event-loop pattern exists elsewhere in this
        # synchronous, single-process harness, so a monotonic-anchored
        # `time.sleep()` for the remaining time until the frozen trigger
        # offset IS the established pattern (identical in kind to every
        # other `time.sleep(<frozen duration>)` call already used
        # throughout this harness for the steady-state stabilization wait).
        trigger_planned_monotonic = traffic_start_monotonic + timing["trigger_at_traffic_elapsed_seconds"]
        remaining = trigger_planned_monotonic - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        print(f"=== TRIGGER REPLICA CHANGE ({replicas_before}->{replicas_after}) while traffic continues ===")
        trigger_result = trigger_replica_transition(replicas_after)
        _trigger_monotonic_box[0] = trigger_result["triggered_at_monotonic"]
        trigger_result["trigger_planned_monotonic"] = trigger_planned_monotonic
        trigger_result["trigger_timing_error_seconds"] = round(trigger_result["triggered_at_monotonic"] - trigger_planned_monotonic, 3)
        if trigger_result["command_returncode"] != 0:
            invalidity_reasons.append(f"transition_trigger_command_failed: {trigger_result['command_stderr']}")

        print(f"=== POST-TRANSITION OBSERVATION (traffic continues to t={timing['total_traffic_seconds']:.0f}s) ===")
        gen_exit = gen_proc.wait(timeout=timing["total_traffic_seconds"] + 60)
        traffic_end_epoch = datetime.now(timezone.utc).timestamp()
        transport_lifecycle["traffic_start_epoch"] = traffic_start_epoch
        transport_lifecycle["traffic_end_epoch"] = traffic_end_epoch
        procs_to_stop = [(n, p) for n, p in procs_to_stop if p is not gen_proc]
        if gen_exit != 0:
            invalidity_reasons.append(f"collector_or_generator_structural_failure: traffic_generator exit code {gen_exit}")

        _take_and_record_dataplane_sample(traffic_start_monotonic)
        dataplane_anchor_sample_taken = True
        replica_timeline_samples.append(sample_replica_transition_state(traffic_start_monotonic, _trigger_monotonic_box[0]))

        print("=== STOP_TRAFFIC / FINAL CAPTURE ===")
        _stop_periodic_sampling.set()
        sampler_thread.join(timeout=5)
        transport_lifecycle["periodic_samples_during_traffic"] = tunnel_periodic_samples

        _stop_dataplane_sampling.set()
        dataplane_sampler_thread.join(timeout=5)

        _stop_replica_sampling.set()
        replica_sampler_thread.join(timeout=5)
        write_replica_transition_timeline_csv(attempt_dir / "cluster" / "replica_transition_timeline.csv", replica_timeline_samples)

        dataplane_ingress_timeline = {
            "attempt_id": attempt_id,
            "dataplane_ingress_sampling_interval_seconds": DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7,
            "dataplane_ingress_sampling_protocol_version": 7,
            "dataplane_ingress_transition_resample_delay_seconds": DATAPLANE_INGRESS_TRANSITION_RESAMPLE_DELAY_SECONDS,
            "dataplane_reactive_resample_count": dataplane_reactive_resample_count,
            "dataplane_anchor_sample_taken": dataplane_anchor_sample_taken,
            "dataplane_acquisition_worst_case_cadence_assumed_seconds": DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8,
            "dataplane_acquisition_gap_exceeded_count": dataplane_acquisition_gap_exceeded_count,
            "nodeport": dataplane_nodeport,
            "periodic_samples": sorted(dataplane_periodic_samples, key=lambda s: s["elapsed_seconds_since_traffic_start"]),
        }
        write_json_atomic(attempt_dir / "cluster" / "loadtest_dataplane_ingress_timeline.json", dataplane_ingress_timeline)

        container_running_at_end, container_exit_code_at_end = (
            check_relay_container_state(relay_container_name["name"]) if relay_container_name["name"] else (None, None)
        )
        transport_lifecycle["running_at_traffic_end"] = container_running_at_end
        transport_lifecycle["alive_at_traffic_end"] = container_running_at_end
        transport_lifecycle["host_listener_present_at_traffic_end"] = check_local_listener_present(tunnel_port) if tunnel_port else None
        transport_lifecycle["local_listener_present_at_traffic_end"] = transport_lifecycle["host_listener_present_at_traffic_end"]
        if container_running_at_end is False:
            transport_lifecycle["exit_code_at_traffic_end"] = container_exit_code_at_end
        transport_lifecycle["listener_owner_at_traffic_end"] = (
            resolve_local_listener_owner(tunnel_port) if tunnel_port else
            {"listener_owner_pid": None, "listener_owner_command": None}
        )

        loadtest_endpoints_post = fetch_loadtest_endpoints_json()
        write_json_atomic(attempt_dir / "cluster" / "loadtest_endpoints_post_traffic.json", loadtest_endpoints_post)

        pod_names_post = sorted(get_pod_names())
        write_json_atomic(attempt_dir / "cluster" / "pod_identities_post_traffic.json", {"pod_names": pod_names_post})
        # Phase 15B.5B sec.9: unlike the steady-state flow (which treats ANY
        # pre/post pod-identity change as invalidating -- replicas are held
        # fixed throughout), a replica-transition slot EXPECTS pod-identity
        # change by design (new pods created for scale-up, pods removed for
        # scale-down). This is never treated as an invalidity reason here.

        events_post = fetch_events_json()
        write_json_atomic(attempt_dir / "cluster" / "kubernetes_events_post.json", events_post)
        normalized_events = normalize_kubernetes_events(events_post)
        with open(attempt_dir / "cluster" / "kubernetes_events_normalized.csv", "w", newline="", encoding="utf-8") as ef:
            ew = csv.DictWriter(ef, fieldnames=KUBERNETES_EVENTS_NORMALIZED_CSV_FIELDNAMES)
            ew.writeheader()
            for row in normalized_events:
                ew.writerow(row)

        pods_snapshot_json = safe_fetch_current_pod_snapshot_json()
        current_snapshot = parse_pod_readiness_snapshot(
            pods_snapshot_json,
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).timestamp(),
            -1.0,
        )
        prev_logs_dir = attempt_dir / "cluster" / "previous_container_logs"
        prev_logs_dir.mkdir(parents=True, exist_ok=True)
        for row in current_snapshot:
            if int(row.get("restart_count", 0)) > 0:
                log_text, available = fetch_previous_container_logs(row["pod_name"])
                (prev_logs_dir / f"{row['pod_name']}.log").write_text(log_text)
                write_json_atomic(prev_logs_dir / f"{row['pod_name']}_availability.json", {"pod_name": row["pod_name"], "previous_log_available": available})

        print("=== Waiting for collectors to finish ===")
        metrics_exit = metrics_proc.wait(timeout=collector_duration + 60)
        readiness_exit = readiness_proc.wait(timeout=collector_duration + 60)
        procs_to_stop = [(n, p) for n, p in procs_to_stop if p not in (metrics_proc, readiness_proc)]
        if metrics_exit != 0:
            invalidity_reasons.append(f"collector_or_generator_structural_failure: metrics_collector exit code {metrics_exit}")
        if readiness_exit != 0:
            invalidity_reasons.append("collector_or_generator_structural_failure: pod readiness collector non-zero exit")

        readiness_metadata_path = attempt_dir / "cluster" / "pod_readiness_metadata.json"
        if readiness_metadata_path.exists():
            readiness_metadata = json.loads(readiness_metadata_path.read_text())
            contamination_ok, contamination_msg = check_no_external_contamination_during_run(readiness_metadata)
            if not contamination_ok:
                invalidity_reasons.append(contamination_msg)

        post_run_context = capture_minikube_resource_context()
        write_json_atomic(attempt_dir / "cluster" / "minikube_resource_context_post_run.json", post_run_context)

        prom_health_diagnostic = check_prometheus_diagnostic_health()
        prom_health = prom_health_diagnostic["healthy"]
        prom_health_error_type = prom_health_diagnostic["error_type"]
        prom_health_error_message = prom_health_diagnostic["error_message"]

    finally:
        cleanup.run_all()

    end_utc = datetime.now(timezone.utc).isoformat()

    if transport_lifecycle:
        docker_logs_path = Path(transport_lifecycle.get("docker_logs_path", ""))
        transport_lifecycle["captured_log_text"] = docker_logs_path.read_text() if docker_logs_path.exists() else None
        write_json_atomic(attempt_dir / "cluster" / "loadtest_transport_lifecycle.json", transport_lifecycle)

    # -------------------------------------------------------------------
    # Post-processing: reuses the IDENTICAL pure route-integrity/materiality
    # /V8-partitioning functions the steady-state flow uses (governing
    # Phase 15B.5B sec.17/19) -- Axis A structural validity for a
    # transition slot is evaluated by the SAME frozen rules, never a
    # transition-specific relaxation.
    #
    # `scored_window`/materiality fraction: FROZEN by CORR-44 (Phase
    # 15B.5C), superseding Phase 15B.5B's own disclosed-as-unfrozen
    # placeholder choice with the identical numeric values (no behavior
    # change for any already-executed attempt -- this run_kind has never
    # had an OFFICIAL attempt) -- see `replica_transition_route_scored_
    # window()`/`REPLICA_TRANSITION_MATERIALITY_FRACTION`'s own docstrings
    # for the full rationale (duration-relative materiality fraction,
    # confirmed from CORR-39's own original text; pre-trigger traffic is
    # genuine baseline, never discardable warmup).
    # -------------------------------------------------------------------
    traffic_dir = PROJECT_ROOT / "persistent-data/traffic" / attempt_id
    metrics_dir = PROJECT_ROOT / "persistent-data/metrics" / attempt_id
    try:
        (attempt_dir / "traffic").symlink_to(traffic_dir)
    except FileExistsError:
        pass
    try:
        (attempt_dir / "metrics").symlink_to(metrics_dir)
    except FileExistsError:
        pass

    traffic_metadata = {}
    traffic_meta_path = traffic_dir / "metadata.json"
    if traffic_meta_path.exists():
        traffic_metadata = json.loads(traffic_meta_path.read_text())
        ok, msg = check_offered_equals_sent(traffic_metadata)
        if not ok:
            invalidity_reasons.append(f"offered_not_equal_sent_due_to_undersized_generator_max_in_flight: {msg}")
        ok, msg = check_no_dropped_overload(traffic_metadata)
        if not ok:
            invalidity_reasons.append(f"generator_dropped_overload_due_to_harness_capacity_not_application_response: {msg}")
    else:
        invalidity_reasons.append("collector_or_generator_structural_failure: traffic metadata.json missing")

    route_integrity_result: Dict[str, Any] = {"classification": None}
    traffic_summary_path = traffic_dir / "traffic_summary.csv"
    dataplane_samples_for_windows = dataplane_periodic_samples
    if traffic_summary_path.exists():
        summary_rows = load_traffic_summary_csv(traffic_summary_path)
        traffic_duration_seconds = timing["total_traffic_seconds"]
        scored_window = replica_transition_route_scored_window(traffic_duration_seconds)
        materiality_threshold_seconds = replica_transition_materiality_threshold_seconds(traffic_duration_seconds)

        requests_rows: List[Dict[str, Any]] = []
        requests_csv_path = traffic_dir / "requests.csv"
        if requests_csv_path.exists():
            requests_rows = load_requests_csv(requests_csv_path)

        metrics_rows: List[Dict[str, Any]] = []
        metrics_csv_path = metrics_dir / "metrics.csv"
        if metrics_csv_path.exists():
            metrics_rows = load_metrics_csv(metrics_csv_path)

        all_zero_success_runs = find_all_zero_success_runs(summary_rows)
        material_intervals = [
            run for run in all_zero_success_runs
            if is_interval_material(
                run[0], run[1], traffic_duration_seconds, scored_window,
                materiality_fraction_threshold=REPLICA_TRANSITION_MATERIALITY_FRACTION,
            )
        ]

        periodic_samples = transport_lifecycle.get("periodic_samples_during_traffic", [])

        evidence_windows: List[Dict[str, Any]] = []
        for run_start, run_end in material_intervals:
            for window_start, window_end in partition_into_evidence_windows(
                run_start, run_end, min_window_seconds=MIN_EVIDENCE_WINDOW_SECONDS_V8
            ):
                observer_evidence = aggregate_observer_evidence_for_window(window_start, window_end, periodic_samples)
                delivered_evidence = compute_delivered_request_evidence_windows(
                    window_start_elapsed=window_start,
                    window_end_elapsed=window_end,
                    traffic_start_epoch=traffic_start_epoch,
                    requests_rows=requests_rows,
                    metrics_rows=metrics_rows,
                    counter_rows=[],
                )
                dataplane_evidence = compute_dataplane_ingress_evidence_windows(
                    window_start_elapsed=window_start,
                    window_end_elapsed=window_end,
                    traffic_start_epoch=traffic_start_epoch,
                    dataplane_samples=dataplane_samples_for_windows,
                )
                window_verdict = classify_route_integrity_window(
                    delivered_request_evidence=delivered_evidence["delivered_request_evidence"],
                    tunnel_process_alive=observer_evidence["tunnel_process_alive"],
                    local_listener_present=observer_evidence["local_listener_present"],
                    nodeport_dataplane_ingress_evidence=dataplane_evidence["nodeport_dataplane_ingress_evidence"],
                )
                route_evidence_sources = list(delivered_evidence["delivery_evidence_sources"])
                if dataplane_evidence["nodeport_dataplane_ingress_evidence"]:
                    route_evidence_sources.append("nodeport_dataplane_ingress")
                evidence_windows.append(
                    {
                        "material_interval": [run_start, run_end],
                        "window_start_elapsed": window_start,
                        "window_end_elapsed": window_end,
                        "tunnel_process_alive": observer_evidence["tunnel_process_alive"],
                        "local_listener_present": observer_evidence["local_listener_present"],
                        "relay_container_running": observer_evidence["tunnel_process_alive"],
                        "host_listener_present": observer_evidence["local_listener_present"],
                        "loadtest_ready_endpoint_count": observer_evidence["loadtest_ready_endpoint_count"],
                        "endpoint_query_error": observer_evidence["endpoint_query_error"],
                        "observer_sample_count": observer_evidence["sample_count"],
                        "http_response_delivery_evidence": delivered_evidence["http_response_delivery_evidence"],
                        "counter_delivery_evidence": delivered_evidence["counter_delivery_evidence"],
                        "rolling_server_rps_observed": delivered_evidence["rolling_server_rps_observed"],
                        "rolling_server_rps_supporting_only": delivered_evidence["rolling_server_rps_supporting_only"],
                        "nodeport_dataplane_ingress_evidence": dataplane_evidence["nodeport_dataplane_ingress_evidence"],
                        "dataplane_counter_intervals_used": dataplane_evidence["dataplane_counter_intervals_used"],
                        "dataplane_counter_intervals_rejected_cross_window": dataplane_evidence["dataplane_counter_intervals_rejected_cross_window"],
                        "dataplane_counter_intervals_rejected_reset": dataplane_evidence["dataplane_counter_intervals_rejected_reset"],
                        "dataplane_observer_error_present": dataplane_evidence["dataplane_observer_error_present"],
                        "dataplane_backend_regimes_with_evidence": dataplane_evidence["dataplane_backend_regimes_with_evidence"],
                        "delivered_request_evidence": delivered_evidence["delivered_request_evidence"],
                        "delivery_evidence_sources": delivered_evidence["delivery_evidence_sources"],
                        "route_evidence_sources": route_evidence_sources,
                        "window_route_status": window_verdict["window_route_status"],
                        "reason": window_verdict["reason"],
                    }
                )

        route_integrity_result = classify_route_integrity(material_intervals=material_intervals, evidence_windows=evidence_windows)
        route_integrity_timeline = {
            "attempt_id": attempt_id,
            "route_integrity_evidence_window_seconds": ROUTE_INTEGRITY_EVIDENCE_WINDOW_SECONDS,
            # CORR-44 (Phase 15B.5C): frozen before any official
            # replica_transition attempt -- see `replica_transition_route_
            # scored_window()`/`replica_transition_materiality_threshold_
            # seconds()` in dynamics_calibration_lib.py.
            "scored_window_seconds": list(scored_window),
            "scored_window_frozen": True,
            "scored_window_correction_reference": "CORR-44",
            "materiality_fraction": REPLICA_TRANSITION_MATERIALITY_FRACTION,
            "materiality_threshold_seconds": materiality_threshold_seconds,
            "all_zero_success_runs": all_zero_success_runs,
            "material_intervals": material_intervals,
            "evidence_windows": evidence_windows,
            "classification": route_integrity_result["classification"],
            "reason": route_integrity_result["reason"],
        }
        write_json_atomic(attempt_dir / "cluster" / "loadtest_route_integrity_timeline.json", route_integrity_timeline)

        if route_integrity_result["classification"] == "FAILED":
            invalidity_reasons.append(
                "wrong_traffic_path_used: official workload route confirmed to have stopped forwarding during "
                "traffic (tunnel/helper/listener failure in at least one evidence window of a material "
                "zero-success interval)"
            )
        elif route_integrity_result["classification"] == "UNVERIFIED":
            invalidity_reasons.append(
                "official_workload_route_integrity_unverified: at least one evidence window within a material "
                f"zero-success delivery interval ({route_integrity_result['material_intervals']}) lacks positive "
                "delivered-request evidence"
            )

    readiness_rows: List[Dict[str, Any]] = []
    readiness_path = attempt_dir / "cluster" / "pod_readiness_history.csv"
    if readiness_path.exists():
        readiness_rows = load_pod_readiness_history(readiness_path)
    restarted_during_run: List[str] = []
    if len(readiness_rows) >= 2:
        first_ts = min(r["elapsed_seconds"] for r in readiness_rows)
        last_ts = max(r["elapsed_seconds"] for r in readiness_rows)
        first_rows = [r for r in readiness_rows if r["elapsed_seconds"] == first_ts]
        last_rows = [r for r in readiness_rows if r["elapsed_seconds"] == last_ts]
        restarted_during_run = restarted_pods_between(first_rows, last_rows)

    # Phase 15B.5B sec.14/15: the three objectively-defined replica lags,
    # plus the four fields with no frozen settling definition, explicitly
    # marked NOT_FORMALLY_OPERATIONALIZED rather than invented.
    transition_metrics = (
        build_transition_metrics_summary(
            trigger_monotonic=_trigger_monotonic_box[0],
            timeline_samples=replica_timeline_samples,
            target_replicas=replicas_after,
            direction=direction,
        )
        if _trigger_monotonic_box[0] is not None
        else {"error": "transition was never triggered -- no lags computable"}
    )
    write_json_atomic(attempt_dir / "analysis" / "replica_transition_metrics.json", transition_metrics)

    execution_structural_validity = EXECUTION_INVALID if invalidity_reasons else EXECUTION_VALID

    attempt_metadata: Dict[str, Any] = {
        "calibration_protocol_version": manifest["protocol_version"],
        "campaign_sequence_index": slot["campaign_sequence_index"],
        "planned_slot_id": slot["planned_slot_id"],
        "execution_attempt_number": args.attempt_number,
        "attempt_id": attempt_id,
        "official": official,
        "wall_clock_start_utc": start_utc,
        "wall_clock_end_utc": end_utc,
        "run_kind": slot["run_kind"],
        "scientific_role": TRANSITION_SCIENTIFIC_ROLE,
        "cell_or_transition_id": slot["cell_or_transition_id"],
        "repeat_number": slot["repeat_number"],
        "replicas_before": replicas_before,
        "replicas_after": replicas_after,
        "transition_direction": direction,
        "offered_rps": slot.get("offered_rps"),
        "intensity": manifest["scope"]["application_intensity"],
        "schedule_path": slot["schedule_path"],
        "schedule_sha256": schedule_hash,
        "connection_mode": manifest["arrival_process"]["connection_mode"],
        "timeout_seconds": manifest["generator_capacity"]["default_timeout_seconds"],
        "max_in_flight": manifest["generator_capacity"]["frozen_calibration_max_in_flight"],
        "target_url": tunnel_url,
        "hpa_absence_verified": hpa_absent,
        "rl_absence_verified": True,
        "kind_state": "stopped",
        "app_resource_spec": {"cpu_request": "100m", "cpu_limit": "500m", "memory_request": "64Mi", "memory_limit": "256Mi"},
        "minikube_outer_resource_limits": pre_run_context,
        "prometheus_health": prom_health,
        "prometheus_health_diagnostic_error_type": prom_health_error_type,
        "prometheus_health_diagnostic_error_message": prom_health_error_message,
        "metrics_server_resolution": "15s",
        "protocol_version_applied": 2,
        "execution_structural_validity": execution_structural_validity,
        "invalidity_reasons": invalidity_reasons,
        "transition_timing_plan": timing,
        "transition_trigger": trigger_result,
        "transition_metrics": transition_metrics,
        "restarted_pods_during_run": restarted_during_run,
        "corr_reference": "CORR-37",
        "route_integrity": {
            "classification": route_integrity_result.get("classification"),
            "reason": route_integrity_result.get("reason"),
            "material_intervals": route_integrity_result.get("material_intervals", []),
            "full_temporal_coverage": route_integrity_result.get("full_temporal_coverage"),
            "confirmed_window_count": route_integrity_result.get("confirmed_window_count"),
            "unverified_window_count": route_integrity_result.get("unverified_window_count"),
            "failed_window_count": route_integrity_result.get("failed_window_count"),
        },
        "route_integrity_corr_reference": "CORR-39",
        "workload_transport": {
            "protocol_version": 5,
            "type": transport_lifecycle.get("transport_type", TRANSPORT_TYPE_DOCKER_NETWORK_TCP_RELAY),
            "ingress_service": LOADTEST_SVC,
            "kubernetes_access_mode": "NodePort",
            "client_location": "mac_host",
            "connection_mode": manifest["arrival_process"]["connection_mode"],
            "session_affinity": "None",
        },
    }
    attempt_metadata.update(build_protocol_provenance())
    attempt_metadata["schedule_sha256_verified_against"] = {"global_sequence": schedule_hash, "on_disk_recomputed": schedule_hash}
    write_json_atomic(attempt_dir / "metadata.json", attempt_metadata)

    if official:
        ledger = ensure_ledger(sequence)
        ledger_state = ledger_state_for_execution_validity(execution_structural_validity)
        if ledger_state == LEDGER_STATE_VALID:
            update_slot_state(ledger, args.campaign_sequence_index, ledger_state, attempt_id=attempt_id, mark_valid=True)
        else:
            update_slot_state(ledger, args.campaign_sequence_index, ledger_state, attempt_id=attempt_id)
        atomic_write_ledger(ledger)
    else:
        print("=== NON-OFFICIAL smoke validation: campaign ledger untouched (no read, no write) ===")

    print(json.dumps(attempt_metadata, indent=2, default=str))
    print(f"\nexecution_structural_validity: {execution_structural_validity}")
    if invalidity_reasons:
        print(f"Invalidity reasons: {invalidity_reasons}")
    if restarted_during_run:
        print(f"Pods restarted during run: {restarted_during_run}")
    return 0 if execution_structural_validity == EXECUTION_VALID else 1


def write_workload_transition_timeline_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=WORKLOAD_TRANSITION_TIMELINE_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


WORKLOAD_TRANSITION_TIMELINE_CSV_FIELDNAMES = [
    "elapsed_from_traffic_start", "elapsed_from_workload_step", "phase", "target_rps",
    "request_rate_rps", "latency_p50_seconds", "latency_p95_seconds", "latency_p99_seconds",
    "requests_in_progress", "cpu_cores", "memory_bytes",
    "replicas_desired", "replicas_current", "replicas_ready", "replicas_available",
]


def execute_workload_transition_slot(
    args: argparse.Namespace,
    manifest: dict,
    sequence: List[dict],
    slot: dict,
    schedule_hash: str,
    ledger: dict,
    official: bool = True,
) -> int:
    """CORR-47 (Phase 15B.6B): the workload-transition analogue of
    `execute_replica_transition_slot()`, reusing the identical V5
    transport / collector / CORR-46-hardened infrastructure. Implements
    the state machine required by the governing task's sec.11:

        PRECHECK -> ISOLATE_HPA -> SET_FIXED_REPLICA_COUNT ->
        WAIT_INITIAL_READY (60s no traffic) -> START_COLLECTORS ->
        START_V5_RELAY -> START EXACT SCHEDULE REPLAY -> (the schedule
        itself, not the harness, steps offered RPS at t=60s) ->
        STOP_TRAFFIC -> FINAL CAPTURE -> ANALYSIS -> LEDGER -> CLEANUP ->
        RESTORE_HPA -> RESTORE CANONICAL 1 REPLICA

    No replica-count change is intentionally commanded during the run
    (sec.9/11) -- the workload step is entirely schedule-owned (the
    schedule CSV's own `pre_step`/`post_step` phases). This function
    therefore has NO trigger-command equivalent of `trigger_replica_
    transition()`; it only records the PLANNED workload-step elapsed time
    and, descriptively, the schedule's own first post-step-phase request
    timestamp (never reinterpreted as a derived application-response lag
    -- sec.14/CORR-47 sec.5).

    `official=False` (NON-OFFICIAL smoke validation only): artifacts are
    written under `TRANSPORT_VALIDATION_ROOT/workload_transition_smoke/`,
    never under `CALIBRATION_ROOT`, and the official campaign ledger is
    never read for a state transition nor written to at all.
    """
    fixed_replicas = int(slot["replicas"])
    rps_before = float(slot["rps_before"])
    rps_after = float(slot["rps_after"])
    timing = resolve_workload_transition_timing_plan(manifest)

    if official:
        attempt_id = make_attempt_id(slot["planned_slot_id"], args.attempt_number)
        attempt_root = CALIBRATION_ROOT
    else:
        attempt_id = f"NONOFFICIAL_{slot['planned_slot_id']}_smoke_{utc_timestamp_id()}"
        attempt_root = TRANSPORT_VALIDATION_ROOT / "workload_transition_smoke"

    print(f"=== run_dynamics_calibration_slot [WORKLOAD_TRANSITION{'' if official else ' NON-OFFICIAL'}]: "
          f"index={args.campaign_sequence_index} attempt_id={attempt_id} "
          f"(replicas={fixed_replicas}, {rps_before}->{rps_after} RPS) (pending preflight) ===")

    print("=== PRECHECK (preflight) ===")
    checks = preflight_checks()
    for ok, msg in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {msg}")
    failed = [msg for ok, msg in checks if not ok]
    if failed:
        print("Preflight FAILED -- stopping before any mutation. No HPA change, no traffic. No attempt directory created.")
        return 2

    attempt_dir = attempt_root / attempt_id
    if attempt_dir.exists():
        raise SystemExit(f"Attempt directory already exists, refusing to overwrite: {attempt_dir}")
    (attempt_dir / "cluster").mkdir(parents=True)
    (attempt_dir / "analysis").mkdir(parents=True)

    pre_run_context = capture_minikube_resource_context()
    write_json_atomic(attempt_dir / "cluster" / "minikube_resource_context_pre_run.json", pre_run_context)

    cleanup = CleanupRegistry()
    invalidity_reasons: List[str] = []
    procs_to_stop: List[Tuple[str, subprocess.Popen]] = []
    transport_lifecycle: Dict[str, Any] = {}
    relay_container_name: Dict[str, Optional[str]] = {"name": None}

    def _stop_background_procs():
        for name, p in procs_to_stop:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()

    def _stop_relay_container():
        name = relay_container_name["name"]
        if not name:
            return
        running, exit_code = check_relay_container_state(name)
        already_dead_before_cleanup = running is False
        transport_lifecycle["already_dead_before_cleanup"] = already_dead_before_cleanup
        transport_lifecycle["exit_code_before_cleanup"] = exit_code if already_dead_before_cleanup else None
        logs_proc = _run(["docker", "logs", name], timeout=15)
        docker_logs_path = Path(f"/tmp/relay_{attempt_id}.log")
        docker_logs_path.write_text((logs_proc.stdout or "") + (logs_proc.stderr or ""))
        transport_lifecycle["docker_logs_path"] = str(docker_logs_path)
        _run(["docker", "rm", "-f", name], timeout=15)
        transport_lifecycle["exit_code_after_cleanup"] = exit_code
        transport_lifecycle["cleanup_action"] = classify_tunnel_termination_action(bool(already_dead_before_cleanup))

    def _restore_hpa_and_idle():
        restore_hpa()
        time.sleep(2)
        wait_timeout = restoration_wait_seconds(initial_replica_count_for_restoration(slot))
        state = wait_for_idle_one_replica(timeout_seconds=wait_timeout)
        converged = all(v == 1 for v in state.values())
        if converged:
            print(f"  Restored HPA; idle state converged within {wait_timeout:.0f}s wait: {state}")
        else:
            print(
                f"  Restored HPA (confirmed applied), but natural scale-down convergence to 1/1/1/1 "
                f"did NOT complete within the {wait_timeout:.0f}s wait. Last observed state: {state}. "
                f"HPA itself IS restored and will continue converging on its own. No new official run "
                f"should start until 1/1/1/1 is independently confirmed."
            )

    cleanup.register("stop_background_processes", _stop_background_procs)
    cleanup.register("stop_relay_container", _stop_relay_container)
    cleanup.register("restore_hpa_and_wait_idle", _restore_hpa_and_idle)
    cleanup.install_signal_and_exit_handlers()

    start_utc = datetime.now(timezone.utc).isoformat()
    tunnel_url: Optional[str] = None
    hpa_absent: Optional[bool] = None
    traffic_start_monotonic: Optional[float] = None
    traffic_start_epoch: Optional[float] = None
    traffic_end_epoch: Optional[float] = None
    dataplane_periodic_samples: List[Dict[str, Any]] = []
    dataplane_reactive_resample_count = 0
    dataplane_anchor_sample_taken = False
    dataplane_acquisition_gap_exceeded_count = 0
    tunnel_periodic_samples: List[Dict[str, Any]] = []

    try:
        print("=== ISOLATE_HPA ===")
        delete_hpa()
        time.sleep(2)
        hpa_absent = verify_no_hpa_targets_deployment()
        if not hpa_absent:
            invalidity_reasons.append("hpa_unexpectedly_active: HPA still present after delete")

        print(f"=== SET_FIXED_REPLICA_COUNT ({fixed_replicas}) ===")
        replica_state = scale_and_wait(fixed_replicas)
        replicas_ok = all(v == fixed_replicas for v in replica_state.values())
        if not replicas_ok:
            invalidity_reasons.append(f"wrong_initial_replica_count_achieved: {replica_state}")
        pod_names_pre = sorted(get_pod_names())
        write_json_atomic(attempt_dir / "cluster" / "pod_identities_pre_stabilization.json", {"pod_names": pod_names_pre})

        print(f"=== WAIT_FOR_INITIAL_READY/STABILIZED ({timing['pre_run_replica_stabilization_seconds']:.0f}s, no traffic) ===")
        time.sleep(timing["pre_run_replica_stabilization_seconds"])

        events_pre = fetch_events_json()
        write_json_atomic(attempt_dir / "cluster" / "kubernetes_events_pre.json", events_pre)

        loadtest_endpoints_pre = fetch_loadtest_endpoints_json()
        write_json_atomic(attempt_dir / "cluster" / "loadtest_endpoints_pre_traffic.json", loadtest_endpoints_pre)

        print("=== START_V5_RELAY ===")
        transport_result = start_loadtest_transport(attempt_id)
        tunnel_url = transport_result.get("target_url")
        if "error" in transport_result:
            invalidity_reasons.append(f"wrong_traffic_path_used: failed to establish V5 relay transport before traffic: {transport_result['error']}")
            transport_lifecycle.update(transport_result)
        else:
            relay_container_name["name"] = transport_result["container_name"]
            transport_lifecycle.update(transport_result)
            if not tunnel_url:
                invalidity_reasons.append("wrong_traffic_path_used: V5 relay launched but its host port could not be resolved")

        prom_proc, started_prom = start_prometheus_port_forward()
        if started_prom and prom_proc is not None:
            procs_to_stop.append(("prometheus_port_forward", prom_proc))

        gen_cap = manifest["generator_capacity"]
        collector_duration = int(timing["total_traffic_seconds"]) + 30

        print("=== START_COLLECTORS ===")
        metrics_proc = subprocess.Popen(
            [VENV_PYTHON, "-m", "collectors.metrics_collector",
             "--experiment-id", attempt_id, "--mode", "manual",
             "--traffic-profile", slot["cell_or_transition_id"],
             "--duration-seconds", str(collector_duration), "--seed", "42"],
        )
        procs_to_stop.append(("metrics_collector", metrics_proc))

        readiness_proc = subprocess.Popen(
            [VENV_PYTHON, "-m", "scripts.pod_readiness_collector",
             "--experiment-dir", str(attempt_dir / "cluster"),
             "--duration-seconds", str(collector_duration)],
        )
        procs_to_stop.append(("pod_readiness_collector", readiness_proc))

        time.sleep(1)

        schedule_full_path = PROJECT_ROOT / slot["schedule_path"]
        profile_path = schedule_full_path.parent / f"{slot['cell_or_transition_id']}.yaml"
        if not profile_path.exists():
            invalidity_reasons.append(f"wrong_or_unhashed_schedule: traffic profile YAML not found: {profile_path}")

        tunnel_port = _extract_port_from_url(tunnel_url)
        relay_name_for_sampling = relay_container_name["name"]
        _stop_periodic_sampling = threading.Event()

        # Route-integrity/tunnel sampler and dataplane sampler are reused
        # VERBATIM from `execute_replica_transition_slot()` -- neither
        # depends on replica-count-transition semantics, and both are
        # already CORR-46-hardened (sec.19 of the governing task).
        def _sample_route_integrity_periodically(sample_start_mono: float) -> None:
            while not _stop_periodic_sampling.wait(10):
                try:
                    endpoints = fetch_loadtest_endpoints_json()
                    container_running, container_exit_code = (
                        check_relay_container_state(relay_name_for_sampling) if relay_name_for_sampling else (None, None)
                    )
                    listener_present = check_local_listener_present(tunnel_port) if tunnel_port else None
                except subprocess.TimeoutExpired as exc:
                    tunnel_periodic_samples.append(
                        {
                            "elapsed_seconds_since_traffic_start": round(time.monotonic() - sample_start_mono, 1),
                            "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                            "tunnel_process_alive": None,
                            "tunnel_process_exit_code": None,
                            "local_listener_present": None,
                            "listener_port": tunnel_port,
                            "relay_container_running": None,
                            "relay_container_exit_code": None,
                            "host_listener_present": None,
                            "loadtest_endpoints": None,
                            "endpoint_query_error": str(exc),
                        }
                    )
                    continue
                tunnel_periodic_samples.append(
                    {
                        "elapsed_seconds_since_traffic_start": round(time.monotonic() - sample_start_mono, 1),
                        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                        "tunnel_process_alive": container_running,
                        "tunnel_process_exit_code": container_exit_code,
                        "local_listener_present": listener_present,
                        "listener_port": tunnel_port,
                        "relay_container_running": container_running,
                        "relay_container_exit_code": container_exit_code,
                        "host_listener_present": listener_present,
                        "loadtest_endpoints": parse_endpoints_ready_count(endpoints) if "_error" not in endpoints else None,
                        "endpoint_query_error": endpoints.get("_error"),
                    }
                )

        dataplane_nodeport = transport_lifecycle.get("nodeport")
        _stop_dataplane_sampling = threading.Event()
        _dataplane_samples_lock = threading.Lock()
        _last_dataplane_sample_taken_mono: List[Optional[float]] = [None]

        def _take_and_record_dataplane_sample(sample_start_mono: float) -> Dict[str, Any]:
            nonlocal dataplane_acquisition_gap_exceeded_count
            t_start = time.monotonic()
            sample = sample_dataplane_ingress_rules(dataplane_nodeport)
            with _dataplane_samples_lock:
                previous_taken_mono = _last_dataplane_sample_taken_mono[0]
                gap = None if previous_taken_mono is None else round(t_start - previous_taken_mono, 4)
                _last_dataplane_sample_taken_mono[0] = t_start
                gap_exceeded = gap is not None and gap > DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8
                if gap_exceeded:
                    dataplane_acquisition_gap_exceeded_count += 1
                sample["elapsed_seconds_since_traffic_start"] = round(time.monotonic() - sample_start_mono, 2)
                sample["sampled_at_utc"] = datetime.now(timezone.utc).isoformat()
                sample["dataplane_mode"] = "kube_proxy_iptables_nf_tables"
                sample["service_namespace"] = APP_NS
                sample["service_name"] = LOADTEST_SVC
                sample["actual_gap_seconds_from_previous_sample"] = gap
                sample["acquisition_gap_exceeded_assumed_bound"] = gap_exceeded
                dataplane_periodic_samples.append(sample)
            return sample

        def _sample_dataplane_ingress_periodically(sample_start_mono: float) -> None:
            nonlocal dataplane_reactive_resample_count
            previous_sample: Optional[Dict[str, Any]] = None
            while not _stop_dataplane_sampling.wait(DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7):
                sample = _take_and_record_dataplane_sample(sample_start_mono)
                if should_trigger_immediate_resample(previous_sample, sample):
                    if _stop_dataplane_sampling.wait(DATAPLANE_INGRESS_TRANSITION_RESAMPLE_DELAY_SECONDS):
                        previous_sample = sample
                        break
                    dataplane_reactive_resample_count += 1
                    sample = _take_and_record_dataplane_sample(sample_start_mono)
                previous_sample = sample

        print("=== START_TRAFFIC (background; schedule owns the workload step, no harness trigger command) ===")
        traffic_start_epoch = datetime.now(timezone.utc).timestamp()
        traffic_start_monotonic = time.monotonic()

        sampler_thread = threading.Thread(target=_sample_route_integrity_periodically, args=(traffic_start_monotonic,), daemon=True)
        sampler_thread.start()
        dataplane_sampler_thread = threading.Thread(target=_sample_dataplane_ingress_periodically, args=(traffic_start_monotonic,), daemon=True)
        dataplane_sampler_thread.start()

        gen_proc = subprocess.Popen(
            [VENV_PYTHON, "-m", "loadgen.traffic_generator",
             "--profile", str(profile_path),
             "--experiment-id", attempt_id,
             "--target-url", tunnel_url or "http://localhost:1",
             "--replay-schedule", str(schedule_full_path),
             "--arrival-process", "deterministic",
             "--connection-mode", manifest["arrival_process"]["connection_mode"],
             "--timeout", str(gen_cap["default_timeout_seconds"]),
             "--max-in-flight", str(gen_cap["frozen_calibration_max_in_flight"])],
        )
        procs_to_stop.append(("traffic_generator", gen_proc))

        print(f"=== WORKLOAD STEP (schedule-owned; planned at traffic_elapsed={timing['trigger_at_traffic_elapsed_seconds']:.0f}s) ===")
        print(f"=== POST-STEP OBSERVATION (traffic continues to t={timing['total_traffic_seconds']:.0f}s) ===")
        gen_exit = gen_proc.wait(timeout=timing["total_traffic_seconds"] + 60)
        traffic_end_epoch = datetime.now(timezone.utc).timestamp()
        transport_lifecycle["traffic_start_epoch"] = traffic_start_epoch
        transport_lifecycle["traffic_end_epoch"] = traffic_end_epoch
        procs_to_stop = [(n, p) for n, p in procs_to_stop if p is not gen_proc]
        if gen_exit != 0:
            invalidity_reasons.append(f"collector_or_generator_structural_failure: traffic_generator exit code {gen_exit}")

        _take_and_record_dataplane_sample(traffic_start_monotonic)
        dataplane_anchor_sample_taken = True

        print("=== STOP_TRAFFIC / FINAL CAPTURE ===")
        _stop_periodic_sampling.set()
        sampler_thread.join(timeout=5)
        transport_lifecycle["periodic_samples_during_traffic"] = tunnel_periodic_samples

        _stop_dataplane_sampling.set()
        dataplane_sampler_thread.join(timeout=5)

        dataplane_ingress_timeline = {
            "attempt_id": attempt_id,
            "dataplane_ingress_sampling_interval_seconds": DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7,
            "dataplane_ingress_sampling_protocol_version": 7,
            "dataplane_ingress_transition_resample_delay_seconds": DATAPLANE_INGRESS_TRANSITION_RESAMPLE_DELAY_SECONDS,
            "dataplane_reactive_resample_count": dataplane_reactive_resample_count,
            "dataplane_anchor_sample_taken": dataplane_anchor_sample_taken,
            "dataplane_acquisition_worst_case_cadence_assumed_seconds": DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8,
            "dataplane_acquisition_gap_exceeded_count": dataplane_acquisition_gap_exceeded_count,
            "nodeport": dataplane_nodeport,
            "periodic_samples": sorted(dataplane_periodic_samples, key=lambda s: s["elapsed_seconds_since_traffic_start"]),
        }
        write_json_atomic(attempt_dir / "cluster" / "loadtest_dataplane_ingress_timeline.json", dataplane_ingress_timeline)

        container_running_at_end, container_exit_code_at_end = (
            check_relay_container_state(relay_container_name["name"]) if relay_container_name["name"] else (None, None)
        )
        transport_lifecycle["running_at_traffic_end"] = container_running_at_end
        transport_lifecycle["alive_at_traffic_end"] = container_running_at_end
        transport_lifecycle["host_listener_present_at_traffic_end"] = check_local_listener_present(tunnel_port) if tunnel_port else None
        transport_lifecycle["local_listener_present_at_traffic_end"] = transport_lifecycle["host_listener_present_at_traffic_end"]
        if container_running_at_end is False:
            transport_lifecycle["exit_code_at_traffic_end"] = container_exit_code_at_end
        transport_lifecycle["listener_owner_at_traffic_end"] = (
            resolve_local_listener_owner(tunnel_port) if tunnel_port else
            {"listener_owner_pid": None, "listener_owner_command": None}
        )

        loadtest_endpoints_post = fetch_loadtest_endpoints_json()
        write_json_atomic(attempt_dir / "cluster" / "loadtest_endpoints_post_traffic.json", loadtest_endpoints_post)

        pod_names_post = sorted(get_pod_names())
        write_json_atomic(attempt_dir / "cluster" / "pod_identities_post_traffic.json", {"pod_names": pod_names_post})
        # Unlike replica_transition, a workload-transition slot holds
        # replicas FIXED by design -- an unexpected pod-identity change
        # here is evaluated honestly (fixed-replica invariant, below) but
        # is NOT automatically treated as invalidating, per the governing
        # task's own sec.12 instruction not to invent a new rule.

        events_post = fetch_events_json()
        write_json_atomic(attempt_dir / "cluster" / "kubernetes_events_post.json", events_post)
        normalized_events = normalize_kubernetes_events(events_post)
        with open(attempt_dir / "cluster" / "kubernetes_events_normalized.csv", "w", newline="", encoding="utf-8") as ef:
            ew = csv.DictWriter(ef, fieldnames=KUBERNETES_EVENTS_NORMALIZED_CSV_FIELDNAMES)
            ew.writeheader()
            for row in normalized_events:
                ew.writerow(row)

        pods_snapshot_json = safe_fetch_current_pod_snapshot_json()
        current_snapshot = parse_pod_readiness_snapshot(
            pods_snapshot_json,
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).timestamp(),
            -1.0,
        )
        prev_logs_dir = attempt_dir / "cluster" / "previous_container_logs"
        prev_logs_dir.mkdir(parents=True, exist_ok=True)
        for row in current_snapshot:
            if int(row.get("restart_count", 0)) > 0:
                log_text, available = fetch_previous_container_logs(row["pod_name"])
                (prev_logs_dir / f"{row['pod_name']}.log").write_text(log_text)
                write_json_atomic(prev_logs_dir / f"{row['pod_name']}_availability.json", {"pod_name": row["pod_name"], "previous_log_available": available})

        print("=== Waiting for collectors to finish ===")
        metrics_exit = metrics_proc.wait(timeout=collector_duration + 60)
        readiness_exit = readiness_proc.wait(timeout=collector_duration + 60)
        procs_to_stop = [(n, p) for n, p in procs_to_stop if p not in (metrics_proc, readiness_proc)]
        if metrics_exit != 0:
            invalidity_reasons.append(f"collector_or_generator_structural_failure: metrics_collector exit code {metrics_exit}")
        if readiness_exit != 0:
            invalidity_reasons.append("collector_or_generator_structural_failure: pod readiness collector non-zero exit")

        readiness_metadata_path = attempt_dir / "cluster" / "pod_readiness_metadata.json"
        if readiness_metadata_path.exists():
            readiness_metadata = json.loads(readiness_metadata_path.read_text())
            contamination_ok, contamination_msg = check_no_external_contamination_during_run(readiness_metadata)
            if not contamination_ok:
                invalidity_reasons.append(contamination_msg)

        post_run_context = capture_minikube_resource_context()
        write_json_atomic(attempt_dir / "cluster" / "minikube_resource_context_post_run.json", post_run_context)

        prom_health_diagnostic = check_prometheus_diagnostic_health()
        prom_health = prom_health_diagnostic["healthy"]
        prom_health_error_type = prom_health_diagnostic["error_type"]
        prom_health_error_message = prom_health_diagnostic["error_message"]

    finally:
        cleanup.run_all()

    end_utc = datetime.now(timezone.utc).isoformat()

    if transport_lifecycle:
        docker_logs_path = Path(transport_lifecycle.get("docker_logs_path", ""))
        transport_lifecycle["captured_log_text"] = docker_logs_path.read_text() if docker_logs_path.exists() else None
        write_json_atomic(attempt_dir / "cluster" / "loadtest_transport_lifecycle.json", transport_lifecycle)

    # -------------------------------------------------------------------
    # Post-processing: reuses the IDENTICAL pure route-integrity/
    # materiality/V8-partitioning functions the replica-transition and
    # steady-state flows use (CORR-47 sec.15-18) -- Axis A structural
    # validity for a workload-transition slot is evaluated by the SAME
    # frozen rules, using the workload-transition-specific scored-window/
    # materiality-threshold values (CORR-47, not silently inherited from
    # CORR-44).
    # -------------------------------------------------------------------
    traffic_dir = PROJECT_ROOT / "persistent-data/traffic" / attempt_id
    metrics_dir = PROJECT_ROOT / "persistent-data/metrics" / attempt_id
    try:
        (attempt_dir / "traffic").symlink_to(traffic_dir)
    except FileExistsError:
        pass
    try:
        (attempt_dir / "metrics").symlink_to(metrics_dir)
    except FileExistsError:
        pass

    traffic_metadata = {}
    traffic_meta_path = traffic_dir / "metadata.json"
    if traffic_meta_path.exists():
        traffic_metadata = json.loads(traffic_meta_path.read_text())
        ok, msg = check_offered_equals_sent(traffic_metadata)
        if not ok:
            invalidity_reasons.append(f"offered_not_equal_sent_due_to_undersized_generator_max_in_flight: {msg}")
        ok, msg = check_no_dropped_overload(traffic_metadata)
        if not ok:
            invalidity_reasons.append(f"generator_dropped_overload_due_to_harness_capacity_not_application_response: {msg}")
    else:
        invalidity_reasons.append("collector_or_generator_structural_failure: traffic metadata.json missing")

    requests_rows: List[Dict[str, Any]] = []
    requests_csv_path = traffic_dir / "requests.csv"
    if requests_csv_path.exists():
        requests_rows = load_requests_csv(requests_csv_path)

    metrics_rows: List[Dict[str, Any]] = []
    metrics_csv_path = metrics_dir / "metrics.csv"
    if metrics_csv_path.exists():
        metrics_rows = load_metrics_csv(metrics_csv_path)

    # CORR-47 sec.13, boundary-hardened by CORR-48: the workload-transition
    # trajectory artifact is ASSEMBLED from already-collected raw streams
    # (metrics.csv, already produced by the same metrics_collector
    # invocation every run kind uses) rather than a new expensive live
    # poller, restricted to the official [0, total_traffic_seconds)
    # traffic window -- the collector's own post-traffic observation
    # buffer (total_traffic_seconds+30s) is excluded from the SCIENTIFIC
    # trajectory, though it remains fully preserved, untouched, in the
    # raw metrics.csv file itself.
    workload_trajectory_rows = build_workload_transition_trajectory_rows(
        metrics_rows=metrics_rows,
        trigger_at_traffic_elapsed_seconds=timing["trigger_at_traffic_elapsed_seconds"],
        rps_before=rps_before,
        rps_after=rps_after,
        total_traffic_seconds=timing["total_traffic_seconds"],
    )
    write_workload_transition_timeline_csv(attempt_dir / "cluster" / "workload_transition_timeline.csv", workload_trajectory_rows)

    fixed_replica_invariant = evaluate_fixed_replica_invariant(workload_trajectory_rows, fixed_replicas)

    workload_step = {
        "workload_step_owner": "schedule",
        "planned_workload_step_elapsed_seconds": timing["trigger_at_traffic_elapsed_seconds"],
        "first_post_step_request_elapsed_seconds": first_post_step_request_elapsed_seconds(requests_rows),
    }

    route_integrity_result: Dict[str, Any] = {"classification": None}
    traffic_summary_path = traffic_dir / "traffic_summary.csv"
    dataplane_samples_for_windows = dataplane_periodic_samples
    if traffic_summary_path.exists():
        summary_rows = load_traffic_summary_csv(traffic_summary_path)
        traffic_duration_seconds = timing["total_traffic_seconds"]
        scored_window = workload_transition_route_scored_window(traffic_duration_seconds)
        materiality_threshold_seconds = workload_transition_materiality_threshold_seconds(traffic_duration_seconds)

        all_zero_success_runs = find_all_zero_success_runs(summary_rows)
        material_intervals = [
            run for run in all_zero_success_runs
            if is_interval_material(
                run[0], run[1], traffic_duration_seconds, scored_window,
                materiality_fraction_threshold=WORKLOAD_TRANSITION_MATERIALITY_FRACTION,
            )
        ]

        periodic_samples = transport_lifecycle.get("periodic_samples_during_traffic", [])

        evidence_windows: List[Dict[str, Any]] = []
        for run_start, run_end in material_intervals:
            for window_start, window_end in partition_into_evidence_windows(
                run_start, run_end, min_window_seconds=MIN_EVIDENCE_WINDOW_SECONDS_V8
            ):
                observer_evidence = aggregate_observer_evidence_for_window(window_start, window_end, periodic_samples)
                delivered_evidence = compute_delivered_request_evidence_windows(
                    window_start_elapsed=window_start,
                    window_end_elapsed=window_end,
                    traffic_start_epoch=traffic_start_epoch,
                    requests_rows=requests_rows,
                    metrics_rows=metrics_rows,
                    counter_rows=[],
                )
                dataplane_evidence = compute_dataplane_ingress_evidence_windows(
                    window_start_elapsed=window_start,
                    window_end_elapsed=window_end,
                    traffic_start_epoch=traffic_start_epoch,
                    dataplane_samples=dataplane_samples_for_windows,
                )
                window_verdict = classify_route_integrity_window(
                    delivered_request_evidence=delivered_evidence["delivered_request_evidence"],
                    tunnel_process_alive=observer_evidence["tunnel_process_alive"],
                    local_listener_present=observer_evidence["local_listener_present"],
                    nodeport_dataplane_ingress_evidence=dataplane_evidence["nodeport_dataplane_ingress_evidence"],
                )
                route_evidence_sources = list(delivered_evidence["delivery_evidence_sources"])
                if dataplane_evidence["nodeport_dataplane_ingress_evidence"]:
                    route_evidence_sources.append("nodeport_dataplane_ingress")
                evidence_windows.append(
                    {
                        "material_interval": [run_start, run_end],
                        "window_start_elapsed": window_start,
                        "window_end_elapsed": window_end,
                        "tunnel_process_alive": observer_evidence["tunnel_process_alive"],
                        "local_listener_present": observer_evidence["local_listener_present"],
                        "relay_container_running": observer_evidence["tunnel_process_alive"],
                        "host_listener_present": observer_evidence["local_listener_present"],
                        "loadtest_ready_endpoint_count": observer_evidence["loadtest_ready_endpoint_count"],
                        "endpoint_query_error": observer_evidence["endpoint_query_error"],
                        "observer_sample_count": observer_evidence["sample_count"],
                        "http_response_delivery_evidence": delivered_evidence["http_response_delivery_evidence"],
                        "counter_delivery_evidence": delivered_evidence["counter_delivery_evidence"],
                        "rolling_server_rps_observed": delivered_evidence["rolling_server_rps_observed"],
                        "rolling_server_rps_supporting_only": delivered_evidence["rolling_server_rps_supporting_only"],
                        "nodeport_dataplane_ingress_evidence": dataplane_evidence["nodeport_dataplane_ingress_evidence"],
                        "dataplane_counter_intervals_used": dataplane_evidence["dataplane_counter_intervals_used"],
                        "dataplane_counter_intervals_rejected_cross_window": dataplane_evidence["dataplane_counter_intervals_rejected_cross_window"],
                        "dataplane_counter_intervals_rejected_reset": dataplane_evidence["dataplane_counter_intervals_rejected_reset"],
                        "dataplane_observer_error_present": dataplane_evidence["dataplane_observer_error_present"],
                        "dataplane_backend_regimes_with_evidence": dataplane_evidence["dataplane_backend_regimes_with_evidence"],
                        "delivered_request_evidence": delivered_evidence["delivered_request_evidence"],
                        "delivery_evidence_sources": delivered_evidence["delivery_evidence_sources"],
                        "route_evidence_sources": route_evidence_sources,
                        "window_route_status": window_verdict["window_route_status"],
                        "reason": window_verdict["reason"],
                    }
                )

        route_integrity_result = classify_route_integrity(material_intervals=material_intervals, evidence_windows=evidence_windows)
        route_integrity_timeline = {
            "attempt_id": attempt_id,
            "route_integrity_evidence_window_seconds": ROUTE_INTEGRITY_EVIDENCE_WINDOW_SECONDS,
            "scored_window_seconds": list(scored_window),
            "scored_window_frozen": True,
            "scored_window_correction_reference": "CORR-47",
            "materiality_fraction": WORKLOAD_TRANSITION_MATERIALITY_FRACTION,
            "materiality_threshold_seconds": materiality_threshold_seconds,
            "all_zero_success_runs": all_zero_success_runs,
            "material_intervals": material_intervals,
            "evidence_windows": evidence_windows,
            "classification": route_integrity_result["classification"],
            "reason": route_integrity_result["reason"],
        }
        write_json_atomic(attempt_dir / "cluster" / "loadtest_route_integrity_timeline.json", route_integrity_timeline)

        if route_integrity_result["classification"] == "FAILED":
            invalidity_reasons.append(
                "wrong_traffic_path_used: official workload route confirmed to have stopped forwarding during "
                "traffic (tunnel/helper/listener failure in at least one evidence window of a material "
                "zero-success interval)"
            )
        elif route_integrity_result["classification"] == "UNVERIFIED":
            invalidity_reasons.append(
                "official_workload_route_integrity_unverified: at least one evidence window within a material "
                f"zero-success delivery interval ({route_integrity_result['material_intervals']}) lacks positive "
                "delivered-request evidence"
            )

    readiness_rows: List[Dict[str, Any]] = []
    readiness_path = attempt_dir / "cluster" / "pod_readiness_history.csv"
    if readiness_path.exists():
        readiness_rows = load_pod_readiness_history(readiness_path)
    restarted_during_run: List[str] = []
    if len(readiness_rows) >= 2:
        first_ts = min(r["elapsed_seconds"] for r in readiness_rows)
        last_ts = max(r["elapsed_seconds"] for r in readiness_rows)
        first_rows = [r for r in readiness_rows if r["elapsed_seconds"] == first_ts]
        last_rows = [r for r in readiness_rows if r["elapsed_seconds"] == last_ts]
        restarted_during_run = restarted_pods_between(first_rows, last_rows)

    execution_structural_validity = EXECUTION_INVALID if invalidity_reasons else EXECUTION_VALID

    attempt_metadata: Dict[str, Any] = {
        "calibration_protocol_version": manifest["protocol_version"],
        "campaign_sequence_index": slot["campaign_sequence_index"],
        "planned_slot_id": slot["planned_slot_id"],
        "execution_attempt_number": args.attempt_number,
        "attempt_id": attempt_id,
        "official": official,
        "wall_clock_start_utc": start_utc,
        "wall_clock_end_utc": end_utc,
        "run_kind": slot["run_kind"],
        "scientific_role": WORKLOAD_TRANSITION_SCIENTIFIC_ROLE,
        "cell_or_transition_id": slot["cell_or_transition_id"],
        "repeat_number": slot["repeat_number"],
        "replicas": fixed_replicas,
        "rps_before": rps_before,
        "rps_after": rps_after,
        "intensity": manifest["scope"]["application_intensity"],
        "schedule_path": slot["schedule_path"],
        "schedule_sha256": schedule_hash,
        "connection_mode": manifest["arrival_process"]["connection_mode"],
        "timeout_seconds": manifest["generator_capacity"]["default_timeout_seconds"],
        "max_in_flight": manifest["generator_capacity"]["frozen_calibration_max_in_flight"],
        "target_url": tunnel_url,
        "hpa_absence_verified": hpa_absent,
        "rl_absence_verified": True,
        "kind_state": "stopped",
        "app_resource_spec": {"cpu_request": "100m", "cpu_limit": "500m", "memory_request": "64Mi", "memory_limit": "256Mi"},
        "minikube_outer_resource_limits": pre_run_context,
        "prometheus_health": prom_health,
        "prometheus_health_diagnostic_error_type": prom_health_error_type,
        "prometheus_health_diagnostic_error_message": prom_health_error_message,
        "metrics_server_resolution": "15s",
        "protocol_version_applied": 2,
        "execution_structural_validity": execution_structural_validity,
        "invalidity_reasons": invalidity_reasons,
        "workload_transition_timing_plan": timing,
        "workload_step": workload_step,
        "fixed_replica_invariant": fixed_replica_invariant,
        "derived_metrics_status": WORKLOAD_TRANSITION_DERIVED_METRICS_STATUS,
        "derived_metrics_status_reason": WORKLOAD_TRANSITION_DERIVED_METRICS_STATUS_REASON,
        "restarted_pods_during_run": restarted_during_run,
        "corr_reference": "CORR-37",
        "route_integrity": {
            "classification": route_integrity_result.get("classification"),
            "reason": route_integrity_result.get("reason"),
            "material_intervals": route_integrity_result.get("material_intervals", []),
            "full_temporal_coverage": route_integrity_result.get("full_temporal_coverage"),
            "confirmed_window_count": route_integrity_result.get("confirmed_window_count"),
            "unverified_window_count": route_integrity_result.get("unverified_window_count"),
            "failed_window_count": route_integrity_result.get("failed_window_count"),
        },
        "route_integrity_corr_reference": "CORR-47",
        "workload_transport": {
            "protocol_version": 5,
            "type": transport_lifecycle.get("transport_type", TRANSPORT_TYPE_DOCKER_NETWORK_TCP_RELAY),
            "ingress_service": LOADTEST_SVC,
            "kubernetes_access_mode": "NodePort",
            "client_location": "mac_host",
            "connection_mode": manifest["arrival_process"]["connection_mode"],
            "session_affinity": "None",
        },
    }
    attempt_metadata.update(build_protocol_provenance())
    attempt_metadata["schedule_sha256_verified_against"] = {"global_sequence": schedule_hash, "on_disk_recomputed": schedule_hash}
    write_json_atomic(attempt_dir / "metadata.json", attempt_metadata)

    if official:
        ledger = ensure_ledger(sequence)
        ledger_state = ledger_state_for_execution_validity(execution_structural_validity)
        if ledger_state == LEDGER_STATE_VALID:
            update_slot_state(ledger, args.campaign_sequence_index, ledger_state, attempt_id=attempt_id, mark_valid=True)
        else:
            update_slot_state(ledger, args.campaign_sequence_index, ledger_state, attempt_id=attempt_id)
        atomic_write_ledger(ledger)
    else:
        print("=== NON-OFFICIAL smoke validation: campaign ledger untouched (no read, no write) ===")

    print(json.dumps(attempt_metadata, indent=2, default=str))
    print(f"\nexecution_structural_validity: {execution_structural_validity}")
    if invalidity_reasons:
        print(f"Invalidity reasons: {invalidity_reasons}")
    if restarted_during_run:
        print(f"Pods restarted during run: {restarted_during_run}")
    return 0 if execution_structural_validity == EXECUTION_VALID else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    argv: Optional[List[str]] = None,
    sequence_override: Optional[List[dict]] = None,
    schedule_registry_override: Optional[dict] = None,
    calibration_root: Path = CALIBRATION_ROOT,
    ledger_path: Path = LEDGER_PATH,
) -> int:
    """Phase 16I.1: `sequence_override`, `schedule_registry_override`,
    `calibration_root`, and `ledger_path` are new, CODE-LEVEL-ONLY keyword
    parameters (never exposed as CLI flags -- `build_arg_parser()` is
    completely unchanged) that let a trusted, in-repository caller (e.g.
    `scripts/run_iteration2_dynamics_calibration_slot.py`) reuse this exact,
    unmodified execution/collection/V8-validity/route-integrity/HPA-isolation
    pipeline against a separately-frozen campaign context, without ever
    touching the historical `global_campaign_sequence.json`,
    `schedule_registry.json`, `persistent-data/dynamics_calibration/`, or its
    ledger. Every default reproduces the exact historical CLI behavior
    byte-for-byte: an invocation with no new kwargs is unchanged."""
    args = build_arg_parser().parse_args(argv)

    manifest = load_manifest()
    sequence = sequence_override if sequence_override is not None else load_global_sequence()
    slot = resolve_planned_slot(args.campaign_sequence_index, sequence=sequence)
    schedule_hash = verify_schedule_hash(slot, registry=schedule_registry_override)

    ledger = ensure_ledger(sequence, path=ledger_path)

    if args.dry_run:
        # Phase 15B.3N.3 (CORR-40): read-only structural resolution of the
        # V5 transport's inputs -- no mutation (no container created, no
        # HPA deleted, no Deployment scaled, no traffic sent).
        transport_discovery = {
            "docker_network": resolve_minikube_docker_network(),
            "node_ip": resolve_minikube_node_ip(),
            "nodeport": resolve_loadtest_nodeport(),
            "relay_image_identity": resolve_relay_image_identity(),
        }
        report = dry_run_report(slot, schedule_hash, manifest, ledger, transport_discovery, calibration_root=calibration_root)
        print(json.dumps(report, indent=2))
        return 0

    # Phase 15B.5B: replica-transition slots use a dedicated execution path
    # (see `execute_replica_transition_slot`'s own docstring) -- the
    # steady-state body below is completely unmodified and unreachable for
    # `run_kind=replica_transition`, so this branch introduces zero
    # regression risk to the 29 official steady-state attempts already
    # executed through it.
    if is_replica_transition_slot(slot):
        return execute_replica_transition_slot(args, manifest, sequence, slot, schedule_hash, ledger, official=True)

    # CORR-47 (Phase 15B.6B): workload-transition slots use their own
    # dedicated execution path (see `execute_workload_transition_slot`'s
    # own docstring) -- the steady-state body below remains completely
    # unmodified and unreachable for `run_kind=workload_transition`.
    if is_workload_transition_slot(slot):
        return execute_workload_transition_slot(args, manifest, sequence, slot, schedule_hash, ledger, official=True)

    session_id = args.session_id or f"dyncal_session_{utc_timestamp_id()}"
    attempt_id = make_attempt_id(slot["planned_slot_id"], args.attempt_number)

    print(f"=== run_dynamics_calibration_slot: index={args.campaign_sequence_index} attempt_id={attempt_id} (pending preflight) ===")

    # Phase 15B.3E: preflight is READ-ONLY and must run BEFORE any official
    # scientific attempt directory is created -- a prior invocation crashed
    # partway through a read-only preflight subprocess call (an unrelated
    # tooling timeout, not a scientific finding) yet had already created a
    # full attempt directory tree, leaving an artifact that looked like a
    # scientific attempt but contained zero real data. Preflight failure
    # (or crash) must never register as, or resemble, an attempt.
    print("=== Preflight ===")
    checks = preflight_checks()
    for ok, msg in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {msg}")
    failed = [msg for ok, msg in checks if not ok]
    if failed:
        print("Preflight FAILED -- stopping before any mutation. No HPA change, no traffic. No attempt directory created.")
        return 2

    attempt_dir = calibration_root / attempt_id
    if attempt_dir.exists():
        raise SystemExit(f"Attempt directory already exists, refusing to overwrite: {attempt_dir}")
    (attempt_dir / "cluster").mkdir(parents=True)
    (attempt_dir / "routing").mkdir(parents=True)
    (attempt_dir / "analysis").mkdir(parents=True)

    pre_run_context = capture_minikube_resource_context()
    write_json_atomic(attempt_dir / "cluster" / "minikube_resource_context_pre_run.json", pre_run_context)

    cleanup = CleanupRegistry()
    invalidity_reasons: List[str] = []
    procs_to_stop: List[Tuple[str, subprocess.Popen]] = []
    # Phase 15B.3M.1 (pre-V5 SSH tunnel) / Phase 15B.3N.3 (CORR-40, V5 Docker
    # relay): passive, non-invasive lifecycle evidence for the official
    # load-generator transport. Prior to Phase 15B.3M.1 the harness only
    # ever confirmed the tunnel's URL resolved once at launch
    # (`wrong_traffic_path_used`) and never recorded whether the transport
    # remained alive for the run's actual duration. This makes NO new
    # network calls and sends NO probe traffic through the transport -- it
    # only inspects the already-running container/process's own state and
    # its own logs.
    transport_lifecycle: Dict[str, Any] = {}
    relay_container_name: Dict[str, Optional[str]] = {"name": None}

    def _stop_background_procs():
        for name, p in procs_to_stop:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()

    def _stop_relay_container():
        # V5 (CORR-40): the relay is a detached `docker run -d` container,
        # not a Popen child of this process -- it cannot be stopped via
        # `procs_to_stop`/`Popen.terminate()`. Captures final state and
        # logs BEFORE removal, even if the relay already died on its own,
        # so route-integrity evidence is never lost to cleanup ordering.
        name = relay_container_name["name"]
        if not name:
            return
        running, exit_code = check_relay_container_state(name)
        already_dead_before_cleanup = running is False
        transport_lifecycle["already_dead_before_cleanup"] = already_dead_before_cleanup
        transport_lifecycle["exit_code_before_cleanup"] = exit_code if already_dead_before_cleanup else None
        logs_proc = _run(["docker", "logs", name], timeout=15)
        docker_logs_path = Path(f"/tmp/relay_{attempt_id}.log")
        docker_logs_path.write_text((logs_proc.stdout or "") + (logs_proc.stderr or ""))
        transport_lifecycle["docker_logs_path"] = str(docker_logs_path)
        _run(["docker", "rm", "-f", name], timeout=15)
        transport_lifecycle["exit_code_after_cleanup"] = exit_code
        transport_lifecycle["cleanup_action"] = classify_tunnel_termination_action(bool(already_dead_before_cleanup))

    def _restore_hpa_and_idle():
        # HPA restoration itself is the priority operation and happens
        # immediately, unconditionally of how long convergence takes.
        restore_hpa()
        time.sleep(2)
        wait_timeout = restoration_wait_seconds(int(slot["replicas"]))
        state = wait_for_idle_one_replica(timeout_seconds=wait_timeout)
        converged = all(v == 1 for v in state.values())
        if converged:
            print(f"  Restored HPA; idle state converged within {wait_timeout:.0f}s wait: {state}")
        else:
            print(
                f"  Restored HPA (confirmed applied), but natural scale-down convergence to 1/1/1/1 "
                f"did NOT complete within the {wait_timeout:.0f}s wait derived from starting_replicas="
                f"{slot['replicas']}. Last observed state: {state}. HPA itself IS restored and will "
                f"continue converging on its own -- this does NOT silently claim canonical idle state. "
                f"No new official run should start until 1/1/1/1 is independently confirmed."
            )

    cleanup.register("stop_background_processes", _stop_background_procs)
    cleanup.register("stop_relay_container", _stop_relay_container)
    cleanup.register("restore_hpa_and_wait_idle", _restore_hpa_and_idle)
    cleanup.install_signal_and_exit_handlers()

    start_utc = datetime.now(timezone.utc).isoformat()
    tunnel_url: Optional[str] = None
    hpa_absent: Optional[bool] = None
    try:
        print("=== HPA isolation ===")
        delete_hpa()
        time.sleep(2)
        hpa_absent = verify_no_hpa_targets_deployment()
        if not hpa_absent:
            invalidity_reasons.append("hpa_unexpectedly_active: HPA still present after delete")

        print(f"=== Scaling to {slot['replicas']} replicas ===")
        replica_state = scale_and_wait(int(slot["replicas"]))
        replicas_ok = all(v == int(slot["replicas"]) for v in replica_state.values())
        if not replicas_ok:
            invalidity_reasons.append(f"wrong_replica_count_achieved: {replica_state}")
        pod_names_pre = sorted(get_pod_names())
        write_json_atomic(attempt_dir / "cluster" / "pod_identities_pre_stabilization.json", {"pod_names": pod_names_pre})

        print(f"=== {manifest['steady_state_timing']['fixed_replica_stabilization_seconds']}s stabilization (no traffic) ===")
        time.sleep(manifest["steady_state_timing"]["fixed_replica_stabilization_seconds"])

        print("=== Capturing pre-traffic Kubernetes Events (evidence hardening) ===")
        events_pre = fetch_events_json()
        write_json_atomic(attempt_dir / "cluster" / "kubernetes_events_pre.json", events_pre)

        # Phase 15B.3M.2 (CORR-39 resolution): distinguishes "pods Ready"
        # from "the demo-app-loadtest Service actually had those pods as
        # usable endpoints" -- a read-only Kubernetes API snapshot, no
        # traffic through the loadtest path itself.
        loadtest_endpoints_pre = fetch_loadtest_endpoints_json()
        write_json_atomic(attempt_dir / "cluster" / "loadtest_endpoints_pre_traffic.json", loadtest_endpoints_pre)

        print("=== Load-generator transport (Docker-network raw TCP relay -> NodePort, Protocol V5/CORR-40) ===")
        transport_result = start_loadtest_transport(attempt_id)
        tunnel_url: Optional[str] = transport_result.get("target_url")
        if "error" in transport_result:
            # CORR-54: a required pre-traffic transport-establishment failure --
            # no traffic has been generated yet. Raised, not appended to
            # invalidity_reasons, so this can never be scored as a scientific
            # execution_structural_validity result (see PreTrafficToolingFailure's
            # own docstring above).
            transport_lifecycle.update(transport_result)
            raise PreTrafficToolingFailure(
                "required_v5_transport_resolution_failed_before_traffic",
                {"transport_error": transport_result["error"], **transport_result},
            )
        relay_container_name["name"] = transport_result["container_name"]
        transport_lifecycle.update(transport_result)
        if not tunnel_url:
            # CORR-54: same pre-traffic classification -- the relay launched but
            # its host port could not be resolved, so no traffic can be sent.
            raise PreTrafficToolingFailure(
                "v5_relay_launched_but_host_port_unresolved_before_traffic",
                dict(transport_result),
            )

        prom_proc, started_prom = start_prometheus_port_forward()
        if started_prom and prom_proc is not None:
            procs_to_stop.append(("prometheus_port_forward", prom_proc))

        traffic_duration = manifest["steady_state_timing"]["traffic_duration_seconds"]
        collector_duration = traffic_duration + 30
        gen_cap = manifest["generator_capacity"]

        print("=== Starting collectors ===")
        metrics_proc = subprocess.Popen(
            [VENV_PYTHON, "-m", "collectors.metrics_collector",
             "--experiment-id", attempt_id, "--mode", "manual",
             "--traffic-profile", slot["cell_or_transition_id"],
             "--duration-seconds", str(collector_duration), "--seed", "42"],
        )
        procs_to_stop.append(("metrics_collector", metrics_proc))

        readiness_proc = subprocess.Popen(
            [VENV_PYTHON, "-m", "scripts.pod_readiness_collector",
             "--experiment-dir", str(attempt_dir / "cluster"),
             "--duration-seconds", str(collector_duration)],
        )
        procs_to_stop.append(("pod_readiness_collector", readiness_proc))

        counter_proc = subprocess.Popen(
            [VENV_PYTHON, "-m", "scripts.per_pod_counter_collector",
             "--experiment-dir", str(attempt_dir / "routing"),
             "--duration-seconds", str(collector_duration)],
        )
        procs_to_stop.append(("per_pod_counter_collector", counter_proc))

        time.sleep(1)

        schedule_full_path = PROJECT_ROOT / slot["schedule_path"]
        profile_path = schedule_full_path.parent / f"{slot['cell_or_transition_id']}.yaml"
        if not profile_path.exists():
            invalidity_reasons.append(f"wrong_or_unhashed_schedule: traffic profile YAML not found: {profile_path}")

        # Phase 15B.3M.2/15B.3M.3 (CORR-39 resolution + proof-sufficiency
        # hardening): a single end-of-traffic poll() cannot explain WHEN
        # during the run the transport might have died, and component
        # liveness alone was never sufficient forwarding-integrity proof
        # (Phase 15B.3M.3's core correction, unchanged in substance under
        # V5/CORR-40). Sample the same evidence layers at the same ~10s
        # cadence the existing readiness/counter collectors already use,
        # for the duration of the blocking traffic_generator call: (1)
        # relay container liveness (`docker inspect`), (2) whether the
        # expected local TCP listener is still bound (read-only `lsof`),
        # (3) the demo-app-loadtest Service's own Endpoints (read-only
        # Kubernetes API call), (4) delivered-request evidence is computed
        # afterward, offline, from the already-collected metrics/counter
        # files (see the post-processing section) -- never from a live
        # probe. Passive process/socket-state inspection and read-only API
        # calls only -- no probe HTTP traffic is ever sent through the
        # transport. `tunnel_process_alive`/`local_listener_present` keys
        # are retained (aliased to the container-based checks) so the
        # existing, UNCHANGED `aggregate_observer_evidence_for_window()`/
        # `classify_route_integrity_window()` pure functions need no
        # redesign -- only the observer's data source changed.
        tunnel_port = _extract_port_from_url(tunnel_url)
        relay_name_for_sampling = relay_container_name["name"]
        tunnel_periodic_samples: List[Dict[str, Any]] = []
        _stop_periodic_sampling = threading.Event()

        def _sample_route_integrity_periodically() -> None:
            sample_start_mono = time.monotonic()
            while not _stop_periodic_sampling.wait(10):
                endpoints = fetch_loadtest_endpoints_json()
                container_running, container_exit_code = (
                    check_relay_container_state(relay_name_for_sampling) if relay_name_for_sampling else (None, None)
                )
                listener_present = check_local_listener_present(tunnel_port) if tunnel_port else None
                tunnel_periodic_samples.append(
                    {
                        "elapsed_seconds_since_traffic_start": round(time.monotonic() - sample_start_mono, 1),
                        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                        # canonical CORR-39 evidence-window keys (unchanged
                        # names, consumed by the unmodified pure functions):
                        "tunnel_process_alive": container_running,
                        "tunnel_process_exit_code": container_exit_code,
                        "local_listener_present": listener_present,
                        "listener_port": tunnel_port,
                        # V5/CORR-40 transport-neutral aliases, for the new
                        # loadtest_transport_lifecycle.json artifact:
                        "relay_container_running": container_running,
                        "relay_container_exit_code": container_exit_code,
                        "host_listener_present": listener_present,
                        "loadtest_endpoints": parse_endpoints_ready_count(endpoints) if "_error" not in endpoints else None,
                        "endpoint_query_error": endpoints.get("_error"),
                    }
                )

        sampler_thread = threading.Thread(target=_sample_route_integrity_periodically, daemon=True)
        sampler_thread.start()

        # Phase 15B.3O.1 (CORR-41): a SEPARATE, faster-cadence sampler for
        # the Kubernetes-dataplane NodePort ingress evidence -- deliberately
        # not folded into `_sample_route_integrity_periodically()`'s 10s
        # loop, since sec.10 of the governing phase requires a cadence
        # faster than the 10s evidence window so every window can contain a
        # fully-interior sample pair. Read-only (`docker exec ... iptables-
        # save -c`); sends no traffic through the loadtest path.
        #
        # Phase 15B.3P.1 (CORR-42, Protocol V7): the fixed 3.0s cadence
        # above was found to structurally under-sample some 10s route
        # windows at its REAL (non-nominal) inter-sample gap, and to leave
        # short trailing windows with too few interior samples to form a
        # confirming pair -- see docs/experimental_design_corrections.md
        # CORR-42. Three combined, acquisition-only fixes (NONE of which
        # touch classify_route_integrity_window/classify_route_integrity/
        # compute_dataplane_ingress_evidence_windows): (1) a reduced
        # baseline cadence (V7 constant); (2) an immediate reactive
        # follow-up sample the instant a regime/rule-identity transition is
        # detected, so a legitimate same-identity confirming pair has the
        # best chance of forming even when a reset lands inside a sparse
        # window; (3) one final anchor sample taken the instant the traffic
        # generator process exits, directly targeting short/trailing
        # windows at the end of a run. All three only ever ADD samples to
        # the same `dataplane_periodic_samples` list that
        # `compute_dataplane_ingress_evidence_windows()` already consumes
        # generically -- no fabricated deltas, no cross-window/cross-
        # identity smearing.
        dataplane_nodeport = transport_lifecycle.get("nodeport")
        dataplane_periodic_samples: List[Dict[str, Any]] = []
        dataplane_reactive_resample_count = 0
        dataplane_anchor_sample_taken = False
        dataplane_acquisition_gap_exceeded_count = 0
        _stop_dataplane_sampling = threading.Event()
        _dataplane_samples_lock = threading.Lock()
        _last_dataplane_sample_taken_mono: List[Optional[float]] = [None]

        def _take_and_record_dataplane_sample(sample_start_mono: float) -> Dict[str, Any]:
            # Phase 15B.3P.3: records the REAL wall-clock gap since the
            # previous sample (of any kind -- periodic, reactive, or
            # anchor) was taken, and whether it exceeded
            # DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8. This is
            # pure OBSERVABILITY -- it is recorded on the artifact for
            # post-hoc audit and never fed into classify_route_integrity_
            # window()/classify_route_integrity() or into whether a
            # sample pair is accepted; a large gap can only ever result in
            # FEWER confirming pairs being available (honestly reflected
            # as UNVERIFIED where evidence is genuinely insufficient),
            # never in a fabricated positive delta. See CORR-43/V8's
            # amended rationale: this project does not claim a hard,
            # OS/Docker-enforced real-time upper bound on the inter-sample
            # gap -- only an empirically well-supported, now EXPLICITLY
            # MONITORED assumption.
            nonlocal dataplane_acquisition_gap_exceeded_count
            t_start = time.monotonic()
            sample = sample_dataplane_ingress_rules(dataplane_nodeport)
            with _dataplane_samples_lock:
                previous_taken_mono = _last_dataplane_sample_taken_mono[0]
                gap = None if previous_taken_mono is None else round(t_start - previous_taken_mono, 4)
                _last_dataplane_sample_taken_mono[0] = t_start
                gap_exceeded = gap is not None and gap > DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8
                if gap_exceeded:
                    dataplane_acquisition_gap_exceeded_count += 1
                sample["elapsed_seconds_since_traffic_start"] = round(time.monotonic() - sample_start_mono, 2)
                sample["sampled_at_utc"] = datetime.now(timezone.utc).isoformat()
                sample["dataplane_mode"] = "kube_proxy_iptables_nf_tables"
                sample["service_namespace"] = APP_NS
                sample["service_name"] = LOADTEST_SVC
                sample["actual_gap_seconds_from_previous_sample"] = gap
                sample["acquisition_gap_exceeded_assumed_bound"] = gap_exceeded
                dataplane_periodic_samples.append(sample)
            return sample

        dataplane_sample_start_mono = time.monotonic()

        def _sample_dataplane_ingress_periodically() -> None:
            nonlocal dataplane_reactive_resample_count
            previous_sample: Optional[Dict[str, Any]] = None
            while not _stop_dataplane_sampling.wait(DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7):
                sample = _take_and_record_dataplane_sample(dataplane_sample_start_mono)
                if should_trigger_immediate_resample(previous_sample, sample):
                    if _stop_dataplane_sampling.wait(DATAPLANE_INGRESS_TRANSITION_RESAMPLE_DELAY_SECONDS):
                        previous_sample = sample
                        break
                    dataplane_reactive_resample_count += 1
                    sample = _take_and_record_dataplane_sample(dataplane_sample_start_mono)
                previous_sample = sample

        dataplane_sampler_thread = threading.Thread(target=_sample_dataplane_ingress_periodically, daemon=True)
        dataplane_sampler_thread.start()

        print("=== Traffic execution (schedule REPLAY, connection_mode=fresh) ===")
        traffic_start_epoch = datetime.now(timezone.utc).timestamp()
        gen_result = subprocess.run(
            [VENV_PYTHON, "-m", "loadgen.traffic_generator",
             "--profile", str(profile_path),
             "--experiment-id", attempt_id,
             "--target-url", tunnel_url or "http://localhost:1",
             "--replay-schedule", str(schedule_full_path),
             "--arrival-process", "deterministic",
             "--connection-mode", manifest["arrival_process"]["connection_mode"],
             "--timeout", str(gen_cap["default_timeout_seconds"]),
             "--max-in-flight", str(gen_cap["frozen_calibration_max_in_flight"])],
        )
        traffic_end_epoch = datetime.now(timezone.utc).timestamp()
        transport_lifecycle["traffic_start_epoch"] = traffic_start_epoch
        transport_lifecycle["traffic_end_epoch"] = traffic_end_epoch
        if gen_result.returncode != 0:
            invalidity_reasons.append(f"collector_or_generator_structural_failure: traffic_generator exit code {gen_result.returncode}")

        # Phase 15B.3P.1 (CORR-42): final anchor sample, taken the instant
        # the generator subprocess has exited and BEFORE the periodic
        # sampler is stopped -- directly targets short/trailing windows
        # whose material interval ends near the end of the traffic run.
        _take_and_record_dataplane_sample(dataplane_sample_start_mono)
        dataplane_anchor_sample_taken = True

        _stop_periodic_sampling.set()
        sampler_thread.join(timeout=5)
        transport_lifecycle["periodic_samples_during_traffic"] = tunnel_periodic_samples

        _stop_dataplane_sampling.set()
        dataplane_sampler_thread.join(timeout=5)
        dataplane_ingress_timeline = {
            "attempt_id": attempt_id,
            "dataplane_ingress_sampling_interval_seconds": DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7,
            "dataplane_ingress_sampling_protocol_version": 7,
            "dataplane_ingress_transition_resample_delay_seconds": DATAPLANE_INGRESS_TRANSITION_RESAMPLE_DELAY_SECONDS,
            "dataplane_reactive_resample_count": dataplane_reactive_resample_count,
            "dataplane_anchor_sample_taken": dataplane_anchor_sample_taken,
            # Phase 15B.3P.3: pure observability, added to the already-frozen
            # V7/V8 acquisition schedule without changing it -- see
            # `DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8`'s own
            # amended docstring. Never fed into classification.
            "dataplane_acquisition_worst_case_cadence_assumed_seconds": DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8,
            "dataplane_acquisition_gap_exceeded_count": dataplane_acquisition_gap_exceeded_count,
            "nodeport": dataplane_nodeport,
            "periodic_samples": sorted(dataplane_periodic_samples, key=lambda s: s["elapsed_seconds_since_traffic_start"]),
        }
        write_json_atomic(attempt_dir / "cluster" / "loadtest_dataplane_ingress_timeline.json", dataplane_ingress_timeline)

        # Passive liveness/listener check (no new network call, no probe
        # traffic through the transport itself) -- records whether the relay
        # container and its expected local listener were still present the
        # instant traffic finished, closing the evidence gap that left
        # index12's tunnel state unknowable (pre-V5) and remains closed for
        # the V5 container-based transport.
        container_running_at_end, container_exit_code_at_end = (
            check_relay_container_state(relay_container_name["name"]) if relay_container_name["name"] else (None, None)
        )
        transport_lifecycle["running_at_traffic_end"] = container_running_at_end
        transport_lifecycle["alive_at_traffic_end"] = container_running_at_end
        transport_lifecycle["host_listener_present_at_traffic_end"] = check_local_listener_present(tunnel_port) if tunnel_port else None
        transport_lifecycle["local_listener_present_at_traffic_end"] = transport_lifecycle["host_listener_present_at_traffic_end"]
        if container_running_at_end is False:
            transport_lifecycle["exit_code_at_traffic_end"] = container_exit_code_at_end
        # Phase 15B.3N.1/15B.3N.3: diagnostic-only listener-owner enrichment
        # (never consulted by route-integrity classification) -- lets a
        # future review directly compare the relay container against
        # whatever process/container actually owns the published listener.
        transport_lifecycle["listener_owner_at_traffic_end"] = (
            resolve_local_listener_owner(tunnel_port) if tunnel_port else
            {"listener_owner_pid": None, "listener_owner_command": None}
        )

        loadtest_endpoints_post = fetch_loadtest_endpoints_json()
        write_json_atomic(attempt_dir / "cluster" / "loadtest_endpoints_post_traffic.json", loadtest_endpoints_post)

        pod_names_post = sorted(get_pod_names())
        write_json_atomic(attempt_dir / "cluster" / "pod_identities_post_traffic.json", {"pod_names": pod_names_post})
        if pod_names_post != pod_names_pre:
            invalidity_reasons.append(
                f"unexpected_pod_identity_change_during_scored_window: pre={pod_names_pre} post={pod_names_post}"
            )

        print("=== Capturing post-traffic Kubernetes Events (before cleanup) ===")
        events_post = fetch_events_json()
        write_json_atomic(attempt_dir / "cluster" / "kubernetes_events_post.json", events_post)
        normalized_events = normalize_kubernetes_events(events_post)
        with open(attempt_dir / "cluster" / "kubernetes_events_normalized.csv", "w", newline="", encoding="utf-8") as ef:
            ew = csv.DictWriter(ef, fieldnames=KUBERNETES_EVENTS_NORMALIZED_CSV_FIELDNAMES)
            ew.writeheader()
            for row in normalized_events:
                ew.writerow(row)

        print("=== Checking for container restarts (evidence hardening: fetch --previous logs immediately) ===")
        pods_snapshot_json = safe_fetch_current_pod_snapshot_json()
        current_snapshot = parse_pod_readiness_snapshot(
            pods_snapshot_json,
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).timestamp(),
            -1.0,
        )
        # Any pod with restart_count > 0 gets its --previous log captured now,
        # while the pod object still exists -- harmless even if the restart
        # predates this run's own window; evidence-preserving, never fabricated.
        prev_logs_dir = attempt_dir / "cluster" / "previous_container_logs"
        prev_logs_dir.mkdir(parents=True, exist_ok=True)
        for row in current_snapshot:
            if int(row.get("restart_count", 0)) > 0:
                log_text, available = fetch_previous_container_logs(row["pod_name"])
                (prev_logs_dir / f"{row['pod_name']}.log").write_text(log_text)
                write_json_atomic(prev_logs_dir / f"{row['pod_name']}_availability.json", {"pod_name": row["pod_name"], "previous_log_available": available})

        print("=== Waiting for collectors to finish ===")
        metrics_exit = metrics_proc.wait(timeout=collector_duration + 60)
        readiness_exit = readiness_proc.wait(timeout=collector_duration + 60)
        counter_exit = counter_proc.wait(timeout=collector_duration + 60)
        procs_to_stop = [(n, p) for n, p in procs_to_stop if p not in (metrics_proc, readiness_proc, counter_proc)]
        if metrics_exit != 0:
            invalidity_reasons.append(f"collector_or_generator_structural_failure: metrics_collector exit code {metrics_exit}")
        if readiness_exit != 0 or counter_exit != 0:
            invalidity_reasons.append("collector_or_generator_structural_failure: pod/counter collector non-zero exit")

        readiness_metadata_path = attempt_dir / "cluster" / "pod_readiness_metadata.json"
        if readiness_metadata_path.exists():
            readiness_metadata = json.loads(readiness_metadata_path.read_text())
            contamination_ok, contamination_msg = check_no_external_contamination_during_run(readiness_metadata)
            if not contamination_ok:
                invalidity_reasons.append(contamination_msg)

        post_run_context = capture_minikube_resource_context()
        write_json_atomic(attempt_dir / "cluster" / "minikube_resource_context_post_run.json", post_run_context)

        # Phase 15B.3D: Prometheus health must be checked HERE, before the
        # `finally` block below tears down the Prometheus port-forward
        # tunnel -- checking afterward (as this metadata field previously
        # did) always reports False regardless of Prometheus's actual
        # health during the run, since curl then has nothing listening on
        # localhost:9090 at all. This field is diagnostic-only and never
        # consulted for execution_structural_validity/steady_state_fit_eligibility.
        prom_health_diagnostic = check_prometheus_diagnostic_health()
        prom_health = prom_health_diagnostic["healthy"]
        prom_health_error_type = prom_health_diagnostic["error_type"]
        prom_health_error_message = prom_health_diagnostic["error_message"]

    except PreTrafficToolingFailure as exc:
        # CORR-54: deliberately short-circuits BEFORE any scientific
        # structural-classification or ledger-write code below is ever
        # reached -- no execution_structural_validity value is computed, no
        # route_integrity value is computed, and atomic_write_ledger() is
        # never called for this attempt. `finally` below still runs
        # cleanup.run_all() exactly as it would for any other exception.
        write_json_atomic(
            attempt_dir / "pre_traffic_tooling_failure.json",
            {
                "attempt_id": attempt_id,
                "campaign_sequence_index": args.campaign_sequence_index,
                "planned_slot_id": slot["planned_slot_id"],
                "attempt_number": args.attempt_number,
                "reason": exc.reason,
                "detail": exc.detail,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(f"\nPRE_TRAFFIC_TOOLING_FAILURE: {exc.reason}")
        print(json.dumps(exc.detail, indent=2, default=str))
        print(
            "No traffic was generated. No scientific structural-validity or "
            "route-integrity result was computed. The live validation ledger "
            "was not written for this attempt -- the slot remains scientifically "
            "unresolved. This physical attempt number is consumed; its artifact "
            "directory is preserved."
        )
        return EXIT_PRE_TRAFFIC_TOOLING_FAILURE

    finally:
        cleanup.run_all()

    end_utc = datetime.now(timezone.utc).isoformat()

    # Phase 15B.3N.3 (CORR-40, Protocol V5): persist the transport-neutral
    # lifecycle evidence gathered above, plus whatever the relay container
    # itself wrote to its own logs (read-only; `_stop_relay_container()`
    # already captured these to disk during cleanup). Written to a NEW,
    # transport-neutral artifact name -- the pre-V5 SSH-specific
    # `loadtest_tunnel_lifecycle.json` name/parsing code is retained
    # unchanged for reading historical (attempt1/attempt2 and all pre-V5)
    # attempts, never reused for V5 attempts.
    if transport_lifecycle:
        docker_logs_path = Path(transport_lifecycle.get("docker_logs_path", ""))
        transport_lifecycle["captured_log_text"] = docker_logs_path.read_text() if docker_logs_path.exists() else None
        write_json_atomic(attempt_dir / "cluster" / "loadtest_transport_lifecycle.json", transport_lifecycle)

    # -------------------------------------------------------------------
    # Post-processing (pure Python, no cluster access): traffic structural
    # checks, decision-aligned extraction, CORR-35 routing analysis.
    # -------------------------------------------------------------------
    traffic_dir = PROJECT_ROOT / "persistent-data/traffic" / attempt_id
    metrics_dir = PROJECT_ROOT / "persistent-data/metrics" / attempt_id
    try:
        (attempt_dir / "traffic").symlink_to(traffic_dir)
    except FileExistsError:
        pass
    try:
        (attempt_dir / "metrics").symlink_to(metrics_dir)
    except FileExistsError:
        pass

    traffic_metadata = {}
    traffic_meta_path = traffic_dir / "metadata.json"
    if traffic_meta_path.exists():
        traffic_metadata = json.loads(traffic_meta_path.read_text())
        ok, msg = check_offered_equals_sent(traffic_metadata)
        if not ok:
            invalidity_reasons.append(f"offered_not_equal_sent_due_to_undersized_generator_max_in_flight: {msg}")
        ok, msg = check_no_dropped_overload(traffic_metadata)
        if not ok:
            invalidity_reasons.append(f"generator_dropped_overload_due_to_harness_capacity_not_application_response: {msg}")
    else:
        invalidity_reasons.append("collector_or_generator_structural_failure: traffic metadata.json missing")

    readiness_path = attempt_dir / "cluster" / "pod_readiness_history.csv"
    readiness_rows: List[Dict[str, Any]] = []
    if readiness_path.exists():
        readiness_rows = load_pod_readiness_history(readiness_path)

    decision_aligned_v2: Dict[str, Any] = {}
    metrics_csv_path = metrics_dir / "metrics.csv"
    metrics_rows: List[Dict[str, Any]] = []
    if metrics_csv_path.exists() and readiness_rows:
        metrics_rows = load_metrics_csv(metrics_csv_path)
        decision_aligned_v2 = extract_decision_aligned_v2(
            metrics_rows,
            readiness_rows,
            manifest["steady_state_timing"]["decision_aligned_extraction_timestamps_relative_to_traffic_onset"],
            staleness_threshold=30.0,
            intended_replicas=int(slot["replicas"]),
        )
        write_json_atomic(attempt_dir / "analysis" / "decision_aligned_observations.json", decision_aligned_v2)
    else:
        invalidity_reasons.append("collector_or_generator_structural_failure: metrics.csv or pod_readiness_history.csv missing")

    routing_summary: Dict[str, Any] = {}
    counters_path = attempt_dir / "routing" / "per_pod_request_counters.csv"
    counter_rows: List[Dict[str, Any]] = []
    if readiness_path.exists() and counters_path.exists():
        counter_rows = load_per_pod_counters(counters_path)
        scored_window = tuple(manifest["steady_state_timing"]["scored_window_seconds"])
        boundaries, ready_set_at, pod_samples, relevant_pod_set_at = build_routing_inputs(readiness_rows, counter_rows, scored_window)
        if len(boundaries) >= 2 and pod_samples:
            result = compute_expected_and_actual(boundaries, ready_set_at, pod_samples, relevant_pod_set_at)
            routing_summary = gate_and_classify_routing(result)
            if routing_summary["classification"] == "STRUCTURAL_ROUTING_FAILURE":
                invalidity_reasons.append("structural_routing_failure_per_per_pod_routing_diagnostics")
        else:
            routing_summary = {
                "classification": "INSUFFICIENT_ROUTING_SAMPLE",
                "reason": "insufficient boundaries/samples collected",
                "retained_interval_count": 0,
                "conservation_assessable": False,
                "routing_evidence_sufficient": False,
            }
        write_json_atomic(attempt_dir / "routing" / "routing_summary.json", routing_summary)
    else:
        invalidity_reasons.append("collector_or_generator_structural_failure: readiness/counter artifacts missing")

    # Phase 15B.3M.2 (CORR-39 resolution) / 15B.3M.3 (proof-sufficiency
    # hardening) / 15B.3M.4 (temporal-coverage hardening) / 15B.3M.5
    # (window-local delivery-attribution hardening): official external
    # workload-route integrity -- distinct from, and checked independently
    # of, the offered==sent/dropped_overload==0 checks above (those only
    # prove the generator attempted the frozen schedule against its own
    # configured destination, never that it was delivered into the
    # intended Kubernetes workload path). Pure post-hoc analysis of
    # already-collected files; touches no live cluster/tunnel state.
    # Phase 15B.3M.4: EVERY materially-qualifying zero-success run (not
    # only the longest) is partitioned into
    # ROUTE_INTEGRITY_EVIDENCE_WINDOW_SECONDS-wide evidence windows, each
    # independently classified -- one early delivered request can no
    # longer confirm route integrity for the remainder of a long interval
    # with no further evidence; see classify_route_integrity's own
    # docstring for the exact attempt-level aggregation rule.
    # Phase 15B.3M.5: decisive per-window delivered-request evidence now
    # attributes an HTTP response by the request's DISPATCH time (not its
    # completion time), and treats server-side `request_rate_rps` (a
    # 60s-rolling Prometheus rate) as supporting context only, never
    # decisive per-window proof -- see
    # compute_delivered_request_evidence_windows()'s own docstring.
    route_integrity_result: Dict[str, Any] = {"classification": None}
    traffic_summary_path = traffic_dir / "traffic_summary.csv"
    if traffic_summary_path.exists():
        summary_rows = load_traffic_summary_csv(traffic_summary_path)
        traffic_duration_seconds = float(manifest["steady_state_timing"]["traffic_duration_seconds"])
        scored_window = tuple(manifest["steady_state_timing"]["scored_window_seconds"])

        requests_rows: List[Dict[str, Any]] = []
        requests_csv_path = traffic_dir / "requests.csv"
        if requests_csv_path.exists():
            requests_rows = load_requests_csv(requests_csv_path)

        all_zero_success_runs = find_all_zero_success_runs(summary_rows)
        material_intervals = [
            run
            for run in all_zero_success_runs
            if is_interval_material(run[0], run[1], traffic_duration_seconds, scored_window)
        ]

        periodic_samples = transport_lifecycle.get("periodic_samples_during_traffic", [])
        traffic_start_epoch = transport_lifecycle.get("traffic_start_epoch")

        evidence_windows: List[Dict[str, Any]] = []
        for run_start, run_end in material_intervals:
            # Phase 15B.3P.2 (CORR-43, Protocol V8): merge a pathologically
            # short trailing remainder window into the preceding window
            # rather than leaving it as its own, potentially unconfirmable,
            # standalone window -- see MIN_EVIDENCE_WINDOW_SECONDS_V8's own
            # docstring for the deterministic derivation. Ordinary,
            # exact-multiple interval lengths (and interval lengths already
            # fully covered by a single window) are entirely unaffected.
            for window_start, window_end in partition_into_evidence_windows(
                run_start, run_end, min_window_seconds=MIN_EVIDENCE_WINDOW_SECONDS_V8
            ):
                observer_evidence = aggregate_observer_evidence_for_window(window_start, window_end, periodic_samples)
                delivered_evidence = compute_delivered_request_evidence_windows(
                    window_start_elapsed=window_start,
                    window_end_elapsed=window_end,
                    traffic_start_epoch=traffic_start_epoch,
                    requests_rows=requests_rows,
                    metrics_rows=metrics_rows,
                    counter_rows=counter_rows,
                )
                dataplane_evidence = compute_dataplane_ingress_evidence_windows(
                    window_start_elapsed=window_start,
                    window_end_elapsed=window_end,
                    traffic_start_epoch=traffic_start_epoch,
                    dataplane_samples=dataplane_periodic_samples,
                )
                window_verdict = classify_route_integrity_window(
                    delivered_request_evidence=delivered_evidence["delivered_request_evidence"],
                    tunnel_process_alive=observer_evidence["tunnel_process_alive"],
                    local_listener_present=observer_evidence["local_listener_present"],
                    nodeport_dataplane_ingress_evidence=dataplane_evidence["nodeport_dataplane_ingress_evidence"],
                )
                route_evidence_sources = list(delivered_evidence["delivery_evidence_sources"])
                if dataplane_evidence["nodeport_dataplane_ingress_evidence"]:
                    route_evidence_sources.append("nodeport_dataplane_ingress")
                evidence_windows.append(
                    {
                        "material_interval": [run_start, run_end],
                        "window_start_elapsed": window_start,
                        "window_end_elapsed": window_end,
                        "tunnel_process_alive": observer_evidence["tunnel_process_alive"],
                        "local_listener_present": observer_evidence["local_listener_present"],
                        # V5/CORR-40 transport-neutral aliases (same values,
                        # clearer names for the container-based V5 transport):
                        "relay_container_running": observer_evidence["tunnel_process_alive"],
                        "host_listener_present": observer_evidence["local_listener_present"],
                        "loadtest_ready_endpoint_count": observer_evidence["loadtest_ready_endpoint_count"],
                        "endpoint_query_error": observer_evidence["endpoint_query_error"],
                        "observer_sample_count": observer_evidence["sample_count"],
                        "http_response_delivery_evidence": delivered_evidence["http_response_delivery_evidence"],
                        "http_response_request_indices": delivered_evidence["http_response_request_indices"],
                        "http_response_dispatch_times": delivered_evidence["http_response_dispatch_times"],
                        "http_response_completion_times": delivered_evidence["http_response_completion_times"],
                        "counter_delivery_evidence": delivered_evidence["counter_delivery_evidence"],
                        "counter_intervals_used": delivered_evidence["counter_intervals_used"],
                        "counter_intervals_rejected_as_cross_window": delivered_evidence["counter_intervals_rejected_as_cross_window"],
                        "rolling_server_rps_observed": delivered_evidence["rolling_server_rps_observed"],
                        "rolling_server_rps_supporting_only": delivered_evidence["rolling_server_rps_supporting_only"],
                        # Phase 15B.3O.1/CORR-41: Kubernetes-dataplane ingress evidence.
                        "nodeport_dataplane_ingress_evidence": dataplane_evidence["nodeport_dataplane_ingress_evidence"],
                        "dataplane_counter_intervals_used": dataplane_evidence["dataplane_counter_intervals_used"],
                        "dataplane_counter_intervals_rejected_cross_window": dataplane_evidence["dataplane_counter_intervals_rejected_cross_window"],
                        "dataplane_counter_intervals_rejected_reset": dataplane_evidence["dataplane_counter_intervals_rejected_reset"],
                        "dataplane_observer_error_present": dataplane_evidence["dataplane_observer_error_present"],
                        "dataplane_backend_regimes_with_evidence": dataplane_evidence["dataplane_backend_regimes_with_evidence"],
                        "delivered_request_evidence": delivered_evidence["delivered_request_evidence"],
                        "delivery_evidence_sources": delivered_evidence["delivery_evidence_sources"],
                        "route_evidence_sources": route_evidence_sources,
                        "window_route_status": window_verdict["window_route_status"],
                        "reason": window_verdict["reason"],
                    }
                )

        route_integrity_result = classify_route_integrity(
            material_intervals=material_intervals,
            evidence_windows=evidence_windows,
        )

        route_integrity_timeline = {
            "attempt_id": attempt_id,
            "route_integrity_evidence_window_seconds": ROUTE_INTEGRITY_EVIDENCE_WINDOW_SECONDS,
            "all_zero_success_runs": all_zero_success_runs,
            "material_intervals": material_intervals,
            "evidence_windows": evidence_windows,
            "classification": route_integrity_result["classification"],
            "reason": route_integrity_result["reason"],
        }
        write_json_atomic(attempt_dir / "cluster" / "loadtest_route_integrity_timeline.json", route_integrity_timeline)

        if route_integrity_result["classification"] == "FAILED":
            invalidity_reasons.append(
                "wrong_traffic_path_used: official workload route confirmed to have stopped forwarding during "
                "traffic (tunnel/helper/listener failure in at least one evidence window of a material "
                "zero-success interval)"
            )
        elif route_integrity_result["classification"] == "UNVERIFIED":
            invalidity_reasons.append(
                "official_workload_route_integrity_unverified: at least one "
                f"{ROUTE_INTEGRITY_EVIDENCE_WINDOW_SECONDS:.0f}s evidence window within a material zero-success "
                f"delivery interval ({route_integrity_result['material_intervals']}) lacks positive "
                "delivered-request evidence -- component-level health (tunnel process, listener, Service "
                "endpoints) is supporting evidence only and full temporal coverage was not established (CORR-39)"
            )

    # Health-instability evidence: pods whose restart_count increased across
    # the readiness collector's own persisted history (first vs last tick).
    restarted_during_run: List[str] = []
    if len(readiness_rows) >= 2:
        first_ts = min(r["elapsed_seconds"] for r in readiness_rows)
        last_ts = max(r["elapsed_seconds"] for r in readiness_rows)
        first_rows = [r for r in readiness_rows if r["elapsed_seconds"] == first_ts]
        last_rows = [r for r in readiness_rows if r["elapsed_seconds"] == last_ts]
        restarted_during_run = restarted_pods_between(first_rows, last_rows)

    execution_structural_validity = EXECUTION_INVALID if invalidity_reasons else EXECUTION_VALID
    steady_state_fit_eligibility = decision_aligned_v2.get("run_level_classification", "STEADY_STATE_FIT_INELIGIBLE")

    attempt_metadata = build_attempt_metadata(
        calibration_protocol_version=manifest["protocol_version"],
        campaign_sequence_index=slot["campaign_sequence_index"],
        planned_slot_id=slot["planned_slot_id"],
        execution_attempt_number=args.attempt_number,
        attempt_id=attempt_id,
        session_id=session_id,
        wall_clock_start_utc=start_utc,
        wall_clock_end_utc=end_utc,
        run_kind=slot["run_kind"],
        cell_or_transition_id=slot["cell_or_transition_id"],
        repeat_number=slot["repeat_number"],
        replicas=slot.get("replicas"),
        offered_rps=slot.get("offered_rps"),
        intensity=manifest["scope"]["application_intensity"],
        schedule_path=slot["schedule_path"],
        schedule_sha256=schedule_hash,
        connection_mode=manifest["arrival_process"]["connection_mode"],
        timeout_seconds=manifest["generator_capacity"]["default_timeout_seconds"],
        max_in_flight=manifest["generator_capacity"]["frozen_calibration_max_in_flight"],
        target_url=tunnel_url,
        hpa_absence_verified=hpa_absent,
        rl_absence_verified=True,
        kind_state="stopped",
        app_resource_spec={"cpu_request": "100m", "cpu_limit": "500m", "memory_request": "64Mi", "memory_limit": "256Mi"},
        minikube_outer_resource_limits=pre_run_context,
        prometheus_health=prom_health,
        prometheus_health_diagnostic_error_type=prom_health_error_type,
        prometheus_health_diagnostic_error_message=prom_health_error_message,
        metrics_server_resolution="15s",
        protocol_version_applied=2,
        execution_structural_validity=execution_structural_validity,
        invalidity_reasons=invalidity_reasons,
        steady_state_fit_eligibility=steady_state_fit_eligibility,
        decision_points_observable_count=decision_aligned_v2.get("decision_points_observable_count", 0),
        decision_points_fit_eligible_count=decision_aligned_v2.get("decision_points_fit_eligible_count", 0),
    )
    attempt_metadata["restarted_pods_during_run"] = restarted_during_run
    attempt_metadata["corr_reference"] = "CORR-37"
    # Phase 15B.3M.4: metadata.json carries only the SUMMARY of the
    # windowed route-integrity classification -- the full per-window
    # timeline (potentially many windows across multiple material
    # intervals) is persisted separately at
    # cluster/loadtest_route_integrity_timeline.json, never duplicated
    # wholesale into metadata.json.
    attempt_metadata["route_integrity"] = {
        "classification": route_integrity_result.get("classification"),
        "reason": route_integrity_result.get("reason"),
        "material_intervals": route_integrity_result.get("material_intervals", []),
        "full_temporal_coverage": route_integrity_result.get("full_temporal_coverage"),
        "confirmed_window_count": route_integrity_result.get("confirmed_window_count"),
        "unverified_window_count": route_integrity_result.get("unverified_window_count"),
        "failed_window_count": route_integrity_result.get("failed_window_count"),
    }
    attempt_metadata["route_integrity_corr_reference"] = "CORR-39"
    # Phase 15B.3N.3 (CORR-40, Protocol V5): explicit transport-epoch
    # provenance on every attempt -- prevents future analysis from mixing
    # pre-V5 (SSH-tunnel-to-ClusterIP) and V5 (Docker-relay-to-NodePort)
    # measurements unknowingly. See analysis/transport_epoch_compatibility.json
    # for the still-PENDING cross-epoch model-fitting/HPA-260 questions this
    # does not itself resolve.
    attempt_metadata["workload_transport"] = {
        "protocol_version": 5,
        "type": transport_lifecycle.get("transport_type", TRANSPORT_TYPE_DOCKER_NETWORK_TCP_RELAY),
        "ingress_service": LOADTEST_SVC,
        "kubernetes_access_mode": "NodePort",
        "client_location": "mac_host",
        "connection_mode": manifest["arrival_process"]["connection_mode"],
        "session_affinity": "None",
    }
    # Phase 15B.3E (CORR-38 provenance hardening): explicit, unambiguous
    # protocol-layer provenance for every NEW attempt -- does not alter the
    # meaning of the historical calibration_protocol_version/
    # protocol_version_applied fields above, only adds to them.
    attempt_metadata.update(build_protocol_provenance())
    attempt_metadata["schedule_sha256_verified_against"] = {
        "global_sequence": schedule_hash,
        "on_disk_recomputed": schedule_hash,
    }
    write_json_atomic(attempt_dir / "metadata.json", attempt_metadata)

    ledger = ensure_ledger(sequence, path=ledger_path)
    ledger_state = ledger_state_for_execution_validity(execution_structural_validity)
    if ledger_state == LEDGER_STATE_VALID:
        update_slot_state(ledger, args.campaign_sequence_index, ledger_state, attempt_id=attempt_id, mark_valid=True)
    else:
        update_slot_state(ledger, args.campaign_sequence_index, ledger_state, attempt_id=attempt_id)
    atomic_write_ledger(ledger, path=ledger_path)

    print(json.dumps(attempt_metadata, indent=2, default=str))
    print(f"\nexecution_structural_validity: {execution_structural_validity}")
    print(f"steady_state_fit_eligibility: {steady_state_fit_eligibility}")
    if invalidity_reasons:
        print(f"Invalidity reasons: {invalidity_reasons}")
    if restarted_during_run:
        print(f"Pods restarted during run: {restarted_during_run}")
    return 0 if execution_structural_validity == EXECUTION_VALID else 1


if __name__ == "__main__":
    sys.exit(main())
