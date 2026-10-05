"""scripts/run_bridge_trial.py

Phase 15B.3N.4 (Gate B): NON-OFFICIAL pre-V5<->V5 transport bridge trial.
Static-replica (HPA removed), otherwise-identical steady-state measurement
of one already-frozen dynamics-calibration cell, through EITHER the
historical pre-V5 SSH-tunnel-to-ClusterIP transport OR the frozen V5
Docker-network-relay-to-NodePort transport -- to determine whether
model-critical server/Kubernetes measurements shift systematically with
the transport mechanism.

NOT an official calibration attempt: never touches the campaign ledger,
never counted as calibration/fitting support. Every output directory is
prefixed `NON_OFFICIAL_PROTOCOL_TRANSITION_VALIDATION_`.

Mirrors scripts/run_dynamics_calibration_slot.py's own steady-state
measurement structure (HPA isolation, scale-and-wait, 60s stabilization,
schedule replay, collectors, route-integrity evidence collection,
decision-aligned extraction, cleanup) as closely as possible, reusing its
own functions directly, so the ONLY deliberate difference between a
`--transport old` and `--transport v5` run of the same cell is the
transport mechanism itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import dynamics_calibration_lib as lib  # noqa: E402
from scripts.run_dynamics_calibration_slot import (  # noqa: E402
    APP_NS,
    MONITORING_NS,
    VENV_PYTHON,
    _run,
    authoritative_ready_count_at_target,
    capture_minikube_resource_context,
    check_local_listener_present,
    check_relay_container_state,
    delete_hpa,
    fetch_events_json,
    fetch_loadtest_endpoints_json,
    get_pod_names,
    load_metrics_csv,
    load_per_pod_counters,
    load_requests_csv,
    load_traffic_summary_csv,
    parse_endpoints_ready_count,
    restoration_wait_seconds,
    restore_hpa,
    resolve_local_listener_owner,
    scale_and_wait,
    start_loadtest_transport,
    start_loadtest_tunnel,
    verify_no_hpa_targets_deployment,
    wait_for_idle_one_replica,
    _extract_port_from_url,
)

OUTPUT_ROOT = PROJECT_ROOT / "persistent-data/transport_validation/protocol_v5_transition/bridge"

TRAFFIC_DURATION_SECONDS = 180.0
STABILIZATION_SECONDS = 60.0
SCORED_WINDOW = (60.0, 180.0)
DECISION_TIMESTAMPS = [60, 90, 120, 150]
CONNECTION_MODE = "fresh"
TIMEOUT_SECONDS = 5.0
MAX_IN_FLIGHT = 400


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-name", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--expected-schedule-sha256", required=True)
    parser.add_argument("--replicas", required=True, type=int)
    parser.add_argument("--transport", required=True, choices=["old", "v5"])
    parser.add_argument("--repeat", required=True, type=int)
    parser.add_argument("--output-root", required=False, default=None,
                         help="Override OUTPUT_ROOT (relative to PROJECT_ROOT). Default: persistent-data/transport_validation/protocol_v5_transition/bridge")
    args = parser.parse_args()

    schedule_path = PROJECT_ROOT / args.schedule
    actual_hash = sha256_of(schedule_path)
    if actual_hash != args.expected_schedule_sha256:
        print(f"SCHEDULE HASH MISMATCH: expected {args.expected_schedule_sha256}, got {actual_hash}")
        return 2

    output_root = (PROJECT_ROOT / args.output_root) if args.output_root else OUTPUT_ROOT

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"NON_OFFICIAL_PROTOCOL_TRANSITION_VALIDATION_bridge_{args.cell_name}_{args.transport}_repeat{args.repeat}_{ts}"
    out_dir = output_root / experiment_id
    (out_dir / "cluster").mkdir(parents=True)
    (out_dir / "routing").mkdir(parents=True)

    result: Dict[str, Any] = {
        "scientific_role": "NON_OFFICIAL_PROTOCOL_TRANSITION_VALIDATION",
        "formal_calibration_eligible": False,
        "campaign_ledger_eligible": False,
        "model_fitting_eligible": False,
        "experiment_id": experiment_id,
        "gate": "B_cross_epoch_bridge",
        "cell_name": args.cell_name,
        "schedule_path": args.schedule,
        "schedule_sha256": actual_hash,
        "replicas": args.replicas,
        "transport": args.transport,
        "repeat": args.repeat,
        "traffic_duration_seconds": TRAFFIC_DURATION_SECONDS,
        "connection_mode": CONNECTION_MODE,
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_in_flight": MAX_IN_FLIGHT,
        "invalidity_reasons": [],
    }

    pre_run_context = capture_minikube_resource_context()
    (out_dir / "cluster" / "minikube_resource_context_pre_run.json").write_text(json.dumps(pre_run_context, indent=2))

    print(f"=== [{experiment_id}] HPA isolation ===")
    delete_hpa()
    time.sleep(2)
    hpa_absent = verify_no_hpa_targets_deployment()
    if not hpa_absent:
        result["invalidity_reasons"].append("hpa_unexpectedly_active")

    print(f"=== Scaling to {args.replicas} replicas ===")
    replica_state = scale_and_wait(args.replicas)
    if not all(v == args.replicas for v in replica_state.values()):
        result["invalidity_reasons"].append(f"wrong_replica_count_achieved: {replica_state}")
    result["pod_names_pre_run"] = sorted(get_pod_names())

    print(f"=== {STABILIZATION_SECONDS:.0f}s stabilization (no traffic) ===")
    time.sleep(STABILIZATION_SECONDS)

    events_pre = fetch_events_json()
    (out_dir / "cluster" / "kubernetes_events_pre.json").write_text(json.dumps(events_pre, default=str))
    endpoints_pre = fetch_loadtest_endpoints_json()
    (out_dir / "cluster" / "loadtest_endpoints_pre_traffic.json").write_text(json.dumps(endpoints_pre, default=str))

    # -------------------------------------------------------------------
    # Transport launch
    # -------------------------------------------------------------------
    transport_lifecycle: Dict[str, Any] = {"transport": args.transport}
    target_url: Optional[str] = None
    relay_name: Optional[str] = None
    tunnel_proc: Optional[subprocess.Popen] = None

    if args.transport == "v5":
        print("=== V5 transport (Docker-network raw TCP relay) ===")
        launch = start_loadtest_transport(experiment_id)
        transport_lifecycle.update(launch)
        if "error" in launch:
            result["invalidity_reasons"].append(f"wrong_traffic_path_used: {launch['error']}")
        else:
            target_url = launch["target_url"]
            relay_name = launch["container_name"]
    else:
        print("=== Pre-V5 transport (minikube service --url SSH tunnel) ===")
        tunnel_proc, target_url = start_loadtest_tunnel(experiment_id)
        transport_lifecycle.update({
            "command": ["minikube", "service", "demo-app-loadtest", "-n", APP_NS, "--url"],
            "pid": tunnel_proc.pid,
            "destination_url": target_url or None,
        })
        if not target_url:
            result["invalidity_reasons"].append("wrong_traffic_path_used: could not resolve SSH tunnel URL")

    if not target_url:
        target_url = "http://localhost:1"

    tunnel_port = _extract_port_from_url(target_url)

    prom_ready = _run(["curl", "-s", "-f", "http://localhost:9090/-/ready"], timeout=5).returncode == 0
    started_prom = False
    prom_proc: Optional[subprocess.Popen] = None
    if not prom_ready:
        prom_proc = subprocess.Popen(
            ["kubectl", "port-forward", "-n", MONITORING_NS, "svc/kube-prom-stack-kube-prome-prometheus", "9090:9090"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        started_prom = True
        time.sleep(3)

    collector_duration = TRAFFIC_DURATION_SECONDS + 30

    print("=== Starting collectors ===")
    metrics_proc = subprocess.Popen(
        [VENV_PYTHON, "-m", "collectors.metrics_collector", "--experiment-id", experiment_id, "--mode", "test",
         "--traffic-profile", args.cell_name, "--duration-seconds", str(collector_duration), "--seed", "42"],
    )
    readiness_proc = subprocess.Popen(
        [VENV_PYTHON, "-m", "scripts.pod_readiness_collector", "--experiment-dir", str(out_dir / "cluster"),
         "--duration-seconds", str(collector_duration)],
    )
    counter_proc = subprocess.Popen(
        [VENV_PYTHON, "-m", "scripts.per_pod_counter_collector", "--experiment-dir", str(out_dir / "routing"),
         "--duration-seconds", str(collector_duration)],
    )
    time.sleep(1)

    periodic_samples: List[Dict[str, Any]] = []
    stop_sampling = threading.Event()

    def _sample():
        start_mono = time.monotonic()
        while not stop_sampling.wait(10):
            endpoints = fetch_loadtest_endpoints_json()
            if args.transport == "v5" and relay_name:
                running, exit_code = check_relay_container_state(relay_name)
            elif args.transport == "old" and tunnel_proc is not None:
                running, exit_code = (tunnel_proc.poll() is None), tunnel_proc.returncode
            else:
                running, exit_code = None, None
            listener_present = check_local_listener_present(tunnel_port) if tunnel_port else None
            periodic_samples.append({
                "elapsed_seconds_since_traffic_start": round(time.monotonic() - start_mono, 1),
                "sampled_at_utc": utc_now(),
                "tunnel_process_alive": running,
                "tunnel_process_exit_code": exit_code,
                "local_listener_present": listener_present,
                "listener_port": tunnel_port,
                "loadtest_endpoints": parse_endpoints_ready_count(endpoints) if "_error" not in endpoints else None,
                "endpoint_query_error": endpoints.get("_error"),
            })

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()

    print(f"=== [{args.transport}] Traffic execution: {args.cell_name} (schedule REPLAY) ===")
    traffic_start_epoch = time.time()
    gen = subprocess.run(
        [VENV_PYTHON, "-m", "loadgen.traffic_generator",
         "--profile", str((schedule_path.parent / f"{args.cell_name}.yaml")),
         "--experiment-id", experiment_id, "--target-url", target_url,
         "--replay-schedule", str(schedule_path), "--arrival-process", "deterministic",
         "--connection-mode", CONNECTION_MODE, "--timeout", str(TIMEOUT_SECONDS),
         "--max-in-flight", str(MAX_IN_FLIGHT)],
    )
    traffic_end_epoch = time.time()
    if gen.returncode != 0:
        result["invalidity_reasons"].append(f"traffic_generator_exit_{gen.returncode}")

    stop_sampling.set()
    sampler.join(timeout=5)
    transport_lifecycle["periodic_samples_during_traffic"] = periodic_samples
    transport_lifecycle["traffic_start_epoch"] = traffic_start_epoch
    transport_lifecycle["traffic_end_epoch"] = traffic_end_epoch

    if args.transport == "v5" and relay_name:
        running_end, exit_end = check_relay_container_state(relay_name)
    elif args.transport == "old" and tunnel_proc is not None:
        running_end, exit_end = (tunnel_proc.poll() is None), tunnel_proc.returncode
    else:
        running_end, exit_end = None, None
    transport_lifecycle["alive_at_traffic_end"] = running_end
    transport_lifecycle["exit_code_at_traffic_end"] = exit_end
    transport_lifecycle["local_listener_present_at_traffic_end"] = check_local_listener_present(tunnel_port) if tunnel_port else None
    transport_lifecycle["listener_owner_at_traffic_end"] = resolve_local_listener_owner(tunnel_port) if tunnel_port else None

    result["pod_names_post_run"] = sorted(get_pod_names())
    events_post = fetch_events_json()
    (out_dir / "cluster" / "kubernetes_events_post.json").write_text(json.dumps(events_post, default=str))
    endpoints_post = fetch_loadtest_endpoints_json()
    (out_dir / "cluster" / "loadtest_endpoints_post_traffic.json").write_text(json.dumps(endpoints_post, default=str))

    print("=== Waiting for collectors ===")
    metrics_exit = metrics_proc.wait(timeout=collector_duration + 60)
    readiness_exit = readiness_proc.wait(timeout=collector_duration + 60)
    counter_exit = counter_proc.wait(timeout=collector_duration + 60)
    if metrics_exit != 0:
        result["invalidity_reasons"].append(f"metrics_collector_exit_{metrics_exit}")
    if readiness_exit != 0 or counter_exit != 0:
        result["invalidity_reasons"].append("readiness_or_counter_collector_nonzero_exit")

    print("=== Cleanup: transport ===")
    if args.transport == "v5" and relay_name:
        logs = _run(["docker", "logs", relay_name], timeout=15)
        (out_dir / "cluster" / "relay_docker_logs.txt").write_text((logs.stdout or "") + (logs.stderr or ""))
        _run(["docker", "rm", "-f", relay_name], timeout=15)
    elif args.transport == "old" and tunnel_proc is not None:
        if tunnel_proc.poll() is None:
            tunnel_proc.terminate()
            try:
                tunnel_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tunnel_proc.kill()
    if started_prom and prom_proc is not None:
        prom_proc.terminate()

    (out_dir / "cluster" / "loadtest_transport_lifecycle.json").write_text(json.dumps(transport_lifecycle, indent=2, default=str))

    print("=== Restoring HPA, waiting for natural convergence ===")
    restore_hpa()
    time.sleep(2)
    wait_timeout = restoration_wait_seconds(args.replicas)
    idle_state = wait_for_idle_one_replica(timeout_seconds=wait_timeout)
    result["idle_state_after_restore"] = idle_state
    result["idle_converged"] = all(v == 1 for v in idle_state.values())

    # -------------------------------------------------------------------
    # Post-processing: load artifacts, route-integrity classification,
    # decision-aligned extraction -- reusing the same pure pipeline the
    # official harness uses.
    # -------------------------------------------------------------------
    traffic_dir = PROJECT_ROOT / "persistent-data/traffic" / experiment_id
    metrics_dir = PROJECT_ROOT / "persistent-data/metrics" / experiment_id

    metrics_rows: List[Dict[str, Any]] = []
    if (metrics_dir / "metrics.csv").exists():
        metrics_rows = load_metrics_csv(metrics_dir / "metrics.csv")

    counter_rows: List[Dict[str, Any]] = []
    counters_path = out_dir / "routing" / "per_pod_request_counters.csv"
    if counters_path.exists():
        counter_rows = load_per_pod_counters(counters_path)

    requests_rows: List[Dict[str, Any]] = []
    if (traffic_dir / "requests.csv").exists():
        requests_rows = load_requests_csv(traffic_dir / "requests.csv")

    route_integrity_result: Dict[str, Any] = {"classification": None}
    if (traffic_dir / "traffic_summary.csv").exists():
        summary_rows = load_traffic_summary_csv(traffic_dir / "traffic_summary.csv")
        all_runs = lib.find_all_zero_success_runs(summary_rows)
        material_intervals = [r for r in all_runs if lib.is_interval_material(r[0], r[1], TRAFFIC_DURATION_SECONDS, SCORED_WINDOW)]
        traffic_start_epoch_for_windows = transport_lifecycle.get("traffic_start_epoch")
        evidence_windows: List[Dict[str, Any]] = []
        for run_start, run_end in material_intervals:
            for w_start, w_end in lib.partition_into_evidence_windows(run_start, run_end):
                observer = lib.aggregate_observer_evidence_for_window(w_start, w_end, periodic_samples)
                delivered = lib.compute_delivered_request_evidence_windows(
                    window_start_elapsed=w_start, window_end_elapsed=w_end,
                    traffic_start_epoch=traffic_start_epoch_for_windows,
                    requests_rows=requests_rows, metrics_rows=metrics_rows, counter_rows=counter_rows,
                )
                verdict = lib.classify_route_integrity_window(
                    delivered_request_evidence=delivered["delivered_request_evidence"],
                    tunnel_process_alive=observer["tunnel_process_alive"],
                    local_listener_present=observer["local_listener_present"],
                )
                evidence_windows.append({
                    "material_interval": [run_start, run_end], "window_start_elapsed": w_start, "window_end_elapsed": w_end,
                    "tunnel_process_alive": observer["tunnel_process_alive"], "local_listener_present": observer["local_listener_present"],
                    "delivered_request_evidence": delivered["delivered_request_evidence"],
                    "window_route_status": verdict["window_route_status"], "reason": verdict["reason"],
                })
        route_integrity_result = lib.classify_route_integrity(material_intervals=material_intervals, evidence_windows=evidence_windows)
        (out_dir / "cluster" / "loadtest_route_integrity_timeline.json").write_text(
            json.dumps({"all_zero_success_runs": all_runs, "material_intervals": material_intervals, "evidence_windows": evidence_windows}, indent=2, default=str)
        )
        if route_integrity_result["classification"] == lib.ROUTE_INTEGRITY_FAILED:
            result["invalidity_reasons"].append("wrong_traffic_path_used: route FAILED")
        elif route_integrity_result["classification"] == lib.ROUTE_INTEGRITY_UNVERIFIED:
            result["invalidity_reasons"].append("official_workload_route_integrity_unverified")

    result["route_integrity"] = {
        "classification": route_integrity_result.get("classification"),
        "reason": route_integrity_result.get("reason"),
        "material_intervals": route_integrity_result.get("material_intervals", []),
    }
    result["structurally_valid"] = (len(result["invalidity_reasons"]) == 0) and (route_integrity_result.get("classification") != lib.ROUTE_INTEGRITY_FAILED) and (route_integrity_result.get("classification") != lib.ROUTE_INTEGRITY_UNVERIFIED)

    # Traffic outcome totals
    traffic_meta = {}
    if (traffic_dir / "metadata.json").exists():
        traffic_meta = json.loads((traffic_dir / "metadata.json").read_text())
    result["traffic_outcome_totals"] = traffic_meta.get("outcome_totals")
    result["client_latency_stats"] = traffic_meta.get("client_latency_stats")
    result["total_offered"] = traffic_meta.get("total_offered")
    result["total_sent"] = traffic_meta.get("total_sent")

    # Decision-aligned extraction (readiness authoritative + metrics)
    readiness_rows: List[Dict[str, Any]] = []
    readiness_path = out_dir / "cluster" / "pod_readiness_history.csv"
    if readiness_path.exists():
        with open(readiness_path) as f:
            for row in csv.DictReader(f):
                row["elapsed_seconds"] = float(row["elapsed_seconds"])
                row["restart_count"] = int(row["restart_count"])
                readiness_rows.append(row)

    decision_points = []
    for target in DECISION_TIMESTAMPS:
        sample = lib.select_decision_sample(float(target), metrics_rows, staleness_threshold_seconds=30.0) if metrics_rows else None
        ready_count = authoritative_ready_count_at_target(readiness_rows, float(target)) if readiness_rows else None
        point: Dict[str, Any] = {"target_decision_timestamp_s": target, "ready_count_authoritative": ready_count}
        if sample is not None:
            point.update({
                "selected_source_sample_elapsed_s": sample.get("elapsed_seconds"),
                "request_rate_rps": sample.get("request_rate_rps"),
                "cpu_cores": sample.get("cpu_cores"),
                "latency_p95_seconds": sample.get("latency_p95_seconds"),
                "memory_bytes": sample.get("memory_bytes"),
                "requests_in_progress": sample.get("requests_in_progress"),
                "error_rate_rps": sample.get("error_rate_rps"),
                "replicas_current_prometheus_sourced": sample.get("replicas_current"),
            })
        decision_points.append(point)
    result["decision_points"] = decision_points

    restarted = []
    if len(readiness_rows) >= 2:
        first_ts = min(r["elapsed_seconds"] for r in readiness_rows)
        last_ts = max(r["elapsed_seconds"] for r in readiness_rows)
        first_rows = [r for r in readiness_rows if r["elapsed_seconds"] == first_ts]
        last_rows = [r for r in readiness_rows if r["elapsed_seconds"] == last_ts]
        restarted = lib.restarted_pods_between(first_rows, last_rows)
    result["restarted_pods_during_run"] = restarted

    result["completed_at_utc"] = utc_now()
    (out_dir / "metadata.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({"experiment_id": experiment_id, "structurally_valid": result["structurally_valid"], "route_integrity": result["route_integrity"]["classification"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
