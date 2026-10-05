#!/usr/bin/env python3
"""scripts/summarize_calibration_level.py

Canonical summary tool for empirical single-replica capacity calibration runs.
Parses persistent artifacts directly:
- persistent-data/traffic/<experiment-id>/requests.csv
- persistent-data/metrics/<experiment-id>/metrics.csv
- metadata.json files for provenance and runtime context.
"""

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional


def compute_percentiles(sorted_values: List[float], percentiles: List[float]) -> Dict[str, float]:
    """Compute percentiles using standard linear interpolation (continuous estimation)."""
    if not sorted_values:
        return {}
    n = len(sorted_values)
    results = {}
    for p in percentiles:
        key = f"p{int(p * 100)}"
        if n == 1:
            results[key] = sorted_values[0]
            continue
        idx = p * (n - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        fraction = idx - lower
        if lower == upper:
            results[key] = sorted_values[lower]
        else:
            results[key] = sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])
    return results


def compute_rank_percentiles(sorted_values: List[float], percentiles: List[float]) -> Dict[str, float]:
    """Compute rank-based percentiles using integer indexing (int(n * p)), matching loadgen metadata."""
    if not sorted_values:
        return {}
    n = len(sorted_values)
    results = {}
    for p in percentiles:
        key = f"p{int(p * 100)}_rank"
        idx = min(int(n * p), n - 1)
        results[key] = sorted_values[idx]
    return results


def analyze_level(experiment_id: str, project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Parse raw traffic and metrics artifacts and compute canonical calibration statistics.

    Statistical Methodology:
    1. Client-Side Request Metrics:
       - Evaluated over ALL requests in the authoritative workload window (requests.csv).
       - Client observations do not experience Prometheus rate lookback constraints.
       - Reports standard continuous percentiles (linear interpolation) and rank-based percentiles.
    2. Prometheus Server-Side Metrics:
       - Metrics derived from rate(...[1m]) queries have a 60-second moving lookback.
       - 'steady_window' defines samples where elapsed_seconds >= 60.0 (fully saturated lookback).
       - 'whole_run' reports metrics across all valid samples for full transparency.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    traffic_dir = project_root / "persistent-data" / "traffic" / experiment_id
    metrics_dir = project_root / "persistent-data" / "metrics" / experiment_id

    if not traffic_dir.exists():
        raise FileNotFoundError(f"Traffic directory not found: {traffic_dir}")
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    # 1. Parse traffic metadata & requests.csv
    with open(traffic_dir / "metadata.json", "r", encoding="utf-8") as f:
        t_meta = json.load(f)

    with open(traffic_dir / "requests.csv", "r", encoding="utf-8") as f:
        requests = list(csv.DictReader(f))

    # 2. Parse metrics metadata & metrics.csv
    m_meta = {}
    if (metrics_dir / "metadata.json").exists():
        with open(metrics_dir / "metadata.json", "r", encoding="utf-8") as f:
            m_meta = json.load(f)

    with open(metrics_dir / "metrics.csv", "r", encoding="utf-8") as f:
        metrics_samples = list(csv.DictReader(f))

    # --- Client-Side Request Analysis (All requests in workload window) ---
    actual_runtime = float(t_meta.get("actual_runtime_seconds", 90.0))
    requested_duration = float(t_meta.get("requested_duration_seconds", 90.0))
    total_offered = int(t_meta.get("total_offered", len(requests)))
    sent_requests = [r for r in requests if r.get("outcome") != "dropped_overload"]
    total_sent = len(sent_requests)

    offered_rps = round(total_offered / requested_duration, 4) if requested_duration > 0 else 0.0
    dispatch_rate_workload_window = round(total_sent / requested_duration, 4) if requested_duration > 0 else 0.0
    process_runtime_send_rps = round(total_sent / actual_runtime, 4) if actual_runtime > 0 else 0.0
    achieved_send_rps = dispatch_rate_workload_window

    success_requests = [r for r in requests if r.get("outcome") == "success_2xx"]
    success_count = len(success_requests)
    client_4xx = sum(1 for r in requests if r.get("outcome") == "client_error_4xx")
    server_5xx = sum(1 for r in requests if r.get("outcome") == "server_error_5xx")
    timeouts = sum(1 for r in requests if r.get("outcome") == "timeout")
    conn_errors = sum(1 for r in requests if r.get("outcome") == "connection_error")
    dropped_overload = sum(1 for r in requests if r.get("outcome") == "dropped_overload")
    success_rate = round(success_count / total_offered, 4) if total_offered > 0 else 0.0

    # Client latency statistics
    latencies = sorted([
        float(r["client_latency_seconds"])
        for r in success_requests
        if r.get("client_latency_seconds") not in ("", None)
    ])
    linear_pcts = compute_percentiles(latencies, [0.50, 0.95, 0.99])
    rank_pcts = compute_rank_percentiles(latencies, [0.50, 0.95, 0.99])

    # Dispatch lag statistics
    lags = [
        float(r["dispatch_lag_seconds"])
        for r in requests
        if r.get("dispatch_lag_seconds") not in ("", None)
    ]
    mean_lag = round(sum(lags) / len(lags), 6) if lags else 0.0
    max_lag = round(max(lags), 6) if lags else 0.0

    # --- Prometheus Server-Side Telemetry Analysis ---
    valid_samples = [s for s in metrics_samples if s.get("sample_valid", "").lower() == "true"]
    steady_samples = [s for s in valid_samples if float(s.get("elapsed_seconds", 0.0)) >= 60.0]

    # Whole-run statistics
    all_cpu = [float(s["cpu_cores"]) for s in valid_samples if s.get("cpu_cores") not in ("", None)]
    all_mem = [float(s["memory_bytes"]) / (1024 * 1024) for s in valid_samples if s.get("memory_bytes") not in ("", None)]
    all_req_rate = [float(s["request_rate_rps"]) for s in valid_samples if s.get("request_rate_rps") not in ("", None)]
    all_p95 = [float(s["latency_p95_seconds"]) for s in valid_samples if s.get("latency_p95_seconds") not in ("", None)]

    whole_run_mean_cpu = round(sum(all_cpu) / len(all_cpu), 4) if all_cpu else None
    whole_run_max_cpu = round(max(all_cpu), 4) if all_cpu else None
    whole_run_mean_req_rate = round(sum(all_req_rate) / len(all_req_rate), 4) if all_req_rate else None
    whole_run_mean_mem = round(sum(all_mem) / len(all_mem), 2) if all_mem else None
    whole_run_max_mem = round(max(all_mem), 2) if all_mem else None

    # Steady-window statistics (elapsed_seconds >= 60.0)
    steady_cpu = [float(s["cpu_cores"]) for s in steady_samples if s.get("cpu_cores") not in ("", None)]
    steady_mem = [float(s["memory_bytes"]) / (1024 * 1024) for s in steady_samples if s.get("memory_bytes") not in ("", None)]
    steady_req_rate = [float(s["request_rate_rps"]) for s in steady_samples if s.get("request_rate_rps") not in ("", None)]
    steady_err_rate = [float(s["error_rate_rps"]) for s in steady_samples if s.get("error_rate_rps") not in ("", None)]
    steady_p50 = [float(s["latency_p50_seconds"]) for s in steady_samples if s.get("latency_p50_seconds") not in ("", None)]
    steady_p95 = [float(s["latency_p95_seconds"]) for s in steady_samples if s.get("latency_p95_seconds") not in ("", None)]
    steady_p99 = [float(s["latency_p99_seconds"]) for s in steady_samples if s.get("latency_p99_seconds") not in ("", None)]
    steady_replicas = [int(float(s["replicas_current"])) for s in steady_samples if s.get("replicas_current") not in ("", None)]

    steady_mean_cpu = round(sum(steady_cpu) / len(steady_cpu), 4) if steady_cpu else None
    steady_max_cpu = round(max(steady_cpu), 4) if steady_cpu else None
    steady_mean_req_rate = round(sum(steady_req_rate) / len(steady_req_rate), 4) if steady_req_rate else None
    steady_mean_err_rate = round(sum(steady_err_rate) / len(steady_err_rate), 4) if steady_err_rate else 0.0
    steady_server_p50 = round(sum(steady_p50) / len(steady_p50), 4) if steady_p50 else None
    steady_server_p95 = round(sum(steady_p95) / len(steady_p95), 4) if steady_p95 else None
    steady_server_p99 = round(sum(steady_p99) / len(steady_p99), 4) if steady_p99 else None
    steady_mean_mem = round(sum(steady_mem) / len(steady_mem), 2) if steady_mem else None
    steady_max_mem = round(max(steady_mem), 2) if steady_mem else None

    summary = {
        "experiment_id": experiment_id,
        "profile": t_meta.get("profile"),
        "requested_duration_seconds": t_meta.get("requested_duration_seconds"),
        "actual_runtime_seconds": actual_runtime,
        "traffic": {
            "offered_rps": offered_rps,
            "offered_requests": total_offered,
            "dispatched_requests": total_sent,
            "dispatch_rate_workload_window": dispatch_rate_workload_window,
            "successful_requests": success_count,
            "total_offered": total_offered,
            "total_sent": total_sent,
            "achieved_send_rps": achieved_send_rps,
            "process_runtime_send_rps": process_runtime_send_rps,
            "success_count": success_count,
            "success_rate": success_rate,
            "client_4xx": client_4xx,
            "server_5xx": server_5xx,
            "timeouts": timeouts,
            "conn_errors": conn_errors,
            "dropped_overload": dropped_overload,
            "client_latency": {
                "p50_seconds": round(linear_pcts.get("p50", 0.0), 6),
                "p95_seconds": round(linear_pcts.get("p95", 0.0), 6),
                "p99_seconds": round(linear_pcts.get("p99", 0.0), 6),
                "p50_rank_seconds": round(rank_pcts.get("p50_rank", 0.0), 6),
                "p95_rank_seconds": round(rank_pcts.get("p95_rank", 0.0), 6),
                "p99_rank_seconds": round(rank_pcts.get("p99_rank", 0.0), 6),
            },
            "dispatch_lag": {
                "mean_seconds": mean_lag,
                "max_seconds": max_lag,
            },
        },
        "prometheus_steady_window": {
            "window_definition": "elapsed_seconds >= 60.0 (fully stabilized 1m lookback)",
            "sample_count": len(steady_samples),
            "observed_request_rate_rps": steady_mean_req_rate,
            "observed_error_rate_rps": steady_mean_err_rate,
            "mean_observed_cpu_rate": steady_mean_cpu,
            "max_observed_cpu_rate": steady_max_cpu,
            "cpu_limit_utilization_pct": round((steady_mean_cpu / 0.5) * 100, 2) if steady_mean_cpu else None,
            "server_p50_seconds": steady_server_p50,
            "server_p95_seconds": steady_server_p95,
            "server_p99_seconds": steady_server_p99,
            "mean_memory_mb": steady_mean_mem,
            "max_memory_mb": steady_max_mem,
            "replicas_current": steady_replicas[-1] if steady_replicas else 1,
        },
        "prometheus_whole_run": {
            "window_definition": "all valid samples (t = 0.4s to 80.0s)",
            "sample_count": len(valid_samples),
            "mean_observed_cpu_rate": whole_run_mean_cpu,
            "max_observed_cpu_rate": whole_run_max_cpu,
            "mean_request_rate_rps": whole_run_mean_req_rate,
            "server_p95_max_seconds": round(max(all_p95), 4) if all_p95 else None,
            "mean_memory_mb": whole_run_mean_mem,
            "max_memory_mb": whole_run_max_mem,
        },
        "metrics_metadata": m_meta,
        # Top-level backward compatibility aliases for existing tooling
        "total_offered": total_offered,
        "total_sent": total_sent,
        "achieved_send_rps": achieved_send_rps,
        "success_count": success_count,
        "success_rate": success_rate,
        "client_4xx": client_4xx,
        "server_5xx": server_5xx,
        "timeouts": timeouts,
        "conn_errors": conn_errors,
        "dropped_overload": dropped_overload,
        "client_latency": {
            "p50_seconds": round(linear_pcts.get("p50", 0.0), 6),
            "p95_seconds": round(linear_pcts.get("p95", 0.0), 6),
            "p99_seconds": round(linear_pcts.get("p99", 0.0), 6),
            "p95_rank_seconds": round(rank_pcts.get("p95_rank", 0.0), 6),
        },
        "dispatch_lag": {
            "mean_seconds": mean_lag,
            "max_seconds": max_lag,
        },
        "server_metrics": {
            "mean_observed_cpu_rate": steady_mean_cpu,
            "max_observed_cpu_rate": whole_run_max_cpu,
            "cpu_limit_utilization_pct": round((steady_mean_cpu / 0.5) * 100, 2) if steady_mean_cpu else None,
            "mean_memory_mb": steady_mean_mem,
            "max_memory_mb": steady_max_mem,
            "mean_request_rate_rps": steady_mean_req_rate,
            "mean_error_rate_rps": steady_mean_err_rate,
            "server_p50_mean_seconds": steady_server_p50,
            "server_p95_steady_seconds": steady_server_p95,
            "server_p99_mean_seconds": steady_server_p99,
            "replicas_current": steady_replicas[-1] if steady_replicas else 1,
        },
    }
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: summarize_calibration_level.py <experiment_id>")
        sys.exit(1)
    res = analyze_level(sys.argv[1])
    print(json.dumps(res, indent=2))
