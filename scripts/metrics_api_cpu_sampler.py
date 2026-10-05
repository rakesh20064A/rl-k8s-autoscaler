#!/usr/bin/env python3
"""scripts/metrics_api_cpu_sampler.py

Phase 11C.4D: minimal diagnostic sampler for the raw Kubernetes resource
Metrics API (metrics.k8s.io/v1beta1), the same data source the HPA controller
itself reads for a Resource CPU metric -- distinct from, and not a substitute
for, Prometheus's own `container_cpu_usage_seconds_total` (cAdvisor-sourced,
useful for reproducible server-side characterization but NOT the HPA's own
input).

Metrics Server here is configured at --metric-resolution=15s: polling this
script's target endpoint faster than that only re-reads the same cached
observation. Each poll's underlying observation is uniquely identified by the
API response's own `timestamp` field (when the metric window was actually
computed), NOT by this script's own poll wall-clock time. Repeated polls that
return the same `timestamp` are recorded (for audit) but explicitly flagged
as NOT independent observations and excluded from summary statistics.

Read-only. Performs no scaling, no HPA/Deployment modification.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_cpu_quantity_to_millicores(qty: Optional[str]) -> Optional[float]:
    """Parse a Kubernetes CPU quantity string as returned by metrics.k8s.io
    ('5988839n' nanocores, '12345u' microcores, '42m' millicores, or a bare
    core count like '0.5') into millicores."""
    if qty is None or qty == "":
        return None
    qty = qty.strip()
    try:
        if qty.endswith("n"):
            return float(qty[:-1]) / 1_000_000.0
        if qty.endswith("u"):
            return float(qty[:-1]) / 1_000.0
        if qty.endswith("m"):
            return float(qty[:-1])
        return float(qty) * 1000.0
    except ValueError:
        return None


def fetch_pod_metrics(namespace: str, pod: str, timeout_seconds: float = 5.0) -> Optional[Dict[str, Any]]:
    """Read-only call to the raw resource Metrics API for one pod. Returns the
    parsed JSON response, or None on any failure (never fabricates a value)."""
    try:
        result = subprocess.run(
            [
                "kubectl", "get", "--raw",
                f"/apis/metrics.k8s.io/v1beta1/namespaces/{namespace}/pods/{pod}",
            ],
            capture_output=True, text=True, timeout=timeout_seconds,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except Exception:
        return None


def extract_cpu_sample(raw: Dict[str, Any], container_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Extract the CPU usage sample for one container from a PodMetrics response."""
    containers = raw.get("containers") or []
    target = None
    for c in containers:
        if container_name is None or c.get("name") == container_name:
            target = c
            break
    if target is None:
        return None
    cpu_raw = (target.get("usage") or {}).get("cpu")
    cpu_millicores = parse_cpu_quantity_to_millicores(cpu_raw)
    return {
        "metrics_api_timestamp": raw.get("timestamp"),
        "metrics_api_window": raw.get("window"),
        "pod_name": raw.get("metadata", {}).get("name"),
        "container_name": target.get("name"),
        "cpu_quantity_raw": cpu_raw,
        "cpu_millicores": cpu_millicores,
    }


def run_sampler(
    namespace: str,
    pod: str,
    experiment_id: str,
    duration_seconds: float,
    poll_interval_seconds: float,
    cpu_request_millicores: float,
    output_base_dir: str,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / output_base_dir / experiment_id
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    samples_path = output_dir / "metrics_api_samples.csv"
    metadata_path = output_dir / "metadata.json"

    fieldnames = [
        "poll_index", "poll_wallclock_utc", "metrics_api_timestamp", "metrics_api_window",
        "pod_name", "container_name", "cpu_quantity_raw", "cpu_millicores",
        "cpu_utilization_pct_of_request", "is_new_underlying_sample",
    ]

    all_rows: List[Dict[str, Any]] = []
    seen_timestamps = set()
    start_mono = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()
    poll_index = 0

    with open(samples_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        while (time.monotonic() - start_mono) < duration_seconds:
            poll_wallclock_utc = datetime.now(timezone.utc).isoformat()
            raw = fetch_pod_metrics(namespace, pod)
            if raw is not None:
                sample = extract_cpu_sample(raw)
                if sample is not None:
                    ts = sample["metrics_api_timestamp"]
                    is_new = ts not in seen_timestamps
                    if is_new:
                        seen_timestamps.add(ts)
                    util_pct = (
                        round((sample["cpu_millicores"] / cpu_request_millicores) * 100.0, 3)
                        if sample["cpu_millicores"] is not None else None
                    )
                    row = {
                        "poll_index": poll_index,
                        "poll_wallclock_utc": poll_wallclock_utc,
                        "metrics_api_timestamp": ts,
                        "metrics_api_window": sample["metrics_api_window"],
                        "pod_name": sample["pod_name"],
                        "container_name": sample["container_name"],
                        "cpu_quantity_raw": sample["cpu_quantity_raw"],
                        "cpu_millicores": sample["cpu_millicores"],
                        "cpu_utilization_pct_of_request": util_pct,
                        "is_new_underlying_sample": is_new,
                    }
                    writer.writerow(row)
                    all_rows.append(row)
            poll_index += 1
            time.sleep(poll_interval_seconds)

    end_utc = datetime.now(timezone.utc).isoformat()

    unique_rows = [r for r in all_rows if r["is_new_underlying_sample"] and r["cpu_millicores"] is not None]
    unique_millicores = [r["cpu_millicores"] for r in unique_rows]

    summary: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "namespace": namespace,
        "pod": pod,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "requested_duration_seconds": duration_seconds,
        "poll_interval_seconds": poll_interval_seconds,
        "cpu_request_millicores": cpu_request_millicores,
        "total_polls": poll_index,
        "total_successful_polls": len(all_rows),
        "unique_underlying_metrics_api_samples": len(unique_rows),
        "note": (
            "unique_underlying_metrics_api_samples counts only polls whose "
            "metrics.k8s.io 'timestamp' field differs from every prior poll "
            "in this run -- Metrics Server updates roughly every 15s "
            "regardless of poll_interval_seconds, so repeated polls between "
            "updates share the same timestamp and are NOT independent "
            "observations."
        ),
    }
    if unique_millicores:
        summary["cpu_millicores_median"] = statistics.median(unique_millicores)
        summary["cpu_millicores_min"] = min(unique_millicores)
        summary["cpu_millicores_max"] = max(unique_millicores)
        summary["cpu_utilization_pct_median"] = round(
            (statistics.median(unique_millicores) / cpu_request_millicores) * 100.0, 3
        )
        for boundary_name, boundary_millicores in (("220pct_tol", 242.0), ("230pct_tol", 253.0), ("250pct_tol", 275.0)):
            above = sum(1 for v in unique_millicores if v > boundary_millicores)
            summary[f"fraction_above_{boundary_name}_{int(boundary_millicores)}m"] = round(
                above / len(unique_millicores), 3
            )

    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--poll-interval-seconds", type=float, default=3.0)
    parser.add_argument("--cpu-request-millicores", type=float, default=100.0)
    parser.add_argument("--output-base-dir", default="persistent-data/hpa_mapping")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_sampler(
        namespace=args.namespace,
        pod=args.pod,
        experiment_id=args.experiment_id,
        duration_seconds=args.duration_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        cpu_request_millicores=args.cpu_request_millicores,
        output_base_dir=args.output_base_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
