"""scripts/per_pod_counter_collector.py

Phase 15B.3A: actively persists RAW per-pod `demo_app_requests_total`
counter samples throughout a calibration attempt, at approximately the
frozen 10-second cadence -- the standard aggregate `metrics_collector.py`
is not sufficient for CORR-35 routing reconstruction, which needs each
pod's own raw counter trajectory, never a `rate()`-smoothed aggregate.

PromQL: `demo_app_requests_total{namespace="...",route="/work",job="demo-app"}`
(never `job="demo-app-loadtest"` -- CORR-22 sec.4's documented double-scrape
caveat). Each pod exposes its own counter, so this instant query already
returns one time series per pod; no `by (pod)` aggregation is applied here
because none is needed (or wanted -- aggregation would collapse the very
per-pod distinction this collector exists to capture).

Phase 15B.3C.3 (CORR-38 sec.C) hardening: Phase 15B.3C.2's
`detect_pod_series_gap` only detects a pod PREVIOUSLY returned by
Prometheus that has since vanished -- it cannot detect a pod known to
Kubernetes but NEVER once returned (absent from the very first tick, or
absent for the entire run), which is exactly what happened with `6lld4`
in index2. Each tick now ALSO lists the authoritative currently-known
Kubernetes pod population (the same `kubectl get pods -o json` source and
label-selector convention already used by `pod_readiness_collector.py`)
and records, per known pod, whether Prometheus returned its series --
persisted to `pod_counter_observability_history.csv`, a snapshot
comparison rather than a previous-tick-relative one. A whole-Prometheus-
query failure, a Kubernetes-listing failure, and a single known pod's
series being absent from an otherwise-successful query are recorded as
three distinct conditions, never collapsed into one generic error count.

Read-only. Blocking, foreground loop -- run as a background subprocess from
the orchestrator, same pattern as pod_readiness_collector.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.dynamics_calibration_lib import (  # noqa: E402
    compute_missing_known_pods,
    detect_pod_identity_change,
    detect_pod_series_gap,
    parse_pod_readiness_snapshot,
    parse_prometheus_vector_by_pod,
)
from scripts.pod_readiness_collector import fetch_pods_json  # noqa: E402

CSV_FIELDNAMES: List[str] = ["timestamp_utc", "timestamp_epoch", "elapsed_seconds", "pod", "raw_counter_value"]

OBSERVABILITY_CSV_FIELDNAMES: List[str] = [
    "timestamp_utc",
    "elapsed_seconds",
    "pod_uid",
    "pod_name",
    "phase",
    "ready_status",
    "prometheus_series_present",
    "counter_value",
    "missing_series_reason",
]


def query_instant_vector(base_url: str, promql: str, timeout: float = 5.0) -> dict:
    url = f"{base_url.rstrip('/')}/api/v1/query?{urllib.parse.urlencode({'query': promql})}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rl-k8s-per-pod-counter-collector"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return {"status": "error", "error": f"HTTP {resp.status}"}
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return {"status": "error", "error": f"Connection failed: {exc.reason}"}
    except Exception as exc:  # noqa: BLE001 -- collector must never crash on a bad sample
        return {"status": "error", "error": str(exc)}


def run(
    experiment_dir: Path,
    prometheus_url: str,
    namespace: str,
    route: str,
    job: str,
    duration_seconds: float,
    sampling_interval_seconds: float,
    label_selector: str = "app=demo-app",
) -> int:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    csv_path = experiment_dir / "per_pod_request_counters.csv"
    observability_csv_path = experiment_dir / "pod_counter_observability_history.csv"
    metadata_path = experiment_dir / "per_pod_request_counters_metadata.json"

    promql = f'demo_app_requests_total{{namespace="{namespace}",route="{route}",job="{job}"}}'

    interrupted = {"flag": False}

    def _handle_signal(signum, _frame):
        interrupted["flag"] = True

    orig_sigint = signal.signal(signal.SIGINT, _handle_signal)
    orig_sigterm = signal.signal(signal.SIGTERM, _handle_signal)

    start_mono = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()
    sample_count = 0
    whole_prometheus_query_error_count = 0
    kubernetes_listing_error_count = 0
    pod_series_gap_log: List[dict] = []
    identity_change_log: List[dict] = []
    previous_pod_names: List[str] = []
    previous_uid_by_name: dict = {}

    with open(csv_path, "w", newline="", encoding="utf-8") as f, open(
        observability_csv_path, "w", newline="", encoding="utf-8"
    ) as of:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        f.flush()
        obs_writer = csv.DictWriter(of, fieldnames=OBSERVABILITY_CSV_FIELDNAMES)
        obs_writer.writeheader()
        of.flush()

        tick = 0
        while True:
            elapsed = time.monotonic() - start_mono
            if elapsed >= duration_seconds or interrupted["flag"]:
                break

            now = datetime.now(timezone.utc)

            pods_json = fetch_pods_json(namespace, label_selector)
            kubernetes_listing_failed = "_error" in pods_json
            if kubernetes_listing_failed:
                kubernetes_listing_error_count += 1
            known_pods = parse_pod_readiness_snapshot(
                pods_json,
                timestamp_utc=now.isoformat(),
                timestamp_epoch=now.timestamp(),
                elapsed_seconds=round(elapsed, 3),
            )
            current_uid_by_name = {p["pod_name"]: p["pod_uid"] for p in known_pods}
            changed_identities = detect_pod_identity_change(previous_uid_by_name, current_uid_by_name)
            if changed_identities:
                identity_change_log.append(
                    {"timestamp_utc": now.isoformat(), "elapsed_seconds": round(elapsed, 3), "pod_names_with_changed_uid": changed_identities}
                )
            previous_uid_by_name = current_uid_by_name

            response = query_instant_vector(prometheus_url, promql)
            whole_query_failed = response.get("status") != "success"
            if whole_query_failed:
                whole_prometheus_query_error_count += 1
                pod_values: dict = {}
            else:
                pod_values = parse_prometheus_vector_by_pod(response)
                current_pod_names = sorted(pod_values.keys())
                # Phase 15B.3C.2 (previous-tick-relative gap signal; kept
                # alongside the Phase 15B.3C.3 snapshot-based check below,
                # since it remains a cheap, valid diagnostic for a pod
                # that WAS being returned and stops being returned).
                vanished = detect_pod_series_gap(previous_pod_names, current_pod_names)
                if vanished:
                    pod_series_gap_log.append(
                        {"timestamp_utc": now.isoformat(), "elapsed_seconds": round(elapsed, 3), "pods_vanished_since_previous_poll": vanished}
                    )
                previous_pod_names = current_pod_names
                for pod, value in pod_values.items():
                    writer.writerow(
                        {
                            "timestamp_utc": now.isoformat(),
                            "timestamp_epoch": now.timestamp(),
                            "elapsed_seconds": round(elapsed, 3),
                            "pod": pod,
                            "raw_counter_value": value,
                        }
                    )
                    sample_count += 1
            f.flush()

            # Phase 15B.3C.3 (CORR-38 sec.C): snapshot-based observability
            # row per Kubernetes-known pod, regardless of whether it has
            # ever been returned by Prometheus -- catches a pod missing
            # from the first tick or missing for the entire run, which
            # `detect_pod_series_gap` cannot.
            missing = compute_missing_known_pods(known_pods, list(pod_values.keys())) if not whole_query_failed else []
            missing_names = {m["pod_name"] for m in missing}
            for known_pod in known_pods:
                name = known_pod.get("pod_name", "")
                present = name in pod_values
                if kubernetes_listing_failed:
                    reason = ""
                elif whole_query_failed:
                    reason = "whole_prometheus_query_failed"
                elif name in missing_names:
                    reason = "known_to_kubernetes_but_absent_from_prometheus_result"
                else:
                    reason = ""
                obs_writer.writerow(
                    {
                        "timestamp_utc": now.isoformat(),
                        "elapsed_seconds": round(elapsed, 3),
                        "pod_uid": known_pod.get("pod_uid", ""),
                        "pod_name": name,
                        "phase": known_pod.get("phase", ""),
                        "ready_status": known_pod.get("ready_status", ""),
                        "prometheus_series_present": present,
                        "counter_value": pod_values.get(name, ""),
                        "missing_series_reason": reason,
                    }
                )
            of.flush()

            tick += 1
            next_tick_mono = start_mono + tick * sampling_interval_seconds
            sleep_for = next_tick_mono - time.monotonic()
            if sleep_for > 0:
                time.sleep(min(sleep_for, duration_seconds))

    signal.signal(signal.SIGINT, orig_sigint)
    signal.signal(signal.SIGTERM, orig_sigterm)

    end_utc = datetime.now(timezone.utc).isoformat()
    metadata = {
        "start_utc": start_utc,
        "end_utc": end_utc,
        "requested_duration_seconds": duration_seconds,
        "actual_duration_seconds": round(time.monotonic() - start_mono, 2),
        "sampling_interval_seconds": sampling_interval_seconds,
        "namespace": namespace,
        "label_selector": label_selector,
        "promql": promql,
        "sample_row_count": sample_count,
        "query_error_count": whole_prometheus_query_error_count,  # backward-compatible alias
        "whole_prometheus_query_error_count": whole_prometheus_query_error_count,
        "kubernetes_listing_error_count": kubernetes_listing_error_count,
        "interrupted": interrupted["flag"],
        "pod_series_gap_log": pod_series_gap_log,
        "identity_change_log": identity_change_log,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Actively persist raw per-pod demo_app_requests_total counter samples.")
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--namespace", default="autoscaler-app")
    parser.add_argument("--route", default="/work")
    parser.add_argument("--job", default="demo-app")
    parser.add_argument("--label-selector", default="app=demo-app")
    parser.add_argument("--duration-seconds", required=True, type=float)
    parser.add_argument("--sampling-interval-seconds", default=10.0, type=float)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    return run(
        experiment_dir=args.experiment_dir,
        prometheus_url=args.prometheus_url,
        namespace=args.namespace,
        route=args.route,
        job=args.job,
        duration_seconds=args.duration_seconds,
        sampling_interval_seconds=args.sampling_interval_seconds,
        label_selector=args.label_selector,
    )


if __name__ == "__main__":
    sys.exit(main())
