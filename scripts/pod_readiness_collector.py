"""scripts/pod_readiness_collector.py

Phase 15B.3A: actively captures and persists per-pod Kubernetes Ready-
condition history throughout a calibration attempt, at approximately the
frozen 10-second cadence (`configs/experiment.yaml: prometheus.scrape_interval_seconds`).

Per CORR-35 sec.13 / the manifest's `per_pod_routing_diagnostics.ready_set_interval_reconstruction`:
the Kubernetes API does not retain historical per-pod Ready membership once
a pod is gone, so this must be captured LIVE, not reconstructed afterward.
Sources state directly from `kubectl get pods -o json` (this project's
established preference for authoritative object state over coarser
Prometheus/kube-state-metrics polling, per `collectors/hpa_status_collector.py`).

Phase 15B.3C hardening (CORR-37's own recommendation, informed by attempt1's
restart forensic audit): the SAME `kubectl get pods -o json` poll now ALSO
feeds a per-CONTAINER state history (`pod_container_state_history.csv` --
restartCount, current/lastState, termination reason/exit code,
started/finishedAt) so a future restart's cause is recoverable from
PRESERVED evidence alone, without depending on live cluster Events still
being within their retention window at forensic-review time.

Read-only: issues no mutating kubectl command. Blocking, foreground loop
(matches `collectors/metrics_collector.py`'s own pattern) -- run as a
background subprocess from the orchestrator.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.dynamics_calibration_lib import (  # noqa: E402
    classify_external_container_contamination,
    parse_pod_container_state_snapshot,
    parse_pod_readiness_snapshot,
)

CSV_FIELDNAMES: List[str] = [
    "timestamp_utc",
    "timestamp_epoch",
    "elapsed_seconds",
    "pod_name",
    "pod_uid",
    "phase",
    "ready_status",
    "ready_last_transition_time",
    "deletion_timestamp",
    "restart_count",
]

CONTAINER_STATE_CSV_FIELDNAMES: List[str] = [
    "timestamp_utc",
    "timestamp_epoch",
    "elapsed_seconds",
    "pod_name",
    "container_name",
    "container_id",
    "restart_count",
    "current_state",
    "last_state",
    "last_termination_reason",
    "last_termination_exit_code",
    "last_termination_started_at",
    "last_termination_finished_at",
]

# Phase 15B.3E: a continuous watch for Docker Desktop's own built-in
# Kubernetes containers becoming RUNNING during an official attempt --
# preflight alone only checks once, before mutation; Docker Desktop could
# in principle restart these containers mid-campaign (as happened before
# Phase 15B.3D's live run started). "CPU usage where practical" is
# intentionally left blank here: sampling live CPU per matched container
# would require an additional `docker stats` subprocess call per tick,
# which is unnecessary in the common (clean) case and is not required to
# detect the one condition this observer exists to catch -- an unrelated
# container becoming RUNNING.
EXTERNAL_CONTAMINATION_CSV_FIELDNAMES: List[str] = [
    "timestamp_utc",
    "elapsed_seconds",
    "container_name",
    "container_id",
    "status",
    "cpu_percent",
]

# CORR-46: a per-tick observation-error audit trail, distinct from (and in
# addition to) the pre-existing `fetch_error_count` metadata counter -- a
# bare count cannot answer "when" or "why" a fetch failed. Written only on
# a recoverable fetch failure (see `fetch_pods_json`'s own timeout/OSError
# catch); a healthy attempt with zero fetch errors produces a header-only,
# empty file.
OBSERVATION_ERRORS_CSV_FIELDNAMES: List[str] = [
    "timestamp_utc",
    "elapsed_seconds",
    "collector",
    "error_type",
    "timeout_seconds",
    "message",
    "continued_after_error",
]


def fetch_docker_ps_all() -> Tuple[List[Tuple[str, str, str]], Optional[str]]:
    """Returns (rows, error) where each row is (name, id, status_text) for
    EVERY container Docker currently knows about (running or stopped) --
    `-a` is required to also detect a matched container that has been
    stopped (e.g. by this project's own remediation) but could restart.
    """
    try:
        proc = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.ID}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return [], str(exc)
    if proc.returncode != 0:
        return [], proc.stderr.strip()
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            rows.append((parts[0], parts[1], parts[2]))
    return rows, None


def fetch_pods_json(namespace: str, label_selector: str) -> dict:
    """CORR-46: a single transient `kubectl` timeout must be a recoverable
    observation error, not an unhandled exception that crashes this whole
    collector subprocess (as happened during index33 attempt1's
    structural-INVALID classification). Mirrors `fetch_docker_ps_all()`'s
    own pre-existing `(subprocess.TimeoutExpired, OSError)` catch in this
    same file -- this was already the established local convention, it
    simply had not been applied to this specific call site."""
    try:
        proc = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace, "-l", label_selector, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"items": [], "_error": str(exc)}
    if proc.returncode != 0:
        return {"items": [], "_error": proc.stderr.strip()}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"items": [], "_error": f"JSON decode error: {exc}"}


def run(experiment_dir: Path, namespace: str, label_selector: str, duration_seconds: float, sampling_interval_seconds: float) -> int:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    csv_path = experiment_dir / "pod_readiness_history.csv"
    container_state_csv_path = experiment_dir / "pod_container_state_history.csv"
    contamination_csv_path = experiment_dir / "external_container_contamination_history.csv"
    observation_errors_csv_path = experiment_dir / "pod_readiness_observation_errors.csv"
    metadata_path = experiment_dir / "pod_readiness_metadata.json"

    interrupted = {"flag": False}

    def _handle_signal(signum, _frame):
        interrupted["flag"] = True

    orig_sigint = signal.signal(signal.SIGINT, _handle_signal)
    orig_sigterm = signal.signal(signal.SIGTERM, _handle_signal)

    start_mono = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()
    sample_count = 0
    container_state_row_count = 0
    error_count = 0
    contamination_row_count = 0
    contamination_check_error_count = 0
    contamination_detected_ticks: List[float] = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f, open(
        container_state_csv_path, "w", newline="", encoding="utf-8"
    ) as cf, open(contamination_csv_path, "w", newline="", encoding="utf-8") as xf, open(
        observation_errors_csv_path, "w", newline="", encoding="utf-8"
    ) as ef:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        f.flush()
        container_writer = csv.DictWriter(cf, fieldnames=CONTAINER_STATE_CSV_FIELDNAMES)
        container_writer.writeheader()
        cf.flush()
        contamination_writer = csv.DictWriter(xf, fieldnames=EXTERNAL_CONTAMINATION_CSV_FIELDNAMES)
        contamination_writer.writeheader()
        xf.flush()
        observation_errors_writer = csv.DictWriter(ef, fieldnames=OBSERVATION_ERRORS_CSV_FIELDNAMES)
        observation_errors_writer.writeheader()
        ef.flush()

        tick = 0
        while True:
            elapsed = time.monotonic() - start_mono
            if elapsed >= duration_seconds or interrupted["flag"]:
                break

            now = datetime.now(timezone.utc)
            pods_json = fetch_pods_json(namespace, label_selector)
            if "_error" in pods_json:
                error_count += 1
                observation_errors_writer.writerow(
                    {
                        "timestamp_utc": now.isoformat(),
                        "elapsed_seconds": round(elapsed, 3),
                        "collector": "pod_readiness_collector",
                        "error_type": "TimeoutExpired_or_OSError_or_JSONDecodeError",
                        "timeout_seconds": 15,
                        "message": pods_json["_error"],
                        "continued_after_error": True,
                    }
                )
                ef.flush()
            rows = parse_pod_readiness_snapshot(
                pods_json,
                timestamp_utc=now.isoformat(),
                timestamp_epoch=now.timestamp(),
                elapsed_seconds=round(elapsed, 3),
            )
            for row in rows:
                writer.writerow(row)
                sample_count += 1
            f.flush()

            container_rows = parse_pod_container_state_snapshot(
                pods_json,
                timestamp_utc=now.isoformat(),
                timestamp_epoch=now.timestamp(),
                elapsed_seconds=round(elapsed, 3),
            )
            for row in container_rows:
                container_writer.writerow(row)
                container_state_row_count += 1
            cf.flush()

            # Phase 15B.3E: continuous watch for Docker Desktop's own
            # built-in Kubernetes containers becoming RUNNING mid-attempt
            # -- preflight alone only checks once, before mutation.
            docker_rows, docker_err = fetch_docker_ps_all()
            if docker_err is not None:
                contamination_check_error_count += 1
            else:
                matched_names = set(classify_external_container_contamination([r[0] for r in docker_rows]))
                running_matched = []
                for name, container_id, status_text in docker_rows:
                    if name not in matched_names:
                        continue
                    contamination_writer.writerow(
                        {
                            "timestamp_utc": now.isoformat(),
                            "elapsed_seconds": round(elapsed, 3),
                            "container_name": name,
                            "container_id": container_id,
                            "status": status_text,
                            "cpu_percent": "",
                        }
                    )
                    contamination_row_count += 1
                    if status_text.lower().startswith("up"):
                        running_matched.append(name)
                if running_matched:
                    contamination_detected_ticks.append(round(elapsed, 3))
            xf.flush()

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
        "pod_snapshot_row_count": sample_count,
        "container_state_row_count": container_state_row_count,
        "fetch_error_count": error_count,
        "external_contamination_row_count": contamination_row_count,
        "external_contamination_check_error_count": contamination_check_error_count,
        "external_contamination_detected_ticks_elapsed_seconds": contamination_detected_ticks,
        "interrupted": interrupted["flag"],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Actively persist per-pod Kubernetes Ready-condition history.")
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--namespace", default="autoscaler-app")
    parser.add_argument("--label-selector", default="app=demo-app")
    parser.add_argument("--duration-seconds", required=True, type=float)
    parser.add_argument("--sampling-interval-seconds", default=10.0, type=float)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    return run(
        experiment_dir=args.experiment_dir,
        namespace=args.namespace,
        label_selector=args.label_selector,
        duration_seconds=args.duration_seconds,
        sampling_interval_seconds=args.sampling_interval_seconds,
    )


if __name__ == "__main__":
    sys.exit(main())
