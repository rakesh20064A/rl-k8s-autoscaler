"""Host-side HPA status collector for Phase 11C HPA tuning experiments.

Companion to collectors/metrics_collector.py, scoped ONLY to HorizontalPodAutoscaler
and target-Deployment state. Reuses PrometheusClient from metrics_collector.py for
secondary/corroborating observations only.

Phase 11C.2 hardening (see task_outputs/phase_11c2_hpa_tuning_hardening_and_smoke.md):
Phase 11C.1's original design queried kube-state-metrics via Prometheus as the
PRIMARY source for HPA numeric state, sampled every 5s. But Prometheus scrapes
kube-state-metrics only every ~10s (this project's ServiceMonitor/scrape
interval), so a 5s poll of Prometheus does not give true 5s resolution for
those fields -- most 5s samples would just re-read the same cached scrape.

This version reads the AUTHORITATIVE state directly from the Kubernetes API
(via `kubectl get hpa/deployment -o json`) at the collector's own sampling
cadence, for every field where precise timing matters: HPA
status.currentReplicas/desiredReplicas/currentMetrics/conditions/lastScaleTime,
and Deployment spec.replicas/status.replicas/readyReplicas/availableReplicas.
Prometheus/kube-state-metrics values are retained ONLY as secondary,
clearly-labelled (`prom_*`) corroborating observations -- never as the
authoritative timing source, and Prometheus being unreachable no longer fails
the collector (kubectl is the required, authoritative path; Prometheus is
optional).

Phase 11C.3 hardening (see task_outputs/phase_11c3_final_pre_tuning_corrections.md):
Kubernetes Events are documented by the API itself as best-effort, limited-
retention, SUPPLEMENTAL information -- they are never this project's primary
scaling-timing evidence. The authoritative evidence hierarchy for Phase 11C is:
(1) HPA status.desiredReplicas, (2) Deployment spec.replicas, (3) Deployment
current/ready/available replicas, (4) HPA status.lastScaleTime -- all already
captured in hpa_status.csv above. Kubernetes Events (e.g. SuccessfulRescale)
are corroboration only, never the sole source for reaction-delay or
scaling-action counts. Consistent with this, the collector now BASELINES
whatever HPA-related Events are already visible at startup (see
_baseline_existing_events()) so a stale, pre-existing Event is never mistaken
for something that happened during the trial -- only a genuinely new Event, or
a baselined Event whose occurrence count/timestamp advances AFTER the
baseline, is ever written to hpa_events.jsonl.

This script performs NO scaling actions and NO HPA/Deployment modifications of
any kind -- every kubectl/Prometheus call here is a read. It refuses to
overwrite an existing output directory, uses drift-free monotonic scheduling
(matching metrics_collector.py's own pattern), and never fabricates a value:
data that is absent or a failed lookup is recorded as an empty field, never as
an invented zero.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Reuse the already-implemented, already-tested Prometheus client for the
# secondary/corroborating observations only.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collectors.metrics_collector import PrometheusClient  # noqa: E402


HPA_STATUS_CSV_FIELDNAMES: List[str] = [
    "timestamp_utc",
    "timestamp_epoch",
    "elapsed_seconds",
    # Authoritative (Kubernetes API, read at this collector's own cadence)
    "hpa_status_current_replicas",
    "hpa_status_desired_replicas",
    "hpa_current_cpu_utilization_percent",
    "hpa_current_cpu_average_value_millicores",
    "hpa_target_cpu_utilization_percent",
    "hpa_min_replicas",
    "hpa_max_replicas",
    "able_to_scale",
    "scaling_active",
    "scaling_limited",
    "last_scale_time",
    "deployment_spec_desired_replicas",
    "deployment_status_replicas",
    "deployment_status_ready_replicas",
    "deployment_status_available_replicas",
    # Secondary / corroborating only (Prometheus <- kube-state-metrics, ~10s
    # scrape cadence -- never treat these as more precise than that cadence)
    "prom_current_cpu_utilization_percent",
    "prom_current_cpu_average_value_millicores",
    "prom_desired_replicas",
    "kubectl_error_count",
]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_cpu_quantity_to_millicores(qty: Optional[str]) -> Optional[float]:
    """Parse a Kubernetes CPU quantity string ('10m', '0.01', '1') to millicores."""
    if qty is None or qty == "":
        return None
    qty = qty.strip()
    try:
        if qty.endswith("m"):
            return float(qty[:-1])
        return float(qty) * 1000.0
    except ValueError:
        return None


def _run_kubectl_json(args: List[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Run a read-only `kubectl ... -o json` command. Returns (parsed_json, error)."""
    try:
        result = subprocess.run(
            ["kubectl"] + args + ["-o", "json"],
            capture_output=True, text=True, timeout=5.0,
        )
        if result.returncode != 0:
            return None, result.stderr.strip() or f"kubectl exited {result.returncode}"
        return json.loads(result.stdout), None
    except Exception as exc:
        return None, str(exc)


def _fetch_hpa_state(hpa_name: str, namespace: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    return _run_kubectl_json(["get", "hpa", hpa_name, "-n", namespace])


def _fetch_deployment_state(deployment_name: str, namespace: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    return _run_kubectl_json(["get", "deployment", deployment_name, "-n", namespace])


def _extract_condition(hpa_json: Dict[str, Any], condition_type: str) -> Optional[int]:
    for cond in hpa_json.get("status", {}).get("conditions", []) or []:
        if cond.get("type") == condition_type:
            return 1 if cond.get("status") == "True" else 0
    return None


def _extract_cpu_resource_metric(
    metrics_list: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float]]:
    """From a currentMetrics or spec.metrics list, extract (utilization_pct, average_millicores)
    for the first Resource/cpu entry found."""
    for m in metrics_list or []:
        if m.get("type") != "Resource":
            continue
        resource = m.get("resource", {})
        if resource.get("name") != "cpu":
            continue
        # currentMetrics uses "current"; spec.metrics uses "target"
        target_like = resource.get("current") or resource.get("target") or {}
        util = target_like.get("averageUtilization")
        avg_millicores = _parse_cpu_quantity_to_millicores(target_like.get("averageValue"))
        return (float(util) if util is not None else None), avg_millicores
    return None, None


def _event_progress_key(item: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    """A (count, latest_time) tuple used to detect whether a repeated Event has
    materially advanced since it was last observed. Handles both the classic
    count/lastTimestamp fields and the newer series.count/series.lastObservedTime
    fields, falling back to eventTime where neither is present."""
    series = item.get("series") or {}
    count = item.get("count")
    if count is None:
        count = series.get("count")
    latest_time = item.get("lastTimestamp") or series.get("lastObservedTime") or item.get("eventTime")
    return count, latest_time


def _fetch_new_or_advanced_events(
    hpa_name: str,
    namespace: str,
    seen_state: Dict[str, Tuple[Optional[int], Optional[str]]],
) -> List[Dict[str, Any]]:
    """Read-only kubectl call for HPA-related Events that are new OR have
    materially advanced (repeat-occurrence count increased, or latest
    timestamp advanced) since the last poll -- UID alone is not sufficient,
    since Kubernetes updates an existing Event object in place for repeats."""
    try:
        result = subprocess.run(
            [
                "kubectl", "get", "events", "-n", namespace,
                "--field-selector",
                f"involvedObject.name={hpa_name},involvedObject.kind=HorizontalPodAutoscaler",
                "-o", "json",
            ],
            capture_output=True, text=True, timeout=5.0,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
        new_or_advanced = []
        for item in data.get("items", []):
            uid = item.get("metadata", {}).get("uid")
            if uid is None:
                continue
            progress = _event_progress_key(item)
            prev_progress = seen_state.get(uid)
            if prev_progress == progress:
                continue  # unchanged since last poll
            seen_state[uid] = progress
            series = item.get("series") or {}
            new_or_advanced.append({
                "uid": uid,
                "observed_at_utc": _now_utc_iso(),
                "type": item.get("type"),
                "reason": item.get("reason"),
                "message": item.get("message"),
                "count": item.get("count"),
                "series_count": series.get("count"),
                "first_timestamp": item.get("firstTimestamp"),
                "last_timestamp": item.get("lastTimestamp"),
                "event_time": item.get("eventTime"),
                "series_last_observed_time": series.get("lastObservedTime"),
            })
        return new_or_advanced
    except Exception:
        return []


class HPAStatusCollector:
    """Monotonic-scheduled, read-only HPA + Deployment status sampler."""

    def __init__(
        self,
        experiment_id: str,
        hpa_name: str = "demo-app-hpa",
        deployment_name: Optional[str] = None,
        namespace: str = "autoscaler-app",
        prometheus_url: str = "http://localhost:9090",
        sampling_interval_seconds: float = 5.0,
        duration_seconds: Optional[float] = None,
        output_base_dir: str = "persistent-data/hpa",
        project_root: Optional[Path] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.hpa_name = hpa_name
        self.deployment_name = deployment_name  # resolved from the HPA if None
        self.namespace = namespace
        self.sampling_interval = sampling_interval_seconds
        self.duration_seconds = duration_seconds

        self.prom = PrometheusClient(base_url=prometheus_url)

        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent
        self.output_dir = project_root / output_base_dir / experiment_id
        self.status_csv_path = self.output_dir / "hpa_status.csv"
        self.events_path = self.output_dir / "hpa_events.jsonl"
        self.events_baseline_path = self.output_dir / "hpa_events_baseline.json"

        self.stop_requested = False
        self._seen_event_state: Dict[str, Tuple[Optional[int], Optional[str]]] = {}

    def _baseline_existing_events(self) -> List[Dict[str, Any]]:
        """Snapshot HPA-related Events already visible at collector startup.

        Kubernetes Events are best-effort, limited-retention, supplemental
        information (per the API's own documentation) -- they are never this
        project's primary scaling-timing evidence (see CORR-19). A pre-existing
        Event object (e.g. a stale Warning from long before this trial began)
        must not be misread as something that happened DURING the trial.

        This performs one read of currently-visible events and initializes
        self._seen_event_state from them via the same progress-key machinery
        used during the run, WITHOUT writing anything to hpa_events.jsonl --
        only a genuinely new event, or a baselined event whose occurrence
        count/timestamp advances AFTER this baseline, will ever be emitted as
        a trial occurrence. Returns the baselined events for diagnostic
        persistence (hpa_events_baseline.json) -- historical cluster Events
        are never deleted, only recorded as pre-existing context.
        """
        return _fetch_new_or_advanced_events(self.hpa_name, self.namespace, self._seen_event_state)

    def _setup_directory(self) -> None:
        if self.output_dir.exists():
            raise FileExistsError(
                f"HPA status output directory already exists: {self.output_dir}. "
                "Refusing to overwrite existing experiment data."
            )
        self.output_dir.mkdir(parents=True, exist_ok=False)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        self.stop_requested = True

    def _resolve_deployment_name(self, hpa_json: Optional[Dict[str, Any]]) -> None:
        if self.deployment_name is not None:
            return
        if hpa_json is not None:
            ref_name = hpa_json.get("spec", {}).get("scaleTargetRef", {}).get("name")
            if ref_name:
                self.deployment_name = ref_name
                return
        self.deployment_name = "demo-app"  # project default fallback

    def _prom_query_label(self, metric: str, extra_labels: str = "") -> tuple:
        labels = f'horizontalpodautoscaler="{self.hpa_name}",namespace="{self.namespace}"'
        if extra_labels:
            labels += f",{extra_labels}"
        return self.prom.query(f"{metric}{{{labels}}}")

    def _collect_sample(self, elapsed_seconds: float) -> Dict[str, Any]:
        kubectl_errors = 0

        hpa_json, hpa_err = _fetch_hpa_state(self.hpa_name, self.namespace)
        if hpa_err:
            kubectl_errors += 1

        if self.deployment_name is None:
            self._resolve_deployment_name(hpa_json)

        deploy_json, deploy_err = _fetch_deployment_state(self.deployment_name, self.namespace)
        if deploy_err:
            kubectl_errors += 1

        # --- Authoritative fields, from the Kubernetes API ---
        hpa_status_current_replicas: Optional[int] = None
        hpa_status_desired_replicas: Optional[int] = None
        hpa_current_util: Optional[float] = None
        hpa_current_avg_mcpu: Optional[float] = None
        hpa_target_util: Optional[float] = None
        hpa_min_replicas: Optional[int] = None
        hpa_max_replicas: Optional[int] = None
        able_to_scale: Optional[int] = None
        scaling_active: Optional[int] = None
        scaling_limited: Optional[int] = None
        last_scale_time: Optional[str] = None

        if hpa_json is not None:
            status = hpa_json.get("status", {})
            spec = hpa_json.get("spec", {})
            hpa_status_current_replicas = status.get("currentReplicas")
            hpa_status_desired_replicas = status.get("desiredReplicas")
            hpa_min_replicas = spec.get("minReplicas")
            hpa_max_replicas = spec.get("maxReplicas")
            last_scale_time = status.get("lastScaleTime")

            hpa_current_util, hpa_current_avg_mcpu = _extract_cpu_resource_metric(
                status.get("currentMetrics", [])
            )
            hpa_target_util, _ = _extract_cpu_resource_metric(spec.get("metrics", []))

            able_to_scale = _extract_condition(hpa_json, "AbleToScale")
            scaling_active = _extract_condition(hpa_json, "ScalingActive")
            scaling_limited = _extract_condition(hpa_json, "ScalingLimited")

        deployment_spec_desired: Optional[int] = None
        deployment_status_replicas: Optional[int] = None
        deployment_status_ready: Optional[int] = None
        deployment_status_available: Optional[int] = None
        if deploy_json is not None:
            deployment_spec_desired = deploy_json.get("spec", {}).get("replicas")
            dep_status = deploy_json.get("status", {})
            deployment_status_replicas = dep_status.get("replicas")
            deployment_status_ready = dep_status.get("readyReplicas")
            deployment_status_available = dep_status.get("availableReplicas")

        # --- Secondary / corroborating fields, from Prometheus <- kube-state-metrics ---
        prom_util, _ = self._prom_query_label(
            "kube_horizontalpodautoscaler_status_target_metric",
            'metric_name="cpu",metric_target_type="utilization"',
        )
        prom_avg_cores, _ = self._prom_query_label(
            "kube_horizontalpodautoscaler_status_target_metric",
            'metric_name="cpu",metric_target_type="average"',
        )
        prom_desired, _ = self._prom_query_label(
            "kube_horizontalpodautoscaler_status_desired_replicas"
        )

        def _blank_or_int(v):
            return "" if v is None else int(v)

        def _blank_or_float(v):
            return "" if v is None else v

        return {
            "timestamp_utc": _now_utc_iso(),
            "timestamp_epoch": f"{time.time():.3f}",
            "elapsed_seconds": f"{elapsed_seconds:.3f}",
            "hpa_status_current_replicas": _blank_or_int(hpa_status_current_replicas),
            "hpa_status_desired_replicas": _blank_or_int(hpa_status_desired_replicas),
            "hpa_current_cpu_utilization_percent": _blank_or_float(hpa_current_util),
            "hpa_current_cpu_average_value_millicores": _blank_or_float(hpa_current_avg_mcpu),
            "hpa_target_cpu_utilization_percent": _blank_or_float(hpa_target_util),
            "hpa_min_replicas": _blank_or_int(hpa_min_replicas),
            "hpa_max_replicas": _blank_or_int(hpa_max_replicas),
            "able_to_scale": _blank_or_int(able_to_scale),
            "scaling_active": _blank_or_int(scaling_active),
            "scaling_limited": _blank_or_int(scaling_limited),
            "last_scale_time": last_scale_time or "",
            "deployment_spec_desired_replicas": _blank_or_int(deployment_spec_desired),
            "deployment_status_replicas": _blank_or_int(deployment_status_replicas),
            "deployment_status_ready_replicas": _blank_or_int(deployment_status_ready),
            "deployment_status_available_replicas": _blank_or_int(deployment_status_available),
            "prom_current_cpu_utilization_percent": _blank_or_float(prom_util),
            "prom_current_cpu_average_value_millicores": (
                "" if prom_avg_cores is None else round(prom_avg_cores * 1000, 3)
            ),
            "prom_desired_replicas": _blank_or_int(prom_desired),
            "kubectl_error_count": kubectl_errors,
        }

    def run(self) -> None:
        self._setup_directory()

        orig_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        orig_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)

        # kubectl/HPA reachability is the required precondition -- Prometheus
        # is secondary/optional and no longer gates startup.
        hpa_json, hpa_err = _fetch_hpa_state(self.hpa_name, self.namespace)
        if hpa_json is None:
            raise RuntimeError(
                f"Cannot read hpa/{self.hpa_name} in namespace {self.namespace} via kubectl: {hpa_err}"
            )
        self._resolve_deployment_name(hpa_json)

        # Baseline pre-existing Events BEFORE the sampling loop starts, so
        # anything already visible at startup is never mistaken for a trial
        # occurrence (see CORR-19 and _baseline_existing_events()'s docstring).
        baselined_events = self._baseline_existing_events()
        with open(self.events_baseline_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "baselined_at_utc": _now_utc_iso(),
                    "baselined_event_count": len(baselined_events),
                    "events": baselined_events,
                },
                f,
                indent=2,
            )
            f.write("\n")
        print(
            f"hpa_status_collector: baselined {len(baselined_events)} pre-existing "
            f"event(s) at startup (see hpa_events_baseline.json); these will NOT "
            f"appear in hpa_events.jsonl unless they advance during this trial."
        )

        start_mono = time.monotonic()
        exp_deadline_mono = (
            start_mono + self.duration_seconds if self.duration_seconds else None
        )

        status_file = open(self.status_csv_path, "w", encoding="utf-8", newline="")
        writer = csv.DictWriter(status_file, fieldnames=HPA_STATUS_CSV_FIELDNAMES)
        writer.writeheader()
        status_file.flush()

        events_file = open(self.events_path, "w", encoding="utf-8")

        try:
            sample_idx = 0
            while not self.stop_requested:
                sample_target_mono = start_mono + sample_idx * self.sampling_interval
                if exp_deadline_mono is not None and sample_target_mono >= exp_deadline_mono - 1e-4:
                    break

                now = time.monotonic()
                if sample_target_mono > now:
                    time.sleep(sample_target_mono - now)
                if self.stop_requested:
                    break

                elapsed = time.monotonic() - start_mono
                row = self._collect_sample(elapsed)
                writer.writerow(row)
                status_file.flush()

                for event in _fetch_new_or_advanced_events(
                    self.hpa_name, self.namespace, self._seen_event_state
                ):
                    events_file.write(json.dumps(event) + "\n")
                events_file.flush()

                sample_idx += 1
        finally:
            status_file.close()
            events_file.close()
            signal.signal(signal.SIGINT, orig_sigint)
            signal.signal(signal.SIGTERM, orig_sigterm)
            print(
                f"hpa_status_collector stopped for experiment '{self.experiment_id}' "
                f"after {sample_idx} sample(s)."
            )


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Host-side, read-only HPA + Deployment status collector (Phase 11C)."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--hpa-name", default="demo-app-hpa")
    parser.add_argument("--deployment-name", default=None, help="Defaults to the HPA's own scaleTargetRef.name")
    parser.add_argument("--namespace", default="autoscaler-app")
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--sampling-interval-seconds", type=float, default=5.0)
    parser.add_argument("--duration-seconds", type=float, default=None)
    return parser.parse_args(args)


def main() -> None:
    parsed = parse_args()
    try:
        collector = HPAStatusCollector(
            experiment_id=parsed.experiment_id,
            hpa_name=parsed.hpa_name,
            deployment_name=parsed.deployment_name,
            namespace=parsed.namespace,
            prometheus_url=parsed.prometheus_url,
            sampling_interval_seconds=parsed.sampling_interval_seconds,
            duration_seconds=parsed.duration_seconds,
        )
        collector.run()
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
