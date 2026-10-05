"""Host-side Prometheus metrics collector for Kubernetes autoscaling experiments.

Observational telemetry collector that queries Prometheus HTTP API, records
scientific metrics to CSV, emits collector lifecycle events to JSONL, and
tracks experiment metadata and provenance in JSON.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request
import yaml


class PrometheusError(Exception):
    """Base exception for Prometheus client errors."""
    pass


class PrometheusUnreachableError(PrometheusError):
    """Raised when Prometheus server is not reachable or ready."""
    pass


class PrometheusTargetError(PrometheusError):
    """Raised when the scrape target is missing or unhealthy at startup."""
    pass


CSV_FIELDNAMES: List[str] = [
    "timestamp_utc",
    "timestamp_epoch",
    "elapsed_seconds",
    "request_rate_rps",
    "success_rate_rps",
    "error_rate_rps",
    "latency_p50_seconds",
    "latency_p95_seconds",
    "latency_p99_seconds",
    "requests_in_progress",
    "cpu_cores",
    "memory_bytes",
    "replicas_current",
    "replicas_desired",
    "replicas_ready",
    "replicas_available",
    "last_work_intensity_diagnostic",
    "target_up",
    "sample_valid",
    "collection_error",
]


def format_metric_value(name: str, val: Optional[float]) -> str:
    """Format numeric telemetry values for CSV serialization.

    Preserves scientific distinction:
    - None or NaN/inf -> empty string ""
    - 0.0 -> "0.0"
    - replicas_* and memory_bytes -> integer string
    - rates and latencies -> cleanly formatted float
    """
    if val is None:
        return ""
    if isinstance(val, (int, float)) and (math.isnan(val) or math.isinf(val)):
        return ""
    if name.startswith("replicas_") or name == "memory_bytes":
        return str(int(round(val)))
    if val == 0 or val == 0.0:
        return "0.0"

    formatted = f"{val:.6f}".rstrip("0")
    if formatted.endswith("."):
        formatted += "0"
    return formatted


class PrometheusClient:
    """Direct HTTP client for Prometheus server."""

    def __init__(self, base_url: str = "http://localhost:9090", timeout_seconds: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds

    def check_readiness(self) -> Tuple[bool, str]:
        """Check if Prometheus server is responsive and ready via /-/ready."""
        url = f"{self.base_url}/-/ready"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rl-k8s-metrics-collector"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    return True, "Ready"
                return False, f"Unexpected HTTP status {resp.status}"
        except urllib.error.URLError as exc:
            return False, f"Connection failed: {exc.reason}"
        except Exception as exc:
            return False, f"Unexpected error: {exc}"

    def get_targets(self) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Fetch active scrape targets from /api/v1/targets."""
        url = f"{self.base_url}/api/v1/targets"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rl-k8s-metrics-collector"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    return [], f"HTTP {resp.status}"
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") != "success":
                    return [], data.get("error", "Unknown Prometheus error")
                return data.get("data", {}).get("activeTargets", []), None
        except urllib.error.URLError as exc:
            return [], f"Connection failed: {exc.reason}"
        except Exception as exc:
            return [], f"Unexpected error: {exc}"

    def check_target_health(
        self,
        namespace: str = "autoscaler-app",
        app_name: str = "demo-app",
    ) -> Tuple[bool, str]:
        """Verify scrape health of demo-app targets.

        Returns (is_up, message).
        """
        targets, err = self.get_targets()
        if err:
            return False, f"Failed to retrieve targets: {err}"

        matched_targets = []
        for target in targets:
            labels = target.get("labels", {})
            ns = labels.get("namespace")
            app = labels.get("app")
            job = labels.get("job")
            container = labels.get("container")
            service = labels.get("service")
            if ns == namespace and (app == app_name or (job and app_name in job) or container == app_name or service == app_name):
                matched_targets.append(target)

        if not matched_targets:
            return False, f"Target '{app_name}' not found in namespace '{namespace}'"

        unhealthy = []
        for t in matched_targets:
            health = t.get("health", "unknown")
            if health.lower() != "up":
                last_err = t.get("lastError", "unspecified error")
                unhealthy.append(f"health={health} ({last_err})")

        if unhealthy:
            return False, f"{len(unhealthy)}/{len(matched_targets)} targets unhealthy: {', '.join(unhealthy)}"

        return True, "up"

    def query(self, promql: str) -> Tuple[Optional[float], Optional[str]]:
        """Execute instant PromQL query.

        Returns:
            (float_value, None) on success
            (None, None) if metric is absent or returns NaN (no error)
            (None, error_str) on query error or communication failure
        """
        query_url = f"{self.base_url}/api/v1/query?{urllib.parse.urlencode({'query': promql})}"
        try:
            req = urllib.request.Request(query_url, headers={"User-Agent": "rl-k8s-metrics-collector"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    return None, f"HTTP {resp.status}"
                body = json.loads(resp.read().decode("utf-8"))

            if body.get("status") != "success":
                return None, body.get("error", "Unknown Prometheus query error")

            data = body.get("data", {})
            result_type = data.get("resultType")
            results = data.get("result", [])

            if not results:
                # Absent metric (e.g. idle rate or empty set)
                return None, None

            if result_type == "vector":
                raw_val = results[0].get("value", [None, None])[1]
            elif result_type == "scalar":
                raw_val = results[1]
            else:
                return None, f"Unsupported resultType '{result_type}'"

            if raw_val is None:
                return None, None

            val_str = str(raw_val).strip()
            if val_str.lower() in ("nan", "+inf", "-inf"):
                return None, None

            try:
                return float(val_str), None
            except ValueError:
                return None, f"Unable to parse float from '{val_str}'"

        except urllib.error.URLError as exc:
            return None, f"Query network error: {exc.reason}"
        except Exception as exc:
            return None, f"Query exception: {exc}"


def get_runtime_provenance(config: Dict[str, Any]) -> Dict[str, Any]:
    """Query runtime environment provenance dynamically without hardcoding."""
    provenance: Dict[str, Any] = {
        "git_commit": None,
        "minikube_version": None,
        "kubernetes_version": None,
        "demo_app_image": None,
        "demo_app_resources": {
            "cpu_request": "unknown",
            "cpu_limit": "unknown",
            "memory_request": "unknown",
            "memory_limit": "unknown",
        },
        "scaling_bounds": {
            "min_replicas": config.get("scaling", {}).get("min_replicas", 1),
            "max_replicas": config.get("scaling", {}).get("max_replicas", 8),
        },
    }

    # Git commit
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            provenance["git_commit"] = proc.stdout.strip()
    except Exception:
        pass

    # Minikube version
    try:
        proc = subprocess.run(
            ["minikube", "version", "--short"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            provenance["minikube_version"] = proc.stdout.strip()
    except Exception:
        pass

    # Kubernetes version
    try:
        proc = subprocess.run(
            ["kubectl", "version", "--output=json"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            k8s_data = json.loads(proc.stdout)
            server_ver = k8s_data.get("serverVersion", {}).get("gitVersion")
            client_ver = k8s_data.get("clientVersion", {}).get("gitVersion")
            provenance["kubernetes_version"] = server_ver or client_ver
    except Exception:
        pass

    # demo-app deployment details
    app_ns = config.get("kubernetes", {}).get("app_namespace", "autoscaler-app")
    dep_name = config.get("kubernetes", {}).get("deployment_name", "demo-app")
    try:
        proc = subprocess.run(
            ["kubectl", "get", "deployment", dep_name, "-n", app_ns, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            dep_data = json.loads(proc.stdout)
            containers = dep_data.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
            if containers:
                container = containers[0]
                provenance["demo_app_image"] = container.get("image")
                resources = container.get("resources", {})
                reqs = resources.get("requests", {})
                limits = resources.get("limits", {})
                provenance["demo_app_resources"] = {
                    "cpu_request": reqs.get("cpu", "unknown"),
                    "cpu_limit": limits.get("cpu", "unknown"),
                    "memory_request": reqs.get("memory", "unknown"),
                    "memory_limit": limits.get("memory", "unknown"),
                }
    except Exception:
        pass

    return provenance


class MetricsCollector:
    """Host-side observational metrics collector for autoscaling experiments."""

    def __init__(
        self,
        experiment_id: str,
        mode: str,
        traffic_profile: str,
        seed: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        config_path: str = "configs/experiment.yaml",
        project_root: Optional[Path] = None,
    ) -> None:
        self.experiment_id = experiment_id
        if mode not in ("hpa", "rl", "manual", "test"):
            raise ValueError(f"Invalid mode '{mode}'. Expected one of: 'hpa', 'rl', 'manual', 'test'")
        self.mode = mode
        self.traffic_profile = traffic_profile
        self.duration_seconds = duration_seconds
        self.config_path = config_path

        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = project_root

        self.config = self._load_config()
        self.seed = seed if seed is not None else self.config.get("traffic", {}).get("default_seed", 42)

        # Output paths
        output_base = self.project_root / self.config.get("collection", {}).get("output_dir", "persistent-data")
        self.experiment_dir = output_base / "metrics" / self.experiment_id
        self.csv_path = self.experiment_dir / "metrics.csv"
        self.metadata_path = self.experiment_dir / "metadata.json"
        self.events_path = self.experiment_dir / "events.jsonl"

        # Prometheus client
        prom_url = self.config.get("prometheus", {}).get("url", "http://localhost:9090")
        self.sampling_interval = float(self.config.get("collection", {}).get("sampling_interval_seconds", 10))
        self.prom_client = PrometheusClient(base_url=prom_url, timeout_seconds=5.0)

        # State tracking
        self.interrupted = False
        self.stop_requested = False
        self.start_epoch: Optional[float] = None
        self.start_utc: Optional[str] = None
        self.total_samples = 0
        self.valid_samples = 0
        self.invalid_samples = 0
        self.query_error_count = 0

        # Build PromQL queries
        app_ns = self.config.get("kubernetes", {}).get("app_namespace", "autoscaler-app")
        dep_name = self.config.get("kubernetes", {}).get("deployment_name", "demo-app")
        route = self.config.get("traffic", {}).get("workload_endpoint", "/work")

        self.queries: Dict[str, str] = {
            "request_rate_rps": f'sum(rate(demo_app_requests_total{{namespace="{app_ns}",route="{route}"}}[1m]))',
            "success_rate_rps": f'sum(rate(demo_app_requests_total{{namespace="{app_ns}",route="{route}",status_code=~"2.."}}[1m]))',
            "error_rate_rps": f'sum(rate(demo_app_requests_total{{namespace="{app_ns}",route="{route}",status_code=~"5.."}}[1m]))',
            "latency_p50_seconds": f'histogram_quantile(0.50, sum(rate(demo_app_request_duration_seconds_bucket{{namespace="{app_ns}",route="{route}"}}[1m])) by (le))',
            "latency_p95_seconds": f'histogram_quantile(0.95, sum(rate(demo_app_request_duration_seconds_bucket{{namespace="{app_ns}",route="{route}"}}[1m])) by (le))',
            "latency_p99_seconds": f'histogram_quantile(0.99, sum(rate(demo_app_request_duration_seconds_bucket{{namespace="{app_ns}",route="{route}"}}[1m])) by (le))',
            "requests_in_progress": f'sum(demo_app_requests_in_progress{{namespace="{app_ns}",route="{route}"}})',
            "cpu_cores": f'sum(rate(container_cpu_usage_seconds_total{{namespace="{app_ns}",pod=~"{dep_name}-.*"}}[1m]))',
            "memory_bytes": f'sum(container_memory_working_set_bytes{{namespace="{app_ns}",pod=~"{dep_name}-.*"}})',
            "replicas_current": f'kube_deployment_status_replicas{{namespace="{app_ns}",deployment="{dep_name}"}}',
            "replicas_desired": f'kube_deployment_spec_replicas{{namespace="{app_ns}",deployment="{dep_name}"}}',
            "replicas_ready": f'kube_deployment_status_replicas_ready{{namespace="{app_ns}",deployment="{dep_name}"}}',
            "replicas_available": f'kube_deployment_status_replicas_available{{namespace="{app_ns}",deployment="{dep_name}"}}',
            "last_work_intensity_diagnostic": f'max(demo_app_last_work_intensity{{namespace="{app_ns}"}})',
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load experiment configuration YAML."""
        full_config_path = self.project_root / self.config_path
        if not full_config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {full_config_path}")
        with open(full_config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def log_event(self, event_type: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Append an event line to events.jsonl."""
        now = time.time()
        event_record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timestamp_epoch": round(now, 3),
            "event": event_type,
            "details": details or {},
        }
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_record) + "\n")
            f.flush()

    def _setup_directory(self) -> None:
        """Create target experiment directory or refuse if it already exists."""
        if self.experiment_dir.exists():
            raise FileExistsError(
                f"Experiment output directory already exists: {self.experiment_dir}. "
                "Refusing to overwrite existing experiment data."
            )
        self.experiment_dir.mkdir(parents=True, exist_ok=False)

    def _write_metadata(
        self,
        end_time_utc: Optional[str] = None,
        actual_duration_seconds: Optional[float] = None,
        validity_status: str = "in_progress",
        invalid_reason: Optional[str] = None,
    ) -> None:
        """Write metadata.json atomically preserving requested vs actual duration."""
        provenance = get_runtime_provenance(self.config)
        metadata = {
            "experiment_id": self.experiment_id,
            "mode": self.mode,
            "traffic_profile": self.traffic_profile,
            "random_seed": self.seed,
            "start_time_utc": self.start_utc,
            "end_time_utc": end_time_utc,
            "requested_duration_seconds": self.duration_seconds,
            "actual_duration_seconds": actual_duration_seconds,
            "duration_seconds": actual_duration_seconds,
            "sampling_interval_seconds": self.sampling_interval,
            "prometheus_url": self.prom_client.base_url,
            "prometheus_scrape_interval_seconds": self.config.get("prometheus", {}).get("scrape_interval_seconds", 10),
            "git_commit": provenance.get("git_commit"),
            "minikube_version": provenance.get("minikube_version"),
            "kubernetes_version": provenance.get("kubernetes_version"),
            "demo_app_image": provenance.get("demo_app_image"),
            "demo_app_resources": provenance.get("demo_app_resources"),
            "scaling_bounds": provenance.get("scaling_bounds"),
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "invalid_samples": self.invalid_samples,
            "query_error_count": self.query_error_count,
            "interrupted": self.interrupted,
            "validity_status": validity_status,
            "invalid_reason": invalid_reason,
        }

        # Write to temporary file and rename for durability
        tmp_path = self.metadata_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(self.metadata_path)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle termination signals gracefully."""
        sig_name = signal.Signals(signum).name
        sys.stderr.write(f"\nReceived signal {sig_name}. Initiating graceful shutdown...\n")
        self.interrupted = True
        self.stop_requested = True

    def collect_sample(self, elapsed: float) -> Tuple[Dict[str, Any], bool, List[str]]:
        """Collect one sample tick from Prometheus.

        Sample Validity Policy:
        A sample is scientifically VALID (sample_valid = True) iff:
        1. Target health check passes (target_up == True).
        2. Required infrastructure/workload telemetry is present:
           - replicas_current is not None
           - cpu_cores is not None
           - memory_bytes is not None
        3. No critical query errors occurred during the tick (query_errors_in_tick == 0).

        Legitimately absent application telemetry (e.g. idle request rates or unobserved
        latency quantiles) does NOT invalidate the sample.
        """
        sample_time_epoch = time.time()
        sample_time_utc = datetime.now(timezone.utc).isoformat()
        errors: List[str] = []

        # 1. Target scrape health check
        app_ns = self.config.get("kubernetes", {}).get("app_namespace", "autoscaler-app")
        dep_name = self.config.get("kubernetes", {}).get("deployment_name", "demo-app")
        target_up, target_msg = self.prom_client.check_target_health(namespace=app_ns, app_name=dep_name)

        if not target_up:
            errors.append(f"target_unhealthy: {target_msg}")
            self.log_event("target_unhealthy", {"reason": target_msg})

        # 2. Query individual metrics
        metric_values: Dict[str, Optional[float]] = {}
        query_errors: Dict[str, str] = {}
        query_errors_in_tick = 0

        for m_name, query_str in self.queries.items():
            val, err = self.prom_client.query(query_str)
            if err:
                query_errors_in_tick += 1
                self.query_error_count += 1
                query_errors[m_name] = err
                errors.append(f"{m_name}_error: {err}")
                self.log_event("query_error", {"metric": m_name, "error": err, "query": query_str})
                metric_values[m_name] = None
            else:
                metric_values[m_name] = val

        # 3. Scientific zero vs null policy for traffic rates:
        # If scrape target is unhealthy, do not infer zero rates.
        if not target_up:
            metric_values["request_rate_rps"] = None
            metric_values["success_rate_rps"] = None
            metric_values["error_rate_rps"] = None
        else:
            total_rate = metric_values.get("request_rate_rps")
            total_rate_err = "request_rate_rps" in query_errors

            if total_rate is None or total_rate_err:
                # If total /work request series is absent or errored:
                # All request rate categories remain null. Do NOT invent zero.
                metric_values["request_rate_rps"] = None
                metric_values["success_rate_rps"] = None
                metric_values["error_rate_rps"] = None
            else:
                # Total request series exists!
                # Absence of status-specific series without query error implies genuine 0.0.
                if metric_values.get("success_rate_rps") is None and "success_rate_rps" not in query_errors:
                    metric_values["success_rate_rps"] = 0.0
                if metric_values.get("error_rate_rps") is None and "error_rate_rps" not in query_errors:
                    metric_values["error_rate_rps"] = 0.0

        # 4. Determine scientific sample validity
        essential_present = (
            metric_values.get("replicas_current") is not None
            and metric_values.get("cpu_cores") is not None
            and metric_values.get("memory_bytes") is not None
        )
        sample_valid = target_up and essential_present and (query_errors_in_tick == 0)

        # 5. Construct CSV row representation
        row: Dict[str, Any] = {
            "timestamp_utc": sample_time_utc,
            "timestamp_epoch": f"{sample_time_epoch:.3f}",
            "elapsed_seconds": f"{elapsed:.2f}",
            "target_up": target_up,
            "sample_valid": sample_valid,
            "collection_error": "; ".join(errors) if errors else "",
        }

        for m_name in self.queries.keys():
            row[m_name] = format_metric_value(m_name, metric_values.get(m_name))

        return row, sample_valid, errors

    def run(self) -> None:
        """Run the metrics collection loop with monotonic drift-free scheduling."""
        # Step 1: Verify Prometheus readiness and target health before creating directories
        ready, ready_msg = self.prom_client.check_readiness()
        if not ready:
            raise PrometheusUnreachableError(
                f"Prometheus at {self.prom_client.base_url} is not ready: {ready_msg}. "
                "Ensure Prometheus is running and port-forward is active before starting collector."
            )

        app_ns = self.config.get("kubernetes", {}).get("app_namespace", "autoscaler-app")
        dep_name = self.config.get("kubernetes", {}).get("deployment_name", "demo-app")
        target_up, target_msg = self.prom_client.check_target_health(namespace=app_ns, app_name=dep_name)
        if not target_up:
            raise PrometheusTargetError(
                f"Scrape target '{dep_name}' in namespace '{app_ns}' is not healthy: {target_msg}. "
                "Refusing to begin collection with unhealthy target."
            )

        # Step 2: Set up directory (refuses to overwrite)
        self._setup_directory()

        # Step 3: Register signal handlers
        orig_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        orig_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)

        start_mono = time.monotonic()
        self.start_epoch = time.time()
        self.start_utc = datetime.now(timezone.utc).isoformat()

        exp_deadline_mono: Optional[float] = (
            (start_mono + self.duration_seconds)
            if self.duration_seconds is not None
            else None
        )

        # Step 4: Initialize output files
        self._write_metadata(validity_status="in_progress")
        self.log_event("collector_started", {
            "experiment_id": self.experiment_id,
            "mode": self.mode,
            "traffic_profile": self.traffic_profile,
            "seed": self.seed,
            "sampling_interval_seconds": self.sampling_interval,
            "requested_duration_seconds": self.duration_seconds,
        })

        print(f"Metrics collection started for experiment '{self.experiment_id}'")
        print(f"Mode: {self.mode} | Traffic profile: {self.traffic_profile} | Seed: {self.seed}")
        print(f"Output directory: {self.experiment_dir}")
        print(f"Sampling cadence: {self.sampling_interval}s | Requested duration: {self.duration_seconds}s")

        csv_file = open(self.csv_path, "w", encoding="utf-8", newline="")
        import csv
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        csv_writer.writeheader()
        csv_file.flush()

        sample_idx = 0

        try:
            while not self.stop_requested:
                planned_sample_mono = start_mono + (sample_idx * self.sampling_interval)

                # If requested duration is set, never schedule a sample at or beyond the deadline
                if exp_deadline_mono is not None and planned_sample_mono >= (exp_deadline_mono - 1e-4):
                    break

                # Wait until planned sample time (or experiment deadline)
                now_mono = time.monotonic()
                if now_mono < planned_sample_mono:
                    wait_target = (
                        min(planned_sample_mono, exp_deadline_mono)
                        if exp_deadline_mono is not None
                        else planned_sample_mono
                    )
                    while not self.stop_requested:
                        rem = wait_target - time.monotonic()
                        if rem <= 0.001:
                            break
                        time.sleep(min(0.1, rem))

                    if exp_deadline_mono is not None and time.monotonic() >= (exp_deadline_mono - 1e-4):
                        break

                if self.stop_requested:
                    break

                # Monotonic elapsed time calculation
                elapsed = time.monotonic() - start_mono

                row, is_valid, sample_errors = self.collect_sample(elapsed)
                csv_writer.writerow(row)
                csv_file.flush()
                os.fsync(csv_file.fileno())

                self.total_samples += 1
                if is_valid:
                    self.valid_samples += 1
                else:
                    self.invalid_samples += 1

                print(
                    f"[{row['timestamp_utc']}] sample={self.total_samples} elapsed={row['elapsed_seconds']}s "
                    f"replicas={row.get('replicas_current', '')} cpu={row.get('cpu_cores', '')} "
                    f"rps={row.get('request_rate_rps', '')} valid={is_valid}"
                )

                sample_idx += 1

                # Calculate next target to sleep toward
                next_sample_mono = start_mono + (sample_idx * self.sampling_interval)
                if exp_deadline_mono is not None:
                    if next_sample_mono >= (exp_deadline_mono - 1e-4):
                        # The next sample would be at or after experiment deadline.
                        # Sleep only until the experiment deadline, then exit.
                        sleep_target = exp_deadline_mono
                    else:
                        sleep_target = next_sample_mono
                else:
                    sleep_target = next_sample_mono

                # Sleep in slices toward sleep_target
                while not self.stop_requested:
                    rem = sleep_target - time.monotonic()
                    if rem <= 0.001:
                        break
                    time.sleep(min(0.1, rem))

                if exp_deadline_mono is not None and time.monotonic() >= (exp_deadline_mono - 1e-4):
                    print(f"Requested duration ({self.duration_seconds}s) reached.")
                    break

        finally:
            csv_file.close()
            # Restore signals
            signal.signal(signal.SIGINT, orig_sigint)
            signal.signal(signal.SIGTERM, orig_sigterm)

            # Finalize metadata and exit event using monotonic elapsed time
            end_mono = time.monotonic()
            end_utc = datetime.now(timezone.utc).isoformat()
            actual_duration = round(end_mono - start_mono, 2)

            if self.mode in ("test", "manual"):
                validity_status = "test_non_benchmark"
                invalid_reason = "Smoke test / non-benchmark execution"
            elif self.interrupted:
                validity_status = "incomplete"
                invalid_reason = "Collection interrupted before completion"
            elif self.invalid_samples > self.valid_samples or self.valid_samples == 0:
                validity_status = "invalid"
                invalid_reason = f"Predominantly invalid samples: {self.invalid_samples}/{self.total_samples}"
            else:
                validity_status = "valid"
                invalid_reason = None

            self.log_event("collector_stopped", {
                "exit_reason": "interrupted" if self.interrupted else "completed",
                "requested_duration_seconds": self.duration_seconds,
                "actual_duration_seconds": actual_duration,
                "total_samples": self.total_samples,
                "valid_samples": self.valid_samples,
                "invalid_samples": self.invalid_samples,
                "validity_status": validity_status,
            })

            self._write_metadata(
                end_time_utc=end_utc,
                actual_duration_seconds=actual_duration,
                validity_status=validity_status,
                invalid_reason=invalid_reason,
            )

            print(
                f"Metrics collection finalized: total_samples={self.total_samples}, "
                f"valid={self.valid_samples}, status={validity_status}, "
                f"actual_duration={actual_duration}s (requested={self.duration_seconds}s)"
            )


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Host-side Prometheus metrics collector for Kubernetes autoscaling experiments"
    )
    parser.add_argument(
        "--config",
        dest="config",
        default="configs/experiment.yaml",
        help="Path to experiment config YAML (default: configs/experiment.yaml)",
    )
    parser.add_argument(
        "--experiment-id",
        "--run-id",
        dest="experiment_id",
        required=True,
        help="Unique identifier for experiment run",
    )
    parser.add_argument(
        "--mode",
        "--controller",
        dest="mode",
        choices=["hpa", "rl", "manual", "test"],
        required=True,
        help="Operating mode: hpa, rl, manual, or test",
    )
    parser.add_argument(
        "--traffic-profile",
        "--profile",
        dest="traffic_profile",
        default="unknown",
        help="Traffic profile name (e.g. constant_load, step_change)",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="Random seed for experiment run",
    )
    parser.add_argument(
        "--duration-seconds",
        dest="duration_seconds",
        type=float,
        default=None,
        help="Optional collection duration in seconds",
    )
    return parser.parse_args(args)


def main() -> None:
    """Entry point for python -m collectors.metrics_collector."""
    parsed = parse_args()
    try:
        collector = MetricsCollector(
            experiment_id=parsed.experiment_id,
            mode=parsed.mode,
            traffic_profile=parsed.traffic_profile,
            seed=parsed.seed,
            duration_seconds=parsed.duration_seconds,
            config_path=parsed.config,
        )
        collector.run()
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
