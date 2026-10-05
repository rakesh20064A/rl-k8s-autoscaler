"""Unit tests for host-side Prometheus metrics collector.

All tests run completely offline and mock HTTP communication.
"""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

import yaml

from collectors.metrics_collector import (
    MetricsCollector,
    PrometheusClient,
    PrometheusTargetError,
    PrometheusUnreachableError,
    format_metric_value,
    parse_args,
)


class MockMonotonicClock:
    """Mock clock for deterministic, instantaneous simulation of monotonic time and sleep."""

    def __init__(self, start_time: float = 1000.0, query_duration: float = 0.0) -> None:
        self.current = start_time
        self.query_duration = query_duration
        self.sleep_calls: List[float] = []
        self.sample_timestamps: List[float] = []

    def monotonic(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.current += seconds

    def on_sample(self) -> None:
        """Simulate sample execution time and record sample timestamp."""
        self.sample_timestamps.append(self.current)
        if self.query_duration > 0:
            self.current += self.query_duration


class TestFormatMetricValue(unittest.TestCase):
    """Test format_metric_value logic for CSV serialization."""

    def test_null_and_nan_values(self):
        self.assertEqual(format_metric_value("latency_p50_seconds", None), "")
        self.assertEqual(format_metric_value("latency_p95_seconds", float("nan")), "")
        self.assertEqual(format_metric_value("cpu_cores", float("inf")), "")
        self.assertEqual(format_metric_value("cpu_cores", float("-inf")), "")

    def test_real_zeros(self):
        self.assertEqual(format_metric_value("request_rate_rps", 0), "0.0")
        self.assertEqual(format_metric_value("request_rate_rps", 0.0), "0.0")
        self.assertEqual(format_metric_value("error_rate_rps", 0.0), "0.0")

    def test_integer_metrics(self):
        self.assertEqual(format_metric_value("replicas_current", 1.0), "1")
        self.assertEqual(format_metric_value("replicas_desired", 4.0), "4")
        self.assertEqual(format_metric_value("replicas_ready", 0.0), "0")
        self.assertEqual(format_metric_value("memory_bytes", 28340224.0), "28340224")

    def test_float_formatting(self):
        self.assertEqual(format_metric_value("request_rate_rps", 12.5), "12.5")
        self.assertEqual(format_metric_value("latency_p50_seconds", 0.045), "0.045")
        self.assertEqual(format_metric_value("cpu_cores", 0.1), "0.1")


class TestPrometheusClient(unittest.TestCase):
    """Test direct HTTP Prometheus client methods with mocks."""

    def setUp(self):
        self.client = PrometheusClient(base_url="http://localhost:9090", timeout_seconds=2.0)

    @patch("urllib.request.urlopen")
    def test_readiness_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        ready, msg = self.client.check_readiness()
        self.assertTrue(ready)
        self.assertEqual(msg, "Ready")

    @patch("urllib.request.urlopen")
    def test_readiness_failure(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        ready, msg = self.client.check_readiness()
        self.assertFalse(ready)
        self.assertIn("Connection failed", msg)

    @patch("urllib.request.urlopen")
    def test_targets_healthy(self, mock_urlopen):
        payload = {
            "status": "success",
            "data": {
                "activeTargets": [
                    {
                        "labels": {
                            "namespace": "autoscaler-app",
                            "app": "demo-app",
                            "job": "autoscaler-app/demo-app",
                        },
                        "health": "up",
                        "lastError": "",
                    }
                ]
            },
        }
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        is_up, msg = self.client.check_target_health("autoscaler-app", "demo-app")
        self.assertTrue(is_up)
        self.assertEqual(msg, "up")

    @patch("urllib.request.urlopen")
    def test_targets_unhealthy(self, mock_urlopen):
        payload = {
            "status": "success",
            "data": {
                "activeTargets": [
                    {
                        "labels": {
                            "namespace": "autoscaler-app",
                            "app": "demo-app",
                        },
                        "health": "down",
                        "lastError": "context deadline exceeded",
                    }
                ]
            },
        }
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        is_up, msg = self.client.check_target_health("autoscaler-app", "demo-app")
        self.assertFalse(is_up)
        self.assertIn("1/1 targets unhealthy", msg)
        self.assertIn("context deadline exceeded", msg)

    @patch("urllib.request.urlopen")
    def test_target_missing(self, mock_urlopen):
        payload = {
            "status": "success",
            "data": {
                "activeTargets": [
                    {
                        "labels": {
                            "namespace": "monitoring",
                            "app": "prometheus",
                        },
                        "health": "up",
                    }
                ]
            },
        }
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        is_up, msg = self.client.check_target_health("autoscaler-app", "demo-app")
        self.assertFalse(is_up)
        self.assertIn("Target 'demo-app' not found", msg)

    @patch("urllib.request.urlopen")
    def test_query_vector_success(self, mock_urlopen):
        payload = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [{"metric": {}, "value": [1700000000.0, "35.5"]}],
            },
        }
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        val, err = self.client.query("test_query")
        self.assertIsNone(err)
        self.assertEqual(val, 35.5)

    @patch("urllib.request.urlopen")
    def test_query_vector_nan(self, mock_urlopen):
        payload = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [{"metric": {}, "value": [1700000000.0, "NaN"]}],
            },
        }
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        val, err = self.client.query("histogram_quantile(...)")
        self.assertIsNone(err)
        self.assertIsNone(val)

    @patch("urllib.request.urlopen")
    def test_query_empty_result(self, mock_urlopen):
        payload = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [],
            },
        }
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        val, err = self.client.query("empty_query")
        self.assertIsNone(err)
        self.assertIsNone(val)

    @patch("urllib.request.urlopen")
    def test_query_scalar(self, mock_urlopen):
        payload = {
            "status": "success",
            "data": {
                "resultType": "scalar",
                "result": [1700000000.0, "10.0"],
            },
        }
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        val, err = self.client.query("scalar_query")
        self.assertIsNone(err)
        self.assertEqual(val, 10.0)

    @patch("urllib.request.urlopen")
    def test_query_api_error(self, mock_urlopen):
        payload = {
            "status": "error",
            "error": "bad PromQL syntax",
        }
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        val, err = self.client.query("bad_syntax")
        self.assertIsNone(val)
        self.assertIn("bad PromQL syntax", err)

    @patch("urllib.request.urlopen")
    def test_query_network_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Connection timed out")

        val, err = self.client.query("timeout_query")
        self.assertIsNone(val)
        self.assertIn("Query network error", err)


class TestMetricsCollectorLifecycle(unittest.TestCase):
    """Test full collector setup, immutability, collection, and teardown."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        # Create mock configs/experiment.yaml
        self.config_dir = self.root_path / "configs"
        self.config_dir.mkdir(parents=True)
        self.config_file = self.config_dir / "experiment.yaml"

        self.mock_config: Dict[str, Any] = {
            "kubernetes": {
                "app_namespace": "autoscaler-app",
                "deployment_name": "demo-app",
            },
            "scaling": {
                "min_replicas": 1,
                "max_replicas": 8,
            },
            "prometheus": {
                "url": "http://localhost:9090",
                "scrape_interval_seconds": 10,
            },
            "collection": {
                "sampling_interval_seconds": 10,
                "output_dir": "persistent-data",
            },
            "traffic": {
                "default_seed": 42,
                "workload_endpoint": "/work",
            },
        }
        with open(self.config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.mock_config, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_refuse_overwrite_existing_directory(self):
        existing_exp = "exp_collision_test"
        target_dir = self.root_path / "persistent-data" / "metrics" / existing_exp
        target_dir.mkdir(parents=True)
        dummy_file = target_dir / "metrics.csv"
        dummy_file.write_text("existing data\n")

        collector = MetricsCollector(
            experiment_id=existing_exp,
            mode="test",
            traffic_profile="constant_load",
            project_root=self.root_path,
        )

        with self.assertRaises(FileExistsError):
            collector._setup_directory()

        # Verify existing data is strictly preserved
        self.assertEqual(dummy_file.read_text(), "existing data\n")

    def test_startup_fails_if_prometheus_unreachable(self):
        collector = MetricsCollector(
            experiment_id="exp_unreachable",
            mode="test",
            traffic_profile="constant_load",
            project_root=self.root_path,
        )

        with patch.object(collector.prom_client, "check_readiness", return_value=(False, "Connection refused")):
            with self.assertRaises(PrometheusUnreachableError):
                collector.run()

        # Verify directory was NOT created
        self.assertFalse(collector.experiment_dir.exists())

    def test_startup_fails_if_target_unhealthy(self):
        collector = MetricsCollector(
            experiment_id="exp_unhealthy_target",
            mode="test",
            traffic_profile="constant_load",
            project_root=self.root_path,
        )

        with patch.object(collector.prom_client, "check_readiness", return_value=(True, "Ready")):
            with patch.object(collector.prom_client, "check_target_health", return_value=(False, "Target down")):
                with self.assertRaises(PrometheusTargetError):
                    collector.run()

        # Verify directory was NOT created
        self.assertFalse(collector.experiment_dir.exists())

    def test_sample_collection_csv_null_handling(self):
        collector = MetricsCollector(
            experiment_id="exp_csv_test",
            mode="test",
            traffic_profile="constant_load",
            project_root=self.root_path,
        )
        collector._setup_directory()

        # Mock target healthy
        collector.prom_client.check_target_health = MagicMock(return_value=(True, "up"))

        # Mock query return values: total rate present (15.2), 5xx absent -> genuine 0.0 error rate
        def mock_query(q):
            if "status_code=~\"5..\"" in q:
                return None, None  # 5xx series absent
            if "status_code=~\"2..\"" in q:
                return 15.2, None
            if "demo_app_requests_total" in q:
                return 15.2, None
            if "histogram_quantile" in q:
                return None, None  # idle/absent latency
            if "demo_app_requests_in_progress" in q:
                return 0.0, None
            if "container_cpu_usage" in q:
                return 0.054321, None
            if "container_memory" in q:
                return 25000000.0, None
            if "replicas" in q:
                return 2.0, None
            if "last_work_intensity" in q:
                return None, None
            return None, None

        collector.prom_client.query = MagicMock(side_effect=mock_query)

        row, is_valid, errors = collector.collect_sample(elapsed=10.0)

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertEqual(row["target_up"], True)
        self.assertEqual(row["sample_valid"], True)
        self.assertEqual(row["request_rate_rps"], "15.2")
        self.assertEqual(row["success_rate_rps"], "15.2")
        self.assertEqual(row["error_rate_rps"], "0.0")  # Genuine zero inferred from total rate present + absent 5xx
        self.assertEqual(row["latency_p50_seconds"], "")  # Null represented as empty
        self.assertEqual(row["latency_p95_seconds"], "")
        self.assertEqual(row["latency_p99_seconds"], "")
        self.assertEqual(row["replicas_current"], "2")  # Integer formatting
        self.assertEqual(row["memory_bytes"], "25000000")
        self.assertEqual(row["last_work_intensity_diagnostic"], "")

    def test_target_unhealthy_during_collection(self):
        collector = MetricsCollector(
            experiment_id="exp_midrun_unhealthy",
            mode="test",
            traffic_profile="step_load",
            project_root=self.root_path,
        )
        collector._setup_directory()

        # Mock target unhealthy
        collector.prom_client.check_target_health = MagicMock(return_value=(False, "pod terminated"))
        collector.prom_client.query = MagicMock(return_value=(None, None))

        row, is_valid, errors = collector.collect_sample(elapsed=20.0)

        self.assertFalse(is_valid)
        self.assertEqual(row["target_up"], False)
        self.assertEqual(row["sample_valid"], False)
        self.assertIn("target_unhealthy", row["collection_error"])
        # Telemetry should be empty, not fabricated zeros
        self.assertEqual(row["request_rate_rps"], "")
        self.assertEqual(row["cpu_cores"], "")

    def test_metadata_provenance_and_finalization(self):
        collector = MetricsCollector(
            experiment_id="exp_meta_test",
            mode="test",
            traffic_profile="step_load",
            seed=99,
            duration_seconds=30.0,
            project_root=self.root_path,
        )
        collector._setup_directory()
        collector.start_epoch = 1700000000.0
        collector.start_utc = "2026-09-06T00:00:00Z"
        collector.total_samples = 3
        collector.valid_samples = 3

        collector._write_metadata(
            end_time_utc="2026-09-06T00:00:30Z",
            actual_duration_seconds=30.05,
            validity_status="test_non_benchmark",
            invalid_reason="Smoke test / non-benchmark execution",
        )

        with open(collector.metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["experiment_id"], "exp_meta_test")
        self.assertEqual(data["mode"], "test")
        self.assertEqual(data["traffic_profile"], "step_load")
        self.assertEqual(data["random_seed"], 99)
        self.assertEqual(data["requested_duration_seconds"], 30.0)
        self.assertEqual(data["actual_duration_seconds"], 30.05)
        self.assertEqual(data["total_samples"], 3)
        self.assertEqual(data["valid_samples"], 3)
        self.assertEqual(data["invalid_samples"], 0)
        self.assertEqual(data["validity_status"], "test_non_benchmark")
        self.assertIn("demo_app_resources", data)
        self.assertIn("scaling_bounds", data)


class TestRateAndValiditySemantics(unittest.TestCase):
    """Test scientific rate zero/null rules, latency null rules, and sample validity."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        self.config_dir = self.root_path / "configs"
        self.config_dir.mkdir(parents=True)
        self.config_file = self.config_dir / "experiment.yaml"

        self.mock_config: Dict[str, Any] = {
            "kubernetes": {
                "app_namespace": "autoscaler-app",
                "deployment_name": "demo-app",
            },
            "scaling": {"min_replicas": 1, "max_replicas": 8},
            "prometheus": {"url": "http://localhost:9090"},
            "collection": {"sampling_interval_seconds": 10, "output_dir": "persistent-data"},
            "traffic": {"default_seed": 42, "workload_endpoint": "/work"},
        }
        with open(self.config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.mock_config, f)

        self.collector = MetricsCollector(
            experiment_id="exp_semantics",
            mode="test",
            traffic_profile="test_profile",
            project_root=self.root_path,
        )
        self.collector._setup_directory()
        self.collector.prom_client.check_target_health = MagicMock(return_value=(True, "up"))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_total_request_series_absent_all_rates_null(self):
        """If total request counter series is absent, all request-rate fields must be null."""
        def mock_query(q):
            if "replicas" in q:
                return 1.0, None
            if "container_cpu" in q:
                return 0.01, None
            if "container_memory" in q:
                return 40000000.0, None
            # All application metrics absent (None, None)
            return None, None

        self.collector.prom_client.query = MagicMock(side_effect=mock_query)
        row, is_valid, errors = self.collector.collect_sample(elapsed=0.0)

        self.assertTrue(is_valid)  # Idle app is scientifically valid
        self.assertEqual(row["request_rate_rps"], "")
        self.assertEqual(row["success_rate_rps"], "")
        self.assertEqual(row["error_rate_rps"], "")
        self.assertEqual(row["latency_p50_seconds"], "")
        self.assertEqual(row["latency_p95_seconds"], "")
        self.assertEqual(row["latency_p99_seconds"], "")

    def test_total_request_series_present_5xx_absent_error_rate_zero(self):
        """If total request series exists and 5xx is absent, error rate is genuine 0.0."""
        def mock_query(q):
            if "status_code=~\"5..\"" in q:
                return None, None  # No 5xx series in Prometheus
            if "status_code=~\"2..\"" in q:
                return 25.0, None
            if "demo_app_requests_total" in q:
                return 25.0, None
            if "replicas" in q:
                return 2.0, None
            if "container_cpu" in q:
                return 0.08, None
            if "container_memory" in q:
                return 45000000.0, None
            return None, None

        self.collector.prom_client.query = MagicMock(side_effect=mock_query)
        row, is_valid, errors = self.collector.collect_sample(elapsed=10.0)

        self.assertTrue(is_valid)
        self.assertEqual(row["request_rate_rps"], "25.0")
        self.assertEqual(row["success_rate_rps"], "25.0")
        self.assertEqual(row["error_rate_rps"], "0.0")  # Verified genuine 0.0

    def test_total_request_series_present_2xx_absent_success_rate_zero(self):
        """If total request series exists and 2xx is absent, success rate is genuine 0.0."""
        def mock_query(q):
            if "status_code=~\"2..\"" in q:
                return None, None  # No 2xx series in Prometheus
            if "status_code=~\"5..\"" in q:
                return 10.0, None
            if "demo_app_requests_total" in q:
                return 10.0, None
            if "replicas" in q:
                return 1.0, None
            if "container_cpu" in q:
                return 0.05, None
            if "container_memory" in q:
                return 30000000.0, None
            return None, None

        self.collector.prom_client.query = MagicMock(side_effect=mock_query)
        row, is_valid, errors = self.collector.collect_sample(elapsed=20.0)

        self.assertTrue(is_valid)
        self.assertEqual(row["request_rate_rps"], "10.0")
        self.assertEqual(row["success_rate_rps"], "0.0")  # Verified genuine 0.0
        self.assertEqual(row["error_rate_rps"], "10.0")

    def test_latency_absent_is_null_never_zero(self):
        """Absence of latency observation must serialize as null (''), never '0 seconds'."""
        def mock_query(q):
            if "histogram_quantile" in q:
                return None, None
            if "replicas" in q:
                return 1.0, None
            if "container_cpu" in q:
                return 0.01, None
            if "container_memory" in q:
                return 20000000.0, None
            return None, None

        self.collector.prom_client.query = MagicMock(side_effect=mock_query)
        row, _, _ = self.collector.collect_sample(elapsed=0.0)

        self.assertEqual(row["latency_p50_seconds"], "")
        self.assertEqual(row["latency_p95_seconds"], "")
        self.assertEqual(row["latency_p99_seconds"], "")

    def test_query_failure_marks_sample_invalid(self):
        """A query error on any metric must mark sample_valid=False and document collection_error."""
        def mock_query(q):
            if "cpu_cores" in q or "container_cpu" in q:
                return None, "HTTP 500 internal server error"
            if "replicas" in q:
                return 1.0, None
            if "container_memory" in q:
                return 20000000.0, None
            return None, None

        self.collector.prom_client.query = MagicMock(side_effect=mock_query)
        row, is_valid, errors = self.collector.collect_sample(elapsed=10.0)

        self.assertFalse(is_valid)
        self.assertIn("cpu_cores_error", row["collection_error"])
        self.assertEqual(self.collector.query_error_count, 1)


class TestTimingAndMonotonicScheduling(unittest.TestCase):
    """Test monotonic drift-free scheduling and duration boundary enforcement."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        self.config_dir = self.root_path / "configs"
        self.config_dir.mkdir(parents=True)
        self.config_file = self.config_dir / "experiment.yaml"

        self.mock_config: Dict[str, Any] = {
            "kubernetes": {"app_namespace": "autoscaler-app", "deployment_name": "demo-app"},
            "scaling": {"min_replicas": 1, "max_replicas": 8},
            "prometheus": {"url": "http://localhost:9090"},
            "collection": {"sampling_interval_seconds": 10, "output_dir": "persistent-data"},
            "traffic": {"default_seed": 42, "workload_endpoint": "/work"},
        }
        with open(self.config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.mock_config, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_35s_duration_10s_cadence_no_overshoot(self):
        """Verify 35s duration at 10s cadence stops at t=35s and takes exactly 4 samples (t=0,10,20,30)."""
        collector = MetricsCollector(
            experiment_id="exp_overshoot_35s",
            mode="test",
            traffic_profile="test_profile",
            duration_seconds=35.0,
            project_root=self.root_path,
        )
        clock = MockMonotonicClock(start_time=1000.0, query_duration=0.1)

        # Mock readiness, target health, and query responses
        collector.prom_client.check_readiness = MagicMock(return_value=(True, "Ready"))
        collector.prom_client.check_target_health = MagicMock(return_value=(True, "up"))
        collector.prom_client.query = MagicMock(return_value=(1.0, None))

        orig_collect = collector.collect_sample

        def instrumented_collect(elapsed):
            clock.on_sample()
            return orig_collect(elapsed)

        collector.collect_sample = instrumented_collect

        with patch("time.monotonic", side_effect=clock.monotonic):
            with patch("time.sleep", side_effect=clock.sleep):
                collector.run()

        # Exactly 4 samples taken: at t=0, 10, 20, 30
        self.assertEqual(collector.total_samples, 4)
        self.assertEqual(collector.valid_samples, 4)

        # Verify sample times relative to start
        sample_times_rel = [round(t - 1000.0, 2) for t in clock.sample_timestamps]
        self.assertEqual(sample_times_rel, [0.0, 10.0, 20.0, 30.0])

        # Verify process stopped at exactly 35.0s, not 40.0s
        total_simulated_elapsed = round(clock.current - 1000.0, 2)
        self.assertAlmostEqual(total_simulated_elapsed, 35.0, delta=0.1)

        # Verify metadata
        with open(collector.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.assertEqual(metadata["requested_duration_seconds"], 35.0)
        self.assertAlmostEqual(metadata["actual_duration_seconds"], 35.0, delta=0.1)

    def test_25s_duration_10s_cadence_stops_at_25s(self):
        """Verify 25s duration at 10s cadence stops at t=25s and takes exactly 3 samples (t=0,10,20)."""
        collector = MetricsCollector(
            experiment_id="exp_overshoot_25s",
            mode="test",
            traffic_profile="test_profile",
            duration_seconds=25.0,
            project_root=self.root_path,
        )
        clock = MockMonotonicClock(start_time=5000.0, query_duration=0.1)

        collector.prom_client.check_readiness = MagicMock(return_value=(True, "Ready"))
        collector.prom_client.check_target_health = MagicMock(return_value=(True, "up"))
        collector.prom_client.query = MagicMock(return_value=(1.0, None))

        orig_collect = collector.collect_sample

        def instrumented_collect(elapsed):
            clock.on_sample()
            return orig_collect(elapsed)

        collector.collect_sample = instrumented_collect

        with patch("time.monotonic", side_effect=clock.monotonic):
            with patch("time.sleep", side_effect=clock.sleep):
                collector.run()

        # Exactly 3 samples taken (t=0, 10, 20)
        self.assertEqual(collector.total_samples, 3)
        sample_times_rel = [round(t - 5000.0, 2) for t in clock.sample_timestamps]
        self.assertEqual(sample_times_rel, [0.0, 10.0, 20.0])

        # No t=30 sample taken; stopped at 25s
        total_simulated_elapsed = round(clock.current - 5000.0, 2)
        self.assertAlmostEqual(total_simulated_elapsed, 25.0, delta=0.1)

    def test_no_cumulative_drift_with_variable_query_duration(self):
        """Verify that slow queries do not introduce cumulative drift to sample targets."""
        collector = MetricsCollector(
            experiment_id="exp_no_drift",
            mode="test",
            traffic_profile="test_profile",
            duration_seconds=30.0,
            project_root=self.root_path,
        )
        # Simulate query taking 0.45 seconds on every sample
        clock = MockMonotonicClock(start_time=2000.0, query_duration=0.45)

        collector.prom_client.check_readiness = MagicMock(return_value=(True, "Ready"))
        collector.prom_client.check_target_health = MagicMock(return_value=(True, "up"))
        collector.prom_client.query = MagicMock(return_value=(1.0, None))

        orig_collect = collector.collect_sample

        def instrumented_collect(elapsed):
            clock.on_sample()
            return orig_collect(elapsed)

        collector.collect_sample = instrumented_collect

        with patch("time.monotonic", side_effect=clock.monotonic):
            with patch("time.sleep", side_effect=clock.sleep):
                collector.run()

        # Samples start exactly at target intervals despite query overhead
        sample_times_rel = [round(t - 2000.0, 2) for t in clock.sample_timestamps]
        self.assertEqual(sample_times_rel, [0.0, 10.0, 20.0])


class TestCLIArgs(unittest.TestCase):
    """Test CLI argument parsing and aliases."""

    def test_valid_args(self):
        args = parse_args([
            "--experiment-id", "exp_100",
            "--mode", "hpa",
            "--traffic-profile", "constant",
            "--seed", "123",
            "--duration-seconds", "60",
        ])
        self.assertEqual(args.experiment_id, "exp_100")
        self.assertEqual(args.mode, "hpa")
        self.assertEqual(args.traffic_profile, "constant")
        self.assertEqual(args.seed, 123)
        self.assertEqual(args.duration_seconds, 60.0)

    def test_aliases(self):
        args = parse_args([
            "--run-id", "run_200",
            "--controller", "rl",
            "--profile", "bursty",
        ])
        self.assertEqual(args.experiment_id, "run_200")
        self.assertEqual(args.mode, "rl")
        self.assertEqual(args.traffic_profile, "bursty")

    def test_invalid_mode_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--experiment-id", "exp_bad", "--mode", "invalid_mode"])


if __name__ == "__main__":
    unittest.main()
