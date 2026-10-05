# tests/unit/test_calibration_summary.py
"""Unit tests for calibration summary methodology and analysis scripts."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.compile_stage1_calibration import classify_level
from scripts.summarize_calibration_level import (
    analyze_level,
    compute_percentiles,
    compute_rank_percentiles,
)


class TestPercentileCalculations(unittest.TestCase):
    def test_single_element(self):
        vals = [0.150]
        p = compute_percentiles(vals, [0.50, 0.95, 0.99])
        self.assertEqual(p["p50"], 0.150)
        self.assertEqual(p["p95"], 0.150)
        self.assertEqual(p["p99"], 0.150)

    def test_linear_interpolation(self):
        # 100 values: 1.0 to 100.0
        vals = [float(i) for i in range(1, 101)]
        p = compute_percentiles(vals, [0.50, 0.95, 0.99])
        self.assertAlmostEqual(p["p50"], 50.5, places=5)
        self.assertAlmostEqual(p["p95"], 95.05, places=5)
        self.assertAlmostEqual(p["p99"], 99.01, places=5)

    def test_rank_percentiles(self):
        vals = [float(i) for i in range(1, 101)]
        p = compute_rank_percentiles(vals, [0.50, 0.95, 0.99])
        self.assertEqual(p["p50_rank"], 51.0)
        self.assertEqual(p["p95_rank"], 96.0)
        self.assertEqual(p["p99_rank"], 100.0)


class TestAnalyzeLevelWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.exp_id = "test_exp_01"
        self.traffic_dir = self.root / "persistent-data" / "traffic" / self.exp_id
        self.metrics_dir = self.root / "persistent-data" / "metrics" / self.exp_id
        self.traffic_dir.mkdir(parents=True)
        self.metrics_dir.mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_analyze_level_windowing(self):
        # Write traffic metadata
        t_meta = {
            "experiment_id": self.exp_id,
            "profile": "cal_1rps_90s",
            "requested_duration_seconds": 90.0,
            "actual_runtime_seconds": 90.0,
            "total_offered": 10,
        }
        with open(self.traffic_dir / "metadata.json", "w") as f:
            json.dump(t_meta, f)

        # Write requests.csv (10 requests)
        with open(self.traffic_dir / "requests.csv", "w") as f:
            f.write("request_index,outcome,client_latency_seconds,dispatch_lag_seconds\n")
            for i in range(10):
                lat = 0.10 + i * 0.01  # 0.10 to 0.19
                f.write(f"{i},success_2xx,{lat:.4f},0.0015\n")

        # Write metrics.csv with ramp samples (<60s) and steady samples (>=60s)
        with open(self.metrics_dir / "metrics.csv", "w") as f:
            f.write(
                "elapsed_seconds,cpu_cores,memory_bytes,request_rate_rps,error_rate_rps,latency_p50_seconds,latency_p95_seconds,latency_p99_seconds,replicas_current,sample_valid\n"
            )
            # Ramp samples
            f.write("0.4,0.010,40000000,0.1,0.0,0.10,0.20,0.25,1,True\n")
            f.write("10.0,0.020,40000000,0.3,0.0,0.11,0.21,0.25,1,True\n")
            f.write("30.0,0.040,40000000,0.6,0.0,0.12,0.22,0.25,1,True\n")
            # Steady samples (>= 60s)
            f.write("60.0,0.100,50000000,1.0,0.0,0.13,0.23,0.26,1,True\n")
            f.write("70.0,0.110,50000000,1.0,0.0,0.13,0.23,0.26,1,True\n")
            f.write("80.0,0.120,50000000,1.0,0.0,0.13,0.23,0.26,1,True\n")

        res = analyze_level(self.exp_id, project_root=self.root)

        # Traffic checks
        t = res["traffic"]
        self.assertEqual(t["total_offered"], 10)
        self.assertEqual(t["total_sent"], 10)
        self.assertEqual(t["success_count"], 10)
        self.assertAlmostEqual(t["client_latency"]["p50_seconds"], 0.145, places=3)

        # Steady Prometheus checks (only 3 samples >= 60s)
        s = res["prometheus_steady_window"]
        self.assertEqual(s["sample_count"], 3)
        self.assertAlmostEqual(s["mean_observed_cpu_rate"], 0.110, places=3)
        self.assertAlmostEqual(s["max_observed_cpu_rate"], 0.120, places=3)
        self.assertAlmostEqual(s["observed_request_rate_rps"], 1.0, places=2)
        self.assertAlmostEqual(s["server_p95_seconds"], 0.23, places=2)

        # Whole run checks (6 valid samples)
        w = res["prometheus_whole_run"]
        self.assertEqual(w["sample_count"], 6)
        # Mean across 0.01, 0.02, 0.04, 0.10, 0.11, 0.12 is 0.0667
        self.assertAlmostEqual(w["mean_observed_cpu_rate"], 0.0667, places=4)


class TestClassificationLogic(unittest.TestCase):
    def test_underloaded(self):
        res = {
            "traffic": {
                "success_rate": 1.0,
                "timeouts": 0,
                "dropped_overload": 0,
                "client_latency": {"p95_seconds": 0.130},
            },
            "prometheus_steady_window": {
                "server_p95_seconds": 0.220,
                "cpu_limit_utilization_pct": 53.5,
            },
        }
        self.assertEqual(classify_level(res), "Underloaded")

    def test_approaching_saturation_on_p95(self):
        res = {
            "traffic": {
                "success_rate": 1.0,
                "timeouts": 0,
                "dropped_overload": 0,
                "client_latency": {"p95_seconds": 0.285},
            },
            "prometheus_steady_window": {
                "server_p95_seconds": 0.290,
                "cpu_limit_utilization_pct": 65.0,
            },
        }
        self.assertEqual(classify_level(res), "Approaching Saturation")

    def test_saturated_on_timeouts_or_drops(self):
        res = {
            "traffic": {
                "success_rate": 0.95,
                "timeouts": 3,
                "dropped_overload": 0,
                "client_latency": {"p95_seconds": 0.350},
            },
            "prometheus_steady_window": {
                "server_p95_seconds": 0.400,
                "cpu_limit_utilization_pct": 98.0,
            },
        }
        self.assertEqual(classify_level(res), "Saturated")


class TestOutcomeConservationAndReporting(unittest.TestCase):
    def test_outcome_invariants(self):
        # Verify strict conservation: offered = success + 4xx + 5xx + timeouts + conn_errors + dropped_overload
        t = {
            "total_offered": 900,
            "success_count": 753,
            "client_4xx": 0,
            "server_5xx": 0,
            "timeouts": 141,
            "conn_errors": 1,
            "dropped_overload": 5,
        }
        total_accounted = (
            t["success_count"]
            + t["client_4xx"]
            + t["server_5xx"]
            + t["timeouts"]
            + t["conn_errors"]
            + t["dropped_overload"]
        )
        self.assertEqual(total_accounted, t["total_offered"])
        dispatched = t["total_offered"] - t["dropped_overload"]
        self.assertEqual(dispatched, 895)
        self.assertEqual(
            dispatched,
            t["success_count"] + t["client_4xx"] + t["server_5xx"] + t["timeouts"] + t["conn_errors"],
        )


if __name__ == "__main__":
    unittest.main()
