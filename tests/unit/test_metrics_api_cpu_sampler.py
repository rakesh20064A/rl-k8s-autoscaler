"""Unit tests for scripts/metrics_api_cpu_sampler.py (Phase 11C.4D).

All tests mock subprocess.run (kubectl) -- no network calls, no kubectl
calls, no Kubernetes API access. Nothing here talks to a live cluster.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.metrics_api_cpu_sampler import (
    extract_cpu_sample,
    fetch_pod_metrics,
    parse_cpu_quantity_to_millicores,
    run_sampler,
)


SAMPLE_POD_METRICS = {
    "kind": "PodMetrics",
    "metadata": {"name": "demo-app-647f665b49-rpztw", "namespace": "autoscaler-app"},
    "timestamp": "2026-09-08T10:39:37Z",
    "window": "14.964s",
    "containers": [
        {"name": "demo-app", "usage": {"cpu": "5988839n", "memory": "54004Ki"}}
    ],
}


class TestParseCpuQuantity(unittest.TestCase):
    def test_nanocores(self):
        self.assertAlmostEqual(parse_cpu_quantity_to_millicores("5988839n"), 5.988839, places=5)

    def test_microcores(self):
        self.assertAlmostEqual(parse_cpu_quantity_to_millicores("12345u"), 12.345, places=3)

    def test_millicores(self):
        self.assertEqual(parse_cpu_quantity_to_millicores("42m"), 42.0)

    def test_bare_cores(self):
        self.assertEqual(parse_cpu_quantity_to_millicores("0.5"), 500.0)

    def test_none_and_empty(self):
        self.assertIsNone(parse_cpu_quantity_to_millicores(None))
        self.assertIsNone(parse_cpu_quantity_to_millicores(""))

    def test_malformed(self):
        self.assertIsNone(parse_cpu_quantity_to_millicores("not-a-number"))


class TestExtractCpuSample(unittest.TestCase):
    def test_extracts_expected_fields(self):
        sample = extract_cpu_sample(SAMPLE_POD_METRICS)
        self.assertEqual(sample["metrics_api_timestamp"], "2026-09-08T10:39:37Z")
        self.assertEqual(sample["metrics_api_window"], "14.964s")
        self.assertEqual(sample["pod_name"], "demo-app-647f665b49-rpztw")
        self.assertEqual(sample["container_name"], "demo-app")
        self.assertEqual(sample["cpu_quantity_raw"], "5988839n")
        self.assertAlmostEqual(sample["cpu_millicores"], 5.988839, places=5)

    def test_missing_container_returns_none(self):
        self.assertIsNone(extract_cpu_sample(SAMPLE_POD_METRICS, container_name="does-not-exist"))

    def test_no_containers_returns_none(self):
        self.assertIsNone(extract_cpu_sample({"containers": []}))


class TestFetchPodMetrics(unittest.TestCase):
    @patch("scripts.metrics_api_cpu_sampler.subprocess.run")
    def test_successful_fetch(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(SAMPLE_POD_METRICS))
        result = fetch_pod_metrics("autoscaler-app", "demo-app-647f665b49-rpztw")
        self.assertEqual(result["timestamp"], "2026-09-08T10:39:37Z")

    @patch("scripts.metrics_api_cpu_sampler.subprocess.run")
    def test_nonzero_exit_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        self.assertIsNone(fetch_pod_metrics("autoscaler-app", "missing-pod"))

    @patch("scripts.metrics_api_cpu_sampler.subprocess.run")
    def test_exception_returns_none(self, mock_run):
        mock_run.side_effect = Exception("timeout")
        self.assertIsNone(fetch_pod_metrics("autoscaler-app", "demo-app"))


class TestRunSamplerDedup(unittest.TestCase):
    """Verify repeated polls sharing the same underlying metrics.k8s.io
    'timestamp' are correctly excluded from unique-sample statistics."""

    @patch("scripts.metrics_api_cpu_sampler.time.sleep", return_value=None)
    @patch("scripts.metrics_api_cpu_sampler.fetch_pod_metrics")
    def test_repeated_timestamp_not_double_counted(self, mock_fetch, mock_sleep):
        # Two distinct underlying Metrics Server updates, each observed twice
        # by faster polling -- 4 polls total, 2 unique underlying samples.
        responses = [
            {**SAMPLE_POD_METRICS, "timestamp": "T1", "containers": [{"name": "demo-app", "usage": {"cpu": "100000000n"}}]},
            {**SAMPLE_POD_METRICS, "timestamp": "T1", "containers": [{"name": "demo-app", "usage": {"cpu": "100000000n"}}]},
            {**SAMPLE_POD_METRICS, "timestamp": "T2", "containers": [{"name": "demo-app", "usage": {"cpu": "200000000n"}}]},
            {**SAMPLE_POD_METRICS, "timestamp": "T2", "containers": [{"name": "demo-app", "usage": {"cpu": "200000000n"}}]},
        ]
        mock_fetch.side_effect = responses

        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.metrics_api_cpu_sampler.time.monotonic", side_effect=[0, 0, 1, 2, 3, 4]):
                summary = run_sampler(
                    namespace="autoscaler-app",
                    pod="demo-app-647f665b49-rpztw",
                    experiment_id="test_dedup",
                    duration_seconds=3.5,
                    poll_interval_seconds=0.0,
                    cpu_request_millicores=100.0,
                    output_base_dir="scratch",
                    project_root=Path(tmp),
                )

        self.assertEqual(summary["total_successful_polls"], 4)
        self.assertEqual(summary["unique_underlying_metrics_api_samples"], 2)
        self.assertEqual(summary["cpu_millicores_min"], 100.0)
        self.assertEqual(summary["cpu_millicores_max"], 200.0)

    def test_refuses_existing_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "scratch" / "dup_id"
            existing.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                run_sampler(
                    namespace="autoscaler-app",
                    pod="demo-app",
                    experiment_id="dup_id",
                    duration_seconds=0.0,
                    poll_interval_seconds=0.0,
                    cpu_request_millicores=100.0,
                    output_base_dir="scratch",
                    project_root=Path(tmp),
                )


if __name__ == "__main__":
    unittest.main()
