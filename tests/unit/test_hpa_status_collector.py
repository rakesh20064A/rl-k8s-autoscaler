"""Unit tests for collectors/hpa_status_collector.py (Phase 11C.1 / hardened 11C.2).

All tests mock subprocess.run (kubectl) and PrometheusClient.query -- no network
calls, no kubectl calls, no Kubernetes API access. Nothing here talks to a live
cluster.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from collectors.hpa_status_collector import (
    HPA_STATUS_CSV_FIELDNAMES,
    HPAStatusCollector,
    _event_progress_key,
    _extract_condition,
    _extract_cpu_resource_metric,
    _fetch_new_or_advanced_events,
    _parse_cpu_quantity_to_millicores,
)


SAMPLE_HPA_JSON = {
    "spec": {
        "scaleTargetRef": {"name": "demo-app"},
        "minReplicas": 1,
        "maxReplicas": 8,
        "metrics": [
            {"type": "Resource", "resource": {"name": "cpu", "target": {"type": "Utilization", "averageUtilization": 230}}}
        ],
    },
    "status": {
        "currentReplicas": 1,
        "desiredReplicas": 2,
        "lastScaleTime": "2026-09-08T10:00:00Z",
        "currentMetrics": [
            {"type": "Resource", "resource": {"name": "cpu", "current": {"averageUtilization": 85, "averageValue": "85m"}}}
        ],
        "conditions": [
            {"type": "AbleToScale", "status": "True"},
            {"type": "ScalingActive", "status": "True"},
            {"type": "ScalingLimited", "status": "False"},
        ],
    },
}

SAMPLE_DEPLOYMENT_JSON = {
    "spec": {"replicas": 2},
    "status": {"replicas": 2, "readyReplicas": 1, "availableReplicas": 1},
}


class TestQuantityParsing(unittest.TestCase):
    def test_millicore_suffix(self):
        self.assertEqual(_parse_cpu_quantity_to_millicores("85m"), 85.0)

    def test_bare_cores(self):
        self.assertEqual(_parse_cpu_quantity_to_millicores("0.085"), 85.0)

    def test_none_and_empty(self):
        self.assertIsNone(_parse_cpu_quantity_to_millicores(None))
        self.assertIsNone(_parse_cpu_quantity_to_millicores(""))

    def test_malformed_returns_none(self):
        self.assertIsNone(_parse_cpu_quantity_to_millicores("not-a-number"))


class TestExtraction(unittest.TestCase):
    def test_extract_condition_true(self):
        self.assertEqual(_extract_condition(SAMPLE_HPA_JSON, "AbleToScale"), 1)

    def test_extract_condition_false(self):
        self.assertEqual(_extract_condition(SAMPLE_HPA_JSON, "ScalingLimited"), 0)

    def test_extract_condition_missing(self):
        self.assertIsNone(_extract_condition(SAMPLE_HPA_JSON, "SomeOtherCondition"))

    def test_extract_current_cpu_metric(self):
        util, avg = _extract_cpu_resource_metric(SAMPLE_HPA_JSON["status"]["currentMetrics"])
        self.assertEqual(util, 85.0)
        self.assertEqual(avg, 85.0)

    def test_extract_target_cpu_metric(self):
        util, _ = _extract_cpu_resource_metric(SAMPLE_HPA_JSON["spec"]["metrics"])
        self.assertEqual(util, 230.0)

    def test_extract_from_empty_list(self):
        self.assertEqual(_extract_cpu_resource_metric([]), (None, None))


class TestHPAStatusCollectorLifecycle(unittest.TestCase):
    def test_refuse_overwrite_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            collector = HPAStatusCollector(
                experiment_id="dup_test",
                project_root=tmp_root,
                output_base_dir="persistent-data/hpa",
            )
            (tmp_root / "persistent-data" / "hpa" / "dup_test").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                collector.run()

    @patch("collectors.hpa_status_collector._fetch_hpa_state", return_value=(None, "hpa not found"))
    def test_startup_fails_if_hpa_unreachable_via_kubectl(self, _mock_fetch):
        """kubectl/HPA reachability is now the required precondition -- Prometheus
        being unreachable must NOT be what gates startup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            collector = HPAStatusCollector(experiment_id="unreachable_test", project_root=tmp_root)
            with self.assertRaises(RuntimeError):
                collector.run()

    @patch("subprocess.run")
    @patch("collectors.hpa_status_collector.PrometheusClient.check_readiness", return_value=(False, "unreachable"))
    @patch("collectors.hpa_status_collector._fetch_hpa_state", return_value=(SAMPLE_HPA_JSON, None))
    def test_startup_succeeds_even_if_prometheus_unreachable(self, _mock_hpa, _mock_prom_ready, mock_subproc):
        """Prometheus is secondary/optional -- its unreachability must not be fatal.
        subprocess.run is mocked so the events baseline call (Section 3, this
        phase) makes no real kubectl call -- this test must stay fully offline."""
        empty_events = MagicMock()
        empty_events.returncode = 0
        empty_events.stdout = json.dumps({"items": []})
        mock_subproc.return_value = empty_events

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            collector = HPAStatusCollector(
                experiment_id="prom_down_test",
                project_root=tmp_root,
                duration_seconds=0.01,  # stop almost immediately
            )
            with patch("collectors.hpa_status_collector._fetch_deployment_state", return_value=(SAMPLE_DEPLOYMENT_JSON, None)):
                with patch("collectors.hpa_status_collector.PrometheusClient.query", return_value=(None, None)):
                    collector.run()  # must not raise
            self.assertTrue((tmp_root / "persistent-data" / "hpa" / "prom_down_test" / "hpa_status.csv").exists())


class TestEventBaselining(unittest.TestCase):
    """Kubernetes Events are supplemental, best-effort, limited-retention
    evidence (CORR-19) -- an Event object already visible when the collector
    starts must never be misread as something that happened DURING the trial."""

    def _kubectl_events_result(self, items):
        result = MagicMock()
        result.returncode = 0
        result.stdout = json.dumps({"items": items})
        return result

    def _run_collector_with_events(self, pre_existing_items, mid_run_items):
        """Runs a 2-sample collector: startup baseline sees pre_existing_items,
        then the one sampling tick sees mid_run_items. Returns the events
        actually written to hpa_events.jsonl plus the baseline file contents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            collector = HPAStatusCollector(
                experiment_id="baseline_test",
                project_root=tmp_root,
                duration_seconds=0.05,
                sampling_interval_seconds=0.02,
            )
            # _fetch_hpa_state/_fetch_deployment_state/PrometheusClient.query are
            # all patched directly below, so subprocess.run is only ever called
            # by the events lookup in this test -- safe to sequence unconditionally.
            call_sequence = [
                self._kubectl_events_result(pre_existing_items),  # baseline read
                self._kubectl_events_result(mid_run_items),        # first sample tick
            ]

            def events_side_effect(*_args, **_kwargs):
                return call_sequence.pop(0) if len(call_sequence) > 1 else call_sequence[-1]

            with patch("collectors.hpa_status_collector._fetch_hpa_state", return_value=(SAMPLE_HPA_JSON, None)):
                with patch("collectors.hpa_status_collector._fetch_deployment_state", return_value=(SAMPLE_DEPLOYMENT_JSON, None)):
                    with patch("collectors.hpa_status_collector.PrometheusClient.query", return_value=(None, None)):
                        with patch("subprocess.run", side_effect=events_side_effect):
                            collector.run()

            events_dir = tmp_root / "persistent-data" / "hpa" / "baseline_test"
            emitted = [
                json.loads(line)
                for line in (events_dir / "hpa_events.jsonl").read_text().splitlines()
                if line.strip()
            ]
            baseline = json.loads((events_dir / "hpa_events_baseline.json").read_text())
            return emitted, baseline

    def test_pre_existing_event_not_emitted_on_startup(self):
        pre_existing = [{"metadata": {"uid": "old-1"}, "reason": "FailedGetScale", "count": 2, "lastTimestamp": "T_OLD"}]
        emitted, baseline = self._run_collector_with_events(pre_existing, pre_existing)
        self.assertEqual(len(emitted), 0, "pre-existing event must not be emitted as a trial occurrence")
        self.assertEqual(baseline["baselined_event_count"], 1)
        self.assertEqual(baseline["events"][0]["uid"], "old-1")

    def test_unchanged_pre_existing_event_stays_unemitted(self):
        pre_existing = [{"metadata": {"uid": "old-2"}, "reason": "FailedGetScale", "count": 1, "lastTimestamp": "T_OLD"}]
        # mid_run_items identical -> no advancement
        emitted, _baseline = self._run_collector_with_events(pre_existing, pre_existing)
        self.assertEqual(len(emitted), 0)

    def test_pre_existing_event_with_incremented_count_is_emitted(self):
        pre_existing = [{"metadata": {"uid": "old-3"}, "reason": "SuccessfulRescale", "count": 1, "lastTimestamp": "T_OLD"}]
        advanced = [{"metadata": {"uid": "old-3"}, "reason": "SuccessfulRescale", "count": 2, "lastTimestamp": "T_NEW"}]
        emitted, _baseline = self._run_collector_with_events(pre_existing, advanced)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["uid"], "old-3")
        self.assertEqual(emitted[0]["count"], 2)

    def test_pre_existing_event_with_advanced_series_count_is_emitted(self):
        pre_existing = [{"metadata": {"uid": "old-4"}, "reason": "SuccessfulRescale", "series": {"count": 1, "lastObservedTime": "T_OLD"}}]
        advanced = [{"metadata": {"uid": "old-4"}, "reason": "SuccessfulRescale", "series": {"count": 2, "lastObservedTime": "T_NEW"}}]
        emitted, _baseline = self._run_collector_with_events(pre_existing, advanced)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["series_count"], 2)

    def test_newly_created_uid_after_baseline_is_emitted(self):
        pre_existing = [{"metadata": {"uid": "old-5"}, "reason": "FailedGetScale", "count": 1, "lastTimestamp": "T_OLD"}]
        with_new = pre_existing + [{"metadata": {"uid": "new-1"}, "reason": "SuccessfulRescale", "count": 1, "lastTimestamp": "T_NEW"}]
        emitted, baseline = self._run_collector_with_events(pre_existing, with_new)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["uid"], "new-1")
        self.assertEqual(baseline["baselined_event_count"], 1)


class TestCollectSampleAuthoritativeFields(unittest.TestCase):
    """Verify kubectl (not Prometheus) is now the source of authoritative fields,
    and that HPA-recommendation vs Deployment-mutation vs Ready-capacity are
    kept as distinct fields, never collapsed."""

    def setUp(self):
        self.collector = HPAStatusCollector(experiment_id="sample_test")

    @patch("collectors.hpa_status_collector.PrometheusClient.query", return_value=(None, None))
    @patch("collectors.hpa_status_collector._fetch_deployment_state", return_value=(SAMPLE_DEPLOYMENT_JSON, None))
    @patch("collectors.hpa_status_collector._fetch_hpa_state", return_value=(SAMPLE_HPA_JSON, None))
    def test_authoritative_fields_distinguish_recommendation_from_mutation_from_ready(
        self, _mock_hpa, _mock_deploy, _mock_prom
    ):
        self.collector.deployment_name = "demo-app"
        row = self.collector._collect_sample(elapsed_seconds=12.5)

        # HPA's own recommendation
        self.assertEqual(row["hpa_status_current_replicas"], 1)
        self.assertEqual(row["hpa_status_desired_replicas"], 2)
        # Deployment's actual spec mutation (HPA writes this)
        self.assertEqual(row["deployment_spec_desired_replicas"], 2)
        # Deployment's actual current pod count
        self.assertEqual(row["deployment_status_replicas"], 2)
        # Actual Ready capacity -- deliberately lags the above in this fixture
        self.assertEqual(row["deployment_status_ready_replicas"], 1)
        self.assertEqual(row["deployment_status_available_replicas"], 1)

        self.assertEqual(row["hpa_current_cpu_utilization_percent"], 85.0)
        self.assertEqual(row["hpa_current_cpu_average_value_millicores"], 85.0)
        self.assertEqual(row["hpa_target_cpu_utilization_percent"], 230.0)
        self.assertEqual(row["hpa_min_replicas"], 1)
        self.assertEqual(row["hpa_max_replicas"], 8)
        self.assertEqual(row["able_to_scale"], 1)
        self.assertEqual(row["scaling_active"], 1)
        self.assertEqual(row["scaling_limited"], 0)
        self.assertEqual(row["last_scale_time"], "2026-09-08T10:00:00Z")
        self.assertEqual(row["kubectl_error_count"], 0)

    @patch("collectors.hpa_status_collector.PrometheusClient.query", return_value=(None, None))
    @patch("collectors.hpa_status_collector._fetch_deployment_state", return_value=(None, "not found"))
    @patch("collectors.hpa_status_collector._fetch_hpa_state", return_value=(None, "not found"))
    def test_absent_authoritative_data_recorded_as_empty_not_zero(self, _mock_hpa, _mock_deploy, _mock_prom):
        self.collector.deployment_name = "demo-app"
        row = self.collector._collect_sample(elapsed_seconds=0.5)

        for field in [
            "hpa_status_current_replicas", "hpa_status_desired_replicas",
            "hpa_current_cpu_utilization_percent", "hpa_current_cpu_average_value_millicores",
            "hpa_target_cpu_utilization_percent", "hpa_min_replicas", "hpa_max_replicas",
            "able_to_scale", "scaling_active", "scaling_limited",
            "deployment_spec_desired_replicas", "deployment_status_replicas",
            "deployment_status_ready_replicas", "deployment_status_available_replicas",
        ]:
            self.assertEqual(row[field], "", f"field {field} should be blank, got {row[field]!r}")
        self.assertEqual(row["last_scale_time"], "")
        self.assertEqual(row["kubectl_error_count"], 2)  # both hpa and deployment lookups failed

    @patch("collectors.hpa_status_collector.PrometheusClient.query")
    @patch("collectors.hpa_status_collector._fetch_deployment_state", return_value=(SAMPLE_DEPLOYMENT_JSON, None))
    @patch("collectors.hpa_status_collector._fetch_hpa_state", return_value=(SAMPLE_HPA_JSON, None))
    def test_prometheus_fields_are_clearly_secondary_and_labelled(self, _mock_hpa, _mock_deploy, mock_prom_query):
        mock_prom_query.side_effect = lambda promql: (
            (10.0, None) if "utilization" in promql else
            (0.01, None) if "average" in promql else
            (2.0, None) if "desired_replicas" in promql else
            (None, None)
        )
        self.collector.deployment_name = "demo-app"
        row = self.collector._collect_sample(elapsed_seconds=1.0)

        self.assertEqual(row["prom_current_cpu_utilization_percent"], 10.0)
        self.assertEqual(row["prom_current_cpu_average_value_millicores"], 10.0)
        self.assertEqual(row["prom_desired_replicas"], 2)
        # Authoritative fields must be independent of the prom_* values above
        self.assertEqual(row["hpa_current_cpu_utilization_percent"], 85.0)


class TestEventHardening(unittest.TestCase):
    """Kubernetes updates an existing Event object in place for repeats -- UID
    alone must not be treated as sufficient to detect a new occurrence."""

    def _kubectl_events_result(self, items):
        result = MagicMock()
        result.returncode = 0
        result.stdout = json.dumps({"items": items})
        return result

    def test_new_uid_is_reported(self):
        seen: dict = {}
        item = {"metadata": {"uid": "uid-1"}, "reason": "SuccessfulRescale", "count": 1, "lastTimestamp": "T1"}
        with patch("subprocess.run", return_value=self._kubectl_events_result([item])):
            events = _fetch_new_or_advanced_events("demo-app-hpa", "autoscaler-app", seen)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["uid"], "uid-1")
        self.assertIn("uid-1", seen)

    def test_unchanged_event_is_not_reported_twice(self):
        seen: dict = {}
        item = {"metadata": {"uid": "uid-1"}, "reason": "SuccessfulRescale", "count": 1, "lastTimestamp": "T1"}
        with patch("subprocess.run", return_value=self._kubectl_events_result([item])):
            first = _fetch_new_or_advanced_events("demo-app-hpa", "autoscaler-app", seen)
            second = _fetch_new_or_advanced_events("demo-app-hpa", "autoscaler-app", seen)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 0)

    def test_count_increase_is_reported_as_advanced(self):
        seen: dict = {}
        item_v1 = {"metadata": {"uid": "uid-1"}, "reason": "SuccessfulRescale", "count": 1, "lastTimestamp": "T1"}
        item_v2 = {"metadata": {"uid": "uid-1"}, "reason": "SuccessfulRescale", "count": 2, "lastTimestamp": "T2"}
        with patch("subprocess.run", return_value=self._kubectl_events_result([item_v1])):
            first = _fetch_new_or_advanced_events("demo-app-hpa", "autoscaler-app", seen)
        with patch("subprocess.run", return_value=self._kubectl_events_result([item_v2])):
            second = _fetch_new_or_advanced_events("demo-app-hpa", "autoscaler-app", seen)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["count"], 2)

    def test_series_count_advance_is_reported(self):
        """Newer Kubernetes API versions track repeats under `series.count` /
        `series.lastObservedTime` instead of top-level count/lastTimestamp."""
        seen: dict = {}
        item_v1 = {
            "metadata": {"uid": "uid-2"}, "reason": "SuccessfulRescale",
            "series": {"count": 1, "lastObservedTime": "T1"},
        }
        item_v2 = {
            "metadata": {"uid": "uid-2"}, "reason": "SuccessfulRescale",
            "series": {"count": 3, "lastObservedTime": "T2"},
        }
        with patch("subprocess.run", return_value=self._kubectl_events_result([item_v1])):
            first = _fetch_new_or_advanced_events("demo-app-hpa", "autoscaler-app", seen)
        with patch("subprocess.run", return_value=self._kubectl_events_result([item_v2])):
            second = _fetch_new_or_advanced_events("demo-app-hpa", "autoscaler-app", seen)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["series_count"], 3)

    def test_preserved_fields_present(self):
        seen: dict = {}
        item = {
            "metadata": {"uid": "uid-3"}, "type": "Normal", "reason": "SuccessfulRescale",
            "message": "New size: 2; reason: cpu resource utilization above target",
            "count": 1, "firstTimestamp": "T0", "lastTimestamp": "T1",
        }
        with patch("subprocess.run", return_value=self._kubectl_events_result([item])):
            events = _fetch_new_or_advanced_events("demo-app-hpa", "autoscaler-app", seen)
        e = events[0]
        for field in ["uid", "observed_at_utc", "type", "reason", "message", "count", "first_timestamp", "last_timestamp"]:
            self.assertIn(field, e)

    def test_progress_key_prefers_top_level_then_series_then_event_time(self):
        self.assertEqual(_event_progress_key({"count": 1, "lastTimestamp": "T1"}), (1, "T1"))
        self.assertEqual(
            _event_progress_key({"series": {"count": 2, "lastObservedTime": "T2"}}), (2, "T2")
        )
        self.assertEqual(_event_progress_key({"eventTime": "T3"}), (None, "T3"))


class TestCSVSchema(unittest.TestCase):
    def test_fieldnames_stable(self):
        expected = [
            "timestamp_utc", "timestamp_epoch", "elapsed_seconds",
            "hpa_status_current_replicas", "hpa_status_desired_replicas",
            "hpa_current_cpu_utilization_percent", "hpa_current_cpu_average_value_millicores",
            "hpa_target_cpu_utilization_percent", "hpa_min_replicas", "hpa_max_replicas",
            "able_to_scale", "scaling_active", "scaling_limited", "last_scale_time",
            "deployment_spec_desired_replicas", "deployment_status_replicas",
            "deployment_status_ready_replicas", "deployment_status_available_replicas",
            "prom_current_cpu_utilization_percent", "prom_current_cpu_average_value_millicores",
            "prom_desired_replicas", "kubectl_error_count",
        ]
        self.assertEqual(HPA_STATUS_CSV_FIELDNAMES, expected)


if __name__ == "__main__":
    unittest.main()
