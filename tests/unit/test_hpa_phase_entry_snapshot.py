"""Unit tests for scripts/hpa_phase_entry_snapshot.py (Phase 11C.3).

Fully offline -- reads only local fixture files, no network/kubectl/Prometheus.
"""

import csv
from pathlib import Path
import tempfile
import unittest

from scripts.hpa_phase_entry_snapshot import (
    compute_phase_boundaries,
    find_phase_entry_row,
    phase_entry_snapshots,
)


TINY_PROFILE_YAML = """
profile: tiny_test
seed: 42
endpoint: /work
intensity: 3
phases:
  - type: constant
    name: phase_a
    duration_seconds: 10
    target_rps: 2
  - type: constant
    name: phase_b
    duration_seconds: 20
    target_rps: 6
"""

HPA_STATUS_FIELDNAMES = [
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


class TestComputePhaseBoundaries(unittest.TestCase):
    def test_cumulative_offsets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "tiny.yaml"
            profile_path.write_text(TINY_PROFILE_YAML)
            boundaries = compute_phase_boundaries(profile_path)
        self.assertEqual(boundaries, {"phase_a": 0.0, "phase_b": 10.0})


class TestFindPhaseEntryRow(unittest.TestCase):
    def test_returns_first_row_at_or_after_start(self):
        rows = [
            {"elapsed_seconds": "0.0", "hpa_status_desired_replicas": "1"},
            {"elapsed_seconds": "5.0", "hpa_status_desired_replicas": "1"},
            {"elapsed_seconds": "10.0", "hpa_status_desired_replicas": "2"},
            {"elapsed_seconds": "15.0", "hpa_status_desired_replicas": "2"},
        ]
        row = find_phase_entry_row(rows, 10.0)
        self.assertEqual(row["hpa_status_desired_replicas"], "2")
        self.assertEqual(row["elapsed_seconds"], "10.0")

    def test_returns_earliest_when_multiple_qualify(self):
        rows = [
            {"elapsed_seconds": "12.0", "x": "late"},
            {"elapsed_seconds": "10.0", "x": "earliest"},
            {"elapsed_seconds": "11.0", "x": "middle"},
        ]
        row = find_phase_entry_row(rows, 10.0)
        self.assertEqual(row["x"], "earliest")

    def test_returns_none_if_no_sample_after_start(self):
        rows = [{"elapsed_seconds": "0.0"}, {"elapsed_seconds": "5.0"}]
        self.assertIsNone(find_phase_entry_row(rows, 100.0))

    def test_ignores_blank_elapsed_seconds(self):
        rows = [{"elapsed_seconds": ""}, {"elapsed_seconds": "10.0", "x": "ok"}]
        row = find_phase_entry_row(rows, 5.0)
        self.assertEqual(row["x"], "ok")


class TestPhaseEntrySnapshots(unittest.TestCase):
    def _write_hpa_status_csv(self, path: Path, rows):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HPA_STATUS_FIELDNAMES)
            writer.writeheader()
            for row in rows:
                full_row = {k: "" for k in HPA_STATUS_FIELDNAMES}
                full_row.update(row)
                writer.writerow(full_row)

    def test_snapshot_per_phase_reflects_actual_observed_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            profile_path = tmp_root / "tiny.yaml"
            profile_path.write_text(TINY_PROFILE_YAML)

            hpa_status_path = tmp_root / "hpa_status.csv"
            self._write_hpa_status_csv(hpa_status_path, [
                {"elapsed_seconds": "0.0", "hpa_status_current_replicas": "1", "hpa_status_desired_replicas": "1"},
                {"elapsed_seconds": "5.0", "hpa_status_current_replicas": "1", "hpa_status_desired_replicas": "1"},
                # phase_b starts at t=10.0 -- a candidate that already scaled
                # during phase_a would show desired=2 here, not 1
                {"elapsed_seconds": "10.0", "hpa_status_current_replicas": "2", "hpa_status_desired_replicas": "2"},
                {"elapsed_seconds": "15.0", "hpa_status_current_replicas": "2", "hpa_status_desired_replicas": "2"},
            ])

            snapshots = phase_entry_snapshots(hpa_status_path, profile_path)

        self.assertEqual(snapshots["phase_a"]["observed_elapsed_seconds"], 0.0)
        self.assertEqual(snapshots["phase_a"]["hpa_status_desired_replicas"], "1")

        # This is exactly the Phase-11C.3 correction: phase_b's entry state
        # must be read from what was ACTUALLY observed, not assumed as 1.
        self.assertEqual(snapshots["phase_b"]["observed_elapsed_seconds"], 10.0)
        self.assertEqual(snapshots["phase_b"]["hpa_status_desired_replicas"], "2")
        self.assertEqual(snapshots["phase_b"]["hpa_status_current_replicas"], "2")

    def test_missing_snapshot_is_none_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            profile_path = tmp_root / "tiny.yaml"
            profile_path.write_text(TINY_PROFILE_YAML)

            hpa_status_path = tmp_root / "hpa_status.csv"
            # Collector stopped before phase_b (t=10.0) ever began.
            self._write_hpa_status_csv(hpa_status_path, [
                {"elapsed_seconds": "0.0", "hpa_status_desired_replicas": "1"},
                {"elapsed_seconds": "5.0", "hpa_status_desired_replicas": "1"},
            ])

            snapshots = phase_entry_snapshots(hpa_status_path, profile_path)

        self.assertIsNotNone(snapshots["phase_a"])
        self.assertIsNone(snapshots["phase_b"])


if __name__ == "__main__":
    unittest.main()
