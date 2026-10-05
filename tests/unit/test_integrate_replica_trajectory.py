"""Unit tests for scripts/integrate_replica_trajectory.py (Phase 11C.4I).

All tests operate on synthetic in-memory row lists or small temp CSV files --
no Kubernetes, Prometheus, or network access.
"""

from pathlib import Path
import tempfile
import unittest

from scripts.integrate_replica_trajectory import (
    find_transitions,
    integrate_step_function,
    read_hpa_status_rows,
)


def _rows(pairs, field="deployment_status_replicas"):
    """Build a minimal row list: pairs of (elapsed_seconds, value_or_None)."""
    out = []
    for t, v in pairs:
        row = {"_elapsed_seconds": t}
        row[field] = "" if v is None else str(v)
        out.append(row)
    return out


class TestIntegrateStepFunction(unittest.TestCase):
    def test_constant_one_replica(self):
        rows = _rows([(0.0, 1), (300.0, 1), (600.0, 1)])
        result = integrate_step_function(rows, "deployment_status_replicas", scored_end=1230.0)
        self.assertAlmostEqual(result["replica_seconds"], 1230.0)
        self.assertAlmostEqual(result["average"], 1.0)
        self.assertEqual(result["peak"], 1)

    def test_one_to_two_to_one_trajectory(self):
        # 1 replica [0,100), 2 replicas [100,200), 1 replica [200,1230)
        rows = _rows([(0.0, 1), (100.0, 2), (200.0, 1)])
        result = integrate_step_function(rows, "deployment_status_replicas", scored_end=1230.0)
        expected = 1 * 100 + 2 * 100 + 1 * 1030
        self.assertAlmostEqual(result["replica_seconds"], expected)
        self.assertEqual(result["peak"], 2)

    def test_final_sample_before_scored_end_carries_forward(self):
        # last sample at t=1225, value 1 -- must be held through t=1230
        rows = _rows([(0.0, 1), (1225.0, 1)])
        result = integrate_step_function(rows, "deployment_status_replicas", scored_end=1230.0)
        self.assertAlmostEqual(result["replica_seconds"], 1230.0)

    def test_first_sample_slightly_after_zero_uses_standardized_start(self):
        # first sample at t=0.2 (not t=0) -- [0, 0.2) must use start_value=1
        rows = _rows([(0.2, 1), (1230.0 - 0.001, 1)])
        result = integrate_step_function(
            rows, "deployment_status_replicas", scored_end=1230.0, start_value=1.0
        )
        self.assertAlmostEqual(result["replica_seconds"], 1230.0)

    def test_transition_exactly_at_phase_boundary(self):
        # transition exactly at t=330 (a real phase boundary in hpa_tuning_v2)
        rows = _rows([(0.0, 1), (330.0, 2)])
        result = integrate_step_function(rows, "deployment_status_replicas", scored_end=1230.0)
        expected = 1 * 330 + 2 * 900
        self.assertAlmostEqual(result["replica_seconds"], expected)

    def test_null_missing_value_is_skipped_not_a_new_segment(self):
        # middle row has an empty value (simulates a collector error) --
        # must NOT reset/break the step function; value should continue at 2.
        rows = _rows([(0.0, 1), (100.0, 2), (150.0, None), (1230.0 - 0.001, 2)])
        result = integrate_step_function(rows, "deployment_status_replicas", scored_end=1230.0)
        expected = 1 * 100 + 2 * 1130
        self.assertAlmostEqual(result["replica_seconds"], expected)

    def test_samples_after_scored_end_are_ignored(self):
        rows = _rows([(0.0, 1), (1000.0, 2), (1300.0, 5)])
        result = integrate_step_function(rows, "deployment_status_replicas", scored_end=1230.0)
        expected = 1 * 1000 + 2 * 230
        self.assertAlmostEqual(result["replica_seconds"], expected)
        self.assertEqual(result["peak"], 2)

    def test_irregular_sampling_intervals(self):
        rows = _rows([(0.0, 1), (7.3, 1), (412.9, 2), (413.0, 2), (1100.5, 1)])
        result = integrate_step_function(rows, "deployment_status_replicas", scored_end=1230.0)
        expected = 1 * 412.9 + 2 * (1100.5 - 412.9) + 1 * (1230.0 - 1100.5)
        self.assertAlmostEqual(result["replica_seconds"], expected)

    def test_no_usable_data_raises(self):
        rows = _rows([(0.0, None), (5.0, None)])
        with self.assertRaises(ValueError):
            integrate_step_function(rows, "deployment_status_replicas", scored_end=1230.0)


class TestReadHpaStatusRows(unittest.TestCase):
    def test_reads_and_sorts_by_elapsed_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hpa_status.csv"
            path.write_text(
                "elapsed_seconds,deployment_status_replicas\n"
                "10.0,1\n"
                "0.0,1\n"
                "5.0,1\n"
            )
            rows = read_hpa_status_rows(path)
            self.assertEqual([r["_elapsed_seconds"] for r in rows], [0.0, 5.0, 10.0])

    def test_drops_rows_with_unparseable_elapsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hpa_status.csv"
            path.write_text(
                "elapsed_seconds,deployment_status_replicas\n"
                "0.0,1\n"
                ",1\n"
                "not-a-number,1\n"
            )
            rows = read_hpa_status_rows(path)
            self.assertEqual(len(rows), 1)


class TestFindTransitions(unittest.TestCase):
    def test_finds_only_value_changes(self):
        rows = _rows([(0.0, 1), (10.0, 1), (20.0, 2), (30.0, 2), (40.0, 1)])
        for r in rows:
            r["deployment_status_replicas"] = r["deployment_status_replicas"]
        transitions = find_transitions(rows, "deployment_status_replicas", scored_end=1230.0)
        self.assertEqual([t["elapsed_seconds"] for t in transitions], [0.0, 20.0, 40.0])
        self.assertEqual([t["value"] for t in transitions], [1.0, 2.0, 1.0])

    def test_ignores_rows_at_or_after_scored_end(self):
        rows = _rows([(0.0, 1), (1230.0, 2), (1300.0, 3)])
        transitions = find_transitions(rows, "deployment_status_replicas", scored_end=1230.0)
        self.assertEqual([t["elapsed_seconds"] for t in transitions], [0.0])


if __name__ == "__main__":
    unittest.main()
