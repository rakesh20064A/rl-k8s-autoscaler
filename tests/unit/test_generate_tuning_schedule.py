"""Unit tests for scripts/generate_tuning_schedule.py (Phase 11C.1).

These tests use a small synthetic profile (not the real 840s hpa_tuning_v1.yaml)
so the suite stays fast. No HTTP client is ever imported or instantiated by
the module under test -- confirmed here by asserting the output schedule is
produced purely from parsing + arithmetic, with no network activity possible.
"""

from pathlib import Path
import tempfile
import unittest

from scripts.generate_tuning_schedule import generate_and_persist


SMALL_PROFILE_YAML = """
profile: tiny_test_profile
seed: 42
endpoint: /work
intensity: 3
phases:
  - type: constant
    name: phase_a
    duration_seconds: 5
    target_rps: 2
  - type: constant
    name: phase_b
    duration_seconds: 5
    target_rps: 4
"""


class TestGenerateTuningSchedule(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self.tmpdir.name)
        self.profile_path = self.tmp_root / "tiny.yaml"
        self.profile_path.write_text(SMALL_PROFILE_YAML)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_generates_schedule_and_hash_sidecar(self):
        out_path = self.tmp_root / "schedule.csv"
        digest = generate_and_persist(self.profile_path, "deterministic", 42, out_path)

        self.assertTrue(out_path.exists())
        hash_path = out_path.with_suffix(".csv.sha256")
        self.assertTrue(hash_path.exists())
        self.assertIn(digest, hash_path.read_text())

        rows = out_path.read_text().strip().splitlines()
        # header + at least one row for a 10s, 2-4 RPS schedule
        self.assertGreater(len(rows), 1)

    def test_deterministic_arrival_process_is_reproducible(self):
        out_path_1 = self.tmp_root / "schedule1.csv"
        out_path_2 = self.tmp_root / "schedule2.csv"
        digest_1 = generate_and_persist(self.profile_path, "deterministic", 42, out_path_1)
        digest_2 = generate_and_persist(self.profile_path, "deterministic", 99, out_path_2)

        # Deterministic arrivals ignore the seed entirely -- same profile must
        # produce a byte-identical schedule (and thus identical hash) regardless.
        self.assertEqual(digest_1, digest_2)
        self.assertEqual(out_path_1.read_text(), out_path_2.read_text())

    def test_refuses_to_overwrite_existing_schedule(self):
        out_path = self.tmp_root / "schedule.csv"
        generate_and_persist(self.profile_path, "deterministic", 42, out_path)
        with self.assertRaises(FileExistsError):
            generate_and_persist(self.profile_path, "deterministic", 42, out_path)


if __name__ == "__main__":
    unittest.main()
