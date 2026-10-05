"""tests/unit/test_rl_env_episode_ledger.py

Phase 16V: tests for autoscaler/src/rl_env/episode_ledger.py -- append-only
JSONL ledger semantics. Pure filesystem I/O in a tmp_path; no other
dependency.
"""

from __future__ import annotations

import json

from autoscaler.src.rl_env import episode_ledger


class TestAppendAndRead:
    def test_append_creates_file_and_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "ledger.jsonl"
        episode_ledger.append_row(path, {"a": 1})
        assert path.exists()

    def test_rows_are_appended_in_order_and_never_overwritten(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        episode_ledger.append_row(path, {"seq": 1})
        episode_ledger.append_row(path, {"seq": 2})
        episode_ledger.append_row(path, {"seq": 3})
        rows = list(episode_ledger.read_rows(path))
        assert [r["seq"] for r in rows] == [1, 2, 3]

    def test_each_row_is_one_json_line(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        episode_ledger.append_row(path, {"x": "y"})
        episode_ledger.append_row(path, {"x": "z"})
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"x": "y"}
        assert json.loads(lines[1]) == {"x": "z"}

    def test_read_rows_on_nonexistent_ledger_yields_nothing(self, tmp_path):
        path = tmp_path / "does_not_exist.jsonl"
        assert list(episode_ledger.read_rows(path)) == []

    def test_count_rows(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        assert episode_ledger.count_rows(path) == 0
        episode_ledger.append_row(path, {"a": 1})
        episode_ledger.append_row(path, {"a": 2})
        assert episode_ledger.count_rows(path) == 2

    def test_prior_rows_survive_a_later_append(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        episode_ledger.append_row(path, {"seq": 1, "status": "truncated"})
        before = path.read_text()
        episode_ledger.append_row(path, {"seq": 2, "status": "tooling_failed"})
        after = path.read_text()
        assert after.startswith(before)  # strictly additive, no rewrite


class TestNextPhysicalAttemptNumber:
    """Phase16AB (task sec.7): the retry-provenance mechanism."""

    def test_fresh_episode_id_gets_attempt_one(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        assert episode_ledger.next_physical_attempt_number(path, "training_mixed_42_1") == 1

    def test_second_attempt_at_the_same_episode_id_gets_attempt_two(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        episode_ledger.append_row(path, {"episode_id": "training_mixed_42_1", "status": "tooling_failed"})
        assert episode_ledger.next_physical_attempt_number(path, "training_mixed_42_1") == 2

    def test_attempt_number_is_scoped_per_episode_id_not_global(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        episode_ledger.append_row(path, {"episode_id": "training_mixed_42_1", "status": "tooling_failed"})
        episode_ledger.append_row(path, {"episode_id": "training_mixed_42_1", "status": "tooling_failed"})
        # A DIFFERENT episode_id must not be affected by another slot's attempts.
        assert episode_ledger.next_physical_attempt_number(path, "training_step_42_2") == 1
        assert episode_ledger.next_physical_attempt_number(path, "training_mixed_42_1") == 3

    def test_computed_from_disk_not_in_memory_state(self, tmp_path):
        """Simulates a fresh process (no in-memory history) querying after
        a PRIOR process already recorded a failure -- the whole point of
        reading from the ledger file rather than a counter that would reset
        on every restart."""
        path = tmp_path / "ledger.jsonl"
        episode_ledger.append_row(path, {"episode_id": "training_mixed_42_1", "status": "tooling_failed"})
        # A brand new call with zero prior in-memory state still sees it.
        assert episode_ledger.next_physical_attempt_number(path, "training_mixed_42_1") == 2
