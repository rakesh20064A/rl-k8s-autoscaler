from __future__ import annotations

from pathlib import Path

import pytest

from demo_ui import experiment_store as es


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "runs.json"


class TestLifecycle:
    def test_start_run_creates_running_record(self, store_path):
        run_id = es.start_run("HPA", "constant", path=store_path)
        rec = es.get_run(run_id, store_path)
        assert rec["status"] == "running"
        assert rec["controller"] == "HPA"
        assert rec["profile_name"] == "constant"
        assert rec["completed_at"] is None

    def test_active_run_returns_only_running(self, store_path):
        run_id = es.start_run("HPA", "constant", path=store_path)
        assert es.active_run(store_path)["run_id"] == run_id
        es.update_run(run_id, status="completed", summary={}, path=store_path)
        assert es.active_run(store_path) is None

    def test_update_run_stamps_completed_at_once(self, store_path):
        run_id = es.start_run("HPA", "constant", path=store_path)
        es.update_run(run_id, status="completed", summary={"x": 1}, path=store_path)
        rec = es.get_run(run_id, store_path)
        first_completed_at = rec["completed_at"]
        assert first_completed_at is not None
        es.update_run(run_id, stage="COMPLETED", path=store_path)
        assert es.get_run(run_id, store_path)["completed_at"] == first_completed_at

    def test_update_unknown_run_id_is_a_noop(self, store_path):
        es.start_run("HPA", "constant", path=store_path)
        es.update_run("nonexistent", status="completed", path=store_path)  # must not raise
        assert len(es._load(store_path)) == 1


class TestHistoryAndPersistence:
    def test_completed_runs_newest_first(self, store_path):
        r1 = es.start_run("HPA", "constant", path=store_path)
        es.update_run(r1, status="completed", summary={}, path=store_path)
        r2 = es.start_run("PPO", "constant", path=store_path)
        es.update_run(r2, status="completed", summary={}, path=store_path)
        runs = es.completed_runs(store_path)
        assert [r["run_id"] for r in runs] == [r2, r1]

    def test_running_run_excluded_from_completed_runs(self, store_path):
        es.start_run("HPA", "constant", path=store_path)
        assert es.completed_runs(store_path) == []

    def test_persists_across_fresh_reads(self, store_path):
        """Simulates a Streamlit restart: a brand-new read of the same path
        (no shared in-memory state) must see the same completed history."""
        run_id = es.start_run("HPA", "constant", path=store_path)
        es.update_run(run_id, status="completed", summary={"mean_replicas": 2.0}, path=store_path)
        reloaded = es._load(store_path)
        assert len(reloaded) == 1
        assert reloaded[0]["run_id"] == run_id
        assert reloaded[0]["summary"]["mean_replicas"] == 2.0

    def test_no_duplicate_record_from_repeated_status_polls(self, store_path):
        """Mirrors RealExperimentController.status() being polled many times
        by the UI fragment: repeated update_run() calls for the SAME run_id
        must mutate one record, never append a new one."""
        run_id = es.start_run("HPA", "constant", path=store_path)
        for stage in ("STARTING", "EPISODE_RUNNING", "TRAFFIC_STARTED", "COMPLETED"):
            es.update_run(run_id, stage=stage, path=store_path)
        assert len(es._load(store_path)) == 1


class TestCompatiblePairing:
    def _completed(self, store_path, controller, profile, h, seed, dur, summary=None):
        rid = es.start_run(controller, profile, path=store_path)
        es.update_run(
            rid, status="completed", workload_hash=h, workload_seed=seed,
            workload_duration_s=dur, summary=summary or {}, path=store_path,
        )
        return es.get_run(rid, store_path)

    def test_pair_found_for_matching_workload(self, store_path):
        hpa = self._completed(store_path, "HPA", "constant", "h1", 42, 600.0)
        ppo = self._completed(store_path, "PPO", "constant", "h1", 42, 600.0)
        pair = es.latest_compatible_pair(hpa, store_path)
        assert pair["hpa"]["run_id"] == hpa["run_id"]
        assert pair["ppo"]["run_id"] == ppo["run_id"]

    def test_missing_counterpart_reported_as_none(self, store_path):
        hpa = self._completed(store_path, "HPA", "constant", "h1", 42, 600.0)
        pair = es.latest_compatible_pair(hpa, store_path)
        assert pair["hpa"] is not None
        assert pair["ppo"] is None

    def test_different_workload_never_paired(self, store_path):
        """An HPA Ramp run must never be paired with a PPO Constant run."""
        self._completed(store_path, "PPO", "constant", "hC", 1, 60.0)
        hpa_ramp = self._completed(store_path, "HPA", "ramp", "hR", 2, 120.0)
        pair = es.latest_compatible_pair(hpa_ramp, store_path)
        assert pair["hpa"]["run_id"] == hpa_ramp["run_id"]
        assert pair["ppo"] is None

    def test_pair_uses_latest_when_multiple_matches(self, store_path):
        hpa = self._completed(store_path, "HPA", "constant", "h1", 42, 600.0)
        ppo_old = self._completed(store_path, "PPO", "constant", "h1", 42, 600.0)
        ppo_new = self._completed(store_path, "PPO", "constant", "h1", 42, 600.0)
        pair = es.latest_compatible_pair(hpa, store_path)
        assert pair["ppo"]["run_id"] == ppo_new["run_id"]
        assert pair["ppo"]["run_id"] != ppo_old["run_id"]
