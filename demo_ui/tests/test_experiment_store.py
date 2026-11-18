from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from demo_ui import experiment_store as es


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "runs.json"


def _mark_alive(run_id: str, store_path: Path, status_path: "Path | None" = None) -> None:
    """Simulates a genuinely running process: a live pid (this test
    process's own) and, optionally, a status file with a fresh mtime."""
    fields = {"pid": os.getpid()}
    if status_path is not None:
        status_path.write_text("{}")
        fields["status_path"] = str(status_path)
    es.update_run(run_id, path=store_path, **fields)


class TestArmFieldBackfill:
    """Task sec.5: `arm` is derived from `controller` via the same
    `_controller_family()` substring logic Comparison pairing uses --
    written for every NEW run (`start_run`/`update_run`) and backfilled
    once for records written before this phase."""

    def test_start_run_writes_arm_from_controller(self, store_path):
        run_id = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        assert es.get_run(run_id, store_path)["arm"] == "PPO"

    def test_update_run_setting_controller_also_sets_arm(self, store_path):
        run_id = es.start_run("unknown_controller", "demo_cpu_short", path=store_path)
        assert es.get_run(run_id, store_path)["arm"] is None
        es.update_run(run_id, controller="hpa_v2_cpu_memory", path=store_path)
        assert es.get_run(run_id, store_path)["arm"] == "HPA"

    def test_update_run_without_controller_never_touches_arm(self, store_path):
        run_id = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        es.update_run(run_id, stage="EPISODE_RUNNING", path=store_path)
        assert es.get_run(run_id, store_path)["arm"] == "PPO"

    def test_backfill_fills_existing_none_arm_records_from_controller(self, store_path):
        """Simulates records written before this phase (every one has
        `arm=None`, same shape `start_run` used to produce) -- the
        one-time migration must fill them in place, never touching any
        other field (task sec.5: mutates RECORDS only, never
        `summary`/samples content)."""
        store_path.write_text(json.dumps([
            {"run_id": "r1", "controller": "ppo_v2", "arm": None, "summary": {"n_samples": 3}},
            {"run_id": "r2", "controller": "hpa_v2_cpu_memory", "arm": None, "summary": None},
            {"run_id": "r3", "controller": "some_future_hpa_controller", "arm": None, "summary": None},
            {"run_id": "r4", "controller": "unrecognized", "arm": None, "summary": None},
            {"run_id": "r5", "controller": "ppo_v2", "arm": "PPO", "summary": None},  # already set -- untouched
        ]))
        es._backfill_arm_field_once(store_path)
        records = {r["run_id"]: r for r in es._load(store_path)}
        assert records["r1"]["arm"] == "PPO"
        assert records["r1"]["summary"] == {"n_samples": 3}  # untouched
        assert records["r2"]["arm"] == "HPA"
        assert records["r3"]["arm"] == "HPA"  # forward-compatible substring match
        assert records["r4"]["arm"] is None  # unrecognized controller -- never fabricated
        assert records["r5"]["arm"] == "PPO"

    def test_backfill_is_idempotent_and_a_noop_on_an_already_backfilled_store(self, store_path):
        run_id = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        before = es._load(store_path)
        es._backfill_arm_field_once(store_path)
        assert es._load(store_path) == before
        assert es.get_run(run_id, store_path)["arm"] == "PPO"


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
        _mark_alive(run_id, store_path)
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

    def test_pair_found_for_real_v2_controller_ids(self, store_path):
        """The real V2 controller ids are `ppo_v2` and `hpa_v2_cpu_memory`
        (see components/controls_v2.py) -- not the bare `"hpa"`/`"ppo"`
        this used to require an exact (lowercased) match against."""
        ppo = self._completed(store_path, "ppo_v2", "demo_cpu_short", "h1", 42, 40.0)
        hpa = self._completed(store_path, "hpa_v2_cpu_memory", "demo_cpu_short", "h1", 42, 40.0)
        pair = es.latest_compatible_pair(ppo, store_path)
        assert pair["ppo"]["run_id"] == ppo["run_id"]
        assert pair["hpa"]["run_id"] == hpa["run_id"]

    def test_pair_members_both_have_the_arm_field_populated(self, store_path):
        """Task sec.5/6: pairing itself keys off `controller`, never
        `arm`, but every record pairing produces must still carry a
        correct, populated `arm` label (never `None`) -- this is what
        Comparison and any other view render wherever they want the
        short "PPO"/"HPA" label instead of the raw controller id."""
        ppo = self._completed(store_path, "ppo_v2", "demo_cpu_short", "h1", 42, 40.0)
        hpa = self._completed(store_path, "hpa_v2_cpu_memory", "demo_cpu_short", "h1", 42, 40.0)
        pair = es.latest_compatible_pair(ppo, store_path)
        assert pair["ppo"]["arm"] == "PPO"
        assert pair["hpa"]["arm"] == "HPA"
        assert ppo["arm"] == "PPO"
        assert hpa["arm"] == "HPA"


class TestInvalidNoTrafficStatus:
    """Task sec.3: `invalid_no_traffic` is TERMINAL (History/the Live
    Details run selector still show it) but never a Comparison pairing
    candidate and never the Live Details default-selected run."""

    def _completed_with_samples(self, store_path, controller, profile, rates, h="h1", seed=42, dur=120.0):
        rid = es.start_run(controller, profile, path=store_path)
        samples = [{"time_s": float(i) * 30.0, "request_rate_rps": r} for i, r in enumerate(rates)]
        es.update_run(
            rid, status="completed", workload_hash=h, workload_seed=seed, workload_duration_s=dur,
            summary={"samples": samples}, path=store_path,
        )
        return es.get_run(rid, store_path)

    def test_is_zero_traffic_run_true_when_every_sample_is_zero_or_none(self, store_path):
        rec = self._completed_with_samples(store_path, "ppo_v2", "demo_cpu_short", [0.0, None, 0.0])
        assert es.is_zero_traffic_run(rec) is True

    def test_is_zero_traffic_run_false_when_any_sample_has_real_traffic(self, store_path):
        rec = self._completed_with_samples(store_path, "ppo_v2", "demo_cpu_short", [0.0, 12.5, 0.0])
        assert es.is_zero_traffic_run(rec) is False

    def test_is_zero_traffic_run_false_when_no_samples_at_all(self, store_path):
        rid = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        es.update_run(rid, status="error", summary={}, path=store_path)
        rec = es.get_run(rid, store_path)
        assert es.is_zero_traffic_run(rec) is False

    def test_invalidate_zero_traffic_runs_marks_only_the_zero_traffic_ones(self, store_path):
        zero = self._completed_with_samples(store_path, "ppo_v2", "demo_cpu_short", [0.0, 0.0])
        real = self._completed_with_samples(store_path, "hpa_v2_cpu_memory", "demo_cpu_short", [5.0, 6.0])

        changed = es.invalidate_zero_traffic_runs(store_path)

        assert changed == [zero["run_id"]]
        assert es.get_run(zero["run_id"], store_path)["status"] == es.INVALID_NO_TRAFFIC
        assert es.get_run(real["run_id"], store_path)["status"] == "completed"

    def test_invalidate_zero_traffic_runs_never_touches_a_running_record(self, store_path):
        rid = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        es.update_run(rid, summary={"samples": [{"time_s": 0.0, "request_rate_rps": 0.0}]}, path=store_path)
        changed = es.invalidate_zero_traffic_runs(store_path)
        assert changed == []
        assert es.get_run(rid, store_path)["status"] == "running"

    def test_invalidate_zero_traffic_runs_is_idempotent(self, store_path):
        zero = self._completed_with_samples(store_path, "ppo_v2", "demo_cpu_short", [0.0])
        first = es.invalidate_zero_traffic_runs(store_path)
        second = es.invalidate_zero_traffic_runs(store_path)
        assert first == [zero["run_id"]]
        assert second == []

    def test_invalidate_zero_traffic_runs_never_deletes_the_record_or_summary(self, store_path):
        zero = self._completed_with_samples(store_path, "ppo_v2", "demo_cpu_short", [0.0])
        es.invalidate_zero_traffic_runs(store_path)
        rec = es.get_run(zero["run_id"], store_path)
        assert rec is not None
        assert rec["summary"]["samples"]

    def test_invalid_no_traffic_is_terminal_but_never_a_pairing_candidate(self, store_path):
        ppo = self._completed_with_samples(store_path, "ppo_v2", "demo_cpu_short", [0.0])
        es.update_run(ppo["run_id"], status=es.INVALID_NO_TRAFFIC, path=store_path)
        hpa = self._completed_with_samples(store_path, "hpa_v2_cpu_memory", "demo_cpu_short", [5.0])

        assert es.get_run(ppo["run_id"], store_path) in es.completed_runs(store_path)  # still in History
        pair = es.latest_compatible_pair(hpa, store_path)
        assert pair["hpa"]["run_id"] == hpa["run_id"]
        assert pair["ppo"] is None  # the invalidated run must never be offered as the counterpart


class TestActiveRunLiveness:
    """Covers the "Mac restarted, /tmp wiped" bug: a `"running"` record
    whose process is actually dead must never lock the Experiment page's
    ARM/traffic controls forever."""

    def test_dead_pid_is_marked_stale_and_unlocks_controls(self, store_path):
        run_id = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        # A pid that cannot belong to a real live process in this test run.
        dead_pid = 2**31 - 1
        es.update_run(run_id, path=store_path, pid=dead_pid)
        assert es.active_run(store_path) is None
        rec = es.get_run(run_id, store_path)
        assert rec["status"] == es.STALE_STATUS
        assert rec["completed_at"] is not None  # ended, not silently dropped

    def test_missing_status_file_with_no_pid_is_marked_stale(self, store_path, tmp_path):
        """Mirrors the actual production bug: a record created before this
        fix carries neither a pid nor a status_path (the status file lived
        in the system temp dir and was wiped by a host restart), so there
        is no way left to verify it is still running."""
        run_id = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        assert es.active_run(store_path) is None
        assert es.get_run(run_id, store_path)["status"] == es.STALE_STATUS

    def test_missing_status_file_but_pid_present_is_stale(self, store_path, tmp_path):
        run_id = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        missing_status_path = tmp_path / "gone.json"  # never written -- simulates the wiped /tmp file
        es.update_run(run_id, path=store_path, pid=os.getpid(), status_path=str(missing_status_path))
        assert es.active_run(store_path) is None
        assert es.get_run(run_id, store_path)["status"] == es.STALE_STATUS

    def test_stale_status_file_is_marked_stale_even_with_live_pid(self, store_path, tmp_path):
        run_id = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        status_path = tmp_path / "status.json"
        status_path.write_text("{}")
        old = time.time() - (es.STALE_STATUS_FILE_AGE_S + 60)
        os.utime(status_path, (old, old))
        es.update_run(run_id, path=store_path, pid=os.getpid(), status_path=str(status_path))
        assert es.active_run(store_path) is None
        assert es.get_run(run_id, store_path)["status"] == es.STALE_STATUS

    def test_live_pid_and_fresh_status_file_still_locks_controls(self, store_path, tmp_path):
        run_id = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        status_path = tmp_path / "status.json"
        status_path.write_text("{}")  # just written -- fresh mtime
        es.update_run(run_id, path=store_path, pid=os.getpid(), status_path=str(status_path))
        active = es.active_run(store_path)
        assert active is not None
        assert active["run_id"] == run_id
        assert active["status"] == "running"

    def test_stale_marking_never_deletes_run_history(self, store_path):
        run_id = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        es.update_run(run_id, path=store_path, pid=2**31 - 1)
        es.active_run(store_path)
        records = es._load(store_path)
        assert len(records) == 1  # updated in place, never removed
        assert records[0]["run_id"] == run_id

    def test_multiple_stale_records_are_all_cleaned_up(self, store_path):
        r1 = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        r2 = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        r3 = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        assert es.active_run(store_path) is None
        for rid in (r1, r2, r3):
            assert es.get_run(rid, store_path)["status"] == es.STALE_STATUS


class TestDeadPidButActuallyCompletedRegression:
    """Regression test for the exact production incident this phase fixes:
    run `4d4edc8274e6` (PID 20102, profile `demo_constant_cpu_normal`) was
    left recorded `status="running"` after a Streamlit restart orphaned
    its `RealExperimentControllerV2` instance. The process (PID 20102)
    had actually finished successfully -- its OWN status file read
    `{"status": "completed", ..., "summary": {..., "n_samples": 16}}` --
    but the store record's pid was dead AND its status file was stale by
    mtime, so the OLD `active_run()` unconditionally overwrote it with a
    fabricated `STALE_STATUS` verdict, throwing away the true outcome and
    leaving the UI's controls disabled with no way to show what actually
    happened. `active_run()`/`reconcile()` must now check the status
    file's own content FIRST and surface COMPLETED, never INTERRUPTED."""

    def test_dead_pid_and_stale_status_file_but_status_file_says_completed(self, store_path, tmp_path):
        run_id = es.start_run("ppo_v2", "demo_constant_cpu_normal", path=store_path)
        status_path = tmp_path / "status.json"
        summary = {
            "episode_id": "demo_v2_ppo_v2_demo_constant_cpu_normal_1", "arm": "ppo_v2",
            "profile": "demo_constant_cpu_normal", "n_samples": 16, "demo_only": True, "partial": False,
            "samples": [{"time_s": float(i) * 30.0} for i in range(16)],
        }
        status_path.write_text(json.dumps({
            "status": "completed", "stage": "COMPLETED", "arm": "ppo_v2",
            "profile": "demo_constant_cpu_normal", "summary": summary,
        }))
        # A pid that cannot belong to a real live process in this test run
        # (the dead PID 20102 scenario), AND a status file whose mtime is
        # old enough to independently fail the staleness check too -- the
        # OLD code's dead-pid-OR-stale-mtime liveness check alone would
        # have marked this STALE_STATUS without ever looking at the
        # file's content.
        old = time.time() - (es.STALE_STATUS_FILE_AGE_S + 600)
        os.utime(status_path, (old, old))
        dead_pid = 2**31 - 1
        es.update_run(run_id, path=store_path, pid=dead_pid, status_path=str(status_path))

        assert es.is_run_alive(es.get_run(run_id, store_path)) is False  # liveness alone says "not alive"

        active = es.active_run(store_path)  # must NOT fabricate STALE_STATUS over this

        assert active is None  # it is terminal, not "still running" -- but also not STALE
        rec = es.get_run(run_id, store_path)
        assert rec["status"] == "completed"
        assert rec["stage"] == "COMPLETED"
        assert rec["status"] != es.STALE_STATUS
        assert rec["summary"]["n_samples"] == 16
        assert rec["completed_at"] is not None

        # The run now shows up as a real completed run -- History/Comparison
        # can use it -- never silently dropped.
        assert run_id in [r["run_id"] for r in es.completed_runs(store_path)]

    def test_reconcile_reports_the_run_as_newly_terminal_not_newly_stale(self, store_path, tmp_path):
        run_id = es.start_run("ppo_v2", "demo_constant_cpu_normal", path=store_path)
        status_path = tmp_path / "status.json"
        status_path.write_text(json.dumps({"status": "completed", "stage": "COMPLETED", "summary": {"n_samples": 1}}))
        old = time.time() - (es.STALE_STATUS_FILE_AGE_S + 600)
        os.utime(status_path, (old, old))
        es.update_run(run_id, path=store_path, pid=2**31 - 1, status_path=str(status_path))

        result = es.reconcile(store_path)

        assert result["active"] is None
        assert [r["run_id"] for r in result["newly_terminal"]] == [run_id]
        assert result["newly_stale"] == []

    def test_refused_status_file_also_surfaces_instead_of_stale(self, store_path, tmp_path):
        run_id = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        status_path = tmp_path / "status.json"
        status_path.write_text(json.dumps({"status": "refused", "stage": "REFUSED", "error": "controller lock is 'ppo', not 'none'"}))
        old = time.time() - (es.STALE_STATUS_FILE_AGE_S + 600)
        os.utime(status_path, (old, old))
        es.update_run(run_id, path=store_path, pid=2**31 - 1, status_path=str(status_path))

        es.active_run(store_path)

        rec = es.get_run(run_id, store_path)
        assert rec["status"] == "refused"
        assert "controller lock" in rec["error"]

    def test_genuinely_dead_run_with_no_terminal_status_file_still_marked_stale_with_a_reason(self, store_path, tmp_path):
        """The fix must not swallow the real stale case -- a run that is
        ACTUALLY dead (no terminal outcome recorded anywhere) must still
        be marked `STALE_STATUS`, now with a human-readable reason."""
        run_id = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        status_path = tmp_path / "status.json"
        status_path.write_text(json.dumps({"status": "running", "stage": "EPISODE_RUNNING"}))
        old = time.time() - (es.STALE_STATUS_FILE_AGE_S + 600)
        os.utime(status_path, (old, old))
        es.update_run(run_id, path=store_path, pid=2**31 - 1, status_path=str(status_path))

        assert es.active_run(store_path) is None
        rec = es.get_run(run_id, store_path)
        assert rec["status"] == es.STALE_STATUS
        assert rec["error"]  # a reason was recorded, not just a bare label


class TestComparisonPairingAfterReconciliation:
    """Task sec.7: Comparison must only ever pair COMPLETED runs on the
    same workload, and reconciliation (which can now turn a stale-looking
    record into a real COMPLETED one, or a dead one into STALE_STATUS)
    must never change that. `latest_compatible_pair()`/`_same_workload()`
    require `workload_hash`/`workload_seed`/`workload_duration_s` to all
    be non-None and matching -- a real `scripts/run_demo_episode_v2.py`
    run's summary does not populate those fields at all, so two such runs
    sharing only `profile_name` correctly do NOT pair (this is the
    existing, unchanged identity rule -- not something this phase's
    reconciliation fix is meant to change)."""

    def test_reconciled_completed_run_without_workload_identity_is_not_paired(self, store_path, tmp_path):
        status_path = tmp_path / "status.json"
        status_path.write_text(json.dumps({
            "status": "completed", "stage": "COMPLETED", "summary": {"n_samples": 16, "profile": "demo_constant_cpu_normal"},
        }))
        old = time.time() - (es.STALE_STATUS_FILE_AGE_S + 600)
        os.utime(status_path, (old, old))
        run_id = es.start_run("ppo_v2", "demo_constant_cpu_normal", path=store_path)
        es.update_run(run_id, path=store_path, pid=2**31 - 1, status_path=str(status_path))
        es.active_run(store_path)  # reconciliation surfaces the real "completed" outcome
        reconciled_run = es.get_run(run_id, store_path)
        assert reconciled_run["status"] == "completed"

        other = es.start_run("hpa_v2_cpu_memory", "demo_constant_cpu_normal", path=store_path)
        es.update_run(other, path=store_path, status="completed", summary={"n_samples": 16})

        pair = es.latest_compatible_pair(reconciled_run, path=store_path)
        # Neither carries workload_hash/seed/duration -- same profile is
        # not enough on its own to be treated as the same workload.
        assert pair["ppo"] is None or pair["ppo"]["run_id"] != reconciled_run["run_id"] or pair["hpa"] is None

    def test_interrupted_run_never_appears_in_a_pair_even_if_workload_identity_set(self, store_path):
        run_id = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        es.update_run(
            run_id, path=store_path, pid=2**31 - 1, status=es.STALE_STATUS, stage="INTERRUPTED",
            workload_hash="h1", workload_seed=42, workload_duration_s=40.0,
        )
        completed = es.start_run("hpa_v2_cpu_memory", "demo_cpu_short", path=store_path)
        es.update_run(
            completed, path=store_path, status="completed",
            workload_hash="h1", workload_seed=42, workload_duration_s=40.0, summary={},
        )

        rec = es.get_run(run_id, store_path)
        pair = es.latest_compatible_pair(rec, path=store_path)
        assert pair["ppo"] is None  # the INTERRUPTED run itself is never offered as "the" ppo result
        assert pair["hpa"]["run_id"] == completed  # its genuinely completed counterpart still is


class TestExpectedDurationLiveness:
    def test_elapsed_past_expected_duration_plus_grace_is_not_alive(self, store_path):
        run_id = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        es.update_run(
            run_id, path=store_path, pid=os.getpid(), expected_duration_s=10.0,
            started_at=time.time() - (10.0 + 301.0),
        )
        rec = es.get_run(run_id, store_path)
        assert es.is_run_alive(rec) is False

    def test_elapsed_within_expected_duration_plus_grace_is_alive(self, store_path):
        run_id = es.start_run("ppo_v2", "demo_cpu_short", path=store_path)
        es.update_run(
            run_id, path=store_path, pid=os.getpid(), expected_duration_s=600.0,
            started_at=time.time() - 10.0,
        )
        rec = es.get_run(run_id, store_path)
        assert es.is_run_alive(rec) is True


class TestStoreLocationAndMigration:
    def test_store_path_lives_under_persistent_data_demo_ui(self):
        assert es.STORE_PATH.parts[-3:-1] == ("persistent-data", "demo_ui")

    def test_legacy_store_is_migrated_once(self, tmp_path, monkeypatch):
        legacy = tmp_path / "legacy_runs.json"
        legacy.write_text(json.dumps([{"run_id": "legacy1", "started_at": 1.0, "status": "completed"}]))
        new_path = tmp_path / "new_runs.json"
        monkeypatch.setattr(es, "_LEGACY_STORE_PATH", legacy)
        monkeypatch.setattr(es, "STORE_PATH", new_path)

        es._migrate_legacy_store_once()

        assert new_path.exists()
        migrated = es._load(new_path)
        assert migrated[0]["run_id"] == "legacy1"

    def test_migration_never_overwrites_an_existing_new_store(self, tmp_path, monkeypatch):
        legacy = tmp_path / "legacy_runs.json"
        legacy.write_text(json.dumps([{"run_id": "legacy1", "started_at": 1.0, "status": "completed"}]))
        new_path = tmp_path / "new_runs.json"
        new_path.write_text(json.dumps([{"run_id": "already_here", "started_at": 2.0, "status": "completed"}]))
        monkeypatch.setattr(es, "_LEGACY_STORE_PATH", legacy)
        monkeypatch.setattr(es, "STORE_PATH", new_path)

        es._migrate_legacy_store_once()

        migrated = es._load(new_path)
        assert [r["run_id"] for r in migrated] == ["already_here"]
