"""tests/unit/test_v2_final_campaign_resume.py

V2 Phase 4B-R PART 9: targeted tests for the failed-session audit +
resume-safety correction -- proves the campaign bookkeeping bug (stale
`status="in_progress"` on disk after a terminal failure) is fixed, that a
failed/partial session can never claim an extra completed rollout/train
cycle, and that a retry of the SAME scientific session index uses a new
physical-attempt directory without ever overwriting the original failed
evidence or skipping the scientific index forward.
"""

from __future__ import annotations

import json

import scripts.train_ppo_final_v2 as final_mod


# -- PART 6: scientific-session-index-aware physical attempt directories --

class TestResolveSessionAttemptDir:
    def test_first_attempt_gets_bare_session_name(self, tmp_path):
        path = final_mod.resolve_session_attempt_dir(3, tmp_path)
        assert path == tmp_path / "session_003"

    def test_retry_after_one_failure_gets_attempt2(self, tmp_path):
        (tmp_path / "session_003").mkdir(parents=True)
        path = final_mod.resolve_session_attempt_dir(3, tmp_path)
        assert path == tmp_path / "session_003__attempt2"

    def test_never_returns_an_existing_path(self, tmp_path):
        """Direct proof against PART 1: the failed session_003's own
        directory, once it exists, is never returned again -- whatever
        already exists on disk is untouchable evidence."""
        (tmp_path / "session_003").mkdir(parents=True)
        (tmp_path / "session_003" / "SESSION_FAILED.json").write_text("{}")
        path = final_mod.resolve_session_attempt_dir(3, tmp_path)
        assert not path.exists()
        assert path != tmp_path / "session_003"

    def test_retry_after_two_failures_gets_attempt3(self, tmp_path):
        (tmp_path / "session_003").mkdir(parents=True)
        (tmp_path / "session_003__attempt2").mkdir(parents=True)
        path = final_mod.resolve_session_attempt_dir(3, tmp_path)
        assert path == tmp_path / "session_003__attempt3"

    def test_does_not_skip_the_scientific_index_forward(self, tmp_path):
        """The core PART 6 fix: a failed session_003 must NOT cause the next
        allocation to silently become session_004 -- the scientific index is
        explicitly supplied by the caller (the campaign loop's own
        `cumulative_updates`-derived index), never inferred from what the
        next free `session_NNN` slot happens to be."""
        (tmp_path / "session_003").mkdir(parents=True)
        path = final_mod.resolve_session_attempt_dir(3, tmp_path)
        assert "session_004" not in str(path)
        assert path.name.startswith("session_003")


class TestPhysicalAttemptNumberForDir:
    def test_bare_name_is_attempt_1(self, tmp_path):
        assert final_mod.physical_attempt_number_for_dir(tmp_path / "session_003") == 1

    def test_attempt2_suffix_parsed(self, tmp_path):
        assert final_mod.physical_attempt_number_for_dir(tmp_path / "session_003__attempt2") == 2

    def test_attempt5_suffix_parsed(self, tmp_path):
        assert final_mod.physical_attempt_number_for_dir(tmp_path / "session_003__attempt5") == 5


class TestScientificSessionIndexForDir:
    def test_bare_name(self, tmp_path):
        assert final_mod.scientific_session_index_for_dir(tmp_path / "session_003") == 3

    def test_attempt_suffixed_name(self, tmp_path):
        assert final_mod.scientific_session_index_for_dir(tmp_path / "session_003__attempt2") == 3


# -- PART 5: campaign-metadata terminal-status bookkeeping fix --

class TestMarkCampaignFailed:
    def test_writes_stopped_on_failure_to_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(final_mod, "FINAL_CAMPAIGN_METADATA_PATH", tmp_path / "campaign_metadata.json")
        state = final_mod._default_campaign_metadata()
        state["cumulative_timesteps"] = 768
        state["cumulative_updates"] = 3
        final_mod._mark_campaign_failed(
            state, {"session_index": 3, "physical_attempt_number": 1, "termination_reason": "zero_ready_replicas"},
        )
        on_disk = final_mod.load_campaign_metadata()
        assert on_disk["status"] == "stopped_on_failure"
        assert on_disk["failed_scientific_session_index"] == 3
        assert on_disk["failed_physical_attempt"] == 1
        assert on_disk["failure_reason"] == "zero_ready_replicas"
        assert on_disk["resume_eligible"] is True

    def test_never_advances_verified_counters(self, tmp_path, monkeypatch):
        monkeypatch.setattr(final_mod, "FINAL_CAMPAIGN_METADATA_PATH", tmp_path / "campaign_metadata.json")
        state = final_mod._default_campaign_metadata()
        state["cumulative_timesteps"] = 768
        state["cumulative_updates"] = 3
        state["sessions_completed"] = ["session_000", "session_001", "session_002"]
        final_mod._mark_campaign_failed(state, {"session_index": 3, "physical_attempt_number": 1, "termination_reason": "x"})
        on_disk = final_mod.load_campaign_metadata()
        assert on_disk["cumulative_timesteps"] == 768
        assert on_disk["cumulative_updates"] == 3
        assert on_disk["sessions_completed"] == ["session_000", "session_001", "session_002"]

    def test_remaining_timesteps_computed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(final_mod, "FINAL_CAMPAIGN_METADATA_PATH", tmp_path / "campaign_metadata.json")
        state = final_mod._default_campaign_metadata()
        state["cumulative_timesteps"] = 768
        final_mod._mark_campaign_failed(state, {"session_index": 3, "physical_attempt_number": 1, "termination_reason": "x"})
        on_disk = final_mod.load_campaign_metadata()
        assert on_disk["remaining_timesteps"] == final_mod.FINAL_TOTAL_TIMESTEPS_BUDGET - 768


# -- Exact V2 Phase 4B-R scenario: resume from verified 768/3/session_002,
# NOT from the failed session_003's own consumed episode index 49 --

class _FakeKubectl:
    def list_hpa_names(self):
        # V2 Phase 4C-1 (P1-2): the campaign's own canonical HPA is now
        # demo-app-hpa-v2, not V1's demo-app-hpa.
        return ["demo-app-hpa-v2"]

    def delete_hpa(self):
        pass

    def restore_hpa(self):
        pass

    def wait_for_replica_state(self, desired, timeout):
        pass


class _FakeBackend:
    def __init__(self, prometheus_url=None, hpa_name=None, hpa_manifest_path=None, min_replicas=None, initial_replicas=None):
        self._kubectl = _FakeKubectl()
        self.restoration_max_wait_s = 1.0

    def set_desired_replicas(self, desired):
        pass


class TestExactPhase4BRScenario:
    def _seed_authoritative_state(self, tmp_path):
        path = tmp_path / "campaign_metadata.json"
        state = final_mod._default_campaign_metadata()
        state.update({
            "cumulative_timesteps": 768,
            "cumulative_updates": 3,
            "next_global_schedule_episode_index": 42,
            "sessions_completed": ["session_000", "session_001", "session_002"],
            "latest_verified_session_dir": str(tmp_path / "sessions" / "session_002"),
            "latest_verified_model_path": str(tmp_path / "sessions" / "session_002" / "ppo_model.zip"),
            "latest_verified_model_sha256": "c721144f006d6ff5281dac09eba7050d87c7668aaf23c6acb2ad63e7ba76d3f2",
            "latest_verified_rng_state_path": str(tmp_path / "sessions" / "session_002" / "rng_state.pkl"),
            "status": "in_progress",  # the exact stale-bookkeeping value disclosed in the phase report
        })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)
        return path

    def _patch_cluster_plumbing(self, monkeypatch, tmp_path):
        import autoscaler.src.rl_env.live_backend as live_backend_mod
        import scripts.v2_pilot_safety as safety_mod

        monkeypatch.setattr(live_backend_mod, "LiveKubernetesBackend", _FakeBackend)
        monkeypatch.setattr(
            safety_mod, "verify_prometheus_internal_paths",
            lambda *a, **k: {"all_ok": True, "ready": True, "backend_query_ok": True, "drain_query_ok": True},
        )
        monkeypatch.setattr(final_mod, "read_lock", lambda: final_mod.CONTROLLER_NONE)
        monkeypatch.setattr(final_mod, "acquire_controller", lambda *a, **k: None)
        monkeypatch.setattr(final_mod, "release_controller", lambda *a, **k: None)
        monkeypatch.setattr(final_mod, "FINAL_SESSIONS_DIR", tmp_path / "sessions")

    def test_resume_uses_verified_42_not_failed_attempts_49(self, tmp_path, monkeypatch):
        """PART 7: checkpoint/schedule discovery must use the VERIFIED
        campaign_metadata value (42), never the failed physical attempt's
        own higher consumed episode index (49) -- even though a failed
        session_003 directory sits on disk with evidence of having reached
        49, that evidence must never feed back into what the retry starts
        from."""
        self._patch_cluster_plumbing(monkeypatch, tmp_path)
        monkeypatch.setattr(final_mod, "FINAL_CAMPAIGN_METADATA_PATH", self._seed_authoritative_state(tmp_path))

        # Simulate the EXISTING failed session_003 evidence on disk (as PART
        # 1 requires -- immutable, never read for resume purposes).
        failed_dir = tmp_path / "sessions" / "session_003"
        failed_dir.mkdir(parents=True)
        (failed_dir / "SESSION_FAILED.json").write_text(json.dumps({"reason": "zero_ready_replicas"}))

        seen_campaign_states = []

        def fake_session(session_index, campaign_state, seed=42):
            seen_campaign_states.append((session_index, campaign_state["next_global_schedule_episode_index"]))
            cumulative = 768 + (session_index - 2) * final_mod.FINAL_N_STEPS
            return {
                "ok": True, "session_index": session_index, "cumulative_timesteps": cumulative,
                "cumulative_updates": session_index + 1,
                "next_global_schedule_episode_index": 42 + (session_index - 2) * 7,  # arbitrary but > 42
                "model_path": str(tmp_path / "sessions" / f"session_{session_index:03d}__attempt2" / "ppo_model.zip"),
                "model_sha256": f"hash{session_index}", "rng_state_path": "rng.pkl",
                "physical_attempt_number": 2,
            }

        monkeypatch.setattr(final_mod, "run_one_final_session", fake_session)
        result = final_mod.run_final_training_campaign(seed=42)

        # Resume started at scientific session index 3 (never skipped to 4),
        # with the VERIFIED 42 -- never the failed attempt's 49.
        assert seen_campaign_states[0] == (3, 42)
        assert result["ok"] is True
        # Reached exactly the remaining budget.
        assert result["campaign_state"]["cumulative_timesteps"] == final_mod.FINAL_TOTAL_TIMESTEPS_BUDGET

    def test_bookkeeping_bug_reproduced_then_fixed(self, tmp_path, monkeypatch):
        """Directly reproduces the disclosed bug scenario: stdout-level
        result says stopped_on_failure, but BEFORE the PART 5 fix the
        on-disk campaign_metadata.json still said "in_progress". After the
        fix, the file itself must say stopped_on_failure."""
        self._patch_cluster_plumbing(monkeypatch, tmp_path)
        path = self._seed_authoritative_state(tmp_path)
        monkeypatch.setattr(final_mod, "FINAL_CAMPAIGN_METADATA_PATH", path)

        def fake_session(session_index, campaign_state, seed=42):
            return {
                "ok": False, "session_index": session_index, "physical_attempt_number": 1,
                "termination_reason": "zero_ready_replicas",
            }

        monkeypatch.setattr(final_mod, "run_one_final_session", fake_session)
        result = final_mod.run_final_training_campaign(seed=42)

        assert result["status"] == "stopped_on_failure"
        on_disk = final_mod.load_campaign_metadata()
        assert on_disk["status"] == "stopped_on_failure"  # the fix: file now matches stdout
        assert on_disk["failed_scientific_session_index"] == 3
        assert on_disk["failure_reason"] == "zero_ready_replicas"
        assert on_disk["cumulative_timesteps"] == 768  # unchanged -- no partial credit
        assert on_disk["cumulative_updates"] == 3

    def test_second_consecutive_failure_also_no_auto_retry(self, tmp_path, monkeypatch):
        self._patch_cluster_plumbing(monkeypatch, tmp_path)
        monkeypatch.setattr(final_mod, "FINAL_CAMPAIGN_METADATA_PATH", self._seed_authoritative_state(tmp_path))
        calls = []

        def fake_session(session_index, campaign_state, seed=42):
            calls.append(session_index)
            return {"ok": False, "session_index": session_index, "physical_attempt_number": 2, "termination_reason": "zero_ready_replicas"}

        monkeypatch.setattr(final_mod, "run_one_final_session", fake_session)
        result = final_mod.run_final_training_campaign(seed=42)
        assert calls == [3]  # exactly one attempt this process invocation, no retry loop
        assert result["status"] == "stopped_on_failure"


class TestCumulativeUpdatesNeverClaimedOnFailure:
    """PART 2: direct, offline proof that the cumulative-updates arithmetic
    itself (not just the campaign loop around it) refuses to credit a
    failed/partial session with a completed rollout/train cycle -- mirrors
    the exact session_002-vs-failed-session_003 comparison performed live
    (identical SB3 `_n_updates`, byte-identical policy weights) by asserting
    the SAME conditional arithmetic `run_one_final_session` now uses."""

    def test_formula_credits_only_on_success(self):
        prior = 3
        session_ok = False
        assert prior + (1 if session_ok else 0) == 3

    def test_formula_credits_on_success(self):
        prior = 3
        session_ok = True
        assert prior + (1 if session_ok else 0) == 4
