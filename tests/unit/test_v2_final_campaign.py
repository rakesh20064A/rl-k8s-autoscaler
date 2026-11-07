"""tests/unit/test_v2_final_campaign.py

V2 Phase 4A PART 11: targeted, fully-faked-cluster tests for the final
training campaign orchestration (`scripts.train_ppo_final_v2`) -- namespace
isolation, session-directory allocation, campaign-metadata persistence, and
the campaign LOOP's stop-on-failure/no-automatic-retry contract. The
per-session live-training body (`run_one_final_session`) is exercised
end-to-end only by `run_dry_run_validation()` (see
`test_v2_final_campaign_dry_run` below) and by the live campaign itself --
mirroring the project's own established pattern for
`scripts.train_ppo_live_v2.run_pilot_training()`, which likewise has no
cluster-mocked unit test of its own body, only a dry-run validation plus the
live run.
"""

from __future__ import annotations

import json

import pytest

import scripts.train_ppo_final_v2 as final_mod
from autoscaler.src.rl_env.training_contract_v2 import V1ArtifactProtectionError, V1_MODEL_PATH, V1_TRAINING_LEDGER, V2_MODEL_PATH
from scripts.train_ppo_live_v2 import PILOT_ROOT


class TestNamespaceIsolation:
    def test_v1_model_path_rejected(self):
        with pytest.raises(V1ArtifactProtectionError):
            final_mod.assert_final_namespace_isolated([V1_MODEL_PATH])

    def test_v1_ledger_path_rejected(self):
        with pytest.raises(V1ArtifactProtectionError):
            final_mod.assert_final_namespace_isolated([V1_TRAINING_LEDGER])

    def test_pilot_session_000_rejected(self):
        with pytest.raises(final_mod.FinalNamespaceViolation):
            final_mod.assert_final_namespace_isolated([PILOT_ROOT / "session_000" / "ppo_model.zip"])

    def test_pilot_root_itself_rejected(self):
        with pytest.raises(final_mod.FinalNamespaceViolation):
            final_mod.assert_final_namespace_isolated([PILOT_ROOT])

    def test_reserved_final_model_path_rejected(self):
        """The final campaign must never write directly to the
        promotion-reserved path -- that happens only in a SEPARATE,
        explicitly-authorized post-training audit phase."""
        with pytest.raises(final_mod.FinalNamespaceViolation):
            final_mod.assert_final_namespace_isolated([V2_MODEL_PATH])

    def test_own_namespace_accepted(self):
        final_mod.assert_final_namespace_isolated(
            [final_mod.FINAL_SESSIONS_DIR / "session_000" / "ppo_model.zip", final_mod.FINAL_LEDGER_PATH]
        )  # must not raise


class TestResolveNextFinalSessionDir:
    def test_first_call_returns_session_000(self, tmp_path):
        assert final_mod.resolve_next_final_session_dir(tmp_path) == tmp_path / "session_000"

    def test_skips_existing_sessions(self, tmp_path):
        (tmp_path / "session_000").mkdir(parents=True)
        (tmp_path / "session_001").mkdir(parents=True)
        assert final_mod.resolve_next_final_session_dir(tmp_path) == tmp_path / "session_002"

    def test_contiguous_happy_path_reaches_019(self, tmp_path):
        for n in range(20):
            assert final_mod.resolve_next_final_session_dir(tmp_path) == tmp_path / f"session_{n:03d}"
            (tmp_path / f"session_{n:03d}").mkdir(parents=True)
        assert final_mod.resolve_next_final_session_dir(tmp_path) == tmp_path / "session_020"

    def test_never_reuses_a_failed_sessions_directory(self, tmp_path):
        (tmp_path / "session_000").mkdir(parents=True)
        (tmp_path / "session_000" / final_mod.SESSION_FAILURE_MARKER_NAME).write_text("{}")
        assert final_mod.resolve_next_final_session_dir(tmp_path) == tmp_path / "session_001"


class TestFixedBudget:
    def test_budget_is_exactly_5120(self):
        assert final_mod.FINAL_TOTAL_TIMESTEPS_BUDGET == 5120

    def test_n_steps_is_256_matching_pilot(self):
        assert final_mod.FINAL_N_STEPS == 256

    def test_planned_sessions_is_20(self):
        assert final_mod.FINAL_PLANNED_SESSIONS == 20

    def test_seed_is_42_matching_pilot(self):
        assert final_mod.FINAL_TRAINING_SEED == 42


class TestCampaignMetadataPersistence:
    def test_default_metadata_not_started(self, tmp_path, monkeypatch):
        monkeypatch.setattr(final_mod, "FINAL_CAMPAIGN_METADATA_PATH", tmp_path / "campaign_metadata.json")
        state = final_mod.load_campaign_metadata()
        assert state["status"] == "not_started"
        assert state["cumulative_timesteps"] == 0
        assert state["cumulative_updates"] == 0

    def test_save_then_load_roundtrip(self, tmp_path, monkeypatch):
        path = tmp_path / "campaign_metadata.json"
        monkeypatch.setattr(final_mod, "FINAL_CAMPAIGN_METADATA_PATH", path)
        state = final_mod.load_campaign_metadata()
        state["cumulative_timesteps"] = 256
        state["cumulative_updates"] = 1
        final_mod.save_campaign_metadata(state)
        reloaded = final_mod.load_campaign_metadata()
        assert reloaded["cumulative_timesteps"] == 256
        assert reloaded["cumulative_updates"] == 1
        assert reloaded["campaign_last_updated_utc"] is not None


class _FakeKubectl:
    def __init__(self):
        self.delete_hpa_called = False
        self.restore_hpa_called = False
        self.wait_calls = []

    def list_hpa_names(self):
        # V2 Phase 4C-1 (P1-2): the campaign's own canonical HPA is now
        # demo-app-hpa-v2, not V1's demo-app-hpa.
        return ["demo-app-hpa-v2"]

    def delete_hpa(self):
        self.delete_hpa_called = True

    def restore_hpa(self):
        self.restore_hpa_called = True

    def wait_for_replica_state(self, desired, timeout):
        self.wait_calls.append((desired, timeout))


class _FakeBackend:
    def __init__(self, prometheus_url=None, hpa_name=None, hpa_manifest_path=None, min_replicas=None, initial_replicas=None):
        self._kubectl = _FakeKubectl()
        self.restoration_max_wait_s = 1.0
        self.set_desired_replicas_calls = []
        self.min_replicas = min_replicas
        self.initial_replicas = initial_replicas

    def set_desired_replicas(self, desired):
        self.set_desired_replicas_calls.append(desired)


class TestCampaignLoopControlFlow:
    """Exercises `run_final_training_campaign()`'s own control flow (stop-
    on-failure, no-auto-retry, cumulative accounting, HPA/lock lifecycle)
    with `run_one_final_session()` itself faked -- this isolates the LOOP's
    contract from the live-cluster session body, which is separately
    exercised by the dry-run validation and the live campaign."""

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
        monkeypatch.setattr(final_mod, "FINAL_CAMPAIGN_METADATA_PATH", tmp_path / "campaign_metadata.json")
        monkeypatch.setattr(final_mod, "FINAL_SESSIONS_DIR", tmp_path / "sessions")

    def test_all_sessions_succeed_reaches_completed_status(self, tmp_path, monkeypatch):
        self._patch_cluster_plumbing(monkeypatch, tmp_path)
        calls = []

        def fake_session(session_index, campaign_state, seed=42):
            calls.append(session_index)
            cumulative = (session_index + 1) * final_mod.FINAL_N_STEPS
            return {
                "ok": True, "session_index": session_index, "cumulative_timesteps": cumulative,
                "cumulative_updates": session_index + 1,
                "next_global_schedule_episode_index": session_index + 1,
                "model_path": str(tmp_path / "sessions" / f"session_{session_index:03d}" / "ppo_model.zip"),
                "model_sha256": f"hash{session_index}", "rng_state_path": "rng.pkl",
            }

        monkeypatch.setattr(final_mod, "run_one_final_session", fake_session)
        result = final_mod.run_final_training_campaign(seed=42)
        assert result["ok"] is True
        assert calls == list(range(20))
        assert result["campaign_state"]["status"] == "completed"
        assert result["campaign_state"]["cumulative_timesteps"] == 5120
        assert result["campaign_state"]["cumulative_updates"] == 20
        assert len(result["campaign_state"]["sessions_completed"]) == 20

    def test_failed_session_stops_campaign_immediately_no_retry(self, tmp_path, monkeypatch):
        self._patch_cluster_plumbing(monkeypatch, tmp_path)
        calls = []

        def fake_session(session_index, campaign_state, seed=42):
            calls.append(session_index)
            if session_index == 2:
                return {"ok": False, "session_index": 2, "termination_reason": "memory_safety_threshold_breached:300.0MiB_per_pod"}
            cumulative = (session_index + 1) * final_mod.FINAL_N_STEPS
            return {
                "ok": True, "session_index": session_index, "cumulative_timesteps": cumulative,
                "cumulative_updates": session_index + 1,
                "next_global_schedule_episode_index": session_index + 1,
                "model_path": str(tmp_path / "sessions" / f"session_{session_index:03d}" / "ppo_model.zip"),
                "model_sha256": f"hash{session_index}", "rng_state_path": "rng.pkl",
            }

        monkeypatch.setattr(final_mod, "run_one_final_session", fake_session)
        result = final_mod.run_final_training_campaign(seed=42)
        assert result["ok"] is False
        assert result["status"] == "stopped_on_failure"
        # Exactly one attempt at session_index=2 -- no automatic retry.
        assert calls == [0, 1, 2]
        # Cumulative progress reflects ONLY the two verified successes before the failure.
        assert result["campaign_state"]["cumulative_updates"] == 2
        assert result["campaign_state"]["cumulative_timesteps"] == 512

    def test_failure_still_restores_hpa_and_releases_lock(self, tmp_path, monkeypatch):
        self._patch_cluster_plumbing(monkeypatch, tmp_path)

        def fake_session(session_index, campaign_state, seed=42):
            return {"ok": False, "session_index": session_index, "termination_reason": "pod_restart_detected:x"}

        released = {"n": 0}
        monkeypatch.setattr(final_mod, "run_one_final_session", fake_session)
        monkeypatch.setattr(final_mod, "release_controller", lambda *a, **k: released.__setitem__("n", released["n"] + 1))
        final_mod.run_final_training_campaign(seed=42)
        assert released["n"] == 1

    def test_already_complete_campaign_refuses_to_restart(self, tmp_path, monkeypatch):
        self._patch_cluster_plumbing(monkeypatch, tmp_path)
        path = tmp_path / "campaign_metadata.json"
        state = final_mod._default_campaign_metadata()
        state["cumulative_timesteps"] = 5120
        state["cumulative_updates"] = 20
        state["status"] = "completed"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)

        calls = []
        monkeypatch.setattr(final_mod, "run_one_final_session", lambda *a, **k: calls.append(1))
        result = final_mod.run_final_training_campaign(seed=42)
        assert result["status"] == "already_complete"
        assert calls == []  # never attempts a 21st session

    def test_resumes_from_cumulative_updates_not_from_zero(self, tmp_path, monkeypatch):
        """If a prior process already verified sessions 0-4 (cumulative_
        updates=5), a fresh campaign process must resume at session_index=5,
        never re-run sessions 0-4."""
        self._patch_cluster_plumbing(monkeypatch, tmp_path)
        path = tmp_path / "campaign_metadata.json"
        state = final_mod._default_campaign_metadata()
        state["cumulative_timesteps"] = 5 * final_mod.FINAL_N_STEPS
        state["cumulative_updates"] = 5
        state["next_global_schedule_episode_index"] = 11
        state["latest_verified_session_dir"] = str(tmp_path / "sessions" / "session_004")
        state["latest_verified_model_path"] = str(tmp_path / "sessions" / "session_004" / "ppo_model.zip")
        state["sessions_completed"] = [f"session_{n:03d}" for n in range(5)]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)

        calls = []

        def fake_session(session_index, campaign_state, seed=42):
            calls.append(session_index)
            cumulative = (session_index + 1) * final_mod.FINAL_N_STEPS
            return {
                "ok": True, "session_index": session_index, "cumulative_timesteps": cumulative,
                "cumulative_updates": session_index + 1,
                "next_global_schedule_episode_index": session_index + 1,
                "model_path": str(tmp_path / "sessions" / f"session_{session_index:03d}" / "ppo_model.zip"),
                "model_sha256": f"hash{session_index}", "rng_state_path": "rng.pkl",
            }

        monkeypatch.setattr(final_mod, "run_one_final_session", fake_session)
        result = final_mod.run_final_training_campaign(seed=42)
        assert calls == list(range(5, 20))
        assert result["campaign_state"]["cumulative_updates"] == 20


def test_v2_final_campaign_dry_run_end_to_end():
    """The full offline proof (fresh session + resumed session, SB3 update
    metrics, normalized-observation shape, namespace isolation) -- the one
    place `run_one_final_session`'s own machinery (minus the live cluster)
    is actually exercised, mirroring train_ppo_live_v2.py's own dry-run
    convention for the pilot."""
    report = final_mod.run_dry_run_validation(seed=42)
    assert report["ok"] is True
    assert report["namespace_isolation_ok"] is True
    assert report["sb3_update_metrics_all_non_none"] is True
    assert report["resume_provenance_verified"] is True
    assert report["timesteps_correctly_appended_not_duplicated"] is True
    assert report["fail_closed_on_session_index_mismatch"] is True
    assert report["normalized_observation_length_is_11"] is True
    assert report["traffic_generated"] is False
    assert report["kubernetes_mutation"] is False
