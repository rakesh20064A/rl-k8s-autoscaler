"""tests/unit/test_run_demo_episode_v2.py

V2 Phase 5E: offline tests for scripts/run_demo_episode_v2.py -- the
training/evaluation-protection check (never touches a real cluster) and
the demo-profile firewall. Runs under the scientific .venv (no streamlit
dependency).
"""

from __future__ import annotations

import json

import pytest

import scripts.run_demo_episode_v2 as runner
from autoscaler.src.rl_env.controller_exclusivity import CONTROLLER_NONE


def _fake_episode_result(arm_id: str, profile_name: str, episode_id: str) -> dict:
    """Minimal fake in the exact shape `eval_v2_episodes.py`'s real
    recorders return (and `EpisodeSample.to_dict()` produces) -- enough
    for `run_demo_episode_v2()`'s own post-processing (`_to_status_sample`
    reads `"t_s"`; the summary reads `"episode_id"`/`"samples"`)."""
    return {
        "episode_id": episode_id, "arm_id": arm_id, "profile": profile_name, "pod_restarts_total": 0,
        "samples": [{
            "decision_index": 0, "t_s": 0.0, "timestamp_utc": "fake", "profile": profile_name, "arm_id": arm_id,
            "normalized_observation": [0.5, 0.5, 0.5], "raw_observation": {}, "sample_valid": True,
            "p95_latency_ms": 100.0, "error_rate": 0.0, "request_rate_rps": 1.0,
            "desired_replicas": 2, "ready_replicas": 2, "max_per_pod_memory_mib": 80.0,
            "pod_restart_counts": {}, "hpa_status": None, "ppo_decision": None,
        }],
    }


class TestDemoProfileFirewall:
    def test_resolves_a_real_demo_profile(self):
        path = runner.resolve_demo_profile_path("demo_cpu_short")
        assert path.is_file()

    def test_refuses_an_eval_profile_name(self):
        with pytest.raises(runner.DemoProfileFirewallViolation):
            runner.resolve_demo_profile_path("eval_cpu_step")

    def test_refuses_a_training_profile_name(self):
        with pytest.raises(runner.DemoProfileFirewallViolation):
            runner.resolve_demo_profile_path("constant_cpu")

    def test_refuses_unknown_name(self):
        with pytest.raises(runner.DemoProfileFirewallViolation):
            runner.resolve_demo_profile_path("not_a_real_profile")


class TestTrainingProtectionCheck:
    def test_ok_when_everything_clear(self):
        result = runner.check_training_protection(
            read_lock_fn=lambda: CONTROLLER_NONE, pgrep_fn=lambda pattern: [], list_hpa_names_fn=lambda: ["demo-app-hpa-v2"],
        )
        assert result["ok"] is True
        assert result["reasons"] == []

    def test_refuses_when_lock_is_not_none(self):
        result = runner.check_training_protection(
            read_lock_fn=lambda: "ppo", pgrep_fn=lambda pattern: [], list_hpa_names_fn=lambda: [],
        )
        assert result["ok"] is False
        assert any("controller lock" in r for r in result["reasons"])

    def test_refuses_when_trainer_process_running(self):
        def fake_pgrep(pattern):
            return ["12345"] if pattern == "train_ppo_final_v2.py" else []

        result = runner.check_training_protection(
            read_lock_fn=lambda: CONTROLLER_NONE, pgrep_fn=fake_pgrep, list_hpa_names_fn=lambda: [],
        )
        assert result["ok"] is False
        assert any("train_ppo_final_v2.py" in r for r in result["reasons"])

    def test_refuses_when_evaluation_campaign_process_running(self):
        def fake_pgrep(pattern):
            return ["54321"] if pattern == "evaluate_v2_campaign.py" else []

        result = runner.check_training_protection(
            read_lock_fn=lambda: CONTROLLER_NONE, pgrep_fn=fake_pgrep, list_hpa_names_fn=lambda: [],
        )
        assert result["ok"] is False
        assert any("evaluate_v2_campaign.py" in r for r in result["reasons"])

    def test_refuses_when_unexpected_hpa_present(self):
        result = runner.check_training_protection(
            read_lock_fn=lambda: CONTROLLER_NONE, pgrep_fn=lambda pattern: [],
            list_hpa_names_fn=lambda: ["demo-app-hpa-v2", "demo-app-hpa"],
        )
        assert result["ok"] is False
        assert any("unexpected HPA" in r for r in result["reasons"])

    def test_skips_hpa_check_when_no_lister_injected(self):
        result = runner.check_training_protection(
            read_lock_fn=lambda: CONTROLLER_NONE, pgrep_fn=lambda pattern: [], list_hpa_names_fn=None,
        )
        assert result["ok"] is True

    def test_reports_all_failing_reasons_at_once(self):
        result = runner.check_training_protection(
            read_lock_fn=lambda: "cpu_memory_hpa", pgrep_fn=lambda pattern: ["1"],
            list_hpa_names_fn=lambda: ["some-other-hpa"],
        )
        assert result["ok"] is False
        assert len(result["reasons"]) == 4  # lock + 2 processes + unexpected HPA


class TestRunDemoEpisodeV2RefusesWithoutClusterChange:
    def test_refuses_and_writes_refused_status_without_calling_any_episode_fn(self, tmp_path):
        status_file = tmp_path / "status.json"
        calls = {"n": 0}

        def failing_protection():
            return {"ok": False, "reasons": ["controller lock is 'ppo', not 'none'"]}

        with pytest.raises(runner.DemoRefusedError):
            runner.run_demo_episode_v2(
                "ppo_v2", "demo_cpu_short", status_file=status_file, samples_jsonl=None,
                protection_check_fn=failing_protection,
            )
        assert calls["n"] == 0  # no episode function was ever reached

        import json

        status = json.loads(status_file.read_text())
        assert status["status"] == "refused"
        assert "controller lock" in status["error"]

    def test_invalid_arm_raises_before_any_protection_check(self, tmp_path):
        with pytest.raises(ValueError):
            runner.run_demo_episode_v2("not_a_real_arm", "demo_cpu_short")


class TestDemoArmsAndAllowlist:
    def test_only_two_arms_selectable(self):
        assert runner.DEMO_ARMS == ("ppo_v2", "hpa_v2_cpu_memory")

    def test_demo_profile_allowlist_includes_the_three_legacy_short_profiles(self):
        assert {"demo_cpu_short", "demo_memory_short", "demo_mixed_short"}.issubset(
            set(runner.DEMO_PROFILE_ALLOWLIST)
        )

    def test_demo_profile_allowlist_includes_the_36_pattern_resource_load_catalog(self):
        from scripts.generate_demo_profiles import ALL_PROFILE_NAMES

        assert set(ALL_PROFILE_NAMES).issubset(set(runner.DEMO_PROFILE_ALLOWLIST))
        assert len(ALL_PROFILE_NAMES) == 36


class TestPpoRunRemovesIdleHpaFirstAndAlwaysRestoresIdleState:
    """V2 Phase 9E / CORR-63 lineage: `demo-app-hpa-v2` present is the
    canonical IDLE state, never a conflict to refuse on -- PPO must have
    it removed before it can acquire the controller, and whatever happens
    next (success, an unexpected exception, or an explicit STOP), the
    canonical idle state must always be restored afterward. All fakes --
    no real cluster/kubectl/subprocess access."""

    def _ok_protection(self):
        return {"ok": True, "reasons": []}

    def test_ppo_run_deletes_idle_hpa_before_the_episode_starts_then_succeeds(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(runner, "remove_hpa_before_ppo_start", lambda: calls.append("remove_hpa") or ["demo-app-hpa-v2"])

        def fake_run_ppo(*args, **kwargs):
            calls.append("run_ppo")
            return _fake_episode_result("ppo_v2", "demo_cpu_short", "ep1")

        monkeypatch.setattr(runner, "run_ppo_v2_eval_episode", fake_run_ppo)
        monkeypatch.setattr(runner, "restore_canonical_idle_state", lambda: calls.append("restore") or {"restored": True})
        monkeypatch.setattr(runner, "clear_lock", lambda: calls.append("clear_lock"))

        summary = runner.run_demo_episode_v2(
            "ppo_v2", "demo_cpu_short", status_file=tmp_path / "status.json", samples_jsonl=None,
            protection_check_fn=self._ok_protection,
        )

        assert calls == ["remove_hpa", "run_ppo", "restore", "clear_lock"]
        assert summary["episode_id"] == "ep1"

    def test_successful_ppo_run_cleanup_reapplies_hpa_and_releases_lock(self, tmp_path, monkeypatch):
        cleanup_calls = []
        monkeypatch.setattr(runner, "remove_hpa_before_ppo_start", lambda: [])
        monkeypatch.setattr(
            runner, "run_ppo_v2_eval_episode",
            lambda *a, **k: _fake_episode_result("ppo_v2", "demo_cpu_short", "ep1"),
        )

        def fake_restore():
            cleanup_calls.append("restore")
            return {"restored": True, "hpa_names": ["demo-app-hpa-v2"], "deleted_hpa_names": [], "replicas": 2}

        monkeypatch.setattr(runner, "restore_canonical_idle_state", fake_restore)
        monkeypatch.setattr(runner, "clear_lock", lambda: cleanup_calls.append("clear_lock"))

        runner.run_demo_episode_v2(
            "ppo_v2", "demo_cpu_short", status_file=tmp_path / "status.json", samples_jsonl=None,
            protection_check_fn=self._ok_protection,
        )

        assert cleanup_calls == ["restore", "clear_lock"]

    def test_exception_mid_run_still_restores_canonical_idle_state(self, tmp_path, monkeypatch):
        cleanup_calls = []
        monkeypatch.setattr(runner, "remove_hpa_before_ppo_start", lambda: [])

        def fake_run_ppo_raises(*args, **kwargs):
            raise RuntimeError("injected tooling failure mid-episode")

        monkeypatch.setattr(runner, "run_ppo_v2_eval_episode", fake_run_ppo_raises)
        monkeypatch.setattr(runner, "restore_canonical_idle_state", lambda: cleanup_calls.append("restore") or {"restored": True})
        monkeypatch.setattr(runner, "clear_lock", lambda: cleanup_calls.append("clear_lock"))

        with pytest.raises(RuntimeError, match="injected tooling failure"):
            runner.run_demo_episode_v2(
                "ppo_v2", "demo_cpu_short", status_file=tmp_path / "status.json", samples_jsonl=None,
                protection_check_fn=self._ok_protection,
            )

        assert cleanup_calls == ["restore", "clear_lock"]

    def test_stop_requested_mid_run_still_restores_canonical_idle_state(self, tmp_path, monkeypatch):
        cleanup_calls = []
        monkeypatch.setattr(runner, "remove_hpa_before_ppo_start", lambda: [])

        def fake_run_ppo_stops(*args, **kwargs):
            raise runner.DemoStopRequested("stop requested (signal 15)")

        monkeypatch.setattr(runner, "run_ppo_v2_eval_episode", fake_run_ppo_stops)
        monkeypatch.setattr(runner, "restore_canonical_idle_state", lambda: cleanup_calls.append("restore") or {"restored": True})
        monkeypatch.setattr(runner, "clear_lock", lambda: cleanup_calls.append("clear_lock"))

        status_file = tmp_path / "status.json"
        with pytest.raises(runner.DemoStopRequested):
            runner.run_demo_episode_v2(
                "ppo_v2", "demo_cpu_short", status_file=status_file, samples_jsonl=None,
                protection_check_fn=self._ok_protection,
            )

        assert cleanup_calls == ["restore", "clear_lock"]
        status = json.loads(status_file.read_text())
        assert status["status"] == "stopped"

    def test_hpa_arm_run_never_calls_the_ppo_prestart_helper_but_still_cleans_up(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(runner, "remove_hpa_before_ppo_start", lambda: calls.append("remove_hpa") or [])
        monkeypatch.setattr(
            runner, "run_hpa_v2_cpu_memory_eval_episode",
            lambda *a, **k: _fake_episode_result("hpa_v2_cpu_memory", "demo_cpu_short", "ep2"),
        )
        monkeypatch.setattr(runner, "restore_canonical_idle_state", lambda: calls.append("restore") or {"restored": True})
        monkeypatch.setattr(runner, "clear_lock", lambda: calls.append("clear_lock"))

        summary = runner.run_demo_episode_v2(
            "hpa_v2_cpu_memory", "demo_cpu_short", status_file=tmp_path / "status.json", samples_jsonl=None,
            protection_check_fn=self._ok_protection,
        )

        assert "remove_hpa" not in calls
        assert calls == ["restore", "clear_lock"]
        assert summary["episode_id"] == "ep2"

    def test_unexpected_hpa_present_still_refuses_without_any_mutation(self, tmp_path, monkeypatch):
        """Regression: an HPA other than demo-app-hpa-v2 must still cause
        the run to refuse to start -- this guard must not be weakened by
        the idle-HPA-removal fix."""
        calls = []
        monkeypatch.setattr(runner, "remove_hpa_before_ppo_start", lambda: calls.append("remove_hpa") or [])
        monkeypatch.setattr(runner, "restore_canonical_idle_state", lambda: calls.append("restore") or {"restored": True})
        monkeypatch.setattr(runner, "clear_lock", lambda: calls.append("clear_lock"))

        def failing_protection():
            return runner.check_training_protection(
                read_lock_fn=lambda: CONTROLLER_NONE, pgrep_fn=lambda pattern: [],
                list_hpa_names_fn=lambda: ["demo-app-hpa-v2", "some-other-hpa"],
            )

        status_file = tmp_path / "status.json"
        with pytest.raises(runner.DemoRefusedError):
            runner.run_demo_episode_v2(
                "ppo_v2", "demo_cpu_short", status_file=status_file, samples_jsonl=None,
                protection_check_fn=failing_protection,
            )

        assert calls == []  # no mutation and no cleanup attempted -- refused before anything started
        status = json.loads(status_file.read_text())
        assert status["status"] == "refused"
        assert "unexpected HPA" in status["error"]


class TestTerminalStatusWrittenAfterCleanup:
    """V2 Phase 9F: task sec.3's ordering -- (a) write the run summary,
    (b) restore the canonical idle cluster state, (c) write the terminal
    status LAST. Verified by having the fake cleanup function itself peek
    at the status file mid-cleanup: it must already see the full summary,
    but NOT yet the terminal status/stage -- that must appear only once
    cleanup has returned."""

    def _ok_protection(self):
        return {"ok": True, "reasons": []}

    def test_success_writes_summary_before_cleanup_and_completed_status_after(self, tmp_path, monkeypatch):
        status_file = tmp_path / "status.json"
        monkeypatch.setattr(runner, "remove_hpa_before_ppo_start", lambda: [])
        monkeypatch.setattr(
            runner, "run_ppo_v2_eval_episode",
            lambda *a, **k: _fake_episode_result("ppo_v2", "demo_cpu_short", "ep1"),
        )
        seen_during_cleanup = {}

        def fake_restore():
            seen_during_cleanup["stage"] = json.loads(status_file.read_text())["stage"]
            seen_during_cleanup["has_summary"] = "summary" in json.loads(status_file.read_text())
            return {"restored": True}

        monkeypatch.setattr(runner, "restore_canonical_idle_state", fake_restore)
        monkeypatch.setattr(runner, "clear_lock", lambda: None)

        summary = runner.run_demo_episode_v2(
            "ppo_v2", "demo_cpu_short", status_file=status_file, samples_jsonl=None,
            protection_check_fn=self._ok_protection,
        )

        # Mid-cleanup, the status file already carries the full summary
        # (RE-ATTACH can show it) but is NOT yet terminal -- `stage` is the
        # intermediate "FINISHING" marker, not "COMPLETED".
        assert seen_during_cleanup["stage"] == "FINISHING"
        assert seen_during_cleanup["has_summary"] is True

        final = json.loads(status_file.read_text())
        assert final["status"] == "completed"
        assert final["stage"] == "COMPLETED"
        assert final["cleanup"] == {"restored": True}
        assert final["summary"]["partial"] is False
        assert summary["partial"] is False

    def test_unexpected_exception_still_writes_a_terminal_error_status_with_partial_summary(self, tmp_path, monkeypatch):
        """Regression: an exception not caught by any of the SPECIFIC
        `except` clauses (neither `DemoStopRequested` nor the
        Safety/Tooling/CpuOnly trio) must still leave a real terminal
        status behind -- never a status file stuck mid-run with nothing
        but a dead pid for reconciliation to go on."""
        status_file = tmp_path / "status.json"
        monkeypatch.setattr(runner, "remove_hpa_before_ppo_start", lambda: [])

        def fake_run_ppo_raises_unanticipated(*args, **kwargs):
            raise KeyError("boom -- not one of the specifically-handled exception types")

        monkeypatch.setattr(runner, "run_ppo_v2_eval_episode", fake_run_ppo_raises_unanticipated)
        monkeypatch.setattr(runner, "restore_canonical_idle_state", lambda: {"restored": True})
        monkeypatch.setattr(runner, "clear_lock", lambda: None)

        with pytest.raises(KeyError):
            runner.run_demo_episode_v2(
                "ppo_v2", "demo_cpu_short", status_file=status_file, samples_jsonl=None,
                protection_check_fn=self._ok_protection,
            )

        final = json.loads(status_file.read_text())
        assert final["status"] == "error"
        assert "unexpected failure" in final["error"]
        assert final["summary"]["partial"] is True
        assert final["cleanup"] == {"restored": True}

    def test_stopped_run_summary_is_marked_partial(self, tmp_path, monkeypatch):
        status_file = tmp_path / "status.json"
        monkeypatch.setattr(runner, "remove_hpa_before_ppo_start", lambda: [])

        def fake_run_ppo_stops(*args, **kwargs):
            raise runner.DemoStopRequested("stop requested (signal 15)")

        monkeypatch.setattr(runner, "run_ppo_v2_eval_episode", fake_run_ppo_stops)
        monkeypatch.setattr(runner, "restore_canonical_idle_state", lambda: {"restored": True})
        monkeypatch.setattr(runner, "clear_lock", lambda: None)

        with pytest.raises(runner.DemoStopRequested):
            runner.run_demo_episode_v2(
                "ppo_v2", "demo_cpu_short", status_file=status_file, samples_jsonl=None,
                protection_check_fn=self._ok_protection,
            )

        final = json.loads(status_file.read_text())
        assert final["status"] == "stopped"
        assert final["summary"]["partial"] is True


class TestMainSignalAndAtexitSetup:
    """V2 Phase 9F: SIGINT handling and an `atexit` cleanup safety net,
    alongside the pre-existing SIGTERM handling."""

    def test_main_registers_sigint_sigterm_and_atexit_cleanup(self, monkeypatch):
        calls = {"signals": [], "atexit": []}
        monkeypatch.setattr(runner.signal, "signal", lambda sig, handler: calls["signals"].append((sig, handler)))
        monkeypatch.setattr(runner.atexit, "register", lambda fn: calls["atexit"].append(fn))
        monkeypatch.setattr(runner, "run_demo_episode_v2", lambda *a, **k: {"episode_id": "e1"})

        rc = runner.main(["--arm", "ppo_v2", "--profile", "demo_cpu_short"])

        assert rc == 0
        assert (runner.signal.SIGTERM, runner._raise_stop_requested) in calls["signals"]
        assert (runner.signal.SIGINT, runner._raise_stop_requested) in calls["signals"]
        assert runner._restore_idle_state_safely in calls["atexit"]
