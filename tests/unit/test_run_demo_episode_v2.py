"""tests/unit/test_run_demo_episode_v2.py

V2 Phase 5E: offline tests for scripts/run_demo_episode_v2.py -- the
training/evaluation-protection check (never touches a real cluster) and
the demo-profile firewall. Runs under the scientific .venv (no streamlit
dependency).
"""

from __future__ import annotations

import pytest

import scripts.run_demo_episode_v2 as runner
from autoscaler.src.rl_env.controller_exclusivity import CONTROLLER_NONE


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

    def test_demo_profile_allowlist_matches_the_three_short_profiles(self):
        assert set(runner.DEMO_PROFILE_ALLOWLIST) == {"demo_cpu_short", "demo_memory_short", "demo_mixed_short"}
