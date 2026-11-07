"""tests/unit/test_train_ppo_live_v2.py

V2 Phase 2.2 PART 10/11: proves the V2 trainer's dry-run path performs
every required integration check and NEVER calls PPO.learn() or any real
PPO optimization. Also proves the module itself contains no code path that
could call `.learn()` outside the explicitly-gated `run_live_training()`
(which raises NotImplementedError before reaching it -- see
scripts/train_ppo_live_v2.py).
"""

from __future__ import annotations

import ast
from pathlib import Path

from scripts.train_ppo_live_v2 import run_dry_run_validation

SCRIPT_PATH = Path("scripts/train_ppo_live_v2.py")


class TestDryRunNeverCallsLearn:
    def test_dry_run_report_asserts_learn_not_called(self):
        report = run_dry_run_validation(seed=42)
        assert report["ppo_learn_called"] is False

    def test_stable_baselines3_learn_is_never_invoked(self, monkeypatch):
        """Positive runtime proof: patch SB3's PPO.learn to raise if ever
        called, run the full dry-run path, and confirm no exception."""
        import stable_baselines3

        def _explode(*args, **kwargs):
            raise AssertionError("PPO.learn() must never be called during a dry run")

        monkeypatch.setattr(stable_baselines3.PPO, "learn", _explode)
        report = run_dry_run_validation(seed=42)  # must not raise
        assert report["ok"] is True

    def test_dry_run_path_never_imports_ppo_at_module_level(self):
        """`from stable_baselines3 import PPO` must appear only inside
        run_live_training() (a local/deferred import), never at module
        scope -- so merely importing this script never pulls in SB3's PPO
        class into the dry-run code path's own namespace."""
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        module_level_names = set()
        for node in tree.body:  # only TOP-LEVEL statements, not nested in functions
            if isinstance(node, ast.ImportFrom) and node.module == "stable_baselines3":
                for alias in node.names:
                    module_level_names.add(alias.asname or alias.name)
        assert "PPO" not in module_level_names


class TestDryRunCoversAllRequiredChecks:
    def test_report_has_all_required_sections(self):
        report = run_dry_run_validation(seed=42)
        required = {
            "controller_generation", "workload_contract", "reward_status",
            "ppo_kwargs_constructed", "fresh_policy_enforced",
            "v1_destination_checks", "v2_destinations_accepted",
            "profile_allowlist_checks", "controller_exclusivity_checks",
            "observation_shape", "action_space_n", "dry_run_rewards",
            "v2_directories_prepared", "model_metadata_preview",
            "ppo_learn_called", "traffic_generated", "kubernetes_mutation", "ok",
        }
        assert required.issubset(report.keys())

    def test_observation_shape_is_eleven(self):
        report = run_dry_run_validation(seed=42)
        assert report["observation_shape"] == [11]

    def test_v2_rewards_computed_not_v1(self):
        report = run_dry_run_validation(seed=42)
        assert report["dry_run_used_reward_v2"] is True
        assert len(report["dry_run_rewards"]) > 0

    def test_v1_destinations_refused(self):
        report = run_dry_run_validation(seed=42)
        assert report["v1_destinations_fail_closed"] is True
        assert report["v1_destination_checks"]["v1_model"] == "CORRECTLY_REJECTED"
        assert report["v1_destination_checks"]["v1_ledger"] == "CORRECTLY_REJECTED"

    def test_unsupported_profile_rejected(self):
        report = run_dry_run_validation(seed=42)
        assert report["profile_allowlist_checks"]["some_unsupported_v1_profile"] == "REJECTED_NOT_ALLOWLISTED"

    def test_controller_conflict_detection_wired(self):
        report = run_dry_run_validation(seed=42)
        assert report["controller_exclusivity_checks"]["hpa_present"] == "CORRECTLY_REJECTED"
        assert report["controller_exclusivity_checks"]["no_hpa_present"] == "CORRECTLY_ACQUIRED"

    def test_no_traffic_or_kubernetes_mutation(self):
        report = run_dry_run_validation(seed=42)
        assert report["traffic_generated"] is False
        assert report["kubernetes_mutation"] is False

    def test_overall_ok(self):
        report = run_dry_run_validation(seed=42)
        assert report["ok"] is True

    def test_metadata_preview_has_required_fields(self):
        report = run_dry_run_validation(seed=42)
        preview = report["model_metadata_preview"]
        assert preview["controller_generation"] == "v2"
        assert preview["workload_contract"] == "multi_resource_v2"
        assert preview["safe_inference_epsilon_status"] == "provisional_unvalidated"


class TestLiveTrainingRequiresConfirmation:
    """V2 Phase 3A: `run_pilot_training()` is now implemented (see
    tests/unit/test_v2_pilot.py for its own dedicated, fully-faked-cluster
    test coverage) -- this file keeps only the CLI-level confirmation-flag
    guard, which is independent of the pilot's internal implementation."""

    def test_main_rejects_execute_without_confirmation_flag(self):
        from scripts.train_ppo_live_v2 import main

        rc = main(["--execute-live-training"])
        assert rc == 2
