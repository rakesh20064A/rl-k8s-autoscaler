"""tests/unit/test_rl_live_common.py

Phase 16V: tests for scripts/rl_live_common.py's workload firewall and
governing-hash verification. Pure, offline -- no I/O beyond reading
already-committed project files.
"""

from __future__ import annotations

import pytest

from scripts import rl_live_common as common


class TestWorkloadFirewallTraining:
    @pytest.mark.parametrize("profile", common.TRAINING_PROFILES)
    def test_training_profiles_are_accepted(self, profile):
        common.assert_training_eligible(profile)  # must not raise

    def test_final_holdout_is_rejected_for_training(self):
        with pytest.raises(common.WorkloadFirewallViolation):
            common.assert_training_eligible(common.FINAL_HOLDOUT_PROFILE)

    def test_unseen_eval_is_rejected_for_training(self):
        with pytest.raises(common.WorkloadFirewallViolation):
            common.assert_training_eligible("unseen_eval")

    def test_unknown_profile_is_rejected(self):
        with pytest.raises(common.WorkloadFirewallViolation):
            common.assert_training_eligible("some_made_up_profile")


class TestWorkloadFirewallFinalEvaluation:
    def test_final_holdout_is_accepted_for_evaluation(self):
        common.assert_final_evaluation_eligible(common.FINAL_HOLDOUT_PROFILE)  # must not raise

    @pytest.mark.parametrize("profile", common.TRAINING_PROFILES)
    def test_training_profiles_are_rejected_for_final_evaluation(self, profile):
        with pytest.raises(common.WorkloadFirewallViolation):
            common.assert_final_evaluation_eligible(profile)

    def test_unseen_eval_is_rejected_for_final_evaluation(self):
        with pytest.raises(common.WorkloadFirewallViolation):
            common.assert_final_evaluation_eligible("unseen_eval")


class TestGoverningHashes:
    def test_all_governing_hashes_currently_match(self):
        # This proves the Phase16U artifacts are byte-identical to what
        # Phase16V read at implementation time -- if this test ever fails
        # in the future, it means a Phase16U artifact or a frozen upstream
        # config/doc changed without a disclosed amendment.
        checks = common.verify_governing_hashes()
        mismatches = [c for c in checks if not c["match"]]
        assert mismatches == []

    def test_verify_governing_hashes_returns_nonempty_list(self):
        checks = common.verify_governing_hashes()
        assert len(checks) >= 10

    def test_load_frozen_ppo_hyperparameters_matches_expected_shape(self):
        hp = common.load_frozen_ppo_hyperparameters()
        assert hp["hyperparameters"]["n_steps"] == 256
        assert hp["hyperparameters"]["total_timesteps"] == 12288
        assert hp["hyperparameters"]["seeds"] == [42]

    def test_load_workload_split_matches_expected_shape(self):
        split = common.load_workload_split()
        names = {p["name"] for p in split["training_and_development_workloads"]["profiles"]}
        assert names == set(common.TRAINING_PROFILES)
        assert split["final_evaluation_workload"]["seed"] == 777
