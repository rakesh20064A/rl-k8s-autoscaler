"""tests/unit/test_training_contract_v2.py

V2 Phase 2.1 PART 9/10: proves the V2 training namespace is fully distinct
from V1 (no path collisions), the fresh-policy-by-default guard works, and
model metadata contains every required field. No training is executed by
this file or by the module it tests.
"""

from __future__ import annotations

import pytest

from autoscaler.src.rl_env.training_contract_v2 import (
    RL_V2_CONFIG_PATH,
    V1_MODEL_PATH,
    V1_TRAINING_LEDGER,
    V2_CHECKPOINT_DIR,
    V2_MODEL_DIR,
    V2_MODEL_PATH,
    V2_TENSORBOARD_DIR,
    V2_TRAINING_DATA_DIR,
    V2_TRAINING_LEDGER,
    V2_TRAINING_PROFILE_ALLOWLIST,
    V1ArtifactProtectionError,
    assert_fresh_policy_or_justified,
    assert_v1_artifacts_untouched,
    build_v2_model_metadata,
)


class TestNamespaceSeparationFromV1:
    def test_v2_model_path_distinct_from_v1(self):
        assert V2_MODEL_PATH != V1_MODEL_PATH
        assert V2_MODEL_PATH.resolve() != V1_MODEL_PATH.resolve()
        assert "ppo_v2" in str(V2_MODEL_DIR)
        assert "ppo_v2" not in str(V1_MODEL_PATH)

    def test_v2_training_ledger_distinct_from_v1(self):
        assert V2_TRAINING_LEDGER != V1_TRAINING_LEDGER
        assert "rl_training_v2" in str(V2_TRAINING_LEDGER)
        assert "rl_training_v2" not in str(V1_TRAINING_LEDGER)

    def test_v2_checkpoint_and_tensorboard_dirs_under_v2_namespace(self):
        assert str(V2_CHECKPOINT_DIR).startswith(str(V2_TRAINING_DATA_DIR))
        assert str(V2_TENSORBOARD_DIR).startswith(str(V2_TRAINING_DATA_DIR))

    def test_rl_v2_config_path_is_the_v2_file_not_frozen_v1(self):
        assert RL_V2_CONFIG_PATH.name == "rl_v2.yaml"

    def test_no_v2_path_is_a_substring_match_for_a_v1_path(self):
        v1_paths = {str(V1_MODEL_PATH), str(V1_TRAINING_LEDGER)}
        v2_paths = {str(V2_MODEL_PATH), str(V2_TRAINING_LEDGER), str(V2_CHECKPOINT_DIR), str(V2_TENSORBOARD_DIR)}
        assert v1_paths.isdisjoint(v2_paths)


class TestV1ArtifactProtection:
    def test_writing_to_v1_model_path_is_refused(self):
        with pytest.raises(V1ArtifactProtectionError):
            assert_v1_artifacts_untouched([V1_MODEL_PATH])

    def test_writing_to_v1_training_ledger_is_refused(self):
        with pytest.raises(V1ArtifactProtectionError):
            assert_v1_artifacts_untouched([V1_TRAINING_LEDGER])

    def test_writing_to_v2_paths_is_allowed(self):
        assert_v1_artifacts_untouched([V2_MODEL_PATH, V2_TRAINING_LEDGER])  # must not raise

    def test_mixed_list_with_one_v1_path_still_refused(self):
        with pytest.raises(V1ArtifactProtectionError):
            assert_v1_artifacts_untouched([V2_MODEL_PATH, V1_MODEL_PATH])


class TestFreshPolicyDefault:
    def test_no_justification_is_fine_fresh_default(self):
        assert_fresh_policy_or_justified(None)  # must not raise

    def test_empty_string_justification_rejected(self):
        with pytest.raises(ValueError):
            assert_fresh_policy_or_justified("")

    def test_whitespace_only_justification_rejected(self):
        with pytest.raises(ValueError):
            assert_fresh_policy_or_justified("   ")

    def test_real_justification_accepted(self):
        assert_fresh_policy_or_justified("Warm-starting from V1 CPU-only policy approved per <doc>")  # must not raise


class TestModelMetadata:
    def test_all_required_fields_present(self):
        md = build_v2_model_metadata(
            training_seed=42, ppo_hyperparameters={"learning_rate": 3e-4, "n_steps": 2048},
        )
        d = md.to_dict()
        required = {
            "controller_generation", "observation_contract_hash", "reward_contract_hash",
            "workload_contract", "training_seed", "ppo_hyperparameters",
            "safe_inference_epsilon_status", "training_profiles", "git_commit",
        }
        assert required.issubset(d.keys())

    def test_controller_generation_is_v2(self):
        md = build_v2_model_metadata(training_seed=1, ppo_hyperparameters={})
        assert md.controller_generation == "v2"

    def test_workload_contract_is_multi_resource_v2(self):
        md = build_v2_model_metadata(training_seed=1, ppo_hyperparameters={})
        assert md.workload_contract == "multi_resource_v2"

    def test_epsilon_status_is_provisional(self):
        md = build_v2_model_metadata(training_seed=1, ppo_hyperparameters={})
        assert md.safe_inference_epsilon_status == "provisional_unvalidated"

    def test_default_training_profiles_is_full_allowlist(self):
        md = build_v2_model_metadata(training_seed=1, ppo_hyperparameters={})
        assert set(md.training_profiles) == set(V2_TRAINING_PROFILE_ALLOWLIST)

    def test_non_allowlisted_profile_rejected(self):
        with pytest.raises(ValueError):
            build_v2_model_metadata(
                training_seed=1, ppo_hyperparameters={}, training_profiles=["some_unlisted_v1_profile"],
            )

    def test_subset_of_allowlist_accepted(self):
        md = build_v2_model_metadata(
            training_seed=1, ppo_hyperparameters={}, training_profiles=["constant_cpu", "constant_memory"],
        )
        assert md.training_profiles == ["constant_cpu", "constant_memory"]

    def test_transfer_learning_defaults_to_none_fresh(self):
        md = build_v2_model_metadata(training_seed=1, ppo_hyperparameters={})
        assert md.transfer_learning_justification is None

    def test_explicit_transfer_learning_justification_recorded(self):
        md = build_v2_model_metadata(
            training_seed=1, ppo_hyperparameters={}, transfer_learning_justification="justified reason",
        )
        assert md.transfer_learning_justification == "justified reason"

    def test_empty_transfer_learning_justification_rejected(self):
        with pytest.raises(ValueError):
            build_v2_model_metadata(training_seed=1, ppo_hyperparameters={}, transfer_learning_justification="")

    def test_contract_hashes_are_sha256_hex_strings(self):
        md = build_v2_model_metadata(training_seed=1, ppo_hyperparameters={})
        assert len(md.observation_contract_hash) == 64
        assert len(md.reward_contract_hash) == 64
        int(md.observation_contract_hash, 16)  # raises ValueError if not valid hex
        int(md.reward_contract_hash, 16)

    def test_training_seed_recorded_exactly(self):
        md = build_v2_model_metadata(training_seed=12345, ppo_hyperparameters={})
        assert md.training_seed == 12345

    def test_ppo_hyperparameters_recorded_as_a_copy(self):
        hp = {"learning_rate": 3e-4}
        md = build_v2_model_metadata(training_seed=1, ppo_hyperparameters=hp)
        hp["learning_rate"] = 999  # mutating the original must not affect the recorded metadata
        assert md.ppo_hyperparameters["learning_rate"] == 3e-4
