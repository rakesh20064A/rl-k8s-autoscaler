"""tests/unit/test_v2_evaluation_preregistration.py

V2 Phase 5A / CORR-60: offline structural checks on the evaluation
pre-registration record. Created before any evaluation run and before
session_047 exists (training is still live) -- this test only checks the
record's own structure/consistency, never anything about a real result.
"""

from __future__ import annotations

import json
from pathlib import Path

PREREG_PATH = Path("persistent-data/rl_training_v2/final_min2/EVALUATION_PREREGISTRATION.json")
PREREG_MD_PATH = Path("docs/evaluation_v2_preregistration.md")


def _load():
    with open(PREREG_PATH, encoding="utf-8") as f:
        return json.load(f)


class TestFilesExist:
    def test_json_record_exists_and_parses(self):
        record = _load()
        assert record["corr_id"] == "CORR-60"

    def test_readable_markdown_copy_exists(self):
        assert PREREG_MD_PATH.is_file()
        content = PREREG_MD_PATH.read_text(encoding="utf-8")
        assert "CORR-60" in content


class TestModelUnderTest:
    def test_only_session_047_final_checkpoint(self):
        record = _load()
        assert "session_047" in record["model_under_test"]["rule"]


class TestArms:
    def test_exactly_three_arms(self):
        record = _load()
        assert len(record["arms"]) == 3

    def test_arm_ids_present(self):
        record = _load()
        arm_ids = {a["arm_id"] for a in record["arms"]}
        assert arm_ids == {"PPO_V2", "HPA_V2_CPU_MEMORY", "HPA_CPU_ONLY"}

    def test_all_arms_share_replica_bounds(self):
        record = _load()
        for arm in record["arms"]:
            assert arm["min_replicas"] == 2
            assert arm["max_replicas"] == 8

    def test_hpa_arms_reference_existing_manifests(self):
        record = _load()
        for arm in record["arms"]:
            if "manifest" in arm:
                assert Path(arm["manifest"]).is_file(), arm["manifest"]


class TestWorkloads:
    def test_three_eval_profiles_referenced_and_exist(self):
        record = _load()
        profiles = record["workloads"]["profiles"]
        assert len(profiles) == 3
        for p in profiles:
            assert Path(p).is_file()

    def test_three_repeats_per_profile(self):
        record = _load()
        assert record["workloads"]["repeats_per_profile_per_arm"] == 3


class TestEpsilonRule:
    def test_cap_and_floor(self):
        record = _load()
        epsilon = record["epsilon_rule"]
        assert epsilon["cap"] == 0.05
        assert epsilon["floor"] == 0.0

    def test_value_not_yet_computed_is_disclosed(self):
        """Training is still live (session_047 does not exist yet) -- the
        epsilon value must be explicitly disclosed as not-yet-computed,
        never a guessed or silently-defaulted number."""
        record = _load()
        assert "TO BE COMPUTED" in record["epsilon_rule"]["value"]


class TestValidityAndDecisionRules:
    def test_tooling_failure_gets_exactly_one_retry(self):
        record = _load()
        assert "retried" in record["validity_rules"]["tooling_failure"]
        assert "exactly once" in record["validity_rules"]["tooling_failure"]

    def test_safety_termination_never_retried(self):
        record = _load()
        assert "NEVER retried" in record["validity_rules"]["safety_termination"]

    def test_better_requires_all_three_repeats(self):
        record = _load()
        assert "ALL 3" in record["decision_rule"]["claiming_better"]

    def test_no_aggregate_winner_rule_present(self):
        record = _load()
        assert "no_aggregate_winner" in record["decision_rule"]


class TestTrainingContractReferenceMatchesAmendment:
    def test_matches_phase_4ha_amended_budget(self):
        record = _load()
        ref = record["training_contract_reference"]
        assert ref["total_timesteps_budget"] == 12288
        assert ref["total_planned_sessions"] == 48

    def test_trainer_hash_matches_live_file(self):
        import hashlib

        record = _load()
        expected = record["training_contract_reference"]["trainer_hash_at_this_preregistration"]
        actual = hashlib.sha256(Path("scripts/train_ppo_final_v2.py").read_bytes()).hexdigest()
        assert actual == expected
