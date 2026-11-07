"""Tests for the new V2 config namespace (configs/rl_v2.yaml,
configs/reward_v2.yaml): must load identically to V1 through the SAME,
UNMODIFIED loader (autoscaler/src/rl_env/config.py), and V1's own files
must be completely unaffected by the new files' existence.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from autoscaler.src.rl_env.config import (
    DEFAULT_REWARD_YAML,
    DEFAULT_RL_YAML,
    load_env_config_from_yaml,
)

RL_V2_PATH = Path("configs/rl_v2.yaml")
REWARD_V2_PATH = Path("configs/reward_v2.yaml")


def test_v2_files_exist_alongside_v1():
    assert RL_V2_PATH.exists()
    assert REWARD_V2_PATH.exists()
    assert DEFAULT_RL_YAML.exists()
    assert DEFAULT_REWARD_YAML.exists()


def test_v2_config_loads_without_loader_changes():
    """The existing, UNMODIFIED load_env_config_from_yaml() must accept
    rl_v2.yaml directly -- no version-check extension was needed because
    rl_contract_version is intentionally left at 2 (the observation/action
    contract has not changed yet). Paired with the FROZEN reward.yaml here,
    not reward_v2.yaml: since V2 Phase 2.1, reward_v2.yaml holds a
    genuinely different schema (autoscaler.src.rl_env.reward_v2's own
    RewardConfigV2, loaded via load_reward_config_v2_from_yaml(), NOT this
    loader) -- see test_reward_v2.py for that module's own coverage."""
    cfg = load_env_config_from_yaml(rl_yaml_path=RL_V2_PATH, reward_yaml_path=DEFAULT_REWARD_YAML)
    assert cfg is not None


def test_v2_config_produces_identical_env_config_to_v1():
    """Proves rl_v2.yaml's observation/action/safe_action/identity
    additions are completely inert to the frozen loader -- paired with the
    frozen reward.yaml on both sides so this isolates rl_v2.yaml's own
    effect (see test_v2_config_loads_without_loader_changes for why
    reward_v2.yaml is no longer used with this loader)."""
    v1 = load_env_config_from_yaml()
    v2 = load_env_config_from_yaml(rl_yaml_path=RL_V2_PATH, reward_yaml_path=DEFAULT_REWARD_YAML)
    assert dataclasses.asdict(v1) == dataclasses.asdict(v2)


def test_v1_config_files_unchanged_by_v2_additions():
    """V2 files must never be edits-in-disguise of V1 -- re-parses V1 and
    confirms its own contract versions/observation entries are exactly
    what they always were."""
    with open(DEFAULT_RL_YAML, encoding="utf-8") as f:
        rl_v1 = yaml.safe_load(f)
    assert rl_v1["rl_contract_version"] == 2
    assert "safe_action" not in rl_v1  # the V2 addition must not leak into V1

    with open(DEFAULT_REWARD_YAML, encoding="utf-8") as f:
        reward_v1 = yaml.safe_load(f)
    assert reward_v1["reward_contract_version"] == 2


def test_v2_rl_config_has_safe_action_block():
    with open(RL_V2_PATH, encoding="utf-8") as f:
        rl_v2 = yaml.safe_load(f)
    assert rl_v2["safe_action"]["near_tie_epsilon"] == 0.05
    assert rl_v2["safe_action"]["epsilon_status"] == "provisional_unvalidated"


def test_v2_rl_config_has_identity_metadata():
    """V2 Phase 1.1: explicit identity fields, additive-only -- must not
    affect load_env_config_from_yaml() (see
    test_v2_config_produces_identical_env_config_to_v1) and must not bump
    rl_contract_version (the observation/reward contract itself has not
    changed yet)."""
    with open(RL_V2_PATH, encoding="utf-8") as f:
        rl_v2 = yaml.safe_load(f)
    assert rl_v2["controller_generation"] == "v2"
    assert rl_v2["workload_contract"] == "multi_resource_v2"
    assert rl_v2["inference_policy"] == "safe_near_tie_hold"
    assert rl_v2["reward_status"] == "v2_designed_pending_training_validation"
    assert rl_v2["rl_contract_version"] == 2


def test_v2_rl_config_preserves_observation_order_and_action_map():
    with open(DEFAULT_RL_YAML, encoding="utf-8") as f:
        rl_v1 = yaml.safe_load(f)
    with open(RL_V2_PATH, encoding="utf-8") as f:
        rl_v2 = yaml.safe_load(f)
    assert rl_v2["observation_v2"] == rl_v1["observation_v2"]
    assert rl_v2["action_space"] == rl_v1["action_space"]


def test_v2_reward_config_is_deliberately_redesigned_not_identical_to_v1():
    """V2 Phase 2.1 PART 6: unlike V2 Phase 1's reward_v2.yaml (which was
    byte-identical to V1, deliberately inert), reward_v2.yaml now holds a
    genuinely redesigned multi-metric formula (autoscaler.src.rl_env.
    reward_v2) -- this test documents that divergence is INTENTIONAL, not a
    drift regression. See test_reward_v2.py for the formula's own numeric
    proof and tests/unit/test_reward_v1_audit.py for the V1 evidence this
    responds to. reward.py/configs/reward.yaml (frozen V1) are completely
    untouched -- confirmed by test_v1_config_files_unchanged_by_v2_additions
    above."""
    with open(DEFAULT_REWARD_YAML, encoding="utf-8") as f:
        reward_v1 = yaml.safe_load(f)
    with open(REWARD_V2_PATH, encoding="utf-8") as f:
        reward_v2 = yaml.safe_load(f)
    assert reward_v1["weights"]["w_sla"] != reward_v2["weights"]["w_sla"]
    assert "pressure" in reward_v2 and "pressure" not in reward_v1
    assert "waste" in reward_v2 and "waste" not in reward_v1
    assert "resource" not in reward_v2  # V1's unconditional per-replica cost is gone, replaced by waste
    assert "efficiency" not in reward_v2  # V1's positive CPU-band bonus is gone, replaced by pressure+waste
    assert reward_v2["reward_status"] == "v2_designed_pending_training_validation"
