"""tests/unit/test_config_v2.py

V2 Phase 2.2 PART 5/11: proves the explicit V2 config loader exists, is
distinct from the frozen V1 loader, validates the required identity
metadata, and produces observation/gating values numerically identical to
V1 (the contract hasn't changed) while using the genuinely redesigned
RewardConfigV2 for reward.
"""

from __future__ import annotations

import dataclasses

import pytest

from autoscaler.src.rl_env.config import DEFAULT_RL_YAML, load_env_config_from_yaml
from autoscaler.src.rl_env.config_v2 import (
    DEFAULT_RL_V2_YAML,
    EnvConfigV2,
    V2ConfigContractError,
    load_env_config_v2_from_yaml,
)
from autoscaler.src.rl_env.reward_v2 import RewardConfigV2


class TestV2LoaderIsExplicitAndDistinct:
    def test_loader_function_exists_and_is_separate_from_v1(self):
        import autoscaler.src.rl_env.config as v1_config

        assert load_env_config_v2_from_yaml is not v1_config.load_env_config_from_yaml

    def test_default_paths_point_to_v2_files_not_v1(self):
        assert DEFAULT_RL_V2_YAML.name == "rl_v2.yaml"
        assert DEFAULT_RL_V2_YAML != DEFAULT_RL_YAML

    def test_returns_env_config_v2_type(self):
        cfg = load_env_config_v2_from_yaml()
        assert isinstance(cfg, EnvConfigV2)
        assert isinstance(cfg.reward, RewardConfigV2)


class TestV1LoaderUnaffected:
    def test_v1_loader_still_uses_v1_defaults(self):
        """Calling the FROZEN loader with no arguments must still behave
        exactly as before -- this test would fail if V2 Phase 2.2 had
        changed any V1 default."""
        cfg = load_env_config_from_yaml()
        assert cfg.decision_interval_s == 30.0
        from autoscaler.src.rl_env.reward import RewardConfig

        assert isinstance(cfg.reward, RewardConfig)
        assert cfg.reward.w_sla == 1.0  # frozen V1 value, must be untouched


class TestIdentityMetadataValidated:
    def test_controller_generation_is_v2(self):
        cfg = load_env_config_v2_from_yaml()
        assert cfg.controller_generation == "v2"

    def test_workload_contract_is_multi_resource_v2(self):
        cfg = load_env_config_v2_from_yaml()
        assert cfg.workload_contract == "multi_resource_v2"

    def test_reward_status_is_v2_designed_pending_training_validation(self):
        cfg = load_env_config_v2_from_yaml()
        assert cfg.reward_status == "v2_designed_pending_training_validation"

    def test_rejects_an_rl_yaml_without_v2_identity(self, tmp_path):
        """Fail-closed: if someone points this loader at a file lacking the
        V2 identity fields (e.g. accidentally the frozen rl.yaml), it must
        raise, never silently proceed."""
        with pytest.raises(V2ConfigContractError):
            load_env_config_v2_from_yaml(rl_yaml_path=DEFAULT_RL_YAML)


class TestObservationContractUnchanged:
    def test_observation_config_numerically_identical_to_v1(self):
        """The 11-element observation contract has NOT changed -- only the
        reward differs. Direct proof: V2's ObservationConfig/GatingConfig
        are byte-identical to V1's."""
        v1 = load_env_config_from_yaml()
        v2 = load_env_config_v2_from_yaml()
        assert dataclasses.asdict(v1.observation) == dataclasses.asdict(v2.observation)
        assert dataclasses.asdict(v1.gating) == dataclasses.asdict(v2.gating)
        assert v1.decision_interval_s == v2.decision_interval_s
        assert v1.initial_replicas == v2.initial_replicas

    def test_observation_size_still_eleven(self):
        from autoscaler.src.rl_env.observation import OBSERVATION_SIZE

        assert OBSERVATION_SIZE == 11
