"""autoscaler/src/rl_env/config_v2.py

V2 Phase 2.2 PART 5: an EXPLICIT, separate V2 config loader --
`load_env_config_v2_from_yaml()` -- distinct from the frozen
`config.load_env_config_from_yaml()`. No filename-default ambiguity: V1
code must keep calling `load_env_config_from_yaml()` (defaults to
configs/rl.yaml + configs/reward.yaml), V2 code must call
`load_env_config_v2_from_yaml()` (defaults to configs/rl_v2.yaml +
configs/reward_v2.yaml) -- there is no shared default path either could
silently fall back to.

Reuses the FROZEN loader for observation/gating/decision-interval parsing
(proven in V2 Phase 2's own tests to produce numerically IDENTICAL values
from rl_v2.yaml as from the frozen rl.yaml -- the observation/action
contract has not changed) rather than re-deriving that parsing logic here,
which would risk silent drift between the two. The reward side is NOT
reused from that loader -- `reward_v2.yaml`'s schema is a genuinely
different, redesigned formula (V2 Phase 2.1 PART 6), loaded via
`reward_v2.load_reward_config_v2_from_yaml()` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from .action_selection import DEFAULT_NEAR_TIE_EPSILON
from .config import DEFAULT_REWARD_YAML, PROJECT_ROOT, load_env_config_from_yaml
from .gating import GatingConfig
from .observation import ObservationConfig
from .reward_v2 import DEFAULT_REWARD_V2_YAML, RewardConfigV2, load_reward_config_v2_from_yaml

DEFAULT_RL_V2_YAML = PROJECT_ROOT / "configs" / "rl_v2.yaml"
DEFAULT_EXPERIMENT_YAML = PROJECT_ROOT / "configs" / "experiment.yaml"

DEFAULT_EPSILON_STATUS = "provisional_unvalidated"

EXPECTED_CONTROLLER_GENERATION = "v2"
EXPECTED_WORKLOAD_CONTRACT = "multi_resource_v2"


@dataclass(frozen=True)
class EnvConfigV2:
    decision_interval_s: float
    pre_episode_stabilization_s: float
    initial_replicas: int
    observation: ObservationConfig
    gating: GatingConfig
    reward: RewardConfigV2
    max_consecutive_invalid_steps: int
    zero_ready_termination_seconds: float
    controller_generation: str
    workload_contract: str
    reward_status: str


class V2ConfigContractError(RuntimeError):
    """Raised when rl_v2.yaml/reward_v2.yaml's identity metadata does not
    match what this loader requires -- fail-closed, never silently
    proceeds with an unexpected/stale contract."""


def load_env_config_v2_from_yaml(
    rl_yaml_path: Path = DEFAULT_RL_V2_YAML,
    reward_yaml_path: Path = DEFAULT_REWARD_V2_YAML,
    experiment_yaml_path: Path = DEFAULT_EXPERIMENT_YAML,
) -> EnvConfigV2:
    """Loads the V2 observation/gating/decision-interval contract from
    rl_yaml_path (via the frozen loader, paired with the frozen
    DEFAULT_REWARD_YAML purely as a structurally-required but UNUSED
    placeholder -- see module docstring) and the V2 reward formula from
    reward_yaml_path (via reward_v2's own loader). Validates the identity
    metadata fields explicitly; raises V2ConfigContractError if
    rl_yaml_path is not actually a V2-identified file."""
    base = load_env_config_from_yaml(
        rl_yaml_path=rl_yaml_path, reward_yaml_path=DEFAULT_REWARD_YAML, experiment_yaml_path=experiment_yaml_path,
    )
    reward_v2 = load_reward_config_v2_from_yaml(reward_yaml_path)

    with open(rl_yaml_path, "r", encoding="utf-8") as f:
        rl_v2_raw = yaml.safe_load(f)
    with open(reward_yaml_path, "r", encoding="utf-8") as f:
        reward_v2_raw = yaml.safe_load(f)

    controller_generation = rl_v2_raw.get("controller_generation")
    workload_contract = rl_v2_raw.get("workload_contract")
    reward_status = reward_v2_raw.get("reward_status")

    if controller_generation != EXPECTED_CONTROLLER_GENERATION:
        raise V2ConfigContractError(
            f"{rl_yaml_path} controller_generation={controller_generation!r}, "
            f"expected {EXPECTED_CONTROLLER_GENERATION!r}"
        )
    if workload_contract != EXPECTED_WORKLOAD_CONTRACT:
        raise V2ConfigContractError(
            f"{rl_yaml_path} workload_contract={workload_contract!r}, expected {EXPECTED_WORKLOAD_CONTRACT!r}"
        )

    # V2 Phase 4D-B / CORR-55: `base` above was built from `experiment_cfg
    # ["scaling"]` (configs/experiment.yaml, SHARED with V1 -- min_replicas=1,
    # max_replicas=8, initial_replicas=1, and NEVER edited for this V2-only
    # contract change). `rl_yaml_path`'s own `scaling_v2` block is the ONE
    # V2-only override: required (fail-closed, never silently falls back to
    # the V1-shared values) so a V2 config that forgot to carry this block
    # is caught immediately rather than silently training against the old
    # floor=1 contract. `base.observation`'s min/max replicas already come
    # from THIS SAME file's `ready_replica_norm.raw_min/raw_max` (via the
    # shared loader, config.py), so no observation override is needed here
    # -- only gating/initial_replicas/reward's min_replicas/max_replicas,
    # none of which the shared loader sources from rl_yaml_path.
    scaling_v2 = rl_v2_raw.get("scaling_v2")
    if scaling_v2 is None:
        raise V2ConfigContractError(f"{rl_yaml_path} is missing the required scaling_v2 block")
    v2_min_replicas = int(scaling_v2["min_replicas"])
    v2_max_replicas = int(scaling_v2["max_replicas"])
    v2_initial_replicas = int(scaling_v2["initial_replicas"])

    gating_v2 = replace(base.gating, min_replicas=v2_min_replicas, max_replicas=v2_max_replicas)
    reward_v2 = replace(reward_v2, min_replicas=v2_min_replicas, max_replicas=v2_max_replicas)

    return EnvConfigV2(
        decision_interval_s=base.decision_interval_s,
        pre_episode_stabilization_s=base.pre_episode_stabilization_s,
        initial_replicas=v2_initial_replicas,
        observation=base.observation,
        gating=gating_v2,
        reward=reward_v2,
        max_consecutive_invalid_steps=base.max_consecutive_invalid_steps,
        zero_ready_termination_seconds=base.zero_ready_termination_seconds,
        controller_generation=controller_generation,
        workload_contract=workload_contract,
        reward_status=reward_status,
    )


def load_safe_action_epsilon_from_v2_config(
    rl_yaml_path: Path = DEFAULT_RL_V2_YAML,
) -> "tuple[float, str]":
    """Reads the `safe_action.near_tie_epsilon` / `safe_action.epsilon_status`
    block directly out of rl_yaml_path -- additive, does not touch
    `EnvConfigV2` or `load_env_config_v2_from_yaml()` (both frozen for
    session_002 resumability). Lets V2 callers (e.g.
    scripts/evaluate_ppo_vs_hpa_v2.py) source epsilon from rl_v2.yaml's own
    contract instead of only action_selection.py's Python-level default.
    Falls back to that same default/status if the block or keys are absent,
    so this never raises on a well-formed but older rl_v2.yaml."""
    with open(rl_yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    safe_action = raw.get("safe_action") or {}
    epsilon = float(safe_action.get("near_tie_epsilon", DEFAULT_NEAR_TIE_EPSILON))
    epsilon_status = safe_action.get("epsilon_status", DEFAULT_EPSILON_STATUS)
    return epsilon, epsilon_status
