"""autoscaler/src/rl_env/config.py

Loads the environment's frozen numeric contract from the project's
authoritative YAML files (configs/rl.yaml, configs/reward.yaml,
configs/experiment.yaml) rather than duplicating magic numbers. The
dataclass defaults in observation.py/gating.py/reward.py already match
these values exactly (so pure-function unit tests can construct them
standalone without touching the filesystem) -- `load_env_config_from_yaml`
is the single place that reads the YAML and is used both by the
environment's default construction path and by a test that cross-checks
the two never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .gating import GatingConfig
from .observation import ObservationConfig
from .reward import RewardConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_RL_YAML = PROJECT_ROOT / "configs" / "rl.yaml"
DEFAULT_REWARD_YAML = PROJECT_ROOT / "configs" / "reward.yaml"
DEFAULT_EXPERIMENT_YAML = PROJECT_ROOT / "configs" / "experiment.yaml"


@dataclass(frozen=True)
class EnvConfig:
    decision_interval_s: float
    pre_episode_stabilization_s: float
    initial_replicas: int
    observation: ObservationConfig
    gating: GatingConfig
    reward: RewardConfig
    max_consecutive_invalid_steps: int = 4
    zero_ready_termination_seconds: float = 120.0


def _obs_entry(entries: list, name: str) -> dict:
    for e in entries:
        if e["name"] == name:
            return e
    raise KeyError(f"observation_v2 entry {name!r} not found")


def load_env_config_from_yaml(
    rl_yaml_path: Path = DEFAULT_RL_YAML,
    reward_yaml_path: Path = DEFAULT_REWARD_YAML,
    experiment_yaml_path: Path = DEFAULT_EXPERIMENT_YAML,
) -> EnvConfig:
    with open(rl_yaml_path, "r", encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)
    with open(reward_yaml_path, "r", encoding="utf-8") as f:
        reward_cfg = yaml.safe_load(f)
    with open(experiment_yaml_path, "r", encoding="utf-8") as f:
        experiment_cfg = yaml.safe_load(f)

    if rl_cfg.get("rl_contract_version") != 2:
        raise ValueError(
            f"{rl_yaml_path} rl_contract_version is not 2 -- "
            "autoscaler/src/rl_env implements contract v2 only."
        )
    if reward_cfg.get("reward_contract_version") != 2:
        raise ValueError(
            f"{reward_yaml_path} reward_contract_version is not 2 -- "
            "autoscaler/src/rl_env implements contract v2 only."
        )

    entries = rl_cfg["observation_v2"]
    cpu = _obs_entry(entries, "cpu_per_replica")
    cpu_delta = _obs_entry(entries, "cpu_per_replica_delta")
    memory = _obs_entry(entries, "memory_per_replica")
    in_progress = _obs_entry(entries, "in_progress_per_replica")
    request_rate = _obs_entry(entries, "request_rate_norm")
    request_rate_delta = _obs_entry(entries, "request_rate_delta")
    latency = _obs_entry(entries, "p95_latency_norm")
    latency_delta = _obs_entry(entries, "p95_latency_delta")
    ready_replica = _obs_entry(entries, "ready_replica_norm")
    seconds_since_scale = _obs_entry(entries, "seconds_since_last_scale_norm")

    scaling = experiment_cfg["scaling"]
    min_replicas = int(scaling["min_replicas"])
    max_replicas = int(scaling["max_replicas"])

    observation = ObservationConfig(
        cpu_raw_min=float(cpu["raw_min"]),
        cpu_raw_max=float(cpu["raw_max"]),
        cpu_delta_raw_min=float(cpu_delta["raw_min"]),
        cpu_delta_raw_max=float(cpu_delta["raw_max"]),
        memory_raw_min=float(memory["raw_min"]),
        memory_raw_max=float(memory["raw_max"]),
        in_progress_raw_min=float(in_progress["raw_min"]),
        in_progress_raw_max=float(in_progress["raw_max"]),
        request_rate_raw_min=float(request_rate["raw_min"]),
        request_rate_raw_max=float(request_rate["raw_max"]),
        request_rate_delta_raw_min=float(request_rate_delta["raw_min"]),
        request_rate_delta_raw_max=float(request_rate_delta["raw_max"]),
        latency_raw_min_ms=float(latency["raw_min"]),
        latency_raw_max_ms=float(latency["raw_max"]),
        latency_delta_raw_min_ms=float(latency_delta["raw_min"]),
        latency_delta_raw_max_ms=float(latency_delta["raw_max"]),
        min_replicas=int(ready_replica["raw_min"]),
        max_replicas=int(ready_replica["raw_max"]),
        seconds_since_scale_cap_s=float(seconds_since_scale["raw_max"]),
        ewma_span_steps=int(rl_cfg["history"]["ewma_span_steps"]),
    )

    gating = GatingConfig(
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        scale_down_cooldown_seconds=float(experiment_cfg["scaling"]["cooldown_seconds"]),
    )

    weights = reward_cfg["weights"]
    reward = RewardConfig(
        w_sla=float(weights["w_sla"]),
        sla_p95_target_ms=float(reward_cfg["sla"]["p95_target_ms"]),
        sla_excess_clip=float(reward_cfg["sla"]["excess_clip"]),
        w_reliability=float(weights["w_reliability"]),
        w_resource=float(weights["w_resource"]),
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        w_action=float(weights["w_action"]),
        w_reversal=float(weights["w_reversal"]),
        reversal_window_seconds=float(reward_cfg["reversal"]["reversal_window_seconds"]),
        w_boundary=float(weights["w_boundary"]),
        w_efficiency=float(weights["w_efficiency"]),
        efficiency_cpu_low=float(reward_cfg["efficiency"]["target_cpu_low"]),
        efficiency_cpu_high=float(reward_cfg["efficiency"]["target_cpu_high"]),
    )

    return EnvConfig(
        decision_interval_s=float(experiment_cfg["rl"]["control_interval_seconds"]),
        pre_episode_stabilization_s=60.0,
        initial_replicas=int(scaling["initial_replicas"]),
        observation=observation,
        gating=gating,
        reward=reward,
    )
