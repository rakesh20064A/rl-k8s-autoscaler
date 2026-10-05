"""autoscaler/src/rl_env/ppo_config.py

Phase 15B.1 correction (CORR-29): Phase 15A.2's SB3 compatibility smoke test
verified that `PPO` accepts every field name configured in
`configs/rl.yaml: ppo`, but constructed the model WITHOUT actually passing
`net_arch` through `policy_kwargs` -- so it never proved the config-to-
`policy_kwargs` wiring produces the configured `[128, 128]` hidden layers
(the network it inspected used SB3's own default 64-unit layers instead).
This module is the small, tested adapter that closes that gap: it maps
`configs/rl.yaml: ppo` to the exact kwargs `stable_baselines3.PPO.__init__`
expects, including `net_arch` via `policy_kwargs`. It does not train
anything and is not itself a training script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_RL_YAML = PROJECT_ROOT / "configs" / "rl.yaml"

_DIRECT_FIELDS = (
    "policy", "learning_rate", "n_steps", "batch_size", "n_epochs",
    "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
)


def build_ppo_kwargs(rl_yaml_path: Path = DEFAULT_RL_YAML, seed: int = None) -> Dict[str, Any]:
    """Return kwargs suitable for `stable_baselines3.PPO(env=..., **kwargs)`,
    derived entirely from `configs/rl.yaml: ppo` -- no hyperparameter is
    invented or tuned here. `net_arch` is placed inside `policy_kwargs`
    (never a top-level PPO kwarg in any SB3 version)."""
    with open(rl_yaml_path, "r", encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)
    ppo_cfg = rl_cfg["ppo"]

    kwargs: Dict[str, Any] = {field: ppo_cfg[field] for field in _DIRECT_FIELDS}
    kwargs["policy_kwargs"] = {"net_arch": list(ppo_cfg["net_arch"])}
    if seed is not None:
        kwargs["seed"] = seed
    return kwargs
