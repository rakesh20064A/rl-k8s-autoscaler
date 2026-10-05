"""tests/unit/test_rl_env_ppo_config.py

Phase 15B.1 (CORR-29): verifies the config-to-`policy_kwargs` adapter
actually produces the configured `[128, 128]` hidden-layer PPO policy --
the exact gap Phase 15A.2's smoke test left unproven (it inspected a PPO
model constructed WITHOUT passing net_arch, so it exercised SB3's default
64-unit layers instead). No `.learn()` is called; no model is saved.
"""

from stable_baselines3 import PPO

from autoscaler.src.rl_env.config import load_env_config_from_yaml
from autoscaler.src.rl_env.environment import KubernetesAutoscalerEnv
from autoscaler.src.rl_env.ppo_config import build_ppo_kwargs
from autoscaler.src.rl_env.scripted_backend import ScriptedBackend
from autoscaler.src.rl_env.types import RawMetricSnapshot

CFG = load_env_config_from_yaml()


def _dense_timeline(times):
    snap = RawMetricSnapshot(
        request_rate_rps=5.0, error_rate_rps=0.0, latency_p95_ms=100.0,
        requests_in_progress=1.0, cpu_cores=0.2, memory_bytes=40_000_000.0,
    )
    return {t: snap for t in times}


def test_build_ppo_kwargs_maps_net_arch_into_policy_kwargs():
    kwargs = build_ppo_kwargs()
    assert kwargs["policy_kwargs"] == {"net_arch": [128, 128]}
    assert kwargs["policy"] == "MlpPolicy"
    assert kwargs["learning_rate"] == 3.0e-4
    assert kwargs["n_steps"] == 2048
    assert kwargs["batch_size"] == 64
    assert "net_arch" not in kwargs  # must not ALSO appear as a top-level kwarg


def test_constructed_ppo_policy_actually_has_128_128_hidden_layers():
    """The exact defect this module corrects relative to Phase 15A.2."""
    backend = ScriptedBackend(timeline=_dense_timeline([0.0, 30.0]), initial_replicas=1)
    env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)

    kwargs = build_ppo_kwargs(seed=42)
    kwargs["n_steps"] = 8  # smoke-test-speed override only, not a config/compat change
    kwargs["batch_size"] = 8
    kwargs["verbose"] = 0

    model = PPO(env=env, **kwargs)

    policy_layers = model.policy.mlp_extractor.policy_net
    linear_layers = [layer for layer in policy_layers if hasattr(layer, "out_features")]
    hidden_sizes = [layer.out_features for layer in linear_layers]

    assert hidden_sizes == [128, 128], (
        f"expected [128, 128] hidden layers from configs/rl.yaml net_arch, got {hidden_sizes} "
        "(SB3 defaults to [64, 64] when policy_kwargs is not passed -- this is exactly the gap "
        "Phase 15A.2 left unproven, corrected here per CORR-29)."
    )

    # No training, no model artifact.
    action, _ = model.predict(env.reset(seed=1)[0], deterministic=True)
    assert env.action_space.contains(int(action))
