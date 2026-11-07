"""tests/unit/test_v2_training_instrumentation.py

V2 Phase 4A PART 2/11: proves the new SB3 training-metrics instrumentation
(`scripts.v2_training_instrumentation`) reads SB3's own already-populated
`model.logger.name_to_value` dict -- never fabricates/estimates a value, and
never mutates the model, its hyperparameters, or its training behavior.
"""

from __future__ import annotations

import inspect

import gymnasium as gym
from stable_baselines3 import PPO

from scripts.v2_training_instrumentation import (
    REQUIRED_SB3_METRIC_KEYS,
    build_update_metrics_row,
    extract_sb3_update_metrics,
)


def _tiny_model(seed=42):
    env = gym.make("CartPole-v1")
    return PPO("MlpPolicy", env, n_steps=32, batch_size=16, n_epochs=2, verbose=0, seed=seed)


class TestExtractSb3UpdateMetrics:
    def test_all_required_keys_present_after_one_update(self):
        model = _tiny_model()
        model.learn(total_timesteps=32)
        metrics = extract_sb3_update_metrics(model)
        assert set(metrics.keys()) == set(REQUIRED_SB3_METRIC_KEYS)
        assert all(v is not None for v in metrics.values())

    def test_values_are_numeric(self):
        """SB3 records some metrics as plain Python floats and others as
        numpy scalar types -- both are numeric and JSON-serializable via
        `json.dumps(..., default=str)`; neither is a string/None/object."""
        model = _tiny_model()
        model.learn(total_timesteps=32)
        metrics = extract_sb3_update_metrics(model)
        for k, v in metrics.items():
            assert isinstance(v, (int, float)) or hasattr(v, "item"), f"{k} was {type(v)}"

    def test_never_fabricates_before_any_train_call(self):
        """A model that has never called train() (no learn() yet) must
        report None for every key -- never a guessed/default 0.0."""
        model = _tiny_model()
        metrics = extract_sb3_update_metrics(model)
        assert all(v is None for v in metrics.values())

    def test_does_not_mutate_model(self):
        model = _tiny_model()
        model.learn(total_timesteps=32)
        before_num_timesteps = model.num_timesteps
        before_params = {k: v.clone() for k, v in model.policy.state_dict().items()}
        extract_sb3_update_metrics(model)
        extract_sb3_update_metrics(model)
        assert model.num_timesteps == before_num_timesteps
        after_params = model.policy.state_dict()
        for k, v in before_params.items():
            assert (v == after_params[k]).all(), f"parameter {k} changed from a read-only metrics extraction"

    def test_second_update_metrics_differ_from_first(self):
        """Not a strict scientific requirement, but confirms this is really
        reading per-update values, not a frozen snapshot from construction
        time."""
        model = _tiny_model()
        model.learn(total_timesteps=32)
        first = extract_sb3_update_metrics(model)
        model.learn(total_timesteps=32, reset_num_timesteps=False)
        second = extract_sb3_update_metrics(model)
        assert first != second or first["loss"] != second["loss"]


class TestBuildUpdateMetricsRow:
    def test_row_contains_caller_supplied_identity_fields(self):
        model = _tiny_model()
        model.learn(total_timesteps=32)
        row = build_update_metrics_row(update_index=3, cumulative_timesteps=1024, session_index=3, model=model)
        assert row["update_index"] == 3
        assert row["cumulative_timesteps"] == 1024
        assert row["session_index"] == 3
        for key in REQUIRED_SB3_METRIC_KEYS:
            assert key in row


class TestInstrumentationIsReadOnly:
    def test_functions_never_call_learn_predict_or_set_parameters(self):
        """Structural proof: the two callables this module exposes are a
        pure read tap -- their OWN bodies (not the module's prose docstring,
        which legitimately mentions `model.learn()` in explaining when to
        call these functions) must never themselves drive training."""
        for fn in (extract_sb3_update_metrics, build_update_metrics_row):
            source = inspect.getsource(fn)
            for forbidden in (".learn(", ".predict(", ".set_parameters(", ".train("):
                assert forbidden not in source, f"found {forbidden!r} in {fn.__name__} -- must be read-only"

    def test_functions_have_no_reward_coupling(self):
        for fn in (extract_sb3_update_metrics, build_update_metrics_row):
            source = inspect.getsource(fn)
            assert "reward_v2" not in source
            assert "compute_reward" not in source
