"""scripts/v2_training_instrumentation.py

V2 Phase 4A PART 2: fixes the SB3 training-metrics instrumentation gap
disclosed in the V2 Phase 3D report -- `scripts/train_ppo_live_v2.py`'s
pilot never read or persisted `model.logger`'s recorded `train/*` values
(approx_kl, clip_fraction, entropy_loss, explained_variance, learning_rate,
loss, policy_gradient_loss, value_loss), so they existed only transiently in
the live process's memory and were lost once it exited.

`stable_baselines3.PPO.train()` calls `self.logger.record("train/<name>",
value)` for each of these metrics on every call (confirmed directly against
the installed SB3 2.6.0: `model.logger.name_to_value` is populated
immediately after `model.learn()` returns, with NO dependency on
`verbose>=1` or an explicit `logger.dump()` call -- `record()` always writes
into `name_to_value`, independent of whether anything is ever printed or
flushed to a file). This module therefore only READS that already-existing,
already-populated dict -- it changes no PPO hyperparameter, no training
behavior, no action/reward/observation semantics; it is a pure read-only
instrumentation tap taken AFTER `model.learn()` has already returned for one
completed rollout/update.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# The exact 8 metrics the V2 Phase 3D report disclosed as missing --
# `train/n_updates` and `train/clip_range` also exist in SB3's logger but
# are not part of the required set; they are intentionally not extracted
# here to keep this module's contract exactly matched to what was asked.
REQUIRED_SB3_METRIC_KEYS = (
    "approx_kl",
    "clip_fraction",
    "entropy_loss",
    "explained_variance",
    "learning_rate",
    "loss",
    "policy_gradient_loss",
    "value_loss",
)


def extract_sb3_update_metrics(model: Any) -> Dict[str, Optional[float]]:
    """Reads `model.logger.name_to_value["train/<key>"]` for each of
    `REQUIRED_SB3_METRIC_KEYS`, returning `None` (never a fabricated/
    estimated value) for any key SB3 has not yet recorded (e.g. called
    before any `train()` call has ever run). Never mutates `model` or its
    logger -- a pure read."""
    name_to_value: Dict[str, Any] = getattr(getattr(model, "logger", None), "name_to_value", {}) or {}
    result: Dict[str, Optional[float]] = {}
    for key in REQUIRED_SB3_METRIC_KEYS:
        value = name_to_value.get(f"train/{key}")
        # SB3 records some metrics as plain Python floats and others as
        # numpy scalar types (float32/float64) -- cast to plain `float` so
        # every value is JSON-serializable via a plain `json.dumps()` (e.g.
        # `episode_ledger.append_row()`, which does not pass `default=str`),
        # never leaving a numpy scalar leaking into a persisted ledger row.
        result[key] = None if value is None else float(value)
    return result


def build_update_metrics_row(
    *,
    update_index: int,
    cumulative_timesteps: int,
    session_index: int,
    model: Any,
) -> Dict[str, Any]:
    """Builds one `ppo_update_metrics.jsonl` row for a single completed PPO
    update -- `update_index`/`cumulative_timesteps` are supplied by the
    caller (the campaign script), which is the sole authority on cumulative
    progress; this function never recomputes or guesses them."""
    row: Dict[str, Any] = {
        "update_index": update_index,
        "cumulative_timesteps": cumulative_timesteps,
        "session_index": session_index,
    }
    row.update(extract_sb3_update_metrics(model))
    return row
