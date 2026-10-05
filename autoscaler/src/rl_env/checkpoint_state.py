"""autoscaler/src/rl_env/checkpoint_state.py

Phase 16W: RNG-state and session-metadata capture/restore for resumable
live PPO training, plus the pure, deterministic training-workload-position
function. Exists because `stable_baselines3.PPO.save()`/`.load()`, verified
directly against the installed SB3 2.6.0 source
(`task_outputs/phase_16w_checkpoint_resume_protocol.yaml` documents the
exact inspection), DOES fully restore:

  - policy network parameters (`policy` state dict)
  - optimizer state -- Adam momentum/variance (`policy.optimizer` state
    dict; PPO's `_get_torch_save_params()` includes both, confirmed by
    direct instantiation: `model.get_parameters().keys() ==
    {'policy', 'policy.optimizer'}`)
  - `num_timesteps` / `_total_timesteps` / `_num_timesteps_at_start` /
    `seed` / `lr_schedule` (plain pickled attributes)

but does NOT restore:

  - the rollout buffer (explicitly excluded -- `_excluded_save_params()`
    includes `"rollout_buffer"`) -- checkpoints MUST therefore only be
    taken at a COMPLETED rollout boundary, never mid-rollout.
  - the environment object itself (excluded; `load()`'s `force_reset=True`
    default sets `_last_obs=None`, which forces `_setup_learn()` to call
    `env.reset()` at the start of every resumed session regardless of
    `reset_num_timesteps` -- an in-progress episode at session-end is
    therefore unavoidably, and correctly, ended by the mere act of
    resuming, exactly as it would be by an unplanned crash).
  - RNG state: `_setup_model()` (invoked during `PPO.load()`'s
    reconstruction) unconditionally calls `self.set_random_seed(self.seed)`,
    which RESETS Python/NumPy/PyTorch RNGs to seed 42's initial state on
    EVERY load -- confirmed by direct source inspection
    (`OnPolicyAlgorithm._setup_model` -> `BaseAlgorithm.set_random_seed`).
    Without the explicit capture/restore this module provides, every
    resumed session's stochastic action sampling would restart from the
    same point rather than continuing the sequence a genuinely
    uninterrupted run would have produced.

`learning_rate` and `clip_range` are frozen as plain constant floats in
`phase_16u_ppo_hyperparameters.json` (not progress-dependent schedules) --
verified via `stable_baselines3.common.utils.get_schedule_fn`, a float
input becomes `constant_fn`, which ignores `progress_remaining` entirely.
The per-session recomputation of `_total_timesteps` therefore has ZERO
effect on either hyperparameter's value across sessions -- no
schedule-continuity concern exists for this specific frozen configuration.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class RngSnapshot:
    python_state: Any
    numpy_state: Any
    torch_cpu_state: Any


def capture_rng_state() -> RngSnapshot:
    """Captures Python's, NumPy's, and PyTorch's (CPU) global RNG state.
    Must be called AFTER PPO.save() (or at the equivalent point in the
    training loop) and its output restored via `restore_rng_state()`
    immediately AFTER the NEXT session's `PPO.load()` call (which will have
    just reset all three), BEFORE that session's `model.learn()` call."""
    import numpy as np
    import torch

    return RngSnapshot(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu_state=torch.get_rng_state(),
    )


def restore_rng_state(snapshot: RngSnapshot) -> None:
    import numpy as np
    import torch

    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_state)
    torch.set_rng_state(snapshot.torch_cpu_state)


def save_rng_state(snapshot: RngSnapshot, path: Path) -> str:
    """Saves the snapshot via pickle (RNG state objects are not JSON-safe)
    and returns its SHA-256 for provenance/hash-pinning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(snapshot, f)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rng_state(path: Path) -> RngSnapshot:
    with open(path, "rb") as f:
        return pickle.load(f)


@dataclass(frozen=True)
class SessionCheckpointMetadata:
    """Everything needed to fail-closed-verify a resume, beyond what the
    SB3 .zip and the RNG pickle already contain natively."""

    session_index: int
    cumulative_timesteps: int
    cumulative_ppo_updates: int
    cumulative_episodes_completed: int
    n_steps: int
    training_seed: int
    sb3_checkpoint_sha256: str
    rng_state_sha256: str
    governing_hashes_snapshot: Dict[str, str]
    created_utc: str

    def to_json(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def from_json(path: Path) -> "SessionCheckpointMetadata":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SessionCheckpointMetadata(**data)


class CheckpointProvenanceError(RuntimeError):
    """Raised when a resume's expected state does not match the checkpoint
    on disk. Callers must fail closed -- never guess or proceed anyway."""


def verify_resume_provenance(
    metadata: SessionCheckpointMetadata,
    sb3_checkpoint_path: Path,
    rng_state_path: Path,
    expected_next_session_index: int,
    expected_n_steps: int,
    expected_training_seed: int,
    current_governing_hashes: Dict[str, str],
) -> None:
    """Raises CheckpointProvenanceError on ANY mismatch. A clean pass is
    required before a resumed session may call PPO.load()."""
    if metadata.session_index + 1 != expected_next_session_index:
        raise CheckpointProvenanceError(
            f"session_index continuity broken: checkpoint recorded session "
            f"{metadata.session_index}, expected to resume into session "
            f"{expected_next_session_index} (i.e. checkpoint session+1)."
        )
    if metadata.cumulative_timesteps % metadata.n_steps != 0:
        raise CheckpointProvenanceError(
            f"cumulative_timesteps ({metadata.cumulative_timesteps}) is not an exact "
            f"multiple of n_steps ({metadata.n_steps}) -- checkpoint was not taken at "
            "a completed rollout boundary."
        )
    if metadata.n_steps != expected_n_steps:
        raise CheckpointProvenanceError(
            f"n_steps mismatch: checkpoint={metadata.n_steps}, expected={expected_n_steps}"
        )
    if metadata.training_seed != expected_training_seed:
        raise CheckpointProvenanceError(
            f"training_seed mismatch: checkpoint={metadata.training_seed}, expected={expected_training_seed}"
        )
    if not sb3_checkpoint_path.exists():
        raise CheckpointProvenanceError(f"SB3 checkpoint file missing: {sb3_checkpoint_path}")
    actual_sb3_hash = hashlib.sha256(sb3_checkpoint_path.read_bytes()).hexdigest()
    if actual_sb3_hash != metadata.sb3_checkpoint_sha256:
        raise CheckpointProvenanceError(
            f"SB3 checkpoint hash mismatch: file={actual_sb3_hash}, metadata={metadata.sb3_checkpoint_sha256}"
        )
    if not rng_state_path.exists():
        raise CheckpointProvenanceError(f"RNG state file missing: {rng_state_path}")
    actual_rng_hash = hashlib.sha256(rng_state_path.read_bytes()).hexdigest()
    if actual_rng_hash != metadata.rng_state_sha256:
        raise CheckpointProvenanceError(
            f"RNG state hash mismatch: file={actual_rng_hash}, metadata={metadata.rng_state_sha256}"
        )
    for path, expected_hash in metadata.governing_hashes_snapshot.items():
        actual_hash = current_governing_hashes.get(path)
        if actual_hash != expected_hash:
            raise CheckpointProvenanceError(
                f"governing hash drift detected for {path!r}: checkpoint recorded "
                f"{expected_hash}, current is {actual_hash} -- a frozen protocol/config "
                "file changed between sessions. Refusing to resume."
            )


TRAINING_PROFILE_CYCLE = ("constant", "step", "burst", "ramp", "periodic", "mixed")


def resolve_profile_for_episode(episode_index: int, training_seed: int) -> str:
    """Pure, deterministic function of (episode_index, training_seed) only
    -- requires NO mutable cursor to be persisted across sessions. Matches
    phase_16u_training_and_evaluation_workload_split.json's
    `episode_scheduling_rule` ("round-robin cycle through the 6 profiles in
    a fixed order, reshuffled once per full cycle using the frozen training
    seed"): the order within cycle N is a deterministic permutation of the
    6 profiles seeded by `(training_seed, N)`, so resuming at episode index
    K after any number of interrupted/resumed sessions reconstructs the
    exact same profile that episode K would have used in one uninterrupted
    run -- no drift, no need to persist "which profile comes next."
    """
    cycle_length = len(TRAINING_PROFILE_CYCLE)
    cycle_number = episode_index // cycle_length
    position_in_cycle = episode_index % cycle_length

    # random.Random() only supports int/float/str/bytes/bytearray seeds
    # without triggering Python's deprecated hash-based seeding path (which
    # is scheduled for removal) -- derive a single deterministic int seed
    # from (training_seed, cycle_number) via SHA-256 rather than passing a
    # tuple directly.
    digest = hashlib.sha256(f"{training_seed}:{cycle_number}".encode("utf-8")).hexdigest()
    combined_seed = int(digest[:16], 16)

    order = list(TRAINING_PROFILE_CYCLE)
    rng = random.Random(combined_seed)
    rng.shuffle(order)
    return order[position_in_cycle]
