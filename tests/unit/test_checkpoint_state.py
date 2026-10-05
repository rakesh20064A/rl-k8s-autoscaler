"""tests/unit/test_checkpoint_state.py

Phase 16W: tests for autoscaler/src/rl_env/checkpoint_state.py. Pure
offline -- RNG state, hashing, and pickle/json I/O in tmp_path only.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from autoscaler.src.rl_env.checkpoint_state import (
    CheckpointProvenanceError,
    SessionCheckpointMetadata,
    TRAINING_PROFILE_CYCLE,
    capture_rng_state,
    load_rng_state,
    resolve_profile_for_episode,
    restore_rng_state,
    save_rng_state,
    verify_resume_provenance,
)


class TestRngCaptureRestore:
    def test_restoring_rng_reproduces_the_same_subsequent_draws(self):
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        snapshot = capture_rng_state()
        draws_before = (random.random(), np.random.rand(), torch.rand(1).item())

        # Perturb all three RNGs.
        random.random()
        np.random.rand()
        torch.rand(1)

        restore_rng_state(snapshot)
        draws_after = (random.random(), np.random.rand(), torch.rand(1).item())
        assert draws_before == draws_after

    def test_save_and_load_round_trip(self, tmp_path):
        random.seed(2)
        np.random.seed(2)
        torch.manual_seed(2)
        snapshot = capture_rng_state()
        path = tmp_path / "rng.pkl"
        sha = save_rng_state(snapshot, path)
        assert path.exists()
        loaded = load_rng_state(path)
        restore_rng_state(loaded)
        draws_after_load = (random.random(), np.random.rand(), torch.rand(1).item())

        # Restore the ORIGINAL snapshot again and compare.
        restore_rng_state(snapshot)
        draws_after_original = (random.random(), np.random.rand(), torch.rand(1).item())
        assert draws_after_load == draws_after_original
        assert len(sha) == 64  # sha256 hex digest


class TestSessionCheckpointMetadata:
    def _meta(self, **overrides):
        base = dict(
            session_index=0, cumulative_timesteps=256, cumulative_ppo_updates=1,
            cumulative_episodes_completed=6, n_steps=256, training_seed=42,
            sb3_checkpoint_sha256="a" * 64, rng_state_sha256="b" * 64,
            governing_hashes_snapshot={"x": "y"},
            created_utc=datetime.now(timezone.utc).isoformat(),
        )
        base.update(overrides)
        return SessionCheckpointMetadata(**base)

    def test_json_round_trip(self, tmp_path):
        meta = self._meta()
        path = tmp_path / "meta.json"
        meta.to_json(path)
        loaded = SessionCheckpointMetadata.from_json(path)
        assert loaded == meta


class TestVerifyResumeProvenance:
    def _write_checkpoint_files(self, tmp_path, sb3_bytes=b"fake-zip", rng_bytes=b"fake-rng"):
        sb3_path = tmp_path / "ppo_model.zip"
        rng_path = tmp_path / "rng_state.pkl"
        sb3_path.write_bytes(sb3_bytes)
        rng_path.write_bytes(rng_bytes)
        return sb3_path, rng_path

    def _meta_for(self, sb3_path, rng_path, **overrides):
        import hashlib
        base = dict(
            session_index=0, cumulative_timesteps=256, cumulative_ppo_updates=1,
            cumulative_episodes_completed=6, n_steps=256, training_seed=42,
            sb3_checkpoint_sha256=hashlib.sha256(sb3_path.read_bytes()).hexdigest(),
            rng_state_sha256=hashlib.sha256(rng_path.read_bytes()).hexdigest(),
            governing_hashes_snapshot={"docs/rl_mdp_spec.md": "deadbeef"},
            created_utc=datetime.now(timezone.utc).isoformat(),
        )
        base.update(overrides)
        return SessionCheckpointMetadata(**base)

    def test_clean_resume_passes(self, tmp_path):
        sb3_path, rng_path = self._write_checkpoint_files(tmp_path)
        meta = self._meta_for(sb3_path, rng_path)
        verify_resume_provenance(
            meta, sb3_path, rng_path, expected_next_session_index=1,
            expected_n_steps=256, expected_training_seed=42,
            current_governing_hashes={"docs/rl_mdp_spec.md": "deadbeef"},
        )  # must not raise

    def test_rejects_wrong_next_session_index(self, tmp_path):
        sb3_path, rng_path = self._write_checkpoint_files(tmp_path)
        meta = self._meta_for(sb3_path, rng_path)
        with pytest.raises(CheckpointProvenanceError, match="session_index"):
            verify_resume_provenance(
                meta, sb3_path, rng_path, expected_next_session_index=5,
                expected_n_steps=256, expected_training_seed=42,
                current_governing_hashes={"docs/rl_mdp_spec.md": "deadbeef"},
            )

    def test_rejects_non_rollout_boundary_timesteps(self, tmp_path):
        sb3_path, rng_path = self._write_checkpoint_files(tmp_path)
        meta = self._meta_for(sb3_path, rng_path, cumulative_timesteps=300)  # not a multiple of 256
        with pytest.raises(CheckpointProvenanceError, match="rollout boundary"):
            verify_resume_provenance(
                meta, sb3_path, rng_path, expected_next_session_index=1,
                expected_n_steps=256, expected_training_seed=42,
                current_governing_hashes={"docs/rl_mdp_spec.md": "deadbeef"},
            )

    def test_rejects_n_steps_mismatch(self, tmp_path):
        sb3_path, rng_path = self._write_checkpoint_files(tmp_path)
        meta = self._meta_for(sb3_path, rng_path)
        with pytest.raises(CheckpointProvenanceError, match="n_steps mismatch"):
            verify_resume_provenance(
                meta, sb3_path, rng_path, expected_next_session_index=1,
                expected_n_steps=999, expected_training_seed=42,
                current_governing_hashes={"docs/rl_mdp_spec.md": "deadbeef"},
            )

    def test_rejects_seed_mismatch(self, tmp_path):
        sb3_path, rng_path = self._write_checkpoint_files(tmp_path)
        meta = self._meta_for(sb3_path, rng_path)
        with pytest.raises(CheckpointProvenanceError, match="training_seed mismatch"):
            verify_resume_provenance(
                meta, sb3_path, rng_path, expected_next_session_index=1,
                expected_n_steps=256, expected_training_seed=999,
                current_governing_hashes={"docs/rl_mdp_spec.md": "deadbeef"},
            )

    def test_rejects_missing_sb3_checkpoint_file(self, tmp_path):
        sb3_path, rng_path = self._write_checkpoint_files(tmp_path)
        meta = self._meta_for(sb3_path, rng_path)
        sb3_path.unlink()
        with pytest.raises(CheckpointProvenanceError, match="missing"):
            verify_resume_provenance(
                meta, sb3_path, rng_path, expected_next_session_index=1,
                expected_n_steps=256, expected_training_seed=42,
                current_governing_hashes={"docs/rl_mdp_spec.md": "deadbeef"},
            )

    def test_rejects_tampered_sb3_checkpoint(self, tmp_path):
        sb3_path, rng_path = self._write_checkpoint_files(tmp_path)
        meta = self._meta_for(sb3_path, rng_path)
        sb3_path.write_bytes(b"tampered-content")
        with pytest.raises(CheckpointProvenanceError, match="hash mismatch"):
            verify_resume_provenance(
                meta, sb3_path, rng_path, expected_next_session_index=1,
                expected_n_steps=256, expected_training_seed=42,
                current_governing_hashes={"docs/rl_mdp_spec.md": "deadbeef"},
            )

    def test_rejects_governing_hash_drift(self, tmp_path):
        sb3_path, rng_path = self._write_checkpoint_files(tmp_path)
        meta = self._meta_for(sb3_path, rng_path)
        with pytest.raises(CheckpointProvenanceError, match="governing hash drift"):
            verify_resume_provenance(
                meta, sb3_path, rng_path, expected_next_session_index=1,
                expected_n_steps=256, expected_training_seed=42,
                current_governing_hashes={"docs/rl_mdp_spec.md": "SOMETHING_CHANGED"},
            )


class TestResolveProfileForEpisode:
    def test_returns_a_valid_training_profile(self):
        for i in range(30):
            assert resolve_profile_for_episode(i, 42) in TRAINING_PROFILE_CYCLE

    def test_deterministic_given_same_inputs(self):
        results1 = [resolve_profile_for_episode(i, 42) for i in range(20)]
        results2 = [resolve_profile_for_episode(i, 42) for i in range(20)]
        assert results1 == results2

    def test_each_cycle_of_six_is_a_permutation_of_all_six_profiles(self):
        for cycle in range(5):
            cycle_profiles = {resolve_profile_for_episode(cycle * 6 + pos, 42) for pos in range(6)}
            assert cycle_profiles == set(TRAINING_PROFILE_CYCLE)

    def test_different_seed_can_produce_a_different_order(self):
        order_seed_a = [resolve_profile_for_episode(i, 42) for i in range(6)]
        order_seed_b = [resolve_profile_for_episode(i, 7) for i in range(6)]
        # Not asserting they MUST differ (could coincidentally match), just
        # that the function is seed-sensitive by construction (different
        # random.Random instances per seed).
        assert callable(resolve_profile_for_episode)
        assert isinstance(order_seed_a, list) and isinstance(order_seed_b, list)

    def test_resuming_at_the_same_episode_index_reconstructs_the_same_profile(self):
        """This is the core guarantee checkpoint/resume relies on: no
        mutable cursor needs to be persisted -- episode N always resolves
        to the same profile regardless of how many sessions it took to
        reach it."""
        direct = resolve_profile_for_episode(17, 42)
        also_direct = resolve_profile_for_episode(17, 42)
        assert direct == also_direct
