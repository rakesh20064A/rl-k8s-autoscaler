"""tests/unit/test_train_ppo_live_session.py

Phase 16W: tests for scripts/train_ppo_live_session.py. Only the
--dry-run path is exercised; run_live_session() is proven gated/unreachable
by default, never invoked for real. Includes the required synthetic
equivalence test (task sec.11): a continuous mock PPO run vs. an otherwise
identical checkpoint/resume mock run, compared for continuity/provenance.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

import scripts.train_ppo_live_session as session_mod
from autoscaler.src.rl_env.checkpoint_state import (
    CheckpointProvenanceError,
    SessionCheckpointMetadata,
    capture_rng_state,
    load_rng_state,
    restore_rng_state,
    save_rng_state,
    verify_resume_provenance,
)
from autoscaler.src.rl_env.config import load_env_config_from_yaml
from autoscaler.src.rl_env.environment import KubernetesAutoscalerEnv
from autoscaler.src.rl_env.scripted_backend import ScriptedBackend
from autoscaler.src.rl_env.training_wrapper import LedgeredEpisodeWrapper
from autoscaler.src.rl_env.types import RawMetricSnapshot

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _real_checkpoint_dir_snapshot():
    """Phase16AD note: a REAL checkpoint
    (persistent-data/rl/checkpoints/phase16w_sessions/session_000/) now
    legitimately exists as preserved Phase16AC scientific evidence -- this
    module's invariant is therefore no longer "CHECKPOINT_DIR never
    exists" (only ever true before any real live execution had occurred),
    but "the dry-run/offline test path never MODIFIES it". Tests that
    deliberately monkeypatch `session_mod.CHECKPOINT_DIR` to a tmp_path
    (see TestRealResumeProvenanceFlowFromASession000LikeCheckpoint) revert
    that patch via pytest's own monkeypatch teardown before this fixture's
    post-yield check runs, so they never trip this guard."""
    if not session_mod.CHECKPOINT_DIR.exists():
        return None
    return sorted(str(p.relative_to(session_mod.CHECKPOINT_DIR)) for p in session_mod.CHECKPOINT_DIR.rglob("*"))


@pytest.fixture(autouse=True)
def _assert_dry_run_never_modifies_the_real_checkpoint_dir():
    """Guards against ever regressing back to writing dry-run artifacts
    under the real persistent-data/rl/checkpoints path (see
    run_dry_run_validation()'s docstring for why that must never happen)."""
    before = _real_checkpoint_dir_snapshot()
    yield
    after = _real_checkpoint_dir_snapshot()
    assert before == after, (
        f"a test modified the real {session_mod.CHECKPOINT_DIR} -- dry-run/offline "
        "tests must never write to or alter the real checkpoint path"
    )


class TestDryRunValidation:
    def test_dry_run_report_is_ok(self):
        report = session_mod.run_dry_run_validation()
        assert report["ok"] is True
        assert report["traffic_generated"] is False
        assert report["kubernetes_mutation"] is False

    def test_dry_run_appends_timesteps_without_duplication(self):
        report = session_mod.run_dry_run_validation()
        assert report["timesteps_correctly_appended_not_duplicated"] is True
        assert report["num_timesteps_after_resumed_session"] == report["num_timesteps_before_resume"] + 32

    def test_dry_run_fails_closed_on_bad_resume(self):
        report = session_mod.run_dry_run_validation()
        assert report["fail_closed_on_session_index_mismatch"] is True

    def test_dry_run_never_references_live_backend(self):
        source = Path(session_mod.__file__).read_text()
        dry_run_src = source.split("def run_dry_run_validation")[1].split("\ndef ")[0]
        assert "LiveKubernetesBackend" not in dry_run_src
        assert "ScriptedBackend" in dry_run_src


class TestCliGating:
    def test_default_invocation_is_dry_run_and_exits_zero(self):
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "train_ppo_live_session.py")],
            capture_output=True, text=True, timeout=60, cwd=PROJECT_ROOT,
        )
        assert proc.returncode == 0
        assert '"traffic_generated": false' in proc.stdout

    def test_execute_session_without_confirmation_flag_refuses(self):
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "train_ppo_live_session.py"), "--execute-session"],
            capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT,
        )
        assert proc.returncode == 2
        assert "Refusing" in proc.stderr

    def test_no_cli_flag_can_override_frozen_scientific_hyperparameters(self):
        source = Path(session_mod.__file__).read_text()
        add_argument_lines = [line for line in source.splitlines() if "add_argument(" in line]
        flags = []
        for line in add_argument_lines:
            if '"--' in line:
                flags.append(line.split('"--')[1].split('"')[0])
        allowed = {"dry-run", "execute-session", "i-confirm-live-cluster-interaction", "rollouts-per-session"}
        assert set(flags) == allowed


class TestLiveSessionUnreachableByDefault:
    def test_run_live_session_checks_hpa_before_any_other_live_action(self):
        source = Path(session_mod.__file__).read_text()
        live_fn_src = source.split("def run_live_session")[1].split("\ndef ")[0]
        hpa_idx = live_fn_src.find("hpa_exists")
        learn_idx = live_fn_src.find("model.learn")
        assert hpa_idx != -1 and learn_idx != -1
        assert hpa_idx < learn_idx

    def test_run_live_session_verifies_provenance_before_loading_a_resumed_checkpoint(self):
        source = Path(session_mod.__file__).read_text()
        live_fn_src = source.split("def run_live_session")[1].split("\ndef ")[0]
        verify_idx = live_fn_src.find("verify_resume_provenance")
        load_idx = live_fn_src.find("PPO.load")
        assert verify_idx != -1 and load_idx != -1
        assert verify_idx < load_idx


class TestSyntheticContinuousVsCheckpointedEquivalence:
    """Phase16W task sec.11's required synthetic equivalence test. Compares
    a continuous mock PPO run against a checkpoint/resume mock run built
    from IDENTICAL initial conditions (same seed, same ScriptedBackend
    timeline). Reports exactly what was verified -- cumulative timestep
    continuity always; weight-tensor closeness only to the tolerance
    actually observed, never claimed as bitwise-exact without proof."""

    def _timeline(self, n=600):
        return {
            i * 30.0: RawMetricSnapshot(
                request_rate_rps=5.0, error_rate_rps=0.0, latency_p95_ms=100.0,
                requests_in_progress=1.0, cpu_cores=0.2, memory_bytes=40_000_000.0,
            )
            for i in range(n)
        }

    def _build_env(self, seed):
        from stable_baselines3 import PPO  # noqa: F401  (import ordering parity with session script)
        env_config = load_env_config_from_yaml()
        backend = ScriptedBackend(timeline=self._timeline(), initial_replicas=1)
        raw_env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=270.0, config=env_config)
        return LedgeredEpisodeWrapper(
            raw_env, ledger_path=Path("/dev/null"), traffic_profile="test", profile_hash="n/a",
            seed=seed, controller_code_hash="n/a", config_hash="n/a", role="training",
        )

    def _ppo_kwargs(self, seed):
        from scripts.train_ppo_live import build_ppo_kwargs_for_live_training
        kwargs = build_ppo_kwargs_for_live_training(seed)
        kwargs["n_steps"] = 32
        kwargs["batch_size"] = 16
        return kwargs

    def test_continuous_vs_checkpointed_cumulative_timesteps_match(self, tmp_path):
        from stable_baselines3 import PPO

        seed = 4242
        torch.manual_seed(seed)
        env_continuous = self._build_env(seed)
        model_continuous = PPO(env=env_continuous, **self._ppo_kwargs(seed))
        model_continuous.learn(total_timesteps=64, reset_num_timesteps=False)  # 2 rollouts of 32

        torch.manual_seed(seed)
        env_session0 = self._build_env(seed)
        model_session0 = PPO(env=env_session0, **self._ppo_kwargs(seed))
        model_session0.learn(total_timesteps=32, reset_num_timesteps=False)
        ckpt_path = tmp_path / "ckpt.zip"
        rng_path = tmp_path / "rng.pkl"
        model_session0.save(ckpt_path)
        save_rng_state(capture_rng_state(), rng_path)

        env_session1 = self._build_env(seed)
        model_session1 = PPO.load(ckpt_path, env=env_session1)
        restore_rng_state(load_rng_state(rng_path))
        model_session1.learn(total_timesteps=32, reset_num_timesteps=False)

        assert model_continuous.num_timesteps == 64
        assert model_session1.num_timesteps == 64
        assert model_continuous.num_timesteps == model_session1.num_timesteps

    def test_continuous_vs_checkpointed_policy_weights_NECESSARILY_diverge(self, tmp_path):
        """EMPIRICAL FINDING, not a bug: a continuous 2-rollout run and a
        checkpoint/resume 1+1-rollout run start from IDENTICAL weights and
        an IDENTICAL RNG stream, yet their final weights measurably diverge
        (observed max abs diff ~1e-2 here, far above float32 noise ~1e-6).
        Root cause, confirmed by inspection (see checkpoint_state.py's
        module docstring): SB3 never pickles the environment object, and
        `PPO.load()`'s `force_reset=True` default forces a fresh
        `env.reset()` at the start of every resumed session's `learn()`
        call -- so the checkpointed run's "rollout 2" starts a BRAND NEW
        episode, while the continuous run's rollout 2 CONTINUES whatever
        episode was already in progress after rollout 1. This difference in
        the underlying TRAINING DATA (not any RNG or weight-restoration
        defect) is the entire source of the divergence, and is an inherent,
        physically-unavoidable property of checkpointing ANY stateful
        Gymnasium environment through SB3 -- not specific to this project's
        implementation. This test asserts the divergence IS observed
        (documenting the finding), rather than asserting closeness that
        would misrepresent it."""
        from stable_baselines3 import PPO

        seed = 4242

        torch.manual_seed(seed)
        env_continuous = self._build_env(seed)
        model_continuous = PPO(env=env_continuous, **self._ppo_kwargs(seed))
        model_continuous.learn(total_timesteps=64, reset_num_timesteps=False)

        torch.manual_seed(seed)
        env_session0 = self._build_env(seed)
        model_session0 = PPO(env=env_session0, **self._ppo_kwargs(seed))
        model_session0.learn(total_timesteps=32, reset_num_timesteps=False)
        ckpt_path = tmp_path / "ckpt.zip"
        rng_path = tmp_path / "rng.pkl"
        model_session0.save(ckpt_path)
        save_rng_state(capture_rng_state(), rng_path)

        env_session1 = self._build_env(seed)
        model_session1 = PPO.load(ckpt_path, env=env_session1)
        restore_rng_state(load_rng_state(rng_path))
        model_session1.learn(total_timesteps=32, reset_num_timesteps=False)

        sd_a = model_continuous.policy.state_dict()
        sd_b = model_session1.policy.state_dict()
        assert sd_a.keys() == sd_b.keys()
        max_abs_diff = max((sd_a[k] - sd_b[k]).abs().max().item() for k in sd_a.keys())
        assert max_abs_diff > 1e-5, (
            "expected a measurable divergence caused by the forced env reset at the "
            f"session boundary, but weights matched far more closely than expected "
            f"(max_abs_diff={max_abs_diff}) -- re-investigate whether the forced-reset "
            "explanation above still holds for this SB3 version."
        )

    def test_two_independent_resumes_from_the_same_checkpoint_are_deterministic(self, tmp_path):
        """The property checkpoint/resume actually needs to guarantee:
        resuming is itself REPRODUCIBLE. Two independent resumes from the
        exact same (checkpoint, RNG-state) pair, each run once, must
        produce IDENTICAL results -- both go through the same forced
        env-reset, so that source of divergence is held constant, isolating
        whether the save/load/RNG-restore machinery itself is deterministic."""
        from stable_baselines3 import PPO

        seed = 4242
        torch.manual_seed(seed)
        env_session0 = self._build_env(seed)
        model_session0 = PPO(env=env_session0, **self._ppo_kwargs(seed))
        model_session0.learn(total_timesteps=32, reset_num_timesteps=False)
        ckpt_path = tmp_path / "ckpt.zip"
        rng_path = tmp_path / "rng.pkl"
        model_session0.save(ckpt_path)
        save_rng_state(capture_rng_state(), rng_path)

        def resume_once():
            env = self._build_env(seed)
            model = PPO.load(ckpt_path, env=env)
            restore_rng_state(load_rng_state(rng_path))
            model.learn(total_timesteps=32, reset_num_timesteps=False)
            return model

        model_resume_a = resume_once()
        model_resume_b = resume_once()

        assert model_resume_a.num_timesteps == model_resume_b.num_timesteps == 64
        sd_a = model_resume_a.policy.state_dict()
        sd_b = model_resume_b.policy.state_dict()
        max_abs_diff = max((sd_a[k] - sd_b[k]).abs().max().item() for k in sd_a.keys())
        assert max_abs_diff < 1e-6, (
            f"two independent resumes from the identical checkpoint+RNG-state diverged "
            f"(max_abs_diff={max_abs_diff}) -- resume is not deterministic as required."
        )

    def test_rng_restoration_measurably_changes_the_resumed_outcome(self, tmp_path):
        """Proves the RNG-restoration step is not a no-op: resuming WITH
        `restore_rng_state()` vs. resuming WITHOUT it (a naive resume that
        only calls `PPO.load()`, which per `_setup_model()` resets RNG to
        the seed's initial state) must produce DIFFERENT outcomes -- if
        they matched, the fix would not be doing anything."""
        from stable_baselines3 import PPO

        seed = 4242
        torch.manual_seed(seed)
        env_session0 = self._build_env(seed)
        model_session0 = PPO(env=env_session0, **self._ppo_kwargs(seed))
        model_session0.learn(total_timesteps=32, reset_num_timesteps=False)
        ckpt_path = tmp_path / "ckpt.zip"
        rng_path = tmp_path / "rng.pkl"
        model_session0.save(ckpt_path)
        save_rng_state(capture_rng_state(), rng_path)

        env_with_restore = self._build_env(seed)
        model_with_restore = PPO.load(ckpt_path, env=env_with_restore)
        restore_rng_state(load_rng_state(rng_path))
        model_with_restore.learn(total_timesteps=32, reset_num_timesteps=False)

        env_naive = self._build_env(seed)
        model_naive = PPO.load(ckpt_path, env=env_naive)
        # Deliberately skip restore_rng_state() -- naive resume.
        model_naive.learn(total_timesteps=32, reset_num_timesteps=False)

        sd_a = model_with_restore.policy.state_dict()
        sd_b = model_naive.policy.state_dict()
        max_abs_diff = max((sd_a[k] - sd_b[k]).abs().max().item() for k in sd_a.keys())
        assert max_abs_diff > 1e-6, (
            "expected RNG restoration to measurably change the outcome relative to a "
            f"naive resume, but they matched (max_abs_diff={max_abs_diff}) -- the "
            "restore_rng_state() step may not be having any effect."
        )


class TestNextRolloutIndexArithmetic:
    """Phase16AD task sec.4: regression tests for the exact defect found in
    Phase16AC -- `(find_latest_checkpoint() or -1) + 1` miscomputed the
    next rollout index as `0` (not `1`) whenever the latest checkpoint was
    legitimately index `0`, because Python's `or` treats the int `0` as
    falsy. `_next_rollout_index()` is the corrected, generically-shaped
    replacement."""

    def test_fresh_start_none_gives_rollout_zero(self):
        assert session_mod._next_rollout_index(None) == 0

    def test_first_resume_zero_gives_rollout_one(self):
        """This is EXACTLY the case that broke in Phase16AC: the latest
        checkpoint is session 0, so the next rollout must be 1."""
        assert session_mod._next_rollout_index(0) == 1

    def test_later_resume_one_gives_rollout_two(self):
        assert session_mod._next_rollout_index(1) == 2

    def test_general_shape_holds_for_further_indices(self):
        for prior, expected in [(2, 3), (5, 6), (41, 42)]:
            assert session_mod._next_rollout_index(prior) == expected

    def test_old_buggy_expression_would_have_been_wrong_at_zero(self):
        """Documents the exact bug being regression-protected against: the
        original inline expression, evaluated directly here, must NOT equal
        the corrected function's output for prior_index=0 -- if it ever did
        (e.g. a future refactor reintroducing the pattern), this assertion
        would fail, flagging the regression."""
        old_buggy_result = (0 or -1) + 1  # noqa: SIM222 (deliberately reproducing the historical bug)
        assert old_buggy_result == 0
        assert session_mod._next_rollout_index(0) == 1
        assert old_buggy_result != session_mod._next_rollout_index(0)


class TestRealResumeProvenanceFlowFromASession000LikeCheckpoint:
    """Phase16AD task sec.5: do not test only the standalone arithmetic
    helper -- also exercise the REAL `find_latest_checkpoint()` (reading
    from an on-disk fixture shaped exactly like a real `session_000/`) and
    the REAL, un-mocked `verify_resume_provenance()`, proving the full
    chain the Phase16AC defect broke now works end-to-end offline."""

    def _build_real_session_000(self, checkpoint_dir: Path) -> dict:
        session_dir = checkpoint_dir / "session_000"
        session_dir.mkdir(parents=True)
        sb3_path = session_dir / session_mod.SB3_CHECKPOINT_NAME
        rng_path = session_dir / session_mod.RNG_STATE_NAME
        sb3_path.write_bytes(b"fake-ppo-model-bytes")
        rng_path.write_bytes(b"fake-rng-state-bytes")
        meta = SessionCheckpointMetadata(
            session_index=0, cumulative_timesteps=256, cumulative_ppo_updates=1,
            cumulative_episodes_completed=6, n_steps=256, training_seed=42,
            sb3_checkpoint_sha256=hashlib.sha256(sb3_path.read_bytes()).hexdigest(),
            rng_state_sha256=hashlib.sha256(rng_path.read_bytes()).hexdigest(),
            governing_hashes_snapshot={"docs/rl_mdp_spec.md": "deadbeef"},
            created_utc=datetime.now(timezone.utc).isoformat(),
        )
        meta.to_json(session_dir / session_mod.METADATA_NAME)
        return {"session_dir": session_dir, "sb3_path": sb3_path, "rng_path": rng_path, "meta": meta}

    def test_real_find_latest_checkpoint_sees_session_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "CHECKPOINT_DIR", tmp_path)
        self._build_real_session_000(tmp_path)
        assert session_mod.find_latest_checkpoint() == 0

    def test_corrected_rollout_index_feeds_a_real_passing_verify_resume_provenance_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "CHECKPOINT_DIR", tmp_path)
        fixture = self._build_real_session_000(tmp_path)

        prior_index = session_mod.find_latest_checkpoint()
        assert prior_index == 0
        rollout_index = session_mod._next_rollout_index(prior_index)
        assert rollout_index == 1

        # The REAL, un-mocked provenance check -- must not raise.
        verify_resume_provenance(
            fixture["meta"], fixture["sb3_path"], fixture["rng_path"],
            expected_next_session_index=rollout_index, expected_n_steps=256,
            expected_training_seed=42,
            current_governing_hashes={"docs/rl_mdp_spec.md": "deadbeef"},
        )

    def test_old_buggy_rollout_index_would_have_failed_the_real_provenance_check(self, tmp_path, monkeypatch):
        """Regression protection (task sec.12): proves that IF the old
        buggy expression's result were used instead, the real
        verify_resume_provenance() call -- unmocked -- correctly rejects
        it. This is exactly the CheckpointProvenanceError Phase16AC
        observed."""
        monkeypatch.setattr(session_mod, "CHECKPOINT_DIR", tmp_path)
        fixture = self._build_real_session_000(tmp_path)

        prior_index = session_mod.find_latest_checkpoint()
        old_buggy_rollout_index = (prior_index or -1) + 1  # noqa: SIM222
        assert old_buggy_rollout_index == 0

        with pytest.raises(CheckpointProvenanceError, match="session_index continuity broken"):
            verify_resume_provenance(
                fixture["meta"], fixture["sb3_path"], fixture["rng_path"],
                expected_next_session_index=old_buggy_rollout_index, expected_n_steps=256,
                expected_training_seed=42,
                current_governing_hashes={"docs/rl_mdp_spec.md": "deadbeef"},
            )

    def test_checkpoint_write_target_after_resume_would_be_session_001(self, tmp_path, monkeypatch):
        """Phase16AD task sec.8: offline proof that the NEXT checkpoint
        this rollout would write targets session_001, never session_000
        (overwrite) or session_002 (skip)."""
        monkeypatch.setattr(session_mod, "CHECKPOINT_DIR", tmp_path)
        self._build_real_session_000(tmp_path)
        prior_index = session_mod.find_latest_checkpoint()
        rollout_index = session_mod._next_rollout_index(prior_index)
        assert session_mod._session_dir(rollout_index).name == "session_001"
        assert session_mod._session_dir(rollout_index) != session_mod._session_dir(prior_index)


class TestResumeSourceTargetsSession000:
    """Phase16AD task sec.6: offline, source-level proof that a resumed
    rollout loads its model/RNG from the PRIOR session directory (never a
    hard-coded or freshly-initialized source), and that
    `reset_num_timesteps=False` governs continuation -- without actually
    running PPO or touching any live backend."""

    def _run_live_session_source(self) -> str:
        source = Path(session_mod.__file__).read_text()
        return source.split("def run_live_session")[1].split("\ndef ")[0]

    def test_ppo_load_targets_the_prior_session_directory(self):
        src = self._run_live_session_source()
        assert "PPO.load(prior_dir / SB3_CHECKPOINT_NAME, env=wrapped)" in src

    def test_rng_restore_targets_the_prior_session_directory(self):
        src = self._run_live_session_source()
        assert "restore_rng_state(load_rng_state(prior_dir / RNG_STATE_NAME))" in src

    def test_fresh_model_branch_is_guarded_by_prior_index_is_none(self):
        src = self._run_live_session_source()
        fresh_branch_idx = src.find("if prior_index is None:\n            model = PPO(")
        assert fresh_branch_idx != -1, (
            "expected the fresh-PPO-model construction to be gated on "
            "`prior_index is None` specifically -- never on a falsy check "
            "that could also match index 0."
        )

    def test_reset_num_timesteps_is_false_for_the_learn_call(self):
        src = self._run_live_session_source()
        assert "model.learn(total_timesteps=FROZEN_N_STEPS, reset_num_timesteps=False)" in src

    def test_no_falsy_or_fallback_remains_in_rollout_index_computation(self):
        """Direct regression guard: the exact buggy substring must never
        reappear in this function."""
        src = self._run_live_session_source()
        assert "find_latest_checkpoint() or -1" not in src
