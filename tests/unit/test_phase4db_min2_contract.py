"""tests/unit/test_phase4db_min2_contract.py

V2 Phase 4D-B / CORR-55: the V2 replica floor changes from 1 to 2 for both
V2 controllers (PPO and demo-app-hpa-v2), and the 768-step campaign under
persistent-data/rl_training_v2/final/ is abandoned in favor of a fresh
final_min2/ namespace. This file proves: the new floor is config-driven
end-to-end (not a scattered literal); V1 is completely unaffected; the
environment's clamp/boundary semantics work at the new floor; the fresh
namespace is distinct and the old one is protected; and a resume attempt
against the old campaign now fails closed on governing-hash drift.

Entirely offline: no Kubernetes/Prometheus access, ScriptedBackend only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from autoscaler.src.rl_env.checkpoint_state import (
    CheckpointProvenanceError,
    SessionCheckpointMetadata,
    verify_resume_provenance,
)
from autoscaler.src.rl_env.config import load_env_config_from_yaml
from autoscaler.src.rl_env.config_v2 import V2ConfigContractError, load_env_config_v2_from_yaml
from autoscaler.src.rl_env.environment_v2 import KubernetesAutoscalerEnvV2
from autoscaler.src.rl_env.reward_v2 import RewardConfigV2, compute_reward_v2
from autoscaler.src.rl_env.scripted_backend import ScriptedBackend
from autoscaler.src.rl_env.types import OVERRIDE_REPLICA_BOUND_CLAMP, RawMetricSnapshot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SESSION_002_DIR = PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final" / "sessions" / "session_002"

HOLD, SCALE_DOWN, SCALE_UP = 1, 0, 2


def _timeline(decision_interval_s: float, n_steps: int) -> dict:
    timeline = {}
    t = 0.0
    for _ in range(n_steps + 1):
        timeline[t] = RawMetricSnapshot(
            request_rate_rps=10.0, error_rate_rps=0.0, latency_p95_ms=50.0,
            requests_in_progress=1.0, cpu_cores=0.1, memory_bytes=20_000_000.0,
            target_up=True, query_error=False,
        )
        t += decision_interval_s
    return timeline


class TestV2FloorIsConfigDriven:
    def test_v2_loader_requires_scaling_v2_block(self, tmp_path):
        """A V2 config missing the scaling_v2 block must fail closed, not
        silently fall back to V1's shared experiment.yaml floor."""
        import yaml

        rl_yaml = PROJECT_ROOT / "configs" / "rl_v2.yaml"
        with open(rl_yaml, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        del raw["scaling_v2"]
        broken_path = tmp_path / "rl_v2_no_scaling.yaml"
        with open(broken_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f)
        with pytest.raises(V2ConfigContractError):
            load_env_config_v2_from_yaml(rl_yaml_path=broken_path)

    def test_v2_floor_2_max_8_initial_2(self):
        cfg = load_env_config_v2_from_yaml()
        assert cfg.gating.min_replicas == 2
        assert cfg.gating.max_replicas == 8
        assert cfg.initial_replicas == 2
        assert cfg.observation.min_replicas == 2
        assert cfg.observation.max_replicas == 8
        assert cfg.reward.min_replicas == 2
        assert cfg.reward.max_replicas == 8

    def test_v1_floor_unchanged_at_1(self):
        cfg = load_env_config_from_yaml()
        assert cfg.gating.min_replicas == 1
        assert cfg.gating.max_replicas == 8
        assert cfg.initial_replicas == 1
        assert cfg.observation.min_replicas == 1

    def test_shared_experiment_yaml_never_edited(self):
        """configs/experiment.yaml is SHARED with V1 -- its own scaling
        block must still read min=1/max=8/initial=1, proving the V2
        override lives entirely in rl_v2.yaml's own scaling_v2 block."""
        import yaml

        with open(PROJECT_ROOT / "configs" / "experiment.yaml", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        assert raw["scaling"]["min_replicas"] == 1
        assert raw["scaling"]["max_replicas"] == 8
        assert raw["scaling"]["initial_replicas"] == 1


class TestRewardV2WasteUsesV2Floor:
    def test_reward_v2_config_from_loader_has_floor_2(self):
        cfg = load_env_config_v2_from_yaml()
        assert isinstance(cfg.reward, RewardConfigV2)
        assert cfg.reward.min_replicas == 2

    def test_waste_does_not_apply_at_floor_2(self):
        cfg = load_env_config_v2_from_yaml()
        result = compute_reward_v2(
            decision_state_valid=True, outcome_state_valid=True,
            p95_latency_ms=50.0, error_rate=0.0,
            deployment_current_replicas=2,  # exactly at the new floor
            cpu_per_replica=0.05, memory_per_replica=0.1,
            requested_delta=0, effective_delta=0, override_reason="none", is_reversal=False,
            config=cfg.reward,
        )
        assert result.components["waste"] == pytest.approx(0.0)

    def test_waste_applies_above_floor_2(self):
        cfg = load_env_config_v2_from_yaml()
        result = compute_reward_v2(
            decision_state_valid=True, outcome_state_valid=True,
            p95_latency_ms=50.0, error_rate=0.0,
            deployment_current_replicas=3,  # above the new floor
            cpu_per_replica=0.05, memory_per_replica=0.1,
            requested_delta=0, effective_delta=0, override_reason="none", is_reversal=False,
            config=cfg.reward,
        )
        assert result.components["waste"] == cfg.reward.w_waste
        assert result.components["waste"] > 0.0

    def test_waste_would_have_wrongly_applied_at_1_under_old_default(self):
        """Documents the second, independently-discovered instance of the
        latent floor=1 assumption this phase fixed: before this fix,
        RewardConfigV2's own Python default (min_replicas=1) was used
        regardless of config, so replicas=2 would have incorrectly
        triggered waste under the OLD code path. The real loader now
        prevents this by overriding min_replicas=2."""
        stale_default_config = RewardConfigV2()  # the bare default, min_replicas=1
        assert stale_default_config.min_replicas == 1
        loader_config = load_env_config_v2_from_yaml().reward
        assert loader_config.min_replicas == 2
        assert loader_config.min_replicas != stale_default_config.min_replicas


class TestEnvironmentV2ClampAtNewFloor:
    def test_reset_starts_at_2(self):
        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg.decision_interval_s, 3)
        backend = ScriptedBackend(timeline=timeline, initial_replicas=2, min_replicas=2, max_replicas=8)
        env = KubernetesAutoscalerEnvV2(backend=backend, episode_duration_s=cfg.decision_interval_s * 3, config=cfg)
        _obs, info = env.reset()
        assert info["deployment_current_replicas_after"] == 2

    def test_scale_down_at_floor_2_clamps_with_replica_bound_clamp(self):
        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg.decision_interval_s, 3)
        backend = ScriptedBackend(timeline=timeline, initial_replicas=2, min_replicas=2, max_replicas=8)
        env = KubernetesAutoscalerEnvV2(backend=backend, episode_duration_s=cfg.decision_interval_s * 3, config=cfg)
        env.reset()
        _obs, _reward, _term, _trunc, info = env.step(SCALE_DOWN)
        assert info["action_override_reason"] == OVERRIDE_REPLICA_BOUND_CLAMP
        assert info["deployment_desired_replicas_after"] == 2  # clamped, never went to 1
        assert info["effective_replica_delta"] == 0

    def test_scale_up_at_max_8_clamps(self):
        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg.decision_interval_s, 3)
        backend = ScriptedBackend(timeline=timeline, initial_replicas=8, min_replicas=2, max_replicas=8)
        env = KubernetesAutoscalerEnvV2(backend=backend, episode_duration_s=cfg.decision_interval_s * 3, config=cfg)
        env.reset()
        _obs, _reward, _term, _trunc, info = env.step(SCALE_UP)
        assert info["action_override_reason"] == OVERRIDE_REPLICA_BOUND_CLAMP
        assert info["deployment_desired_replicas_after"] == 8
        assert info["effective_replica_delta"] == 0

    def test_observation_shape_still_eleven(self):
        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg.decision_interval_s, 3)
        backend = ScriptedBackend(timeline=timeline, initial_replicas=2, min_replicas=2, max_replicas=8)
        env = KubernetesAutoscalerEnvV2(backend=backend, episode_duration_s=cfg.decision_interval_s * 3, config=cfg)
        obs, _info = env.reset()
        assert obs.shape == (11,)
        obs2, *_rest = env.step(HOLD)
        assert obs2.shape == (11,)


class TestFreshCampaignNamespace:
    def test_final_root_points_to_min2_namespace(self):
        import scripts.train_ppo_final_v2 as final_mod

        assert final_mod.FINAL_ROOT.name == "final_min2"
        assert final_mod.FINAL_ROOT.parent.name == "rl_training_v2"

    def test_legacy_final_root_distinct_and_still_the_old_path(self):
        import scripts.train_ppo_final_v2 as final_mod

        assert final_mod.LEGACY_FINAL_ROOT.name == "final"
        assert final_mod.LEGACY_FINAL_ROOT != final_mod.FINAL_ROOT

    def test_old_campaign_directory_untouched_on_disk(self):
        """The old campaign's session_002 evidence must still exist, with
        its original session count, completely untouched by this phase."""
        assert LEGACY_SESSION_002_DIR.is_dir()
        assert (LEGACY_SESSION_002_DIR / "ppo_model.zip").is_file()
        assert (LEGACY_SESSION_002_DIR / "metadata.json").is_file()

    def test_fresh_namespace_not_yet_created_on_disk(self, tmp_path, monkeypatch):
        """No session/ledger/metadata file for a NOT-YET-STARTED namespace
        should exist -- exercised against a tmp path (monkeypatched onto
        FINAL_CAMPAIGN_METADATA_PATH/FINAL_SESSIONS_DIR), never the real
        repo path, which (as of V2 Phase 4E onward) legitimately holds the
        live training campaign's own on-disk state and is therefore not a
        valid proxy for "freshly prepared, not yet started" (V2 Phase 5A /
        CORR-60)."""
        import scripts.train_ppo_final_v2 as final_mod

        monkeypatch.setattr(final_mod, "FINAL_CAMPAIGN_METADATA_PATH", tmp_path / "campaign_metadata.json")
        monkeypatch.setattr(final_mod, "FINAL_SESSIONS_DIR", tmp_path / "sessions")

        assert not final_mod.FINAL_CAMPAIGN_METADATA_PATH.exists()
        assert not (final_mod.FINAL_SESSIONS_DIR / "session_000").exists()

    def test_namespace_isolation_refuses_legacy_final_root(self):
        import scripts.train_ppo_final_v2 as final_mod

        with pytest.raises(final_mod.FinalNamespaceViolation):
            final_mod.assert_final_namespace_isolated(
                [final_mod.LEGACY_FINAL_ROOT / "sessions" / "session_003" / "ppo_model.zip"],
            )

    def test_namespace_isolation_accepts_fresh_namespace(self):
        import scripts.train_ppo_final_v2 as final_mod

        final_mod.assert_final_namespace_isolated(
            [final_mod.FINAL_SESSIONS_DIR / "session_000" / "ppo_model.zip"],
        )  # must not raise

    def test_namespace_isolation_refuses_both_reserved_model_paths(self):
        import scripts.train_ppo_final_v2 as final_mod
        from autoscaler.src.rl_env.training_contract_v2 import V2_MODEL_PATH

        with pytest.raises(final_mod.FinalNamespaceViolation):
            final_mod.assert_final_namespace_isolated([V2_MODEL_PATH])
        with pytest.raises(final_mod.FinalNamespaceViolation):
            final_mod.assert_final_namespace_isolated([final_mod.FINAL_MODEL_PATH_MIN2])


class TestOldCampaignResumeRefused:
    def test_session_002_resume_refused_on_governing_hash_drift(self):
        """Direct proof against the REAL session_002 evidence: resuming it
        now fails closed because configs/rl_v2.yaml's hash (changed by
        this very correction) no longer matches what session_002 recorded
        at training time."""
        meta = SessionCheckpointMetadata.from_json(LEGACY_SESSION_002_DIR / "metadata.json")
        current_hashes = {
            rel: hashlib.sha256((PROJECT_ROOT / rel).read_bytes()).hexdigest()
            for rel in meta.governing_hashes_snapshot
        }
        # Sanity: the drift must specifically be in rl_v2.yaml (the file
        # this phase edited), not a spurious mismatch elsewhere.
        assert current_hashes["configs/rl_v2.yaml"] != meta.governing_hashes_snapshot["configs/rl_v2.yaml"]

        with pytest.raises(CheckpointProvenanceError, match="governing hash drift"):
            verify_resume_provenance(
                meta,
                LEGACY_SESSION_002_DIR / "ppo_model.zip",
                LEGACY_SESSION_002_DIR / "rng_state.pkl",
                expected_next_session_index=meta.session_index + 1,
                expected_n_steps=meta.n_steps,
                expected_training_seed=meta.training_seed,
                current_governing_hashes=current_hashes,
            )

    def test_fresh_namespace_has_no_inherited_state(self, tmp_path, monkeypatch):
        """A fresh campaign namespace with no metadata file yet must make
        `load_campaign_metadata()` return the default (session_000, zero
        cumulative counters), never anything derived from the old final/
        campaign's own state. Exercised against a tmp path (V2 Phase 5A /
        CORR-60 -- see test_fresh_namespace_not_yet_created_on_disk above
        for why the real repo path is no longer a valid proxy for this)."""
        import scripts.train_ppo_final_v2 as final_mod

        monkeypatch.setattr(final_mod, "FINAL_CAMPAIGN_METADATA_PATH", tmp_path / "campaign_metadata.json")
        monkeypatch.setattr(final_mod, "FINAL_SESSIONS_DIR", tmp_path / "sessions")

        assert not final_mod.FINAL_CAMPAIGN_METADATA_PATH.exists()
        state = final_mod.load_campaign_metadata()
        assert state["cumulative_timesteps"] == 0
        assert state["cumulative_updates"] == 0
        assert state["sessions_completed"] == []


class TestEachV2ScriptPassesExplicitReplicaFloor:
    @pytest.mark.parametrize("path", [
        "scripts/train_ppo_final_v2.py",
        "scripts/train_ppo_live_v2.py",
        "scripts/evaluate_ppo_vs_hpa_v2.py",
    ])
    def test_script_passes_explicit_min_and_initial_replicas(self, path):
        source = open(PROJECT_ROOT / path, encoding="utf-8").read()
        assert "min_replicas=env_config" in source
        assert "initial_replicas=env_config" in source

    def test_v1_scripts_untouched(self):
        """Sanity: no V1 script was edited by this phase."""
        import subprocess

        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT, check=False,
        )
        changed = set(result.stdout.split())
        v1_forbidden = {
            "scripts/evaluate_ppo_vs_hpa.py", "configs/rl.yaml", "configs/reward.yaml",
            "configs/experiment.yaml", "k8s/hpa/hpa.yaml", "k8s/app/deployment.yaml",
        }
        assert not (changed & v1_forbidden)


class TestHpaV2DryRunReportsNewFloor:
    def test_dry_run_report_includes_min_replicas_2(self):
        from scripts.evaluate_ppo_vs_hpa_v2 import run_dry_run_validation

        report = run_dry_run_validation()
        assert report["min_replicas"] == 2
        assert report["max_replicas"] == 8
        assert report["initial_replicas"] == 2
        assert report["env_class"] == "KubernetesAutoscalerEnvV2"
