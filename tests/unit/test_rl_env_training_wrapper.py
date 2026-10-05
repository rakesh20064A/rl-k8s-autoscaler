"""tests/unit/test_rl_env_training_wrapper.py

Phase 16V: tests for autoscaler/src/rl_env/training_wrapper.py's
LedgeredEpisodeWrapper -- episode-outcome classification and ledger-append
semantics. Uses ScriptedBackend exclusively; no live Kubernetes/Prometheus.
"""

from __future__ import annotations

from autoscaler.src.rl_env import episode_ledger
from autoscaler.src.rl_env.config import load_env_config_from_yaml
from autoscaler.src.rl_env.environment import KubernetesAutoscalerEnv
from autoscaler.src.rl_env.scripted_backend import ScriptedBackend
from autoscaler.src.rl_env.training_wrapper import (
    STATUS_SAFETY_ABORTED,
    STATUS_TOOLING_FAILED,
    STATUS_TRUNCATED,
    LedgeredEpisodeWrapper,
)
from autoscaler.src.rl_env.types import RawMetricSnapshot

ACTION_HOLD = 1
CFG = load_env_config_from_yaml()


def _valid_snapshot(**overrides):
    base = dict(
        request_rate_rps=5.0, error_rate_rps=0.0, latency_p95_ms=100.0,
        requests_in_progress=1.0, cpu_cores=0.2, memory_bytes=40_000_000.0,
    )
    base.update(overrides)
    return RawMetricSnapshot(**base)


def _make_wrapper(ledger_path, episode_duration_s, timeline, **kwargs):
    backend = ScriptedBackend(timeline=timeline, initial_replicas=1)
    env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=episode_duration_s, config=CFG)
    return LedgeredEpisodeWrapper(
        env, ledger_path=ledger_path, traffic_profile="test_profile", profile_hash="abc123",
        seed=1, controller_code_hash="def456", config_hash="ghi789", role="training", **kwargs,
    )


class TestTruncatedEpisode:
    def test_normal_completion_logs_truncated_and_valid(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        timeline = {t: _valid_snapshot() for t in [0.0, 30.0, 60.0]}
        wrapper = _make_wrapper(ledger_path, episode_duration_s=60.0, timeline=timeline)
        wrapper.reset(seed=1)
        for _ in range(3):
            _, _, terminated, truncated, _ = wrapper.step(ACTION_HOLD)
            if terminated or truncated:
                break
        rows = list(episode_ledger.read_rows(ledger_path))
        assert len(rows) == 1
        assert rows[0]["status"] == STATUS_TRUNCATED
        assert rows[0]["valid"] is True
        assert rows[0]["role"] == "training"
        assert rows[0]["traffic_profile"] == "test_profile"


class TestSafetyAbortedEpisode:
    def test_termination_logs_safety_aborted_and_invalid(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        # Only t=0 has scripted data -> every subsequent step is invalid ->
        # 4th consecutive invalid step terminates (docs/rl_mdp_spec.md sec.13).
        timeline = {0.0: _valid_snapshot()}
        wrapper = _make_wrapper(ledger_path, episode_duration_s=10_000.0, timeline=timeline)
        wrapper.reset(seed=1)
        for _ in range(6):
            _, _, terminated, truncated, _ = wrapper.step(ACTION_HOLD)
            if terminated or truncated:
                break
        rows = list(episode_ledger.read_rows(ledger_path))
        assert len(rows) == 1
        assert rows[0]["status"] == STATUS_SAFETY_ABORTED
        assert rows[0]["valid"] is False
        assert len(rows[0]["safety_events"]) > 0


class TestToolingFailure:
    def test_log_tooling_failure_writes_invalid_row(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        timeline = {0.0: _valid_snapshot()}
        wrapper = _make_wrapper(ledger_path, episode_duration_s=300.0, timeline=timeline)
        wrapper.reset(seed=1)
        wrapper.log_tooling_failure(RuntimeError("simulated restoration timeout"), phase="reset")
        rows = list(episode_ledger.read_rows(ledger_path))
        assert len(rows) == 1
        assert rows[0]["status"] == STATUS_TOOLING_FAILED
        assert rows[0]["valid"] is False
        assert "simulated restoration timeout" in rows[0]["tooling_failure_error"]
        assert rows[0]["tooling_failure_phase"] == "reset"


class TestMultipleEpisodesAppend:
    def test_two_episodes_produce_two_distinct_ledger_rows(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        timeline = {t: _valid_snapshot() for t in [0.0, 30.0, 60.0]}
        wrapper = _make_wrapper(ledger_path, episode_duration_s=30.0, timeline=timeline)

        wrapper.reset(seed=1)
        wrapper.step(ACTION_HOLD)  # truncates immediately (30s episode)

        wrapper.reset(seed=2)
        wrapper.step(ACTION_HOLD)

        rows = list(episode_ledger.read_rows(ledger_path))
        assert len(rows) == 2
        assert rows[0]["episode_id"] != rows[1]["episode_id"]


class TestPpoUpdateIndexRecorded:
    def test_set_ppo_update_index_is_recorded_in_ledger_row(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        timeline = {t: _valid_snapshot() for t in [0.0, 30.0]}
        wrapper = _make_wrapper(ledger_path, episode_duration_s=30.0, timeline=timeline)
        wrapper.reset(seed=1)
        wrapper.set_ppo_update_index(7)
        wrapper.step(ACTION_HOLD)
        rows = list(episode_ledger.read_rows(ledger_path))
        assert rows[0]["ppo_update_index"] == 7


class TestRetryProvenancePhase16AB:
    """Proves a retried attempt at the SAME scientific episode slot (e.g.
    after an earlier tooling failure, as happened in Phase16X-Retry) gets a
    distinct physical_attempt_id while the scientific episode_id -- and the
    prior failure row -- both remain exactly as they were."""

    def test_first_attempt_gets_physical_attempt_one(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        timeline = {t: _valid_snapshot() for t in [0.0, 30.0]}
        wrapper = _make_wrapper(ledger_path, episode_duration_s=30.0, timeline=timeline)
        wrapper.reset(seed=1)
        wrapper.step(ACTION_HOLD)
        rows = list(episode_ledger.read_rows(ledger_path))
        assert rows[0]["physical_attempt_number"] == 1
        # Phase17: physical_attempt_id is scoped to scientific_episode_id
        # (globally unique), not the legacy per-process episode_id.
        assert rows[0]["physical_attempt_id"] == f"{rows[0]['scientific_episode_id']}__attempt1"

    def test_retry_after_prior_tooling_failure_gets_distinct_attempt_number_same_episode_id(self, tmp_path):
        """Simulates exactly the Phase16X-Retry scenario: a prior PROCESS
        already wrote a tooling_failed row for this scientific slot (a
        fresh LedgeredEpisodeWrapper instance, as a real retry would
        construct in a NEW process with zero in-memory history), then a
        retry wrapper is constructed and reaches the SAME episode_id."""
        ledger_path = tmp_path / "ledger.jsonl"
        timeline = {0.0: _valid_snapshot()}

        # First (failed) process attempt.
        wrapper_a = _make_wrapper(ledger_path, episode_duration_s=9999.0, timeline=timeline)
        wrapper_a.reset(seed=1)
        wrapper_a.log_tooling_failure(RuntimeError("route establishment failed"), phase="reset")
        rows_after_failure = list(episode_ledger.read_rows(ledger_path))
        assert len(rows_after_failure) == 1
        failed_episode_id = rows_after_failure[0]["episode_id"]
        failed_scientific_episode_id = rows_after_failure[0]["scientific_episode_id"]
        assert rows_after_failure[0]["physical_attempt_number"] == 1

        # Retry: a BRAND NEW wrapper instance (as a fresh process would
        # construct), same role/profile/seed -> same scientific episode_id.
        timeline2 = {t: _valid_snapshot() for t in [0.0, 30.0]}
        wrapper_b = _make_wrapper(ledger_path, episode_duration_s=30.0, timeline=timeline2)
        wrapper_b.reset(seed=1)
        wrapper_b.step(ACTION_HOLD)

        rows_after_retry = list(episode_ledger.read_rows(ledger_path))
        assert len(rows_after_retry) == 2
        # The ORIGINAL failure row is untouched.
        assert rows_after_retry[0] == rows_after_failure[0]
        # The retry reached the identical SCIENTIFIC episode_id (legacy
        # AND the new globally-unique one, since both wrappers here have no
        # external global_schedule_episode_index authority and so both
        # fall back to the same per-process index 0)...
        assert rows_after_retry[1]["episode_id"] == failed_episode_id
        assert rows_after_retry[1]["scientific_episode_id"] == failed_scientific_episode_id
        # ...but got a DISTINCT, incremented physical attempt identity.
        assert rows_after_retry[1]["physical_attempt_number"] == 2
        assert rows_after_retry[1]["physical_attempt_id"] != rows_after_failure[0]["physical_attempt_id"]
        assert rows_after_retry[1]["physical_attempt_id"] == f"{failed_scientific_episode_id}__attempt2"

    def test_prior_failure_row_never_overwritten_or_deleted(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        wrapper_a = _make_wrapper(ledger_path, episode_duration_s=9999.0, timeline={0.0: _valid_snapshot()})
        wrapper_a.reset(seed=1)
        wrapper_a.log_tooling_failure(RuntimeError("boom"), phase="reset")
        before = ledger_path.read_text()

        wrapper_b = _make_wrapper(ledger_path, episode_duration_s=9999.0, timeline={0.0: _valid_snapshot()})
        wrapper_b.reset(seed=1)
        wrapper_b.log_tooling_failure(RuntimeError("boom again"), phase="reset")
        after = ledger_path.read_text()

        assert after.startswith(before)  # strictly additive -- the original failure line is byte-for-byte preserved
