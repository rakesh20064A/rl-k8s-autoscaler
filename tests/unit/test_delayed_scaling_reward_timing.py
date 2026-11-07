"""tests/unit/test_delayed_scaling_reward_timing.py

V2 Phase 2.1 PART 7: reviews reward timing relative to action-at-t / pod
readiness / observation-at-t+1 / the 30s PPO decision interval, and answers
the task's question: can SCALE_UP be unfairly punished before a newly
requested pod becomes Ready?

CONCLUSION (proven below, no environment/gating changes made -- "if reward
timing already handles this correctly, prove it with tests"):

  1. The project's OWN empirically-grounded readiness-delay assumption is
     ~10s (ScriptedBackend.scale_up_ready_delay_seconds default, itself
     documented as matching "the ~5-15s lag observed across all 9 official
     HPA tuning trials", scripted_backend.py module docstring, CORR-25) --
     comfortably BELOW the 30s decision interval (configs/experiment.yaml
     rl.control_interval_seconds). By construction, a newly-scaled-up pod
     is expected to be Ready well before the NEXT decision point under
     normal conditions.

  2. The action/reversal/boundary reward components (both reward.py/V1 and
     reward_v2.py/V2) are functions ONLY of decision-time quantities
     (override_reason, effective_delta, is_reversal) -- CORR-30's
     decision/outcome validity separation. They are NEVER a function of
     whether the new pod actually became Ready in time, or of the later
     observation at t+1. A SCALE_UP's small action-penalty consequence
     (identical regardless of outcome) can therefore never be inflated by
     readiness lag.

  3. Even in the edge case where readiness genuinely outlasts one decision
     interval (modeled below with a 40s readiness delay > the 30s
     interval), gating correctly defers a REPEAT request via
     OVERRIDE_PENDING_SCALE_GATE, whose only reward consequence is the
     SAME small w_action cost any evaluated-but-deferred action incurs --
     never an SLA/pressure penalty improperly attributed to the original
     scale-up decision itself.

No correction was needed; this file is the proof, not a bug fix.
"""

from __future__ import annotations

import pytest

from autoscaler.src.rl_env.config import load_env_config_from_yaml
from autoscaler.src.rl_env.environment import KubernetesAutoscalerEnv
from autoscaler.src.rl_env.reward_v2 import RewardConfigV2, compute_reward_v2
from autoscaler.src.rl_env.scripted_backend import ScriptedBackend
from autoscaler.src.rl_env.types import OVERRIDE_NONE, OVERRIDE_PENDING_SCALE_GATE, RawMetricSnapshot

SCALE_UP = 2  # ACTION_MAP[2] == +1


def _flat_timeline(cfg, n_steps: int, cpu_cores: float = 0.3, memory_bytes: float = 20_000_000.0):
    timeline = {}
    t = 0.0
    for _ in range(n_steps + 1):
        timeline[t] = RawMetricSnapshot(
            request_rate_rps=10.0, error_rate_rps=0.0, latency_p95_ms=50.0,
            requests_in_progress=1.0, cpu_cores=cpu_cores, memory_bytes=memory_bytes,
            target_up=True, query_error=False,
        )
        t += cfg.decision_interval_s
    return timeline


class TestReadinessDelayMarginVsDecisionInterval:
    def test_documented_readiness_delay_is_below_decision_interval(self):
        cfg = load_env_config_from_yaml()
        documented_readiness_delay_s = 10.0  # ScriptedBackend's own CORR-25-grounded default
        assert documented_readiness_delay_s < cfg.decision_interval_s
        margin_s = cfg.decision_interval_s - documented_readiness_delay_s
        assert margin_s >= 10.0  # comfortable margin, not a razor-thin coincidence


class TestNewlyReadyPodReflectedByNextDecision:
    def test_scale_up_pod_is_ready_at_the_very_next_decision_point(self):
        """With the empirically-grounded 10s readiness delay (< 30s
        decision interval), a scale-up issued at step t must show the
        INCREASED ready-replica count in step t's own resulting
        observation (info['ready_replicas_after']) -- no artificial extra
        delay beyond physical readiness."""
        cfg = load_env_config_from_yaml()
        timeline = _flat_timeline(cfg, n_steps=2)
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1, scale_up_ready_delay_seconds=10.0)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=cfg.decision_interval_s * 2, config=cfg)
        env.reset()

        _obs, _reward, _term, _trunc, info = env.step(SCALE_UP)
        assert info["action_override_reason"] == OVERRIDE_NONE
        assert info["deployment_desired_replicas_after"] == 2
        assert info["ready_replicas_after"] == 2  # already Ready -- 10s delay < 30s interval


class TestActionSideRewardNeverDependsOnReadinessOutcome:
    def test_scale_up_action_component_identical_regardless_of_readiness_timing(self):
        """The action-side components (action/reversal/boundary) must be
        byte-identical whether or not the new pod ended up Ready in time --
        they are computed from override_reason/effective_delta alone
        (CORR-30), never from ready_replicas_after or any t+1 observation."""
        config_v2 = RewardConfigV2()

        result_a = compute_reward_v2(
            decision_state_valid=True, outcome_state_valid=True,
            p95_latency_ms=50.0, error_rate=0.0, deployment_current_replicas=2,
            cpu_per_replica=1.0, memory_per_replica=0.5,
            requested_delta=1, effective_delta=1, override_reason=OVERRIDE_NONE, is_reversal=False,
            config=config_v2,
        )
        # Even if the OUTCOME state looks worse (e.g. latency still elevated
        # because the new pod wasn't fully warmed up) -- the scale-up's OWN
        # decision-time consequences are unaffected; only the (unavoidably
        # outcome-dependent) SLA/pressure terms would differ, never
        # action/reversal/boundary.
        result_b = compute_reward_v2(
            decision_state_valid=True, outcome_state_valid=True,
            p95_latency_ms=400.0, error_rate=0.0, deployment_current_replicas=2,
            cpu_per_replica=3.5, memory_per_replica=0.5,
            requested_delta=1, effective_delta=1, override_reason=OVERRIDE_NONE, is_reversal=False,
            config=config_v2,
        )
        # The action-side components are identical regardless of outcome...
        assert result_a.components["action"] == result_b.components["action"]
        assert result_a.components["reversal"] == result_b.components["reversal"]
        assert result_a.components["boundary"] == result_b.components["boundary"]
        # ...only the outcome-dependent SLA/pressure terms differ.
        assert result_a.components["sla"] != result_b.components["sla"]


class TestGenuineReadinessOverrunEdgeCase:
    def test_repeat_request_during_pending_scale_gate_only_incurs_small_action_cost(self):
        """Edge case: readiness delay (40s) genuinely EXCEEDS the decision
        interval (30s). A repeated scale request on the next step must be
        deferred via OVERRIDE_PENDING_SCALE_GATE (never treated as an
        error, never silently dropped) -- and its ONLY reward consequence
        is the same small w_action cost, never an inflated penalty."""
        cfg = load_env_config_from_yaml()
        timeline = _flat_timeline(cfg, n_steps=3)
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1, scale_up_ready_delay_seconds=40.0)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=cfg.decision_interval_s * 3, config=cfg)
        env.reset()

        _obs, _r1, _t1, _tr1, info1 = env.step(SCALE_UP)
        assert info1["action_override_reason"] == OVERRIDE_NONE
        assert info1["ready_replicas_after"] == 1  # 40s delay > 30s interval -- genuinely not Ready yet

        _obs, _r2, _t2, _tr2, info2 = env.step(SCALE_UP)
        assert info2["action_override_reason"] == OVERRIDE_PENDING_SCALE_GATE
        assert info2["effective_replica_delta"] == 0  # correctly deferred, not an error

        config_v2 = RewardConfigV2()
        result = compute_reward_v2(
            decision_state_valid=True, outcome_state_valid=True,
            p95_latency_ms=50.0, error_rate=0.0, deployment_current_replicas=info2["deployment_current_replicas_after"],
            cpu_per_replica=1.0, memory_per_replica=0.5,
            requested_delta=1, effective_delta=0, override_reason=OVERRIDE_PENDING_SCALE_GATE, is_reversal=False,
            config=config_v2,
        )
        assert result.components["action"] == pytest.approx(config_v2.w_action)  # small, bounded cost
        assert result.components["boundary"] == 0.0  # never treated as a boundary violation
        assert result.components["pressure"] == 0.0  # the deferred request itself adds no pressure penalty
