"""tests/unit/test_analyze_final_evaluation.py

Index39: targeted offline tests for scripts/analyze_final_evaluation.py.
No Kubernetes/Prometheus/model calls -- pure function tests against
synthetic sample/decision fixtures, plus a smoke test against the real
on-disk evidence (read-only).
"""

from __future__ import annotations

import json

import pytest

import scripts.analyze_final_evaluation as ana


def _sample(t, p95=100.0, req=10.0, err=0.0, replicas=1, target_up=True):
    return {
        "time_s": t, "request_rate_rps": req, "error_rate_rps": err,
        "latency_p95_ms": p95, "requests_in_progress": 0.0, "cpu_cores": 0.5,
        "memory_bytes": 1e8, "target_up": target_up,
        "replicas_desired": replicas, "replicas_current": replicas,
        "replicas_ready": replicas, "replicas_available": replicas,
        "latency_p50_ms": p95 / 2, "latency_p99_ms": p95 * 1.2, "success_rate_rps": req,
    }


class TestSlaViolationRate:
    def test_all_below_threshold_is_zero(self):
        ep = {"samples": [_sample(0)] + [_sample(i * 30, p95=100.0) for i in range(1, 6)], "decisions": []}
        m = ana.compute_section10_metrics(ep)
        assert m["sla_violation_rate"] == 0.0

    def test_half_above_threshold(self):
        samples = [_sample(0)]
        samples += [_sample(i * 30, p95=400.0) for i in range(1, 4)]
        samples += [_sample(i * 30, p95=100.0) for i in range(4, 7)]
        m = ana.compute_section10_metrics({"samples": samples, "decisions": []})
        assert m["sla_violation_rate"] == pytest.approx(0.5)


class TestErrorRateMean:
    def test_zero_request_rate_counts_as_zero_not_excluded(self):
        samples = [_sample(0)] + [_sample(30, req=0.0, err=0.0)]
        m = ana.compute_section10_metrics({"samples": samples, "decisions": []})
        assert m["error_rate_mean"] == 0.0

    def test_nonzero_error_rate_computed_correctly(self):
        samples = [_sample(0)] + [_sample(30, req=10.0, err=2.0)]
        m = ana.compute_section10_metrics({"samples": samples, "decisions": []})
        assert m["error_rate_mean"] == pytest.approx(0.2)


class TestResourceProxyMean:
    def test_one_replica_is_zero_proxy(self):
        samples = [_sample(0, replicas=1)] + [_sample(30, replicas=1)]
        m = ana.compute_section10_metrics({"samples": samples, "decisions": []})
        assert m["resource_proxy_mean"] == 0.0

    def test_eight_replicas_is_one_proxy(self):
        samples = [_sample(0, replicas=8)] + [_sample(30, replicas=8)]
        m = ana.compute_section10_metrics({"samples": samples, "decisions": []})
        assert m["resource_proxy_mean"] == pytest.approx(1.0)


class TestActionChurnCount:
    def test_ppo_counts_only_effective_replica_changes(self):
        decisions = [
            {"decision_time_s": 30, "replicas_before": 1, "replicas_after": 1},  # HOLD, not effective
            {"decision_time_s": 60, "replicas_before": 1, "replicas_after": 2},  # effective
            {"decision_time_s": 90, "replicas_before": 2, "replicas_after": 2},  # clamped/no-op
        ]
        ep = {"samples": [_sample(0), _sample(30), _sample(60, replicas=2), _sample(90, replicas=2)], "decisions": decisions}
        m = ana.compute_section10_metrics(ep)
        assert m["action_churn_count"] == 1

    def test_hpa_counts_replica_transitions_without_decisions_log(self):
        samples = [_sample(0, replicas=1), _sample(30, replicas=1), _sample(60, replicas=2), _sample(90, replicas=2)]
        m = ana.compute_section10_metrics({"samples": samples, "decisions": []})
        assert m["action_churn_count"] == 1


class TestReactionTimeSeconds:
    def test_no_breach_is_none(self):
        samples = [_sample(0)] + [_sample(30, p95=100.0)]
        m = ana.compute_section10_metrics({"samples": samples, "decisions": []})
        assert m["reaction_time_seconds"] is None

    def test_breach_with_subsequent_effective_scale_up_ppo(self):
        samples = [_sample(0), _sample(30, p95=400.0, replicas=1), _sample(60, p95=400.0, replicas=2)]
        decisions = [
            {"decision_time_s": 30, "replicas_before": 1, "replicas_after": 1},
            {"decision_time_s": 60, "replicas_before": 1, "replicas_after": 2},
        ]
        m = ana.compute_section10_metrics({"samples": samples, "decisions": decisions})
        assert m["reaction_time_seconds"] == pytest.approx(30.0)

    def test_breach_with_subsequent_scale_up_hpa_no_decisions_log(self):
        samples = [_sample(0, replicas=1), _sample(30, p95=400.0, replicas=1), _sample(60, p95=400.0, replicas=2)]
        m = ana.compute_section10_metrics({"samples": samples, "decisions": []})
        assert m["reaction_time_seconds"] == pytest.approx(30.0)

    def test_breach_resolves_before_scale_up_still_searches_forward(self):
        """A scale-up occurring AFTER the breach has already naturally
        resolved must still count for that onset (no truncation bug)."""
        samples = [
            _sample(0, replicas=1), _sample(30, p95=400.0, replicas=1),
            _sample(60, p95=100.0, replicas=1), _sample(90, p95=100.0, replicas=2),
        ]
        m = ana.compute_section10_metrics({"samples": samples, "decisions": []})
        assert m["reaction_time_seconds"] == pytest.approx(60.0)


class TestLegacyClassification:
    def test_episode_without_status_field_is_classified(self):
        ep = {"n_decision_steps": 24, "workload_duration_seconds": 1500.0}
        classified = ana._classify_legacy(ep)
        assert classified["episode_status"] == "terminated_safety"
        assert classified["full_holdout_completed"] is False
        assert classified["valid_for_primary_comparison"] is False

    def test_episode_with_existing_status_field_is_untouched(self):
        ep = {"episode_status": "completed_full_duration", "valid_for_primary_comparison": True}
        assert ana._classify_legacy(ep) is ep


class TestRealEvidenceSmoke:
    """Read-only smoke test against the actual committed evidence -- no
    mutation, no live calls."""

    def test_run_analysis_against_real_evidence_matches_expected_counts(self):
        if not ana.EVAL_DIR.exists():
            pytest.skip("real evaluation evidence not present in this environment")
        result = ana.run_analysis()
        assert result["hpa_valid_repeats"] == [1, 2, 3]
        assert result["ppo_valid_repeats"] == [3]
        assert result["formal_n3_applicable"] is False
        assert result["holdout_hash"] == ana.EXPECTED_HOLDOUT_HASH

    def test_outputs_are_written_and_readable(self, tmp_path):
        if not ana.EVAL_DIR.exists():
            pytest.skip("real evaluation evidence not present in this environment")
        result = ana.run_analysis()
        metrics_path = tmp_path / "metrics.json"
        ana.write_metrics_json(result, metrics_path)
        with open(metrics_path) as f:
            data = json.load(f)
        assert data["formal_n3_rule_applicable"] is False
        assert "PPO1" in data["per_episode"]
