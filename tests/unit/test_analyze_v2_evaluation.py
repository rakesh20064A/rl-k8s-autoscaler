"""tests/unit/test_analyze_v2_evaluation.py

V2 Phase 5B PART C: offline tests for scripts/analyze_v2_evaluation.py.
PART A (evaluation metrics/aggregation/comparison) is tested against
synthetic run fixtures in the normalized `RunSeries` schema. PART B
(training curve) is tested against synthetic session fixtures, plus one
read-only smoke run against the REAL completed final_min2 sessions
(skipped if none exist yet).
"""

from __future__ import annotations

import json

import pytest

import scripts.analyze_v2_evaluation as av


def _make_series_dict(run_id, profile, arm_id, repeat_index, latencies, replicas, actions=None, restarts=0):
    actions = actions or ["HOLD"] * len(latencies)
    samples = [
        {
            "t_s": i * 30.0, "p95_latency_ms": lat, "error_rate": 0.0, "memory_per_replica_mib": 100.0,
            "replicas": rep, "action": act, "reversal": False, "invalid": False,
        }
        for i, (lat, rep, act) in enumerate(zip(latencies, replicas, actions))
    ]
    return {
        "run_id": run_id, "profile": profile, "arm_id": arm_id, "repeat_index": repeat_index,
        "decision_interval_s": 30.0, "samples": samples, "pod_restarts": restarts, "load_change_times_s": [],
    }


class TestNormalizeRunSeries:
    def test_round_trips_fields(self):
        data = _make_series_dict("r1", "eval_cpu_step", "ppo_v2", 0, [100.0, 200.0], [2, 3])
        series = av.normalize_run_series_from_dict(data)
        assert series.run_id == "r1"
        assert len(series.samples) == 2
        assert series.samples[0].p95_latency_ms == 100.0


class TestComputeRunMetrics:
    def test_sla_violation_rate_counts_at_or_above_threshold(self):
        data = _make_series_dict("r1", "p", "ppo_v2", 0, [500.0, 1000.0, 1500.0, 999.0], [2, 2, 2, 2])
        series = av.normalize_run_series_from_dict(data)
        metrics = av.compute_run_metrics(series)
        assert metrics["sla_violation_rate_1000ms"] == pytest.approx(0.5)  # 2 of 4 >= 1000

    def test_replica_minutes_is_time_integral(self):
        data = _make_series_dict("r1", "p", "ppo_v2", 0, [100.0, 100.0], [2, 4])
        series = av.normalize_run_series_from_dict(data)
        metrics = av.compute_run_metrics(series)
        # 2 samples * 30s decision interval: (2 + 4) replicas * 0.5 minutes = 3.0
        assert metrics["replica_minutes"] == pytest.approx(3.0)

    def test_scaling_actions_excludes_hold(self):
        data = _make_series_dict("r1", "p", "ppo_v2", 0, [100.0] * 3, [2, 3, 2], actions=["SCALE_UP", "HOLD", "SCALE_DOWN"])
        series = av.normalize_run_series_from_dict(data)
        metrics = av.compute_run_metrics(series)
        assert metrics["scaling_actions_count"] == 2

    def test_invalid_samples_excluded_from_latency_stats(self):
        data = _make_series_dict("r1", "p", "ppo_v2", 0, [100.0, 9999.0], [2, 2])
        data["samples"][1]["invalid"] = True
        series = av.normalize_run_series_from_dict(data)
        metrics = av.compute_run_metrics(series)
        assert metrics["mean_p95_latency_ms"] == 100.0
        assert metrics["invalid_samples_count"] == 1

    def test_pod_restarts_passthrough(self):
        data = _make_series_dict("r1", "p", "ppo_v2", 0, [100.0], [2], restarts=3)
        series = av.normalize_run_series_from_dict(data)
        metrics = av.compute_run_metrics(series)
        assert metrics["pod_restarts_count"] == 3

    def test_reaction_time_measures_delay_to_next_scaling_action(self):
        data = _make_series_dict("r1", "p", "ppo_v2", 0, [100.0] * 4, [2] * 4, actions=["HOLD", "HOLD", "SCALE_UP", "HOLD"])
        data["load_change_times_s"] = [30.0]  # at t=30s (sample index 1)
        series = av.normalize_run_series_from_dict(data)
        metrics = av.compute_run_metrics(series)
        # first non-HOLD action at or after t=30 is at t=60 (index 2) -> reaction = 30s
        assert metrics["reaction_time_s"] == pytest.approx(30.0)

    def test_no_data_yields_none_not_crash(self):
        data = {"run_id": "r1", "profile": "p", "arm_id": "ppo_v2", "repeat_index": 0, "samples": []}
        series = av.normalize_run_series_from_dict(data)
        metrics = av.compute_run_metrics(series)
        assert metrics["mean_p95_latency_ms"] is None
        assert metrics["sla_violation_rate_1000ms"] is None


class TestAggregatePerProfileArm:
    def test_mean_and_sd_over_three_repeats(self):
        run_metrics = [
            av.compute_run_metrics(av.normalize_run_series_from_dict(_make_series_dict(f"r{i}", "p", "ppo_v2", i, [v], [2])))
            for i, v in enumerate([100.0, 200.0, 300.0])
        ]
        aggregated = av.aggregate_per_profile_arm(run_metrics)
        agg = aggregated[("p", "ppo_v2")]
        assert agg["n_repeats"] == 3
        assert agg["mean_p95_latency_ms"]["mean"] == pytest.approx(200.0)
        assert agg["mean_p95_latency_ms"]["sd"] == pytest.approx(100.0)

    def test_missing_metric_values_reported_as_none(self):
        data = {"run_id": "r1", "profile": "p", "arm_id": "ppo_v2", "repeat_index": 0, "samples": []}
        run_metrics = [av.compute_run_metrics(av.normalize_run_series_from_dict(data))]
        aggregated = av.aggregate_per_profile_arm(run_metrics)
        agg = aggregated[("p", "ppo_v2")]
        assert agg["mean_p95_latency_ms"]["mean"] is None


class TestIsBetterInAllRepeats:
    def test_lower_is_better_true_when_strictly_better_every_repeat(self):
        assert av.is_better_in_all_repeats([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], lower_is_better=True) is True

    def test_lower_is_better_false_when_one_repeat_ties_or_loses(self):
        assert av.is_better_in_all_repeats([1.0, 5.0, 3.0], [4.0, 5.0, 6.0], lower_is_better=True) is False

    def test_mismatched_lengths_returns_none(self):
        assert av.is_better_in_all_repeats([1.0], [1.0, 2.0], lower_is_better=True) is None

    def test_empty_returns_none(self):
        assert av.is_better_in_all_repeats([], [], lower_is_better=True) is None


class TestCompareArmsPerProfile:
    def test_ppo_declared_better_when_strictly_lower_every_repeat(self):
        run_metrics = []
        for i, (ppo_lat, hpa_lat) in enumerate([(100.0, 500.0), (150.0, 550.0), (120.0, 600.0)]):
            run_metrics.append(av.compute_run_metrics(av.normalize_run_series_from_dict(
                _make_series_dict(f"ppo{i}", "p", "ppo_v2", i, [ppo_lat], [2]))))
            run_metrics.append(av.compute_run_metrics(av.normalize_run_series_from_dict(
                _make_series_dict(f"hpa{i}", "p", "hpa_v2_cpu_memory", i, [hpa_lat], [2]))))
        comparison = av.compare_arms_per_profile(run_metrics, "ppo_v2", "hpa_v2_cpu_memory")
        assert comparison["p"]["mean_p95_latency_ms"] == "ppo_v2_better"

    def test_no_decidable_winner_when_repeats_disagree(self):
        run_metrics = []
        for i, (ppo_lat, hpa_lat) in enumerate([(100.0, 500.0), (600.0, 550.0), (120.0, 600.0)]):
            run_metrics.append(av.compute_run_metrics(av.normalize_run_series_from_dict(
                _make_series_dict(f"ppo{i}", "p", "ppo_v2", i, [ppo_lat], [2]))))
            run_metrics.append(av.compute_run_metrics(av.normalize_run_series_from_dict(
                _make_series_dict(f"hpa{i}", "p", "hpa_v2_cpu_memory", i, [hpa_lat], [2]))))
        comparison = av.compare_arms_per_profile(run_metrics, "ppo_v2", "hpa_v2_cpu_memory")
        assert comparison["p"]["mean_p95_latency_ms"] == "no_decidable_winner"


class TestWriteEvaluationTables:
    def test_csv_and_md_written(self, tmp_path):
        run_metrics = [av.compute_run_metrics(av.normalize_run_series_from_dict(
            _make_series_dict("r1", "p", "ppo_v2", 0, [100.0], [2])))]
        aggregated = av.aggregate_per_profile_arm(run_metrics)
        result = av.write_evaluation_tables(aggregated, tmp_path)
        assert result["csv"].exists()
        assert result["md"].exists()
        assert "p" in result["md"].read_text()


class TestPlotEvaluationFigures:
    def test_figures_written_for_primary_metrics(self, tmp_path):
        run_metrics = [av.compute_run_metrics(av.normalize_run_series_from_dict(
            _make_series_dict("r1", "p", "ppo_v2", 0, [100.0], [2])))]
        aggregated = av.aggregate_per_profile_arm(run_metrics)
        paths = av.plot_evaluation_figures(aggregated, tmp_path)
        assert len(paths) == len(av.PRIMARY_METRICS)
        assert all(p.exists() for p in paths)


class TestAnalyzeEvaluationCampaign:
    def test_reads_normalized_series_from_run_folders(self, tmp_path):
        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "run1"
        run_dir.mkdir(parents=True)
        with open(run_dir / "normalized_series.json", "w", encoding="utf-8") as f:
            json.dump(_make_series_dict("run1", "p", "ppo_v2", 0, [100.0], [2]), f)
        # A run with no normalized series yet is skipped, not guessed at.
        (runs_dir / "run2_no_series").mkdir()

        run_metrics = av.analyze_evaluation_campaign(tmp_path)
        assert len(run_metrics) == 1
        assert run_metrics[0]["run_id"] == "run1"

    def test_missing_campaign_dir_returns_empty(self, tmp_path):
        assert av.analyze_evaluation_campaign(tmp_path / "does_not_exist") == []


# ---------------------------------------------------------------------------
# PART B -- training curve
# ---------------------------------------------------------------------------


def _make_training_session(root, name, session_index, reward_values, replicas_values, actions_values):
    d = root / name
    d.mkdir(parents=True)
    (d / "ppo_model.zip").write_bytes(b"x")
    (d / "metadata.json").write_text("{}")
    with open(d / "step_telemetry.jsonl", "w", encoding="utf-8") as f:
        for reward, replicas, action in zip(reward_values, replicas_values, actions_values):
            row = {
                "reward_total": reward,
                "reward_components": {"reliability": reward, "sla": 0.0, "pressure": 0.0, "waste": 0.0, "action": 0.0, "reversal": 0.0, "boundary": 0.0},
                "sampled_action": action, "effective_action": action,
                "deployment_current_replicas_after": replicas,
            }
            f.write(json.dumps(row) + "\n")
    return d


class TestFindCompletedTrainingSessions:
    def test_only_sessions_with_checkpoint_and_metadata(self, tmp_path):
        _make_training_session(tmp_path, "session_000", 0, [1.0], [2], [1])
        incomplete = tmp_path / "session_001"
        incomplete.mkdir()
        result = av.find_completed_training_sessions(tmp_path)
        assert [p.name for p in result] == ["session_000"]


class TestSummarizeSessionTelemetry:
    def test_mean_reward_and_action_mix_and_replica_distribution(self, tmp_path):
        d = _make_training_session(tmp_path, "session_000", 0, [1.0, 3.0], [2, 4], [1, 2])
        summary = av.summarize_session_telemetry(d)
        assert summary["mean_step_reward"] == pytest.approx(2.0)
        assert summary["action_mix_counts"] == {"SCALE_DOWN": 0, "HOLD": 1, "SCALE_UP": 1}
        assert summary["replica_distribution_counts"] == {2: 1, 4: 1}
        assert summary["mean_replicas"] == pytest.approx(3.0)


class TestAnalyzeTrainingCurve:
    def test_joins_update_metrics_by_session_index(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        _make_training_session(sessions_dir, "session_000", 0, [1.0], [2], [1])
        _make_training_session(sessions_dir, "session_001", 1, [2.0], [3], [1])

        update_metrics_path = tmp_path / "ppo_update_metrics.jsonl"
        with open(update_metrics_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"session_index": 0, "entropy_loss": -1.1, "approx_kl": 0.001, "clip_fraction": 0.0, "explained_variance": 0.01, "value_loss": 10.0}) + "\n")
            f.write(json.dumps({"session_index": 1, "entropy_loss": -1.0, "approx_kl": 0.002, "clip_fraction": 0.01, "explained_variance": 0.02, "value_loss": 8.0}) + "\n")

        curve = av.analyze_training_curve(sessions_dir, update_metrics_path)
        assert len(curve) == 2
        assert curve[0]["session_index"] == 0
        assert curve[0]["entropy_loss"] == -1.1
        assert curve[1]["value_loss"] == 8.0

    def test_missing_update_metrics_file_yields_none_fields_not_crash(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        _make_training_session(sessions_dir, "session_000", 0, [1.0], [2], [1])
        curve = av.analyze_training_curve(sessions_dir, tmp_path / "does_not_exist.jsonl")
        assert curve[0]["entropy_loss"] is None


class TestWriteTrainingCurveOutputs:
    def test_csv_md_and_figures_written(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        _make_training_session(sessions_dir, "session_000", 0, [1.0], [2], [1])
        curve = av.analyze_training_curve(sessions_dir, tmp_path / "no_update_metrics.jsonl")
        output_dir = tmp_path / "out"
        result = av.write_training_curve_outputs(curve, output_dir)
        assert result["csv"].exists()
        assert result["md"].exists()
        assert len(result["figures"]) >= 1


class TestRealTrainingCurveSmoke:
    """V2 Phase 5B: run PART B now, read-only, against whatever real
    final_min2 sessions have completed so far (skipped if none exist)."""

    def test_real_sessions_analyze_without_crashing(self):
        completed = av.find_completed_training_sessions(av.DEFAULT_SESSIONS_DIR)
        if not completed:
            pytest.skip("no completed V2 final_min2 sessions exist yet")
        curve = av.analyze_training_curve()
        assert len(curve) == len(completed)
        assert all(row["mean_step_reward"] is not None for row in curve)
