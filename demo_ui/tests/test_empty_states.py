"""Coverage for the "no fixture data leaking into a fresh/empty LIVE
workflow" requirement: before any real run exists, LIVE graphs/metric
cards must show empty-state placeholders -- never fixture/sample/random
values -- while still rendering their frame/axes/labels/legend. Once a
real run exists, its ACTUAL stored samples (and only those) populate the
same chart containers.
"""

from __future__ import annotations

import pytest

pytest.importorskip("plotly")

from demo_ui import experiment_store  # noqa: E402
from demo_ui.components.charts import line_chart, overlay_line_chart, step_chart  # noqa: E402


class TestChartEmptyStateFrame:
    def test_line_chart_with_no_data_has_no_trace_points_but_keeps_layout(self):
        fig = line_chart("P95 Latency", [], {"p95": []}, "ms", empty_message="No data available", empty_detail="Run an experiment to view this metric")
        assert all(len(tr.x or []) == 0 for tr in fig.data)
        assert fig.layout.title.text == "P95 Latency"
        assert fig.layout.yaxis.title.text == "ms"
        anns = fig.layout.annotations
        assert len(anns) == 1
        assert "No data available" in anns[0].text
        assert "Run an experiment to view this metric" in anns[0].text

    def test_empty_annotation_is_compact_two_line_not_one_long_sentence(self):
        """Regression for the "message stretches horizontally across the
        chart" complaint: primary line is short, detail is a SEPARATE,
        smaller, muted line -- never one long concatenated sentence."""
        fig = line_chart("Replicas over time", [], {"replicas": []}, "replicas", empty_message="No data available", empty_detail="Run an experiment to view this metric")
        text = fig.layout.annotations[0].text
        assert text.count("<br>") == 1  # exactly one line break -- two lines, not a run-on sentence
        primary, secondary = text.split("<br>")
        assert len(primary) < 40  # short, centered primary line
        assert "font-size:10px" in secondary  # secondary line is visually de-emphasized

    def test_empty_chart_pins_sensible_axis_range_not_plotly_auto(self):
        fig = line_chart("Replicas over time", [], {"replicas": []}, "replicas", empty_y_range=(0, 8))
        assert list(fig.layout.yaxis.range) == [0, 8]
        assert list(fig.layout.xaxis.range) == [0, 60]
        assert fig.layout.xaxis.title.text == "Elapsed time (s)"

    def test_populated_chart_does_not_pin_axis_range(self):
        """Once real data exists, Plotly must auto-fit it -- a fixed empty
        placeholder range must never clip real measurements."""
        fig = line_chart("Replicas over time", [0.0, 30.0], {"replicas": [1, 5]}, "replicas", empty_y_range=(0, 8))
        assert fig.layout.yaxis.range is None

    def test_line_chart_with_data_has_no_empty_annotation(self):
        fig = line_chart("P95 Latency", [1.0, 2.0], {"p95": [100.0, 110.0]}, "ms", empty_message="should not appear")
        assert list(fig.data[0].x) == [1.0, 2.0]
        assert not fig.layout.annotations

    def test_step_chart_empty_state(self):
        fig = step_chart("Replicas", [], [], empty_message="No data available", empty_detail="Run an experiment to view this metric", empty_y_range=(0, 8))
        assert any("No data available" in ann.text for ann in fig.layout.annotations)
        assert fig.layout.title.text == "Replicas"
        assert list(fig.layout.yaxis.range) == [0, 8]

    def test_overlay_chart_only_hpa_present_omits_ppo_trace_and_message(self):
        fig = overlay_line_chart("Replicas", [1.0], [2], "HPA", [], [], "PPO", "replicas", empty_message="Matching PPO run not available yet.")
        names = [tr.name for tr in fig.data]
        assert names == ["HPA"]  # PPO trace never added, not even empty
        # a partial pair is not the "fully empty" case, so no overlay message
        assert not fig.layout.annotations
        assert fig.layout.yaxis.range is None  # real HPA data must auto-fit, not be clipped to a placeholder range

    def test_overlay_chart_fully_empty_shows_message(self):
        fig = overlay_line_chart("Replicas", [], [], "HPA", [], [], "PPO", "replicas", empty_message="No comparison data", empty_detail="Run HPA and PPO with the same workload", empty_y_range=(0, 8))
        assert len(fig.data) == 0
        anns = fig.layout.annotations
        assert len(anns) == 1
        assert "No comparison data" in anns[0].text
        assert "Run HPA and PPO with the same workload" in anns[0].text
        assert fig.layout.title.text == "Replicas"
        assert list(fig.layout.yaxis.range) == [0, 8]


class TestExtractSeriesNeverFabricates:
    def test_none_run_returns_empty_series(self):
        assert experiment_store.extract_series(None, "replicas_desired") == ([], [])

    def test_running_run_with_no_samples_yet_returns_empty_series(self, tmp_path):
        store_path = tmp_path / "runs.json"
        run_id = experiment_store.start_run("HPA", "constant", path=store_path)
        rec = experiment_store.get_run(run_id, store_path)
        assert experiment_store.extract_series(rec, "replicas_desired") == ([], [])

    def test_completed_run_with_samples_returns_real_values_only(self, tmp_path):
        store_path = tmp_path / "runs.json"
        run_id = experiment_store.start_run("HPA", "constant", path=store_path)
        samples = [
            {"time_s": 0.0, "replicas_desired": 1, "latency_p95_ms": None},
            {"time_s": 30.0, "replicas_desired": 3, "latency_p95_ms": 120.5},
        ]
        experiment_store.update_run(run_id, status="completed", summary={"samples": samples}, path=store_path)
        rec = experiment_store.get_run(run_id, store_path)

        ts, replicas = experiment_store.extract_series(rec, "replicas_desired")
        assert ts == [0.0, 30.0]
        assert replicas == [1, 3]

        # latency_p95_ms is None at t=0 -- must be skipped, never coerced to 0.
        ts_l, lat = experiment_store.extract_series(rec, "latency_p95_ms")
        assert ts_l == [30.0]
        assert lat == [120.5]

    def test_no_zero_fabrication_for_missing_field(self, tmp_path):
        store_path = tmp_path / "runs.json"
        run_id = experiment_store.start_run("PPO", "constant", path=store_path)
        experiment_store.update_run(run_id, status="completed", summary={"samples": []}, path=store_path)
        rec = experiment_store.get_run(run_id, store_path)
        ts, vals = experiment_store.extract_series(rec, "cpu_cores")
        assert ts == [] and vals == []  # not [0.0]
