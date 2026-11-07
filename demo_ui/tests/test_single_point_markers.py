"""Coverage for the "a chart with exactly one valid sample renders
nothing" bug: `mode="lines"` draws no connecting segment for a single
point, so a real P95 measurement like `[(0, None), (30, 245.38)]` (one
valid coordinate after `extract_series` drops the null) appeared as a
blank plot. Fixed centrally in `demo_ui.components.charts` (all trace
modes switched to "lines+markers"), not per-metric -- these tests prove it
holds for every metric/series the shared helpers draw.
"""

from __future__ import annotations

import pytest

pytest.importorskip("plotly")

from demo_ui import experiment_store  # noqa: E402
from demo_ui.components.charts import line_chart, overlay_line_chart, step_chart  # noqa: E402


def _trace_mode_has_markers(trace) -> bool:
    return "markers" in (trace.mode or "")


class TestLineChartSinglePoint:
    def test_zero_valid_samples_shows_empty_state_no_points(self):
        fig = line_chart("P95 Latency", [], {"p95": []}, "ms", empty_message="No data available")
        assert all(len(tr.x or []) == 0 for tr in fig.data)
        assert fig.layout.annotations

    def test_one_valid_sample_renders_visible_marker(self):
        """[(0, None), (30, 245.38)] after null-filtering is just [(30, 245.38)]."""
        fig = line_chart("P95 Latency", [30.0], {"p95": [245.38]}, "ms")
        assert list(fig.data[0].x) == [30.0]
        assert list(fig.data[0].y) == [245.38]
        assert _trace_mode_has_markers(fig.data[0])
        assert not fig.layout.annotations  # one real point is not the empty case

    def test_two_valid_samples_render_normal_line(self):
        fig = line_chart("P95 Latency", [0.0, 30.0], {"p95": [100.0, 245.38]}, "ms")
        assert list(fig.data[0].x) == [0.0, 30.0]
        assert _trace_mode_has_markers(fig.data[0])
        assert "lines" in fig.data[0].mode

    def test_null_never_converted_to_zero(self):
        """extract_series already filters None out (proven in
        test_empty_states.py); this re-confirms end-to-end that the chart
        itself never sees or fabricates a 0.0 in place of a null sample."""
        run_summary = {"samples": [
            {"time_s": 0.0, "latency_p95_ms": None},
            {"time_s": 30.0, "latency_p95_ms": 245.38},
        ]}
        ts, vals = experiment_store.extract_series({"summary": run_summary}, "latency_p95_ms")
        assert ts == [30.0]
        assert vals == [245.38]
        assert 0.0 not in vals
        fig = line_chart("P95 Latency", ts, {"p95": vals}, "ms")
        assert list(fig.data[0].y) == [245.38]

    @pytest.mark.parametrize("field,value", [
        ("cpu_cores", 0.11),
        ("request_rate_rps", 1.44),
        ("error_rate_rps", 0.0),
    ])
    def test_single_point_marker_applies_to_every_metric(self, field, value):
        """Not a P95-only fix: same behavior for CPU/RPS/error-rate/etc,
        since they all go through the same `line_chart` helper."""
        fig = line_chart(field, [30.0], {field: [value]})
        assert list(fig.data[0].y) == [value]
        assert _trace_mode_has_markers(fig.data[0])


class TestStepChartSinglePoint:
    def test_single_replicas_point_renders_marker(self):
        fig = step_chart("Replicas", [30.0], [3])
        assert list(fig.data[0].x) == [30.0]
        assert list(fig.data[0].y) == [3]
        assert _trace_mode_has_markers(fig.data[0])
        assert not fig.layout.annotations


class TestOverlayChartSinglePoint:
    def test_single_hpa_point_renders_marker_ppo_absent(self):
        fig = overlay_line_chart("Replicas", [30.0], [2], "HPA", [], [], "PPO", "replicas")
        assert len(fig.data) == 1
        assert fig.data[0].name == "HPA"
        assert list(fig.data[0].y) == [2]
        assert _trace_mode_has_markers(fig.data[0])

    def test_single_ppo_point_renders_marker_hpa_absent(self):
        fig = overlay_line_chart("Replicas", [], [], "HPA", [30.0], [1], "PPO", "replicas")
        assert len(fig.data) == 1
        assert fig.data[0].name == "PPO"
        assert list(fig.data[0].y) == [1]
        assert _trace_mode_has_markers(fig.data[0])

    def test_one_point_each_side_shows_both_distinct_markers(self):
        fig = overlay_line_chart("Replicas", [30.0], [4], "HPA", [30.0], [1], "PPO", "replicas")
        assert len(fig.data) == 2
        names = {tr.name for tr in fig.data}
        assert names == {"HPA", "PPO"}
        for tr in fig.data:
            assert _trace_mode_has_markers(tr)
        assert not fig.layout.annotations  # both real -- no "no comparison data" message

    def test_no_fabricated_second_point_introduced_anywhere(self):
        fig = overlay_line_chart("Replicas", [30.0], [4], "HPA", [], [], "PPO", "replicas")
        for tr in fig.data:
            assert len(tr.x) == 1  # never padded to 2 points
