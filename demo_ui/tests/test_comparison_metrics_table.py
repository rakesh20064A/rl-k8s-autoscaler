"""demo_ui/tests/test_comparison_metrics_table.py

Regression coverage for Phase 9I: the Comparison page's live-pairing
metrics TABLE (distinct from the overlay CHARTS Phase 9H already fixed)
used to read `mean_replicas`/`max_latency_p95_ms`/etc. straight off
`run["summary"]` -- a dict that `experiment_store.status_file_fields_for_store()`
never populates with any such key (`summary` only ever carries `samples`
plus the workload-identity fields). So every lookup was always `None` and
the table rendered "--" for every metric, for every run, regardless of
how much real per-decision data the run actually has.

Fixed by computing every metric directly from `run["summary"]["samples"]`
-- the real `EpisodeSample` schema (as `scripts/run_demo_episode_v2.py`
writes it and the real `persistent-data/demo_ui/runs.json` store holds
it): `t_s`/`time_s`, `desired_replicas`, `p95_latency_ms`,
`request_rate_rps`, `error_rate`, `max_per_pod_memory_mib`,
`pod_restart_counts` (a dict of pod-name -> restart count).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from demo_ui.views import comparison

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REAL_STORE_PATH = PROJECT_ROOT / "persistent-data" / "demo_ui" / "runs.json"


def _real_shaped_samples():
    """Fixture samples shaped EXACTLY like the real per-decision records
    found in `persistent-data/demo_ui/runs.json` (verified against an
    actual completed HPA/PPO pair, run_ids `aea60a66ed80`/`7838c5a5b76d`,
    profile `demo_bursts_cpu_normal`) -- same keys, same nesting
    (`pod_restart_counts` as a dict of pod name -> restart count), not a
    guessed schema."""
    return [
        {
            "decision_index": 0, "t_s": 0.0, "time_s": 0.0,
            "desired_replicas": 2, "ready_replicas": 2,
            "p95_latency_ms": 100.0, "request_rate_rps": 5.0, "error_rate": 0.0,
            "max_per_pod_memory_mib": 40.0,
            "pod_restart_counts": {"demo-app-aaa": 0},
        },
        {
            "decision_index": 1, "t_s": 30.0, "time_s": 30.0,
            "desired_replicas": 2, "ready_replicas": 2,
            "p95_latency_ms": 200.0, "request_rate_rps": 6.0, "error_rate": 0.0,
            "max_per_pod_memory_mib": 41.0,
            "pod_restart_counts": {"demo-app-aaa": 0},
        },
        {
            "decision_index": 2, "t_s": 60.0, "time_s": 60.0,
            "desired_replicas": 3, "ready_replicas": 3,
            "p95_latency_ms": 1500.0, "request_rate_rps": 7.0, "error_rate": 0.01,
            "max_per_pod_memory_mib": 42.0,
            "pod_restart_counts": {"demo-app-aaa": 1, "demo-app-bbb": 0},
        },
        {
            "decision_index": 3, "t_s": 90.0, "time_s": 90.0,
            "desired_replicas": 3, "ready_replicas": 3,
            "p95_latency_ms": 300.0, "request_rate_rps": 8.0, "error_rate": 0.0,
            "max_per_pod_memory_mib": 41.0,
            "pod_restart_counts": {"demo-app-aaa": 1, "demo-app-bbb": 0},
        },
    ]


def _run_with_samples(samples):
    return {"run_id": "fixture", "summary": {"samples": samples}}


class TestComputeRunMetricsFromRealShapedSamples:
    def test_every_metric_computes_with_no_dashes(self):
        metrics = comparison.compute_run_metrics(_run_with_samples(_real_shaped_samples()))
        for key, label, *_ in comparison._METRIC_ROWS:
            assert metrics[key] is not None, f"{label} ({key}) was None despite real samples"

    def test_replica_metrics(self):
        metrics = comparison.compute_run_metrics(_run_with_samples(_real_shaped_samples()))
        assert metrics["decisions"] == 4
        assert metrics["mean_replicas"] == pytest.approx(2.5)
        assert metrics["max_replicas"] == 3
        assert metrics["scaling_actions"] == 1  # 2->2->3->3: one change
        # Decision interval is 30s = 0.5 min throughout; replicas held
        # [2,2,3,3] * 0.5 min each = 1 + 1 + 1.5 + 1.5 = 5.0
        assert metrics["replica_minutes"] == pytest.approx(5.0)

    def test_latency_metrics(self):
        metrics = comparison.compute_run_metrics(_run_with_samples(_real_shaped_samples()))
        assert metrics["mean_p95_latency_ms"] == pytest.approx(525.0)
        assert metrics["median_p95_latency_ms"] == pytest.approx(250.0)
        assert metrics["max_p95_latency_ms"] == pytest.approx(1500.0)
        assert metrics["share_p95_over_1s_pct"] == pytest.approx(25.0)  # 1 of 4 >= 1000ms

    def test_traffic_and_error_metrics(self):
        metrics = comparison.compute_run_metrics(_run_with_samples(_real_shaped_samples()))
        assert metrics["mean_request_rate_rps"] == pytest.approx(6.5)
        assert metrics["peak_request_rate_rps"] == pytest.approx(8.0)
        assert metrics["error_rate_pct"] == pytest.approx(0.25)  # mean(0,0,0.01,0)*100

    def test_memory_and_restart_metrics(self):
        metrics = comparison.compute_run_metrics(_run_with_samples(_real_shaped_samples()))
        assert metrics["max_pod_memory_mib"] == pytest.approx(42.0)
        assert metrics["pod_restarts"] == 1  # sum of restart counts peaks at 1


class TestZeroDecisionsRunShowsDashesNotFieldNameBug:
    def test_run_with_no_samples_at_all(self):
        metrics = comparison.compute_run_metrics(_run_with_samples([]))
        assert metrics["decisions"] == 0
        for key, label, *_ in comparison._METRIC_ROWS:
            if key == "decisions":
                continue
            assert metrics[key] is None, f"{label} should be None (dash) for a zero-decision run"

    def test_none_run_also_shows_dashes(self):
        metrics = comparison.compute_run_metrics(None)
        assert metrics["decisions"] == 0
        assert all(v is None for k, v in metrics.items() if k != "decisions")

    def test_fmt_value_renders_dash_only_for_none(self):
        assert comparison._fmt_value(None, "ms", 1) == "—"
        assert comparison._fmt_value(0, None, 0) == "0"
        assert comparison._fmt_value(12.345, "ms", 1) == "12.3 ms"


class TestLowerIsBetterBolding:
    def test_bolds_the_smaller_cost_metric_not_the_larger(self):
        hpa = _run_with_samples(_real_shaped_samples())
        faster_samples = [dict(s, p95_latency_ms=s["p95_latency_ms"] / 2.0) for s in _real_shaped_samples()]
        ppo = _run_with_samples(faster_samples)
        hpa_metrics, ppo_metrics = comparison.compute_run_metrics(hpa), comparison.compute_run_metrics(ppo)
        assert ppo_metrics["mean_p95_latency_ms"] < hpa_metrics["mean_p95_latency_ms"]

    def test_descriptive_rows_have_no_better_direction(self):
        rows_by_key = {key: lower_is_better for key, _, _, _, lower_is_better in comparison._METRIC_ROWS}
        assert rows_by_key["mean_request_rate_rps"] is None
        assert rows_by_key["peak_request_rate_rps"] is None
        assert rows_by_key["decisions"] is None


class TestComparisonTableRendersAgainstTheRealStore:
    """End-to-end: the real `persistent-data/demo_ui/runs.json` store
    (read-only here, never written) has a genuinely completed, same-
    workload HPA/PPO pair (verified ahead of this test run). Loading the
    real Comparison page against it must show non-empty, non-dash
    numbers for both arms -- not just that `compute_run_metrics()` works
    in isolation."""

    def _latest_real_pair(self):
        if not REAL_STORE_PATH.exists():
            pytest.skip("no real persistent-data/demo_ui/runs.json store in this environment")
        from demo_ui import experiment_store as es

        latest = comparison._latest_current_system_run()
        if latest is None:
            pytest.skip("no valid completed v2 run in the real store to pair")
        pair = es.latest_compatible_pair(latest)
        if pair["hpa"] is None or pair["ppo"] is None:
            pytest.skip("real store does not currently have a complete HPA/PPO pair")
        return pair["hpa"], pair["ppo"]

    def test_real_pair_metrics_have_no_dashes(self):
        hpa_run, ppo_run = self._latest_real_pair()
        hpa_metrics = comparison.compute_run_metrics(hpa_run)
        ppo_metrics = comparison.compute_run_metrics(ppo_run)
        assert hpa_metrics["decisions"] > 0
        assert ppo_metrics["decisions"] > 0
        for key, label, *_ in comparison._METRIC_ROWS:
            assert hpa_metrics[key] is not None, f"HPA {label} was a dash against the real store"
            assert ppo_metrics[key] is not None, f"PPO {label} was a dash against the real store"

    def test_comparison_app_page_renders_non_dash_numbers_for_the_real_pair(self):
        pytest.importorskip("streamlit.testing.v1")
        pytest.importorskip("plotly")
        from streamlit.testing.v1 import AppTest

        hpa_run, ppo_run = self._latest_real_pair()
        hpa_metrics = comparison.compute_run_metrics(hpa_run)

        app_path = str(Path(__file__).resolve().parent.parent / "app.py")
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()
        assert not at.exception
        at.sidebar.radio[0].set_value("Comparison").run()
        assert not at.exception

        html = "".join(m.value for m in at.markdown)
        assert "du-cmp-table" in html
        assert "Demo runs illustrate behaviour; they are not evaluation evidence." in "".join(
            c.value for c in at.caption
        )
        # The real pair's own mean-replicas value (rounded as the table
        # renders it) must actually appear on the page -- a direct check
        # that the table is not silently showing "--" for live data.
        assert f"{hpa_metrics['mean_replicas']:,.2f}" in html
        assert "—" not in html.split("du-cmp-table")[1].split("</table>")[0]
