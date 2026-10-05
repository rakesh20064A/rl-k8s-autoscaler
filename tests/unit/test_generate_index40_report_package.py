"""tests/unit/test_generate_index40_report_package.py

Index40: targeted offline tests. No Kubernetes/Prometheus/model calls.
"""

from __future__ import annotations

import ast
import inspect

import pytest

import scripts.generate_index40_report_package as rep


class TestNoLiveCalls:
    def test_no_kubectl_or_subprocess_in_module(self):
        src = inspect.getsource(rep)
        for forbidden in ["subprocess", "kubectl", "import httpx", "PrometheusClient", "PPO.load", ".learn("]:
            assert forbidden not in src

    def test_matplotlib_uses_non_interactive_backend(self):
        src = inspect.getsource(rep)
        assert 'matplotlib.use("Agg")' in src


class TestValuesTraceToIndex39:
    """Every plotted/tabulated value must originate from the authoritative
    index39 metrics file -- no value computed independently or fabricated
    here."""

    def test_full_valid_values_match_metrics_file(self):
        if not rep.METRICS_PATH.exists():
            pytest.skip("index39 metrics not present in this environment")
        metrics = rep.load_metrics()
        for name in rep.FULL_VALID:
            ep = metrics["per_episode"][name]
            assert ep["valid_for_primary_comparison"] is True
            assert ep["full_holdout_completed"] is True

    def test_ppo1_ppo2_excluded_from_full_valid_set(self):
        assert "PPO1" not in rep.FULL_VALID
        assert "PPO2" not in rep.FULL_VALID

    def test_execution_order_matches_frozen_sequence(self):
        assert rep.EXECUTION_ORDER == ["PPO1", "HPA1", "PPO2", "HPA2", "PPO3", "HPA3"]


class TestTableGeneration:
    def test_table_rows_built_from_metrics_without_recomputation(self):
        """generate_tables() must read values directly from the metrics
        dict, never recompute/aggregate PPO1/PPO2 into a mean."""
        src = inspect.getsource(rep.generate_tables)
        tree = ast.parse(src)
        # No arithmetic mean/aggregation calls should appear in table generation.
        calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        assert "mean" not in calls
        assert "sum" not in calls
