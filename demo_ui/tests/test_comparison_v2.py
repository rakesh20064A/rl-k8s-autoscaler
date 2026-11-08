"""demo_ui/tests/test_comparison_v2.py

V2 Phase 5E: Comparison (V2) renders the pre-registered evaluation
campaign's already-computed analysis outputs (tables/verdicts/figures)
from a synthetic campaign fixture, and a clean empty state when no
campaign exists. Never recomputes the scientific comparison itself.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

# A minimal valid 1x1 PNG -- comparison_v2.py only displays whatever PNG
# exists via st.image, so a real matplotlib-generated figure is not needed
# to exercise that code path (matplotlib is not installed in the UI venv).
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _make_synthetic_campaign(tmp_path: Path) -> Path:
    campaign_dir = tmp_path / "evaluation_v2" / "campaign_test"
    runs_dir = campaign_dir / "runs"
    for repeat in range(3):
        for arm in ("ppo_v2", "hpa_v2_cpu_memory"):
            run_dir = runs_dir / f"eval_cpu_step__repeat{repeat}__{arm}__attempt1"
            run_dir.mkdir(parents=True)
            latency = 200.0 if arm == "ppo_v2" else 600.0
            series = {
                "run_id": f"eval_cpu_step__repeat{repeat}__{arm}", "profile": "eval_cpu_step", "arm_id": arm,
                "repeat_index": repeat, "decision_interval_s": 30.0,
                "samples": [
                    {"t_s": 0.0, "p95_latency_ms": latency, "error_rate": 0.0, "memory_per_replica_mib": 80.0,
                     "replicas": 2, "action": None, "reversal": False, "invalid": False},
                ],
                "pod_restarts": 0, "load_change_times_s": [],
            }
            with open(run_dir / "normalized_series.json", "w", encoding="utf-8") as f:
                json.dump(series, f)
    return campaign_dir


def _make_analysis_outputs(campaign_dir: Path, project_root: Path, with_figures: bool) -> Path:
    import sys

    sys.path.insert(0, str(project_root))
    from scripts.analyze_v2_evaluation import aggregate_per_profile_arm, analyze_evaluation_campaign, write_evaluation_tables

    run_metrics = analyze_evaluation_campaign(campaign_dir)
    aggregated = aggregate_per_profile_arm(run_metrics)
    analysis_dir = project_root / "persistent-data" / "traffic" / f"{campaign_dir.name}_analysis"
    write_evaluation_tables(aggregated, analysis_dir)
    if with_figures:
        (analysis_dir / "sla_violation_rate_1000ms.png").write_bytes(_TINY_PNG)
        (analysis_dir / "replica_minutes.png").write_bytes(_TINY_PNG)
    return analysis_dir


@pytest.fixture()
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _render_v2_comparison(campaign_dir_str: str, analysis_dir_str: str) -> None:
    """Module-level, no-closure script body -- required by
    `AppTest.from_function` (it extracts and re-execs the function's own
    source, so it must be runnable in isolation; args are passed via
    `AppTest.from_function(..., args=(...))`)."""
    from pathlib import Path

    from demo_ui.views import comparison_v2

    comparison_v2._derive_analysis_dir = lambda cd: Path(analysis_dir_str)  # noqa: SLF001
    comparison_v2.render(Path(campaign_dir_str))


def _render_v2_comparison_no_patch(campaign_dir_str: str) -> None:
    from pathlib import Path

    from demo_ui.views import comparison_v2

    comparison_v2.render(Path(campaign_dir_str))


class TestRenderWithSyntheticCampaign:
    def test_renders_table_and_verdicts_without_exception(self, tmp_path, project_root):
        campaign_dir = _make_synthetic_campaign(tmp_path)
        analysis_dir = _make_analysis_outputs(campaign_dir, project_root, with_figures=False)

        at = AppTest.from_function(_render_v2_comparison, default_timeout=30, args=(str(campaign_dir), str(analysis_dir)))
        at.run()
        assert not at.exception
        html = "".join(m.value for m in at.markdown)
        assert "Evaluation Summary" in html
        assert "ppo_v2" in html

    def test_better_in_all_3_repeats_verdict_present(self, tmp_path, project_root):
        campaign_dir = _make_synthetic_campaign(tmp_path)
        analysis_dir = _make_analysis_outputs(campaign_dir, project_root, with_figures=False)

        at = AppTest.from_function(_render_v2_comparison, default_timeout=30, args=(str(campaign_dir), str(analysis_dir)))
        at.run()
        assert not at.exception
        # The verdict text lives inside an expander's <pre> block.
        found = False
        for exp in at.expander:
            for m in exp.markdown:
                if "ppo_v2_better" in m.value:
                    found = True
        assert found, "expected a ppo_v2_better verdict (ppo_v2 is deterministically lower-latency every repeat)"

    def test_figures_render_when_present(self, tmp_path, project_root):
        campaign_dir = _make_synthetic_campaign(tmp_path)
        analysis_dir = _make_analysis_outputs(campaign_dir, project_root, with_figures=True)

        at = AppTest.from_function(_render_v2_comparison, default_timeout=30, args=(str(campaign_dir), str(analysis_dir)))
        at.run()
        assert not at.exception
        assert len(at.image) == 2


class TestEmptyStates:
    def test_missing_campaign_dir_shows_clean_empty_state(self, tmp_path):
        missing_dir = tmp_path / "does_not_exist"
        at = AppTest.from_function(_render_v2_comparison_no_patch, default_timeout=30, args=(str(missing_dir),))
        at.run()
        assert not at.exception
        html = "".join(m.value for m in at.markdown)
        assert "No V2 evaluation campaign found" in html

    def test_campaign_exists_but_not_yet_analyzed_shows_command_to_run(self, tmp_path):
        campaign_dir = _make_synthetic_campaign(tmp_path)
        no_analysis_dir = tmp_path / "no_analysis_here"

        at = AppTest.from_function(_render_v2_comparison, default_timeout=30, args=(str(campaign_dir), str(no_analysis_dir)))
        at.run()
        assert not at.exception
        html = "".join(m.value for m in at.markdown)
        assert "analysis has not been generated yet" in html
        assert "analyze_v2_evaluation.py" in html
