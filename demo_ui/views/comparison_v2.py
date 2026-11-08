"""demo_ui/views/comparison_v2.py -- V2 Comparison view: reads the
pre-registered (CORR-60) evaluation campaign's already-computed analysis
outputs (`scripts/analyze_v2_evaluation.py`'s tables/figures/verdicts),
never recomputing the scientific comparison itself in the UI layer.

Clean empty state until BOTH a completed campaign AND its analysis output
exist -- never a fabricated table/figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _derive_analysis_dir(campaign_dir: Path) -> Path:
    """The convention this project's own reports use (see the Phase 5C-A
    report's Phase 5D checklist): `persistent-data/traffic/<campaign_id>_analysis/`."""
    return PROJECT_ROOT / "persistent-data" / "traffic" / f"{campaign_dir.name}_analysis"


def render(campaign_dir: Path) -> None:
    if not campaign_dir.is_dir():
        st.markdown(
            f'<div class="du-panel mono">No V2 evaluation campaign found at {campaign_dir}. '
            f'Run scripts/evaluate_v2_campaign.py --execute once the final model and epsilon '
            f'calibration exist (see the Phase 5C-A report\'s Phase 5D checklist).</div>',
            unsafe_allow_html=True,
        )
        return

    analysis_dir = _derive_analysis_dir(campaign_dir)
    md_path = analysis_dir / "evaluation_summary.md"
    if not md_path.exists():
        st.markdown(
            f'<div class="du-panel mono">Campaign found at {campaign_dir}, but its analysis has not '
            f'been generated yet. Run:<br/><code>.venv/bin/python scripts/analyze_v2_evaluation.py '
            f'evaluation --campaign-dir {campaign_dir} --output-dir {analysis_dir}</code></div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="du-section-label">Evaluation Summary (mean &plusmn; sd over 3 repeats)</div>', unsafe_allow_html=True)
    st.markdown(md_path.read_text(encoding="utf-8"))

    _render_verdicts(campaign_dir)
    _render_figures(analysis_dir)


def _render_verdicts(campaign_dir: Path) -> None:
    from scripts.analyze_v2_evaluation import ALL_METRICS, analyze_evaluation_campaign, compare_arms_per_profile

    run_metrics = analyze_evaluation_campaign(campaign_dir)
    if not run_metrics:
        return

    st.markdown('<div class="du-section-label" style="margin-top:0.8rem;">Pre-registered "better in all 3 repeats" verdicts</div>', unsafe_allow_html=True)
    st.caption('An arm is declared "better" on a metric for a profile only if it wins all 3 repeats -- otherwise: no decidable winner (a trade-off).')

    pairs = [("ppo_v2", "hpa_v2_cpu_memory"), ("ppo_v2", "hpa_v2_cpu_only"), ("hpa_v2_cpu_memory", "hpa_v2_cpu_only")]
    for arm_a, arm_b in pairs:
        if not any(m["arm_id"] in (arm_a, arm_b) for m in run_metrics):
            continue
        comparison = compare_arms_per_profile(run_metrics, arm_a, arm_b)
        if not comparison:
            continue
        with st.expander(f"{arm_a} vs. {arm_b}"):
            rows = []
            for profile, metrics in comparison.items():
                for metric in ALL_METRICS:
                    verdict = metrics.get(metric, "no_decidable_winner")
                    rows.append(f"{profile:<20} {metric:<32} {verdict}")
            st.markdown(f'<pre class="mono" style="font-size:12px;">{chr(10).join(rows)}</pre>', unsafe_allow_html=True)


def _render_figures(analysis_dir: Path) -> None:
    pngs = sorted(analysis_dir.glob("*.png"))
    if not pngs:
        return
    st.markdown('<div class="du-section-label" style="margin-top:0.8rem;">Figures</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(pngs), 3) or 1)
    for i, png in enumerate(pngs):
        with cols[i % len(cols)]:
            st.image(str(png), caption=png.stem)
