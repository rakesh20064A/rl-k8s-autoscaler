"""demo_ui/views/training_v2.py -- V2 Training page: the final_min2
training curve, read-only, regenerated from the real completed sessions
on every render (never written into persistent-data/rl_training_v2/
itself -- any cached output goes under demo_ui/.run_store/, the same
gitignored, demo-only directory experiment_store.py already uses).

Imports scripts/analyze_v2_evaluation.py's training-curve functions
unmodified; this file adds only the Streamlit presentation layer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_ui.components.layout import render_topbar  # noqa: E402

TRAINING_CURVE_CACHE_DIR = Path(__file__).resolve().parent.parent / ".run_store" / "training_curve_cache"

_CHART_CAPTIONS = {
    "entropy_loss": "More negative = the policy is more decisive (less random); flattening out means it has settled on a strategy.",
    "approx_kl": "How much the policy changed this update; small and stable values mean learning is proceeding smoothly, not lurching.",
    "explained_variance": "How well the value function predicts actual returns; closer to 1.0 is better.",
    "value_loss": "Error in the value function's own predictions; a falling trend means it is learning to estimate outcomes more accurately.",
    "mean_step_reward": "Average reward per decision this session; an upward trend means the policy is improving overall behaviour.",
    "mean_replicas": "Average replica count the policy chose this session.",
}


def _total_planned_sessions() -> int:
    """Read-only import of the live budget constant -- never hardcoded, so
    this label can never drift from CORR-59's actual amended value."""
    try:
        from scripts.train_ppo_final_v2 import FINAL_PLANNED_SESSIONS

        return FINAL_PLANNED_SESSIONS
    except Exception:  # noqa: BLE001 -- a display label must never crash the page
        return 48


def _load_real_curve() -> List[Dict[str, Any]]:
    from scripts.analyze_v2_evaluation import analyze_training_curve, write_training_curve_outputs

    curve = analyze_training_curve()
    if curve:
        # Regenerated fresh every render into the UI's own gitignored cache
        # dir (never into persistent-data/rl_training_v2/ itself) -- a
        # convenience artifact for anyone wanting the CSV/figures outside
        # the UI, not a dependency of this view's own rendering below.
        try:
            write_training_curve_outputs(curve, TRAINING_CURVE_CACHE_DIR)
        except Exception:  # noqa: BLE001 -- caching (incl. matplotlib, which the UI venv may lack) is best-effort only
            pass
    return curve


def _build_fixture_curve(n_sessions: int = 6) -> List[Dict[str, Any]]:
    """A small, clearly-labeled DEMO curve for when no real sessions have
    completed yet -- deterministic, not read from any real training data."""
    curve = []
    for i in range(n_sessions):
        frac = i / max(n_sessions - 1, 1)
        curve.append({
            "session_index": i,
            "entropy_loss": -1.1 + 0.3 * frac,
            "approx_kl": 0.0005 + 0.01 * frac,
            "clip_fraction": 0.0 + 0.05 * frac,
            "explained_variance": 0.01 + 0.3 * frac,
            "value_loss": 18.0 - 10.0 * frac,
            "mean_step_reward": -0.5 + 0.4 * frac,
            "mean_replicas": 3.5 + 1.5 * frac,
            "n_steps": 256,
        })
    return curve


def _fmt_num(value: Any, digits: int = 4) -> str:
    """A session whose `ppo_update_metrics.jsonl` row hasn't been flushed
    yet (e.g. the live campaign is mid-session) reports `None` for every
    update-metric field -- never fabricated as 0.0, and never allowed to
    crash this page's formatting."""
    return "—" if value is None else f"{value:.{digits}f}"


def _render_table(curve: List[Dict[str, Any]]) -> None:
    rows = "\n".join(
        f"{row['session_index']:>3}  entropy_loss={_fmt_num(row['entropy_loss'])}  approx_kl={_fmt_num(row['approx_kl'], 5)}  "
        f"explained_variance={_fmt_num(row['explained_variance'])}  value_loss={_fmt_num(row['value_loss'], 3)}  "
        f"mean_step_reward={_fmt_num(row['mean_step_reward'])}  mean_replicas={_fmt_num(row['mean_replicas'], 2)}"
        for row in curve
    )
    st.markdown(f'<pre class="mono" style="font-size:11px;">{rows}</pre>', unsafe_allow_html=True)


def _render_charts(curve: List[Dict[str, Any]]) -> None:
    from demo_ui.components.charts import line_chart

    sessions = [row["session_index"] for row in curve]
    metrics = ("entropy_loss", "approx_kl", "explained_variance", "value_loss", "mean_step_reward", "mean_replicas")
    cols = st.columns(2)
    for i, metric in enumerate(metrics):
        values = [row.get(metric) for row in curve]
        fig = line_chart(metric.replace("_", " ").title(), sessions, {metric: values}, metric)
        with cols[i % 2]:
            st.plotly_chart(fig, width="stretch")
            st.caption(_CHART_CAPTIONS.get(metric, ""))


def _render_action_mix(curve: List[Dict[str, Any]]) -> None:
    action_mixes = [row.get("action_mix_counts") for row in curve if row.get("action_mix_counts")]
    if not action_mixes:
        return
    st.markdown('<div class="du-section-label" style="margin-top:0.6rem;">Action Mix per Session</div>', unsafe_allow_html=True)
    rows = []
    for row in curve:
        mix = row.get("action_mix_counts") or {}
        rows.append(
            f"session {row['session_index']:>3}  "
            f"SCALE_DOWN={mix.get('SCALE_DOWN', 0):>4}  HOLD={mix.get('HOLD', 0):>4}  SCALE_UP={mix.get('SCALE_UP', 0):>4}"
        )
    st.markdown(f'<pre class="mono" style="font-size:11px;">{chr(10).join(rows)}</pre>', unsafe_allow_html=True)


def render() -> None:
    render_topbar(mode_label="TRAINING", is_live=False)
    st.markdown('<div class="du-section-label" style="margin-top:0.6rem;">V2 Training Curve</div>', unsafe_allow_html=True)
    st.caption(
        "Read-only, regenerated from the real completed sessions on every load -- "
        "never writes into persistent-data/rl_training_v2/."
    )

    try:
        curve = _load_real_curve()
    except Exception as exc:  # noqa: BLE001 -- training may be live; a read race must never crash this page
        st.caption(f"Could not read training sessions right now: {exc}")
        curve = []

    total_planned = _total_planned_sessions()

    if not curve:
        st.markdown(
            '<div class="du-panel mono">No completed V2 final_min2 sessions yet.</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Demo fixture data (simulated, not real training results)"):
            st.caption("Optional simulated reference curve -- not generated until you load it.")
            if st.button("Load demo fixture curve", key="du_load_training_fixture"):
                st.session_state["du_show_training_fixture"] = True
            if st.session_state.get("du_show_training_fixture", False):
                fixture_curve = _build_fixture_curve()
                st.caption(f"DEMO ONLY -- simulated, not real training data. ({len(fixture_curve)} fixture sessions.)")
                _render_charts(fixture_curve)
                _render_table(fixture_curve)
        return

    st.markdown(
        f'<div class="du-panel mono">Sessions completed: <b>{len(curve)} / {total_planned}</b></div>',
        unsafe_allow_html=True,
    )
    _render_charts(curve)
    _render_action_mix(curve)
    with st.expander(f"Per-session values ({len(curve)} sessions)"):
        _render_table(curve)
