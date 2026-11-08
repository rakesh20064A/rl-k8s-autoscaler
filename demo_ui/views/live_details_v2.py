"""demo_ui/views/live_details_v2.py -- V2 (multi-metric autoscaler) Live
Details rendering, parallel to `views/live_details.py`'s V1 rendering.

Works off ONE plain dict record shape -- the SAME shape
`scripts/run_demo_episode_v2.py` writes into `experiment_store` for a real
LIVE run, and `demo_ui/fixtures_v2.py` produces for DEMO -- so this module
has exactly one rendering code path, never two. Imported by
`views/live_details.py`, which branches to it when the active/latest run's
`generation == "v2"`.

SLA target is read from the record if present, else defaults to the
project's own CORR-57 value (p95 <= 1000ms) -- never a different, silently
chosen number.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from demo_ui import experiment_store
from demo_ui.components.charts import line_chart, step_chart
from demo_ui.components.layout import render_metric_strip
from demo_ui.theme import PALETTE

SLA_P95_TARGET_MS = 1000.0
_REPLICAS_EMPTY_RANGE = (0, 8)
_LATENCY_EMPTY_RANGE = (0, 2000)
_MEMORY_EMPTY_RANGE = (0, 256)

ACTION_COLORS = {"SCALE_UP": PALETTE["vermilion"], "SCALE_DOWN": PALETTE["ochre"], "HOLD": PALETTE["petrol"]}

# CORR-58's safety-aware tie-break rules, in plain language for a
# non-expert examiner -- never shown as a bare "R3" code alone.
TIE_RULE_EXPLANATIONS = {
    "R1": "R1: one action was clearly most likely -- no tie, it was chosen directly.",
    "R2": "R2: scale-down is never chosen to resolve a tie -- an uncertain policy never shrinks capacity by accident.",
    "R3": "R3: hold vs. scale-up was a near-tie -- scaled up because the system was under stress, held otherwise.",
    "R4": "R4: scale-down vs. scale-up (or a 3-way tie) -- held, the safe middle ground.",
    "R5": "R5: scale-down vs. hold was a near-tie -- held, never guessed down.",
    "R6": "R6: stress could not be determined, or the policy failed to produce a usable decision -- held, the fail-safe default.",
}


def _fmt(value, suffix: str = "", digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def render_run_banner(rec: Optional[Dict[str, Any]], is_active: bool) -> None:
    if rec is None:
        st.markdown(
            '<div class="du-panel mono" style="margin-bottom:0.6rem;">No V2 experiment has been run yet.</div>',
            unsafe_allow_html=True,
        )
        return
    profile_label = rec["profile_name"].replace("_", " ").title()
    prefix = "Current run" if is_active else "Latest run"
    data_source = rec.get("data_source", "LIVE")
    status_label = (rec.get("stage") or rec.get("status", "unknown")).upper()
    st.markdown(
        f'<div class="du-panel mono" style="margin-bottom:0.6rem;">{prefix}: V2 {data_source} &middot; '
        f'{rec["controller"]} &middot; {profile_label} &middot; <b>{status_label}</b> &middot; {rec["run_id"]}</div>',
        unsafe_allow_html=True,
    )
    if data_source == "DEMO":
        st.caption("DEMO ONLY -- not evaluation evidence.")


def render_telemetry(rec: Optional[Dict[str, Any]]) -> None:
    st.markdown('<div class="du-section-label" style="margin-top:0.6rem;">Experiment Telemetry</div>', unsafe_allow_html=True)

    samples = (rec.get("summary") or {}).get("samples") or [] if rec is not None else []
    has_samples = bool(samples)
    empty_detail = None
    if rec is None:
        empty_primary = "No data available"
        empty_detail = "Run a V2 demo episode to view this metric"
    elif not has_samples:
        empty_primary = "Waiting for first measurement…"
    else:
        empty_primary = None

    ts_r, replicas = experiment_store.extract_series(rec, "desired_replicas")
    ts_l, p95 = experiment_store.extract_series(rec, "p95_latency_ms")
    ts_m, mem = experiment_store.extract_series(rec, "max_per_pod_memory_mib")
    ts_q, rps = experiment_store.extract_series(rec, "request_rate_rps")
    ts_e, err = experiment_store.extract_series(rec, "error_rate")

    render_metric_strip([
        ("Arm", rec["controller"] if rec is not None else "—"),
        ("Traffic", rec["profile_name"].replace("_", " ").title() if rec is not None else "—"),
        ("RPS", _fmt(rps[-1] if rps else None)),
        ("Replicas (desired/ready)", str(replicas[-1]) if replicas else "—"),
        ("P95", _fmt(p95[-1] if p95 else None, " ms", 0)),
        ("Memory/pod", _fmt(mem[-1] if mem else None, " MiB", 0)),
    ])

    c1, c2 = st.columns(2)
    c1.plotly_chart(
        step_chart("Replicas over time", ts_r, replicas, "replicas",
                   empty_message=empty_primary, empty_detail=empty_detail, empty_y_range=_REPLICAS_EMPTY_RANGE),
        width="stretch",
    )
    p95_fig = line_chart("P95 Latency vs. SLA target", ts_l, {"p95": p95}, "ms",
                          empty_message=empty_primary, empty_detail=empty_detail, empty_y_range=_LATENCY_EMPTY_RANGE)
    p95_fig.add_hline(
        y=SLA_P95_TARGET_MS, line_dash="dash", line_color=PALETTE["vermilion"],
        annotation_text="SLA target: p95 ≤ 1 s", annotation_position="top left",
    )
    c2.plotly_chart(p95_fig, width="stretch")

    c3, c4 = st.columns(2)
    c3.plotly_chart(
        line_chart("Memory per pod", ts_m, {"memory_mib": mem}, "MiB",
                   empty_message=empty_primary, empty_detail=empty_detail, empty_y_range=_MEMORY_EMPTY_RANGE),
        width="stretch",
    )
    c4.plotly_chart(
        line_chart("Request rate & errors", ts_q, {"rps": rps, "error_rate": err}, "req/s",
                   empty_message=empty_primary, empty_detail=empty_detail),
        width="stretch",
    )


def _render_ppo_decision_panel(rec: Dict[str, Any]) -> None:
    """Per-decision explanation panel for the ppo_v2 arm: action
    probabilities, chosen action, plain argmax, tie set + rule, stressed +
    reasons, override reason -- the exact fields the pre-registered
    safety-aware tie-break (CORR-58) produces."""
    st.markdown('<div class="du-section-label" style="margin-top:0.8rem;">PPO Decision Explanation</div>', unsafe_allow_html=True)
    samples = (rec.get("summary") or {}).get("samples") or []
    decisions = [s for s in samples if s.get("ppo_decision")]
    if not decisions:
        st.markdown('<div class="du-panel mono">No PPO decisions recorded yet.</div>', unsafe_allow_html=True)
        return

    latest = decisions[-1]
    decision = latest["ppo_decision"]
    probs = decision.get("action_probabilities") or {}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="du-metric-label">Action probabilities</div>', unsafe_allow_html=True)
        rows = "".join(
            f'<div style="display:flex;justify-content:space-between;">'
            f'<span style="color:{ACTION_COLORS.get(name, PALETTE["muted"])};">{name}</span>'
            f'<span class="mono">{prob:.3f}</span></div>'
            for name, prob in probs.items()
        )
        st.markdown(f'<div class="du-panel">{rows}</div>', unsafe_allow_html=True)

    with col2:
        chosen = decision.get("effective_action_name", "—")
        argmax = decision.get("raw_argmax_action_name", "—")
        st.markdown(
            f'<div class="du-panel mono">'
            f'<div>Chosen action: <b style="color:{ACTION_COLORS.get(chosen, PALETTE["muted"])};">{chosen}</b></div>'
            f'<div>Plain argmax: <b>{argmax}</b></div>'
            f'<div>Tie set: {decision.get("tie_set") or "—"}</div>'
            f'<div>Stressed: <b>{"YES" if decision.get("stressed") else "no"}</b></div>'
            f'<div>Stress reasons: {", ".join(decision.get("stress_reasons") or []) or "—"}</div>'
            f'<div>Override reason: {decision.get("override_reason", "none")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        tie_rule = decision.get("tie_rule_applied")
        if tie_rule:
            st.caption(TIE_RULE_EXPLANATIONS.get(tie_rule, f"Tie rule applied: {tie_rule}"))

    with st.expander(f"Decision history ({len(decisions)} decisions)"):
        rows = []
        for s in decisions:
            d = s["ppo_decision"]
            rows.append(
                f"{s.get('time_s', s.get('t_s', 0)):8.1f}s  action={d.get('effective_action_name', '—'):<10}  "
                f"argmax={d.get('raw_argmax_action_name', '—'):<10}  stressed={d.get('stressed')}  "
                f"override={d.get('override_reason', 'none')}"
            )
        st.markdown(f'<pre class="mono" style="font-size:12px;">{chr(10).join(rows)}</pre>', unsafe_allow_html=True)


def _render_hpa_panel(rec: Dict[str, Any]) -> None:
    """HPA's own CPU% / memory% vs. targets, in place of the PPO decision
    explanation panel, for the hpa_v2_cpu_memory arm."""
    st.markdown('<div class="du-section-label" style="margin-top:0.8rem;">HPA Status</div>', unsafe_allow_html=True)
    samples = (rec.get("summary") or {}).get("samples") or []
    with_hpa = [s for s in samples if s.get("hpa_status")]
    if not with_hpa:
        st.markdown('<div class="du-panel mono">No HPA status recorded yet.</div>', unsafe_allow_html=True)
        return

    latest = with_hpa[-1]["hpa_status"]
    st.markdown(
        f"""
        <div class="du-panel">
          <div class="du-section-label">{latest.get('hpa_name', 'HPA')}</div>
          <div style="display:flex;justify-content:space-between;margin-top:0.3rem;">
            <div>
              <div class="du-metric-label">CPU</div>
              <div class="mono" style="font-size:18px;font-weight:700;">{_fmt(latest.get('current_cpu_pct'), '%', 0)} / {_fmt(latest.get('cpu_target_pct'), '%', 0)}</div>
            </div>
            <div>
              <div class="du-metric-label">Memory</div>
              <div class="mono" style="font-size:18px;font-weight:700;">{_fmt(latest.get('current_memory_pct'), '%', 0)} / {_fmt(latest.get('memory_target_pct'), '%', 0)}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_or_hpa_panel(rec: Optional[Dict[str, Any]]) -> None:
    if rec is None:
        return
    if rec.get("controller") == "ppo_v2":
        _render_ppo_decision_panel(rec)
    else:
        _render_hpa_panel(rec)


def render(rec: Optional[Dict[str, Any]], is_active: bool) -> None:
    """Entry point called by `views/live_details.py` once it determines
    the active/latest run is a V2 run (`rec.get("generation") == "v2"`)."""
    render_run_banner(rec, is_active)
    render_telemetry(rec)
    render_decision_or_hpa_panel(rec)
