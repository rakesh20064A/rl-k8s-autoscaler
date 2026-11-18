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

from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from autoscaler.src.rl_env.types import ACTION_NAMES as _REAL_RUN_ACTION_NAMES
from demo_ui import experiment_store
from demo_ui.components.charts import line_chart, step_chart
from demo_ui.components.controls_v2 import display_arm_label
from demo_ui.components.layout import render_metric_strip
from demo_ui.theme import PALETTE
from scripts.generate_demo_profiles import LOAD_LABELS, PATTERN_LABELS, RESOURCE_LABELS, parse_profile_name

SLA_P95_TARGET_MS = 1000.0
_REPLICAS_EMPTY_RANGE = (0, 8)
_LATENCY_EMPTY_RANGE = (0, 2000)
_MEMORY_EMPTY_RANGE = (0, 256)

# A REAL run's `effective_action_name`/`raw_argmax_action_name` use
# `autoscaler.src.rl_env.types.ACTION_NAMES` (`SCALE_DOWN_ONE`/`HOLD`/
# `SCALE_UP_ONE`); `demo_ui.fixtures_v2`'s DEMO (sample data) records use
# the shorter `SCALE_DOWN`/`HOLD`/`SCALE_UP`. Both are kept here (a
# superset, never a rename of either side's own data) so this panel colors
# an action consistently regardless of which one produced the record.
ACTION_COLORS = {
    "SCALE_UP": PALETTE["vermilion"], "SCALE_UP_ONE": PALETTE["vermilion"],
    "SCALE_DOWN": PALETTE["ochre"], "SCALE_DOWN_ONE": PALETTE["ochre"],
    "HOLD": PALETTE["petrol"],
}

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


def _traffic_label(profile_name: Optional[str]) -> str:
    """"Pattern / Resource / Load" (Title Case, " / " separators) for a
    36-catalog profile name; falls back to a title-cased version of the
    raw name for a legacy `demo_*_short` profile, which has no such
    decomposition. The ONE place this formatting happens in this module
    (`render_run_banner`/`render_telemetry`/the run selector all share
    it), so a profile is never described two different ways on the same
    page."""
    if not profile_name:
        return "—"
    combo = parse_profile_name(profile_name)
    if combo is None:
        return profile_name.replace("_", " ").title()
    pattern, resource, load = combo
    return (
        f"{PATTERN_LABELS.get(pattern, pattern)} / {RESOURCE_LABELS.get(resource, resource)} / "
        f"{LOAD_LABELS.get(load, load)}"
    )


def run_selector_label(rec: Dict[str, Any]) -> str:
    """Task sec.4's exact format: "PPO · Bursts / CPU / High · 18 Nov
    12:40 · completed" -- controller family (the short "PPO"/"HPA" label,
    never the raw `ppo_v2`/`hpa_v2_cpu_memory` id) · traffic description
    (Title Case, " / " separators) · day-month-time (no leading zero on
    the day) · status (lowercase, whatever `experiment_store` currently
    has for it -- "running"/"completed"/"error"/"stopped"/`STALE_STATUS`/
    "refused")."""
    arm = display_arm_label(rec.get("controller"))
    traffic = _traffic_label(rec.get("profile_name"))
    started_at = rec.get("started_at")
    try:
        when = f"{datetime.fromtimestamp(started_at).day} {datetime.fromtimestamp(started_at).strftime('%b %H:%M')}"
    except (TypeError, ValueError, OSError):
        when = "—"
    status = str(rec.get("status") or "unknown").lower()
    return f"{arm} · {traffic} · {when} · {status}"


def _selectable_v2_runs() -> List[Dict[str, Any]]:
    """Every v2 run worth offering in the Live Details run selector,
    newest-first, with the currently active run (if any) always first
    regardless of its `started_at` (it is, almost by definition, the most
    recently started) -- the active run plus every TERMINAL-status run
    `experiment_store.completed_runs()` already tracks (never a
    duplicate: a run that just went active->terminal between two calls is
    de-duplicated by `run_id`). A run from a different generation (V1) is
    never offered here."""
    active = experiment_store.active_run()
    if active is not None and active.get("generation") != "v2":
        active = None
    runs: List[Dict[str, Any]] = [active] if active is not None else []
    seen = {active["run_id"]} if active is not None else set()
    for rec in experiment_store.completed_runs():
        if rec.get("generation") != "v2" or rec["run_id"] in seen:
            continue
        runs.append(rec)
        seen.add(rec["run_id"])
    return runs


def render_run_selector() -> "tuple[Optional[Dict[str, Any]], bool]":
    """Task sec.4: the run selector at the top of Live Details. Returns
    `(selected_rec_or_None, is_active)`. Defaults to the active run if one
    exists, else the latest completed run (`_selectable_v2_runs()`'s own
    ordering already puts exactly that option first) -- Streamlit's own
    widget-state persistence (keyed on `run_id`, a stable identity across
    a running->terminal transition) keeps a user's explicit pick sticky
    across reruns without any extra bookkeeping here. Returns `(None,
    False)` with no widget at all if there is nothing to select -- the
    caller renders the "no run yet" empty state instead."""
    runs = _selectable_v2_runs()
    if not runs:
        return None, False
    by_id = {r["run_id"]: r for r in runs}
    run_ids = list(by_id.keys())
    active = experiment_store.active_run()
    active_id = active["run_id"] if (active is not None and active.get("generation") == "v2") else None
    # Task sec.3: an `invalid_no_traffic` run is still fully selectable
    # here (its record/samples are kept exactly as they are), but it must
    # never be the auto-selected DEFAULT -- prefer the first run that
    # isn't one, falling back to `run_ids[0]` only if every selectable
    # run happens to be invalidated.
    default_id = active_id if active_id in by_id else next(
        (rid for rid in run_ids if by_id[rid].get("status") != experiment_store.INVALID_NO_TRAFFIC),
        run_ids[0],
    )
    selected_id = st.selectbox(
        "Run", run_ids, index=run_ids.index(default_id),
        format_func=lambda rid: run_selector_label(by_id[rid]), key="du_v2_live_details_run_selector",
    )
    selected = by_id.get(selected_id, by_id[default_id])
    return selected, selected["run_id"] == active_id


def render_run_banner(rec: Optional[Dict[str, Any]], is_active: bool) -> None:
    if rec is None:
        st.markdown(
            '<div class="du-panel mono" style="margin-bottom:0.6rem;">No experiment has been run yet.</div>',
            unsafe_allow_html=True,
        )
        st.info("Start a run on the Experiment page")
        return
    profile_label = _traffic_label(rec["profile_name"])
    prefix = "Current run" if is_active else "Latest run"
    data_source = rec.get("data_source", "LIVE")
    status_label = (rec.get("stage") or rec.get("status", "unknown")).upper()
    st.markdown(
        f'<div class="du-panel mono" style="margin-bottom:0.6rem;">{prefix}: {data_source} &middot; '
        f'{display_arm_label(rec["controller"])} &middot; {profile_label} &middot; <b>{status_label}</b> &middot; {rec["run_id"]}</div>',
        unsafe_allow_html=True,
    )
    if data_source == "DEMO":
        st.caption("SIMULATED DATA -- not evaluation evidence.")


def render_telemetry(rec: Optional[Dict[str, Any]]) -> None:
    st.markdown('<div class="du-section-label" style="margin-top:0.6rem;">Experiment Telemetry</div>', unsafe_allow_html=True)

    samples = (rec.get("summary") or {}).get("samples") or [] if rec is not None else []
    has_samples = bool(samples)
    empty_detail = None
    if rec is None:
        empty_primary = "No data available"
        empty_detail = "Run a demo episode to view this metric"
    elif not has_samples:
        empty_primary = "Waiting for first measurement…"
    else:
        empty_primary = None

    ts_r, replicas = experiment_store.extract_series(rec, "desired_replicas")
    ts_l, p95 = experiment_store.extract_series(rec, "p95_latency_ms")
    ts_m, mem = experiment_store.extract_series(rec, "max_per_pod_memory_mib")
    ts_q, rps = experiment_store.extract_series(rec, "request_rate_rps")
    ts_e, err = experiment_store.extract_series(rec, "error_rate")

    traffic_label = _traffic_label(rec["profile_name"]) if rec is not None else "—"

    render_metric_strip([
        ("Arm", display_arm_label(rec["controller"]) if rec is not None else "—"),
        ("Traffic", traffic_label),
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
    c1.caption("Desired replica count the controller issued, over the run's elapsed time.")
    p95_fig = line_chart("P95 Latency vs. SLA target", ts_l, {"p95": p95}, "ms",
                          empty_message=empty_primary, empty_detail=empty_detail, empty_y_range=_LATENCY_EMPTY_RANGE)
    p95_fig.add_hline(
        y=SLA_P95_TARGET_MS, line_dash="dash", line_color=PALETTE["vermilion"],
        annotation_text="SLA target: p95 ≤ 1 s", annotation_position="top left",
    )
    c2.plotly_chart(p95_fig, width="stretch")
    c2.caption("p95 latency; dashed line = 1s target.")

    c3, c4 = st.columns(2)
    c3.plotly_chart(
        line_chart("Memory per pod", ts_m, {"memory_mib": mem}, "MiB",
                   empty_message=empty_primary, empty_detail=empty_detail, empty_y_range=_MEMORY_EMPTY_RANGE),
        width="stretch",
    )
    c3.caption("Peak memory used by any one pod, in MiB.")
    c4.plotly_chart(
        line_chart("Request rate & errors", ts_q, {"rps": rps, "error_rate": err}, "req/s",
                   empty_message=empty_primary, empty_detail=empty_detail),
        width="stretch",
    )
    c4.caption("Incoming request rate and error rate the workload observed, in req/s.")


def _action_probabilities_as_dict(decision: Dict[str, Any]) -> Dict[str, float]:
    """A REAL run (`scripts/run_demo_episode_v2.py`) writes
    `action_probabilities` as the RAW index-ordered list/tuple the policy
    produced -- index 0/1/2 -> SCALE_DOWN_ONE/HOLD/SCALE_UP_ONE
    (`autoscaler.src.rl_env.types.ACTION_NAMES`, the SAME names
    `effective_action_name`/`raw_argmax_action_name` already use), never a
    name-keyed dict. `demo_ui.fixtures_v2`'s DEMO (sample data) records
    already provide a dict. Accepts either shape so this panel never
    crashes trying to call `.items()` on a list."""
    raw = decision.get("action_probabilities") or {}
    if isinstance(raw, dict):
        return raw
    return {_REAL_RUN_ACTION_NAMES.get(i, str(i)): p for i, p in enumerate(raw)}


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
    probs = _action_probabilities_as_dict(decision)

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
    """Task sec.4: exactly ONE panel, matching the selected run's
    controller family -- never both, never the "HPA not available"/"PPO
    not available" pair this replaces. Uses `experiment_store.
    _controller_family()` (substring, not exact-match) so this stays
    correct for any future controller id containing "ppo"/"hpa", same as
    Comparison pairing and the `arm` backfill already do."""
    if rec is None:
        return
    family = experiment_store._controller_family(rec.get("controller"))
    if family == "ppo":
        _render_ppo_decision_panel(rec)
    elif family == "hpa":
        _render_hpa_panel(rec)


def render(rec: Optional[Dict[str, Any]], is_active: bool) -> None:
    """Renders ONE run (banner + telemetry + its single decision/HPA
    panel). Called directly by `_render_sample_data_dashboard()` (DEMO
    fixtures, never persisted to `experiment_store`) with an already-
    resolved `rec`; the primary LIVE path goes through `render_primary()`
    below instead, which resolves `rec` itself via the run selector."""
    render_run_banner(rec, is_active)
    render_telemetry(rec)
    render_decision_or_hpa_panel(rec)


def render_primary() -> None:
    """Entry point called by `views/live_details.py` for the primary
    (store-backed) LIVE path -- task sec.4's run selector lives here:
    resolves which run to show (defaulting to the active run, else the
    latest completed one) before rendering it exactly like `render()`
    does for a DEMO fixture.

    Task sec.3: the selected run's banner/telemetry/decision panel are
    rendered inside an `st.fragment(run_every=...)`, applied FUNCTIONALLY
    (same technique as `views/experiment.py`'s live status panel) so the
    interval is ~4s while the SELECTED run is the active one and `None`
    (no periodic reruns) otherwise -- a completed run's data can never
    change, so there is nothing to poll for. The selector widget itself
    stays OUTSIDE this fragment: picking a different run is a normal
    full-page rerun, which recomputes `is_active` (and therefore whether
    to keep polling) for whichever run is now selected."""
    rec, is_active = render_run_selector()
    run_every = 4 if is_active else None
    st.fragment(run_every=run_every)(render)(rec, is_active)
