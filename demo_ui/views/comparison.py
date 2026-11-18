"""pages/comparison.py -- Final comparison page (sec.13-14, 16): looks
like a scientific result, not another monitoring dashboard.

When real LIVE experiment runs exist (`experiment_store`), the page leads
with an actual-data comparison driven by the latest completed run and its
workload-compatible counterpart (same profile/hash/seed/duration) -- never
a fabricated graph for an arm that hasn't actually run that workload. Below
that sits the pre-registered evaluation campaign's own analysis output.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List, Optional

import streamlit as st

from demo_ui import experiment_store
from demo_ui.components.charts import overlay_line_chart
from demo_ui.components.layout import render_topbar
from demo_ui.config_v2 import load_config_v2_safe
from demo_ui.views import comparison_v2

# Task sec.1/2 (this phase): the metrics table used to read keys
# (`mean_replicas`, `max_latency_p95_ms`, etc.) directly off
# `run["summary"]` -- but `status_file_fields_for_store()` in
# `experiment_store.py` never populates any such top-level summary
# field; `summary` only ever carries `samples` (plus workload identity
# fields). So every `summary.get(key)` call was always `None` and the
# table always rendered "--" for every metric, for every run, regardless
# of how much real data the run actually has. Fixed by computing every
# metric directly from `run["summary"]["samples"]` -- the SAME per-
# decision records `experiment_store.extract_series()` already reads
# correctly (post Phase 9H) for the overlay charts above this table --
# instead of a summary-level field that was never written.
#
# Each entry: (label, unit-suffix-or-None, decimals, lower_is_better).
# `lower_is_better=True` rows get the smaller of the two arms' values
# bolded (never colour-filled, per task sec.2); `None` means the row is
# purely descriptive (e.g. workload request rate) and nothing is
# highlighted either way.
_METRIC_ROWS: List[tuple] = [
    ("decisions", "Number of decisions", None, 0, None),
    ("mean_replicas", "Mean replicas", None, 2, True),
    ("max_replicas", "Max replicas", None, 0, True),
    ("replica_minutes", "Replica-minutes", None, 1, True),
    ("scaling_actions", "Scaling actions", None, 0, True),
    ("mean_p95_latency_ms", "Mean P95 latency", "ms", 1, True),
    ("median_p95_latency_ms", "Median P95 latency", "ms", 1, True),
    ("max_p95_latency_ms", "Max P95 latency", "ms", 1, True),
    ("share_p95_over_1s_pct", "Decisions with P95 >= 1s", "%", 1, True),
    ("mean_request_rate_rps", "Mean request rate", "req/s", 2, None),
    ("peak_request_rate_rps", "Peak request rate", "req/s", 2, None),
    ("error_rate_pct", "Error rate", "%", 2, True),
    ("max_pod_memory_mib", "Max per-pod memory", "MiB", 1, True),
    ("pod_restarts", "Pod restarts", None, 0, True),
]


def _run_samples(run: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if run is None:
        return []
    return (run.get("summary") or {}).get("samples") or []


def _field_values(samples: List[Dict[str, Any]], field: str) -> List[float]:
    return [s[field] for s in samples if s.get(field) is not None]


def compute_run_metrics(run: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Computes every Comparison-table metric for one run directly from
    its own persisted per-decision samples (`run["summary"]["samples"]`)
    -- the real `EpisodeSample` schema (`desired_replicas`,
    `p95_latency_ms`, `request_rate_rps`, `error_rate`,
    `max_per_pod_memory_mib`, `pod_restart_counts`, `t_s`/`time_s`), never
    a summary-level field. Returns `None` for a metric only when there is
    genuinely nothing to compute it from (no samples at all, or that
    field is absent from every sample) -- never because of a field-name
    mismatch."""
    samples = _run_samples(run)
    result: Dict[str, Optional[float]] = {key: None for key, *_ in _METRIC_ROWS}
    # "Number of decisions" is always known (0 is a real, meaningful
    # value -- a run that genuinely produced no samples), so it is set
    # even when every OTHER metric below stays `None`/"--" for lack of
    # anything to compute them from.
    result["decisions"] = len(samples)
    if not samples:
        return result

    replicas = _field_values(samples, "desired_replicas")
    if replicas:
        result["mean_replicas"] = statistics.mean(replicas)
        result["max_replicas"] = max(replicas)
        result["scaling_actions"] = sum(
            1 for a, b in zip(replicas, replicas[1:]) if a != b
        )

    times = [s.get("t_s", s.get("time_s")) for s in samples]
    times = [t for t in times if t is not None]
    if replicas and len(times) == len(replicas) and len(replicas) >= 1:
        # Replica-minutes: each decision's replica count held for the
        # interval until the NEXT decision; the final decision's interval
        # is assumed equal to the previous (uniform decision cadence) --
        # there is no further decision to bound it, so this is the most
        # defensible estimate rather than treating it as zero-duration.
        minutes_total = 0.0
        for i in range(len(replicas)):
            if i + 1 < len(times):
                dt_s = times[i + 1] - times[i]
            elif len(times) >= 2:
                dt_s = times[-1] - times[-2]
            else:
                dt_s = 0.0
            minutes_total += replicas[i] * (dt_s / 60.0)
        result["replica_minutes"] = minutes_total

    p95s = _field_values(samples, "p95_latency_ms")
    if p95s:
        result["mean_p95_latency_ms"] = statistics.mean(p95s)
        result["median_p95_latency_ms"] = statistics.median(p95s)
        result["max_p95_latency_ms"] = max(p95s)
        result["share_p95_over_1s_pct"] = 100.0 * sum(1 for v in p95s if v >= 1000.0) / len(p95s)

    rates = _field_values(samples, "request_rate_rps")
    if rates:
        result["mean_request_rate_rps"] = statistics.mean(rates)
        result["peak_request_rate_rps"] = max(rates)

    errors = _field_values(samples, "error_rate")
    if errors:
        result["error_rate_pct"] = 100.0 * statistics.mean(errors)

    mem = _field_values(samples, "max_per_pod_memory_mib")
    if mem:
        result["max_pod_memory_mib"] = max(mem)

    restart_totals = [
        sum(s["pod_restart_counts"].values())
        for s in samples
        if isinstance(s.get("pod_restart_counts"), dict)
    ]
    if restart_totals:
        result["pod_restarts"] = max(restart_totals)

    return result


def _fmt_value(value: Optional[float], unit: Optional[str], decimals: int) -> str:
    if value is None:
        return "—"
    text = f"{value:,.{decimals}f}" if decimals else f"{value:,.0f}"
    return f"{text} {unit}" if unit else text


def _render_metrics_table(hpa_run: Dict[str, Any], ppo_run: Dict[str, Any]) -> None:
    hpa_metrics = compute_run_metrics(hpa_run)
    ppo_metrics = compute_run_metrics(ppo_run)

    row_html = []
    for key, label, unit, decimals, lower_is_better in _METRIC_ROWS:
        hpa_val, ppo_val = hpa_metrics.get(key), ppo_metrics.get(key)
        hpa_text, ppo_text = _fmt_value(hpa_val, unit, decimals), _fmt_value(ppo_val, unit, decimals)
        if lower_is_better and hpa_val is not None and ppo_val is not None and hpa_val != ppo_val:
            if hpa_val < ppo_val:
                hpa_text = f"<b>{hpa_text}</b>"
            else:
                ppo_text = f"<b>{ppo_text}</b>"
        row_html.append(f"<tr><td>{label}</td><td>{hpa_text}</td><td>{ppo_text}</td></tr>")

    st.markdown(
        f"""
        <table class="du-cmp-table" style="margin-top:0.5rem;">
          <tr><th>Metric</th><th>HPA</th><th>PPO</th></tr>
          {"".join(row_html)}
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Demo runs illustrate behaviour; they are not evaluation evidence.")


def _latest_current_system_run():
    """The most recently started completed run made by the current
    system (never a run from a different generation). Task sec.3: an
    `invalid_no_traffic` run (zero/near-zero traffic throughout -- never a
    real measurement) is never chosen as the workload this section leads
    with, even if it is the most recent terminal run -- falls back to the
    next-most-recent genuinely-evaluable run, or `None` if every run so
    far is invalidated (the empty state below, never a fabricated
    comparison)."""
    candidates = [
        r for r in experiment_store.completed_runs()
        if r.get("generation") == "v2" and r.get("status") != experiment_store.INVALID_NO_TRAFFIC
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["started_at"])


def _render_live_comparison() -> bool:
    """Always renders the Live Comparison section -- chart frames/axes are
    shown even with zero runs, per the "dashboard structure always exists,
    data appears only once real" requirement. Returns True if at least one
    real completed run exists (used only to adjust the spacing below)."""
    st.markdown('<div class="du-section-label">Live Comparison</div>', unsafe_allow_html=True)
    latest = _latest_current_system_run()
    has_any = latest is not None

    if not has_any:
        st.caption("No compatible experiment pair yet.")
        c1, c2 = st.columns(2)
        empty_primary, empty_detail = "No comparison data", "Run both arms with the same workload"
        c1.plotly_chart(overlay_line_chart("Replicas", [], [], "HPA", [], [], "PPO", "replicas", empty_message=empty_primary, empty_detail=empty_detail, empty_y_range=(0, 8)), use_container_width=True)
        c2.plotly_chart(overlay_line_chart("P95 Latency", [], [], "HPA", [], [], "PPO", "ms", empty_message=empty_primary, empty_detail=empty_detail, empty_y_range=(0, 500)), use_container_width=True)
        return False

    profile_label = latest["profile_name"].replace("_", " ").title()
    st.markdown(f'<div class="mono" style="font-size:13px;">Comparing workload: <b>{profile_label}</b></div>', unsafe_allow_html=True)

    pair = experiment_store.latest_compatible_pair(latest)
    hpa_run, ppo_run = pair["hpa"], pair["ppo"]

    def _run_label(label: str, rec) -> str:
        if rec is None:
            return f"{label} run: not available"
        ts = time.strftime("%H:%M:%S", time.localtime(rec.get("completed_at") or rec["started_at"]))
        return f"{label} run: {rec['run_id']} ({ts})"

    st.caption(f"{_run_label('HPA', hpa_run)}  ·  {_run_label('PPO', ppo_run)}")

    missing_msg = None
    if hpa_run is None:
        missing_msg = "Matching HPA run not available yet." if ppo_run is not None else "No compatible experiment pair yet."
    elif ppo_run is None:
        missing_msg = "Matching PPO run not available yet."

    if hpa_run is not None or ppo_run is not None:
        available = "HPA result available" if hpa_run is not None else None
        if ppo_run is not None:
            available = f"{available} · PPO result available" if available else "PPO result available"
        line = available or ""
        if missing_msg and missing_msg != "No compatible experiment pair yet.":
            line = f"{line} · {missing_msg}" if line else missing_msg
        if line:
            st.caption(line)

    # Pre-existing field-name mismatch fixed in passing while verifying
    # comparison pairing end-to-end (task sec.7): `EpisodeSample`/
    # `scripts/run_demo_episode_v2.py`'s own per-decision sample schema
    # names this field `desired_replicas` (see `_to_status_sample`), never
    # `replicas_desired` -- the old key name here meant `extract_series()`
    # always returned `([], [])` for the Replicas overlay chart even for a
    # correctly-paired HPA/PPO run, silently empty rather than erroring.
    ts_hr, hr = experiment_store.extract_series(hpa_run, "desired_replicas")
    ts_hl, hl = experiment_store.extract_series(hpa_run, "p95_latency_ms")
    ts_pr, pr = experiment_store.extract_series(ppo_run, "desired_replicas")
    ts_pl, pl = experiment_store.extract_series(ppo_run, "p95_latency_ms")

    c1, c2 = st.columns(2)
    c1.plotly_chart(overlay_line_chart("Replicas", ts_hr, hr, "HPA", ts_pr, pr, "PPO", "replicas", empty_message="No comparison data", empty_detail="Run both arms with the same workload", empty_y_range=(0, 8)), use_container_width=True)
    c2.plotly_chart(overlay_line_chart("P95 Latency", ts_hl, hl, "HPA", ts_pl, pl, "PPO", "ms", empty_message="No comparison data", empty_detail="Run both arms with the same workload", empty_y_range=(0, 500)), use_container_width=True)

    if hpa_run is not None and ppo_run is not None:
        _render_metrics_table(hpa_run, ppo_run)
    return True


def render() -> None:
    render_topbar(mode_label="Live cluster", is_live=True)

    _render_live_comparison()

    st.markdown('<div class="du-section-label" style="margin-top:1rem;">Evaluation (CORR-60/61 pre-registered campaign)</div>', unsafe_allow_html=True)
    cfg, error = load_config_v2_safe()
    if error is not None:
        st.caption(error)
    else:
        comparison_v2.render(cfg.evaluation_campaign_dir)
