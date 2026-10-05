"""components/metric_cards.py -- Primary instrument metric strip for the
live monitor page (sec.7-8): one slim horizontal row with thin separators,
not a grid of floating cards.
"""

from __future__ import annotations

import streamlit as st

from demo_ui.components.layout import render_metric_strip
from demo_ui.models import Controller, RunSummary


def render_summary_cards(run: RunSummary, elapsed_s: float) -> None:
    render_metric_strip([
        ("Controller", run.controller.value),
        ("Traffic", run.traffic_profile.value.capitalize()),
        ("Elapsed", f"{elapsed_s:.0f}s"),
        ("RPS", f"{run.rps[-1]:.1f}" if run.rps else "–"),
        ("CPU", f"{run.cpu_pct[-1]:.0f}%" if run.cpu_pct else "–"),
        ("Replicas", str(run.replicas[-1]) if run.replicas else "–"),
        ("P95", f"{run.latency_p95_ms[-1]:.0f} ms" if run.latency_p95_ms else "–"),
        ("SLA Viol.", str(run.sla_violations)),
    ])

    render_metric_strip([
        ("Avg Latency", f"{run.latency_avg_ms[-1]:.0f} ms" if run.latency_avg_ms else "–"),
        ("P99", f"{run.latency_p99_ms[-1]:.0f} ms" if run.latency_p99_ms else "–"),
        ("Successful", str(run.successful_requests)),
        ("Errors", str(run.errors)),
        ("Timeouts", str(run.timeouts)),
        ("Memory", f"{run.memory_pct[-1]:.0f}%" if run.memory_pct else "–"),
    ])

    if run.controller == Controller.PPO and run.ppo is not None:
        st.markdown('<div class="du-section-label" style="margin-top:0.5rem;">PPO Decision State</div>', unsafe_allow_html=True)
        render_metric_strip([
            ("Latest Action", run.ppo.latest_action.value if run.ppo.latest_action else "–"),
            ("Latest Reward", f"{run.ppo.latest_reward:+.3f}" if run.ppo.latest_reward is not None else "–"),
            ("Observation ts", f"{run.ppo.observation_timestamp:.0f}s" if run.ppo.observation_timestamp is not None else "–"),
        ])

    if run.controller == Controller.HPA and run.hpa is not None:
        st.markdown('<div class="du-section-label" style="margin-top:0.5rem;">HPA Controller State</div>', unsafe_allow_html=True)
        render_metric_strip([
            ("CPU Target", f"{run.hpa.cpu_target_pct:.0f}%"),
            ("Current CPU", f"{run.hpa.current_cpu_pct:.0f}%"),
            ("Desired Replicas", str(run.hpa.desired_replicas)),
        ])
