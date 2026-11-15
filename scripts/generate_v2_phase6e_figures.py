#!/usr/bin/env python3
"""scripts/generate_v2_phase6e_figures.py

V2 Phase 6E: final publication figure set, ONE PLOT PER FILE (supersedes
the composite multi-panel figures from Phase 6D / commit 8f41e6d and the
P0/T1-T4 addition / commit 5df661a -- both preserved in git history).
Styled via scripts/paper_style.py. Read-only against persistent-data/ and
configs/ -- writes only under docs/figures/ and docs/tables/. Reuses
scripts/analyze_v2_evaluation.py's data-loading functions and
scripts/generate_v2_phase6d_figures.py's helper functions (median-cost
repeat selection, failed-session detection, raw-sample loading) rather
than re-deriving them.

Also builds FIGURES_MANIFEST (file, caption, source, takeaway, section)
consumed by docs/figures/FIGURES.md's generation (see the phase's own
report script invocation).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import scripts.analyze_v2_evaluation as av  # noqa: E402
import scripts.paper_style as ps  # noqa: E402
from loadgen.traffic_generator import parse_traffic_profile  # noqa: E402
from scripts.generate_v2_phase6d_figures import (  # noqa: E402
    _failed_retried_session_indices,
    _load_attempt_samples,
    _median_cost_run,
    _runs_for,
)

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
TABLES_DIR = PROJECT_ROOT / "docs" / "tables"
EVAL_FINAL_DIR = PROJECT_ROOT / "persistent-data" / "evaluation_v2" / "eval_final"
STRESS_DIR = PROJECT_ROOT / "persistent-data" / "evaluation_v2" / "stress_explore"
SESSIONS_DIR = PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final_min2" / "sessions"
UPDATE_METRICS_PATH = PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final_min2" / "ppo_update_metrics.jsonl"
PREREGISTRATION_PATH = PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final_min2" / "EVALUATION_PREREGISTRATION.json"
STRESS_PLAN_PATH = PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final_min2" / "EXPLORATORY_STRESS_PLAN.json"
RL_V2_PATH = PROJECT_ROOT / "configs" / "rl_v2.yaml"
REWARD_V2_PATH = PROJECT_ROOT / "configs" / "reward_v2.yaml"

EVAL_PROFILE_DIR = PROJECT_ROOT / "configs" / "traffic_profiles" / "v2_eval"
STRESS_PROFILE_DIR = PROJECT_ROOT / "configs" / "traffic_profiles" / "v2_stress"
EVAL_PROFILE_NAMES = ("eval_cpu_step", "eval_memory_ramp", "eval_mixed_burst")
STRESS_PROFILE_NAMES = ("stress_cpu_burst", "stress_memory_ramp", "stress_mixed_burst")

SLA_MS = 1000.0
PPO_ACTION_NAMES = ["SCALE_DOWN_ONE", "HOLD", "SCALE_UP_ONE"]
ACTION_COLORS = {"SCALE_DOWN_ONE": "#56B4E9", "HOLD": "#CCCCCC", "SCALE_UP_ONE": ps.ARM_COLORS["ppo_v2"]}

FIGURES_MANIFEST: List[Dict[str, str]] = []


def _record(file_stub: str, caption: str, source: str, takeaway: str, section: str) -> None:
    FIGURES_MANIFEST.append({
        "file": f"{file_stub}.png / {file_stub}.pdf", "caption": caption, "source": source,
        "takeaway": takeaway, "section": section,
    })


# ---------------------------------------------------------------------------
# Workloads (6 files): RPS vs. time, one profile per file
# ---------------------------------------------------------------------------


def plot_workload(profile_dir: Path, name: str, label_prefix: str) -> None:
    profile = parse_traffic_profile(profile_dir / f"{name}.yaml")
    total = sum(ph.duration_seconds for ph in profile.phases)
    t = np.arange(0.0, total, 1.0)
    rps = np.array([profile.get_phase_and_target_rps(float(ti))[0] for ti in t])

    fig, ax = ps.new_figure(ps.SINGLE_COL_WIDTH_IN, 2.2)
    ax.plot(t, rps, linewidth=1.2, color="#333333")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Request rate (RPS)")
    ps.style_axes(ax)
    stub = f"workloads_{name}"
    ps.save_figure(fig, FIGURES_DIR / stub)
    campaign = "pre-registered eval_final" if label_prefix == "eval" else "exploratory stress_explore (1 repeat, CORR-65)"
    _record(
        stub, f"Request rate (RPS) vs. time, {name} ({profile.resource_mode} resource mode, {campaign}).",
        f"configs/traffic_profiles/{'v2_eval' if label_prefix == 'eval' else 'v2_stress'}/{name}.yaml",
        f"Peak {rps.max():.0f} RPS, {profile.resource_mode} resource mode.", "Design",
    )


# ---------------------------------------------------------------------------
# Training (8 files)
# ---------------------------------------------------------------------------


def _training_line_figure(sessions, values, failed, ylabel, stub, caption, takeaway, color=None) -> None:
    fig, ax = ps.new_figure()
    ax.plot(sessions, values, color=color or ps.ARM_COLORS["ppo_v2"], linewidth=1.2)
    for idx in failed:
        ax.axvline(idx, color=ps.FAILED_SESSION_COLOR, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Training session index")
    ax.set_ylabel(ylabel)
    ps.style_axes(ax)
    if failed:
        ax.legend(
            handles=[plt.Line2D([0], [0], color=ps.FAILED_SESSION_COLOR, linestyle="--", linewidth=0.8, label="Failed/retried session")],
            frameon=False, loc="best", fontsize=6.5,
        )
    ps.save_figure(fig, FIGURES_DIR / stub)
    _record(stub, caption, "persistent-data/rl_training_v2/final_min2/ppo_update_metrics.jsonl, sessions/*/step_telemetry.jsonl", takeaway, "Method")


def plot_training_figures() -> None:
    curve = av.analyze_training_curve(SESSIONS_DIR, UPDATE_METRICS_PATH)
    failed = _failed_retried_session_indices()
    sessions = [c["session_index"] for c in curve]

    _training_line_figure(
        sessions, [c["entropy_loss"] for c in curve], failed, "Entropy loss", "training_entropy",
        f"Entropy loss per training session, 48 sessions; dashed lines mark sessions {failed} (failed, retried).",
        "Entropy loss falls steadily, indicating the policy becomes progressively less random over training.",
    )
    _training_line_figure(
        sessions, [c["explained_variance"] for c in curve], failed, "Explained variance", "training_explained_variance",
        f"Explained variance per training session, 48 sessions; dashed lines mark sessions {failed}.",
        "Explained variance rises from ~0 to ~0.7-0.8, indicating the value function increasingly predicts returns.",
    )
    _training_line_figure(
        sessions, [c["value_loss"] for c in curve], failed, "Value loss", "training_value_loss",
        f"Value loss per training session, 48 sessions; dashed lines mark sessions {failed}.",
        "Value loss trends downward as the value function fits better over the campaign.",
    )
    _training_line_figure(
        sessions, [c["approx_kl"] for c in curve], failed, "Approx. KL divergence", "training_approx_kl",
        f"Approximate KL divergence per PPO update per training session, 48 sessions; dashed lines mark sessions {failed}.",
        "Approx. KL stays within PPO's typical stable-update range throughout training, without runaway policy shifts.",
    )
    _training_line_figure(
        sessions, [c["mean_step_reward"] for c in curve], failed, "Mean step reward", "training_mean_step_reward",
        f"Mean per-step reward per training session, 48 sessions; dashed lines mark sessions {failed}.",
        "Mean step reward improves over training, consistent with the policy learning to avoid penalized states.",
    )

    # reward-component shares (curve[i]["reward_component_means"] is a dict
    # of component_name -> mean value for that session; present for every
    # completed session, per summarize_session_telemetry()).
    component_names = sorted(curve[0]["reward_component_means"].keys()) if curve and curve[0].get("reward_component_means") else []
    fig, ax = ps.new_figure(ps.DOUBLE_COL_WIDTH_IN, 2.8)
    if component_names:
        stacks = []
        for comp in component_names:
            abs_vals = [abs((c.get("reward_component_means") or {}).get(comp) or 0.0) for c in curve]
            stacks.append(abs_vals)
        totals = [sum(vals) or 1.0 for vals in zip(*stacks)]
        shares = [[v / t for v, t in zip(vals, totals)] for vals in stacks]
        colors = plt.get_cmap("tab10").colors
        ax.stackplot(sessions, *shares, labels=component_names, colors=colors[: len(shares)])
        ax.set_ylim(0, 1)
        caption = f"Relative share of each reward component (by |mean component value|) per training session, 48 sessions; dashed lines mark sessions {failed}."
        takeaway = "Shows which reward terms dominate the policy's training signal across the campaign."
    else:
        ax.text(0.5, 0.5, "Per-component reward not logged in step_telemetry.jsonl", ha="center", va="center", fontsize=7, transform=ax.transAxes)
        caption = "Reward-component shares per training session -- NOT AVAILABLE: step_telemetry.jsonl does not log a per-component reward breakdown, only the scalar total reward; this figure honestly reports that gap rather than fabricating a breakdown."
        takeaway = "Per-component reward attribution is not recorded by the training pipeline; only the aggregate reward is available (see training_mean_step_reward)."
    for idx in failed:
        ax.axvline(idx, color=ps.FAILED_SESSION_COLOR, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Training session index")
    ax.set_ylabel("Reward component share")
    ps.style_axes(ax)
    if component_names:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, frameon=False, fontsize=6)
    ps.save_figure(fig, FIGURES_DIR / "training_reward_components")
    _record("training_reward_components", caption, "persistent-data/rl_training_v2/final_min2/sessions/*/step_telemetry.jsonl", takeaway, "Method")

    # action mix (stacked area)
    down, hold, up = [], [], []
    for c in curve:
        counts = c["action_mix_counts"]
        total = sum(counts.values()) or 1
        down.append(counts.get("SCALE_DOWN", 0) / total)
        hold.append(counts.get("HOLD", 0) / total)
        up.append(counts.get("SCALE_UP", 0) / total)
    fig, ax = ps.new_figure(ps.SINGLE_COL_WIDTH_IN, 2.4)
    ax.stackplot(sessions, down, hold, up, labels=["SCALE_DOWN", "HOLD", "SCALE_UP"], colors=["#56B4E9", "#CCCCCC", ps.ARM_COLORS["ppo_v2"]])
    for idx in failed:
        ax.axvline(idx, color=ps.FAILED_SESSION_COLOR, linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Training session index")
    ax.set_ylabel("Action share")
    ps.style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False, fontsize=6.5)
    ps.save_figure(fig, FIGURES_DIR / "training_action_mix")
    _record(
        "training_action_mix", f"Fraction of SCALE_DOWN/HOLD/SCALE_UP actions per training session, 48 sessions; dashed lines mark sessions {failed}.",
        "persistent-data/rl_training_v2/final_min2/sessions/*/step_telemetry.jsonl",
        "HOLD share rises over training as the policy learns to avoid unnecessary scaling actions.", "Method",
    )

    # mean replicas
    mean_replicas = [c["mean_replicas"] for c in curve]
    _training_line_figure(
        sessions, mean_replicas, failed, "Mean replicas", "training_mean_replicas",
        f"Mean replica count per training session, 48 sessions; dashed lines mark sessions {failed}.",
        "Mean replicas drifts toward the lower end of the 2-8 range as the policy learns to avoid over-provisioning.",
        color=ps.ARM_COLORS["hpa_v2_cpu_memory"],
    )


# ---------------------------------------------------------------------------
# Evaluation / stress shared helpers
# ---------------------------------------------------------------------------


def _bar_figure(aggregated, metric_key: str, ylabel: str, stub: str, caption: str, source: str, takeaway: str, section: str, exploratory: bool, value_fmt="{:.4f}") -> None:
    profiles = sorted({p for p, _ in aggregated})
    fig, ax = ps.new_figure(ps.DOUBLE_COL_WIDTH_IN, 2.8)
    width = 0.25
    max_height = 0.0
    for i, arm_id in enumerate(ps.ARM_IDS):
        means = [aggregated.get((p, arm_id), {}).get(metric_key, {}).get("mean") or 0.0 for p in profiles]
        sds = [aggregated.get((p, arm_id), {}).get(metric_key, {}).get("sd") or 0.0 for p in profiles] if not exploratory else None
        x = [j + i * width for j in range(len(profiles))]
        ax.bar(x, means, width=width, yerr=sds, capsize=3, label=ps.arm_label(arm_id), color=ps.arm_color(arm_id), hatch=ps.arm_hatch(arm_id), edgecolor="white", linewidth=0.5)
        for j, mean in enumerate(means):
            sd = (sds[j] or 0.0) if sds else 0.0
            max_height = max(max_height, mean + sd)
    ax.set_ylim(0, max_height * 1.25 if max_height else 1.0)
    ax.set_xticks([j + width for j in range(len(profiles))])
    ax.set_xticklabels(profiles, rotation=15, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ps.style_axes(ax)
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, facecolor=ps.arm_color(a), hatch=ps.arm_hatch(a), edgecolor="white", label=ps.arm_label(a)) for a in ps.ARM_IDS],
        loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=1, frameon=False, fontsize=6.5,
    )
    ps.save_figure(fig, FIGURES_DIR / stub)
    _record(stub, caption, source, takeaway, section)


def _boxplot_figure(series_by_name, profile: str, stub: str, caption: str, source: str, takeaway: str, section: str) -> None:
    fig, ax = ps.new_figure(ps.SINGLE_COL_WIDTH_IN, 2.6)
    data, colors, labels = [], [], []
    for arm_id in ps.ARM_IDS:
        values = []
        for s in series_by_name.values():
            if s.profile != profile or s.arm_id != arm_id:
                continue
            values.extend(sample.p95_latency_ms for sample in s.samples if not sample.invalid and sample.p95_latency_ms is not None)
        if values:
            data.append(values)
            colors.append(ps.arm_color(arm_id))
            labels.append(ps.arm_label(arm_id))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=True, boxprops={"linewidth": 0.6}, whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6}, flierprops={"markersize": 3, "markeredgewidth": 0.6})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.0)
    ax.axhline(SLA_MS, color=ps.SLA_LINE_COLOR, linestyle="--", linewidth=0.8)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([_short_label(lbl) for lbl in labels], rotation=15, ha="right", fontsize=6.5)
    ax.set_ylabel("p95 latency (ms)")
    # A single extreme outlier (e.g. ~3600ms against a ~200-900ms IQR) on a
    # linear axis compresses every box to an unreadably thin line -- symlog
    # keeps the box bodies/colors visible while still showing the outlier,
    # without the log-scale's "undefined at 0" problem (linthresh=200
    # covers the typical sub-SLA latency range linearly).
    all_values = [v for series in data for v in series]
    if all_values and max(all_values) > 3 * SLA_MS:
        ax.set_yscale("symlog", linthresh=200)
    ps.style_axes(ax)
    ps.save_figure(fig, FIGURES_DIR / stub)
    _record(stub, caption, source, takeaway, section)


def _short_label(label: str) -> str:
    return {
        ps.ARM_LABELS["ppo_v2"]: "PPO", ps.ARM_LABELS["hpa_v2_cpu_memory"]: "HPA CPU+mem", ps.ARM_LABELS["hpa_v2_cpu_only"]: "HPA CPU-only",
    }.get(label, label)


def _replicas_over_time_figure(campaign_dir: Path, profile: str, run_picker, stub: str, source: str, takeaway: str, section: str, exploratory: bool) -> None:
    fig, ax = ps.new_figure(ps.SINGLE_COL_WIDTH_IN, 2.4)
    ax_rps = ax.twinx()
    for i, arm_id in enumerate(ps.ARM_IDS):
        run_dir = run_picker(campaign_dir, profile, arm_id)
        samples = _load_attempt_samples(run_dir)
        t = [s["t_s"] / 60.0 for s in samples]
        replicas = [s["desired_replicas"] for s in samples]
        if i == 0:
            rps = [s.get("request_rate_rps") or 0.0 for s in samples]
            ax_rps.fill_between(t, rps, color="#dddddd", alpha=0.6, linewidth=0, zorder=0)
            ax_rps.set_ylabel("Request rate (RPS)", color="#888888", fontsize=7)
            ax_rps.tick_params(axis="y", labelsize=6, colors="#888888")
            ax_rps.spines["top"].set_visible(False)
        ax.plot(t, replicas, color=ps.arm_color(arm_id), marker=ps.arm_marker(arm_id), markersize=3, markevery=4, linewidth=1.1, label=ps.arm_label(arm_id), zorder=3)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Replicas")
    ps.style_axes(ax)
    # A figure-level legend ABOVE the axes, never an in-axes legend here --
    # found live: an in-axes "upper left" legend sat directly on top of the
    # PPO line/markers in several profiles (self-review fix).
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=1, frameon=False, fontsize=6)
    ps.save_figure(fig, FIGURES_DIR / stub)
    repeat_note = "median-cost repeat per arm" if not exploratory else "the only repeat per arm"
    _record(stub, f"Replica count over time, {profile}, all 3 arms ({repeat_note}), with request rate (RPS) as a light background.", source, takeaway, section)


def _p95_over_time_figure(campaign_dir: Path, profile: str, run_picker, stub: str, source: str, takeaway: str, section: str, exploratory: bool) -> None:
    fig, ax = ps.new_figure(ps.SINGLE_COL_WIDTH_IN, 2.4)
    for arm_id in ps.ARM_IDS:
        run_dir = run_picker(campaign_dir, profile, arm_id)
        samples = _load_attempt_samples(run_dir)
        t = [s["t_s"] / 60.0 for s in samples]
        p95 = [s["p95_latency_ms"] if not (s["p95_latency_ms"] == 0.0 and (s.get("request_rate_rps") or 0.0) == 0.0) else float("nan") for s in samples]
        ax.plot(t, p95, color=ps.arm_color(arm_id), marker=ps.arm_marker(arm_id), markersize=3, markevery=4, linewidth=1.1, label=ps.arm_label(arm_id))
    ax.axhline(SLA_MS, color=ps.SLA_LINE_COLOR, linestyle="--", linewidth=0.8, label="1000 ms SLA")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("p95 latency (ms)")
    ps.style_axes(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=1, frameon=False, fontsize=6)
    ps.save_figure(fig, FIGURES_DIR / stub)
    repeat_note = "median-cost repeat per arm" if not exploratory else "the only repeat per arm"
    note = " A p95 of exactly 975ms is a histogram-bucket artefact (true value somewhere in 0.5-1.0s)." if profile in ("eval_memory_ramp", "eval_mixed_burst", "stress_memory_ramp", "stress_mixed_burst") else ""
    _record(stub, f"p95 latency over time, {profile}, all 3 arms ({repeat_note}), with the 1000ms SLA line.{note}", source, takeaway, section)


def _single_run_picker(campaign_dir, profile, arm_id):
    return _runs_for(campaign_dir, profile, arm_id)[0]


# ---------------------------------------------------------------------------
# Evaluation (11 files)
# ---------------------------------------------------------------------------


def plot_evaluation_figures() -> None:
    run_metrics = av.analyze_evaluation_campaign(EVAL_FINAL_DIR)
    aggregated = av.aggregate_per_profile_arm(run_metrics)
    series_by_name: Dict[str, av.RunSeries] = {}
    for run_dir in sorted((EVAL_FINAL_DIR / "runs").iterdir()):
        series_path = run_dir / "normalized_series.json"
        if series_path.exists():
            series_by_name[run_dir.name] = av.normalize_run_series_from_dict(json.load(open(series_path, encoding="utf-8")))

    _bar_figure(
        aggregated, "replica_minutes", "Replica-minutes", "evaluation_replica_minutes",
        "Replica-minutes (cost proxy), mean +/- sd over 3 repeats, per profile x arm, eval_final.",
        "persistent-data/evaluation_v2/eval_final/", "No arm is cheapest across all 3 profiles -- cost ranking is profile-dependent.", "Results", exploratory=False,
    )
    _bar_figure(
        aggregated, "sla_violation_rate_1000ms", "SLA-violation rate @1000ms [primary]", "evaluation_sla_1000ms",
        "SLA-violation rate at the 1000ms threshold (primary SLA metric), mean +/- sd over 3 repeats, per profile x arm, eval_final.",
        "persistent-data/evaluation_v2/eval_final/", "The primary SLA metric sits at or near its floor (~0) for every arm at eval_final's light load levels.", "Results", exploratory=False,
    )
    _bar_figure(
        aggregated, "sla_violation_rate_300ms_reference", "SLA-violation rate @300ms [reference-only]", "evaluation_sla_300ms",
        "SLA-violation rate at the 300ms threshold (reference-only, NOT the primary SLA definition), mean +/- sd over 3 repeats, per profile x arm, eval_final.",
        "persistent-data/evaluation_v2/eval_final/", "At the stricter 300ms reference threshold, violation rates are substantial for every arm -- shown for historical comparability only.", "Results", exploratory=False,
    )
    _boxplot_figure(
        series_by_name, "eval_cpu_step", "evaluation_p95_boxplot_eval_cpu_step",
        "Box plot of per-decision p95 latency, pooled across 3 repeats, eval_cpu_step, eval_final. Dashed line = 1000ms SLA.",
        "persistent-data/evaluation_v2/eval_final/runs/eval_cpu_step*/normalized_series.json",
        "Latency distributions sit mostly below the SLA line at eval_final's light load levels.", "Results",
    )

    for profile in EVAL_PROFILE_NAMES:
        _replicas_over_time_figure(
            EVAL_FINAL_DIR, profile, _median_cost_run, f"evaluation_replicas_over_time_{profile}",
            f"persistent-data/evaluation_v2/eval_final/runs/{profile}*/normalized_series.json, attempt.json",
            "All 3 arms track the same request-rate shape but differ in how aggressively they scale.", "Results", exploratory=False,
        )
        _p95_over_time_figure(
            EVAL_FINAL_DIR, profile, _median_cost_run, f"evaluation_p95_over_time_{profile}",
            f"persistent-data/evaluation_v2/eval_final/runs/{profile}*/normalized_series.json",
            "p95 latency stays mostly below the SLA line across arms at eval_final's light load levels.", "Results", exploratory=False,
        )

    # memory over time, eval_memory_ramp
    fig, ax = ps.new_figure(ps.SINGLE_COL_WIDTH_IN, 2.4)
    for arm_id in ps.ARM_IDS:
        run_dir = _median_cost_run(EVAL_FINAL_DIR, "eval_memory_ramp", arm_id)
        samples = _load_attempt_samples(run_dir)
        t = [s["t_s"] / 60.0 for s in samples]
        mem = [s.get("max_per_pod_memory_mib") for s in samples]
        ax.plot(t, mem, color=ps.arm_color(arm_id), marker=ps.arm_marker(arm_id), markersize=3, markevery=4, linewidth=1.1, label=ps.arm_label(arm_id))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Max per-pod memory (MiB)")
    ps.style_axes(ax)
    ax.legend(loc="best", fontsize=5.5, frameon=False)
    ps.save_figure(fig, FIGURES_DIR / "evaluation_memory_over_time")
    _record(
        "evaluation_memory_over_time", "Max per-pod memory (MiB) over time on eval_memory_ramp, all 3 arms, eval_final (median-cost repeat per arm).",
        "persistent-data/evaluation_v2/eval_final/runs/eval_memory_ramp*/attempt.json",
        "Per-pod memory stays well below the 230Mi safety threshold across arms on this profile.", "Results",
    )


# ---------------------------------------------------------------------------
# Stress (10 files)
# ---------------------------------------------------------------------------


def plot_stress_figures() -> None:
    run_metrics = av.analyze_evaluation_campaign(STRESS_DIR)
    aggregated = av.aggregate_per_profile_arm(run_metrics)
    series_by_name: Dict[str, av.RunSeries] = {}
    for run_dir in sorted((STRESS_DIR / "runs").iterdir()):
        series_path = run_dir / "normalized_series.json"
        if series_path.exists():
            series_by_name[run_dir.name] = av.normalize_run_series_from_dict(json.load(open(series_path, encoding="utf-8")))

    _bar_figure(
        aggregated, "replica_minutes", "Replica-minutes (1 repeat, no sd)", "stress_replica_minutes",
        "Replica-minutes (cost proxy), 1 repeat per cell (no error bars), per profile x arm, stress_explore (exploratory, CORR-65).",
        "persistent-data/evaluation_v2/stress_explore/", "Cost separates much more widely at training-level peaks than in eval_final.", "Results", exploratory=True,
    )
    _bar_figure(
        aggregated, "sla_violation_rate_1000ms", "SLA-violation rate @1000ms [primary]", "stress_sla_1000ms",
        "SLA-violation rate at the 1000ms threshold, 1 repeat per cell, per profile x arm, stress_explore (exploratory, 1 repeat, CORR-65).",
        "persistent-data/evaluation_v2/stress_explore/", "At training-level peaks the primary SLA metric becomes clearly nonzero and differentiates arms, unlike eval_final.", "Results", exploratory=True,
    )
    _bar_figure(
        aggregated, "sla_violation_rate_300ms_reference", "SLA-violation rate @300ms [reference-only]", "stress_sla_300ms",
        "SLA-violation rate at the 300ms reference threshold, 1 repeat per cell, per profile x arm, stress_explore (exploratory, 1 repeat, CORR-65).",
        "persistent-data/evaluation_v2/stress_explore/", "Most arms saturate near 100% violation at the stricter reference threshold under stress load.", "Results", exploratory=True,
    )
    _boxplot_figure(
        series_by_name, "stress_cpu_burst", "stress_p95_boxplot_stress_cpu_burst",
        "Box plot of per-decision p95 latency, 1 repeat, stress_cpu_burst, stress_explore (exploratory, 1 repeat, CORR-65). Dashed line = 1000ms SLA. A p95 of exactly 975ms is a histogram-bucket artefact (true value somewhere in 0.5-1.0s).",
        "persistent-data/evaluation_v2/stress_explore/runs/stress_cpu_burst*/normalized_series.json",
        "Latency distributions widen and shift above the SLA line at stress-test peak loads.", "Results",
    )

    for profile in STRESS_PROFILE_NAMES:
        _replicas_over_time_figure(
            STRESS_DIR, profile, _single_run_picker, f"stress_replicas_over_time_{profile}",
            f"persistent-data/evaluation_v2/stress_explore/runs/{profile}*/normalized_series.json, attempt.json (exploratory, 1 repeat, CORR-65)",
            "On stress_cpu_burst, HPA arms repeatedly hit the 8-pod ceiling while PPO caps at 5 replicas, consistent with PPO's lower cost." if profile == "stress_cpu_burst" else
            "Scaling response differs between arms at training-level peak load.", "Results", exploratory=True,
        )
        _p95_over_time_figure(
            STRESS_DIR, profile, _single_run_picker, f"stress_p95_over_time_{profile}",
            f"persistent-data/evaluation_v2/stress_explore/runs/{profile}*/normalized_series.json (exploratory, 1 repeat, CORR-65)",
            "p95 latency crosses the SLA line for sustained periods at training-level peak load." if profile == "stress_cpu_burst" else
            "p95 latency stays mostly near or below the SLA line on this profile even at stress-level peaks.", "Results", exploratory=True,
        )


# ---------------------------------------------------------------------------
# Discussion (3 files)
# ---------------------------------------------------------------------------


def plot_discussion_figures() -> None:
    eval_metrics = av.analyze_evaluation_campaign(EVAL_FINAL_DIR)
    stress_metrics = av.analyze_evaluation_campaign(STRESS_DIR)

    fig, ax = ps.new_figure(ps.SINGLE_COL_WIDTH_IN, 2.8)
    for m in eval_metrics:
        ax.scatter(m["replica_minutes"], m["mean_p95_latency_ms"], color=ps.arm_color(m["arm_id"]), marker=ps.arm_marker(m["arm_id"]), s=20, facecolors=ps.arm_color(m["arm_id"]), edgecolors="none", zorder=3)
    for m in stress_metrics:
        ax.scatter(m["replica_minutes"], m["mean_p95_latency_ms"], marker=ps.arm_marker(m["arm_id"]), s=26, facecolors="none", edgecolors=ps.arm_color(m["arm_id"]), linewidths=1.1, zorder=3)
    ax.set_xlabel("Replica-minutes")
    ax.set_ylabel("Mean p95 latency (ms)")
    ps.style_axes(ax)
    handles = [plt.Line2D([0], [0], marker=ps.arm_marker(a), color=ps.arm_color(a), linestyle="", markersize=5, label=_short_label(ps.arm_label(a))) for a in ps.ARM_IDS]
    handles.append(plt.Line2D([0], [0], marker="o", color="#333333", linestyle="", markerfacecolor="#333333", markersize=4, label="eval_final (filled)"))
    handles.append(plt.Line2D([0], [0], marker="o", color="#333333", linestyle="", markerfacecolor="none", markersize=4, label="stress_explore (hollow)"))
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=5.5)
    ps.save_figure(fig, FIGURES_DIR / "discussion_cost_vs_latency")
    _record(
        "discussion_cost_vs_latency", "Replica-minutes (x) vs. mean p95 latency (y), one point per run; eval_final filled, stress_explore hollow.",
        "persistent-data/evaluation_v2/{eval_final,stress_explore}/runs/*/normalized_series.json",
        "Stress-test points sit well above/right of the eval_final cluster -- cost and latency both rise at training-level peak load.", "Discussion",
    )

    # PPO policy map (dominant action per bin, categorical)
    cpu_vals, mem_vals, actions = [], [], []
    tie_rule_counts: Dict[str, int] = {}
    near_tie_count = 0
    total_decisions = 0
    action_counts = {name: 0 for name in PPO_ACTION_NAMES}
    for run_dir in sorted(EVAL_FINAL_DIR.glob("runs/*ppo_v2*")):
        attempt_path = run_dir / "attempt.json"
        if not attempt_path.exists():
            continue
        data = json.load(open(attempt_path, encoding="utf-8"))
        if data.get("status") != "VALID":
            continue
        for s in data["result"]["samples"]:
            pd = s.get("ppo_decision")
            if not pd:
                continue
            total_decisions += 1
            action_counts[pd["effective_action_name"]] = action_counts.get(pd["effective_action_name"], 0) + 1
            rule = pd.get("tie_rule_applied", "unknown")
            tie_rule_counts[rule] = tie_rule_counts.get(rule, 0) + 1
            if pd.get("near_tie"):
                near_tie_count += 1
            raw = s.get("raw_observation") or {}
            cpu, mem = raw.get("cpu_per_replica"), raw.get("memory_per_replica")
            if cpu is not None and mem is not None:
                cpu_vals.append(cpu)
                mem_vals.append(mem)
                actions.append(pd["effective_action_name"])

    cpu_arr, mem_arr, action_arr = np.array(cpu_vals), np.array(mem_vals), np.array(actions)
    n_bins = 6
    cpu_edges = np.linspace(cpu_arr.min(), cpu_arr.max(), n_bins + 1)
    mem_edges = np.linspace(mem_arr.min(), mem_arr.max(), n_bins + 1)
    cpu_bin = np.clip(np.digitize(cpu_arr, cpu_edges) - 1, 0, n_bins - 1)
    mem_bin = np.clip(np.digitize(mem_arr, mem_edges) - 1, 0, n_bins - 1)
    action_to_code = {"SCALE_DOWN_ONE": 0, "HOLD": 1, "SCALE_UP_ONE": 2}
    grid = np.full((n_bins, n_bins), np.nan)
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (cpu_bin == i) & (mem_bin == j)
            if mask.sum() == 0:
                continue
            counts = {a: (action_arr[mask] == a).sum() for a in action_to_code}
            dominant = max(counts, key=counts.get)
            grid[j, i] = action_to_code[dominant]

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([ACTION_COLORS["SCALE_DOWN_ONE"], ACTION_COLORS["HOLD"], ACTION_COLORS["SCALE_UP_ONE"]])
    fig, ax = ps.new_figure(ps.SINGLE_COL_WIDTH_IN, 2.8)
    im = ax.imshow(grid, origin="lower", cmap=cmap, aspect="auto", vmin=-0.5, vmax=2.5)
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([f"{e:.2f}" for e in cpu_edges[:-1]], rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels([f"{e:.2f}" for e in mem_edges[:-1]], fontsize=6)
    ax.set_xlabel("cpu_per_replica (cores)")
    ax.set_ylabel("memory_per_replica (fraction of limit)")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.85)
    cbar.ax.set_yticklabels(["SCALE_DOWN", "HOLD", "SCALE_UP"], fontsize=6)
    ps.save_figure(fig, FIGURES_DIR / "discussion_ppo_policy_map")
    up_frac = action_counts.get("SCALE_UP_ONE", 0) / total_decisions if total_decisions else 0.0
    _record(
        "discussion_ppo_policy_map", "Dominant action (SCALE_DOWN/HOLD/SCALE_UP) per (cpu_per_replica, memory_per_replica) bin, eval_final ppo_v2 decisions.",
        "persistent-data/evaluation_v2/eval_final/runs/*ppo_v2*/attempt.json",
        f"HOLD dominates almost every bin; only {action_counts.get('SCALE_UP_ONE', 0)}/{total_decisions} ({up_frac:.1%}) decisions are SCALE_UP, spread across a wide CPU range rather than concentrated at high CPU.", "Discussion",
    )

    # tie-rule frequency
    fig, ax = ps.new_figure()
    rule_labels = {"R1": "R1: no tie", "R3": "R3: HOLD/UP\n(stress-dep.)", "R4": "R4: DOWN\n-> HOLD", "R5": "R5: DOWN/HOLD\n-> HOLD", "R6": "R6: unknown\n-> HOLD"}
    rules_sorted = sorted(tie_rule_counts.keys())
    counts = [tie_rule_counts[r] for r in rules_sorted]
    labels = [rule_labels.get(r, r) for r in rules_sorted]
    bar_colors = ["#CCCCCC" if r == "R1" else ps.ARM_COLORS["ppo_v2"] for r in rules_sorted]
    ax.bar(labels, counts, color=bar_colors)
    ax.set_ylabel("Number of decisions")
    ax.tick_params(axis="x", labelsize=6)
    ps.style_axes(ax)
    ps.save_figure(fig, FIGURES_DIR / "discussion_tie_rule_frequency")
    tie_frac = near_tie_count / total_decisions if total_decisions else 0.0
    _record(
        "discussion_tie_rule_frequency", f"How often each CORR-58 tie-break rule (R1 = no tie; R3-R6 = an actual tie override) fired across eval_final ppo_v2 decisions. An actual tie-break fired in {near_tie_count}/{total_decisions} decisions ({tie_frac:.2%}).",
        "persistent-data/evaluation_v2/eval_final/runs/*ppo_v2*/attempt.json (ppo_decision.tie_rule_applied / near_tie fields)",
        f"The safety-aware tie-break rule rarely intervenes ({tie_frac:.2%} of decisions) -- R1 (no tie) accounts for almost all decisions.", "Discussion",
    )


# ---------------------------------------------------------------------------
# Diagrams (regenerate SVG+PDF+PNG from existing .dot sources)
# ---------------------------------------------------------------------------


def regenerate_diagrams() -> None:
    import subprocess

    for stub in ("diagram_system_architecture", "diagram_rl_control_loop", "diagram_experiment_pipeline", "diagram_tie_break_flow"):
        dot_path = FIGURES_DIR / f"{stub}.dot"
        for ext, extra in (("svg", []), ("pdf", []), ("png", ["-Gdpi=300"])):
            out_path = FIGURES_DIR / f"{stub}.{ext}"
            subprocess.run(["dot", f"-T{ext}", *extra, str(dot_path), "-o", str(out_path)], check=True)
        print(f"regenerated {stub}")


# ---------------------------------------------------------------------------
# Tables (regenerated fresh, same filenames as before)
# ---------------------------------------------------------------------------


def _write_table(rows: List[Dict[str, Any]], columns: List[str], out_path_no_ext: Path) -> None:
    out_path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out_path_no_ext}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    md_lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        md_lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    Path(f"{out_path_no_ext}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    def _tex_escape(s: str) -> str:
        return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    tex_lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline", " & ".join(_tex_escape(c) for c in columns) + r" \\", r"\hline"]
    for row in rows:
        tex_lines.append(" & ".join(_tex_escape(row.get(c, "")) for c in columns) + r" \\")
    tex_lines += [r"\hline", r"\end{tabular}"]
    Path(f"{out_path_no_ext}.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def generate_t1_observation_features() -> None:
    rl_v2 = yaml.safe_load(open(RL_V2_PATH, encoding="utf-8"))
    entries = rl_v2["observation_v2"]
    assert len(entries) == 11, f"expected 11 observation features, found {len(entries)}"
    rows = [{
        "index": e["index"], "name": e["name"], "meaning": e["description"],
        "raw_range": f"[{e['raw_min']}, {e['raw_max']}]", "normalized_range": f"[{e['min']}, {e['max']}]", "temporal": e["temporal"],
    } for e in entries]
    _write_table(rows, ["index", "name", "meaning", "raw_range", "normalized_range", "temporal"], TABLES_DIR / "T1_observation_features")


def generate_t2_reward_components() -> None:
    reward_v2 = yaml.safe_load(open(REWARD_V2_PATH, encoding="utf-8"))
    weights = reward_v2["weights"]
    meanings = {
        "w_reliability": "Penalizes the live 5xx error rate (clipped to [0,1]).",
        "w_sla": "Penalizes squared excess p95 latency above the SLA target (clipped to [0,1]).",
        "w_pressure": "Penalizes the worse of CPU/memory 'unsafe pressure' (distance toward the pod's hard resource limit).",
        "w_waste": "Penalizes over-provisioning (replicas > floor, low CPU+memory, SLA/error already healthy).",
        "w_action": "Penalizes taking any non-HOLD action (discourages unnecessary churn).",
        "w_reversal": "Penalizes reversing a scaling direction within the reversal window (discourages oscillation).",
        "w_boundary": "Penalizes an action clamped at the replica floor/ceiling (discourages pushing against the bound).",
    }
    thresholds = {
        "w_reliability": "n/a",
        "w_sla": f"p95_target_ms={reward_v2['sla']['p95_target_ms']}",
        "w_pressure": f"cpu_safe_high={reward_v2['pressure']['cpu_safe_high']}, cpu_hard_limit={reward_v2['pressure']['cpu_hard_limit']}, memory_safe_high={reward_v2['pressure']['memory_safe_high']}, memory_hard_limit={reward_v2['pressure']['memory_hard_limit']}",
        "w_waste": f"cpu_waste_threshold={reward_v2['waste']['cpu_waste_threshold']}, memory_waste_threshold={reward_v2['waste']['memory_waste_threshold']}",
        "w_action": f"per_action_cost={reward_v2['action']['per_action_cost']}",
        "w_reversal": f"reversal_window_seconds={reward_v2['reversal']['reversal_window_seconds']}, per_reversal_cost={reward_v2['reversal']['per_reversal_cost']}",
        "w_boundary": f"per_boundary_violation_cost={reward_v2['boundary']['per_boundary_violation_cost']}",
    }
    rows = [{"component": name, "weight": weight, "meaning": meanings[name], "threshold_params": thresholds[name]} for name, weight in weights.items()]
    assert len(rows) == 7, f"expected 7 reward components, found {len(rows)}"
    _write_table(rows, ["component", "weight", "meaning", "threshold_params"], TABLES_DIR / "T2_reward_components")


def generate_t3_ppo_hyperparameters() -> None:
    rl_v2 = yaml.safe_load(open(RL_V2_PATH, encoding="utf-8"))
    ppo = rl_v2["ppo"]
    prereg = json.load(open(PREREGISTRATION_PATH, encoding="utf-8"))
    n_sessions = prereg["training_contract_reference"]["total_planned_sessions"]
    total_timesteps = prereg["training_contract_reference"]["total_timesteps_budget"]
    steps_per_session = total_timesteps // n_sessions
    rows = [{"parameter": k, "value": v} for k, v in ppo.items()]
    rows.append({"parameter": "training_budget", "value": f"{n_sessions} sessions x {steps_per_session} steps/session = {total_timesteps} total timesteps (final_min2 campaign)"})
    _write_table(rows, ["parameter", "value"], TABLES_DIR / "T3_ppo_hyperparameters")


def _peak_rps(profile) -> float:
    total = sum(ph.duration_seconds for ph in profile.phases)
    t = np.arange(0.0, total, 1.0)
    return max(profile.get_phase_and_target_rps(float(ti))[0] for ti in t)


def generate_t4_experimental_design() -> None:
    prereg = json.load(open(PREREGISTRATION_PATH, encoding="utf-8"))
    stress_plan = json.load(open(STRESS_PLAN_PATH, encoding="utf-8"))
    arm_rows = []
    for arm_id in ps.ARM_IDS:
        role = {"ppo_v2": "RL policy under test", "hpa_v2_cpu_memory": "resource-metric baseline (CPU+memory)", "hpa_v2_cpu_only": "Kubernetes-default baseline (CPU only)"}[arm_id]
        arm_rows.append({"arm": ps.arm_label(arm_id), "profiles_run_on": "all 3 eval + all 3 stress", "role": role})
    _write_table(arm_rows, ["arm", "profiles_run_on", "role"], TABLES_DIR / "T4a_experimental_design_arms")

    profile_rows = []
    for name in EVAL_PROFILE_NAMES:
        profile = parse_traffic_profile(EVAL_PROFILE_DIR / f"{name}.yaml")
        profile_rows.append({"campaign": "eval_final (pre-registered, CORR-60)", "profile": name, "resource_mode": profile.resource_mode, "peak_rps": _peak_rps(profile), "repeats": prereg["workloads"]["repeats_per_profile_per_arm"]})
    for p in stress_plan["design"]["profiles"]:
        name = p["name"]
        profile = parse_traffic_profile(STRESS_PROFILE_DIR / f"{name}.yaml")
        profile_rows.append({"campaign": "stress_explore (exploratory, CORR-65)", "profile": name, "resource_mode": profile.resource_mode, "peak_rps": _peak_rps(profile), "repeats": stress_plan["runner"]["repeats_per_profile_per_arm"]})
    _write_table(profile_rows, ["campaign", "profile", "resource_mode", "peak_rps", "repeats"], TABLES_DIR / "T4b_experimental_design_profiles")

    decision_rule_rows = [
        {"rule": "reporting", "statement": prereg["decision_rule"]["reporting"]},
        {"rule": "claiming_better", "statement": prereg["decision_rule"]["claiming_better"]},
        {"rule": "no_aggregate_winner", "statement": prereg["decision_rule"]["no_aggregate_winner"]},
        {"rule": "applies_to", "statement": "eval_final only (3 repeats/cell). Does NOT apply to stress_explore (1 repeat/cell, descriptive only, CORR-65)."},
    ]
    _write_table(decision_rule_rows, ["rule", "statement"], TABLES_DIR / "T4c_experimental_design_decision_rule")


def generate_eval_metrics_tables() -> None:
    run_metrics = av.analyze_evaluation_campaign(EVAL_FINAL_DIR)
    aggregated = av.aggregate_per_profile_arm(run_metrics)
    primary = list(av.PRIMARY_METRICS)
    secondary = [m for m in av.ALL_METRICS if m not in primary and "300ms" not in m]
    reference = [m for m in av.ALL_METRICS if "300ms" in m]
    for metrics, fname in [(primary, "eval_final_primary_metrics"), (secondary + reference, "eval_final_secondary_metrics")]:
        rows = []
        for (profile, arm_id), agg in sorted(aggregated.items()):
            row = {"profile": profile, "arm": ps.arm_label(arm_id), "n_repeats": agg["n_repeats"]}
            for metric in metrics:
                mean, sd = agg[metric]["mean"], agg[metric]["sd"]
                row[metric] = f"{mean:.4f} +/- {sd:.4f}" if mean is not None else "n/a"
            rows.append(row)
        _write_table(rows, ["profile", "arm", "n_repeats"] + metrics, TABLES_DIR / fname)


def generate_stress_metrics_table() -> None:
    run_metrics = av.analyze_evaluation_campaign(STRESS_DIR)
    aggregated = av.aggregate_per_profile_arm(run_metrics)
    rows = []
    for (profile, arm_id), agg in sorted(aggregated.items()):
        row = {"profile": profile, "arm": ps.arm_label(arm_id), "n_repeats": agg["n_repeats"], "exploratory": "yes"}
        for metric in av.ALL_METRICS:
            mean = agg[metric]["mean"]
            row[metric] = f"{mean:.4f}" if mean is not None else "n/a"
        rows.append(row)
    _write_table(rows, ["profile", "arm", "n_repeats", "exploratory"] + list(av.ALL_METRICS), TABLES_DIR / "stress_explore_descriptive_metrics")


def generate_training_summary_table() -> None:
    curve = av.analyze_training_curve(SESSIONS_DIR, UPDATE_METRICS_PATH)
    failed = _failed_retried_session_indices()
    first, last = curve[0], curve[-1]
    entropy_vals = [c["entropy_loss"] for c in curve if c["entropy_loss"] is not None]
    ev_vals = [c["explained_variance"] for c in curve if c["explained_variance"] is not None]
    rows = [
        {"field": "n_sessions_analyzed", "value": len(curve)},
        {"field": "first_session_entropy_loss", "value": round(first["entropy_loss"], 4) if first["entropy_loss"] is not None else "n/a"},
        {"field": "last_session_entropy_loss", "value": round(last["entropy_loss"], 4) if last["entropy_loss"] is not None else "n/a"},
        {"field": "min_entropy_loss", "value": round(min(entropy_vals), 4) if entropy_vals else "n/a"},
        {"field": "max_entropy_loss", "value": round(max(entropy_vals), 4) if entropy_vals else "n/a"},
        {"field": "first_session_explained_variance", "value": round(first["explained_variance"], 4) if first["explained_variance"] is not None else "n/a"},
        {"field": "last_session_explained_variance", "value": round(last["explained_variance"], 4) if last["explained_variance"] is not None else "n/a"},
        {"field": "min_explained_variance", "value": round(min(ev_vals), 4) if ev_vals else "n/a"},
        {"field": "max_explained_variance", "value": round(max(ev_vals), 4) if ev_vals else "n/a"},
        {"field": "n_failed_retried_attempts", "value": len(failed)},
        {"field": "failed_attempt_session_indices", "value": ", ".join(str(i) for i in failed)},
    ]
    _write_table(rows, ["field", "value"], TABLES_DIR / "training_summary")


def write_figures_md() -> None:
    lines = [
        "# V2 Phase 6E -- Final Publication Figures, Diagrams, and Tables Index", "",
        "Every figure/diagram this phase produced: file, caption, data source, key takeaway, suggested paper section.", "",
        "| File | Caption | Data source | Key takeaway | Section |", "|---|---|---|---|---|",
    ]
    for entry in FIGURES_MANIFEST:
        lines.append(f"| {entry['file']} | {entry['caption']} | {entry['source']} | {entry['takeaway']} | {entry['section']} |")

    diagrams = [
        ("diagram_system_architecture.svg / .pdf / .png", "System architecture: Minikube cluster (demo-app pods, HPA, Metrics Server @15s, Prometheus) plus host-side load generator/relay and PPO agent (observation builder -> policy -> safety layer -> executor), gated by the controller lock file.", "autoscaler/src/rl_env/, scripts/evaluate_v2_campaign.py build_live_context()", "The PPO agent and each HPA arm are mutually exclusive controllers of the same demo-app deployment, enforced by the controller lock -- never both active at once.", "Method"),
        ("diagram_rl_control_loop.svg / .pdf / .png", "The RL control loop: 11-feature observation -> PPO policy -> action -> safety gating -> Kubernetes API -> metrics -> 7-component reward -> update, one cycle per 30s decision interval.", "autoscaler/src/rl_env/observation.py, action_selection.py, reward_v2.py", "Safety gating sits between the policy's raw action and the actual Kubernetes API call, so an unsafe/tied decision is corrected before it reaches the cluster.", "Method"),
        ("diagram_experiment_pipeline.svg / .pdf / .png", "Project/experiment pipeline: V1 -> V2 design -> 48-session training -> pre-registered evaluation (eval_final, CORR-60) -> exploratory stress test (stress_explore, CORR-65).", "docs/experimental_design_corrections.md", "The stress test is a later, separate, descriptive-only stage -- it never supersedes the pre-registered evaluation.", "Design"),
        ("diagram_tie_break_flow.svg / .pdf / .png", "Decision flow for the safety-aware tie-break rule (R1-R6, CORR-58): from the PPO action-probability tie set to the final selected action.", "autoscaler/src/rl_env/action_selection.py (select_safety_aware_action)", "SCALE_DOWN can never win a tie by construction; an unknown-stress tie fails safe to HOLD.", "Method"),
    ]
    lines += ["", "## Diagrams", "", "| File | Caption | Data source | Key takeaway | Section |", "|---|---|---|---|---|"]
    for file, caption, source, takeaway, section in diagrams:
        lines.append(f"| {file} | {caption} | {source} | {takeaway} | {section} |")

    lines += [
        "", "## Note on the PPO policy map",
        "`discussion_ppo_policy_map` shows the DOMINANT action per (cpu_per_replica, memory_per_replica) bin "
        "(a 3-color categorical heatmap), rather than 3 separate per-action share files, since Phase 6D found "
        "SCALE_DOWN essentially never fires and HOLD dominates almost every bin -- a single dominant-action map "
        "is more informative here than 3 mostly-empty share maps.",
        "", "## Note on reward-component shares",
        "`training_reward_components` reports what `step_telemetry.jsonl` actually logs. If no "
        "`reward_component_*` fields are present, the figure says so explicitly rather than fabricating a "
        "7-component breakdown that isn't in the data.",
    ]
    (FIGURES_DIR / "FIGURES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ps.apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    for name in EVAL_PROFILE_NAMES:
        plot_workload(EVAL_PROFILE_DIR, name, "eval")
    for name in STRESS_PROFILE_NAMES:
        plot_workload(STRESS_PROFILE_DIR, name, "stress")

    plot_training_figures()
    plot_evaluation_figures()
    plot_stress_figures()
    plot_discussion_figures()
    regenerate_diagrams()

    generate_t1_observation_features()
    generate_t2_reward_components()
    generate_t3_ppo_hyperparameters()
    generate_t4_experimental_design()
    generate_eval_metrics_tables()
    generate_stress_metrics_table()
    generate_training_summary_table()

    write_figures_md()
    print(f"done: {len(FIGURES_MANIFEST)} figures + 4 diagrams")


if __name__ == "__main__":
    main()
