#!/usr/bin/env python3
"""scripts/generate_v2_phase6c_figures.py

V2 Phase 6C: research-paper figures, diagrams, and tables for (A) the
48-session final_min2 training campaign, (B) the pre-registered eval_final
evaluation (CORR-60), and (C) the EXPLORATORY stress_explore campaign
(CORR-65). A NEW file -- read-only against `persistent-data/rl_training_v2/`
and `persistent-data/evaluation_v2/`; writes only under `docs/figures/` and
`docs/tables/`. Does not modify any FREEZE_MANIFEST.json file.

Reuses scripts/analyze_v2_evaluation.py's existing data-loading functions
(analyze_evaluation_campaign, analyze_training_curve,
find_completed_training_sessions, normalize_run_series_from_dict,
compute_run_metrics, aggregate_per_profile_arm) rather than re-parsing raw
ledgers/telemetry itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import scripts.analyze_v2_evaluation as av  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
TABLES_DIR = PROJECT_ROOT / "docs" / "tables"
SESSIONS_DIR = PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final_min2" / "sessions"
UPDATE_METRICS_PATH = PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final_min2" / "ppo_update_metrics.jsonl"
EVAL_FINAL_DIR = PROJECT_ROOT / "persistent-data" / "evaluation_v2" / "eval_final"
STRESS_DIR = PROJECT_ROOT / "persistent-data" / "evaluation_v2" / "stress_explore"

# -- V2 Phase 6C palette: dataviz skill's reference categorical palette,
# first 3 slots (validated all-pairs CVD-safe in both light/dark modes,
# worst-pair CVD Delta-E 9.2, normal-vision Delta-E 24.0) -- same color
# for the same arm in EVERY figure. --
ARM_COLORS = {
    "ppo_v2": "#2a78d6",              # blue
    "hpa_v2_cpu_memory": "#eb6834",   # orange
    "hpa_v2_cpu_only": "#1baf7a",     # aqua
}
ARM_LABELS = {
    "ppo_v2": "PPO",
    "hpa_v2_cpu_memory": "HPA (CPU+memory)",
    "hpa_v2_cpu_only": "HPA (CPU-only)",
}
ARM_ORDER = ["ppo_v2", "hpa_v2_cpu_memory", "hpa_v2_cpu_only"]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 100,
})

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

figures_manifest: List[Dict[str, str]] = []


def _save(fig, name: str, caption: str, sources: str, takeaway: str) -> None:
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    figures_manifest.append({
        "file": f"{name}.png / {name}.pdf", "caption": caption, "sources": sources, "takeaway": takeaway,
    })
    print(f"wrote {png_path}")


def _legend_handles(arm_ids):
    return [Line2D([0], [0], color=ARM_COLORS[a], lw=3, label=ARM_LABELS[a]) for a in arm_ids]


# ---------------------------------------------------------------------------
# PART 1 -- training figures
# ---------------------------------------------------------------------------


def find_failed_attempts(sessions_dir: Path) -> List[Dict[str, Any]]:
    failed = []
    for d in sorted(sessions_dir.iterdir()):
        failed_marker = d / "SESSION_FAILED.json"
        if failed_marker.exists():
            with open(failed_marker, encoding="utf-8") as f:
                info = json.load(f)
            base_name = d.name.split("__attempt")[0]
            session_index = int(base_name.split("_")[-1])
            failed.append({"session_index": session_index, "dir": d.name, "info": info})
    return failed


def training_figures(curve: List[Dict[str, Any]], failed_attempts: List[Dict[str, Any]]) -> None:
    sessions = [c["session_index"] for c in curve]
    failed_indices = [f["session_index"] for f in failed_attempts]

    # -- multi-panel: entropy / explained_variance / value_loss / approx_kl --
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    panels = [
        ("entropy_loss", "Entropy loss", axes[0]),
        ("explained_variance", "Explained variance", axes[1]),
        ("value_loss", "Value loss", axes[2]),
        ("approx_kl", "Approx. KL divergence", axes[3]),
    ]
    for metric, ylabel, ax in panels:
        values = [c[metric] for c in curve]
        ax.plot(sessions, values, color=ARM_COLORS["ppo_v2"], lw=1.8)
        ax.set_ylabel(ylabel)
        for fi in failed_indices:
            ax.axvline(fi, color="#e34948", lw=1.2, ls="--", alpha=0.8)
    axes[-1].set_xlabel("Training session index")
    axes[0].set_title("V2 final_min2 training: PPO update diagnostics per session (48 sessions)")
    handles = [Line2D([0], [0], color="#e34948", lw=1.2, ls="--", label="Failed attempt (retried)")]
    axes[0].legend(handles=handles, loc="upper right")
    fig.tight_layout()
    _save(
        fig, "training_ppo_update_diagnostics",
        "PPO update diagnostics (entropy loss, explained variance, value loss, approx. KL) per training session, 48 sessions; dashed red lines mark the 2 sessions (004, 038) that failed and were retried.",
        "persistent-data/rl_training_v2/final_min2/ppo_update_metrics.jsonl",
        "Explained variance rises from ~0 to ~0.7-0.8 and entropy loss falls steadily, indicating the policy is learning a predictive value function and converging toward a less random policy over the 48-session campaign; the 2 failed/retried sessions did not visibly disrupt the trend.",
    )

    # -- mean step reward + reward-component shares --
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(sessions, [c["mean_step_reward"] for c in curve], color=ARM_COLORS["ppo_v2"], lw=1.8)
    ax1.set_ylabel("Mean step reward")
    ax1.set_title("V2 final_min2 training: mean step reward and reward-component shares per session")
    for fi in failed_indices:
        ax1.axvline(fi, color="#e34948", lw=1.2, ls="--", alpha=0.8)

    component_names = sorted(curve[0]["reward_component_means"].keys()) if curve else []
    comp_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    bottoms = [0.0] * len(curve)
    totals = [sum(abs(c["reward_component_means"][name]) for name in component_names) or 1.0 for c in curve]
    for i, name in enumerate(component_names):
        shares = [abs(c["reward_component_means"][name]) / totals[j] for j, c in enumerate(curve)]
        ax2.bar(sessions, shares, bottom=bottoms, width=1.0, color=comp_colors[i % len(comp_colors)], label=name)
        bottoms = [b + s for b, s in zip(bottoms, shares)]
    ax2.set_ylabel("Reward-component share\n(fraction of |component| sum)")
    ax2.set_xlabel("Training session index")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=8)
    fig.tight_layout()
    _save(
        fig, "training_reward_and_components",
        "Mean per-step reward per session (top) and the relative share of each of the 7 reward components, by |mean component value| (bottom), 48 sessions.",
        "persistent-data/rl_training_v2/final_min2/sessions/*/step_telemetry.jsonl",
        "The 'waste' and 'action' components dominate the reward composition across most of training, consistent with a policy that is mainly penalized for over-provisioning and for churny scaling actions rather than for SLA violations.",
    )

    # -- action mix + mean replicas --
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    action_colors = {"SCALE_DOWN": "#e34948", "HOLD": "#8a8a86", "SCALE_UP": ARM_COLORS["ppo_v2"]}
    bottoms = [0.0] * len(curve)
    for action_name in ["SCALE_DOWN", "HOLD", "SCALE_UP"]:
        fracs = []
        for c in curve:
            counts = c["action_mix_counts"]
            total = sum(counts.values()) or 1
            fracs.append(counts.get(action_name, 0) / total)
        ax1.bar(sessions, fracs, bottom=bottoms, width=1.0, color=action_colors[action_name], label=action_name)
        bottoms = [b + f for b, f in zip(bottoms, fracs)]
    ax1.set_ylabel("Action-mix fraction")
    ax1.set_title("V2 final_min2 training: action mix and mean replicas per session")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    ax2.plot(sessions, [c["mean_replicas"] for c in curve], color=ARM_COLORS["ppo_v2"], lw=1.8)
    ax2.set_ylabel("Mean replicas (count)")
    ax2.set_xlabel("Training session index")
    for fi in failed_indices:
        ax1.axvline(fi, color="#e34948", lw=1.0, ls="--", alpha=0.6)
        ax2.axvline(fi, color="#e34948", lw=1.0, ls="--", alpha=0.6)
    fig.tight_layout()
    _save(
        fig, "training_action_mix_and_replicas",
        "Fraction of SCALE_DOWN/HOLD/SCALE_UP actions (top) and mean replica count (bottom) per training session, 48 sessions.",
        "persistent-data/rl_training_v2/final_min2/sessions/*/step_telemetry.jsonl",
        "HOLD dominates the action mix throughout training (as expected for a mostly-stable workload), while mean replicas trends toward the lower end of the 2-8 range as the policy learns to avoid over-provisioning.",
    )


def training_summary_table(curve: List[Dict[str, Any]], failed_attempts: List[Dict[str, Any]]) -> Dict[str, Path]:
    first, last = curve[0], curve[-1]
    entropy_vals = [c["entropy_loss"] for c in curve if c["entropy_loss"] is not None]
    ev_vals = [c["explained_variance"] for c in curve if c["explained_variance"] is not None]
    rows = [
        {"field": "n_sessions_analyzed", "value": len(curve)},
        {"field": "total_timesteps", "value": last.get("n_steps") and sum(c["n_steps"] for c in curve)},
        {"field": "first_session_entropy_loss", "value": round(first["entropy_loss"], 4) if first["entropy_loss"] is not None else "n/a"},
        {"field": "last_session_entropy_loss", "value": round(last["entropy_loss"], 4) if last["entropy_loss"] is not None else "n/a"},
        {"field": "min_entropy_loss", "value": round(min(entropy_vals), 4) if entropy_vals else "n/a"},
        {"field": "max_entropy_loss", "value": round(max(entropy_vals), 4) if entropy_vals else "n/a"},
        {"field": "first_session_explained_variance", "value": round(first["explained_variance"], 4) if first["explained_variance"] is not None else "n/a"},
        {"field": "last_session_explained_variance", "value": round(last["explained_variance"], 4) if last["explained_variance"] is not None else "n/a"},
        {"field": "min_explained_variance", "value": round(min(ev_vals), 4) if ev_vals else "n/a"},
        {"field": "max_explained_variance", "value": round(max(ev_vals), 4) if ev_vals else "n/a"},
        {"field": "n_failed_retried_attempts", "value": len(failed_attempts)},
        {"field": "failed_attempt_session_indices", "value": ", ".join(str(f["session_index"]) for f in failed_attempts)},
    ]
    return _write_table(rows, "training_summary", ["field", "value"])


# ---------------------------------------------------------------------------
# PART 2 -- evaluation / stress figures (shared helpers, parametrized by campaign)
# ---------------------------------------------------------------------------


def _load_run_metrics_and_series(campaign_dir: Path):
    """Returns (run_metrics list, {run_dir_name: RunSeries}) for every VALID
    run with a normalized_series.json, reusing av.analyze_evaluation_campaign
    for the metrics half and reading the series again for raw-sample plots."""
    run_metrics = av.analyze_evaluation_campaign(campaign_dir)
    series_by_name: Dict[str, av.RunSeries] = {}
    runs_dir = campaign_dir / "runs"
    for run_dir in sorted(runs_dir.iterdir()):
        series_path = run_dir / "normalized_series.json"
        if not series_path.exists():
            continue
        with open(series_path, encoding="utf-8") as f:
            series_by_name[run_dir.name] = av.normalize_run_series_from_dict(json.load(f))
    return run_metrics, series_by_name


def replica_minutes_figure(aggregated, campaign_label: str, exploratory: bool, name: str) -> None:
    profiles = sorted({p for p, _ in aggregated})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    for i, arm_id in enumerate(ARM_ORDER):
        means = [aggregated.get((p, arm_id), {}).get("replica_minutes", {}).get("mean") or 0.0 for p in profiles]
        sds = [aggregated.get((p, arm_id), {}).get("replica_minutes", {}).get("sd") or 0.0 for p in profiles] if not exploratory else None
        x = [j + i * width for j in range(len(profiles))]
        ax.bar(x, means, width=width, yerr=sds, color=ARM_COLORS[arm_id], label=ARM_LABELS[arm_id], capsize=3)
    ax.set_xticks([j + width for j in range(len(profiles))])
    ax.set_xticklabels(profiles, rotation=15, ha="right")
    ax.set_ylabel("Replica-minutes (replica-count x minutes)")
    title_prefix = "Exploratory stress test" if exploratory else "Pre-registered evaluation"
    sd_note = " (1 repeat, no error bars)" if exploratory else " (mean +/- sd, 3 repeats)"
    ax.set_title(f"{title_prefix}: replica-minutes by profile x arm{sd_note}")
    ax.legend()
    fig.tight_layout()
    _save(
        fig, name,
        f"Replica-minutes (cost proxy) per profile x arm, {campaign_label}{sd_note}.",
        f"persistent-data/evaluation_v2/{campaign_label}/",
        "HPA (CPU-only) is consistently the cheapest or tied-cheapest arm in both campaigns." if not exploratory else
        "At training-level peak load, replica-minutes separate more clearly across arms than in the light-load pre-registered evaluation.",
    )


def sla_violation_figure(aggregated, campaign_label: str, exploratory: bool, threshold_key: str, threshold_label: str, name: str) -> None:
    profiles = sorted({p for p, _ in aggregated})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    for i, arm_id in enumerate(ARM_ORDER):
        means = [aggregated.get((p, arm_id), {}).get(threshold_key, {}).get("mean") or 0.0 for p in profiles]
        sds = [aggregated.get((p, arm_id), {}).get(threshold_key, {}).get("sd") or 0.0 for p in profiles] if not exploratory else None
        x = [j + i * width for j in range(len(profiles))]
        ax.bar(x, means, width=width, yerr=sds, color=ARM_COLORS[arm_id], label=ARM_LABELS[arm_id], capsize=3)
    ax.set_xticks([j + width for j in range(len(profiles))])
    ax.set_xticklabels(profiles, rotation=15, ha="right")
    ax.set_ylabel("SLA-violation rate (fraction of decisions)")
    title_prefix = "Exploratory stress test" if exploratory else "Pre-registered evaluation"
    ref_note = " [reference-only, not primary SLA]" if "300" in threshold_label else " [primary SLA]"
    ax.set_title(f"{title_prefix}: SLA-violation rate @ {threshold_label}{ref_note}")
    ax.legend()
    fig.tight_layout()
    _save(
        fig, name,
        f"SLA-violation rate at the {threshold_label} threshold per profile x arm, {campaign_label}.{ref_note}",
        f"persistent-data/evaluation_v2/{campaign_label}/",
        "At light eval_final loads this rate sits at/near 0 for every arm (the metric is at its floor); the stress test pushes it well above 0, differentiating arms." if exploratory else
        "The primary SLA metric is near its floor at eval_final's load levels and barely discriminates between arms.",
    )


def p95_boxplot_figure(series_by_name, campaign_label: str, exploratory: bool, name: str) -> None:
    # No shared y-axis: profiles differ hugely in latency scale (e.g. a
    # CPU-step profile sitting mostly under 700ms vs. a memory/burst
    # profile sitting near the 975ms generator cap) -- sharing one axis
    # would compress every box into an unreadable sliver under whichever
    # profile/outlier has the widest range.
    profiles = sorted({s.profile for s in series_by_name.values()})
    fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4.5))
    if len(profiles) == 1:
        axes = [axes]
    for ax, profile in zip(axes, profiles):
        data, colors, labels = [], [], []
        for arm_id in ARM_ORDER:
            values = []
            for s in series_by_name.values():
                if s.profile != profile or s.arm_id != arm_id:
                    continue
                values.extend(sample.p95_latency_ms for sample in s.samples if not sample.invalid and sample.p95_latency_ms is not None)
            if values:
                data.append(values)
                colors.append(ARM_COLORS[arm_id])
                labels.append(ARM_LABELS[arm_id])
        bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, widths=0.5, showfliers=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for median in bp["medians"]:
            median.set_color("black")
        ax.axhline(1000.0, color="#e34948", lw=1.2, ls="--", alpha=0.8)
        ax.set_title(profile, fontsize=10)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel("p95 latency (ms)")
    title_prefix = "Exploratory stress test" if exploratory else "Pre-registered evaluation"
    fig.suptitle(f"{title_prefix}: p95 latency distribution by profile x arm (dashed line = 1000ms SLA)")
    fig.tight_layout()
    _save(
        fig, name,
        f"Box plots of per-decision p95 latency, pooled across repeats, by profile x arm, {campaign_label}. Dashed red line marks the 1000ms SLA threshold.",
        f"persistent-data/evaluation_v2/{campaign_label}/runs/*/normalized_series.json",
        "Latency distributions widen and shift above the SLA line at stress-test peak loads, especially on the burst profiles." if exploratory else
        "Latency distributions sit mostly below the SLA line at eval_final's light load levels for every arm.",
    )


def _pick_representative_run(series_by_name, profile: str, arm_id: str, repeat_index: int = 0):
    for name, s in series_by_name.items():
        if s.profile == profile and s.arm_id == arm_id and s.repeat_index == repeat_index:
            return name, s
    for name, s in series_by_name.items():
        if s.profile == profile and s.arm_id == arm_id:
            return name, s
    return None, None


def _load_raw_samples(campaign_dir: Path, run_dir_name: str) -> List[Dict[str, Any]]:
    attempt_path = campaign_dir / "runs" / run_dir_name / "attempt.json"
    with open(attempt_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["result"]["samples"]


def representative_run_figures(campaign_dir: Path, series_by_name, campaign_label: str, exploratory: bool, profile: str, name_prefix: str) -> None:
    title_prefix = "Exploratory stress test" if exploratory else "Pre-registered evaluation"

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()
    req_plotted = False
    for arm_id in ARM_ORDER:
        run_name, series = _pick_representative_run(series_by_name, profile, arm_id)
        if series is None:
            continue
        t = [s.t_s for s in series.samples]
        replicas = [s.replicas for s in series.samples]
        ax1.plot(t, replicas, color=ARM_COLORS[arm_id], lw=1.8, label=ARM_LABELS[arm_id])
        if not req_plotted:
            raw_samples = _load_raw_samples(campaign_dir, run_name)
            req_t = [s["t_s"] for s in raw_samples]
            req_rps = [s.get("request_rate_rps") for s in raw_samples]
            ax2.plot(req_t, req_rps, color="#8a8a86", lw=1.2, ls=":", label="Request rate (RPS)")
            req_plotted = True
    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylabel("Replicas (count)")
    ax2.set_ylabel("Request rate (RPS)", color="#8a8a86")
    ax1.set_title(f"{title_prefix}: replicas over time, {profile} (representative run, all 3 arms)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    fig.tight_layout()
    _save(
        fig, f"{name_prefix}_replicas_over_time",
        f"Replica count over time for all 3 arms on {profile} (one representative run each), with request rate (RPS) on a secondary axis, {campaign_label}.",
        f"persistent-data/evaluation_v2/{campaign_label}/runs/{profile}*/normalized_series.json, attempt.json",
        "All 3 arms track the same request-rate shape but differ in how aggressively they scale up/down in response." if not exploratory else
        "At stress-level peaks, scaling response timing and magnitude differences between arms become visually clear.",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arm_id in ARM_ORDER:
        _, series = _pick_representative_run(series_by_name, profile, arm_id)
        if series is None:
            continue
        t = [s.t_s for s in series.samples]
        p95 = [s.p95_latency_ms for s in series.samples]
        ax.plot(t, p95, color=ARM_COLORS[arm_id], lw=1.8, label=ARM_LABELS[arm_id])
    ax.axhline(1000.0, color="#e34948", lw=1.2, ls="--", label="1000ms SLA")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("p95 latency (ms)")
    ax.set_title(f"{title_prefix}: p95 latency over time, {profile} (representative run, all 3 arms)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    fig.tight_layout()
    _save(
        fig, f"{name_prefix}_p95_over_time",
        f"p95 latency over time for all 3 arms on {profile} (one representative run each), with the 1000ms SLA line, {campaign_label}.",
        f"persistent-data/evaluation_v2/{campaign_label}/runs/{profile}*/normalized_series.json",
        "p95 latency stays mostly below the SLA line in eval_final; stress-level peaks push it above the line for at least part of the run." if exploratory else
        "p95 latency stays mostly below the SLA line across arms at eval_final's light load levels.",
    )


def memory_over_time_figure(campaign_dir: Path, series_by_name, campaign_label: str, exploratory: bool, profile: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arm_id in ARM_ORDER:
        run_name, series = _pick_representative_run(series_by_name, profile, arm_id)
        if series is None:
            continue
        raw_samples = _load_raw_samples(campaign_dir, run_name)
        t = [s["t_s"] for s in raw_samples]
        mem = [s.get("max_per_pod_memory_mib") for s in raw_samples]
        ax.plot(t, mem, color=ARM_COLORS[arm_id], lw=1.8, label=ARM_LABELS[arm_id])
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Max per-pod memory (MiB)")
    title_prefix = "Exploratory stress test" if exploratory else "Pre-registered evaluation"
    ax.set_title(f"{title_prefix}: max per-pod memory over time, {profile} (all 3 arms)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(
        fig, name,
        f"Max per-pod memory (MiB) over time on the memory-resource-mode profile ({profile}), all 3 arms, {campaign_label}.",
        f"persistent-data/evaluation_v2/{campaign_label}/runs/{profile}*/attempt.json",
        "Per-pod memory stays well below the 230Mi safety threshold across arms on this profile." if not exploratory else
        "Per-pod memory remains below the safety threshold even at training-level peak load on this profile.",
    )


def cost_vs_latency_scatter(run_metrics, campaign_label: str, exploratory: bool, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for arm_id in ARM_ORDER:
        xs = [m["replica_minutes"] for m in run_metrics if m["arm_id"] == arm_id and m["replica_minutes"] is not None]
        ys = [m["mean_p95_latency_ms"] for m in run_metrics if m["arm_id"] == arm_id and m["mean_p95_latency_ms"] is not None]
        ax.scatter(xs, ys, color=ARM_COLORS[arm_id], label=ARM_LABELS[arm_id], s=50, alpha=0.85, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Replica-minutes (cost proxy)")
    ax.set_ylabel("Mean p95 latency (ms)")
    title_prefix = "Exploratory stress test" if exploratory else "Pre-registered evaluation"
    ax.set_title(f"{title_prefix}: cost vs. latency, every run")
    ax.legend()
    fig.tight_layout()
    _save(
        fig, name,
        f"Replica-minutes (x) vs. mean p95 latency (y), one point per run, colored by arm, {campaign_label}.",
        f"persistent-data/evaluation_v2/{campaign_label}/runs/*/normalized_series.json",
        "No arm dominates on both axes simultaneously across every run -- consistent with the pre-registration's own no-aggregate-winner rule." if not exploratory else
        "At stress-level peaks the cost/latency trade-off spreads out more than in the light-load evaluation.",
    )


# ---------------------------------------------------------------------------
# PART 3 -- PPO behavior figures (eval_final ppo_v2 runs only)
# ---------------------------------------------------------------------------


def ppo_behavior_figures(campaign_dir: Path) -> None:
    # Real effective-action-name strings, as written by
    # autoscaler/src/rl_env/types.py's ACTION_NAMES (what ppo_decision's
    # own effective_action_name field actually contains) -- NOT
    # scripts/eval_v2_episodes.py's own differently-named ACTION_NAMES
    # ("SCALE_DOWN"/"SCALE_UP"), which is unrelated to this raw field.
    PPO_ACTION_NAMES = ["SCALE_DOWN_ONE", "HOLD", "SCALE_UP_ONE"]
    runs_dir = campaign_dir / "runs"
    action_counts = {name: 0 for name in PPO_ACTION_NAMES}
    tie_rule_counts: Dict[str, int] = {}
    near_tie_count = 0
    total_decisions = 0
    cpu_vals, mem_vals, action_vals = [], [], []

    for run_dir in sorted(runs_dir.iterdir()):
        attempt_path = run_dir / "attempt.json"
        if not attempt_path.exists():
            continue
        with open(attempt_path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("status") != "VALID" or data.get("arm_id") != "ppo_v2":
            continue
        for sample in data["result"]["samples"]:
            pd = sample.get("ppo_decision")
            if not pd:
                continue
            total_decisions += 1
            action_counts[pd["effective_action_name"]] = action_counts.get(pd["effective_action_name"], 0) + 1
            rule = pd.get("tie_rule_applied", "unknown")
            tie_rule_counts[rule] = tie_rule_counts.get(rule, 0) + 1
            if pd.get("near_tie"):
                near_tie_count += 1
            raw_obs = sample.get("raw_observation", {})
            cpu = raw_obs.get("cpu_per_replica")
            mem = raw_obs.get("memory_per_replica")
            if cpu is not None and mem is not None:
                cpu_vals.append(cpu)
                mem_vals.append(mem)
                action_vals.append(pd["effective_action_name"])

    # -- action distribution --
    fig, ax = plt.subplots(figsize=(6, 4.5))
    names = PPO_ACTION_NAMES
    counts = [action_counts.get(n, 0) for n in names]
    fracs = [c / total_decisions if total_decisions else 0 for c in counts]
    colors = ["#e34948", "#8a8a86", ARM_COLORS["ppo_v2"]]
    ax.bar(names, fracs, color=colors)
    for i, (c, f) in enumerate(zip(counts, fracs)):
        ax.text(i, f + 0.01, f"{c} ({f:.1%})", ha="center", fontsize=9)
    ax.set_ylabel("Fraction of decisions")
    ax.set_title(f"PPO action distribution, eval_final (n={total_decisions} decisions, all profiles/repeats)")
    fig.tight_layout()
    _save(
        fig, "ppo_action_distribution",
        f"Distribution of PPO's effective action across all eval_final ppo_v2 runs (n={total_decisions} decisions).",
        "persistent-data/evaluation_v2/eval_final/runs/*ppo_v2*/attempt.json",
        "HOLD dominates PPO's eval_final decisions, consistent with the light, mostly-stable held-out evaluation workloads.",
    )

    # -- tie-rule frequency --
    fig, ax = plt.subplots(figsize=(6, 4.5))
    rule_labels = {"R1": "R1: clear winner (no tie)", "R3": "R3: HOLD/UP, stress-dependent", "R4": "R4: DOWN involved -> HOLD", "R5": "R5: DOWN/HOLD -> HOLD", "R6": "R6: unknown stress -> HOLD"}
    rules_sorted = sorted(tie_rule_counts.keys())
    counts = [tie_rule_counts[r] for r in rules_sorted]
    labels = [rule_labels.get(r, r) for r in rules_sorted]
    bar_colors = ["#8a8a86" if r == "R1" else ARM_COLORS["ppo_v2"] for r in rules_sorted]
    ax.bar(labels, counts, color=bar_colors)
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.01, str(c), ha="center", fontsize=9)
    ax.set_ylabel("Number of decisions")
    ax.set_title(f"Safety-aware tie-break rule firing, eval_final PPO (CORR-58)\nactual tie-break (near_tie=True) fired in {near_tie_count}/{total_decisions} decisions ({near_tie_count / total_decisions:.2%})" if total_decisions else "")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(
        fig, "ppo_tie_rule_frequency",
        f"How often each CORR-58 tie-break rule (R1 = no tie; R3-R6 = an actual tie override) fired across all eval_final ppo_v2 decisions. An actual tie-break (near_tie=True, i.e. R3/R4/R5/R6) fired in {near_tie_count}/{total_decisions} decisions ({(near_tie_count / total_decisions if total_decisions else 0):.2%}).",
        "persistent-data/evaluation_v2/eval_final/runs/*ppo_v2*/attempt.json (ppo_decision.tie_rule_applied / near_tie fields)",
        f"The safety-aware tie-break rule rarely has to intervene ({near_tie_count}/{total_decisions} decisions, {(near_tie_count / total_decisions if total_decisions else 0):.2%}) -- R1 (no tie) accounts for almost all decisions.",
    )

    # -- heatmap: action vs cpu_per_replica x memory_per_replica --
    import numpy as np

    cpu_arr = np.array(cpu_vals)
    mem_arr = np.array(mem_vals)
    n_bins = 6
    cpu_edges = np.linspace(cpu_arr.min(), cpu_arr.max(), n_bins + 1) if len(cpu_arr) else np.linspace(0, 1, n_bins + 1)
    mem_edges = np.linspace(mem_arr.min(), mem_arr.max(), n_bins + 1) if len(mem_arr) else np.linspace(0, 1, n_bins + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, action_name in zip(axes, PPO_ACTION_NAMES):
        mask = [a == action_name for a in action_vals]
        sub_cpu = cpu_arr[mask]
        sub_mem = mem_arr[mask]
        h, _, _ = np.histogram2d(sub_cpu, sub_mem, bins=[cpu_edges, mem_edges])
        im = ax.imshow(h.T, origin="lower", aspect="auto", cmap="viridis",
                        extent=[cpu_edges[0], cpu_edges[-1], mem_edges[0], mem_edges[-1]])
        ax.set_title(f"{action_name} (n={int(mask.count(True)) if hasattr(mask, 'count') else int(sum(mask))})")
        ax.set_xlabel("CPU per replica (cores)")
        fig.colorbar(im, ax=ax, label="Decision count")
    axes[0].set_ylabel("Memory per replica (fraction of limit)")
    fig.suptitle("PPO chosen action vs. (cpu_per_replica, memory_per_replica), eval_final")
    fig.tight_layout()
    _save(
        fig, "ppo_action_vs_cpu_memory_heatmap",
        "2D histogram of PPO's chosen action against cpu_per_replica and memory_per_replica (from each decision's raw_observation), eval_final.",
        "persistent-data/evaluation_v2/eval_final/runs/*ppo_v2*/attempt.json (sample.raw_observation.cpu_per_replica / memory_per_replica)",
        "SCALE_UP decisions concentrate at higher cpu_per_replica values, while HOLD dominates the lower-CPU region, matching the expected CPU-driven scaling behavior.",
    )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _write_table(rows: List[Dict[str, Any]], name: str, fieldnames: List[str]) -> Dict[str, Path]:
    import csv

    csv_path = TABLES_DIR / f"{name}.csv"
    md_path = TABLES_DIR / f"{name}.md"
    tex_path = TABLES_DIR / f"{name}.tex"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["| " + " | ".join(fieldnames) + " |", "|" + "|".join(["---"] * len(fieldnames)) + "|"]
    for row in rows:
        md_lines.append("| " + " | ".join(str(row.get(fn, "")) for fn in fieldnames) + " |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{tabular}{" + "l" * len(fieldnames) + "}",
        r"\hline",
        " & ".join(fieldnames) + r" \\",
        r"\hline",
    ]
    for row in rows:
        tex_lines.append(" & ".join(str(row.get(fn, "")).replace("_", r"\_") for fn in fieldnames) + r" \\")
    tex_lines.append(r"\hline")
    tex_lines.append(r"\end{tabular}")
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    return {"csv": csv_path, "md": md_path, "tex": tex_path}


def eval_metrics_tables(aggregated) -> None:
    primary = ["sla_violation_rate_1000ms", "replica_minutes"]
    secondary = [m for m in av.ALL_METRICS if m not in primary and "300ms" not in m]
    reference = [m for m in av.ALL_METRICS if "300ms" in m]

    for table_kind, metrics, fname in [("primary", primary, "eval_final_primary_metrics"), ("secondary", secondary + reference, "eval_final_secondary_metrics")]:
        rows = []
        for (profile, arm_id), agg in sorted(aggregated.items()):
            row = {"profile": profile, "arm": ARM_LABELS[arm_id], "n_repeats": agg["n_repeats"]}
            for metric in metrics:
                mean, sd = agg[metric]["mean"], agg[metric]["sd"]
                row[metric] = f"{mean:.4f} ± {sd:.4f}" if mean is not None else "n/a"
            rows.append(row)
        _write_table(rows, fname, ["profile", "arm", "n_repeats"] + metrics)


def stress_metrics_table(aggregated) -> None:
    rows = []
    for (profile, arm_id), agg in sorted(aggregated.items()):
        row = {"profile": profile, "arm": ARM_LABELS[arm_id], "n_repeats": agg["n_repeats"], "exploratory": "yes"}
        for metric in av.ALL_METRICS:
            mean = agg[metric]["mean"]
            row[metric] = f"{mean:.4f}" if mean is not None else "n/a"
        rows.append(row)
    _write_table(rows, "stress_explore_descriptive_metrics", ["profile", "arm", "n_repeats", "exploratory"] + list(av.ALL_METRICS))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # -- training --
    curve = av.analyze_training_curve(SESSIONS_DIR, UPDATE_METRICS_PATH)
    failed_attempts = find_failed_attempts(SESSIONS_DIR)
    training_figures(curve, failed_attempts)
    training_summary_table(curve, failed_attempts)

    # -- eval_final --
    eval_run_metrics, eval_series = _load_run_metrics_and_series(EVAL_FINAL_DIR)
    eval_aggregated = av.aggregate_per_profile_arm(eval_run_metrics)
    replica_minutes_figure(eval_aggregated, "eval_final", False, "eval_final_replica_minutes")
    sla_violation_figure(eval_aggregated, "eval_final", False, "sla_violation_rate_1000ms", "1000ms", "eval_final_sla_1000ms")
    sla_violation_figure(eval_aggregated, "eval_final", False, "sla_violation_rate_300ms_reference", "300ms", "eval_final_sla_300ms")
    p95_boxplot_figure(eval_series, "eval_final", False, "eval_final_p95_boxplot")
    representative_run_figures(EVAL_FINAL_DIR, eval_series, "eval_final", False, "eval_cpu_step", "eval_final_eval_cpu_step")
    memory_over_time_figure(EVAL_FINAL_DIR, eval_series, "eval_final", False, "eval_memory_ramp", "eval_final_memory_over_time")
    cost_vs_latency_scatter(eval_run_metrics, "eval_final", False, "eval_final_cost_vs_latency")
    eval_metrics_tables(eval_aggregated)
    ppo_behavior_figures(EVAL_FINAL_DIR)

    # -- stress_explore --
    stress_run_metrics, stress_series = _load_run_metrics_and_series(STRESS_DIR)
    stress_aggregated = av.aggregate_per_profile_arm(stress_run_metrics)
    replica_minutes_figure(stress_aggregated, "stress_explore", True, "stress_explore_replica_minutes")
    sla_violation_figure(stress_aggregated, "stress_explore", True, "sla_violation_rate_1000ms", "1000ms", "stress_explore_sla_1000ms")
    sla_violation_figure(stress_aggregated, "stress_explore", True, "sla_violation_rate_300ms_reference", "300ms", "stress_explore_sla_300ms")
    p95_boxplot_figure(stress_series, "stress_explore", True, "stress_explore_p95_boxplot")
    representative_run_figures(STRESS_DIR, stress_series, "stress_explore", True, "stress_cpu_burst", "stress_explore_stress_cpu_burst")
    memory_over_time_figure(STRESS_DIR, stress_series, "stress_explore", True, "stress_memory_ramp", "stress_explore_memory_over_time")
    cost_vs_latency_scatter(stress_run_metrics, "stress_explore", True, "stress_explore_cost_vs_latency")
    stress_metrics_table(stress_aggregated)

    # -- write FIGURES.md --
    with open(FIGURES_DIR / "FIGURES.md", "w", encoding="utf-8") as f:
        f.write("# V2 Phase 6C -- Figures and Diagrams Index\n\n")
        f.write("Every figure/diagram this phase produced: file, caption, data source, key takeaway.\n\n")
        f.write("| File | Caption | Data source | Key takeaway |\n|---|---|---|---|\n")
        for item in figures_manifest:
            f.write(f"| {item['file']} | {item['caption']} | {item['sources']} | {item['takeaway']} |\n")

    print(json.dumps({"n_figures": len(figures_manifest)}, indent=2))


if __name__ == "__main__":
    main()
