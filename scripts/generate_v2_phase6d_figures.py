#!/usr/bin/env python3
"""scripts/generate_v2_phase6d_figures.py

V2 Phase 6D: publication-quality plots (P1-P8), styled via
scripts/paper_style.py. Read-only against persistent-data/ -- writes only
under docs/figures/. Reuses scripts/analyze_v2_evaluation.py's existing
data-loading functions (normalize_run_series_from_dict, compute_run_metrics,
find_completed_training_sessions, summarize_session_telemetry) rather than
re-parsing raw files from scratch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import scripts.analyze_v2_evaluation as av  # noqa: E402
import scripts.paper_style as ps  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
EVAL_FINAL_DIR = PROJECT_ROOT / "persistent-data" / "evaluation_v2" / "eval_final"
STRESS_DIR = PROJECT_ROOT / "persistent-data" / "evaluation_v2" / "stress_explore"
SESSIONS_DIR = PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final_min2" / "sessions"
UPDATE_METRICS_PATH = PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final_min2" / "ppo_update_metrics.jsonl"

SLA_MS = 1000.0


def _failed_retried_session_indices() -> list[int]:
    """Sessions with a sibling __attemptN directory had their first
    attempt fail and get retried -- the retry's own data is what
    find_completed_training_sessions()/analyze_training_curve() already
    use; this only identifies WHICH session indices to mark."""
    indices = set()
    for d in SESSIONS_DIR.iterdir():
        if d.is_dir() and "__attempt" in d.name:
            base = d.name.split("__attempt")[0]
            indices.add(int(base.split("_")[-1]))
    return sorted(indices)


def _load_attempt_samples(run_dir: Path) -> list[dict]:
    with open(run_dir / "attempt.json", encoding="utf-8") as f:
        return json.load(f)["result"]["samples"]


def _runs_for(campaign_dir: Path, profile: str, arm_id: str) -> list[Path]:
    runs_dir = campaign_dir / "runs"
    return sorted(d for d in runs_dir.iterdir() if d.name.startswith(f"{profile}__") and f"__{arm_id}__" in d.name)


def _median_cost_run(campaign_dir: Path, profile: str, arm_id: str) -> Path:
    candidates = _runs_for(campaign_dir, profile, arm_id)
    scored = []
    for run_dir in candidates:
        series_path = run_dir / "normalized_series.json"
        if not series_path.exists():
            continue
        series = av.normalize_run_series_from_dict(json.load(open(series_path, encoding="utf-8")))
        metrics = av.compute_run_metrics(series)
        scored.append((metrics["replica_minutes"] or 0.0, run_dir))
    scored.sort(key=lambda t: t[0])
    return scored[len(scored) // 2][1]  # median by cost


# ---------------------------------------------------------------------------
# P1 -- training diagnostics 2x2
# ---------------------------------------------------------------------------


def plot_p1_training_diagnostics() -> None:
    curve = av.analyze_training_curve(SESSIONS_DIR, UPDATE_METRICS_PATH)
    failed = _failed_retried_session_indices()
    sessions = [c["session_index"] for c in curve]

    fig, axes = plt.subplots(2, 2, figsize=(ps.DOUBLE_COL_WIDTH_IN, 4.6), sharex=True)
    panels = [
        ("entropy_loss", "Entropy loss", axes[0, 0]),
        ("explained_variance", "Explained variance", axes[0, 1]),
        ("value_loss", "Value loss", axes[1, 0]),
        ("mean_step_reward", "Mean step reward", axes[1, 1]),
    ]
    for key, ylabel, ax in panels:
        values = [c[key] for c in curve]
        ax.plot(sessions, values, color=ps.ARM_COLORS["ppo_v2"], linewidth=1.2)
        for idx in failed:
            ax.axvline(idx, color=ps.FAILED_SESSION_COLOR, linestyle="--", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ps.style_axes(ax)
    axes[1, 0].set_xlabel("Training session index")
    axes[1, 1].set_xlabel("Training session index")
    fig.legend(
        handles=[plt.Line2D([0], [0], color=ps.FAILED_SESSION_COLOR, linestyle="--", linewidth=0.8, label="Failed/retried session")],
        loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=1, frameon=False,
    )
    ps.save_figure(fig, FIGURES_DIR / "P1_training_diagnostics")


# ---------------------------------------------------------------------------
# P2 -- action mix (stacked area) + mean replicas
# ---------------------------------------------------------------------------


def plot_p2_action_mix_and_replicas() -> None:
    curve = av.analyze_training_curve(SESSIONS_DIR, UPDATE_METRICS_PATH)
    sessions = [c["session_index"] for c in curve]
    down = []
    hold = []
    up = []
    for c in curve:
        counts = c["action_mix_counts"]
        total = sum(counts.values()) or 1
        down.append(counts.get("SCALE_DOWN", 0) / total)
        hold.append(counts.get("HOLD", 0) / total)
        up.append(counts.get("SCALE_UP", 0) / total)
    mean_replicas = [c["mean_replicas"] for c in curve]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(ps.DOUBLE_COL_WIDTH_IN, 4.2), sharex=True)
    ax1.stackplot(
        sessions, down, hold, up,
        labels=["SCALE_DOWN", "HOLD", "SCALE_UP"],
        colors=["#56B4E9", "#CCCCCC", ps.ARM_COLORS["ppo_v2"]],
    )
    ax1.set_ylabel("Action share")
    ax1.set_ylim(0, 1)
    ps.style_axes(ax1)

    ax2.plot(sessions, mean_replicas, color=ps.ARM_COLORS["hpa_v2_cpu_memory"], linewidth=1.2)
    ax2.set_ylabel("Mean replicas")
    ax2.set_xlabel("Training session index")
    ps.style_axes(ax2)

    # Figure-level legend placed OUTSIDE the stacked-area axes (above it) --
    # an in-axes legend would sit on top of the filled area itself, since
    # the stack fills the entire 0-1 y-range at every x.
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=lbl) for c, lbl in zip(
            ["#56B4E9", "#CCCCCC", ps.ARM_COLORS["ppo_v2"]], ["SCALE_DOWN", "HOLD", "SCALE_UP"],
        )],
        loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False,
    )

    ps.save_figure(fig, FIGURES_DIR / "P2_action_mix_and_replicas")


# ---------------------------------------------------------------------------
# P3 -- eval_final primary metrics (replica-minutes, mean+-sd, SLA@1000 noted)
# ---------------------------------------------------------------------------


def plot_p3_eval_primary_metrics() -> None:
    run_metrics = av.analyze_evaluation_campaign(EVAL_FINAL_DIR)
    aggregated = av.aggregate_per_profile_arm(run_metrics)
    profiles = sorted({p for p, _ in aggregated})

    fig, ax = ps.new_figure(ps.DOUBLE_COL_WIDTH_IN, 3.6)
    width = 0.25
    max_height = 0.0
    for i, arm_id in enumerate(ps.ARM_IDS):
        means = [aggregated.get((p, arm_id), {}).get("replica_minutes", {}).get("mean") or 0.0 for p in profiles]
        sds = [aggregated.get((p, arm_id), {}).get("replica_minutes", {}).get("sd") or 0.0 for p in profiles]
        slas = [aggregated.get((p, arm_id), {}).get("sla_violation_rate_1000ms", {}).get("mean") for p in profiles]
        x = [j + i * width for j in range(len(profiles))]
        bars = ax.bar(
            x, means, width=width, yerr=sds, capsize=3, label=ps.arm_label(arm_id),
            color=ps.arm_color(arm_id), hatch=ps.arm_hatch(arm_id), edgecolor="white", linewidth=0.5,
        )
        for bar, sla, sd in zip(bars, slas, sds):
            sla_str = f"{sla:.3f}" if sla is not None else "n/a"
            top = bar.get_height() + sd
            max_height = max(max_height, top)
            ax.annotate(
                f"SLA@1000={sla_str}", (bar.get_x() + bar.get_width() / 2, top), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=5.5, rotation=90,
            )
    ax.set_ylim(0, max_height * 1.45)  # headroom for the rotated SLA labels above each bar
    ax.set_xticks([j + width for j in range(len(profiles))])
    ax.set_xticklabels(profiles, rotation=0, ha="center")
    ax.set_ylabel("Replica-minutes")
    ps.style_axes(ax)
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, facecolor=ps.arm_color(a), hatch=ps.arm_hatch(a), edgecolor="white", label=ps.arm_label(a)) for a in ps.ARM_IDS],
        loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False,
    )
    ps.save_figure(fig, FIGURES_DIR / "P3_eval_final_primary_metrics")


# ---------------------------------------------------------------------------
# P4 / P6 -- time-series small multiples (replicas+RPS, p95+SLA line)
# ---------------------------------------------------------------------------


def _time_series_row(ax_replicas, ax_p95, profile: str, campaign_dir: Path, run_picker) -> None:
    for arm_id in ps.ARM_IDS:
        run_dir = run_picker(campaign_dir, profile, arm_id)
        samples = _load_attempt_samples(run_dir)
        t = [s["t_s"] / 60.0 for s in samples]
        replicas = [s["desired_replicas"] for s in samples]
        rps = [s.get("request_rate_rps") or 0.0 for s in samples]
        # A p95 of exactly 0ms with 0 RPS is a genuine "no traffic in this
        # decision window" sample (seen on the final tick of several ppo_v2
        # runs, e.g. the episode's last decision firing after traffic has
        # already stopped) -- it does NOT mean latency was measured as
        # zero. Plotting it literally draws a false cliff down to 0ms;
        # plot a gap (NaN) instead, which is honest about "no measurement"
        # without fabricating or dropping the surrounding real data.
        p95 = [s["p95_latency_ms"] if not (s["p95_latency_ms"] == 0.0 and (s.get("request_rate_rps") or 0.0) == 0.0) else float("nan") for s in samples]

        if arm_id == ps.ARM_IDS[0]:
            ax_rps = ax_replicas.twinx()
            ax_rps.fill_between(t, rps, color="#dddddd", alpha=0.6, linewidth=0, zorder=0)
            ax_rps.set_ylabel("Request rate (RPS)", color="#888888", fontsize=7)
            ax_rps.tick_params(axis="y", labelsize=6, colors="#888888")
            ax_rps.spines["top"].set_visible(False)

        ax_replicas.plot(t, replicas, color=ps.arm_color(arm_id), marker=ps.arm_marker(arm_id), markersize=3, markevery=4, linewidth=1.1, label=ps.arm_label(arm_id), zorder=3)
        ax_p95.plot(t, p95, color=ps.arm_color(arm_id), marker=ps.arm_marker(arm_id), markersize=3, markevery=4, linewidth=1.1, label=ps.arm_label(arm_id))

    ax_p95.axhline(SLA_MS, color=ps.SLA_LINE_COLOR, linestyle="--", linewidth=0.8)
    ax_replicas.set_ylabel("Replicas")
    ax_p95.set_ylabel("p95 latency (ms)")
    ps.style_axes(ax_replicas)
    ps.style_axes(ax_p95)


def plot_p4_eval_time_series_small_multiples() -> None:
    profiles = ["eval_cpu_step", "eval_memory_ramp", "eval_mixed_burst"]
    fig, axes = plt.subplots(3, 2, figsize=(ps.DOUBLE_COL_WIDTH_IN, 7.0), sharex="col")
    for row, profile in enumerate(profiles):
        _time_series_row(axes[row, 0], axes[row, 1], profile, EVAL_FINAL_DIR, _median_cost_run)
        axes[row, 0].set_title(profile, loc="left", fontsize=8, fontweight="bold", y=1.0, pad=2)
    axes[-1, 0].set_xlabel("Time (min)")
    axes[-1, 1].set_xlabel("Time (min)")
    handles = [plt.Line2D([0], [0], color=ps.arm_color(a), marker=ps.arm_marker(a), markersize=4, label=ps.arm_label(a)) for a in ps.ARM_IDS]
    handles.append(plt.Line2D([0], [0], color=ps.SLA_LINE_COLOR, linestyle="--", label="1000 ms SLA"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=7)
    ps.save_figure(fig, FIGURES_DIR / "P4_eval_final_time_series")


def plot_p6_stress_cpu_burst_time_series() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(ps.DOUBLE_COL_WIDTH_IN, 2.6))

    def _single_run_picker(campaign_dir, profile, arm_id):
        return _runs_for(campaign_dir, profile, arm_id)[0]

    _time_series_row(axes[0], axes[1], "stress_cpu_burst", STRESS_DIR, _single_run_picker)
    axes[0].set_xlabel("Time (min)")
    axes[1].set_xlabel("Time (min)")
    handles = [plt.Line2D([0], [0], color=ps.arm_color(a), marker=ps.arm_marker(a), markersize=4, label=ps.arm_label(a)) for a in ps.ARM_IDS]
    handles.append(plt.Line2D([0], [0], color=ps.SLA_LINE_COLOR, linestyle="--", label="1000 ms SLA"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False, fontsize=7)
    ps.save_figure(fig, FIGURES_DIR / "P6_stress_explore_cpu_burst_time_series")


# ---------------------------------------------------------------------------
# P5 -- stress_explore replica-minutes + SLA@1000 (no titles, no error bars)
# ---------------------------------------------------------------------------


def plot_p5_stress_primary_metrics() -> None:
    run_metrics = av.analyze_evaluation_campaign(STRESS_DIR)
    aggregated = av.aggregate_per_profile_arm(run_metrics)
    profiles = sorted({p for p, _ in aggregated})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ps.DOUBLE_COL_WIDTH_IN, 2.8))
    width = 0.25
    for i, arm_id in enumerate(ps.ARM_IDS):
        rm = [aggregated.get((p, arm_id), {}).get("replica_minutes", {}).get("mean") or 0.0 for p in profiles]
        sla = [aggregated.get((p, arm_id), {}).get("sla_violation_rate_1000ms", {}).get("mean") or 0.0 for p in profiles]
        x = [j + i * width for j in range(len(profiles))]
        ax1.bar(x, rm, width=width, color=ps.arm_color(arm_id), hatch=ps.arm_hatch(arm_id), edgecolor="white", linewidth=0.5, label=ps.arm_label(arm_id))
        ax2.bar(x, sla, width=width, color=ps.arm_color(arm_id), hatch=ps.arm_hatch(arm_id), edgecolor="white", linewidth=0.5, label=ps.arm_label(arm_id))

    for ax, ylabel in ((ax1, "Replica-minutes"), (ax2, "SLA-violation rate @1000ms")):
        ax.set_xticks([j + width for j in range(len(profiles))])
        ax.set_xticklabels(profiles, rotation=20, ha="right", fontsize=6.5)
        ax.set_ylabel(ylabel)
        ax.margins(y=0.15)  # headroom so the figure-level legend above never sits on a bar
        ps.style_axes(ax)
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, facecolor=ps.arm_color(a), hatch=ps.arm_hatch(a), edgecolor="white", label=ps.arm_label(a)) for a in ps.ARM_IDS],
        loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False,
    )
    ps.save_figure(fig, FIGURES_DIR / "P5_stress_explore_primary_metrics")


# ---------------------------------------------------------------------------
# P7 -- cost vs latency scatter, eval filled / stress hollow
# ---------------------------------------------------------------------------


def plot_p7_cost_vs_latency_scatter() -> None:
    fig, ax = ps.new_figure(ps.SINGLE_COL_WIDTH_IN, 3.0)
    eval_metrics = av.analyze_evaluation_campaign(EVAL_FINAL_DIR)
    stress_metrics = av.analyze_evaluation_campaign(STRESS_DIR)

    for m in eval_metrics:
        ax.scatter(
            m["replica_minutes"], m["mean_p95_latency_ms"], color=ps.arm_color(m["arm_id"]),
            marker=ps.arm_marker(m["arm_id"]), s=24, facecolors=ps.arm_color(m["arm_id"]), edgecolors="none", zorder=3,
        )
    for m in stress_metrics:
        ax.scatter(
            m["replica_minutes"], m["mean_p95_latency_ms"], marker=ps.arm_marker(m["arm_id"]), s=30,
            facecolors="none", edgecolors=ps.arm_color(m["arm_id"]), linewidths=1.2, zorder=3,
        )

    ax.set_xlabel("Replica-minutes")
    ax.set_ylabel("Mean p95 latency (ms)")
    ps.style_axes(ax)
    handles = [plt.Line2D([0], [0], marker=ps.arm_marker(a), color=ps.arm_color(a), linestyle="", markersize=6, label=ps.arm_label(a)) for a in ps.ARM_IDS]
    handles.append(plt.Line2D([0], [0], marker="o", color="#333333", linestyle="", markerfacecolor="#333333", markersize=5, label="eval_final (filled)"))
    handles.append(plt.Line2D([0], [0], marker="o", color="#333333", linestyle="", markerfacecolor="none", markersize=5, label="stress_explore (hollow)"))
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=6.5)
    ps.save_figure(fig, FIGURES_DIR / "P7_cost_vs_latency_scatter")


# ---------------------------------------------------------------------------
# P8 -- PPO policy map (action share over cpu/memory-per-replica bins)
# ---------------------------------------------------------------------------


def plot_p8_ppo_policy_map() -> None:
    cpu_vals = []
    mem_vals = []
    actions = []
    for run_dir in EVAL_FINAL_DIR.glob("runs/*ppo_v2*"):
        samples = _load_attempt_samples(run_dir)
        for s in samples:
            pd = s.get("ppo_decision")
            if not pd:
                continue
            raw = s.get("raw_observation") or {}
            cpu = raw.get("cpu_per_replica")
            mem = raw.get("memory_per_replica")
            if cpu is None or mem is None:
                continue
            cpu_vals.append(cpu)
            mem_vals.append(mem)
            actions.append(pd["effective_action_name"])

    cpu_vals = np.array(cpu_vals)
    mem_vals = np.array(mem_vals)
    actions = np.array(actions)

    n_bins = 6
    cpu_edges = np.linspace(cpu_vals.min(), cpu_vals.max(), n_bins + 1)
    mem_edges = np.linspace(mem_vals.min(), mem_vals.max(), n_bins + 1)
    cpu_bin = np.clip(np.digitize(cpu_vals, cpu_edges) - 1, 0, n_bins - 1)
    mem_bin = np.clip(np.digitize(mem_vals, mem_edges) - 1, 0, n_bins - 1)

    action_names = sorted(set(actions))  # real logged names only, e.g. HOLD / SCALE_UP_ONE
    up_name = next((a for a in action_names if "UP" in a), None)

    grid = np.full((n_bins, n_bins), np.nan)
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (cpu_bin == i) & (mem_bin == j)
            if mask.sum() == 0:
                continue
            grid[j, i] = (actions[mask] == up_name).mean() if up_name else 0.0

    fig, ax = ps.new_figure(ps.SINGLE_COL_WIDTH_IN, 3.0)
    im = ax.imshow(grid, origin="lower", cmap="Oranges", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([f"{e:.2f}" for e in cpu_edges[:-1]], rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels([f"{e:.2f}" for e in mem_edges[:-1]], fontsize=6)
    ax.set_xlabel("cpu_per_replica (cores)")
    ax.set_ylabel("memory_per_replica (fraction of limit)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(f"Share of '{up_name}' decisions" if up_name else "Share of SCALE_UP decisions", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ps.style_axes(ax)
    ax.grid(False)
    ps.save_figure(fig, FIGURES_DIR / "P8_ppo_policy_map")


def main() -> None:
    ps.apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_p1_training_diagnostics()
    plot_p2_action_mix_and_replicas()
    plot_p3_eval_primary_metrics()
    plot_p4_eval_time_series_small_multiples()
    plot_p5_stress_primary_metrics()
    plot_p6_stress_cpu_burst_time_series()
    plot_p7_cost_vs_latency_scatter()
    plot_p8_ppo_policy_map()
    print("done")


if __name__ == "__main__":
    main()
