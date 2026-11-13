#!/usr/bin/env python3
"""scripts/analyze_v2_evaluation.py

V2 Phase 5B PART C: analysis tooling for (A) the pre-registered PPO-vs-HPA
evaluation (CORR-60) and (B) the V2 final_min2 training curve. A NEW file
-- read-only against `persistent-data/rl_training_v2/` and
`persistent-data/evaluation_v2/`; writes only under `persistent-data/traffic/`
or a caller-supplied path. Does not modify any of the 43
FREEZE_MANIFEST.json files.

PART A (evaluation): operates on a NORMALIZED per-run series schema (see
`RunSeries`/`normalize_run_series_from_dict()` below) rather than directly
on `scripts/evaluate_v2_campaign.py`'s raw per-arm output shapes
(`run_live_v2_episode()`'s decision records vs. `run_hpa_v2_episode()`'s
replica samples are two DIFFERENT raw shapes today) -- normalizing those
two into this one common schema so PPO and HPA runs are measured
IDENTICALLY (the pre-registration's own requirement) is an explicit,
disclosed Phase 5C gap (see this phase's report); this script's own
metric/aggregation/comparison logic is complete and tested now against
synthetic fixtures in the normalized schema, so Phase 5C only needs to
write the two small adapters, not design the analysis itself.

PART B (training curve): operates directly on the REAL, already-written
`persistent-data/rl_training_v2/final_min2/sessions/*/step_telemetry.jsonl`
and `ppo_update_metrics.jsonl` -- run now, read-only, on whatever sessions
have completed so far.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SESSIONS_DIR = PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final_min2" / "sessions"
DEFAULT_UPDATE_METRICS_PATH = PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final_min2" / "ppo_update_metrics.jsonl"

ACTION_NAMES = {0: "SCALE_DOWN", 1: "HOLD", 2: "SCALE_UP"}
SLA_PRIMARY_MS = 1000.0
SLA_REFERENCE_MS = 300.0

# ---------------------------------------------------------------------------
# PART A -- evaluation analysis
# ---------------------------------------------------------------------------


@dataclass
class RunSample:
    t_s: float
    p95_latency_ms: Optional[float] = None
    error_rate: Optional[float] = None
    memory_per_replica_mib: Optional[float] = None
    replicas: Optional[int] = None
    action: Optional[str] = None  # one of ACTION_NAMES.values(), or None for an HPA sample
    reversal: bool = False
    invalid: bool = False


@dataclass
class RunSeries:
    """The common, normalized per-run shape every arm's raw output must be
    converted into (Phase 5C adapters) before `compute_run_metrics()` can
    treat PPO and HPA runs identically."""

    run_id: str
    profile: str
    arm_id: str
    repeat_index: int
    decision_interval_s: float
    samples: List[RunSample]
    pod_restarts: int = 0
    load_change_times_s: List[float] = field(default_factory=list)


def normalize_run_series_from_dict(data: Dict[str, Any]) -> RunSeries:
    samples = [RunSample(**s) for s in data["samples"]]
    return RunSeries(
        run_id=data["run_id"], profile=data["profile"], arm_id=data["arm_id"],
        repeat_index=data["repeat_index"], decision_interval_s=data.get("decision_interval_s", 30.0),
        samples=samples, pod_restarts=data.get("pod_restarts", 0),
        load_change_times_s=data.get("load_change_times_s", []),
    )


def compute_reaction_time_s(series: RunSeries) -> Optional[float]:
    """Seconds from the nearest preceding `load_change_times_s` entry to
    the next non-HOLD action at or after it. None if there is no declared
    load change, or no scaling action ever follows one."""
    if not series.load_change_times_s:
        return None
    reaction_times: List[float] = []
    for change_t in series.load_change_times_s:
        for sample in series.samples:
            if sample.t_s >= change_t and sample.action not in (None, "HOLD") and not sample.invalid:
                reaction_times.append(sample.t_s - change_t)
                break
    if not reaction_times:
        return None
    return statistics.mean(reaction_times)


def compute_run_metrics(series: RunSeries) -> Dict[str, Any]:
    valid = [s for s in series.samples if not s.invalid]
    valid_latency = [s.p95_latency_ms for s in valid if s.p95_latency_ms is not None]
    valid_error = [s.error_rate for s in valid if s.error_rate is not None]
    valid_memory = [s.memory_per_replica_mib for s in valid if s.memory_per_replica_mib is not None]

    def _violation_rate(threshold: float) -> Optional[float]:
        if not valid_latency:
            return None
        return sum(1 for v in valid_latency if v >= threshold) / len(valid_latency)

    replica_minutes = sum((s.replicas or 0) * series.decision_interval_s / 60.0 for s in valid)
    scaling_actions = sum(1 for s in valid if s.action not in (None, "HOLD"))
    reversals = sum(1 for s in valid if s.reversal)

    return {
        "run_id": series.run_id, "profile": series.profile, "arm_id": series.arm_id,
        "repeat_index": series.repeat_index,
        "sla_violation_rate_1000ms": _violation_rate(SLA_PRIMARY_MS),
        "sla_violation_rate_300ms_reference": _violation_rate(SLA_REFERENCE_MS),
        "replica_minutes": replica_minutes,
        "mean_p95_latency_ms": statistics.mean(valid_latency) if valid_latency else None,
        "median_p95_latency_ms": statistics.median(valid_latency) if valid_latency else None,
        "error_rate": statistics.mean(valid_error) if valid_error else None,
        "max_per_pod_memory_mib": max(valid_memory) if valid_memory else None,
        "scaling_actions_count": scaling_actions,
        "reversals_count": reversals,
        "reaction_time_s": compute_reaction_time_s(series),
        "pod_restarts_count": series.pod_restarts,
        "invalid_samples_count": sum(1 for s in series.samples if s.invalid),
        "n_samples": len(series.samples),
    }


PRIMARY_METRICS = ("sla_violation_rate_1000ms", "replica_minutes")
SECONDARY_METRICS = (
    "mean_p95_latency_ms", "median_p95_latency_ms", "error_rate", "max_per_pod_memory_mib",
    "scaling_actions_count", "reversals_count", "reaction_time_s", "pod_restarts_count", "invalid_samples_count",
)
REFERENCE_METRICS = ("sla_violation_rate_300ms_reference",)
ALL_METRICS = PRIMARY_METRICS + SECONDARY_METRICS + REFERENCE_METRICS

# Metrics where a LOWER value is better (used by the "better in all 3
# repeats" comparison). Every pre-registered metric here is a cost/
# violation-style metric -- none is "higher is better".
LOWER_IS_BETTER = {m: True for m in ALL_METRICS}


def aggregate_per_profile_arm(run_metrics: List[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    """Keyed by (profile, arm_id); each value has mean/sd per metric over
    the repeats present for that (profile, arm_id)."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for m in run_metrics:
        groups.setdefault((m["profile"], m["arm_id"]), []).append(m)

    aggregated: Dict[tuple, Dict[str, Any]] = {}
    for key, rows in groups.items():
        agg: Dict[str, Any] = {"n_repeats": len(rows)}
        for metric in ALL_METRICS:
            values = [r[metric] for r in rows if r[metric] is not None]
            if not values:
                agg[metric] = {"mean": None, "sd": None, "n": 0}
                continue
            agg[metric] = {
                "mean": statistics.mean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        aggregated[key] = agg
    return aggregated


def is_better_in_all_repeats(values_a: Sequence[float], values_b: Sequence[float], lower_is_better: bool = True) -> Optional[bool]:
    """Arm A is 'better' than arm B on this metric ONLY if, paired by
    repeat index, A beats B in EVERY repeat -- per the pre-registration's
    decision rule. Returns None if the repeat counts don't match (cannot
    pair) or either side has no data."""
    if not values_a or not values_b or len(values_a) != len(values_b):
        return None
    if lower_is_better:
        return all(a < b for a, b in zip(values_a, values_b))
    return all(a > b for a, b in zip(values_a, values_b))


def compare_arms_per_profile(run_metrics: List[Dict[str, Any]], arm_a: str, arm_b: str) -> Dict[str, Dict[str, Any]]:
    """Per profile, per metric: whether `arm_a` is better than `arm_b` in
    all repeats, the reverse, or neither (a genuine trade-off / no
    decidable winner)."""
    by_profile: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for m in run_metrics:
        by_profile.setdefault(m["profile"], {}).setdefault(m["arm_id"], []).append(m)

    result: Dict[str, Dict[str, Any]] = {}
    for profile, by_arm in by_profile.items():
        rows_a = sorted(by_arm.get(arm_a, []), key=lambda r: r["repeat_index"])
        rows_b = sorted(by_arm.get(arm_b, []), key=lambda r: r["repeat_index"])
        profile_result: Dict[str, Any] = {}
        for metric in ALL_METRICS:
            values_a = [r[metric] for r in rows_a if r[metric] is not None]
            values_b = [r[metric] for r in rows_b if r[metric] is not None]
            lower_is_better = LOWER_IS_BETTER[metric]
            a_better = is_better_in_all_repeats(values_a, values_b, lower_is_better)
            b_better = is_better_in_all_repeats(values_b, values_a, lower_is_better)
            if a_better:
                verdict = f"{arm_a}_better"
            elif b_better:
                verdict = f"{arm_b}_better"
            else:
                verdict = "no_decidable_winner"
            profile_result[metric] = verdict
        result[profile] = profile_result
    return result


def write_evaluation_tables(aggregated: Dict[tuple, Dict[str, Any]], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "evaluation_summary.csv"
    md_path = output_dir / "evaluation_summary.md"

    rows = []
    for (profile, arm_id), agg in sorted(aggregated.items()):
        row = {"profile": profile, "arm_id": arm_id, "n_repeats": agg["n_repeats"]}
        for metric in ALL_METRICS:
            row[f"{metric}_mean"] = agg[metric]["mean"]
            row[f"{metric}_sd"] = agg[metric]["sd"]
        rows.append(row)

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = ["# V2 Evaluation Summary (mean ± sd over repeats)", ""]
    for (profile, arm_id), agg in sorted(aggregated.items()):
        lines.append(f"## {profile} / {arm_id} (n_repeats={agg['n_repeats']})")
        lines.append("")
        lines.append("| metric | mean | sd |")
        lines.append("|---|---|---|")
        for metric in ALL_METRICS:
            mean, sd = agg[metric]["mean"], agg[metric]["sd"]
            mean_str = f"{mean:.4f}" if mean is not None else "n/a"
            sd_str = f"{sd:.4f}" if sd is not None else "n/a"
            lines.append(f"| {metric} | {mean_str} | {sd_str} |")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"csv": csv_path, "md": md_path}


def plot_evaluation_figures(aggregated: Dict[tuple, Dict[str, Any]], output_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    profiles = sorted({profile for profile, _ in aggregated})

    for metric in PRIMARY_METRICS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        width = 0.25
        arm_ids = sorted({arm_id for _, arm_id in aggregated})
        for i, arm_id in enumerate(arm_ids):
            means = [aggregated.get((p, arm_id), {}).get(metric, {}).get("mean") or 0.0 for p in profiles]
            sds = [aggregated.get((p, arm_id), {}).get(metric, {}).get("sd") or 0.0 for p in profiles]
            x = [j + i * width for j in range(len(profiles))]
            ax.bar(x, means, width=width, yerr=sds, label=arm_id, capsize=3)
        ax.set_xticks([j + width for j in range(len(profiles))])
        ax.set_xticklabels(profiles, rotation=20, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"V2 evaluation: {metric} by profile/arm")
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"{metric}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        paths.append(path)

    return paths


def analyze_evaluation_campaign(campaign_dir: Path) -> List[Dict[str, Any]]:
    """Reads every VALID run's normalized series (from
    `<run_dir>/normalized_series.json`, the Phase 5C adapter output -- see
    module docstring) and returns the per-run metrics list. Runs with no
    normalized series yet (raw-only) are skipped, never guessed at."""
    runs_dir = campaign_dir / "runs"
    run_metrics: List[Dict[str, Any]] = []
    if not runs_dir.exists():
        return run_metrics
    for run_dir in sorted(runs_dir.iterdir()):
        series_path = run_dir / "normalized_series.json"
        if not series_path.exists():
            continue
        with open(series_path, encoding="utf-8") as f:
            series = normalize_run_series_from_dict(json.load(f))
        run_metrics.append(compute_run_metrics(series))
    return run_metrics


# ---------------------------------------------------------------------------
# PART B -- training-curve analysis (real data, read-only)
# ---------------------------------------------------------------------------


def find_completed_training_sessions(sessions_dir: Path) -> List[Path]:
    """A session directory counts as completed only if it has BOTH
    `ppo_model.zip` and `metadata.json` AND no sibling `SESSION_FAILED.json`
    -- a failed physical attempt (e.g. `session_004/`, retried as
    `session_004__attempt2/`) can still carry a stale metadata.json/
    ppo_model.zip pair from a checkpoint flush before the failure was
    detected; `campaign_metadata.json`'s own `sessions_completed` list
    (the authoritative resume record) never includes the failed attempt,
    only its successful retry -- this mirrors that exclusion."""
    if not sessions_dir.exists():
        return []
    return sorted(
        d for d in sessions_dir.iterdir()
        if d.is_dir() and (d / "ppo_model.zip").is_file() and (d / "metadata.json").is_file()
        and not (d / "SESSION_FAILED.json").exists()
    )


def load_update_metrics_by_session(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    by_session: Dict[int, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_session[row["session_index"]] = row
    return by_session


def summarize_session_telemetry(session_dir: Path) -> Dict[str, Any]:
    telemetry_path = session_dir / "step_telemetry.jsonl"
    rows = []
    with open(telemetry_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    reward_totals = [r["reward_total"] for r in rows]
    component_names = sorted(rows[0]["reward_components"].keys()) if rows else []
    component_means = {
        name: statistics.mean(r["reward_components"][name] for r in rows) for name in component_names
    }
    action_counts = {name: 0 for name in ACTION_NAMES.values()}
    for r in rows:
        action_counts[ACTION_NAMES[r["effective_action"]]] += 1
    replica_counts: Dict[int, int] = {}
    for r in rows:
        replicas = r["deployment_current_replicas_after"]
        replica_counts[replicas] = replica_counts.get(replicas, 0) + 1

    return {
        "n_steps": len(rows),
        "mean_step_reward": statistics.mean(reward_totals) if reward_totals else None,
        "reward_component_means": component_means,
        "action_mix_counts": action_counts,
        "replica_distribution_counts": replica_counts,
        "mean_replicas": statistics.mean(r["deployment_current_replicas_after"] for r in rows) if rows else None,
    }


def analyze_training_curve(
    sessions_dir: Path = DEFAULT_SESSIONS_DIR, update_metrics_path: Path = DEFAULT_UPDATE_METRICS_PATH,
) -> List[Dict[str, Any]]:
    completed = find_completed_training_sessions(sessions_dir)
    update_metrics_by_session = load_update_metrics_by_session(update_metrics_path)

    curve: List[Dict[str, Any]] = []
    for session_dir in completed:
        # "session_004__attempt2" -> scientific session index 4 (the
        # retry's own data, not the failed attempt 1's) -- strip any
        # "__attemptN" suffix before parsing the numeric index.
        base_name = session_dir.name.split("__attempt")[0]
        session_index = int(base_name.split("_")[-1])
        summary = summarize_session_telemetry(session_dir)
        update_metrics = update_metrics_by_session.get(session_index, {})
        curve.append({
            "session_index": session_index,
            "entropy_loss": update_metrics.get("entropy_loss"),
            "approx_kl": update_metrics.get("approx_kl"),
            "clip_fraction": update_metrics.get("clip_fraction"),
            "explained_variance": update_metrics.get("explained_variance"),
            "value_loss": update_metrics.get("value_loss"),
            **summary,
        })
    return curve


def write_training_curve_outputs(curve: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_curve.csv"
    md_path = output_dir / "training_curve.md"

    if curve:
        fieldnames = [
            "session_index", "n_steps", "entropy_loss", "approx_kl", "clip_fraction",
            "explained_variance", "value_loss", "mean_step_reward", "mean_replicas",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in curve:
                writer.writerow({k: row.get(k) for k in fieldnames})

    lines = ["# V2 Training Curve Preview (read-only, sessions completed so far)", ""]
    lines.append("| session | entropy_loss | approx_kl | clip_fraction | explained_variance | value_loss | mean_step_reward | mean_replicas |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in curve:
        lines.append(
            f"| {row['session_index']} | {row['entropy_loss']} | {row['approx_kl']} | {row['clip_fraction']} | "
            f"{row['explained_variance']} | {row['value_loss']} | {row['mean_step_reward']:.4f} | {row['mean_replicas']:.3f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    figure_paths = plot_training_curve_figures(curve, output_dir)
    return {"csv": csv_path, "md": md_path, "figures": figure_paths}


def plot_training_curve_figures(curve: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    if not curve:
        return paths

    sessions = [row["session_index"] for row in curve]

    for metric in ("entropy_loss", "approx_kl", "clip_fraction", "explained_variance", "value_loss", "mean_step_reward", "mean_replicas"):
        values = [row.get(metric) for row in curve]
        if all(v is None for v in values):
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sessions, values, marker="o")
        ax.set_xlabel("session_index")
        ax.set_ylabel(metric)
        ax.set_title(f"V2 training curve: {metric}")
        fig.tight_layout()
        path = output_dir / f"{metric}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        paths.append(path)

    return paths


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    eval_parser = sub.add_parser("evaluation")
    eval_parser.add_argument("--campaign-dir", required=True)
    eval_parser.add_argument("--output-dir", required=True)

    curve_parser = sub.add_parser("training-curve")
    curve_parser.add_argument("--sessions-dir", default=str(DEFAULT_SESSIONS_DIR))
    curve_parser.add_argument("--update-metrics-path", default=str(DEFAULT_UPDATE_METRICS_PATH))
    curve_parser.add_argument("--output-dir", required=True)

    args = parser.parse_args(argv)

    if args.mode == "evaluation":
        run_metrics = analyze_evaluation_campaign(Path(args.campaign_dir))
        aggregated = aggregate_per_profile_arm(run_metrics)
        output_dir = Path(args.output_dir)
        tables = write_evaluation_tables(aggregated, output_dir)
        figures = plot_evaluation_figures(aggregated, output_dir)
        print(json.dumps({"n_runs_analyzed": len(run_metrics), "tables": {k: str(v) for k, v in tables.items()}, "figures": [str(p) for p in figures]}, indent=2))
    else:
        curve = analyze_training_curve(Path(args.sessions_dir), Path(args.update_metrics_path))
        outputs = write_training_curve_outputs(curve, Path(args.output_dir))
        print(json.dumps({"n_sessions_analyzed": len(curve), **{k: (str(v) if not isinstance(v, list) else [str(p) for p in v]) for k, v in outputs.items()}}, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
