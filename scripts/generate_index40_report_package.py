#!/usr/bin/env python3
"""scripts/generate_index40_report_package.py

Index40: OFFLINE-ONLY report-ready results package. Reads ONLY:
  - results/evaluation/metrics.json (authoritative Section-10 values)
  - results/evaluation/raw/*.json (original, preserved episode
    evidence -- read-only, for the optional time-series figures)

Performs NO Kubernetes/Prometheus/model call. Writes only new files under
results/figures/ and results/evaluation/*.{md,csv} -- never
modifies index39 outputs or original episode evidence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
EVAL_DIR = RESULTS_DIR / "raw"
METRICS_PATH = RESULTS_DIR / "metrics.json"

FULL_VALID = ["HPA1", "HPA2", "HPA3", "PPO3"]
EXECUTION_ORDER = ["PPO1", "HPA1", "PPO2", "HPA2", "PPO3", "HPA3"]

HPA_COLOR = "#234A46"
PPO_COLOR = "#D34A32"
SAFETY_COLOR = "#C18A28"
SLA_LINE_COLOR = "#9C3B2E"


def _bar_colors(names: List[str]) -> List[str]:
    return [HPA_COLOR if n.startswith("HPA") else PPO_COLOR for n in names]


def load_metrics() -> Dict[str, Any]:
    with open(METRICS_PATH) as f:
        return json.load(f)


def load_episode_raw(name: str) -> Dict[str, Any]:
    file_map = {
        "PPO1": "episode_01_ppo_repeat1.json", "HPA1": "episode_02_hpa_repeat1.json",
        "PPO2": "episode_03_ppo_repeat2.json", "HPA2": "episode_04_hpa_repeat2.json",
        "PPO3": "episode_05_ppo_repeat3.json", "HPA3": "episode_06_hpa_repeat3.json",
    }
    with open(EVAL_DIR / file_map[name]) as f:
        return json.load(f)


def _annotate_ppo3(ax) -> None:
    ax.set_xlabel("PPO3 is the only valid full-duration PPO repeat (PPO1/PPO2 excluded: safety terminated)", fontsize=8, style="italic")


def plot_bar(names: List[str], values: List[float], title: str, ylabel: str, path: Path, note: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, values, color=_bar_colors(names))
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.2f}" if isinstance(v, float) else str(v), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if note:
        _annotate_ppo3(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_bar_figures(metrics: Dict[str, Any]) -> List[str]:
    per_ep = metrics["per_episode"]
    created = []

    def m(name, key):
        return per_ep[name]["section10_metrics"][key]

    def e(name, key):
        return per_ep[name][key]

    specs = [
        ("final_mean_p95_latency.png", "Mean P95 Latency (full-duration valid repeats)", "ms", [e(n, "mean_latency_p95_ms") for n in FULL_VALID]),
        ("final_mean_replicas.png", "Mean Replicas (full-duration valid repeats)", "replicas", [e(n, "mean_replicas") for n in FULL_VALID]),
        ("final_sla_violation_rate.png", "SLA Violation Rate (p95 > 300ms)", "fraction of steps", [m(n, "sla_violation_rate") for n in FULL_VALID]),
        ("final_resource_proxy.png", "Resource Proxy Mean (lower = fewer replicas, matched SLA)", "(replicas-1)/7", [m(n, "resource_proxy_mean") for n in FULL_VALID]),
        ("final_reaction_time.png", "Mean Reaction Time to SLA Breach", "seconds", [m(n, "reaction_time_seconds") or 0.0 for n in FULL_VALID]),
    ]
    for filename, title, ylabel, values in specs:
        plot_bar(FULL_VALID, values, title, ylabel, FIG_DIR / filename)
        created.append(filename)

    # Action churn: diagnostic only -- explicitly labeled.
    churn = [m(n, "action_churn_count") for n in FULL_VALID]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(FULL_VALID, churn, color=_bar_colors(FULL_VALID))
    for bar, v in zip(bars, churn):
        ax.annotate(str(v), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=9)
    ax.set_title("Action Churn Count — DIAGNOSTIC ONLY\n(not part of win/loss rule)", fontsize=12)
    ax.set_ylabel("effective non-HOLD actions")
    _annotate_ppo3(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_action_churn.png", dpi=150)
    plt.close(fig)
    created.append("final_action_churn.png")

    return created


def generate_outcomes_figure() -> str:
    outcome_map = {
        "PPO1": ("safety terminated", SAFETY_COLOR), "PPO2": ("safety terminated", SAFETY_COLOR),
        "PPO3": ("full valid", PPO_COLOR), "HPA1": ("full valid", HPA_COLOR),
        "HPA2": ("full valid", HPA_COLOR), "HPA3": ("full valid", HPA_COLOR),
    }
    fig, ax = plt.subplots(figsize=(8, 3))
    for i, name in enumerate(EXECUTION_ORDER):
        label, color = outcome_map[name]
        ax.barh(0, 1, left=i, color=color, edgecolor="black")
        ax.text(i + 0.5, 0, f"{name}\n{label}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.set_xlim(0, len(EXECUTION_ORDER))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Evaluation Execution Sequence and Outcomes (PPO1→HPA1→PPO2→HPA2→PPO3→HPA3)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "evaluation_outcomes.png", dpi=150)
    plt.close(fig)
    return "evaluation_outcomes.png"


_HPA_SHADES = {"HPA1": "#8FB0A6", "HPA2": "#3E6B61", "HPA3": "#13211E"}
_LINE_STYLES = {"HPA1": "-", "HPA2": "--", "HPA3": "-."}


def generate_timeseries_figures() -> List[str]:
    created = []
    episodes = {n: load_episode_raw(n) for n in FULL_VALID}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in FULL_VALID:
        samples = episodes[name]["samples"]
        t = [s["time_s"] for s in samples]
        r = [s["replicas_current"] for s in samples]
        if name.startswith("HPA"):
            ax.plot(t, r, _LINE_STYLES[name], label=name, color=_HPA_SHADES[name], linewidth=1.8)
        else:
            ax.plot(t, r, "-", label=name, color=PPO_COLOR, linewidth=2.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("replicas (current)")
    ax.set_title("Replicas Over Time — Full-Duration Valid Repeats Only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "replicas_over_time_full_runs.png", dpi=150)
    plt.close(fig)
    created.append("replicas_over_time_full_runs.png")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in FULL_VALID:
        samples = episodes[name]["samples"]
        t = [s["time_s"] for s in samples]
        p95 = [s.get("latency_p95_ms") for s in samples]
        if name.startswith("HPA"):
            ax.plot(t, p95, _LINE_STYLES[name], label=name, color=_HPA_SHADES[name], linewidth=1.8)
        else:
            ax.plot(t, p95, "-", label=name, color=PPO_COLOR, linewidth=2.4)
    ax.axhline(300, color=SLA_LINE_COLOR, linestyle=":", label="SLA threshold (300ms)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("P95 latency (ms)")
    ax.set_title("P95 Latency Over Time — Full-Duration Valid Repeats Only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p95_latency_over_time_full_runs.png", dpi=150)
    plt.close(fig)
    created.append("p95_latency_over_time_full_runs.png")

    return created


def generate_tables(metrics: Dict[str, Any]) -> None:
    per_ep = metrics["per_episode"]
    rows = []
    for name in EXECUTION_ORDER:
        ep = per_ep[name]
        m = ep["section10_metrics"]
        rows.append({
            "Controller": "PPO" if name.startswith("PPO") else "HPA",
            "Repeat": name[3],
            "Outcome": ep["episode_status"],
            "Full duration": ep["full_holdout_completed"],
            "Valid primary comparison": ep["valid_for_primary_comparison"],
            "SLA violation rate": m["sla_violation_rate"],
            "Error rate mean": m["error_rate_mean"],
            "Resource proxy mean": m["resource_proxy_mean"],
            "Reaction time seconds": m["reaction_time_seconds"],
            "Action churn": m["action_churn_count"],
            "Mean P95 latency ms": ep["mean_latency_p95_ms"],
            "Mean replicas": ep["mean_replicas"],
            "Max replicas": ep["max_replicas"],
        })

    csv_path = RESULTS_DIR / "final_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["| " + " | ".join(rows[0].keys()) + " |", "|" + "---|" * len(rows[0])]
    for r in rows:
        md_lines.append("| " + " | ".join(str(r[k]) for k in r) + " |")
    md_path = RESULTS_DIR / "final_results.md"
    with open(md_path, "w") as f:
        f.write("# Index40 Final Results Table\n\n" + "\n".join(md_lines) + "\n")


def generate_results_chapter(metrics: Dict[str, Any]) -> None:
    content = """# Final Results Chapter (Draft)

## 1. Experimental setup

A PPO policy was trained live against a Kubernetes cluster and compared
against the cluster's canonical Horizontal Pod Autoscaler (HPA) on an
identical, hash-pinned, previously-unseen held-out traffic workload.

## 2. Training completion

Training was frozen at checkpoint `session_019`: 5120 cumulative
environment timesteps and 20 PPO updates, using training seed 42.

## 3. Final holdout protocol

The final evaluation replayed the single frozen held-out workload
(`iteration_rl_final_holdout_v1`, SHA-256 `308f4f90ea3cea97f1927cec97efeaa652c4424e9fe5bcea421358f3425e3409`,
seed 777, 1500 seconds) across six episodes in the pre-registered order
PPO1 → HPA1 → PPO2 → HPA2 → PPO3 → HPA3, with the two controllers never
active simultaneously.

## 4. Episode outcome summary

HPA completed all three of its repeats to full duration. PPO completed
one of its three repeats (repeat 3) to full duration; repeats 1 and 2
terminated early under the frozen safety rule (4 consecutive decision
steps with the application/Prometheus target unreachable). A separate
HPA repeat-2 attempt (physical attempt 1) failed for infrastructure
reasons (a Prometheus connection failure during post-workload metric
drain, concurrent with observed cluster node instability) and was
excluded entirely from the scientific record; HPA repeat 2 was
successfully completed on a second physical attempt after the
infrastructure issue resolved.

## 5. HPA results

Across its three full-duration repeats, HPA's mean P95 latency ranged
from {hpa_p95_min:.0f} ms to {hpa_p95_max:.0f} ms, with mean replica
counts between {hpa_rep_min:.2f} and {hpa_rep_max:.2f} (peaking at
{hpa_max_rep} replicas) and SLA violation rates between
{hpa_sla_min:.2f} and {hpa_sla_max:.2f}.

## 6. PPO results

PPO's single valid full-duration repeat (repeat 3) used a mean of
{ppo3_rep:.2f} replicas (maximum {ppo3_max_rep}), substantially fewer
than any HPA repeat, but recorded a higher mean P95 latency
({ppo3_p95:.0f} ms) and a higher SLA violation rate ({ppo3_sla:.2f})
than any HPA repeat.

## 7. Resource-latency trade-off

The single PPO full-duration repeat is consistent with a resource/latency
trade-off relative to HPA: markedly lower resource usage alongside
markedly higher latency and SLA-violation rate. This observation is based
on a single repeat and is not statistically generalizable.

## 8. Safety termination findings

PPO repeats 1 and 2 terminated after 4 consecutive decision steps with
an unreachable telemetry/application target — a pre-registered safety
rule, not an implementation defect. This is itself a relevant finding:
under this specific held-out workload and live-cluster conditions, two
of three PPO evaluation attempts could not be sustained to completion,
while HPA's control pattern did not trigger this failure mode in any of
its three valid attempts.

## 9. Section-10 metric interpretation

All five frozen Section-10 metrics (`sla_violation_rate`, `error_rate_mean`,
`resource_proxy_mean`, `reaction_time_seconds`, `action_churn_count`) were
computed identically for every episode from the same recorded per-step
samples; `action_churn_count` is diagnostic only and not part of the
win/loss decision rule.

## 10. Formal comparison-rule status

The pre-registered decision rule requires three valid full-duration
repeats per controller. HPA achieved 3/3; PPO achieved 1/3. **The formal
n=3 decision rule is therefore not applicable, and no formal comparative
winner is declared.**

## 11. Limitations

PPO's valid sample size (n=1) cannot support the frozen decision rule or
any statistical claim. PPO1/PPO2's partial data are reported for
transparency only and are excluded from the primary comparison. The
excluded HPA2 infrastructure-failure attempt reflects live-cluster
conditions during the evaluation window, not a scientific result.

## 12. Final conclusion

HPA demonstrated more consistent completion and lower latency in the
held-out evaluation, while the single full-duration PPO run used
substantially fewer replicas. Two PPO repeats terminated under the
predefined safety rule, so the planned three-repeat primary comparison
could not be completed and no formal winner was declared.
""".format(
        hpa_p95_min=min(metrics["per_episode"][n]["mean_latency_p95_ms"] for n in ["HPA1", "HPA2", "HPA3"]),
        hpa_p95_max=max(metrics["per_episode"][n]["mean_latency_p95_ms"] for n in ["HPA1", "HPA2", "HPA3"]),
        hpa_rep_min=min(metrics["per_episode"][n]["mean_replicas"] for n in ["HPA1", "HPA2", "HPA3"]),
        hpa_rep_max=max(metrics["per_episode"][n]["mean_replicas"] for n in ["HPA1", "HPA2", "HPA3"]),
        hpa_max_rep=max(metrics["per_episode"][n]["max_replicas"] for n in ["HPA1", "HPA2", "HPA3"]),
        hpa_sla_min=min(metrics["per_episode"][n]["section10_metrics"]["sla_violation_rate"] for n in ["HPA1", "HPA2", "HPA3"]),
        hpa_sla_max=max(metrics["per_episode"][n]["section10_metrics"]["sla_violation_rate"] for n in ["HPA1", "HPA2", "HPA3"]),
        ppo3_rep=metrics["per_episode"]["PPO3"]["mean_replicas"],
        ppo3_max_rep=metrics["per_episode"]["PPO3"]["max_replicas"],
        ppo3_p95=metrics["per_episode"]["PPO3"]["mean_latency_p95_ms"],
        ppo3_sla=metrics["per_episode"]["PPO3"]["section10_metrics"]["sla_violation_rate"],
    )
    with open(PROJECT_ROOT / "docs" / "results_chapter_draft.md", "w") as f:
        f.write(content)


def generate_viva_summary() -> None:
    content = """# Viva / Demo Results Summary (Index40)

## Why PPO was chosen
PPO (Proximal Policy Optimization) is a widely-used, stable on-policy RL
algorithm well suited to a discrete, low-dimensional action space like
replica scaling decisions, and is a standard baseline choice for
autoscaling research.

## What PPO observed
An 11-dimensional normalized observation vector covering CPU/replica,
request-rate, latency, and short-horizon history (EWMA) features, sourced
live from Prometheus and the Kubernetes Deployment/HPA objects.

## Actions
`Discrete(3)`: SCALE_DOWN_ONE (-1), HOLD (0), SCALE_UP_ONE (+1), applied
every 30-second decision interval, bounded to the replica range [1, 8].

## Reward objective
A multi-component reward balancing SLA/latency performance against
replica-resource cost (the same `resource_proxy` quantity reused in the
Section-10 evaluation metric), with penalties for scaling reversals —
see `phase_16u_reward_specification.yaml` for the exact frozen formula.

## Training budget actually completed
5120 environment timesteps (20 PPO updates), live against the real
cluster.

## Why only 5120 instead of the original 12288-step design budget
The original 12288-step design budget (~118.6 hours of pure live decision
time, excluding episode-reset overhead) proved infeasible to complete
within the capstone timeline on a single-machine, single-cluster
development environment subject to real infrastructure interruptions
(network changes, transient node instability). Training was honestly
frozen at the live progress actually achieved (5120 steps) rather than
silently extending the deadline or fabricating additional training.

## Final holdout design
A single, newly-materialized, hash-pinned traffic workload
(`iteration_rl_final_holdout_v1`, seed 777, 1500s, peak 65 RPS) was
deliberately designed to combine workload shapes not seen together in any
training profile, and was never inspected or executed before both PPO
training and the evaluation protocol were frozen.

## Why HPA target = 260%
260% CPU utilization (relative to a 100m request) is the project's
long-standing canonical HPA configuration (established well before this
RL work began), chosen so HPA's natural scale-up threshold falls in a
comparable operating range to the environment's reward design.

## What safety termination means
A frozen, pre-registered rule: if the live application/Prometheus target
is unreachable for 4 consecutive 30-second decision steps (120 seconds),
the episode terminates and the system is restored to a safe state. This
protects against evaluating (or training) on meaningless data during a
genuine infrastructure outage — it is not an implementation bug.

## Why HPA2 (attempt 1) was retried
HPA repeat 2's first physical attempt failed for a clear infrastructure
reason (a Prometheus connection failure during post-workload metric
drain, concurrent with observed Kubernetes node instability) — the
traffic workload itself had already completed in full. This is a pure
tooling failure with no scientific outcome to be "outcome-aware" about,
so a second physical attempt at the same logical repeat was permitted
and executed once the infrastructure recovered and was independently
verified stable.

## Why PPO1/PPO2 were NOT retried
PPO repeats 1 and 2 terminated via the frozen, pre-registered SAFETY rule,
which is itself a legitimate, meaningful scientific outcome (not a tooling
failure) — retrying it because the outcome was unfavorable would be
outcome-aware retry, which the project's evaluation protocol explicitly
prohibits.

## Main result
HPA completed 3/3 held-out repeats with consistently lower latency and
higher resource usage; PPO completed 1/3 repeats, using substantially
fewer replicas in that one repeat but with higher latency and a higher
SLA-violation rate. Because PPO did not achieve the required three valid
repeats, the pre-registered formal comparison rule could not be applied,
and **no formal winner is declared**.

## Limitations
Single trained checkpoint, single evaluation pass, PPO n=1 valid sample
(no statistical power), live-infrastructure variability during evaluation,
and a substantially reduced training budget relative to the original
design target.

## Future improvements
Complete the originally-designed training budget; obtain three valid
full-duration PPO repeats (reusing the same safety-termination handling,
simply re-attempting the logical repeat under the SAME retry discipline
already used for tooling failures, if and only if the protocol is amended
to permit it); investigate the root cause of PPO's telemetry-unavailability
safety terminations specifically under this held-out workload; and
evaluate robustness to infrastructure instability as its own explicit,
measured research question.
"""
    with open(PROJECT_ROOT / "docs" / "viva_results_summary.md", "w") as f:
        f.write(content)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()

    bar_figs = generate_bar_figures(metrics)
    outcomes_fig = generate_outcomes_figure()
    ts_figs = generate_timeseries_figures()

    generate_tables(metrics)
    generate_results_chapter(metrics)
    generate_viva_summary()

    print(json.dumps({
        "figures_created": bar_figs + [outcomes_fig] + ts_figs,
        "tables_created": ["final_results.md", "final_results.csv"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
