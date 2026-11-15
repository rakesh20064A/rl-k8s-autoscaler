#!/usr/bin/env python3
"""scripts/generate_v2_phase6c_diagrams.py

V2 Phase 6C: architecture/pipeline diagrams, rendered with matplotlib
(Graphviz `dot` was checked for and is not installed on this machine; no
system-wide installation was performed). Writes SVG + PNG to docs/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GREY = "#8a8a86"
RED = "#e34948"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

diagrams_manifest = []


def _save(fig, name: str, caption: str, takeaway: str) -> None:
    svg_path = FIGURES_DIR / f"{name}.svg"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    diagrams_manifest.append({"file": f"{name}.svg / {name}.png", "caption": caption, "takeaway": takeaway})
    print(f"wrote {png_path}")


def _box(ax, xy, w, h, text, color, fontsize=9, text_color="white"):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=color, edgecolor="black", linewidth=0.8,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, color=text_color, wrap=True)
    return rect


def _arrow(ax, p1, p2, color="black", style="-|>", lw=1.3, connectionstyle="arc3,rad=0.0", label=None):
    arrow = FancyArrowPatch(p1, p2, arrowstyle=style, color=color, lw=lw, connectionstyle=connectionstyle, mutation_scale=14)
    ax.add_patch(arrow)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.08, label, ha="center", fontsize=7.5, color="#333333")


# ---------------------------------------------------------------------------
# 1. System architecture
# ---------------------------------------------------------------------------


def system_architecture_diagram():
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    ax.set_title("V2 system architecture: Minikube cluster + load generator + PPO agent", fontsize=13)

    # Minikube cluster boundary
    cluster = mpatches.FancyBboxPatch((0.3, 0.3), 7.6, 7.6, boxstyle="round,pad=0.05", facecolor="none", edgecolor="#555555", linewidth=1.5, linestyle="--")
    ax.add_patch(cluster)
    ax.text(0.5, 7.6, "Minikube cluster", fontsize=11, color="#555555", weight="bold")

    _box(ax, (0.8, 6.1), 2.9, 1.0, "demo-app\nDeployment\n(2-8 replicas)", BLUE)
    _box(ax, (4.3, 6.1), 2.9, 1.0, "HPA\n(CPU+mem / CPU-only)", ORANGE)
    _box(ax, (0.8, 4.5), 2.9, 1.0, "Metrics Server\n(--metric-resolution=15s)", GREY)
    _box(ax, (4.3, 4.5), 2.9, 1.0, "Prometheus\n(10s scrape)", GREY)
    _box(ax, (0.8, 2.9), 2.9, 1.0, "Kubernetes API\n(scale, get, apply)", "#444444")
    _box(ax, (4.3, 2.9), 2.9, 1.0, "Controller lock file\nactive_controller.json", "#444444")
    _box(ax, (0.8, 1.0), 6.4, 1.0, "demo-app-loadtest Service (NodePort tunnel)", GREY)

    # External: load generator + relay (top right)
    _box(ax, (9.0, 6.1), 3.4, 1.0, "Load generator + relay\n(loadgen/traffic_generator.py)", AQUA)

    # PPO agent (bottom right)
    _box(ax, (9.0, 1.0), 3.4, 4.4, "", "#f2f2f0", text_color="black")
    ax.text(10.7, 5.1, "PPO agent", fontsize=10, weight="bold", ha="center")
    _box(ax, (9.2, 3.85), 3.0, 0.75, "Observation builder\n(11-feature vector)", BLUE, fontsize=8)
    _box(ax, (9.2, 2.9), 3.0, 0.75, "Policy (PPO)", BLUE, fontsize=8)
    _box(ax, (9.2, 1.95), 3.0, 0.75, "Safety layer\n(tie-break R1-R6)", BLUE, fontsize=8)
    _box(ax, (9.2, 1.05), 3.0, 0.75, "Executor\n(scale via K8s API)", BLUE, fontsize=8)

    # -- straight, non-crossing arrows only --
    _arrow(ax, (4.25, 6.6), (4.3, 6.6), label="scale target")
    _arrow(ax, (2.25, 6.1), (2.25, 5.5))
    _arrow(ax, (5.75, 6.1), (5.75, 5.5))
    _arrow(ax, (2.25, 4.5), (2.25, 3.9))
    _arrow(ax, (5.75, 4.5), (5.75, 3.9))

    # loadtest Service -> load generator (HTTP requests), right along the bottom then up
    _arrow(ax, (7.2, 1.5), (9.0, 1.5), label="HTTP requests", connectionstyle="arc3,rad=0.0")
    # load generator -> loadtest Service (traffic), a single clean arc above the cluster box
    _arrow(ax, (9.0, 6.4), (7.2, 7.5), color=GREY, label="traffic", connectionstyle="arc3,rad=-0.25")

    # PPO internal pipeline
    _arrow(ax, (10.7, 3.85), (10.7, 3.65))
    _arrow(ax, (10.7, 2.9), (10.7, 2.7))
    _arrow(ax, (10.7, 1.95), (10.7, 1.8))

    # metrics -> observation builder (clean horizontal-ish line at a dedicated y)
    _arrow(ax, (7.2, 5.0), (9.2, 4.2), color="#333333", label="metrics", connectionstyle="arc3,rad=-0.1")
    # executor -> Kubernetes API (scale command), dedicated lower path
    _arrow(ax, (9.2, 1.3), (3.7, 2.9), color="#333333", label="scale command", connectionstyle="arc3,rad=0.15")
    # both controllers <-> lock file: single bidirectional-style arrow from lock file up to PPO safety layer area
    _arrow(ax, (7.2, 3.4), (9.2, 2.3), color="#333333", label="controller lock", connectionstyle="arc3,rad=-0.1")

    fig.tight_layout()
    _save(
        fig, "diagram_system_architecture",
        "System architecture: Minikube cluster (demo-app, HPA, Metrics Server, Prometheus, controller lock) plus the external load generator/relay and PPO agent (observation builder -> policy -> safety layer -> executor).",
        "The PPO agent and each HPA arm are mutually exclusive controllers of the same demo-app deployment, enforced by the controller lock file -- never both active at once.",
    )


# ---------------------------------------------------------------------------
# 2. RL control loop
# ---------------------------------------------------------------------------


def rl_control_loop_diagram():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("RL control loop (closed loop, one decision per 30s interval)", fontsize=13)

    positions = {
        "obs": (5, 8.6),
        "policy": (8.3, 6.8),
        "action": (8.3, 4.2),
        "safety": (5, 2.4),
        "k8s": (1.7, 4.2),
        "metrics": (1.7, 6.8),
    }
    labels = {
        "obs": "11-feature\nobservation",
        "policy": "PPO policy\n(action probabilities)",
        "action": "Action\n(DOWN / HOLD / UP)",
        "safety": "Safety gating\n(tie-break R1-R6)",
        "k8s": "Kubernetes API\n(scale deployment)",
        "metrics": "Metrics\n(CPU, memory, latency, RPS)",
    }
    colors = {"obs": BLUE, "policy": ORANGE, "action": AQUA, "safety": RED, "k8s": "#444444", "metrics": GREY}
    for key, (x, y) in positions.items():
        _box(ax, (x - 1.1, y - 0.55), 2.2, 1.1, labels[key], colors[key])

    order = ["obs", "policy", "action", "safety", "k8s", "metrics", "obs"]
    for a, b in zip(order[:-1], order[1:]):
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        _arrow(ax, (x1, y1), (x2, y2), connectionstyle="arc3,rad=0.15")

    ax.text(5, 0.6, "7-component reward (reliability, SLA, pressure, waste, action, reversal, boundary)\ncomputed alongside this loop, drives PPO training only -- not part of inference-time control", ha="center", fontsize=9, color="#333333")

    fig.tight_layout()
    _save(
        fig, "diagram_rl_control_loop",
        "The RL control loop: 11-feature observation -> PPO policy -> action -> safety gating -> Kubernetes API -> metrics -> back to observation, one cycle per 30-second decision interval.",
        "Safety gating sits between the policy's raw action and the actual Kubernetes API call, so an unsafe/tied decision is corrected before it ever reaches the cluster.",
    )


# ---------------------------------------------------------------------------
# 3. Safety-aware tie-break decision flow (R1-R6)
# ---------------------------------------------------------------------------


def tie_break_flow_diagram():
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_title("Safety-aware tie-break decision flow (CORR-58)", fontsize=13)

    _box(ax, (3.5, 9.6), 3.0, 1.0, "Action probabilities\nfrom PPO policy", BLUE)
    _box(ax, (3.5, 8.0), 3.0, 1.0, "Tie set = actions within\nepsilon of the max", GREY)

    _box(ax, (0.3, 6.2), 2.6, 1.1, "R1: exactly 1 action\nin tie set\n-> that action", "#444444")
    _box(ax, (3.7, 6.2), 2.6, 1.1, "DOWN in tie set?", ORANGE)
    _box(ax, (7.1, 6.2), 2.6, 1.1, "Tie set ==\n{HOLD, UP}", AQUA)

    _box(ax, (2.0, 4.2), 3.0, 1.0, "R4: {DOWN,UP} or\n{DOWN,HOLD,UP} -> HOLD", RED)
    _box(ax, (5.3, 4.2), 3.0, 1.0, "R5: {DOWN,HOLD}\n-> HOLD", RED)

    _box(ax, (7.1, 3.0), 2.6, 1.0, "stressed is None?", GREY)
    _box(ax, (5.5, 1.3), 2.0, 1.0, "R6: unknown\nstress -> HOLD", RED)
    _box(ax, (8.0, 1.3), 2.0, 1.0, "R3: UP if stressed\nelse HOLD", AQUA)

    _arrow(ax, (5, 9.6), (5, 9.0))
    _arrow(ax, (5, 8.0), (1.6, 7.3), label="size==1")
    _arrow(ax, (5, 8.0), (5, 7.3), label="size>1")
    _arrow(ax, (5, 6.2), (8.4, 6.9), label="no", connectionstyle="arc3,rad=-0.3")
    _arrow(ax, (5.0, 6.2), (3.5, 5.2), label="DOWN,UP both")
    _arrow(ax, (5.0, 6.2), (6.0, 5.2), label="DOWN,HOLD only")
    _arrow(ax, (8.4, 6.2), (8.4, 4.0))
    _arrow(ax, (8.4, 3.0), (6.5, 2.3), label="yes")
    _arrow(ax, (8.4, 3.0), (9.0, 2.3), label="no")

    ax.text(5, 0.2, "DOWN never wins a tie (R2, structural guarantee): every tie set containing DOWN resolves via R4 or R5, both HOLD.", ha="center", fontsize=9, color="#333333")

    fig.tight_layout()
    _save(
        fig, "diagram_tie_break_flow",
        "Decision flow for the safety-aware tie-break rule (R1-R6, CORR-58): from the PPO action-probability tie set to the final selected action.",
        "SCALE_DOWN can never win a tie by construction -- only R3's {HOLD,UP} branch (gated on decision-time stress) is sensitive to live cluster state; every other branch resolves to a fixed, safe default.",
    )


# ---------------------------------------------------------------------------
# 4. Project / experiment pipeline
# ---------------------------------------------------------------------------


def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.set_title("Project / experiment pipeline", fontsize=13)

    stages = [
        ("V1\n(single-metric\nbaseline)", GREY),
        ("V2 design\n(multi-metric,\nsafety-aware)", BLUE),
        ("48-session\ntraining\n(final_min2)", ORANGE),
        ("Pre-registered\nevaluation\n(eval_final, CORR-60)", AQUA),
        ("Exploratory\nstress test\n(stress_explore, CORR-65)", RED),
    ]
    x = 0.4
    w = 2.2
    centers = []
    for text, color in stages:
        _box(ax, (x, 1.0), w, 1.4, text, color, fontsize=9)
        centers.append(x + w / 2)
        x += w + 0.4
    for i in range(len(centers) - 1):
        _arrow(ax, (centers[i] + w / 2, 1.7), (centers[i + 1] - w / 2, 1.7))

    ax.text(centers[2], 0.5, "CORR-58/60-62", ha="center", fontsize=8, color="#333333")
    ax.text(centers[3], 0.5, "CORR-60, 63, 64", ha="center", fontsize=8, color="#333333")
    ax.text(centers[4], 0.5, "CORR-65", ha="center", fontsize=8, color="#333333")

    fig.tight_layout()
    _save(
        fig, "diagram_experiment_pipeline",
        "Project/experiment pipeline: V1 -> V2 design -> 48-session training -> pre-registered evaluation (eval_final) -> exploratory stress test (stress_explore), with key CORR milestone numbers annotated.",
        "The exploratory stress test is a separate, later stage that never feeds back into or revises the pre-registered evaluation's own results.",
    )


def main():
    system_architecture_diagram()
    rl_control_loop_diagram()
    tie_break_flow_diagram()
    pipeline_diagram()
    return diagrams_manifest


if __name__ == "__main__":
    import json

    result = main()
    print(json.dumps({"n_diagrams": len(result)}, indent=2))
