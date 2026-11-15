#!/usr/bin/env python3
"""scripts/generate_v2_paper_tables.py

V2 Phase 6D item 7 (paper additions): tables T1-T4, generated
programmatically from the real config/data files below -- never
hand-typed. Read-only against configs/, autoscaler/, persistent-data/;
writes only under docs/tables/.

T1 observation features: configs/rl_v2.yaml's own `observation_v2` block
   (name/description/raw_min/raw_max/min/max/source/temporal), the single
   source of truth per that file's own header note -- cross-checked
   against autoscaler/src/rl_env/observation.py's FEATURE_NAMES tuple
   (same 11 names, same order).
T2 reward components: configs/reward_v2.yaml's `weights`/`sla`/`pressure`/
   `waste`/`action`/`reversal`/`boundary` blocks.
T3 PPO hyperparameters + training budget: configs/rl_v2.yaml's `ppo:`
   block, plus the budget actually used for the final_min2 campaign
   (persistent-data/rl_training_v2/final_min2/EVALUATION_PREREGISTRATION.json's
   training_contract_reference: 48 sessions, 12,288 total timesteps --
   NOT configs/rl_v2.yaml's own generic `training.total_timesteps: 500000`,
   which is a different, unrelated default never used by this campaign).
T4 experimental design: arms (scripts/paper_style.py's canonical labels),
   profiles + peak RPS (recomputed from each profile's own YAML, the same
   way scripts/generate_v2_phase6d_figures.py's P0 does, so the numbers
   can never drift from what P0 actually plots), repeats, and the
   pre-registration's decision rule.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.paper_style as ps  # noqa: E402
from loadgen.traffic_generator import parse_traffic_profile  # noqa: E402

TABLES_DIR = PROJECT_ROOT / "docs" / "tables"
RL_V2_PATH = PROJECT_ROOT / "configs" / "rl_v2.yaml"
REWARD_V2_PATH = PROJECT_ROOT / "configs" / "reward_v2.yaml"
PREREGISTRATION_PATH = (
    PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final_min2" / "EVALUATION_PREREGISTRATION.json"
)
STRESS_PLAN_PATH = PROJECT_ROOT / "persistent-data" / "rl_training_v2" / "final_min2" / "EXPLORATORY_STRESS_PLAN.json"
EVAL_PROFILE_DIR = PROJECT_ROOT / "configs" / "traffic_profiles" / "v2_eval"
STRESS_PROFILE_DIR = PROJECT_ROOT / "configs" / "traffic_profiles" / "v2_stress"


def _write_table(rows: List[Dict[str, Any]], columns: List[str], out_path_no_ext: Path) -> None:
    out_path_no_ext.parent.mkdir(parents=True, exist_ok=True)

    with open(f"{out_path_no_ext}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        md_lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    Path(f"{out_path_no_ext}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    def _tex_escape(s: str) -> str:
        return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    tex_lines = [
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\hline",
        " & ".join(_tex_escape(c) for c in columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        tex_lines.append(" & ".join(_tex_escape(row[c]) for c in columns) + r" \\")
    tex_lines += [r"\hline", r"\end{tabular}"]
    Path(f"{out_path_no_ext}.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def generate_t1_observation_features() -> None:
    with open(RL_V2_PATH, encoding="utf-8") as f:
        rl_v2 = yaml.safe_load(f)
    entries = rl_v2["observation_v2"]
    assert len(entries) == 11, f"expected 11 observation features, found {len(entries)}"

    rows = []
    for e in entries:
        rows.append({
            "index": e["index"],
            "name": e["name"],
            "meaning": e["description"],
            "raw_range": f"[{e['raw_min']}, {e['raw_max']}]",
            "normalized_range": f"[{e['min']}, {e['max']}]",
            "temporal": e["temporal"],
        })
    _write_table(rows, ["index", "name", "meaning", "raw_range", "normalized_range", "temporal"], TABLES_DIR / "T1_observation_features")


def generate_t2_reward_components() -> None:
    with open(REWARD_V2_PATH, encoding="utf-8") as f:
        reward_v2 = yaml.safe_load(f)
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
        "w_reliability": "n/a (reliability term has no separate threshold)",
        "w_sla": f"p95_target_ms={reward_v2['sla']['p95_target_ms']}",
        "w_pressure": (
            f"cpu_safe_high={reward_v2['pressure']['cpu_safe_high']}, cpu_hard_limit={reward_v2['pressure']['cpu_hard_limit']}, "
            f"memory_safe_high={reward_v2['pressure']['memory_safe_high']}, memory_hard_limit={reward_v2['pressure']['memory_hard_limit']}"
        ),
        "w_waste": (
            f"cpu_waste_threshold={reward_v2['waste']['cpu_waste_threshold']}, "
            f"memory_waste_threshold={reward_v2['waste']['memory_waste_threshold']}"
        ),
        "w_action": f"per_action_cost={reward_v2['action']['per_action_cost']}",
        "w_reversal": (
            f"reversal_window_seconds={reward_v2['reversal']['reversal_window_seconds']}, "
            f"per_reversal_cost={reward_v2['reversal']['per_reversal_cost']}"
        ),
        "w_boundary": f"per_boundary_violation_cost={reward_v2['boundary']['per_boundary_violation_cost']}",
    }
    rows = []
    for name, weight in weights.items():
        rows.append({
            "component": name, "weight": weight, "meaning": meanings[name], "threshold_params": thresholds[name],
        })
    assert len(rows) == 7, f"expected 7 reward components, found {len(rows)}"
    _write_table(rows, ["component", "weight", "meaning", "threshold_params"], TABLES_DIR / "T2_reward_components")


def generate_t3_ppo_hyperparameters() -> None:
    with open(RL_V2_PATH, encoding="utf-8") as f:
        rl_v2 = yaml.safe_load(f)
    ppo = rl_v2["ppo"]

    with open(PREREGISTRATION_PATH, encoding="utf-8") as f:
        prereg = json.load(f)
    n_sessions = prereg["training_contract_reference"]["total_planned_sessions"]
    total_timesteps = prereg["training_contract_reference"]["total_timesteps_budget"]
    steps_per_session = total_timesteps // n_sessions

    rows = [{"parameter": k, "value": v} for k, v in ppo.items()]
    rows.append({"parameter": "training_budget", "value": f"{n_sessions} sessions x {steps_per_session} steps/session = {total_timesteps} total timesteps (final_min2 campaign, not configs/rl_v2.yaml's own generic training.total_timesteps default)"})
    _write_table(rows, ["parameter", "value"], TABLES_DIR / "T3_ppo_hyperparameters")


def _peak_rps(profile) -> float:
    total = sum(ph.duration_seconds for ph in profile.phases)
    t = np.arange(0.0, total, 1.0)
    return max(profile.get_phase_and_target_rps(float(ti))[0] for ti in t)


def generate_t4_experimental_design() -> None:
    with open(PREREGISTRATION_PATH, encoding="utf-8") as f:
        prereg = json.load(f)
    with open(STRESS_PLAN_PATH, encoding="utf-8") as f:
        stress_plan = json.load(f)

    rows = []
    for arm_id in ps.ARM_IDS:
        rows.append({"arm": ps.arm_label(arm_id), "profiles_run_on": "all 3 eval + all 3 stress", "role": {
            "ppo_v2": "RL policy under test",
            "hpa_v2_cpu_memory": "resource-metric baseline (CPU+memory)",
            "hpa_v2_cpu_only": "Kubernetes-default baseline (CPU only)",
        }[arm_id]})

    eval_profile_names = [Path(p).stem for p in prereg["workloads"]["profiles"]]
    for name in eval_profile_names:
        profile = parse_traffic_profile(EVAL_PROFILE_DIR / f"{name}.yaml")
        rows.append({
            "campaign": "eval_final (pre-registered, CORR-60)", "profile": name,
            "resource_mode": profile.resource_mode, "peak_rps": _peak_rps(profile),
            "repeats": prereg["workloads"]["repeats_per_profile_per_arm"],
        })
    for p in stress_plan["design"]["profiles"]:
        name = p["name"]
        profile = parse_traffic_profile(STRESS_PROFILE_DIR / f"{name}.yaml")
        rows.append({
            "campaign": "stress_explore (exploratory, CORR-65)", "profile": name,
            "resource_mode": profile.resource_mode, "peak_rps": _peak_rps(profile),
            "repeats": stress_plan["runner"]["repeats_per_profile_per_arm"],
        })

    arm_rows = [r for r in rows if "arm" in r]
    profile_rows = [r for r in rows if "profile" in r]
    _write_table(arm_rows, ["arm", "profiles_run_on", "role"], TABLES_DIR / "T4a_experimental_design_arms")
    _write_table(profile_rows, ["campaign", "profile", "resource_mode", "peak_rps", "repeats"], TABLES_DIR / "T4b_experimental_design_profiles")

    decision_rule_rows = [
        {"rule": "reporting", "statement": prereg["decision_rule"]["reporting"]},
        {"rule": "claiming_better", "statement": prereg["decision_rule"]["claiming_better"]},
        {"rule": "no_aggregate_winner", "statement": prereg["decision_rule"]["no_aggregate_winner"]},
        {"rule": "applies_to", "statement": "eval_final only (3 repeats/cell). Does NOT apply to stress_explore (1 repeat/cell, descriptive only, CORR-65)."},
    ]
    _write_table(decision_rule_rows, ["rule", "statement"], TABLES_DIR / "T4c_experimental_design_decision_rule")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    generate_t1_observation_features()
    generate_t2_reward_components()
    generate_t3_ppo_hyperparameters()
    generate_t4_experimental_design()
    print("done")


if __name__ == "__main__":
    main()
