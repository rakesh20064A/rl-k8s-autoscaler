"""scripts/rl_live_common.py

Phase 16V: shared, importable helpers for the direct-PPO live-training
runner (`scripts/train_ppo_live.py`) and final-evaluation harness
(`scripts/evaluate_ppo_vs_hpa.py`). Deliberately free of any
Kubernetes/Prometheus/subprocess call itself (mirrors
`scripts/dynamics_calibration_lib.py`'s own "pure/network-free" convention)
so it can be exercised entirely offline in tests.

Holds:
  - the training/evaluation workload firewall (phase_16u_training_and_
    evaluation_workload_split.json's separation, enforced in code)
  - frozen-hyperparameter loading (phase_16u_ppo_hyperparameters.json)
  - governing-artifact hash verification (Phase16V task sec.1)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_OUTPUTS = PROJECT_ROOT / "task_outputs"

PHASE16U_PROTOCOL_YAML = TASK_OUTPUTS / "phase_16u_direct_ppo_protocol.yaml"
PHASE16U_REWARD_YAML = TASK_OUTPUTS / "phase_16u_reward_specification.yaml"
PHASE16U_WORKLOAD_SPLIT_JSON = TASK_OUTPUTS / "phase_16u_training_and_evaluation_workload_split.json"
PHASE16U_PPO_HYPERPARAMS_JSON = TASK_OUTPUTS / "phase_16u_ppo_hyperparameters.json"
PHASE16U_SAFETY_YAML = TASK_OUTPUTS / "phase_16u_safety_and_fallback_protocol.yaml"
PHASE16U_MANIFEST_JSON = TASK_OUTPUTS / "phase_16u_manifest.json"

TRAINING_PROFILES = ("constant", "step", "burst", "ramp", "periodic", "mixed")
FINAL_HOLDOUT_PROFILE = "iteration_rl_final_holdout_v1"
EXCLUDED_FROM_FORMAL_COMPARISON = ("unseen_eval",)


class WorkloadFirewallViolation(RuntimeError):
    """Raised whenever code attempts to use a workload outside its
    preregistered role. Must never be caught-and-ignored by a caller --
    this exception existing at all means a scientific-integrity boundary
    was about to be crossed."""


def assert_training_eligible(profile_name: str) -> None:
    """Raises WorkloadFirewallViolation unless `profile_name` is one of the
    six preregistered training/development profiles. The final-holdout
    workload and unseen_eval.yaml must NEVER be reachable from the training
    code path."""
    if profile_name == FINAL_HOLDOUT_PROFILE:
        raise WorkloadFirewallViolation(
            f"{profile_name!r} is the FINAL HELD-OUT evaluation workload -- "
            "it must never be used during training (phase_16u_training_and_"
            "evaluation_workload_split.json)."
        )
    if profile_name in EXCLUDED_FROM_FORMAL_COMPARISON:
        raise WorkloadFirewallViolation(
            f"{profile_name!r} is excluded from the formal comparison entirely "
            "(docs/rl_mdp_spec.md sec.22) and must never be used during training."
        )
    if profile_name not in TRAINING_PROFILES:
        raise WorkloadFirewallViolation(
            f"{profile_name!r} is not one of the preregistered training profiles {TRAINING_PROFILES}."
        )


def assert_final_evaluation_eligible(profile_name: str) -> None:
    """Raises WorkloadFirewallViolation unless `profile_name` is the single
    preregistered final-holdout workload. A training profile must never be
    passed to the final-evaluation code path pretending to be the holdout."""
    if profile_name != FINAL_HOLDOUT_PROFILE:
        raise WorkloadFirewallViolation(
            f"{profile_name!r} is not the preregistered final-holdout workload "
            f"({FINAL_HOLDOUT_PROFILE!r}). The formal PPO-vs-HPA comparison may "
            "only use the frozen final-holdout workload."
        )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_ppo_hyperparameters() -> Dict[str, Any]:
    with open(PHASE16U_PPO_HYPERPARAMS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def load_workload_split() -> Dict[str, Any]:
    with open(PHASE16U_WORKLOAD_SPLIT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_governing_hashes() -> List[Dict[str, Any]]:
    """Recomputes SHA-256 of every Phase16U protocol artifact and the frozen
    upstream contract/config files, and compares against the values
    recorded in phase_16u_manifest.json. Returns a list of
    {path, expected, actual, match} -- never raises on mismatch (the caller
    decides whether a mismatch is fatal), so this can be used for reporting
    as well as gating."""
    with open(PHASE16U_MANIFEST_JSON, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    checks: List[Dict[str, Any]] = []

    for rel_path, expected in manifest.get("output_files", {}).items():
        path = TASK_OUTPUTS / rel_path
        actual = sha256_file(path) if path.exists() else None
        checks.append({"path": str(path.relative_to(PROJECT_ROOT)), "expected": expected, "actual": actual, "match": actual == expected})

    for rel_path, expected in manifest.get("read_only_inputs_verified_unchanged", {}).items():
        path = PROJECT_ROOT / rel_path
        actual = sha256_file(path) if path.exists() else None
        checks.append({"path": rel_path, "expected": expected, "actual": actual, "match": actual == expected})

    frozen_contract = manifest.get("frozen_contract_adopted_unchanged", {})
    contract_path = PROJECT_ROOT / frozen_contract.get("document", "docs/rl_mdp_spec.md")
    contract_expected = frozen_contract.get("sha256")
    contract_actual = sha256_file(contract_path) if contract_path.exists() else None
    checks.append({"path": str(contract_path.relative_to(PROJECT_ROOT)), "expected": contract_expected, "actual": contract_actual, "match": contract_actual == contract_expected})

    return checks


def all_governing_hashes_match() -> bool:
    return all(c["match"] for c in verify_governing_hashes())
