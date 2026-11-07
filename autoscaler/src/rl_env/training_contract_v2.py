"""autoscaler/src/rl_env/training_contract_v2.py

V2 Phase 2.1 PART 9: PREPARES (does NOT execute) a separate V2 training
namespace, fully distinct from every frozen V1 path. No training is run by
this module or by this phase -- it defines the paths/constants/metadata
schema a LATER, separately authorized training phase will use, and a
fail-closed guard against accidentally resuming from or overwriting the V1
model.

Distinct from V1:
    V1                                                  V2
    models/ppo/                                         models/ppo_v2/
    persistent-data/rl_training/                         persistent-data/rl_training_v2/
    ppo_training_episode_ledger.jsonl                    ppo_training_episode_ledger_v2.jsonl
    configs/rl.yaml, configs/reward.yaml                 configs/rl_v2.yaml, configs/reward_v2.yaml
    (no profile allowlist concept at this layer)          V2_TRAINING_PROFILE_ALLOWLIST (below)

`autoscaler.src.rl_env.episode_ledger`'s `append_row`/`read_rows`/
`count_rows` are already path-agnostic (take a `ledger_path` argument) --
reused here unmodified, pointed at the V2 ledger path, never the V1 one.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# -- V2 training namespace: distinct directories/files, never overlapping V1 --
V2_MODEL_DIR = PROJECT_ROOT / "models" / "ppo_v2"
V2_MODEL_PATH = V2_MODEL_DIR / "ppo_model_v2.zip"
V2_TRAINING_DATA_DIR = PROJECT_ROOT / "persistent-data" / "rl_training_v2"
V2_TRAINING_LEDGER = V2_TRAINING_DATA_DIR / "ppo_training_episode_ledger_v2.jsonl"
V2_CHECKPOINT_DIR = V2_TRAINING_DATA_DIR / "checkpoints"
V2_TENSORBOARD_DIR = V2_TRAINING_DATA_DIR / "tensorboard"

V1_MODEL_PATH = PROJECT_ROOT / "models" / "ppo" / "ppo_model.zip"
V1_TRAINING_LEDGER = PROJECT_ROOT / "persistent-data" / "rl_training" / "ppo_training_episode_ledger.jsonl"

RL_V2_CONFIG_PATH = PROJECT_ROOT / "configs" / "rl_v2.yaml"
REWARD_V2_CONFIG_PATH = PROJECT_ROOT / "configs" / "reward_v2.yaml"

# V2 profile allowlist: ONLY the V2 multi-resource profiles (configs/
# traffic_profiles/v2/*.yaml) are eligible for V2 training -- mirrors the
# project's existing training/eval firewall pattern (scripts/
# rl_live_common.py's TRAINING_PROFILES / FINAL_HOLDOUT_PROFILE
# allow/deny-list concept) without importing that V1-training-specific
# module, which enforces its OWN, separate V1 profile set.
V2_TRAINING_PROFILE_ALLOWLIST = (
    "constant_cpu",
    "constant_memory",
    "constant_mixed",
    "ramp_mixed",
    "burst_mixed",
)


class V1ArtifactProtectionError(RuntimeError):
    """Raised when V2 training setup would read from or write to a V1
    path. Fail-closed: never silently redirected, never a warning-only."""


def ensure_v2_training_directories() -> None:
    """Creates the V2 training namespace's directories if absent. Never
    touches models/ppo/ or persistent-data/rl_training/ (V1). Idempotent."""
    for d in (V2_MODEL_DIR, V2_TRAINING_DATA_DIR, V2_CHECKPOINT_DIR, V2_TENSORBOARD_DIR):
        d.mkdir(parents=True, exist_ok=True)


def assert_fresh_policy_or_justified(transfer_learning_justification: Optional[str] = None) -> None:
    """Default is FRESH: V2 training must start from a newly-initialized
    policy, never resuming from or warm-starting off the V1 checkpoint,
    UNLESS a separately justified transfer-learning decision is explicitly
    provided as a non-empty string (recorded in the model metadata's
    `transfer_learning_justification` field, never silently assumed)."""
    if transfer_learning_justification is not None and not transfer_learning_justification.strip():
        raise ValueError("transfer_learning_justification must be None or a non-empty, explicit justification string")


def assert_v1_artifacts_untouched(paths_about_to_be_written: List[Path]) -> None:
    """Fail-closed guard: raises if any path V2 training is about to write
    to collides with a frozen V1 path. Intended to be called by the
    (not-yet-written, separately authorized) V2 training entry point
    immediately before any file I/O."""
    protected = {V1_MODEL_PATH.resolve(), V1_TRAINING_LEDGER.resolve()}
    for p in paths_about_to_be_written:
        if p.resolve() in protected:
            raise V1ArtifactProtectionError(f"refusing to write to frozen V1 path: {p}")


def _git_commit_identity() -> Optional[str]:
    """Best-effort -- returns None (never raises) if git is unavailable or
    this is not a git checkout. Metadata completeness must never block on
    an optional provenance field."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=5, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class V2ModelMetadata:
    """Every field the task's PART 9 requires, at minimum. Serializable via
    `to_dict()` for writing alongside a future V2 checkpoint (e.g.
    models/ppo_v2/ppo_model_v2.metadata.json) -- this phase does NOT write
    that file (no training occurs), it only defines the schema and a
    builder function so a later training phase has no ambiguity about what
    must be recorded."""

    controller_generation: str
    observation_contract_hash: str
    reward_contract_hash: str
    workload_contract: str
    training_seed: int
    ppo_hyperparameters: Dict[str, Any]
    safe_inference_epsilon_status: str
    training_profiles: List[str]
    git_commit: Optional[str] = None
    transfer_learning_justification: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_v2_model_metadata(
    *,
    training_seed: int,
    ppo_hyperparameters: Dict[str, Any],
    training_profiles: Optional[List[str]] = None,
    transfer_learning_justification: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> V2ModelMetadata:
    """Builds the required metadata record for a (not-yet-trained) V2
    model. `training_profiles` defaults to the full V2_TRAINING_PROFILE_
    ALLOWLIST; passing a subset is allowed but every entry MUST already be
    in the allowlist (fail-closed, never silently accepts an unlisted
    profile name)."""
    assert_fresh_policy_or_justified(transfer_learning_justification)

    profiles = list(training_profiles) if training_profiles is not None else list(V2_TRAINING_PROFILE_ALLOWLIST)
    not_allowed = [p for p in profiles if p not in V2_TRAINING_PROFILE_ALLOWLIST]
    if not_allowed:
        raise ValueError(f"training_profiles contains non-allowlisted entries: {not_allowed}")

    return V2ModelMetadata(
        controller_generation="v2",
        observation_contract_hash=_sha256_file(RL_V2_CONFIG_PATH),
        reward_contract_hash=_sha256_file(REWARD_V2_CONFIG_PATH),
        workload_contract="multi_resource_v2",
        training_seed=training_seed,
        ppo_hyperparameters=dict(ppo_hyperparameters),
        safe_inference_epsilon_status="provisional_unvalidated",
        training_profiles=profiles,
        git_commit=_git_commit_identity(),
        transfer_learning_justification=transfer_learning_justification,
        extra=extra or {},
    )
