# Part 11 — HPA and Controller Exclusivity

FROM_CODE, drawn from Phase 4B-R4E/R4F's direct, line-cited investigation
(unchanged since — no git diff on any of these files per Part 05).

## Every `demo-app-hpa`/`demo-app-hpa-v2` reference

- `autoscaler/src/rl_env/live_backend.py:64` —
  `DEFAULT_HPA_NAME = "demo-app-hpa"`.
- `autoscaler/src/rl_env/live_backend.py:103` —
  `hpa_manifest_path: str = "k8s/hpa/hpa.yaml"` (the actual default used by
  `restore_hpa()`/`delete_hpa()`).
- `scripts/train_ppo_final_v2.py:492` — `checks["canonical_hpa_present"] =
  "demo-app-hpa" in hpa_names`.
- `scripts/train_ppo_final_v2.py:822` — `unexpected_hpas = [h for h in
  existing_hpas if h != "demo-app-hpa"]`.
- `scripts/train_ppo_live_v2.py:157,296,311` — same `"demo-app-hpa"`-named
  canonical-HPA pre-check pattern.
- `scripts/run_dynamics_calibration_slot.py:133` — `HPA_NAME =
  "demo-app-hpa"` (V1 historical).
- `autoscaler/src/rl_env/controller_exclusivity.py:58-59` — models BOTH
  identities: `CONTROLLER_CPU_HPA: "demo-app-hpa"`,
  `CONTROLLER_CPU_MEMORY_HPA: "demo-app-hpa-v2"`.
- `scripts/evaluate_ppo_vs_hpa_v2.py:26-27` — refuses to start "while any
  HPA object (CPU-only `demo-app-hpa` or the new CPU+memory
  `demo-app-hpa-v2`) is present" — the only script that explicitly
  name-checks BOTH identities for its own exclusivity refusal, though it
  still never restores/applies `demo-app-hpa-v2` itself (see below).
- `tests/unit/test_controller_exclusivity.py` and
  `tests/unit/test_rl_env_live_backend.py:295,297` — both parametrize over
  `["demo-app-hpa", "demo-app-hpa-v2"]`, confirming the exclusivity design
  is tested against both names.
- `tests/unit/test_hpa_v2_manifest.py:47` — asserts
  `v2["metadata"]["name"] == "demo-app-hpa-v2"`, but this test only checks
  the static YAML file's own declared name, plus a pure-function
  reproduction of Kubernetes' "largest recommendation across metrics"
  algorithm — **zero live-cluster testing of `demo-app-hpa-v2` actually
  scaling anything has ever occurred.**

## Which HPA each V2 script actually restores/validates at runtime

**`demo-app-hpa` (V1 CPU-only), and ONLY `demo-app-hpa`** — confirmed: no
call site anywhere passes a different `hpa_manifest_path` to
`LiveKubernetesBackend`; `train_ppo_final_v2.py`/`train_ppo_live_v2.py`'s
own preflight/canonical checks are hardcoded to `"demo-app-hpa"`. The
entire V2 pilot and final-training campaign has, every single time it has
ever run live, removed and restored `demo-app-hpa` — `demo-app-hpa-v2` has
**never** been applied to the live cluster by any training/evaluation
script in this project's history. It exists only as a written manifest +
a unit-tested pure-function algorithm reproduction + the exclusivity
design's name list.

## `controller_exclusivity.py` identities and lock semantics

Three controller identities tracked: `CONTROLLER_PPO`, `CONTROLLER_CPU_HPA`
(→"demo-app-hpa"), `CONTROLLER_CPU_MEMORY_HPA` (→"demo-app-hpa-v2").
`acquire_controller(CONTROLLER_PPO, ...)` refuses unconditionally while
EITHER HPA object is present in the cluster (detected via
`list_hpa_names()`). Lock state is persisted to a local JSON file,
`persistent-data/controller/active_controller.json` — VERIFIED this
session: `{"active_controller": "none", "updated_at_utc":
"2026-10-12T07:53:39.620859+00:00"}` (unchanged since Phase 4B-R2,
confirming no PPO session has held the lock since then).

## Clean-separation recommendation (from Phase 4B-R4F, unresolved)

V1 `demo-app-hpa` should remain the frozen baseline, restored only by V1
scripts (already true). For V2, an EXPLICIT, named `hpa_manifest_path`
parameter should be threaded through so a future "PPO vs. V2 HPA" fairness
comparison (Option A, per R4F Section 6) has a real episode-runner that
applies and restores `demo-app-hpa-v2` specifically — this does not
currently exist anywhere in the codebase. `P1`, unresolved as of this
snapshot.
