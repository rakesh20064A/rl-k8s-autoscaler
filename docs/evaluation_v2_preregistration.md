# V2 Evaluation Pre-Registration (CORR-60 / Phase 5A)

Created 2026-11-08T10:07:46Z, **before any evaluation run and before
session_047 exists** (training is still live under the amended 48-session
budget, CORR-59). Machine-readable copy:
`persistent-data/rl_training_v2/final_min2/EVALUATION_PREREGISTRATION.json`.

## Model under test

Only the **48-update final checkpoint (session_047)**. Intermediate
checkpoints (session_000 through session_046) are used solely for
training-curve analysis and are never a candidate for the formal
PPO-vs-HPA comparison itself.

## Arms

| Arm | Controller | HPA object / manifest | Replica bounds |
|---|---|---|---|
| **PPO-V2** | `ppo` | n/a (direct `kubectl scale`) | 2-8 |
| **HPA-V2 CPU+memory** | `cpu_memory_hpa` | `demo-app-hpa-v2` / `k8s/hpa/hpa_v2_cpu_memory.yaml` | 2-8 |
| **HPA-CPU-only** | not yet defined (Phase 5B gap) | `demo-app-hpa-v2-cpu` / `k8s/hpa/hpa_v2_cpu_only_min2.yaml` | 2-8 |

All three arms: the same `demo-app` image, Metrics Server resolution
verified live at 15 s immediately before every HPA-arm run (never
assumed), Grafana scaled to 0 replicas.

## Workloads

The 3 new held-out evaluation profiles (`configs/traffic_profiles/v2_eval/`):
`eval_cpu_step`, `eval_memory_ramp`, `eval_mixed_burst` — none is a member
of the training allowlist (`V2_TRAINING_PROFILE_ALLOWLIST`), so the live
training schedule could never have selected any of them.

- **3 repeats** per (arm, profile) cell.
- **Order rotated per repeat** using a Latin square over the 3 arms, so no
  arm is systematically first or last within a repeat. The realized order
  for every (profile, repeat) cell must be recorded in the evaluation
  run's own output before any result is read.

## Epsilon rule (near-tie threshold for PPO-V2's deterministic inference)

```
epsilon = 10th percentile of the final model's (session_047) top1-top2
          action-probability margin (action_selection.compute_margin_stats().margin),
          computed over observations from the LAST 5 training sessions'
          step_telemetry.jsonl (sessions 043-047).
cap   = 0.05
floor = 0.0
```

Computed **once**, before evaluation begins, and never re-tuned after
seeing any evaluation result. The computed value is **not yet known** —
session_047 does not exist while training is still live — and must be
appended to `EVALUATION_PREREGISTRATION.json`'s `epsilon_rule.value` field
in Phase 5B, before the first evaluation run. This supersedes the current
provisional `near_tie_epsilon: 0.05` in `configs/rl_v2.yaml` **for
evaluation purposes only** — training itself is unaffected (same CORR-58
scope boundary).

## Metrics

**Primary**:
- SLA-violation rate at 1000 ms (share of decisions with `p95_latency_ms >= 1000`, the current CORR-57 SLA target).
- Replica-minutes (cost: the time-integral of replica count).

**Secondary**: mean/median p95 latency, error rate, max per-pod memory
(MiB), scaling-action count, reversal count, reaction time after a load
change, pod-restart count, invalid-sample count.

**Reference only**: SLA-violation rate at 300 ms (the pre-CORR-57 target),
reported for historical comparability — not the primary SLA definition for
this evaluation.

## Validity rules

- **Tooling failure** (route unreachable, Prometheus unavailable, a
  cluster/network fault) → no scientific outcome for that attempt → retried
  **exactly once**. The failed attempt's evidence is preserved, never
  deleted.
- **Safety termination** (e.g. a memory-safety threshold breach, zero ready
  replicas) → **counted as a scientific outcome**, **never retried**.
- Every attempt — successful, tooling-failed, or safety-terminated,
  including the one permitted tooling-failure retry — is preserved on disk
  under its own directory.

## Decision rule

- Report mean ± sd over the 3 repeats, per profile, per arm, for every
  metric.
- Claim an arm "better" than another on a given metric for a given profile
  **only if it is better in all 3 of the 3 repeats** — a single repeat
  going the other way blocks the claim.
- **No aggregate winner** is declared when metrics disagree (e.g. a lower
  SLA-violation rate but higher replica-minutes) — such cases are reported
  explicitly as a trade-off.

## Scope

Evaluation-only code changes made after this record (Phase 5B:
`scripts/evaluate_ppo_vs_hpa_v2.py` changes, the CPU-only-HPA
`controller_exclusivity.py` extension, etc.) do not affect training in any
way. Training's own governing files, `FREEZE_MANIFEST.json`, and
`FREEZE_MANIFEST_AMENDMENT_001.json` remain the sole authority for what
was learned. Every evaluation-code file's SHA-256 hash will be recorded in
a dedicated evaluation-code manifest **before** the first evaluation run
executes.

## Training contract reference (at the time of this pre-registration)

- `FREEZE_MANIFEST.json` + `FREEZE_MANIFEST_AMENDMENT_001.json`
- Trainer hash: `6c6f00c096c4932540e11baa94d961658b850cbc917e7068ba085ea7805a69ac`
- Total timesteps budget: 12,288 (48 sessions)

See CORR-60 in `docs/experimental_design_corrections.md` for the
append-only correction-log entry summarizing this record.
