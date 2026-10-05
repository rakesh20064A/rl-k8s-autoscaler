# Phase 16U — Direct/Bounded Live-Kubernetes PPO Protocol Preregistration

**Classification: `OFFLINE_PROTOCOL_FORMALIZATION_ONLY`.** No traffic
generated, no Kubernetes/HPA mutation, no PPO training or evaluation, no
surrogate-model refit, no reopening of the consumed Iteration-2B/
Iteration-3 holdouts.

## Key finding before formalizing anything

This project already has a **frozen, authoritative MDP contract**
(`docs/rl_mdp_spec.md`, `rl_contract_version: 2`, `reward_contract_version:
2`, frozen Phase 15A.1, clarified Phase 15B.1A/CORR-30) — the observation
vector (11-dim, exact clip ranges and missing-data semantics), action
encoding (`Discrete(3)`, ±1/HOLD), reward formula (7 weighted components,
exact coefficients), decision interval (30s), episode contract (reset
sequence, termination rules), and cooldown/gating rules were all already
designed and frozen before Phase16T's recommendation. Phase16U therefore
**adopts this contract unchanged** rather than redesigning it, and
confines itself to what the frozen contract explicitly still requires: a
live-execution protocol, a training/evaluation workload split, a
live-feasible PPO training budget, safety/fallback procedures, and
ledgers — none of which existed yet (`autoscaler/src/rl_env/` implements
this contract only against a `ScriptedBackend`; `training/` and
`k8s/rl-agent/` remain empty).

## 1. Correction to Phase16T

`phase_16t_direct_rl_design.md` loosely suggested using historical
Iteration-2B/Iteration-3 transitions as an "off-policy replay buffer" for
PPO. This is corrected in
`phase_16u_direct_ppo_protocol.yaml` (Section `correction_to_phase16t`):
**PPO is on-policy** and must not be trained from a conventional replay
buffer of historical transitions. Historical data may only support
observation-normalization sanity-checks, reward-scale sanity-checks, and
the new final-holdout workload's envelope design — each explicitly
separated from PPO's own rollout/update mechanism. An optional, separately
justified supervised behavior-cloning pretraining step is noted but left
undecided, per the task's own conditional wording.

## 2–3. Observation and action space

Adopted unchanged from the frozen contract — not redesigned. 11-dimensional
live-only observation (no surrogate-model output ever enters it),
`Discrete(3)` ±1/HOLD action space, exact clamp/pending-scale/cooldown
rules cited from `docs/rl_mdp_spec.md` sec.9/12/13.
`phase_16u_safety_and_fallback_protocol.yaml` makes the requested
**pre-action rejection vs. post-action emergency fallback** distinction
explicit (the frozen contract already implements both mechanically but
never used these exact terms).

## 4. Decision timing and episode structure

30s decision interval and 60s pre-episode stabilization are frozen,
unchanged. Episode = one full traffic-profile execution (also frozen,
unchanged — not cropped for training convenience). **New, justified
choice**: PPO-controlled episodes restore to 1 replica via a *direct*
Deployment patch (not through HPA), bounded to a 120s max wait — faster
than the existing HPA-throttled `restoration_wait_seconds()` (up to 570s)
because a direct patch is not subject to HPA's own -1-pod-per-60s
scale-down throttle. HPA-controlled evaluation episodes keep the existing
mechanism unchanged, since there HPA genuinely governs the pacing.

## 5–6. Reward and PPO configuration

`phase_16u_reward_specification.yaml` restates the frozen reward formula
verbatim — zero tuning. `phase_16u_ppo_hyperparameters.json` freezes a
**deliberately reduced** PPO configuration for live, real-time
interaction: `n_steps` 2048→256, `total_timesteps` 500,000→12,288, `eval_freq`
10,000→1,024, `n_eval_episodes` 5→1, training seeds `[42,7,123]`→`[42]` —
every deviation from `configs/rl.yaml` is listed with its numeric reason.
**Feasibility disclosed honestly**: 500,000 live steps would require ~173
days of pure decision time and is infeasible; 12,288 steps (~118.6 hours,
~4.9 days of live wall-clock, including reset/validation overhead) is
~40x smaller than the original simulated-backend target, and whether PPO
learns a non-trivial policy within this budget is explicitly flagged as
an open empirical question for the implementation phase to report
honestly either way.

## 7. Training vs. final evaluation separation

`phase_16u_training_and_evaluation_workload_split.json` designates the six
existing candidate profiles (`constant`, `step`, `burst`, `ramp`,
`periodic`, `mixed`) as the training/development set. `unseen_eval.yaml`
is explicitly excluded from the formal comparison (its epistemic status
was already compromised per `docs/rl_mdp_spec.md` sec.22). Performs the
workload-freeze step that sec.22 required but left undone: a **new**
final-holdout generation specification (`iteration_rl_final_holdout_v1.yaml`,
seed 777, 1500s, a 6-phase shape combining elements no single training
profile combines, peak 65 RPS chosen to avoid reusing any previously
inspected magnitude) — specified precisely enough to create mechanically
later, but **not created, inspected, or run** in this phase.

## 8. Training budget

Frozen at 12,288 environment steps (Section 6/8 above) — a fixed number,
not "train until convergence," with the shortfall relative to the
simulated-backend target disclosed rather than hidden.

## 9–10. Baselines and final comparison metrics

Required baselines: PPO vs. canonical HPA (260%/1–8, unchanged since
CORR-26), no co-control, fail-closed if a live HPA object is detected at
PPO-controller start. Optional diagnostic-only static-4-replica baseline.
Five frozen per-episode metrics (SLA-violation rate, error rate, resource
proxy, reaction time, action churn) and a **conservative, non-inferential
decision rule** frozen now (3 repeats per controller; "descriptively
better" requires consistent wins on both SLA and resource metrics across
all 3 repeats; otherwise reported as "mixed / no clear winner") — chosen
specifically so no decision rule is invented after seeing results, and so
the small sample size is not overclaimed.

## 11. Reproducibility and ledgers

Three new, dedicated ledgers specified (PPO training episodes, PPO final
evaluation, HPA final evaluation) — Iteration-3's scientific ledger is
never reused. Every episode gets exactly one row; safety events and
restoration timeouts are recorded, never discarded or silently retried.

## 12. Implementation boundary

No PPO or environment code was written. No live episode ran. The concrete
list of what a future, separately-authorized implementation phase must
still build (a live `DynamicsBackend`, the three ledgers, the new
held-out workload file, the actual training run, the actual evaluation
campaign) is recorded in `phase_16u_direct_ppo_protocol.yaml` Section 12.

## Verification performed in this phase

- Re-hashed `docs/rl_mdp_spec.md`, `configs/rl.yaml`, `configs/reward.yaml`,
  `configs/experiment.yaml`, `k8s/hpa/hpa.yaml`, `configs/hpa.yaml` —
  all read-only inputs, all confirmed present and used only for citation.
- Zero Python files created or modified (confirmed via mtime scan against
  the Phase16T baseline) — the standing "full test suite + Ruff on changed
  Python" requirement is not triggered.
- No traffic generated, no Kubernetes/HPA object touched, no Iteration-2B/
  Iteration-3 ledger or artifact modified.

## Output files

- `phase_16u_direct_ppo_protocol.yaml`
- `phase_16u_reward_specification.yaml`
- `phase_16u_training_and_evaluation_workload_split.json`
- `phase_16u_ppo_hyperparameters.json`
- `phase_16u_safety_and_fallback_protocol.yaml`
- `phase_16u_completion_report_and_verdict.md`
- `phase_16u_manifest.json`

## Verdict

**`PHASE16U_DIRECT_PPO_PROTOCOL_PREREGISTRATION_COMPLETE`**

**STOP.** PPO is not implemented and no live traffic is generated
automatically. A separately-authorized implementation phase is required
before any of the Section-12 implementation items begin.
