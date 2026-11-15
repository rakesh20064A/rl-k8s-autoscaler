# V2 Phase 6A — Evaluation Results Report

Campaign: `eval_final` (`persistent-data/evaluation_v2/eval_final`).
Pre-registration: `EVALUATION_PREREGISTRATION.json` (CORR-60), amended by
CORR-61 (safety/termination clarification), CORR-62
(`CONTROLLER_CPU_ONLY_HPA_V2` addition), CORR-63 and CORR-64 (two
tooling-defect fixes, below). Analysis tool: `scripts/analyze_v2_evaluation.py`
(read-only against `persistent-data/rl_training_v2/` and
`persistent-data/evaluation_v2/`). All numbers below are taken directly from
`persistent-data/traffic/v2_phase6a_analysis/evaluation_summary.{md,csv}` and
the comparison rule already implemented in that script
(`compare_arms_per_profile()`/`is_better_in_all_repeats()`), not recomputed
by hand.

## 1. Validity

27 of 27 planned runs are `VALID` (9 per arm: `ppo_v2`, `hpa_v2_cpu_memory`,
`hpa_v2_cpu_only`; 3 per profile x arm cell, across `eval_cpu_step`,
`eval_memory_ramp`, `eval_mixed_burst`). No run was invalidated by a safety
termination (no pod restarts, no sustained zero-ready-replicas, no sustained
memory breach, in any of the 27 valid runs' own recorded
`pod_restarts_total`).

Four attempts across the campaign's history were **not** `VALID` — all four
are pure tooling/infrastructure failures with no scientific content (no
`"result"` key in the ledger), each later resolved by an explicit,
append-only `--rerun-after-fix` re-attempt that IS one of the 27 `VALID`
rows above:

| run_id | attempt | status | error | fixed under |
|---|---|---|---|---|
| `eval_cpu_step__repeat0__hpa_v2_cpu_only` | 1 | TOOLING_FAILURE | conflicting HPA `['demo-app-hpa-v2']` | CORR-63 |
| `eval_cpu_step__repeat0__hpa_v2_cpu_only` | 2 | FAILED_AFTER_RETRY | same (one permitted retry burned) | CORR-63 |
| `eval_cpu_step__repeat1__hpa_v2_cpu_memory` | 1 | TOOLING_FAILURE | preflight failed, HPAs `['demo-app-hpa-v2', 'demo-app-hpa-v2-cpu']` | CORR-64 |
| `eval_cpu_step__repeat1__hpa_v2_cpu_memory` | 2 | FAILED_AFTER_RETRY | same | CORR-64 |

Both runs' attempt 3 (tagged `corr_id: "CORR-63"` / `"CORR-64"`,
`rerun_after_fix: true` in the ledger) is `VALID` and is the row counted in
the 27 above. Per the pre-registration's own validity rules, these four
failed attempts carry zero scientific outcome and are preserved on disk,
never deleted or edited.

## 2. Per-profile results (primary metrics, mean ± sd over 3 repeats)

### eval_cpu_step

| arm | SLA-violation-rate@1000ms | replica-minutes |
|---|---|---|
| ppo_v2 | 0.0083 ± 0.0144 | 72.67 ± 0.29 |
| hpa_v2_cpu_memory | 0.0000 ± 0.0000 | 60.50 ± 0.87 |
| hpa_v2_cpu_only | 0.0000 ± 0.0000 | 59.33 ± 0.29 |

### eval_memory_ramp

| arm | SLA-violation-rate@1000ms | replica-minutes |
|---|---|---|
| ppo_v2 | 0.0000 ± 0.0000 | 56.17 ± 0.29 |
| hpa_v2_cpu_memory | 0.0000 ± 0.0000 | 75.17 ± 1.61 |
| hpa_v2_cpu_only | 0.0000 ± 0.0000 | 40.00 ± 0.00 |

### eval_mixed_burst

| arm | SLA-violation-rate@1000ms | replica-minutes |
|---|---|---|
| ppo_v2 | 0.0000 ± 0.0000 | 58.33 ± 2.31 |
| hpa_v2_cpu_memory | 0.0000 ± 0.0000 | 74.67 ± 5.13 |
| hpa_v2_cpu_only | 0.0000 ± 0.0000 | 40.00 ± 0.00 |

Full tables (all primary + secondary + reference-only metrics, all 9
profile x arm cells) are in
`persistent-data/traffic/v2_phase6a_analysis/evaluation_summary.md` /
`.csv`. Figures: `sla_violation_rate_1000ms.png`, `replica_minutes.png` in
the same directory.

## 3. Verdicts (pre-registered decision rule only)

Per `EVALUATION_PREREGISTRATION.json`'s `decision_rule`: an arm is "better"
on a metric for a profile only if it is better in **all 3 of 3 repeats**
(not just on average); no aggregate winner is declared across metrics that
disagree.

**SLA-violation-rate@1000ms**: no decidable winner, any arm vs. any arm, in
any of the 3 profiles. Every arm is statistically indistinguishable from
every other arm on this metric — means are 0 (or effectively 0) across the
board; the one nonzero cell (`ppo_v2` / `eval_cpu_step`, 0.0083 ± 0.0144)
reflects a single repeat with a small violation rate and does not make
`ppo_v2` worse in all 3 repeats, so no claim is supported either way.

**replica-minutes** (cost):
- `eval_cpu_step`: `hpa_v2_cpu_memory` better than `ppo_v2`; `hpa_v2_cpu_only`
  better than `ppo_v2`; `hpa_v2_cpu_memory` vs `hpa_v2_cpu_only` — no
  decidable winner (close, 60.50 vs 59.33, order not consistent across all 3
  repeats).
- `eval_memory_ramp`: `ppo_v2` better than `hpa_v2_cpu_memory`;
  `hpa_v2_cpu_only` better than `ppo_v2`; `hpa_v2_cpu_only` better than
  `hpa_v2_cpu_memory`.
- `eval_mixed_burst`: `ppo_v2` better than `hpa_v2_cpu_memory`;
  `hpa_v2_cpu_only` better than `ppo_v2`; `hpa_v2_cpu_only` better than
  `hpa_v2_cpu_memory`.

No aggregate ranking across profiles or across the two primary metrics is
declared — per the pre-registration's own `no_aggregate_winner` rule, each
profile x metric verdict above stands on its own.

## 4. Trade-offs in plain language

- All three controllers meet the SLA (p95 latency < 1000 ms) almost
  perfectly, and no evidence here shows one controller trading SLA
  compliance for cost or vice versa — the SLA metric simply does not
  discriminate between them at this workload intensity.
- `hpa_v2_cpu_only` is the cheapest (lowest replica-minutes) controller in
  two of three profiles (`eval_memory_ramp`, `eval_mixed_burst`), and ties
  with `hpa_v2_cpu_memory` on the third (`eval_cpu_step`) — it never costs
  more than the other two arms in any profile where a verdict was
  decidable.
- `ppo_v2`'s cost ranks in the middle and flips profile to profile relative
  to the HPA arms (cheaper than `hpa_v2_cpu_memory` on the memory/burst
  profiles, more expensive than both HPA arms on the CPU-step profile) —
  its replica-minutes behavior is workload-dependent rather than uniformly
  better or worse than either HPA baseline.

## 5. Safety events

None. Across all 27 valid runs: `pod_restarts_total == 0` in every run; no
`SafetyTerminationError` occurred for any arm (no sustained >=230Mi per-pod
memory breach, no pod restart, no sustained zero-ready-replicas, no PPO
4-consecutive-invalid-step termination — per the CORR-61 thresholds, which
govern every arm identically).

## 6. Deviations from the original plan (CORR-60 through CORR-64)

- **CORR-60** (Phase 5A): the pre-registration itself — arms, profiles,
  primary/secondary/reference-only metrics, epsilon rule, and the decision
  rule applied throughout this report.
- **CORR-61** (Phase 5C-A): clarified, before any run, which live-cluster
  conditions count as a safety termination (same thresholds training
  already uses, applied identically to all 3 arms) vs. a tooling failure.
- **CORR-62** (Phase 5D, post-training): added
  `CONTROLLER_CPU_ONLY_HPA_V2` to `controller_exclusivity.py` so the
  `hpa_v2_cpu_only` arm could be acquired/released at all; purely additive,
  no existing controller's behavior changed.
- **CORR-63** (Phase 5F): fixed a tooling defect where the
  `hpa_v2_cpu_only` arm's wrapper didn't remove a conflicting leftover
  `demo-app-hpa-v2` before starting; added the `--rerun-after-fix`
  mechanism used to re-attempt the one affected run with zero scientific
  content lost.
- **CORR-64** (Phase 5G): fixed a second instance of the same bug class —
  the campaign-level `restore_canonical_idle_state()` cleanup left
  `demo-app-hpa-v2-cpu` behind after the `hpa_v2_cpu_only` arm, blocking
  the next `hpa_v2_cpu_memory` run; re-attempted via the same
  `--rerun-after-fix` mechanism.

None of CORR-62/63/64 touched any arm's scoring logic, any profile, any
metric definition, or any of the 5 resume-governing files — all four failed
attempts they explain carry zero scientific outcome, and all 27 `VALID`
runs reported above were produced under the code version each fix actually
shipped as (confirmed by the ledger's own `corr_id` tags on the two
rerun rows).

## 7. Limitations

- Each profile x arm cell has only 3 repeats (as pre-registered) — the
  per-repeat sample size is small, so "no decidable winner" verdicts above
  may partly reflect limited statistical power rather than true
  equivalence; they are reported as the pre-registration itself defines
  them (strict better-in-all-3-repeats), not as a claim of proven parity.
- `reaction_time_s` is `n/a` across every cell in the summary table — the
  analysis script's `compute_reaction_time_s()` did not find a usable
  signal for it in this normalized series schema; it is not fabricated or
  imputed here.
- `sla_violation_rate_300ms_reference` (historical-comparability only, not
  this evaluation's primary SLA definition) is high across several cells —
  this is expected and explicitly out of scope for any "better" claim per
  the pre-registration's own `reference_only` classification.
- This report applies only the mean/sd-over-repeats and
  better-in-all-3-repeats rules exactly as pre-registered; no additional
  statistical test (e.g. significance testing beyond the pre-registered
  rule) was introduced.
