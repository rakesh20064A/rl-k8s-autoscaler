# RL Empirical Dynamics Calibration Protocol — v1 (FROZEN, CORR-35 CORRECTED)

**Status**: Frozen by Phase 15B.2 (CORR-31); corrected before its first execution by
Phase 15B.2A (CORR-32), further clarified by Phase 15B.2B (CORR-33), closed for
execution by Phase 15B.2C (CORR-34), then had its routing mathematics corrected for
unequal Ready-serving exposure by Phase 15B.2D (CORR-35) — **no calibration data
existed at any point**. This is the authoritative narrative specification of how the
future empirical `DynamicsBackend` (the piece that will eventually let
`KubernetesAutoscalerEnv` — frozen mechanics-complete since Phase 15B.1/15B.1A —
actually predict CPU/latency/error/memory/in-progress behavior) will be calibrated.
The full frozen, machine-readable matrix (cell list, canonical schedules + SHA-256
hashes, randomized order, the one global 102-slot execution order, model families,
acceptance criteria) is `experiments/manifests/rl_dynamics_calibration_v1.yaml`
— this document explains and justifies it; the manifest is the single source of
exact numbers.

```
dynamics_calibration_protocol_version: 1   (unchanged -- pre-execution corrections, not a post-data methodology change)
frozen_in_phase: 15B.2
corrected_in_phase: 15B.2A (CORR-32), then 15B.2B (CORR-33), then 15B.2C (CORR-34), then 15B.2D (CORR-35)
frozen_date_utc: 2026-09-09
serves: rl_contract_version 2 / reward_contract_version 2 (unchanged)
official_new_calibration_run_count: 102
```

---

## 1. What This Protocol Governs

Unchanged from v1: the empirical dynamics — given a workload level and a replica
count, what CPU/latency/error-rate/memory/in-progress values does the real
application actually produce? `ScriptedBackend` deliberately does not answer this;
this protocol defines what will be measured, how, how much, in what order, how it
will be validated, and what "good enough" means, **before** a single new data point
is collected.

## 2. Existing-Data Suitability Audit — CORR-32-Corrected

**Category A (reusable directly for fitting): NONE.** This is the single largest
change from v1. Independent re-inspection found:
- **CORR-23/Phase 11C.4D** is the **Metrics-API** CPU mapping — built specifically
  to characterize what HPA itself reads. Its primary reported values are
  Metrics-Server-sourced, not the frozen Prometheus/cAdvisor aggregate target this
  protocol fits to. Reclassified: `SANITY_CHECK_OR_SUPPLEMENTAL_CPU_CROSS_CHECK_ONLY`
  — it must never substitute for an official Prometheus/cAdvisor calibration repeat.
- **CORR-22's fresh-mode addendum**, independently re-verified, *does* use the
  correct Prometheus/cAdvisor CPU source, the corrected NodePort path,
  `connection_mode=fresh`, and intensity=3 — structurally compatible in isolation.
  However, it predates this protocol's own newly-frozen decision-aligned extraction
  convention (§9), canonical-schedule-replay discipline (§6), and generator-capacity
  sizing (§5) — none of which it followed. For measurement-semantics uniformity
  across the entire fitting dataset, `historical_run_substitution_for_official_repeats`
  is frozen `false`: **no historical run substitutes for any official repeat.** All
  102 official runs are newly collected under this one protocol.

Historical data remains useful for: matrix-design evidence (grounding load-level
RPS choices, §7), sanity-check priors (e.g. cross-checking the new dedicated 1↔2
transition cell against CORR-25's ~5-15s prior), and external consistency checks
during analysis — never as a direct fitting-label substitute.

**Category B/C (sanity-check-only / never suitable), unchanged in kind from v1**:
all 9 official HPA tuning trials (peaked at exactly 2 replicas — verified from
`experiments/results/hpa_tuning_final/per_trial_metrics.csv`, no exception — the
entire 3-8 replica range is genuinely uncalibrated); Phase 10's keepalive-mode
calibration; Phase 11C.4B's confounded spot-check; any Metrics-Server/Metrics-API
reading or HPA's own `averageUtilization`, as fitting targets, ever.

## 3. Dynamics-Backend Target Outputs (unchanged from v1)

The future empirical backend must produce the same `RawMetricSnapshot` shape
`ScriptedBackend` already produces — no `backend.DynamicsBackend` Protocol change
needed. The exogenous-workload/policy-observation boundary is unchanged:
`docs/rl_mdp_spec.md` §2's rule (no future-workload-schedule value reaches a PPO
policy) stands.

## 4. Frozen Scope: Intensity (unchanged)

`intensity=3`, unchanged from v1 — no conflict found, no STOP triggered.

## 5. Generator-Capacity Audit and Revised Maximum Workload (CORR-32, NEW)

Before freezing any load ceiling, the traffic generator's actual constraints were
inspected directly (`loadgen/traffic_generator.py`): default `timeout_seconds=5.0`,
default `max_in_flight=50`. Worst-case simultaneously-outstanding requests under a
full application stall ≈ `scheduled_total_rps × timeout_seconds`. v1's original L4
level (12 RPS/replica, scaling to 96 RPS total at replicas=8) would require
`96 × 5.0 = 480` potential outstanding requests — nearly 10× the generator's default
`max_in_flight=50`, meaning the generator's own harness capacity, not the
application, would have been the effective bottleneck at that extreme, exactly the
failure mode the task warned against.

**Correction**: `max_in_flight` is frozen at **400** for the entire calibration
campaign (an existing `--max-in-flight` CLI/constructor parameter — no code change)
— a ~25% safety margin above the worst-case bound at the *revised* matrix's own
maximum (64 RPS × 5.0s = 320, occurring at replicas=8/L3). L4 ("overload") is
restricted to replicas `{1,2,4}` only (totals 12/24/48) rather than all 5 anchors,
removing the 72/96 RPS extremes entirely — L3's own already-justified per-replica
design naturally reaches the matrix's highest total (64, at replicas=8), which is a
meaningful "does near-capacity per-pod behavior hold at scale" test, not an
arbitrary new extreme. Any cell reporting `dropped_overload` under the frozen
`max_in_flight=400` is now an explicit **harness-capacity structural defect** for
that cell (§21), never an application-performance observation.

## 6. Frozen Arrival Process and Canonical Schedules (CORR-32, NEW)

**Deterministic, constant, open-loop arrivals** — frozen as the sole arrival mode
(not Poisson), so repeat-to-repeat variation measures the system, not schedule
randomness. The identical canonical schedule is replayed for every repeat of a
cell. **37 canonical schedules have already been generated and SHA-256-hashed
OFFLINE** (no traffic sent) via `scripts/generate_dynamics_calibration_schedules.py`
— a new, narrow, network-free script that reuses `scripts/generate_tuning_schedule.py`'s
existing, tested, pure schedule-generation path (the same one that froze the
HPA-tuning canonical schedule, CORR-24) rather than a new implementation. Schedules
live under `configs/traffic_profiles/rl_dynamics_calibration/{steady_state,replica_transitions,workload_transitions}/`,
indexed in `configs/traffic_profiles/rl_dynamics_calibration/schedule_registry.json`
(cell ID → replicas/RPS/kind/SHA-256/path). This makes the protocol genuinely
mechanical: the execution phase replays these exact files, it does not generate
new ones.

## 7. Frozen Steady-State Matrix — Deduplicated (CORR-32)

**20 unique fitting cells** (down from v1's 23, after removing 1 genuine duplicate
and restricting L4's anchor set):

| | Replicas | Load levels |
|---|---|---|
| Fitting | 1, 2, 4, 6, 8 | L1(2 RPS/replica), L2(5), L3(8) at all 5; L4(12 RPS/replica) at {1,2,4} only; L5(fixed 40 RPS total) at {1,4,8} |
| Interpolation validation | 3, 5, 7 | L1, L2, L3 only |

**The duplicate found and fixed**: at replicas=8, `L2` (5×8=40) and `L5` (fixed 40)
are the exact same physical condition. Merged into one tagged cell,
`dyncal_r8_L2_L5`, executed **once** per repeat with both analytical roles recorded
— never run twice under two names. Every remaining `(replicas, total_rps)` tuple
was independently audited and confirmed physically unique (the same total RPS at a
*different* replica count is never a duplicate — it is a different per-pod load,
exactly the thing this campaign exists to characterize).

## 8. Load-Level Rationale (unchanged reasoning, revised L4 scope per §5)

L1/L2/L3 remain anchored to matrix-design evidence (CORR-22/23, now explicitly
sanity-check-only per §2, not fitting substitutes) — 2/5/8 RPS/replica → 25.1%/
~49-52%/75.9% of the single-replica 500m limit. L4 (12 RPS/replica, `{1,2,4}` only)
probes genuinely untested overload territory without excessive totals. L5 (fixed
40 total, `{1,4,8}`) is the cluster-wide-contention probe, physically grounded in
4×500m pods already exactly consuming the 2-core outer ceiling.

## 9. Decision-Aligned Metric Extraction (CORR-32, NEW — Precisely Frozen)

Per-cell timing, now stated relative to **traffic onset** (not the pre-traffic
stabilization period): `warmup=[0,60)`, `scored=[60,180)` (traffic duration 180s;
plus 60s pre-traffic, no-traffic, fixed-replica stabilization = 240s total per
steady-state run). **Decision-aligned extraction timestamps**: `[60, 90, 120, 150]`
seconds relative to traffic onset — for each, use the **latest valid** Prometheus
observation at-or-before that timestamp, within the existing 30s staleness
tolerance (`configs/experiment.yaml`) — never a sample after the target timestamp.
All four decision-aligned samples from one run may be used in fitting, but **all
train/validation splitting remains at the whole-run/cell-repeat level**, never
across individual decision-aligned samples within a run (which would leak
autocorrelated observations, unchanged principle from v1, now stated with the
exact timestamps that make it unambiguous).

## 10. Replica-Transition Matrix — Corrected to ±1 Only (CORR-32)

v1's `1→2,2→4,4→6,6→8` design used non-adjacent jumps that do not represent one RL
action (the policy changes replicas only by ±1). **Corrected**: four adjacent pairs
spanning the full range — `1↔2, 3↔4, 5↔6, 7↔8` — 8 directional transition types.
Workload during a transition: `5.0 RPS × lower_replica_count_of_the_pair`, held
**constant** across the transition (only replica count changes) — verified
sustainable in both pre- and post-transition states for every pair (e.g. pair
7↔8: 35 RPS total = 5/replica at 7, ~4.4/replica at 8, both comfortably inside the
verified-sustainable L2 regime). Because the workload doesn't change with
direction, **one schedule serves both directions of a pair** (4 distinct schedules
for 8 directional types).

## 11. Workload-Step Transition Matrix — Added to the Run Budget (CORR-32)

v1 *named* fixed-replica workload-step transitions in its narrative but never
budgeted any runs for them — an omission, corrected here. Four types: at
replicas=1, `2↔8` RPS (application response without cluster contention); at
replicas=4, `8↔32` RPS (dynamics closer to shared-host pressure, since 4×500m
already equals the 2-core ceiling; 32 ≤ the revised 64 RPS ceiling, verified safe
under the frozen `max_in_flight=400`). 2 repeats each — **additional** to, not a
substitute for, the replica-count transition runs.

## 12. Transition Timing (CORR-32, Precisely Frozen for Both Kinds)

Both replica-count and workload-step transitions share one exact structure: 60s
pre-run replica stabilization (no traffic) → 60s pre-transition/pre-step traffic at
the "before" condition → **trigger at traffic-elapsed=60s** → 180s post-trigger
observation (300s total per transition run: 60 stabilization + 240 traffic).
Pre-transition baseline window, transition timestamp, and post-transition
settled-response window are all now unambiguous — v1 had stated "60s pre + 180s
post" without defining where traffic began or which interval was scored.

## 13. Repetition/Randomization Strategy (Regenerated)

3 repeats for fitting anchors, 2 for validation/replica-transition/workload-
transition cells (unchanged rationale from v1). **Every randomized order was
regenerated** against the corrected, deduplicated cell lists — deterministic
`random.Random(seed).shuffle()`, distinct seed bases per cell-type
(`42`/`142`/`242`/`342` + repeat index for fitting/validation/replica-transition/
workload-transition respectively) — the exact resulting sequence for every repeat
is embedded verbatim in the manifest.

## 14. Corrected Campaign Size

`fitting: 20×3=60` + `validation: 9×2=18` + `replica-transition: 8×2=16` +
`workload-transition: 4×2=8` = **102 total runs** (v1's 103 was not preserved
merely because it was previously written — recomputed from first principles after
every correction above). Steady-state runs: 240s each. Transition runs (both
kinds): 300s each. Estimated wall-clock: **≈8.9 hours** across multiple sessions.

## 15. Acceptance-Criterion Mathematics — Corrected (CORR-32)

v1's "MAE ≤ 2× repeat SD" was ambiguous (MAE often aggregate vs. SD cell-specific;
near-zero SD would demand near-zero error; 2-repeat validation-cell SD is itself
noisy). **Corrected, explicit formula**:

```
s_eff[cell,metric] = max(
    within_cell_repeat_sd[cell,metric],
    0.5 * pooled_within_cell_sd_across_fitting_anchors[metric],
    epsilon[metric]         # numerical floor only, never a generous allowance
)
normalized_error[cell,metric] = |predicted_cell_mean - observed_cell_mean| / s_eff[cell,metric]
```

Fitting-cell predictions use **leave-one-repeat-out** (never same-data fitted
predictions); interpolation cells use the held-out cell mean. Frozen thresholds:
median normalized error ≤2.0; ≥90% of cells ≤2.0; no cell >4.0 — applied **equally**
to fitting and interpolation cells. p95 latency's normalized error is computed in
the frozen `log1p` space; millisecond-space error is reported separately, never
substituted into the acceptance decision.

## 16. Error-Rate Validation — Corrected to Balanced Metrics (CORR-32)

Plain zero/non-zero classification accuracy can be trivially "passed" by an
always-zero classifier when positive-5xx cells are rare (exactly the expected
regime for this application). **Corrected**: `balanced_accuracy ≥0.80` and
`nonzero_class_recall ≥0.80` (both classes weighted, never a raw-accuracy proxy),
plus the existing probability/rate MAE. If the entire fitting+validation dataset
contains zero non-zero server-visible 5xx observations, classification quality is
explicitly reported `NOT_ESTIMABLE` — never asserted as passing from an all-zero
dataset.

## 17. Monotonicity — Justified Axes Only (CORR-32)

A generic cross-replica monotonicity rule is **not** imposed — the shared 2-core
host means increasing replicas can legitimately create *more* contention, not
less. **Enforced**: at fixed replicas, increasing offered RPS must not produce a
materially *lower* aggregate CPU, server p95, in-progress load, or 5xx rate, beyond
local noise. **Explicitly not enforced**: latency improving monotonically as
replicas increase at fixed RPS, or CPU falling monotonically as replicas increase
at fixed total RPS — both could suppress the exact shared-host effects L4/L5 exist
to discover.

## 18. Stochastic Residual/Noise Modeling — Preregistered (CORR-32, NEW)

v1's candidate families modeled conditional means only, risking PPO overfitting to
a perfectly deterministic simulator. **Two candidate residual schemes preregistered
now** (no winner selected): (1) joint empirical residual bootstrap — fit mean
dynamics on fitting runs, sample one joint `[cpu, latency, memory, in_progress,
error_rate]` residual vector at a time (never independent per-metric noise, to
preserve cross-metric correlation), stratified by nearest supported cell when
sample size permits; (2) a parametric multivariate distribution (e.g. Gaussian
copula) as a fallback if empirical sample size proves inadequate. Residuals are
**never** estimated from interpolation-validation runs. Validated separately via
simulated-vs-observed variability, quantile comparison, and physical-bound
violation counts.

## 19. Memory/In-Progress Simplification — Clarified Decision Rules (CORR-32)

**Memory**: evaluated per-*Ready*-replica first (`memory_bytes/ready_replicas`);
regressed against `(total_offered_rps, replicas)` using fitting-cell data only; if
R²<0.10 or CV<10%, modeled as `aggregate = per_replica_empirical_baseline ×
ready_replicas` (CORR-25-consistent replica semantics) with fitted residual
variation. **In-progress**: evaluated as an aggregate, load-regime-tested — **not**
automatically collapsed to a replica-only constant merely because CV is low (it is
logically a queueing signal tied to offered load, per CORR-32 §18's explicit
correction); if the simplification criterion fires, a load-regime-conditioned
empirical distribution is preferred over a pure per-replica constant.

## 20. Dynamics-Out-of-Support ≠ Prometheus Unavailable (CORR-32)

v1 incorrectly proposed mapping an out-of-calibration-support query onto
`RawMetricSnapshot`'s existing `target_up=False`/`query_error=True` fields — this
conflates simulator-domain ignorance with an actual Prometheus/target outage.
**Corrected**: a distinct future condition, `DYNAMICS_OUT_OF_SUPPORT`, is frozen
conceptually (not implemented in this protocol-correction phase — it requires a
small, targeted extension during the later empirical-backend-implementation phase,
and does not change the 11-element policy observation contract). Recommended
handling: stop/truncate the simulated episode immediately with an explicit reason,
never counted toward Prometheus-downtime statistics, no further steps trained from
that unsupported state.

## 21. Invalid-Cell Policy — Generator/Harness Criteria Added (CORR-32)

Retains validity-before-performance (CORR-17 §7 lineage). **Added structural
criteria**: `offered != sent` due to an undersized generator `max_in_flight`;
generator `dropped_overload` caused by harness capacity rather than application
response. **Clarified as VALID**: a client timeout occurring *after* a request was
legitimately dispatched to a genuinely overloaded application remains a valid
performance outcome, distinct from a harness-capacity artifact.

## 22. Pre-Registered Future Analysis Outputs (unchanged from v1)

Per-run/per-cell tables, repeat-variability table, interpolation-validation
results, transition-response tables (replica- and workload-transition reported
separately), fitted-model comparison table, residual/error plots, calibration-
domain plot, final chosen dynamics-model specification, simulator-vs-real
validation report. None created now.

## 23. Request-Rate Realization — Offered vs. Server-Observed (CORR-33, NEW)

CORR-32 fitted `request_rate_rps` implicitly as if it were a passthrough of the
exogenous workload schedule. It is not: `offered_rps` (the schedule's planned
arrivals — an internal simulator input, **never** exposed directly to the PPO
observation, per `docs/rl_mdp_spec.md` §2) and `request_rate_rps` (the
Prometheus-server-observed rate, one of the 11 frozen observation components) can
diverge, especially under overload, because a `connection_error` outcome means
the request never reached the application and was never counted by
`demo_app_requests_total`. **Corrected**: `request_rate_rps` is now an explicit
**fitting target**, modeled from `(offered_rps, replica_state)` like every other
output (§24), never assumed identical to `offered_rps`. Three candidate model
families are added: bounded lookup/interpolation, a piecewise-linear regression
surface, and a tree-based regressor. Validation reports MAE in RPS, relative
error, the throughput-realization ratio `observed_rps/offered_rps` (never
assumed ≡1.0), and overload-regime error **separately** from the non-overload
regime. See `experiments/manifests/rl_dynamics_calibration_v1.yaml`
`signal_semantics.request_rate` and `candidate_model_families.request_rate`.

## 24. Causal/Modeling Structure — Joint-Input, Not Two-Stage (CORR-33, NEW)

A tempting alternative to §23 is a two-stage pipeline: first predict
`request_rate_rps` from `offered_rps`, then feed that *prediction* into the
other output models (CPU, latency, memory, in-progress, error-rate). **Rejected**
— this creates hidden teacher-forcing at fit time (models trained on the
*true* observed rate) that silently becomes propagated-error at simulation
time (models driven by a *predicted*, noisy rate), an inconsistency between
training and inference regimes with no principled way to characterize its own
error growth. **Frozen structure: joint-input.** All six steady-state output
models — `request_rate_rps`, `cpu_cores`, `latency_p95_ms`, `memory_bytes`,
`in_progress`, `error_rate_rps` — share exactly the same exogenous predictors,
`(offered_rps, replica_state)`, each fit and evaluated independently against
that same input pair. No output model ever consumes another output model's
prediction as an input.

## 25. Traffic Terminology — Offered / Sent / Server-Observed / Client-Completed (CORR-33, NEW)

Four distinct rates, precisely disambiguated to prevent silent substitution:
**offered** — the canonical planned arrivals from the frozen schedule file
(the dynamics model's input); **sent** — what the generator actually
dispatched (must equal offered for structural validity — a mismatch is the
existing `offered_not_equal_sent_due_to_undersized_generator_max_in_flight`
invalidity criterion, CORR-32 §21); **server-observed** — the
Prometheus-counted rate (§23's fitting target; may legitimately be less than
offered/sent under overload); **client-completed** — the generator's own
outcome log (success/timeout/connection_error breakdown), diagnostic-only,
**never** substituted for the RL request-rate observation or for
server-observed rate in any model.

## 26. Per-Pod Routing-Distribution Diagnostics (CORR-33, NEW)

Every run with `ready_replicas ≥ 2` at any point in its scored window now
records a per-pod routing diagnostic, using the exact PromQL pattern already
established by CORR-22 §4 (`job="demo-app"` filtering to avoid the documented
double-scrape double-count), grouped `by (pod)`, read as a raw counter delta
between the scored window's start and end (never `rate()`-smoothed). Three
explicit tiers, applied only when a pod's total request count in the window is
≥50 (below that, diagnostics are recorded but never used to invalidate):
**HEALTHY_VARIANCE** (every pod within [0.5×, 1.5×] fair share and CV ≤0.30 —
no action, ordinary load-balancing noise); **ROUTING_PINNING_SUSPECTED** (a
pod between 1% and 10% of fair share — flagged for manual review, *not*
automatically invalid); **STRUCTURAL_ROUTING_FAILURE** (a pod below 1% of
fair share while others receive substantial traffic — structurally invalid,
retried as a new attempt of the same planned slot, §28). This criterion never
requires perfectly equal distribution and never penalizes ordinary variance —
see `per_pod_routing_diagnostics` in the manifest.

## 27. HPA Isolation and Restoration Mechanics (CORR-33, NEW)

Reuses this project's own already-proven commands rather than inventing new
tooling: `kubectl delete hpa demo-app-hpa -n autoscaler-app --ignore-not-found`
(the exact pattern already used in `scripts/run_rl_experiment.sh` and in the
CORR-21/22/23 diagnostic sessions) before a session begins, and
`kubectl apply -f k8s/hpa/hpa.yaml` with a live-target=260% verification after
a session ends — **including on abort/error**, so a crash mid-campaign never
leaves the cluster HPA-less or RL-controller-active. Between cells, only the
execution harness sets fixed replica counts directly (`kubectl scale` +
`kubectl rollout status` wait); HPA stays absent for the entire session — it
is never merely reconfigured (e.g. min=max) as a substitute for deletion. See
`hpa_isolation_procedure` in the manifest for the full before/between/after
checklist.

## 28. Multi-Session Execution and Run/Attempt Identity (CORR-33, NEW)

The 102-run campaign will span multiple sessions. **Session semantics**: a
documented start checklist (cluster/HPA/Prometheus/Metrics-Server health,
idle stabilization) and end checklist (stop generators/collectors, restore
HPA, verify idle, preserve all artifacts) bound each session; resuming a
later session **continues from the next not-yet-VALID planned slot** in the
already-frozen randomized order (§13) — the order is never regenerated and
already-VALID runs are never rerun; a session boundary is purely operational,
never a scientific event. **Run/attempt identity**: a *planned slot*
(e.g. `fit_dyncal_r4_L3_repeat2`) is distinct from an *execution attempt*
(e.g. `fit_dyncal_r4_L3_repeat2_attempt1_<timestamp>`) — an invalid attempt is
preserved (never deleted or overwritten) and the same slot is retried as the
next attempt number; only VALID attempts enter the fitting/validation/
transition datasets. Required per-run metadata is extended accordingly
(planned slot ID, attempt number, session ID, campaign sequence index,
wall-clock UTC start/end, and the full chronological/config context listed in
`run_identity.required_run_metadata_fields`).

## 29. CPU/Memory Transition Semantics and Client/Server Latency (CORR-33, NEW)

**Memory** (§19): the `per_ready_replica_baseline × ready_replicas`
simplification applies **only** to steady-state cells, where
desired=current=ready=available by construction. During a transition,
aggregate memory must be modeled against **current** replicas and
transition/startup state — never silently multiplied by Ready alone. This
does not change the frozen V2 observation, which still normalizes by Ready.
**CPU**: cAdvisor's aggregate CPU includes not-yet-Ready containers, so during
scale-up `current > ready` can occur; the dynamics model must preserve
aggregate CPU, `current`, and `ready` as distinct fields and must never assume
a newly-created, not-yet-Ready pod contributes zero CPU. **Client/server
latency** (§16 context): both Prometheus/server p95 and client-side p95 are
preserved for every run; their absolute difference, ratio, cross-cell
correlation, and overload-regime divergence are preregistered analyses;
client p95 is **never** a policy input and never silently substitutes for
server p95 in model selection.

## 30. All-Zero 5xx Fallback and Precise Dynamics-Out-of-Support Semantics (CORR-33, NEW)

**All-zero 5xx fallback** (extends §16): if every fitting-and-validation run
shows server-visible 5xx==0, the empirical backend is not fit or claimed as a
meaningful 5xx classifier — it may output 0 **inside the observed calibration
domain only**, labeled `ZERO_ONLY_WITHIN_OBSERVED_CALIBRATION_DOMAIN`, and
must never be represented as predicting overload 5xx behavior outside that
domain. This does not by itself block final acceptance, but must be reported
before any PPO training begins; in that scenario the SLA-latency reward term
remains the primary overload-performance signal. Synthetic 5xx examples are
never manufactured to avoid this outcome. **Dynamics-out-of-support** (§20,
made precise): when a query falls outside the calibrated domain, no fabricated
observation is returned; the episode ends via Gymnasium's `truncated` flag
(not `terminated` — the episode did not reach a terminal MDP state, it left
the region the simulator can speak to, matching SB3's bootstrapped-value
handling of truncation vs. termination), with
`info["termination_reason"]="dynamics_out_of_support"`; this does **not**
increment `consecutive_invalid_steps` and does **not** set
`prometheus_unavailable` — it is a distinct condition requiring its own new
`types.py` constant, never overloading an existing `OVERRIDE_*` value.
Implementation is deferred to the empirical-backend phase; this phase freezes
semantics only.

## 31. Generator Safety Margin — Revised to 20% Headroom (CORR-33)

CORR-32's `max_in_flight=400` was justified by a ~25% margin computed loosely
against the matrix's peak (320). **Restated precisely and more
conservatively**: `required_max_in_flight = ceil(max_total_rps_in_matrix ×
timeout_seconds × 1.20)` — a 20% headroom factor applied directly to the
worst-case first-order bound, not an approximate comparison after the fact.
At the matrix's revised peak (64 RPS × 5.0s timeout), this computes to
`ceil(64 × 5.0 × 1.20) = 384`, still comfortably under the frozen `400`. Tests
must compute this value from the manifest's own
`revised_max_total_rps_ceiling`, `default_timeout_seconds`, and the new
`required_max_in_flight_headroom_factor` field — never hardcode `320` or `384`
independent of those inputs.

## 32. Execution-Harness Requirements (Specification Only) (CORR-33, NEW)

The future execution harness (not built in this phase) must, purely
mechanically and without deciding any scientific parameter itself: load the
frozen manifest; verify all 37 schedule SHA-256 hashes; identify the next
not-yet-VALID planned slot per §28's resumption rule; run the session-start
preflight (§28); temporarily remove HPA (§27); set the exact fixed replica
count and wait for exact readiness; start the NodePort tunnel; start the
Prometheus observer/collector; replay the exact canonical schedule file
byte-for-byte; preserve all artifacts regardless of outcome; validate run
structure against every criterion in `invalid_cell_structural_criteria`
(including the new per-pod-routing and generator-capacity criteria); mark the
attempt VALID or INVALID; on INVALID, retry the same slot as the next attempt
number; on VALID, advance to the next planned slot; and on session exit
(normal or aborted) restore the canonical cluster state (§27). All scientific
parameters — which cells, how many repeats, what schedule, what thresholds —
remain fixed by this frozen protocol; the harness only executes them.
**Superseded by §33**: "the next not-yet-VALID planned slot" is now a single,
precise lookup against the one authoritative global 102-slot sequence below,
not an implicit per-block concept.

## 33. The One Authoritative Global 102-Slot Execution Order (CORR-34, NEW)

§28 froze *session* semantics (resume from "the next not-yet-VALID planned
slot") but never froze a single, explicit, indexed list a harness could
mechanically consult — leaving what "next" means implicitly dependent on
four separate per-block randomized orders (fitting/validation/replica-
transition/workload-transition, §13). A future execution agent would have
had to decide, on its own, how to interleave those four independent
orders — precisely the kind of scientific discretion this protocol exists to
remove. **Frozen now**: `experiments/manifests/rl_dynamics_calibration_v1.yaml`'s
`global_campaign_order` section plus the generated, SHA-256-hashed
`configs/traffic_profiles/rl_dynamics_calibration/global_campaign_sequence.json`
together define the **one** authoritative, ordered list of exactly 102
planned run slots, indexed `1..102` contiguously and uniquely. It did not
previously exist; it is derived now, before any data collection, purely by
**concatenating** the already-frozen within-block orders — introducing no
new randomization — in one precommitted block sequence: fitting-repeat-1
(20) → validation-repeat-1 (9) → replica-transition-repeat-1 (8) →
workload-transition-repeat-1 (4) → fitting-repeat-2 (20) →
validation-repeat-2 (9) → replica-transition-repeat-2 (8) →
workload-transition-repeat-2 (4) → fitting-repeat-3 (20) = 102. Each
block's own internal order is preserved byte-for-byte from CORR-32's frozen
seeded shuffle. The generator, `scripts/generate_global_campaign_sequence.py`,
is a pure, network-free derivation (no `random`/`shuffle` call anywhere in
its own code) and is idempotent — re-running it reproduces a byte-identical
file, verified by test.

## 34. Resume Rule, Restated Against the Global Index (CORR-34)

Resume **always** means: the lowest `campaign_sequence_index` whose planned
slot has no VALID attempt yet. Never a new shuffle. Never "pick an easy run
first." A session boundary (§28) never alters this list or its order — the
list is immutable, identical in spirit to `schedule_registry.json`'s
already-frozen schedule hashes. This restates and sharpens §28's
resumption rule in terms of the one global index rather than four separate
per-block orders, removing the remaining ambiguity about how those orders
interleave across a real multi-session campaign.

## 35. Planned-Slot State Ledger — Specification Only (CORR-34, NEW)

A machine-readable campaign-state ledger format is specified (not created —
no run has been attempted, no ledger file exists): three states,
`NOT_ATTEMPTED` / `VALID` / `INVALID_RETRY_REQUIRED`, one entry per planned
slot, tracking its `campaign_sequence_index`, `planned_slot_id`, the valid
attempt ID if any, the full (append-only) list of all attempt IDs, and its
current state. The ledger must be fully reconstructable from persisted
per-attempt metadata (§29's required fields) alone if the ledger file
itself is ever lost — it is a derived convenience index, not the source of
truth. The frozen manifest and `global_campaign_sequence.json` remain
immutable; only the ledger and per-attempt artifacts change during
execution.

## 36. Per-Pod Routing Classification Made Exhaustive (CORR-34)

§26's 3-tier classification left gaps: a pod at, say, 20%, 40%, or 170% of
fair share, or a run with CV>0.30 while every individual share still looked
near-fair, mapped to **no** tier. **Corrected to four exhaustive tiers,
checked in this exact precedence order** so a severely starved pod is never
masked by an otherwise-acceptable aggregate CV:

1. **`INSUFFICIENT_ROUTING_SAMPLE`** — total traffic across eligible pods
   is below the sample-sufficiency threshold (§37), or the per-pod count
   cannot be reliably reconstructed (§39). Recorded, not a failure; routing
   is simply not used for structural validity on that run.
2. **`STRUCTURAL_ROUTING_FAILURE`** — (checked *before* healthy variance)
   at least one eligible pod's `fair_share_multiple < 0.01` while at least
   one other eligible pod receives substantial traffic. Structurally
   invalid; retried as a new attempt of the same planned slot.
3. **`HEALTHY_VARIANCE`** — every eligible pod's `fair_share_multiple` is
   within `[0.5, 1.5]` **and** the coefficient of variation is `≤0.30`.
   Ordinary load-balancing noise; no action.
4. **`ROUTING_PINNING_SUSPECTED`** — the explicit catch-all for every other
   sufficiently-sampled, non-structural case (a share between 1% and
   49.999% of fair, a share above 150% of fair, a CV above 0.30 even with
   individually near-fair shares, or any other substantial imbalance not
   severe enough to be structural). Never automatically invalidates; always
   preserved and surfaced for review.

Every eligible run now maps to **exactly one** of these four tiers — there
is no unclassified case. Fair share is computed as `fair_fraction =
1/N_eligible`, `observed_fraction_i = pod_count_i / total`,
`fair_share_multiple_i = observed_fraction_i / fair_fraction`; CV is frozen
to be computed over **raw per-pod request counts** (not shares — the two
scale equivalently, but exactly one must be chosen, and this is it).
Per-pod routing distribution remains a structural/path **diagnostic only**
— it is never added to the 11-element RL observation or to any dynamics
output contract, and causes no MDP version change.
**Superseded by §39-42**: the `fair_fraction = 1/N_eligible` whole-window
expectation used above is invalid whenever eligible pods have unequal
Ready-serving exposure (e.g. any replica-transition run) — see CORR-35.

## 37. Sample Sufficiency: Total-Window, Not Per-Pod (CORR-34)

§26's "applied only when a pod's window total is ≥50 requests" was
ambiguous, and read as a *per-pod* requirement would have hidden exactly
the severe-starvation pattern this diagnostic exists to catch (a starved
pod's own count is precisely what may be near zero). **Corrected**: the
threshold is evaluated against the **total** traffic across all eligible
pods — `total_request_count_across_eligible_pods ≥ 50 × N_eligible` — sized
so there is enough traffic for roughly 50 requests/pod under a fair
distribution. Below threshold: `INSUFFICIENT_ROUTING_SAMPLE`, not a
failure. This still lets a pod with 0 or 1 requests be correctly flagged
inside a sufficiently large total window.
**Superseded by §41**: this whole-window total threshold is replaced by a
per-pod EXPECTED-count threshold, again because it silently assumed equal
exposure.

## 38. Eligibility, Counter History, and Reset Handling (CORR-34)

**Eligibility** is Ready-serving exposure, not desired/current membership:
a pod counts toward `N_eligible` only for the portion of the window it was
actually Ready and capable of serving traffic; a newly-starting or
terminating pod is never classified as starved merely for not being Ready
the whole window, and a pod never Ready during the window is excluded
entirely (not counted as a 0-request eligible pod). **Counter history**:
transition-run routing is never computed as a bare
`counter_at_final_timestamp − counter_at_start_timestamp` (a scaled-down
pod may have no live series at the window's end) — instead, the full raw
counter-sample history over the window is queried, and each pod's first
and last valid sample *while its own series exists* is used; terminated
pods' historical series are retained from Prometheus whenever still
available. **Counter resets**: walking a pod's ordered raw samples, a
non-negative diff is summed normally; a negative diff is treated as a
reset and the running total continues from the new value (never
subtracted); `counter_reset_count` is recorded per pod, and a reset alone
never invalidates a run if the count remains reconstructable this way. If
a count cannot be reliably reconstructed, the run is classified
`INSUFFICIENT_ROUTING_SAMPLE` unless an independent structural failure is
separately established — missing traffic counts are never invented or
interpolated. The routing PromQL continues to filter `job="demo-app"`
only, never `job="demo-app-loadtest"` (CORR-22 §4's double-scrape
caveat).
**Eligibility (first/last Ready timestamp, serving duration) and counter
history/reset mechanics above are UNCHANGED by CORR-35** — they remain the
building blocks CORR-35's interval reconstruction is built on top of.

## 39. The CORR-34 Defect: Unequal Ready-Serving Exposure Breaks 1/N (CORR-35, NEW)

§36's `fair_fraction = 1/N_eligible` whole-window expectation silently
assumes every eligible pod had *equal* opportunity to serve traffic across
the entire routing-analysis window. This holds for an ordinary steady-state
cell (all eligible pods Ready for the whole window) but is **false** for
any replica-transition run: if Pod A is Ready for 180s and Pod B only for
the last 20s after scale-up, a 50/50 whole-window expectation is not a
meaningful fairness baseline — Pod B was never in a position to receive
half the traffic, through no fault of routing. **Corrected**: expected
routing opportunity is now computed at the **interval** level, weighted by
the traffic that actually occurred while each pod was Ready — not by
wall-clock Ready duration alone (§40). This automatically incorporates
unequal exposure, changing pod membership, and changing workload intensity
in one formula, without a separate correction term for each.

## 40. Interval-Level Expected Routing Opportunity (CORR-35, NEW)

For each reconstruction interval `j` (aligned to the frozen 10-second
Prometheus/collector cadence, `configs/experiment.yaml:
prometheus.scrape_interval_seconds` — never a coarser 1-minute `rate()`),
compute `delta_total_requests_j`: the reconstructed, reset-aware, positive
delta of the aggregate legitimate request counter
(`demo_app_requests_total{job="demo-app",route="/work"}`, never
`job="demo-app-loadtest"`) over that interval. Determine `ready_pod_set_j`
— the eligible pods actually Ready at interval `j`'s start timestamp,
sourced from the Kubernetes API's per-pod Ready condition (this project's
established preference for authoritative object state over coarser
Prometheus/kube-state-metrics polling, per `hpa_status_collector.py`'s own
precedent) — and `N_ready_j = len(ready_pod_set_j)`. Then:

```
expected_increment_i_j = delta_total_requests_j / N_ready_j   if pod i in ready_pod_set_j, else 0
expected_request_count_i = sum_j(expected_increment_i_j)  over every interval j in the window
```

This is deliberately **not** weighted by wall-clock Ready duration alone —
a pure-duration weighting would misrepresent expected opportunity whenever
traffic intensity varies during the window, exactly what workload-step
transitions and overload-regime cells exist to vary. No sub-interval
precision is invented: a Ready/NotReady transition observed between two
interval boundaries is attributed to whichever boundary sample first
reflects it.

## 41. Actual Reconstruction, the Routing-Opportunity Ratio, and Ratio-CV (CORR-35)

`actual_request_count_i` reuses §38's counter-history and reset mechanics
unchanged (first/last valid sample while the pod's series exists; positive
diffs summed; negative diffs treated as a reset, continuing from the new
value; `counter_reset_count` recorded; missing counts never invented). The
canonical comparison quantity is now:

```
routing_opportunity_ratio_i = actual_request_count_i / expected_request_count_i
```

with `1.0` representing exactly-expected routing. For a steady-state run
where every eligible pod shares equal exposure, `N_ready_j` is constant and
equal to `N_eligible` for every interval, so `expected_request_count_i`
reduces exactly to §36's `1/N_eligible` result — this is a
correction/generalization of the same diagnostic, not a change of its
purpose. Coefficient of variation is now computed over the **normalized
ratios** of assessable pods (§42), not raw request counts (§36's
count-based CV is invalid once pods can have unequal exposure — a
short-lived pod's naturally smaller raw count would inflate count-CV even
under perfectly proportional routing).

## 42. Assessability, Run-Level Sample Sufficiency, and Reconstruction Consistency (CORR-35)

§37's whole-window total-traffic threshold is replaced: a pod is
**assessable** if `expected_request_count_i ≥ 50` — evaluated against
EXPECTED count, never actual count, so a genuinely starved pod
(expected=100, actual=0) remains assessable and can still trigger
structural failure, while a short-lived pod with little Ready-serving
opportunity (expected=7, actual=1) is correctly excluded rather than
falsely flagged as starved. Routing structural classification requires at
least **2 assessable pods**; fewer classifies `INSUFFICIENT_ROUTING_SAMPLE`
(not a failure). A new **reconstruction consistency check** additionally
verifies that summing all eligible pods' independently-reconstructed
`actual_request_count_i` matches the aggregate `delta_total_requests_j`
reconstruction (computed as the reset-aware reconstruction of the *summed*
per-pod raw values at each shared interval boundary) — frozen tolerance
**0 requests**, since both reconstructions apply identical reset-aware
arithmetic to the same exact-integer Counter and neither involves
smoothing or extrapolation. A joint-observability gap (a scrape miss for
one pod at a boundary) excludes that boundary consistently from both sides
rather than fabricating a value; a discrepancy beyond tolerance classifies
`INSUFFICIENT_ROUTING_SAMPLE` unless an independent structural failure is
separately established.

## 43. Classification, Corrected (CORR-35)

The same four tiers, same precedence order, now expressed in ratio terms:
`STRUCTURAL_ROUTING_FAILURE` requires **only** at least 2 assessable pods
and at least one assessable pod's `routing_opportunity_ratio_i < 0.01` —
§36's secondary "another pod receives substantial traffic" clause is
**removed as unnecessary**: if a pod had ≥50 *expected* requests (meaning
the legitimate traffic that occurred during its own Ready exposure was
itself substantial) yet received <1% of that expected opportunity,
starvation is already established numerically by the ratio alone.
`HEALTHY_VARIANCE` requires every assessable pod's ratio within `[0.5,
1.5]` **and** ratio-CV `≤0.30`. `ROUTING_PINNING_SUSPECTED` remains the
exhaustive catch-all for every other sufficiently-sampled, non-structural
case (a ratio between 1% and 49.999% of expected, a ratio above 150% of
expected, a ratio-CV above 0.30 even with individually near-1.0 ratios).
Every eligible run still maps to exactly one tier.

## 44. Worked Examples, Including the First Official Slot (CORR-35, NEW)

Three deterministic protocol-level examples are frozen, directly relevant
to `campaign_sequence_index=1` (`fit_dyncal_r2_L4_repeat1`, a fixed
2-replica steady-state run where equal Ready exposure is expected after
stabilization): (1) 2 Ready pods for the whole window, 2880 total
legitimate requests → expected 1440/pod, actual 1450/1430 →
`HEALTHY_VARIANCE`; (2) the same exposure, actual 0/2880 →
`STRUCTURAL_ROUTING_FAILURE`; (3) a replica-transition example — Pod A
Ready for the full high-traffic interval, Pod B Ready only after
scale-up — demonstrating that expected opportunity is **traffic-weighted**,
not duration-weighted, and does **not** default to a naive 50/50
whole-window split. No traffic was sent to produce these examples; they
are pure arithmetic worked through the frozen formulas above.

## Change Log

- **v1** (Phase 15B.2, CORR-31, 2026-09-09): initial freeze.
- **v1, corrected** (Phase 15B.2A, CORR-32, 2026-09-09): historical-data
  substitution disabled; one duplicate physical cell deduplicated; generator-
  capacity-audited maximum load ceiling (64 RPS, down from 96); deterministic
  arrival process frozen with 37 canonical schedules generated and hashed
  offline; decision-aligned extraction timestamps precisely frozen; replica-
  transition matrix corrected to adjacent ±1 pairs; workload-step-transition
  runs added to the budget (previously omitted); transition timing precisely
  frozen for both kinds; acceptance-criterion mathematics made explicit and
  divide-by-zero-safe; error-rate validation corrected to balanced metrics;
  monotonicity requirements scoped to physically justified axes only;
  stochastic residual-model candidates preregistered; dynamics-out-of-support
  distinguished from Prometheus unavailability; invalid-cell policy extended
  with generator/harness criteria; official run count corrected to 102.
  `protocol_version` kept at 1 (pre-execution correction, not a post-data
  methodology change) per explicit instruction.
- **v1, further clarified** (Phase 15B.2B, CORR-33, 2026-09-09): request-rate
  realization frozen as an explicit fitting target, never assumed identical to
  offered_rps (§23); causal structure frozen as joint-input, two-stage
  offered→predicted-rate→other-outputs explicitly rejected (§24); traffic
  terminology (offered/sent/server-observed/client-completed) precisely
  disambiguated (§25); per-pod routing-distribution diagnostics with an exact
  3-tier criterion added (§26); HPA isolation/restoration mechanics frozen,
  reusing existing repository tooling (§27); multi-session execution and
  run/attempt identity semantics frozen (§28); CPU/memory transition
  semantics and client/server latency preregistration clarified (§29);
  all-zero-5xx fallback and precise `dynamics_out_of_support` (truncated, not
  terminated) semantics frozen (§30); generator safety margin restated as an
  exact 20%-headroom formula, computed value 384 (§31); execution-harness
  requirements specified as a mechanical checklist that decides no scientific
  parameters (§32); residual/noise sampling unit clarified as whole-run
  trajectory blocks, never pointwise-shuffled, to preserve cross-metric and
  temporal correlation (extends §18, see manifest
  `stochastic_residual_model.sampling_unit`). `protocol_version` kept at 1 —
  still no calibration measurements collected. No traffic sent, no
  Kubernetes mutation, no RL/PPO training occurred in this phase; the 20/9/8/4/
  102/37/64/400/5s frozen values and CORR-1 through CORR-32 are unchanged.
- **v1, execution-closed** (Phase 15B.2C, CORR-34, 2026-09-09): froze the ONE
  authoritative global 102-slot execution order (`global_campaign_order` /
  `global_campaign_sequence.json`), derived purely by concatenating the
  already-frozen within-block orders in one precommitted block sequence,
  introducing no new randomization (§33); restated the resume rule against
  this single global index (§34); specified a machine-readable planned-slot
  state ledger format, not yet created (§35); corrected the per-pod routing
  classifier from 3 gap-prone tiers to 4 exhaustive tiers —
  `INSUFFICIENT_ROUTING_SAMPLE` / `STRUCTURAL_ROUTING_FAILURE` /
  `HEALTHY_VARIANCE` / `ROUTING_PINNING_SUSPECTED`, checked in that
  precedence order, with an explicit catch-all so no eligible run is ever
  unclassified (§36); corrected the sample-sufficiency rule from an
  ambiguous per-pod reading to an explicit total-window threshold,
  `≥50×N_eligible` (§37); froze Ready-serving-exposure eligibility, full
  counter-history reconstruction for disappearing pods, and counter-reset
  handling (§38). `protocol_version` kept at 1 — still no calibration
  measurements collected. No traffic sent, no Kubernetes mutation, no
  RL/PPO training occurred in this phase; the 20/9/8/4/102/37/64/400/5s
  frozen values, model families, acceptance criteria, residual/noise design,
  HPA isolation rules, session semantics, and CORR-1 through CORR-33 are
  unchanged.
- **v1, routing mathematics corrected** (Phase 15B.2D, CORR-35, 2026-09-09):
  corrected CORR-34's whole-window `fair_fraction = 1/N_eligible` routing
  expectation, which is invalid whenever eligible pods have unequal
  Ready-serving exposure (e.g. any replica-transition run) (§39); replaced
  it with an interval-level, traffic-weighted expected-routing-opportunity
  calculation — `expected_increment_i_j = delta_total_requests_j /
  N_ready_j`, summed across the window — that reduces exactly to CORR-34's
  result for equal-exposure steady-state runs (§40); replaced count-based
  CV with ratio-based CV over the new canonical
  `routing_opportunity_ratio_i = actual_request_count_i /
  expected_request_count_i` (§41); replaced the whole-window total-traffic
  sample threshold with a per-pod EXPECTED-count assessability rule
  (`≥50`) plus a run-level minimum of 2 assessable pods, and added a
  reconstruction consistency check (aggregate-vs-summed-per-pod, frozen
  tolerance 0 requests) (§42); removed the vague "another pod receives
  substantial traffic" clause from `STRUCTURAL_ROUTING_FAILURE`, now
  sufficient on the ratio alone (§43); froze three deterministic worked
  examples directly relevant to `campaign_sequence_index=1` (§44).
  `protocol_version` kept at 1 — still no calibration measurements
  collected. No traffic sent, no Kubernetes mutation, no RL/PPO training
  occurred in this phase; the 20/9/8/4/102/37/64/400/5s frozen values, the
  global 102-slot campaign order and its SHA-256 (byte-identical to Phase
  15B.2C), the first planned slot (`fit_dyncal_r2_L4_repeat1`), model
  families, acceptance criteria, residual/noise design, HPA isolation
  rules, session semantics, and CORR-1 through CORR-34 are unchanged.
