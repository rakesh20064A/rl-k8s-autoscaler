# RL Empirical Dynamics Calibration Protocol — v2 (CORR-37, POST-DATA CORRECTION)

**Status**: Frozen by Phase 15B.3B, correction reference **CORR-37**, immediately following
**CORR-36**'s discovery. This is the **first post-data** correction to the calibration
protocol. `docs/rl_dynamics_calibration_protocol.md` (v1, corrected through CORR-35) remains
the base narrative document for everything not explicitly redefined here — this file governs
**only** execution-validity/fit-eligibility semantics, the routing-conservation bug-fix
confirmation, and ledger semantics under the resulting two-axis model. The machine-readable
counterpart is `experiments/manifests/rl_dynamics_calibration_v2.yaml`, which explicitly
inherits everything else (the 20/9/8/4/102/37/64/400/5s matrix, all model families, all
acceptance criteria not superseded here) unchanged, by reference, from
`experiments/manifests/rl_dynamics_calibration_v1.yaml`.

```
dynamics_calibration_protocol_version: 2   (NEW major version -- this is a POST-DATA correction, unlike v1's CORR-31..35 which were all pre-execution)
frozen_in_phase: 15B.3B
correction_reference: CORR-37 (follows CORR-36)
frozen_after_observing_first_empirical_attempt: true
first_observed_attempt_id: fit_dyncal_r2_L4_repeat1_attempt1_20260909T140146Z
```

## Why This Had to Be a New Version, Not an In-Place v1 Edit

CORR-31 through CORR-35 corrected the *preregistration* before any calibration data existed —
exactly like CORR-27 corrected the RL MDP contract before any environment code existed. This
correction is different in kind: it was written **after** observing attempt1's real evidence.
Blurring that distinction — silently folding a post-data methodology decision into the same
`protocol_version: 1` label used for pre-data clarifications — would make it impossible for a
future reviewer to tell whether any given rule was chosen before or after seeing results. So
`protocol_version` is bumped to `2`, and the file is new, not edited in place. v1 remains
byte-for-byte intact as the historical record of what rules governed attempt1 when it ran.

## What Attempt1 Revealed (CORR-36 Recap)

At all four decision-aligned timestamps of `campaign_sequence_index=1`
(`fit_dyncal_r2_L4_repeat1`), only 1 of the intended 2 replicas was actually Ready, and one pod
was killed and restarted by kubelet mid-window. Forensic audit (Phase 15B.3B, below) confirmed
via Kubernetes Events that this was a **liveness-probe timeout** (`Liveness probe failed: ...
context deadline exceeded`, 5 consecutive failures at 10s intervals) — the app was too
CPU/request-saturated to answer its own `/health` endpoint within the frozen 3-second probe
timeout, and kubelet reacted exactly as configured. This is not a code defect, not OOMKilled,
not a harness bug — it is a genuine, load-induced reliability finding. The protocol had no rule
for what such a finding means for a nominally "steady-state" fitting cell.

## The Two-Axis Resolution

**Axis A — `execution_structural_validity`**: did the harness actually execute the planned
experiment mechanically correctly (right schedule, right path, right replica target, HPA
absent, no dropped_overload, etc.)? Legitimate application behavior under overload — high
latency, timeouts, connection errors, CPU saturation, readiness loss, a liveness-triggered
restart, or the app's own `/metrics` scrape target going unhealthy — is **never** structural
invalidity on its own. Retrying a run merely because the tested application behaved badly under
the load it was deliberately given would let the campaign selectively erase inconvenient
evidence until a favorable trajectory happened to occur — the opposite of this project's
validity-before-performance discipline (CORR-17 lineage).

**Axis B — `steady_state_fit_eligibility`**: independently of Axis A, can this run — or a
specific decision-aligned sample within it — actually be used as fixed-replica steady-state
fitting data? A decision point is fit-**in**eligible whenever, at that exact timestamp,
desired/current/ready/available do not all equal the cell's intended replica count (`current=2,
ready=1` is never treated as the same operating point as `2/2/2/2`), or when the application's
own metrics endpoint was unobservable under load. Two frozen, non-overlapping exclusion reasons:
`LOAD_INDUCED_HEALTH_INSTABILITY` and `APPLICATION_TARGET_UNOBSERVABLE_UNDER_LOAD`. Neither
implies structural invalidity; both simply mean that specific decision point cannot enter the
regression dataset. Missing/ineligible points are **never** imputed — all four frozen targets
(60/90/120/150s) are always reported, each with its own observability and eligibility flags.

## Cell-Level Stable-Calibration-Support Rule (Frozen Before Attempts 2/3)

Precommitted now, before any repeat of `dyncal_r2_L4` or any other cell is observed: a 3-repeat
fitting cell reaches `STABLE_CALIBRATION_SUPPORT` only if at least 2 of its 3 official repeats
each provide at least 2 of 4 fit-eligible decision points; a 2-repeat interpolation-validation
cell needs at least 1 of 2 repeats meeting that same per-repeat bar. Falling short:
`OUTSIDE_STABLE_CALIBRATION_SUPPORT` for that cell — no simulator output is ever invented there.
This threshold is chosen and frozen now specifically so it cannot be chosen retroactively after
seeing which repeats look favorable.

## Routing Conservation — Confirmed, Not Relaxed

Attempt1's original routing summary reported `expected=591` vs `actual=1427` — not because
exact-zero conservation is unachievable, but because of two compounding bugs in
`compute_expected_and_actual()`: the interval aggregate was computed from Ready-**and**-
observable pods only (silently dropping real traffic whenever the Ready pod for an interval
lacked a joint sample there), and `actual` was reconstructed as an unconstrained whole-span
total per pod, never scoped to the same interval set as `expected`. Fixed: the aggregate for
each interval is now built from **every** pod with a valid joint sample at that interval's two
boundaries (readiness determines only who shares the resulting expected credit, never whose
traffic counts toward the aggregate), and `actual` is the sum of each pod's own per-interval
delta over the identical interval set — which conserves by construction. Re-run against
attempt1's real preserved raw data: `conservation_error == 0.0` exactly. **The frozen zero
tolerance (`per_pod_routing_diagnostics.reconstruction_consistency_check`) is retained
unchanged** — this was a reference-implementation bug fix, never a scientific threshold
relaxation, and it is never applied by choosing whatever value happens to make attempt1 look
clean; the corrected code was validated against a from-first-principles hand-derivation before
being trusted (see `tests/unit/test_dynamics_calibration_harness.py::TestRoutingConservationCorr37`).

## Ledger Semantics Under the Two-Axis Model

The campaign ledger tracks Axis A only — whether the planned slot was executed — never whether
the resulting samples were pleasant. If `execution_structural_validity == VALID`, the slot
becomes ledger state `VALID` with `valid_attempt_id` set, even when
`steady_state_fit_eligibility` is partial or fully ineligible, **provided** no required evidence
artifact is missing due to an actual harness/collector failure (which would itself make Axis A
`INVALID`). Fit-eligibility is recorded in the attempt's own analysis metadata, never as a 4th
ledger state. A slot's provisional/superseded classification history is preserved in that
metadata, never silently erased by a later, better-informed ledger update.

## Change Log

- **v1** (Phase 15B.2 → 15B.2D, CORR-31 through CORR-35): pre-execution preregistration and its
  corrections. `docs/rl_dynamics_calibration_protocol.md`.
- **v2** (Phase 15B.3B, CORR-37, 2026-09-09): first **post-data** correction, made after
  observing `campaign_sequence_index=1`'s real evidence (CORR-36). Freezes
  `execution_structural_validity` (Axis A) and `steady_state_fit_eligibility` (Axis B) as
  independent classifications; freezes `LOAD_INDUCED_HEALTH_INSTABILITY` and
  `APPLICATION_TARGET_UNOBSERVABLE_UNDER_LOAD` as non-structural, fit-eligibility-only exclusion
  reasons; freezes decision-point-level observability/eligibility bookkeeping (all four targets
  always reported, never imputed); freezes the cell-level `STABLE_CALIBRATION_SUPPORT` rule
  (≥2/3 fitting repeats, ≥1/2 validation repeats) before attempts 2/3 of any cell are executed;
  confirms the routing-conservation zero-tolerance requirement is achievable and fixes the
  reference implementation's aggregation-scope bug (not a threshold change); freezes ledger
  semantics so Axis A alone determines ledger state. The 20/9/8/4/102/37/64/400/5s matrix,
  global campaign order, all schedule hashes, HPA baseline (260%), and MDP/reward contracts
  (V2) are all unchanged from v1. No traffic was sent to produce this correction.
