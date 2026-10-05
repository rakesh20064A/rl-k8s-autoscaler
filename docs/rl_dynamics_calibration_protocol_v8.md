# RL Empirical Dynamics Calibration Protocol — v8 (CORR-43, SEVENTH POST-DATA CORRECTION)

**Status**: Frozen by Phase 15B.3P.2, correction reference **CORR-43**, following **CORR-42**
(v7). This is the **seventh post-data** correction to the calibration protocol, and the fourth
(after v5/CORR-40, v6/CORR-41, and v7/CORR-42) to be frozen strictly after reviewing a specific
attempt's real evidence — `campaign_sequence_index=12` attempt4, the SAME attempt whose evidence
motivated v7/CORR-42, re-analyzed here with a deterministic (exhaustive, not Monte-Carlo) timing
model. This file governs **only** evidence-**window partitioning**: `partition_into_evidence_
windows()` gains an optional `min_window_seconds` parameter that merges a pathologically short
trailing remainder window into its preceding neighbor, so no window narrower than a deterministically
derived safety threshold is ever constructed. It does **not** touch CORR-41's route-integrity
classification rules, CORR-42's dataplane-acquisition schedule, the V5 workload-transport mechanism
CORR-40 froze, the base matrix, the workload, or the schedules.
`docs/rl_dynamics_calibration_protocol_v7.md` (CORR-42's acquisition schedule),
`docs/rl_dynamics_calibration_protocol_v6.md` (CORR-41's classification contract), and earlier
protocol docs remain the narrative documents for everything not explicitly redefined here. The
machine-readable counterpart is `experiments/manifests/rl_dynamics_calibration_v8.yaml`, which
inherits everything else unchanged, by reference, from v7.

```
dynamics_calibration_protocol_version: 8   (seventh POST-DATA correction)
frozen_in_phase: 15B.3P.2
correction_reference: CORR-43 (follows CORR-42)
frozen_after_observing_empirical_attempts: true
observed_attempt_ids:
  - fit_dyncal_r8_L3_repeat1_attempt4_20260919T091715Z   (same attempt that motivated CORR-42, re-analyzed deterministically)
route_integrity_protocol_version: 6   (UNCHANGED)
dataplane_acquisition_protocol_version: 2   (UNCHANGED)
window_partitioning_protocol_version: 2   (was implicitly 1 under the CORR-39/Phase 15B.3M.4 original always-separate-remainder rule)
transport_protocol_version: 5   (UNCHANGED)
```

## Why This Had to Be a New Version, Not an In-Place V7/CORR-42 Edit

Phase 15B.3P.1's own report was explicit that V7's cadence reduction and reactive/anchor sampling
were a **probabilistic** improvement (a 4-second trailing window's risk of insufficient interior
samples fell from ~88.9% at V6's realized cadence to ~17.3% at V7's), not a deterministic guarantee.
A deeper, deterministic (exhaustive fine-grid enumeration, not random Monte-Carlo sampling)
re-analysis of the SAME timing geometry proves this precisely: for a periodic sampler with
worst-case realized period `T`, a window of width `r` is guaranteed exactly `floor(r/T)` interior
samples at its worst possible phase offset — this is *tight*, confirmed by exhaustive sweep, not an
approximation. A guarantee of `>=2` samples therefore requires `r >= 2T`; no finite sampling cadence
can guarantee even *one* interior sample once the window width can be adversarially placed below one
full cadence period. This is a property of the **window-partitioning rule** (which can construct an
arbitrarily short trailing window whenever a material interval's length isn't an exact multiple of
10 seconds) — not of the sampling cadence — so no amount of further cadence tuning inside V7's own
acquisition layer could ever fully close it for the narrowest possible remainders. Deciding to
change how windows are constructed, after this deterministic re-analysis, is a post-data methodology
decision by this project's own established precedent (CORR-27/32/37/38/39/40/41/42) — it must be
versioned transparently, not folded into an in-place V7 edit.

## CORR-43 sec.A: Rationale

1. V7/CORR-42's cadence reduction and reactive/anchor sampling reduce, but do not eliminate, the
   probability that a short trailing route-integrity evidence window contains fewer than 2 interior
   dataplane samples.
2. An exhaustive (fine-grid, not Monte-Carlo) phase-offset analysis proves the exact worst case:
   `min_interior_samples(r, T) = floor(r/T)`, achieved at the adversarial phase offset. A guarantee
   of `>=2` samples requires `r >= 2T`; a guarantee of `>=3` samples (needed for resilience to a
   single identity/regime transition landing anywhere in the window, independent of whether CORR-42's
   own reactive resample completes before the window's end) requires `r >= 3T`.
3. The current window-partitioning rule (`partition_into_evidence_windows()`, frozen by Phase
   15B.3M.4/CORR-39) can construct a trailing window of arbitrarily small width whenever a material
   interval's length modulo 10s is small — attempt4's own `[175,179)` (a 4-second remainder) is a
   concrete instance.
4. The fix must not declare short tails automatically `CONFIRMED`, borrow evidence from neighboring
   windows, weaken FAILED precedence, or reduce the full-temporal-coverage requirement — it must
   eliminate the pathologically short window from being **constructed** in the first place, while
   still covering the entire material interval exactly once.
5. Therefore the window-construction change is versioned transparently as CORR-43/Protocol V8,
   rather than hidden inside an in-place CORR-42/V7 edit.

## CORR-43 sec.B: The Deterministic Derivation

Using `DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8 = 3.0` (a conservative, measured-with-
margin bound on V7's real inter-sample gap — 2.0s nominal sleep plus command-duration/scheduling
jitter; Phase 15B.3P.1 measured a live mean of 0.191s and worst-observed 0.306s over 7 real
invocations; Phase 15B.3P.2 re-measured with a larger live sample during its own U1–U6 validation —
see this phase's report for the exact figures):

```
count_guarantee_threshold_seconds       = 2 * T = 6.0s   (>=2 samples, no-transition case)
transition_resilience_threshold_seconds = 3 * T = 9.0s   (>=3 samples, resilient to one transition anywhere in the window)
MIN_EVIDENCE_WINDOW_SECONDS_V8          = 9.0s           (the STRICTER of the two, adopted)
```

Verified by exhaustive fine-grid enumeration (not random sampling) in
`tests/unit/test_dynamics_calibration_harness.py::TestDeterministicTailWindowTimingAnalysisPhase15B3P2`.
The **irreducible limit**: a whole material interval shorter than one full window (10s) has no
preceding window to merge into — this is the sole case this correction cannot close, and it is
honestly retained (a standalone, possibly short, single window), never fabricated into a false
guarantee. Under the existing 25% materiality gate (`is_interval_material()`) and realistic
official-scale traffic durations, a qualifying material interval is at least 45 seconds long, so
this irreducible case does not arise at official scale — it is a diagnostic-scale-only concern,
exercised deliberately by this phase's U3 validation scenario.

## CORR-43 sec.C: The Window-Partitioning Change

`partition_into_evidence_windows()` gains an optional `min_window_seconds: Optional[float] = None`
parameter. When `None` (the default), behavior is byte-for-byte identical to every prior version —
no historical call site and no historical attempt's already-persisted `evidence_windows` are
affected. When supplied (as `MIN_EVIDENCE_WINDOW_SECONDS_V8` is, for every future attempt from this
phase onward): if the trailing window's width is strictly less than `min_window_seconds` **and** at
least one other (full) window already exists, the trailing window is merged into the immediately
preceding window — extending that window's end to the interval's own end — instead of being appended
as its own separate window. This does not apply when the interval length is an exact multiple of the
window size (no remainder exists), when the interval fits entirely within a single window (no
preceding window exists to merge into), or when the remainder is already `>= min_window_seconds`
(already safe standalone — left separate to preserve temporal resolution where merging would cost
resolution for no safety benefit).

**Coverage invariants**, proven for both the default and merged case:
`union(windows) == [interval_start, interval_end)` exactly; windows are pairwise non-overlapping
except at their shared boundary points; no gap; no duplicated covered duration; chronological
ordering preserved. `classify_route_integrity_window()`, `classify_route_integrity()`,
`compute_dataplane_ingress_evidence_windows()`, `compute_delivered_request_evidence_windows()`, and
`aggregate_observer_evidence_for_window()` are byte-for-byte unchanged — this correction only
changes which `(window_start, window_end)` boundaries are fed into them.

## CORR-43 sec.D: Route-Integrity Classification and Acquisition — Completely Unchanged

Every CORR-41/V6 classification rule and every CORR-42/V7 acquisition mechanism is preserved
verbatim: the three-source OR'd `CONFIRMED` rule; **FAILED precedence** (decisive and unconditional,
regardless of window width or any dataplane evidence, old or new); the materiality criterion,
window-local attribution, and the prohibition on cross-window/cross-identity smearing; attempt-level
aggregation and the Axis-A consequence; the V7 dataplane-acquisition schedule (2.0s baseline cadence,
reactive resample, end-of-run anchor sample). `route_integrity_protocol_version` stays **6** and
`dataplane_acquisition_protocol_version` stays **2** — this file introduces a new, independent
`window_partitioning_protocol_version` field (1 → 2) specifically so a partitioning-only change is
never conflated with a classification or acquisition change in any attempt's provenance.

## CORR-43 sec.E: Non-Retroactivity

Attempt4 was **not** re-partitioned or re-classified under this rule — Protocol V8 was frozen
strictly *after* a deterministic re-analysis of attempt4's already-collected evidence. What the
merged partitioning "would have shown" for attempt4's own `[175,179)` window is assessed only via a
separate, clearly labeled `NON_OFFICIAL_COUNTERFACTUAL_ONLY` analysis (this phase's own report) that
never changes attempt4's official stored classification or its already-persisted `evidence_windows`.
Attempt4 remains, permanently: `SEQUENCE_INDEX_12_ATTEMPT4_INVALID`, `ROUTE_INTEGRITY_UNVERIFIED`,
`AXIS_A_INVALID`. Ledger state is unchanged by this phase: `slot12.state = INVALID_RETRY_REQUIRED`,
`slot12.valid_attempt_id = None`, `slot12.all_attempt_ids = [attempt1, attempt2, attempt3,
attempt4]`, `next_resumable_index() == 12`, `slot13 = NOT_ATTEMPTED`. This file applies
**prospectively only**, starting with a future, separately-authorized attempt5 — Phase 15B.3P.2
does not execute attempt5 or index13.

## Phase 15B.3P.3 Audit Amendment (Pre-First-Use)

V8 had not yet been used by any **official** attempt when Phase 15B.3P.3 audited it, so this
section is amended in place per this project's own standing convention (pre-first-use
implementation-correctness fixes are amended in place; anything that changes a declared,
already-used protocol layer requires a new correction/version instead — this amendment does
neither of the latter).

**Finding**: the `floor(r/T)` derivation above is a sound, exhaustively-verified mathematical
*implication* — **if** every real inter-sample gap stays `<= T`, **then** a window of width `>= k·T`
is guaranteed `>= k` interior samples, for any phase offset. What this document did **not** establish
is that `T = 3.0s` is itself a hard, OS/Docker-enforced **real-time** bound. It is not: the production
sampler (`_sample_dataplane_ingress_periodically()`) sleeps a fixed 2.0s **after** each sample's two
sequential `docker exec` calls complete — a relative-sleep design, not an absolute monotonic-deadline
design — so the realized gap is `2.0s + command_duration`, and each `docker exec` call carries only a
5-second timeout. A pathologically slow but still-successful command could in principle push a single
gap above 3.0s, and a Python process invoking Docker on macOS is not a hard-real-time system in the
first place. **This project therefore does not claim "a guaranteed maximum inter-sample gap" as a
real-time property.**

**What remains true, and is now load-bearing instead**: `T=3.0s` is an empirically well-supported
value across three independent live measurements (Phase 15B.3P.1: n=7, max 0.306s command duration,
realized gap ≈2.3s; Phase 15B.3P.2: n=24, max 0.644s, realized gap ≈2.64s; Phase 15B.3P.3: see its
own report for a further, larger live sample) — every one comfortably under the bound. Every real
inter-sample gap is now **explicitly recorded** (`actual_gap_seconds_from_previous_sample`,
`acquisition_gap_exceeded_assumed_bound`, summarized by `summarize_dataplane_acquisition_gaps()`) for
post-hoc audit, and `compute_dataplane_ingress_evidence_windows()` **can never fabricate** a
confirming pair regardless of how large a gap is — an unusually large gap can only ever result in
fewer (or zero) confirming pairs, honestly reflected as `UNVERIFIED`, never as a false `CONFIRMED`.
The window-partitioning correction itself remains fully justified independent of this distinction: it
removes a known **geometric** pathology (an arithmetic-remainder-driven short window) that existed
regardless of how tightly `T` could ever be bounded.

**Multi-transition finding** (Phase 15B.3P.3 sec.7/8): a valid same-identity confirming pair exists
if and only if at least one pair of *consecutive* samples shares an identity — independent of how
many *distinct* identities appear in a window, and independent of the total transition *count*. A
window whose identity changes on every consecutive sample (e.g. `A→B→C→D`, or an oscillating
`A→B→A→B`) yields zero confirming pairs regardless of sample count — honestly `UNVERIFIED`, never
fabricated. This is an inherent limit of any interval/delta-based evidence source when identity churn
outpaces the sampling cadence, not a defect specific to V8, and it was not previously stated
explicitly in this document.

**Implementation hardening added** (pre-first-use, no version bump): a `subprocess.TimeoutExpired`
from either `docker exec` call — previously **uncaught** (not an `OSError` subclass), capable of
silently killing the entire dataplane sampler thread for the rest of an attempt — is now caught
exactly like any other observer failure in `sample_dataplane_ingress_rules()`.

**Version decision**: implementation hardening and wording correction only. `route_integrity_
protocol_version`, `dataplane_acquisition_protocol_version`, and `window_partitioning_protocol_
version` are all unchanged — no declared acquisition-schedule parameter (cadence, resample delay,
anchor trigger) or classification rule changed.

## What Remains Unaffected

Everything CORR-41/V6 and CORR-42/V7 preserved remains preserved: the three-source route-integrity
classification rule and its `route_integrity_protocol_version=6`; the dataplane-acquisition schedule
and its `dataplane_acquisition_protocol_version=2`; `classify_route_integrity_window()`,
`classify_route_integrity()`, and `compute_dataplane_ingress_evidence_windows()` byte-for-byte
unchanged; the 20 fitting cells, 9 interpolation cells, 8 replica-transition types, 4
workload-transition types, 102 global planned slots (unchanged, unrenumbered), 37 canonical
schedules and their SHA-256 hashes, the global campaign sequence and its order, the 64 RPS ceiling,
400 max-in-flight, 5s timeout, decision-aligned timestamps (60/90/120/150), the two-axis validity
model, every routing classification threshold, CPU normalization (Phase 15B.3F.1), restart-evidence
categories (Phase 15B.3J.1), SLA thresholds (Phase 15B.3L.1), `rl_contract_version`/
`reward_contract_version` (both still 2), and — explicitly — the V5 workload-transport mechanism in
every particular (`transport_protocol_version` stays 5). `HPA260_V5_REVALIDATED` and
`CROSS_EPOCH_COMPATIBLE_FOR_MODEL_FITTING` are unchanged — not rerun by this phase. Index12's own
frozen schedule and its SHA-256 (`4594f8ce7f813f39588a362360f17e596797e0b8cf579990e08d3bfed33c4dda`)
are unchanged — no schedule regeneration. No dynamics model was fitted, no PPO trained. Sequence
index 13 was not executed; a fifth attempt (attempt5) of sequence index 12 was not executed by this
phase; attempts 1–4 were not modified, retried, or reclassified.
