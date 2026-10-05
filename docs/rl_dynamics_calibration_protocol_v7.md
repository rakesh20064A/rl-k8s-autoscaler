# RL Empirical Dynamics Calibration Protocol — v7 (CORR-42, SIXTH POST-DATA CORRECTION)

**Status**: Frozen by Phase 15B.3P.1, correction reference **CORR-42**, following **CORR-41**
(v6). This is the **sixth post-data** correction to the calibration protocol, and the third (after
v5/CORR-40 and v6/CORR-41) to be frozen strictly after reviewing a specific attempt's real evidence —
`campaign_sequence_index=12` attempt4, executed under v6/CORR-41's own frozen route-integrity
evidence rules. This file governs **only** the dataplane-ingress evidence **acquisition schedule**:
event-triggered reactive re-sampling on a detected regime/rule-identity transition, one end-of-run
anchor sample taken at traffic-generator-process exit, and a reduced baseline periodic cadence
(3.0s → 2.0s). It does **not** touch CORR-41's route-integrity classification rules, the V5
workload-transport mechanism CORR-40 froze, the base matrix, the workload, or the schedules.
`docs/rl_dynamics_calibration_protocol_v6.md` (CORR-41's route-integrity evidence contract this file
extends), `docs/rl_dynamics_calibration_protocol_v5.md` (V5 transport),
`docs/rl_dynamics_calibration_protocol_v4.md` (CORR-39's original route-integrity contract),
`docs/rl_dynamics_calibration_protocol_v3.md` (routing relevance),
`docs/rl_dynamics_calibration_protocol_v2.md` (two-axis validity/ledger semantics), and
`docs/rl_dynamics_calibration_protocol.md` (v1, the base matrix) remain the narrative documents for
everything not explicitly redefined here. The machine-readable counterpart is
`experiments/manifests/rl_dynamics_calibration_v7.yaml`, which inherits everything else unchanged,
by reference, from v6 (which itself inherits from v5, v4, v3, v2, and v1).

```
dynamics_calibration_protocol_version: 7   (sixth POST-DATA correction)
frozen_in_phase: 15B.3P.1
correction_reference: CORR-42 (follows CORR-41)
frozen_after_observing_empirical_attempts: true
observed_attempt_ids:
  - fit_dyncal_r8_L3_repeat1_attempt4_20260919T091715Z   (campaign_sequence_index=12, V6/CORR-41-governed, route_integrity=UNVERIFIED, 2 of 18 windows)
route_integrity_protocol_version: 6   (UNCHANGED -- this file does not touch classification, only acquisition)
dataplane_acquisition_protocol_version: 2   (was implicitly 1 under CORR-41/V6's fixed 3s schedule)
transport_protocol_version: 5   (UNCHANGED -- this file does not touch the workload transport)
```

## Why This Had to Be a New Version, Not an In-Place V6/CORR-41 Edit

Index12 attempt4 was the first live attempt executed under the V6/CORR-41 dataplane-ingress
evidence source with a fully healthy V6 relay throughout — and its result exposed a real,
structural **acquisition-schedule** gap rather than a classification defect: 16 of 18 route-integrity
evidence windows were `CONFIRMED` exclusively via `nodeport_dataplane_ingress_evidence` (the same
application collapse as attempt3 made HTTP/counter evidence unavailable throughout), but exactly 2
windows were `UNVERIFIED`, forcing overall `route_integrity=UNVERIFIED` and
`execution_structural_validity=INVALID`. Direct reconstruction of both windows from attempt4's raw
sample timeline showed continuous, healthy, monotonically-rising packet counts on both sides of
each — i.e., this was not evidence of route failure, but of the sampler not having collected enough
temporally-local samples to prove what packet counters already show happened. `compute_dataplane_
ingress_evidence_windows()`'s classification logic is not wrong and is not being weakened here — a
formal, general timing analysis (not limited to attempt4's own timestamps) confirmed the fixed 3.0s
nominal cadence's REAL inter-sample gap (empirically 3.31s–4.31s, mean 3.60s) can structurally leave
a full 10-second window with only 2 interior samples, and can leave a short/trailing window with
even fewer — independent of whether the dataplane itself is healthy. Deciding to add reactive and
anchor sampling, and to reduce the baseline cadence, after observing this specific real gap is a
post-data methodology decision by this project's own established precedent (CORR-27/32/37/38/39/
40/41) — it must be versioned transparently, not folded into an unversioned bug-fix label or
silently edited into V6.

## CORR-42 sec.A: Rationale

1. CORR-41's fixed `DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS=3.0` nominal cadence's own
   justification assumed near-zero jitter; the REAL inter-sample gap (including `docker exec`
   command latency and thread-scheduling overhead) is materially larger — empirically 3.31s–4.31s,
   mean 3.60s, measured directly from attempt4's own 52 real samples.
2. A formal, general synthetic timing analysis (offsets between material-interval boundaries, 10s
   route-window boundaries, and realistic-cadence sampler timestamps — not limited to attempt4's
   exact timestamps) found that a meaningful fraction of possible phase alignments (23%–68%,
   depending on realized cadence) leave a full 10-second window with only 2 interior samples, and
   that trailing/remainder windows narrower than roughly one cadence period are sometimes-to-always
   insufficient depending on exact width.
3. Index12 attempt4 demonstrated concrete instances of exactly this structural class:
   `[115.0,125.0)` has its only candidate pair invalidated by a genuine, correctly-detected
   `rule_identity` reset (8 SEPs → 7 SEPs) landing between the window's only two interior samples;
   `[175.0,179.0)` is a 4-second trailing remainder window (74s material interval = 7×10+4) with
   only one interior sample — structurally short of the two needed to form any pair.
4. `compute_dataplane_ingress_evidence_windows()` was confirmed, by direct code reading, to be
   completely agnostic to how or when its input samples were collected — the defect is entirely in
   the acquisition SCHEDULE, not the classification semantics, so the correction is acquisition-only.
5. The literal reading of "sample at every future route-window boundary" was evaluated and
   rejected: route-window boundaries are discovered only **post-hoc**, from `traffic_summary.csv`
   after traffic ends (`find_all_zero_success_runs()` / `partition_into_evidence_windows()`), so the
   harness cannot know them live during a run. A reactive (event-triggered) and anchor
   (generator-exit-triggered) sampling strategy was designed instead, using only real-time-knowable
   events.
6. Therefore the acquisition-schedule change is versioned transparently as CORR-42/Protocol V7,
   rather than hidden inside an in-place CORR-41/V6 edit.

## CORR-42 sec.B: The Corrected Acquisition Schedule

Three combined, acquisition-only corrections (full raw validation evidence:
`persistent-data/transport_validation/protocol_v7_temporal_coverage/`):

```
1. Event-triggered reactive re-sampling
   trigger:  should_trigger_immediate_resample(previous_sample, current_sample) is True --
             current sample's regime or rule_identity differs from the immediately preceding
             sample (a detected kube-proxy reprogramming event: pod restart, endpoint-count
             transition)
   delay:    1.0s after the detected transition
   guarantee: one additional sample is taken under the NEW identity shortly after a detected
             transition -- never fabricates a delta across the transition itself; the
             pre-transition/post-transition pair remains rejected as a RESET exactly as under
             CORR-41
   errors:   an observation_error on either sample being compared is never treated as a
             transition -- this function only ever requests MORE evidence

2. End-of-run anchor sample
   trigger:  the traffic-generator subprocess has exited
   timing:   taken BEFORE the periodic sampler thread is stopped, immediately after generator exit
   guarantee: directly targets short/trailing windows whose material interval ends near the end
             of a run; subject to the SAME cross-window/reset/identity rules as every other
             sample -- an anchor sample landing outside the window it was hoped to help is
             rejected as CROSS_WINDOW exactly like any other sample, never specially privileged

3. Reduced baseline cadence
   value:    DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7 = 2.0s (was 3.0s under CORR-41/V6)
   status:   evaluated as a fallback-only lever per this phase's own governing instructions --
             a modest, quantitatively-justified reduction, not an arbitrary jump to 1s
```

All three only ever ADD samples to the same list `compute_dataplane_ingress_evidence_windows()`
already consumes generically — that function, `classify_route_integrity_window()`, and
`classify_route_integrity()` are byte-for-byte unchanged by this correction. Reference
implementation: `scripts/dynamics_calibration_lib.py:
DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7`, `DATAPLANE_INGRESS_TRANSITION_RESAMPLE_DELAY_SECONDS`,
`should_trigger_immediate_resample()`; `scripts/run_dynamics_calibration_slot.py:
_sample_dataplane_ingress_periodically()`'s transition-detection loop and the post-generator-exit
anchor sample.

## CORR-42 sec.C: Dataplane Ingress Artifact Additions

`cluster/loadtest_dataplane_ingress_timeline.json` gains four new top-level fields:
`dataplane_ingress_sampling_protocol_version` (7), `dataplane_ingress_transition_resample_delay_seconds`,
`dataplane_reactive_resample_count`, and `dataplane_anchor_sample_taken`. Its existing
`dataplane_ingress_sampling_interval_seconds` field now records the V7 baseline cadence (2.0) for
attempts executed under this protocol; historical attempts (1–4) retain whatever value (3.0,
CORR-41/V6) was live when *they* executed, in their own immutable metadata/artifacts. No field of
`cluster/loadtest_route_integrity_timeline.json` changes — CORR-41's per-window evidence fields
are consumed identically regardless of how the underlying samples were acquired.

## CORR-42 sec.D: Route-Integrity Classification — Completely Unchanged

Every CORR-41/V6 classification rule is preserved verbatim: the three-source OR'd `CONFIRMED` rule
(`http_response_delivery_evidence`, `application_counter_delivery_evidence`,
`nodeport_dataplane_ingress_evidence`); **FAILED precedence** (a confirmed harness-path failure is
always decisive, regardless of what any dataplane evidence — old or new-acquisition — shows); the
materiality criterion, 10-second half-open evidence windows, window-local attribution, and the
prohibition on cross-window/cross-identity smearing; attempt-level aggregation (`FAILED` if any
window `FAILED`; `CONFIRMED` only if every material window is `CONFIRMED`; `UNVERIFIED` otherwise);
and the Axis-A consequence (`FAILED`/material `UNVERIFIED` → `execution_structural_validity=
INVALID`). `route_integrity_protocol_version` therefore stays **6** — this file introduces a new,
independent `dataplane_acquisition_protocol_version` field (1 → 2) specifically so an
acquisition-schedule change is never conflated with a classification change in any attempt's
provenance.

## CORR-42 sec.E: Non-Retroactivity

Index12 attempt4 did **not** collect samples under this acquisition schedule — Protocol V7 was
frozen strictly *after*, and as a direct consequence of, reviewing attempt4's own evidence. What
the corrected schedule "would have shown" during attempt4's two `UNVERIFIED` windows is never
inferred, estimated, or backfilled into attempt4's stored, official result — it is assessed only via
a separate, clearly labeled `NON_OFFICIAL_COUNTERFACTUAL_ONLY` offline/synthetic replay of attempt4's
preserved raw timing structure, which never changes attempt4's official classification. Attempt4
remains, permanently: `SEQUENCE_INDEX_12_ATTEMPT4_INVALID`, `ROUTE_INTEGRITY_UNVERIFIED`,
`AXIS_A_INVALID`. Ledger state is unchanged by this phase: `slot12.state = INVALID_RETRY_REQUIRED`,
`slot12.valid_attempt_id = None`, `slot12.all_attempt_ids = [attempt1, attempt2, attempt3,
attempt4]`, `next_resumable_index() == 12`, `slot13 = NOT_ATTEMPTED`. This file applies
**prospectively only**, starting with a future, separately-authorized attempt5 — Phase 15B.3P.1
does not execute attempt5 or index13.

## What Remains Unaffected

Everything CORR-41/V6 preserved remains preserved: the three-source route-integrity classification
rule and its `route_integrity_protocol_version=6`; `classify_route_integrity_window()`,
`classify_route_integrity()`, and `compute_dataplane_ingress_evidence_windows()` byte-for-byte
unchanged; the 20 fitting cells, 9 interpolation cells, 8 replica-transition types, 4
workload-transition types, 102 global planned slots (unchanged, unrenumbered), 37 canonical
schedules and their SHA-256 hashes, the global campaign sequence and its order, the 64 RPS ceiling,
400 max-in-flight, 5s timeout, decision-aligned timestamps (60/90/120/150), the two-axis validity
model, every routing classification threshold, CPU normalization (Phase 15B.3F.1), restart-evidence
categories (Phase 15B.3J.1), SLA thresholds (Phase 15B.3L.1), `rl_contract_version`/
`reward_contract_version` (both still 2), and — explicitly — the V5 workload-transport mechanism in
every particular (socat relay, NodePort, connection mode, client location;
`transport_protocol_version` stays 5). `HPA260_V5_REVALIDATED` and
`CROSS_EPOCH_COMPATIBLE_FOR_MODEL_FITTING` (Phase 15B.3N.4/15B.3N.4A) are unchanged — HPA-260 was
not rerun and the pre-V5/V5 bridge was not rerun by this phase. Index12's own frozen schedule and
its SHA-256 (`4594f8ce7f813f39588a362360f17e596797e0b8cf579990e08d3bfed33c4dda`) are unchanged — no
schedule regeneration. No dynamics model was fitted, no PPO trained. Sequence index 13 was not
executed; a fifth attempt (attempt5) of sequence index 12 was not executed by this phase; attempts
1–4 were not modified, retried, or reclassified.
