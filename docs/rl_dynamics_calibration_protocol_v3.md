# RL Empirical Dynamics Calibration Protocol — v3 (CORR-38, SECOND POST-DATA CORRECTION)

**Status**: Frozen by Phase 15B.3C.3, correction reference **CORR-38**, following **CORR-37**
(v2) and the routing forensic reconstructions of Phase 15B.3C.1/15B.3C.2. This is the
**second post-data** correction to the calibration protocol — v1 (CORR-31 through CORR-35) was
entirely pre-execution preregistration; v2 (CORR-37) was the first post-data correction, frozen
after observing `campaign_sequence_index=1`'s real evidence alone. This file governs **only**
the relevant-pod population required for routing-interval admissibility, the pod-identity key
used to join samples, the interval-partition categories, and future counter-observability
collection requirements. `docs/rl_dynamics_calibration_protocol_v2.md` (execution/fit-eligibility
two-axis model, ledger semantics, cell-level support rule) and
`docs/rl_dynamics_calibration_protocol.md` (v1, the base matrix) remain the narrative documents
for everything not explicitly redefined here. The machine-readable counterpart is
`experiments/manifests/rl_dynamics_calibration_v3.yaml`, which inherits everything else
unchanged, by reference, from v2 (which itself inherits from v1).

```
dynamics_calibration_protocol_version: 3   (second POST-DATA correction)
frozen_in_phase: 15B.3C.3
correction_reference: CORR-38 (follows CORR-37)
frozen_after_observing_empirical_attempts: true
observed_attempt_ids:
  - fit_dyncal_r2_L4_repeat1_attempt1_20260909T140146Z   (campaign_sequence_index=1)
  - fit_dyncal_r4_L5_repeat1_attempt1_20260909T151415Z   (campaign_sequence_index=2)
```

## Why This Had to Be a New Version, Not an In-Place v2 Edit

Phase 15B.3C.2 (`task_outputs/phase_15b3c2_complete_joint_observability_routing_gate.md`)
implemented a stricter complete-joint-observability gate in `compute_expected_and_actual()` and
described it in that phase's own report as "an implementation-conformance tightening, not a
methodology change." That framing is correct for the NARROW claim it was defending (the old
per-pod-independent "observable" check could pass trivial conservation on an incomplete subset,
which the frozen v1 manifest's `joint_observability_gap_handling` text already forbade). It is
**not** sufficient framing for the BROADER claim that the *population* now required to be
completely observed — `relevant_pod_set_j`, including pods that are NotReady but still exist or
carry counter-series evidence — was always the literal scope of the prior wording. v1's own
`per_pod_routing_diagnostics.eligibility.basis` defines eligibility narrowly as "Ready-serving
exposure." Broadening that population is scientifically sound and strictly more conservative
(it can only exclude more, never fabricate less), but it was chosen **after** observing that the
broadened population changes both already-executed runs' routing conclusions. A rule change that
alters which already-collected evidence is admissible, decided after seeing what that evidence
says, is a post-data methodology decision by this project's own established precedent (CORR-27,
CORR-32, CORR-37's own framing) — it must be versioned transparently, never silently folded into
an unversioned "bug fix." `protocol_version` is therefore bumped to `3`. v2 remains byte-for-byte
intact (independently re-hashed before this file was written: SHA-256
`de4434fd9ce1f734fdebc2f8b8e80d4f91c44770656d27eef477512bb22f31ac`, pinned in v3's
`inherits_from`), exactly as v1 remained intact when v2 was frozen.

## Correcting Phase 15B.3C.2's Own Appendix Answer

Phase 15B.3C.2's appendix asked "Was this compatible with CORR-35/37?" and answered "Yes." That
conflated two distinct claims that this correction now separates explicitly:

1. **The OLD subset-based implementation** (checking each pod's own two-boundary presence
   independently, silently dropping a missing pod from the aggregate rather than excluding the
   interval) was **not** compatible with the trustworthy-reconstruction principle CORR-35/v1
   already intended — the manifest's own `joint_observability_gap_handling` text ("excluded
   consistently from BOTH... never estimated or fabricated") already implied whole-interval
   exclusion. This part **is** a pre-existing implementation defect, correctly fixed without a
   new CORR.
2. **The NEW complete-`relevant_pod_set_j` population**, which extends beyond the literal
   Ready/eligible wording of v1's `eligibility.basis`, is a **post-data conservative methodology
   extension** — a genuinely new scientific rule, chosen after seeing that it changes retained
   evidence, and it is this part that CORR-38/v3 versions.

Do not conflate these two statements: (1) is a bug fix requiring no CORR; (2) is a post-data
rule requiring a CORR and a version bump. Phase 15B.3C.2's code was correct; its appendix's
single "Yes" answer was not precise enough about which of the two claims it was compatible with.

## Relevant-Pod-Set Definition (Frozen)

`relevant_pod_set_j` for interval `[t_j, t_(j+1))` is the union of pod identities satisfying at
least one of: (a) the pod's Kubernetes object exists in the active app-pod readiness history at
`t_j`; (b) exists at `t_(j+1)`; (c) is Ready at either boundary (a subset of a/b); (d) has a
Prometheus counter-series sample at either boundary. It is explicitly **not** limited to
`ready_pod_set_j` — a NotReady-but-existing pod can still legitimately carry real traffic
(endpoint/readiness-propagation lag) and must still be checked for complete observability, even
though it never receives expected-routing credit (unchanged from CORR-35's formula: Ready-set
membership determines only who *shares* expected credit, never whose traffic counts toward the
aggregate).

**Identity key**: Kubernetes pod **UID** is canonical; pod name is the human-readable label used
to join against Prometheus's `pod` metric label (Prometheus carries no UID label). A name whose
UID changes across the samples being joined must never be silently merged into one series
(`dynamics_calibration_lib.detect_pod_identity_change()`). Both preserved attempts were
independently re-verified: no pod name maps to more than one UID within either attempt's window
(index1: `rpztw`, `xtzgk` each map to exactly one UID; index2: `6lld4`, `v6lmc`, `wvt92`, `xtzgk`
each map to exactly one UID — `xtzgk`'s UID is identical across both attempts, consistent with
the same long-lived pod object surviving between the two calibration slots without being deleted
in between, not a collision). Historical reconstruction therefore safely joins by name for these
two attempts; future runs persist UID alongside name in the new observability artifact (below)
so any future name/UID mismatch is immediately visible rather than silently merged.

## Complete Joint-Observability Gate and Interval Partition (Unchanged From 15B.3C.2's Code, Now Formally Versioned)

An interval enters `ALLOCATABLE` or `ZERO_READY` only if every pod in `relevant_pod_set_j` has a
reconstructable counter value at **both** boundaries; otherwise the whole interval is
`EXCLUDED_INCOMPLETE_POD_COUNTER_OBSERVABILITY` (missing pod identity recorded, never
reconstructed from the remaining subset). A degenerate interval with no relevant pod at all is
`EXCLUDED_OTHER_NO_RELEVANT_PODS`. Mid-interval pod birth/death is handled without fabrication: a
pod whose only appearance is an interval's later boundary is still relevant to that interval
(conservative literal reading) and its missing earlier boundary excludes it; its own later,
fully-observed interval proceeds normally. **Routing classification thresholds are unchanged**:
structural `<0.01`; healthy `[0.5,1.5]` with CV`<=0.30`; suspected catch-all; minimum 2 assessable
pods with `expected>=50`; exact-zero conservation tolerance. Only the admitted population changes.

## Historical Routing Reclassification (Chain Preserved, Nothing Erased)

| Slot | v1/original code | 15B.3C.1 (domain-partition fix) | 15B.3C.2 / v3 (complete relevant-pod gate) |
|---|---|---|---|
| index1 | `ROUTING_PINNING_SUSPECTED` (buggy: expected=591 vs actual=1427, no conservation) | `HEALTHY_VARIANCE` (conserves: 1034=1034, but only 2 pods, Ready-serving-only completeness) | **`INSUFFICIENT_ROUTING_SAMPLE`** (only 1 of 2 pods ever assessable once the full relevant population is required to be jointly observed) |
| index2 | `ROUTING_PINNING_SUSPECTED` (buggy: expected=253 vs actual=415) | `ROUTING_PINNING_SUSPECTED` (conserves: 253=253, CV=0.621) | **`INSUFFICIENT_ROUTING_SAMPLE`** (0 of 4 pods assessable — `6lld4` is relevant throughout but has counter samples for only 2 of 12 boundaries) |

Each superseding artifact is a new, additively-named file; no prior routing summary or forensic
artifact was overwritten. The progression itself — from a buggy implementation, through an
implementation fix, to a conservative post-data population extension — is preserved as the
audit trail, exactly as CORR-36→CORR-37's provisional-then-corrected ledger history was preserved.

## Axis A / Axis B Are Unchanged

v3 changes routing-diagnostic **admissibility** only; it does not touch the two-axis
execution-validity/fit-eligibility model frozen in v2. `INSUFFICIENT_ROUTING_SAMPLE` alone has
never been, and is not now, an `execution_structural_validity` (Axis A) trigger — only
`STRUCTURAL_ROUTING_FAILURE` is. Both index1 and index2 remain `execution_structural_validity =
VALID` (ledger state `VALID`) and `steady_state_fit_eligibility = STEADY_STATE_FIT_INELIGIBLE`,
unchanged from their Phase 15B.3C.1/15B.3B determinations. Neither slot is retried merely because
routing evidence is now assessed as insufficient — retrying to chase a more convenient routing
conclusion would violate the same validity-before-performance principle (CORR-17 lineage) that
already governs Axis A.

## Future Counter-Observability Collection (Hardening, No Live Run Required to Freeze This)

Phase 15B.3C.2's `detect_pod_series_gap(previous, current)` only detects a pod previously
returned by Prometheus that has since vanished — it cannot detect a pod known to Kubernetes but
**never once** returned by Prometheus (absent from the very first tick, or absent for the entire
run), which is exactly what happened with `6lld4` in index2. `scripts/per_pod_counter_collector.py`
now also lists the currently-known Kubernetes pod population at each tick (the same authoritative
`kubectl get pods -o json` source and label-selector convention already used by
`pod_readiness_collector.py`) and computes `missing_known_pods` as a snapshot difference — this
detects the gap on the first tick, a continuously-missing pod, and a transiently-missing pod
alike, unlike the previous-tick-relative check. Results are persisted to a new artifact,
`routing/pod_counter_observability_history.csv`, one row per known pod per tick, without changing
the existing `per_pod_request_counters.csv` schema. A whole-Prometheus-query failure, a
Kubernetes-listing failure, and a single-pod-series-absent-from-an-otherwise-successful-query are
recorded as three **distinct** conditions, never collapsed into one generic error count. No
counter value is ever fabricated, interpolated, or estimated from a neighboring sample — the
complete-observability exclusion remains the safe fallback regardless of diagnostic detail. A
future `query_range` recovery of genuinely-stored historical samples is noted as an optional,
not-yet-implemented future enhancement; it is never required for index3 and was not implemented
in this phase (no live query was issued). A `sum(...)` aggregate computed from the same
currently-returned series is explicitly documented as **not** an independent completeness source.

## Change Log

- **v1** (Phase 15B.2 → 15B.2D, CORR-31 through CORR-35): pre-execution preregistration and its
  corrections. `docs/rl_dynamics_calibration_protocol.md`.
- **v2** (Phase 15B.3B, CORR-37, 2026-09-09): first post-data correction — two-axis
  execution-validity/fit-eligibility model, ledger semantics, cell-level stable-support rule,
  routing-conservation aggregation-scope bug-fix confirmation (thresholds unchanged).
  `docs/rl_dynamics_calibration_protocol_v2.md`.
- **v3** (Phase 15B.3C.3, CORR-38, 2026-09-09): second post-data correction, frozen after
  observing both `campaign_sequence_index=1` and `=2`'s real evidence and the routing forensic
  reconstructions of Phase 15B.3C.1/15B.3C.2. Freezes `relevant_pod_set_j` as the complete
  population (Kubernetes-known-any-status ∪ Ready ∪ counter-series-known) that must be jointly
  observed at both interval boundaries before a routing interval is admissible; freezes
  Kubernetes pod UID as the canonical pod identity (name as label only); freezes the four-way
  interval partition (`ALLOCATABLE`/`ZERO_READY`/`EXCLUDED_INCOMPLETE_POD_COUNTER_OBSERVABILITY`/
  `EXCLUDED_OTHER_NO_RELEVANT_PODS`); requires future counter collection to record the
  authoritative Kubernetes-known pod population per tick and distinguish query-failure from
  series-absence. Routing classification THRESHOLDS are unchanged. Reclassifies index1
  (`HEALTHY_VARIANCE` → `INSUFFICIENT_ROUTING_SAMPLE`) and index2 (`ROUTING_PINNING_SUSPECTED` →
  `INSUFFICIENT_ROUTING_SAMPLE`) without erasing either prior conclusion's own preserved
  artifacts. The 20/9/8/4/102/37/64/400/5s matrix, decision-aligned timestamps, global campaign
  order, all schedule hashes, HPA baseline (260%), MDP/reward contracts (V2), and the two-axis
  validity model and its ledger semantics are all unchanged from v2/v1. No traffic was sent to
  produce this correction; sequence index 3 was not executed.
