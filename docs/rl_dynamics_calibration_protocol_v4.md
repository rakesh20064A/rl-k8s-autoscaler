# RL Empirical Dynamics Calibration Protocol — v4 (CORR-39, THIRD POST-DATA CORRECTION)

**Status**: Frozen by Phase 15B.3M.2, correction reference **CORR-39**, following **CORR-38**
(v3) and the external-workload-route forensic audit of Phase 15B.3M/15B.3M.1. This is the
**third post-data** correction to the calibration protocol — v1 (CORR-31 through CORR-35) was
entirely pre-execution preregistration; v2 (CORR-37) was frozen after `campaign_sequence_index=1`'s
real evidence; v3 (CORR-38) was frozen after indices 1/2's routing-relevance evidence. This file
governs **only** the distinction between generator schedule execution integrity and official
external workload-route integrity, the `CONFIRMED`/`FAILED`/`UNVERIFIED` route-integrity
classification and its materiality gate, one new `invalid_cell_structural_criteria` entry, the
resulting Axis-A/ledger/cell-support consequences, and the minimum evidence-collection contract
future official attempts must satisfy. `docs/rl_dynamics_calibration_protocol_v3.md` (routing
relevance), `docs/rl_dynamics_calibration_protocol_v2.md` (two-axis validity/ledger semantics),
and `docs/rl_dynamics_calibration_protocol.md` (v1, the base matrix) remain the narrative
documents for everything not explicitly redefined here. The machine-readable counterpart is
`experiments/manifests/rl_dynamics_calibration_v4.yaml`, which inherits everything else
unchanged, by reference, from v3 (which itself inherits from v2 and v1).

```
dynamics_calibration_protocol_version: 4   (third POST-DATA correction)
frozen_in_phase: 15B.3M.2
correction_reference: CORR-39 (follows CORR-38)
frozen_after_observing_empirical_attempts: true
observed_attempt_ids:
  - fit_dyncal_r8_L3_repeat1_attempt1_20260916T130416Z   (campaign_sequence_index=12)
```

## Why This Had to Be a New Version, Not an In-Place v3 Edit

Phase 15B.3M's official attempt for `campaign_sequence_index=12` exhibited a 121-second
(elapsed 59–179s of its 180s traffic window) sustained zero-success delivery interval. Phase
15B.3M.1's forensic audit established that this could not be localized among the application,
the in-cluster `demo-app-loadtest` Service/dataplane, and the external Minikube-service tunnel
that actually carries the official generator's traffic into the cluster — because the harness had
never instrumented sustained tunnel liveness (only its URL resolving once at launch) nor
`demo-app-loadtest`'s own historical Endpoints state. Deciding whether such an attempt may be
accepted as formal calibration evidence, after observing that this specific ambiguity exists in
real, already-collected data, is a post-data methodology decision by this project's own
established precedent (CORR-27/CORR-32/CORR-37/CORR-38) — it must be versioned transparently,
not folded into an unversioned bug-fix label.

## The Core Distinction (CORR-39 sec.A)

**Generator schedule execution integrity** — proven by `offered_requests == sent_requests` and
`outcome_totals.dropped_overload == 0` — means only that the traffic generator attempted every
scheduled request against its own configured destination URL. It has never proven, and must never
be read as proving, **official workload-route integrity**: that the dispatched schedule actually
traversed the designated experiment route (host → Minikube-service tunnel/helper →
`demo-app-loadtest` Service → kube-proxy dataplane → pods) for the scientifically relevant portion
of the run. If that route may have failed, later observations cannot be unambiguously attributed
to "the planned RPS applied to the tested Kubernetes workload" versus "the planned RPS attempted
against a broken harness transport that never reached the workload at all" — these are different
experiments, and an unresolved, material ambiguity between them must never silently default to
`VALID`.

## Route-Integrity Evidence Layers and Classification (CORR-39 sec.B, PROOF-SUFFICIENCY HARDENED — Phase 15B.3M.3)

Reference implementation: `scripts/dynamics_calibration_lib.py` — `longest_zero_success_run()`
(finds the longest maximal run of consecutive per-second traffic-summary rows with
`sent_requests>0` and `success_2xx==0`), `classify_route_integrity()` (now returns a full evidence
dict, never a bare string), and `compute_delivered_request_evidence()`;
`scripts/run_dynamics_calibration_slot.py` — `check_local_listener_present()`. **Phase 15B.3M.4**
further hardens the temporal sufficiency of this evidence — see "Temporal Coverage Hardening"
below, which supersedes interval-wide `compute_delivered_request_evidence()` for classification
purposes with a windowed evaluation.

**This section was amended in place after Phase 15B.3M.2's first implementation was found, before
it ever governed a live attempt, to allow a material interval to be classified `CONFIRMED` on
bare tunnel-process liveness (`Popen.poll() is None`) alone — insufficient, since that proves only
that the helper has not exited, nothing about whether it is still listening, forwarding, or that
any request reached the workload.** Five evidence layers are now consulted, never collapsed into
one boolean:

| Layer | Evidence | Proves | Sufficient for `CONFIRMED`? |
|---|---|---|---|
| A — generator schedule execution | `offered==sent`, `dropped_overload==0` | generator attempted its schedule | No |
| B — tunnel process state | PID, `poll()`, exit code, timestamps | helper process has/hasn't exited | No (necessary only; `False` here → `FAILED`) |
| C — local listener state | read-only `lsof -iTCP:<port> -sTCP:LISTEN` | something is bound to the expected port | No (necessary only; `False` here → `FAILED`) |
| D — loadtest-Service endpoint state | `demo-app-loadtest` Endpoints ready membership | the Service has usable backends at sample time | No (supporting only) |
| E — delivered-request evidence | any HTTP response (2xx/4xx/5xx), non-zero server `/work` RPS, or a per-pod counter increment during the interval | a request actually crossed the harness boundary into the workload | **Yes — the only sufficient layer** |

- **`CONFIRMED`** — either no *material* zero-success interval occurred at all (ordinary
  application-level failures — timeouts, 5xx, readiness loss, restarts, brief connection-error
  blips — remain legitimate, non-invalidating performance outcomes whenever route integrity is
  otherwise unquestioned), or a material interval occurred and **Layer E** establishes requests
  actually reached the tested workload during it (legitimate in-cluster/application behavior under
  load, however severe — success is never required, a 5xx or a hang after genuine receipt both
  count).
- **`FAILED`** — Layer B and/or Layer C directly establish the required tunnel/helper demonstrably
  exited, its listener disappeared, or it otherwise stopped forwarding during the scientifically
  relevant window, for harness-infrastructure reasons. Structural reason: `wrong_traffic_path_used`
  (existing v1 criterion, reused).
- **`UNVERIFIED`** — a material interval occurred, no Layer B/C failure is confirmed, but Layer E
  is also absent. **Layers B/C/D being individually healthy — even all three simultaneously — is
  explicitly insufficient**: process alive + listener present + endpoints valid + zero server RPS +
  100% client `connection_error` is the corrected index12-shaped regression case, and it resolves
  to `UNVERIFIED`, never `CONFIRMED`. This asserts neither that the tunnel failed nor that the
  application is exonerated. Structural reason: `official_workload_route_integrity_unverified`
  (**new** `invalid_cell_structural_criteria` entry, CORR-39).

**Materiality** (unchanged by this hardening — all required): a zero-success interval exists; it
overlaps the official scored window; its duration is at least `0.25` (25%) of the total traffic
duration.

`FAILED` and material `UNVERIFIED` both set `execution_structural_validity=INVALID` →
`INVALID_RETRY_REQUIRED` in the ledger. Neither state is itself a claim about which layer actually
failed; only `FAILED` asserts a confirmed harness-side fault.

**Versioning note**: this hardening amends V4/CORR-39 in place rather than creating V5, because V4
had not yet governed any live attempt when the gap was found — this is a pre-first-use
implementation correction to a just-introduced mechanism, not a post-data reversal of a rule that
had already produced accepted results (unlike the V2/V3/V4 version bumps themselves, each of which
changed a rule's substance after observing what it did to already-collected data).

## Temporal Coverage Hardening (CORR-39 sec.F, Phase 15B.3M.4)

**This section was amended in place after Phase 15B.3M.3's Layer E was found, before it ever
governed a live attempt, to allow ONE delivered request ANYWHERE inside a long material interval
to `CONFIRM` the entire interval, even if the route then stopped forwarding for good and the
remainder carried zero further delivery evidence with the tunnel process, local listener, and
Service endpoints all still nominally healthy.** This is the temporal analogue of the process-
liveness gap Phase 15B.3M.3 itself closed, one level deeper.

**Core principle**: for a material zero-success interval, route-integrity confirmation requires
temporally sufficient delivery evidence across that interval, not merely evidence that at least one
request crossed the route somewhere inside it. Positive route-delivery evidence is temporally
scoped: evidence that the route forwarded one request at one point in a material zero-success
interval does not establish forwarding integrity for the remainder of that interval. A material
interval is `CONFIRMED` only when every evidence window covering that interval has positive
delivered-request evidence and no direct harness-route failure. `CONFIRMED` at t1 does not imply
`CONFIRMED` at t2 for a persistent tunnel/dataplane path that may fail partway through and never
recover.

**Mechanism**: `ROUTE_INTEGRITY_EVIDENCE_WINDOW_SECONDS = 10` (frozen, matching the pre-existing
tunnel/listener/endpoint observer and metrics/counter collector cadence). Reference
implementation: `scripts/dynamics_calibration_lib.py` — `find_all_zero_success_runs()` (every
materially-qualifying zero-success run is now evaluated, not only the longest),
`is_interval_material()` (the unchanged sec.5/sec.B materiality gate applied per candidate
interval), `partition_into_evidence_windows()` (deterministic half-open `[start, end)` windows
aligned to the interval's own start, e.g. `[59,180)` → `[59,69), [69,79), ..., [169,179),
[179,180)`, with the shorter final window always preserved), `compute_delivered_request_evidence_windows()`
(per-window delivery evidence from `requests.csv` HTTP responses, `metrics.csv` server RPS, and
`per_pod_request_counters.csv` counter increases — a counter increase whose bounding samples
straddle two windows is credited to **neither**, never duplicated), `aggregate_observer_evidence_for_window()`
(maps existing periodic tunnel/listener/endpoint samples into whichever window contains each
sample's own timestamp; no sample in a window means `None`/unknown, never fabricated healthy),
`classify_route_integrity_window()` (Layers B/C/E's same three-way rule, applied per window), and a
redesigned `classify_route_integrity()` that now takes `material_intervals` plus a flattened
`evidence_windows` list rather than one interval-wide boolean.

**Attempt-level aggregation** across every material interval in the attempt: `FAILED` if **any**
constituent evidence window is `FAILED` (decisive, never downgraded by other healthy windows);
`CONFIRMED` **only if every** constituent window across every material interval is `CONFIRMED`
(`full_temporal_coverage=true`); `UNVERIFIED` otherwise — positive evidence in one window never
retroactively upgrades a different, evidence-lacking window.

**Regression case closed**: a 120s material interval partitioned into 12 windows, window1 with a
server-side counter increase (`CONFIRMED`) and windows 2–12 with tunnel/listener/endpoints all
nominally healthy but zero delivery evidence (`UNVERIFIED` each), now correctly resolves
attempt-level `UNVERIFIED`, not `CONFIRMED`. A full-coverage case (every window independently
`CONFIRMED`) still resolves `CONFIRMED` however catastrophic client-observed performance is. A
mid-run harness failure (an early run of `CONFIRMED` windows followed by a `FAILED` window) resolves
attempt-level `FAILED`, never downgraded to `UNVERIFIED`.

**Artifacts**: the full per-window temporal audit is persisted to
`cluster/loadtest_route_integrity_timeline.json`; `metadata.json`'s `route_integrity` key carries
only the summary (`classification`, `reason`, `material_intervals`, `full_temporal_coverage`,
`confirmed_window_count`, `unverified_window_count`, `failed_window_count`) — the full per-window
timeline is never duplicated wholesale into `metadata.json`.

**Versioning note**: exactly as with the Phase 15B.3M.3 hardening above, this amends V4/CORR-39 in
place rather than creating V5 — V4 had not yet governed any live attempt when this temporal-
coverage gap was found (index12 attempt1 predates V4 entirely, remains a pre-v4 historical attempt,
and is not re-audited by this section). See `docs/experimental_design_corrections.md` CORR-39 sec.9
and `task_outputs/phase_15b3m4_corr39_temporal_route_coverage_hardening.md`.

## Window-Local Delivered-Request Attribution Hardening (CORR-39 sec.G, Phase 15B.3M.5)

**This section was amended in place after the Phase 15B.3M.4 per-window evidence contract was
found, before it ever governed a live attempt, to allow a window's evidence to be established from
an OBSERVATION timestamp that fell inside the window without the underlying delivery EVENT itself
being attributable to that window.** Two concrete gaps, both closed by direct source audit:

**HTTP-response attribution used completion time, not dispatch time.** A request dispatched at
t=58.5 (before window `[59,69)` begins) whose response completed at t=61.0 (inside it) was being
credited to `[59,69)` — but the route-entry attempt began before the window. **Fixed**: HTTP
delivery evidence is now attributed by `actual_dispatch_elapsed_seconds` (`time.monotonic()`,
captured as the literal first statement inside
`loadgen/traffic_generator.py:_dispatch_single_request()`, before URL construction or the `httpx`
call — confirmed by direct source read). A request counts as window-local evidence for window W
only when its dispatch falls in W and it eventually produced a genuine HTTP response; its
completion time is retained for provenance only, never for window membership.

**`request_rate_rps` is a 60-second rolling rate, not a point-in-time count.** Direct audit of
`collectors/metrics_collector.py` confirmed the exact query:
`sum(rate(demo_app_requests_total{namespace="...",route="..."}[1m]))`, sampled at the collector's
own ~10s cadence. A positive value at t=65 reflects up to 60 seconds of prior request history, not
proof a request entered the route specifically during a 10-second evidence window. **Fixed**:
`request_rate_rps` is now recorded per window as `rolling_server_rps_observed`, explicitly flagged
`rolling_server_rps_supporting_only=true`, and is never included in `delivered_request_evidence`.
The canonical `request_rate_rps` metric and its use elsewhere (dynamics calibration, RL
observation, throughput analysis) is entirely unchanged.

**Raw per-pod counters remain the preferred server-side evidence**, with the same window-local
bounding-sample rule as before (both samples' epoch inside the same window), now with explicit
provenance: an interval overlapping a window without being fully contained in it is recorded as a
rejected cross-window interval for that window, never silently dropped, never smeared across two
windows.

**A genuine pre-existing defect was found and fixed in the process**: both the interval-wide
`compute_delivered_request_evidence()` and the first `compute_delivered_request_evidence_windows()`
read `row.get("pod_name")` from `per_pod_request_counters.csv` rows, but the real column (written by
`scripts/per_pod_counter_collector.py`) is `pod` — the counter-increment evidence source had never
actually fired in either function. Corrected to `row.get("pod")` in both; both were pre-first-use,
so this is amended in place without a version bump.

**Core principle now frozen**: evidence is window-local only when the event or count change can be
attributed to requests whose route-entry/delivery occurred within that window. A metric observed
inside a window is not automatically evidence generated inside that window. Do not confuse an
observation timestamp with an event-attribution timestamp.

**What did not change**: the 10-second evidence-window resolution, the half-open boundary
convention, the 25% materiality threshold, the scored-window-overlap requirement, and the Phase
15B.3M.4 attempt-level aggregation rule (`FAILED` if any window `FAILED`; `CONFIRMED` only if every
window is `CONFIRMED`; `UNVERIFIED` otherwise) are all unchanged — `classify_route_integrity_window()`
and `classify_route_integrity()` required no changes.

**Versioning note**: Protocol V4 is retained — a third pre-first-use hardening pass, not V5. Index12
attempt1 remains `INVALID_RETRY_REQUIRED`, unaffected and not re-audited. See
`docs/experimental_design_corrections.md` CORR-39 sec.10 and
`task_outputs/phase_15b3m5_corr39_window_local_delivery_attribution_hardening.md`.

## Index12 Disposition (CORR-39 sec.C)

`fit_dyncal_r8_L3_repeat1_attempt1_20260916T130416Z`: `route_integrity=UNVERIFIED` (material
interval elapsed 59–179s, 121s of 180s, overlapping the scored window; no tunnel-lifecycle
evidence exists for this pre-CORR-39 attempt). Resolved: `execution_structural_validity=INVALID`,
ledger `INVALID_RETRY_REQUIRED`. All raw evidence is preserved as diagnostic data — not deleted,
not rewritten — but the attempt does not contribute to formal calibration fitting or
`dyncal_r8_L3` cell support. A retry is required; it is not performed in this phase or by this
document.

## Axis-B / Cell-Support Consequence (CORR-39 sec.E)

`assess_decision_point_eligibility()` is unchanged — route integrity is never folded into it as a
new Axis-B eligibility criterion, and no raw decision-point row is ever rewritten. But an
Axis-A-`INVALID` attempt (for any reason, route-integrity or otherwise) contributes **zero**
sufficient-repeat support to its cell, regardless of how many of its own decision points were
locally computed as fit-eligible — this was already implicit in v2/CORR-37's two-axis model (Axis
B is only ever consulted for an Axis-A-`VALID` attempt); CORR-39 makes it explicit for the
route-integrity case, the first case where an attempt's Axis-B fields were fully fit-eligible (2
of 4) before its Axis-A disposition was corrected.

## Future Evidence-Collection Contract (CORR-39 sec.D)

Implemented Phase 15B.3M.1/15B.3M.2/15B.3M.3 (`scripts/run_dynamics_calibration_slot.py`,
`scripts/dynamics_calibration_lib.py`, `loadgen/traffic_generator.py`) and applying starting with
the **very next official attempt**, whichever slot that is — concretely the index12 retry
(attempt2), not "index13 onward" (a documentation error in Phase 15B.3M.2's own report, corrected
here; `build_protocol_provenance()` takes no `campaign_sequence_index` argument and structurally
cannot special-case which slot runs next): tunnel/helper launch command, PID, timestamp,
destination URL, and log path; passive liveness sampling (no probe traffic) at the same ~10s
cadence as the existing readiness/counter collectors, plus an end-of-traffic check; exit
code/timestamp when detected; explicit already-exited-vs-explicitly-terminated cleanup
classification (`classify_tunnel_termination_action()`); **read-only local TCP listener presence**
(`check_local_listener_present()`, `lsof -iTCP:<port> -sTCP:LISTEN`) at the same cadence, plus at
traffic end; per-failed-request `error_type`/`error_message` diagnostics (sanitized, truncated) for
`timeout` and `connection_error` outcomes, without altering outcome classification or timeout
behavior; `demo-app-loadtest` Endpoints snapshots before and after traffic plus periodic
ready-count sampling during it, with query failures recorded distinctly (never silently treated as
zero endpoints); and **delivered-request evidence**, computed entirely offline/post-hoc from
already-collected `metrics.csv`/`per_pod_request_counters.csv`, aligned to the material interval by
wall-clock epoch (`compute_delivered_request_evidence()`). None of this sends additional HTTP
traffic through the tunnel or the loadtest Service — the tested workload itself is never altered.

## What Remains Unaffected

The 20 fitting cells, 9 interpolation cells, 8 replica-transition types, 4 workload-transition
types, 102 global planned slots, 37 canonical schedules and their SHA-256 hashes, the global
campaign sequence and its order, decision-aligned timestamps (60/90/120/150) and their existing
fit-eligibility rule, the 64 RPS ceiling, 400 max-in-flight, 5s timeout, the two-axis validity
model, every routing classification threshold, Protocol V3's `relevant_pod_set_j`/complete-joint-
observability rules, CPU-normalization definitions (Phase 15B.3F.1), restart-evidence categories
(Phase 15B.3J.1), and SLA thresholds/dual-metric-source discipline (Phase 15B.3L.1) are all
unchanged. `rl_contract_version`/`reward_contract_version` (both still 2) and the HPA baseline
(260%) remain untouched, confirmed live. No traffic was sent, no Kubernetes object was mutated,
and no dynamics model was fitted while freezing this correction. `configs/traffic_profiles/
unseen_eval.yaml` was not inspected. Sequence index 13 was not executed; a retry of sequence
index 12 was not executed.
