# RL Empirical Dynamics Calibration Protocol — v6 (CORR-41, FIFTH POST-DATA CORRECTION)

**Status**: Frozen by Phase 15B.3O.1, correction reference **CORR-41**, following **CORR-40**
(v5). This is the **fifth post-data** correction to the calibration protocol, and the second (after
v5/CORR-40) to be frozen strictly after reviewing a specific attempt's real evidence —
`campaign_sequence_index=12` attempt3, executed under v5/CORR-40's own frozen V5 transport. This
file governs **only** the route-integrity evidence/classification layer: it adds one new
decisive-positive evidence source (Kubernetes-dataplane NodePort ingress) alongside CORR-39's
existing two, and the dedicated artifact/provenance fields that go with it. It does **not** touch
the V5 workload-transport mechanism CORR-40 froze, the base matrix, the workload, the schedules, or
any historical classification. `docs/rl_dynamics_calibration_protocol_v5.md` (V5 transport),
`docs/rl_dynamics_calibration_protocol_v4.md` (CORR-39's original route-integrity contract),
`docs/rl_dynamics_calibration_protocol_v3.md` (routing relevance),
`docs/rl_dynamics_calibration_protocol_v2.md` (two-axis validity/ledger semantics), and
`docs/rl_dynamics_calibration_protocol.md` (v1, the base matrix) remain the narrative documents for
everything not explicitly redefined here. The machine-readable counterpart is
`experiments/manifests/rl_dynamics_calibration_v6.yaml`, which inherits everything else unchanged,
by reference, from v5 (which itself inherits from v4, v3, v2, and v1).

```
dynamics_calibration_protocol_version: 6   (fifth POST-DATA correction)
frozen_in_phase: 15B.3O.1
correction_reference: CORR-41 (follows CORR-40)
frozen_after_observing_empirical_attempts: true
observed_attempt_ids:
  - fit_dyncal_r8_L3_repeat1_attempt3_20260919T032529Z   (campaign_sequence_index=12, V5/CORR-40-governed, route_integrity=UNVERIFIED, all 14 windows)
route_integrity_protocol_version: 6   (was 4 under V4/V5 -- CORR-40 changed only the transport, never this)
transport_protocol_version: 5   (UNCHANGED -- this file does not touch the workload transport)
```

## Why This Had to Be a New Version, Not an In-Place V4/CORR-39 Edit

Index12 attempt3 was the first live attempt executed under the V5/CORR-40 transport with a fully
healthy transport throughout (`relay_container_running=true`, `host_listener_present=true` for the
entire run) — and its result exposed a real, structural methodology gap rather than a transport
defect: application readiness collapsed under load, Prometheus scraping and HTTP successes nearly
disappeared, and per-pod counter evidence became unavailable, so CORR-39's classifier correctly (by
its own rules) found all 14 evidence windows across both material zero-success intervals
`UNVERIFIED`, forcing `execution_structural_validity=INVALID`. CORR-39's classification logic is
not wrong and is not being weakened here — the gap is that its two positive-evidence sources both
require the application to be observable, so they cannot distinguish "the harness/route failed"
from "the route is fine but the overloaded application collapsed," and the latter is itself a
legitimate scientific outcome this campaign needs to be able to observe without discarding the
attempt as structurally invalid. Deciding to add a new, endpoint-independent evidence source after
observing this specific real gap is a post-data methodology decision by this project's own
established precedent (CORR-27/32/37/38/39/40) — it must be versioned transparently, not folded
into an unversioned bug-fix label or silently edited into V4.

## CORR-41 sec.A: Rationale

1. CORR-39's only two decisive positive route-evidence sources (HTTP response attributable by
   dispatch time, raw per-pod application-counter increment) both require the application to be
   alive, scraped, and counting.
2. Index12 attempt3 demonstrated the resulting gap directly: transport healthy throughout,
   generator dispatching throughout, application readiness collapsed, all 14 evidence windows
   `UNVERIFIED` for lack of application-dependent evidence — not for lack of harness/route health.
3. A relay-alive-plus-listener-present-only relaxation was explicitly rejected as insufficient: a
   relay could be listening locally while failing to forward toward Kubernetes. A direct, passive,
   Kubernetes-dataplane-side signal was required instead.
4. A live audit of this Minikube cluster's kube-proxy dataplane (iptables backed by `nf_tables`)
   found exactly such a signal: Service-specific, comment-identified NAT/filter rules that increment
   with real traffic independently of whether the application ever responds — validated under
   healthy-endpoint, multi-endpoint, and zero-endpoint conditions (see sec.B and the raw evidence
   artifact).
5. Therefore the evidence-layer change is versioned transparently as CORR-41/Protocol V6, rather
   than hidden inside an in-place CORR-39/V4 edit.

## CORR-41 sec.B: The Kubernetes-Dataplane Ingress Evidence Source

Two mutually-exclusive, kube-proxy-programmed regimes, both uniquely identified by a Service-specific
`iptables`/`nft` rule comment (`autoscaler-app/demo-app-loadtest:http[ has no endpoints]`), were
empirically confirmed live on this project's cluster (full raw evidence:
`persistent-data/transport_validation/protocol_v6_route_evidence/NON_OFFICIAL_ROUTE_EVIDENCE_VALIDATION_corr41_nodeport_dataplane_20260919T054000Z.json`):

```
Regime: endpoint_backed (>=1 Ready endpoint)
  table: nat
  rule:  -A KUBE-SEP-<hash> ... -j DNAT --to-destination <podIP>:8080
  comment: "autoscaler-app/demo-app-loadtest:http"
  cardinality: one chain per Ready endpoint; SUM all matching rules
  verified: 5 curls -> 5 pkts/300 bytes (1 endpoint); 10 curls -> 7+3=10 pkts (2 endpoints)

Regime: no_endpoints (0 Ready endpoints)
  table: filter
  rule:  -A KUBE-EXTERNAL-SERVICES ... --dport 31891 -j REJECT --reject-with icmp-port-unreachable
  comment: "autoscaler-app/demo-app-loadtest:http has no endpoints"
  cardinality: exactly one rule
  verified: scaling to 0 replicas removed EVERY endpoint_backed rule entirely (not merely froze
            it) and produced this REJECT rule instead; 1 curl -> 1/60, +4 curls -> 5/300 total,
            with every request refused (curl exit=7, "connection refused") and zero application
            involvement
```

Both regimes answer exactly one question — *did workload traffic reach the Kubernetes
Service/NodePort dataplane during this window* — and explicitly do **not** answer *did the
application successfully process it*. That second question remains entirely the domain of existing
SLA/performance/restart evidence, untouched by this correction.

**Rule identity is not stable across pod recreation** — a scale-to-0/scale-to-1 cycle changed
`KUBE-SEP-ENIF66D2G3466LGQ` to a brand-new `KUBE-SEP-OGXRNUUKR756PIKN` for the same Service. Any
observed change in `regime` or `rule_identity` between two consecutive samples is a baseline
reset, never a negative delta (sec.D).

**Contamination audit**: `k8s/app/service.yaml`, `k8s/app/service-monitor.yaml`, and
`k8s/app/service-loadtest.yaml` (pre-existing design decisions from Phase 11C.4B/CORR-21 and Phase
11C.4C/CORR-22) confirm Prometheus/Grafana/metrics-server scrape exclusively via the separate
`demo-app` ClusterIP Service — `demo-app-loadtest`'s NodePort and its dataplane counters are
structurally isolated from every other in-cluster traffic source.

**Sampling**: `docker exec minikube iptables-save -c` (nat and filter tables) measured 0.05–0.14s
per invocation — a 3-second sampling cadence (faster than the existing 10-second evidence window,
so every window can contain at least one fully-interior sample pair) keeps observer overhead to a
low single-digit percentage of one core on the 2-core Minikube VM. Reference implementation:
`scripts/dynamics_calibration_lib.py: parse_dataplane_ingress_rules()`,
`compute_dataplane_ingress_evidence_windows()`; `scripts/run_dynamics_calibration_slot.py:
sample_dataplane_ingress_rules()` and its periodic-sampling/classification wiring.

## CORR-41 sec.C: Dataplane Ingress Artifact

`cluster/loadtest_dataplane_ingress_timeline.json` (new, additive — does not replace
`cluster/loadtest_transport_lifecycle.json`, which is unchanged): periodic samples of
`sampled_at_utc`, `elapsed_seconds_since_traffic_start`, `dataplane_mode`, `service_namespace`,
`service_name`, `nodeport`, `regime`, `rule_identity`, `total_packet_count`, `total_byte_count`,
`observation_error`. `cluster/loadtest_route_integrity_timeline.json`'s per-window records gain:
`nodeport_dataplane_ingress_evidence`, `dataplane_counter_intervals_used`,
`dataplane_counter_intervals_rejected_cross_window`, `dataplane_counter_intervals_rejected_reset`,
`dataplane_observer_error_present`, `dataplane_backend_regimes_with_evidence`, and a combined
`route_evidence_sources` list.

## CORR-41 sec.D: Updated Route-Integrity Classification — CORR-39 Extended, Not Replaced

Unchanged from V4/V5: the materiality criterion (25%, scored-window overlap), 10-second half-open
evidence windows, window-local HTTP dispatch-time attribution, raw-counter window-local-only
attribution, rolling-1m server-RPS supporting-only status, and — critically — **FAILED precedence**:
`relay_container_running=false` or an absent expected host listener during a window is decisive and
unconditional, regardless of any positive dataplane evidence in that same window.

**New CONFIRMED rule**: a window is `CONFIRMED` when no direct harness-path failure is proven AND
at least one of `{http_response_delivery_evidence, application_counter_delivery_evidence,
nodeport_dataplane_ingress_evidence}` is true — any one source suffices, none is required, and all
three may coexist. `classify_route_integrity_window()`'s new `nodeport_dataplane_ingress_evidence`
parameter defaults to `None`, so every historical call site and every pre-CORR-41 classification is
reproduced exactly unchanged when the parameter is omitted.

**Endpoint/context separation** (sec.17 of this phase's governing instructions): endpoints=0 +
positive `nodeport_dataplane_ingress_evidence` → route `CONFIRMED`, application/backend
availability recorded separately as failed/unavailable. endpoints>0 + no positive evidence from any
of the three sources → route `UNVERIFIED`, exactly as before. A relay/listener failure is `FAILED`
regardless of endpoint count or any dataplane evidence.

**Observer failure**: a dataplane-counter observation error for a window is never treated as
`FAILED` — it means that one evidence source is unavailable for that window; the other two sources
may still `CONFIRM` it; if none do, `UNVERIFIED`.

**Performance independence** (the phase's central objective): timeout rate, p95 latency, zero Ready
endpoints, restarts, scrape failure, and SLA failure must never, by themselves, make a window's
route integrity `UNVERIFIED` when `nodeport_dataplane_ingress_evidence` (or either existing source)
directly proves workload traffic reached the workload path during that window.

Attempt-level aggregation (`FAILED` if any window `FAILED`; `CONFIRMED` only if every material
window is `CONFIRMED`; `UNVERIFIED` otherwise) and the Axis-A consequence (`FAILED`/material
`UNVERIFIED` → `execution_structural_validity=INVALID`) are both unchanged.

## CORR-41 sec.E: Non-Retroactivity

Index12 attempt3 did **not** collect the dataplane-ingress counter timeline this file defines —
Protocol V6 was frozen strictly *after*, and as a direct consequence of, reviewing attempt3's own
evidence. What those counters "would have shown" during attempt3's two material intervals is never
inferred, estimated, or backfilled. Attempt3 remains, permanently:
`RL_DYNAMICS_CALIBRATION_SEQUENCE_INDEX_12_ATTEMPT3_STRUCTURALLY_INVALID`,
`ROUTE_INTEGRITY_UNVERIFIED`. Ledger state is unchanged by this phase: `slot12.state =
INVALID_RETRY_REQUIRED`, `slot12.valid_attempt_id = None`, `slot12.all_attempt_ids = [attempt1,
attempt2, attempt3]`, `next_resumable_index() == 12`, `slot13 = NOT_ATTEMPTED`. This file applies
**prospectively only**, starting with a future, separately-authorized attempt4 — Phase 15B.3O.1
does not execute attempt4 or index13.

## What Remains Unaffected

Everything CORR-40/V5 preserved remains preserved: the 20 fitting cells, 9 interpolation cells, 8
replica-transition types, 4 workload-transition types, 102 global planned slots (unchanged,
unrenumbered), 37 canonical schedules and their SHA-256 hashes, the global campaign sequence and
its order, the 64 RPS ceiling, 400 max-in-flight, 5s timeout, decision-aligned timestamps
(60/90/120/150), the two-axis validity model, every routing classification threshold, CPU
normalization (Phase 15B.3F.1), restart-evidence categories (Phase 15B.3J.1), SLA thresholds
(Phase 15B.3L.1), `rl_contract_version`/`reward_contract_version` (both still 2), and — explicitly
— the V5 workload-transport mechanism in every particular (socat relay, NodePort, connection mode,
client location; `transport_protocol_version` stays 5). `HPA260_V5_REVALIDATED` and
`CROSS_EPOCH_COMPATIBLE_FOR_MODEL_FITTING` (Phase 15B.3N.4/15B.3N.4A) are unchanged — HPA-260 was
not rerun and the pre-V5/V5 bridge was not rerun by this phase, since the workload transport,
application, resources, and schedules are all unchanged. Index12's own frozen schedule and its
SHA-256 (`4594f8ce7f813f39588a362360f17e596797e0b8cf579990e08d3bfed33c4dda`) are unchanged — no
schedule regeneration. No dynamics model was fitted, no PPO trained. Sequence index 13 was not
executed; a fourth attempt (attempt4) of sequence index 12 was not executed by this phase.
