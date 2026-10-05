# RL Empirical Dynamics Calibration Protocol — v5 (CORR-40, FOURTH POST-DATA CORRECTION)

**Status**: Frozen by Phase 15B.3N.3, correction reference **CORR-40**, following **CORR-39**
(v4). This is the **fourth post-data** correction to the calibration protocol — v1 (CORR-31
through CORR-35) was entirely pre-execution preregistration; v2 (CORR-37) was frozen after
`campaign_sequence_index=1`'s real evidence; v3 (CORR-38) was frozen after indices 1/2's
routing-relevance evidence; v4 (CORR-39) was frozen after index12 attempt1's evidence, then
hardened three times in place before it ever governed a live attempt. This file governs **only**
the concrete official workload-transport mechanism (CORR-40), the transport-neutral evidence
artifact, explicit transport provenance on attempt metadata, and the historical-epoch
compatibility record. `docs/rl_dynamics_calibration_protocol_v4.md` (route-integrity
classification, unchanged in substance), `docs/rl_dynamics_calibration_protocol_v3.md` (routing
relevance), `docs/rl_dynamics_calibration_protocol_v2.md` (two-axis validity/ledger semantics),
and `docs/rl_dynamics_calibration_protocol.md` (v1, the base matrix) remain the narrative
documents for everything not explicitly redefined here. The machine-readable counterpart is
`experiments/manifests/rl_dynamics_calibration_v5.yaml`, which inherits everything else unchanged,
by reference, from v4 (which itself inherits from v3, v2, and v1).

```
dynamics_calibration_protocol_version: 5   (fourth POST-DATA correction)
frozen_in_phase: 15B.3N.3
correction_reference: CORR-40 (follows CORR-39)
frozen_after_observing_empirical_attempts: true
observed_attempt_ids:
  - fit_dyncal_r8_L3_repeat1_attempt1_20260916T130416Z   (campaign_sequence_index=12, pre-CORR-39)
  - fit_dyncal_r8_L3_repeat1_attempt2_20260918T023436Z   (campaign_sequence_index=12, V4/CORR-39-governed, route_integrity=FAILED)
```

## Why This Had to Be a New Version, Not an In-Place v4 Edit

Index12 attempt2 was the first live attempt Protocol V4/CORR-39 ever governed, and its hardened
route-integrity classifier worked exactly as designed: it caught a genuine, decisive harness-side
transport failure (the `minikube service --url` SSH tunnel's local listener disappeared at
elapsed ~60s and never returned, while the wrapping process stayed alive) and correctly returned
`FAILED`, invalidating the attempt. V4's classification logic is not being changed here — it is
not wrong. What changed is a dedicated engineering investigation (Phases 15B.3N.1, 15B.3N.2,
15B.3N.2A) that root-caused the failure as **structural**: the SSH-forwarding mechanism cannot
reliably sustain this project's actual fresh-connection rate (`connection_mode=fresh`, ~64
new TCP connections/second for 180 seconds) for a full official run, reproduced identically three
independent times (attempt2 itself, and two further reconstructions), always failing within the
same ~60–67 second window, regardless of whether `minikube`'s own CLI wrapper or the harness
itself owns the SSH process. Deciding to change the concrete transport mechanism, after observing
that its failure is structural rather than an evidentiary gap, is a post-data methodology decision
by this project's own established precedent (CORR-27/32/37/38/39) — it must be versioned
transparently, not folded into an unversioned bug-fix label or silently edited into v4.

## CORR-40 sec.A: Rationale

1. The V4 SSH transport (`minikube service --url`) was demonstrated to fail structurally under
   the actual frozen workload — reproduced three independent times, always within the same
   ~60–67s window of a 180s run.
2. Direct harness ownership of the SSH process (eliminating the `minikube` CLI wrapper,
   "Candidate E") did **not** solve it — it failed at elapsed 66.8s under the exact real workload
   in Phase 15B.3N.2A's durable reconstruction.
3. A raw TCP relay container on the Minikube Docker network, forwarding to the Minikube node's
   NodePort ("Candidate F"), survived repeated exact-workload validation: a clean baseline run and
   a separately-labeled run with injected mid-run pod churn, both completing the full 180+ seconds
   with zero container/listener exceptions across 121 combined observations.
4. Candidate F preserves: Mac-host traffic generation; fresh TCP connections (`socat
   TCP-LISTEN:...,fork,reuseaddr`, one inbound mapped to one outbound); Kubernetes Service-level
   backend selection (the NodePort routes through the identical kube-proxy dataplane);
   `sessionAffinity=None`; the same application endpoint (`/work`) and intensity; the same frozen
   traffic schedules.
5. Candidate F changes the concrete access path: ClusterIP-via-SSH becomes
   NodePort-via-Docker-network-relay. This is **not** claimed to be an identical path — only a
   path that preserves the properties enumerated above.
6. The latency penalty for Candidate F reported in Phase 15B.3N.2 did **not** survive Phase
   15B.3N.2A's properly counterbalanced 3-block replication (the F−E delta flipped sign between
   blocks) and is explicitly **not** relied upon as a rationale here.
7. Therefore the transport change is versioned transparently as CORR-40/Protocol V5, rather than
   hidden inside an in-place V4 edit.

## CORR-40 sec.B: Frozen V5 Official Transport Route

```
Mac host traffic generator
  -> 127.0.0.1:<Docker-published host port>
  -> dedicated raw TCP relay container (image: alpine/socat, --restart=no,
     attached to the Minikube Docker network)
       socat TCP-LISTEN:<container_port>,fork,reuseaddr TCP:<node_ip>:<nodeport>
  -> Minikube node IP : demo-app-loadtest NodePort
  -> Kubernetes Service dataplane (kube-proxy, unchanged backend-selection semantics)
  -> demo-app pods
```

The relay is **raw TCP only**: no HTTP awareness, no retries, no caching, no connection pooling,
no application-layer load balancing — `fork` gives a strict one-inbound-to-one-outbound-connection
mapping, so Kubernetes (via the NodePort/Service dataplane) remains solely responsible for backend
selection, exactly as it would for any other NodePort client. `--restart=no` is explicit and
non-negotiable: no supervisor, no automatic relaunch, no Docker restart policy may hide a relay
failure — CORR-39's `FAILED` classification must remain reachable. The host port is published on
`127.0.0.1` **only**, never `0.0.0.0`, using Docker's own ephemeral-port allocation
(`-p 127.0.0.1::<container_port>`) when not manually pinned; the actual bound port is discovered
structurally afterward (`docker port`/`docker inspect`), never guessed. Reference implementation:
`scripts/dynamics_calibration_lib.py: build_relay_docker_run_command()`,
`build_relay_target_url()`; `scripts/run_dynamics_calibration_slot.py: start_loadtest_transport()`.

**No fallback**: the transport is one deterministic implementation for an official attempt — there
is no runtime fallback to `minikube service --url` or any other mechanism if the relay fails to
launch or dies mid-run. If the relay cannot launch, join the network, publish the loopback port,
or establish the expected listener before scientific traffic begins, the harness **stops before
traffic** — no attempt whose intended treatment never began is created, per existing preflight
semantics. If the relay fails **during** traffic, it is never restarted; the run continues (or is
preserved as-is) and CORR-39's evidence contract classifies the resulting route integrity exactly
as it would any other transport failure — `FAILED` still invalidates Axis A.

## CORR-40 sec.C: Transport Lifecycle Artifact

`cluster/loadtest_transport_lifecycle.json` (transport-neutral, replacing
`cluster/loadtest_tunnel_lifecycle.json` for V5 attempts going forward — the SSH-specific artifact
and its parsing code are retained unchanged for reading historical attempts, never deleted, never
backfilled onto V5 attempts): `transport_type`, `container_id`, `container_name`, `image`,
`image_id`, `repo_digest`, `docker_network`, `node_ip`, `nodeport`, `host_ip`, `host_port`,
`container_port`, `target_url`, `launch_command`, `launch_time`,
`periodic_samples_during_traffic` (each sample: `sampled_at_utc`,
`elapsed_seconds_since_traffic_start`, `relay_container_running`, `relay_container_exit_code`,
`host_listener_present`, `loadtest_ready_endpoint_count`, `endpoint_query_error`),
`running_at_traffic_end`, `exit_code_at_traffic_end`, `host_listener_present_at_traffic_end`,
`cleanup_action`, `exit_code_after_cleanup`, `docker_logs_path`.

## CORR-40 sec.D: CORR-39 Mapping — Classification Rules Unchanged in Substance

CORR-39's materiality criterion (25%, scored-window overlap), 10-second half-open evidence
windows, window-local HTTP dispatch-time attribution, raw-counter window-local-only attribution
(no duplication across windows), and rolling-1m server-RPS supporting-only status are **all
unchanged**. Only the observer's data source changes: a material evidence window is `FAILED` when
`relay_container_running=false` or the expected host listener is absent (the direct equivalent of
V4's `tunnel_process_alive=False`/`local_listener_present=False`); a window with a healthy
container, present listener, and present endpoints but no window-local delivered-request evidence
is still `UNVERIFIED`, exactly as under V4 — component health remains supporting-only, never
sufficient alone. Attempt-level aggregation (`FAILED` if any window `FAILED`; `CONFIRMED` only if
every material window is `CONFIRMED`; `UNVERIFIED` otherwise) and the Axis-A consequence
(`FAILED`/material `UNVERIFIED` → `execution_structural_validity=INVALID`) are both unchanged.

## CORR-40 sec.E: Historical Transport Epochs

**Pre-V5 epoch**: `SSH tunnel -> Service ClusterIP`. Applies to `campaign_sequence_index` 1–11
(`VALID`, unchanged, not reclassified) and index12 attempt1 (pre-CORR-39, `INVALID`) and attempt2
(V4/CORR-39-governed, `route_integrity=FAILED`, `INVALID`). **V5 epoch**: `Docker-network raw TCP
relay -> Service NodePort`. Applies to the index12 retry (attempt3, **not executed by this
phase, nor by Phase 15B.3N.4**) onward, once separately authorized. Both
`cross_epoch_model_fitting_status` and `hpa_260_status` were resolved by Phase 15B.3N.4's
dedicated non-official transition-validation runs — see `analysis/transport_epoch_compatibility.json`
and `docs/experimental_design_corrections.md` CORR-40 "Phase 15B.3N.4 Transition-Validation
Result" for full evidence:

- `cross_epoch_model_fitting_status = COMPATIBLE_WITH_V5`: a 24-run (4 cells x 2 transports x 3
  repeats, counterbalanced) bridge trial found decision-point medians for server RPS, aggregate
  CPU, server p95 latency, and memory comparable between transports in every cell, with no
  consistent cross-cell directional shift and zero restarts under either transport. One residual
  finding is explicitly flagged, not treated as disqualifying: 2 of 12 V5-transport repeats showed
  a transient right-tail latency/reliability degradation (route_integrity remained `CONFIRMED` in
  both; the V5 relay itself showed zero anomalies throughout both runs) with no counterpart under
  the old transport in any of its 12 repeats — plausibly ordinary application/scheduling-level
  variability rather than a transport-layer effect, but retained here as an open item for future
  V5 recollection of `dyncal_r1_L3`/`dyncal_r8_L1` to keep in view.
- `hpa_260_status = HPA260_V5_REVALIDATED`: 3 V5-transport repeats of the exact frozen historical
  HPA-tuning workload were all structurally valid, met SLA, triggered in the same historical
  scaling phase (`scale_pressure_6rps`, 3/3), and showed no new oscillation or reversal.
  HPA-200%/230% were not retested. Neither this file nor Phase 15B.3N.4 retunes the 260% value;
  it is revalidated as-is under the V5 transport.

## What Remains Unaffected

The 20 fitting cells, 9 interpolation cells, 8 replica-transition types, 4 workload-transition
types, 102 global planned slots (unchanged, unrenumbered), 37 canonical schedules and their
SHA-256 hashes, the global campaign sequence and its order, decision-aligned timestamps
(60/90/120/150) and their existing fit-eligibility rule, the 64 RPS ceiling, 400 max-in-flight, 5s
timeout, the two-axis validity model, every routing classification threshold, Protocol V3's
`relevant_pod_set_j`/complete-joint-observability rules, CPU-normalization definitions (Phase
15B.3F.1), restart-evidence categories (Phase 15B.3J.1), and SLA thresholds/dual-metric-source
discipline (Phase 15B.3L.1) are all unchanged. `rl_contract_version`/`reward_contract_version`
(both still 2) are untouched. The HPA baseline definition (`k8s/hpa/hpa.yaml`,
`averageUtilization: 260`) is **not edited** by this file — its value is unchanged, but its
validity status for future RL-vs-HPA evaluation is explicitly `PENDING_TRANSPORT_TRANSITION_
REVALIDATION` per sec.E, confirmed live. Index12's own frozen schedule and its SHA-256
(`4594f8ce7f813f39588a362360f17e596797e0b8cf579990e08d3bfed33c4dda`) are unchanged — no schedule
regeneration. No dynamics model was fitted, no PPO trained, while freezing this correction.
`configs/traffic_profiles/unseen_eval.yaml` was not inspected. Sequence index 13 was not executed;
a retry (attempt3) of sequence index 12 was not executed by this phase.
