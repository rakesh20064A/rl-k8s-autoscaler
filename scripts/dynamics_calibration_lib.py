"""scripts/dynamics_calibration_lib.py

Phase 15B.3A: pure, unit-testable helper functions for the empirical
dynamics-calibration execution harness (`scripts/run_dynamics_calibration_slot.py`).

Every function here is deliberately free of network/subprocess/Kubernetes
calls so it can be exercised offline in `tests/unit/`. All scientific
parameters (which cells, schedules, thresholds) are read from the FROZEN
manifest/global-sequence/schedule-registry -- this module derives, verifies,
and computes; it never invents or overrides a frozen value.

Implements exactly the CORR-32/33/34/35 rules already frozen in
`experiments/manifests/rl_dynamics_calibration_v1.yaml`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "experiments/manifests/rl_dynamics_calibration_v1.yaml"
MANIFEST_V2_PATH = PROJECT_ROOT / "experiments/manifests/rl_dynamics_calibration_v2.yaml"
MANIFEST_V3_PATH = PROJECT_ROOT / "experiments/manifests/rl_dynamics_calibration_v3.yaml"
PROTOCOL_DOC_V3_PATH = PROJECT_ROOT / "docs/rl_dynamics_calibration_protocol_v3.md"
MANIFEST_V4_PATH = PROJECT_ROOT / "experiments/manifests/rl_dynamics_calibration_v4.yaml"
PROTOCOL_DOC_V4_PATH = PROJECT_ROOT / "docs/rl_dynamics_calibration_protocol_v4.md"
MANIFEST_V5_PATH = PROJECT_ROOT / "experiments/manifests/rl_dynamics_calibration_v5.yaml"
PROTOCOL_DOC_V5_PATH = PROJECT_ROOT / "docs/rl_dynamics_calibration_protocol_v5.md"
MANIFEST_V6_PATH = PROJECT_ROOT / "experiments/manifests/rl_dynamics_calibration_v6.yaml"
PROTOCOL_DOC_V6_PATH = PROJECT_ROOT / "docs/rl_dynamics_calibration_protocol_v6.md"
MANIFEST_V7_PATH = PROJECT_ROOT / "experiments/manifests/rl_dynamics_calibration_v7.yaml"
PROTOCOL_DOC_V7_PATH = PROJECT_ROOT / "docs/rl_dynamics_calibration_protocol_v7.md"
MANIFEST_V8_PATH = PROJECT_ROOT / "experiments/manifests/rl_dynamics_calibration_v8.yaml"
PROTOCOL_DOC_V8_PATH = PROJECT_ROOT / "docs/rl_dynamics_calibration_protocol_v8.md"
SCHEDULE_REGISTRY_PATH = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration/schedule_registry.json"
GLOBAL_SEQUENCE_PATH = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration/global_campaign_sequence.json"
LEDGER_PATH = PROJECT_ROOT / "persistent-data/dynamics_calibration/campaign_ledger.json"
CALIBRATION_ROOT = PROJECT_ROOT / "persistent-data/dynamics_calibration"

# Phase 15B.3N.2A: durable, permanent home for NON_OFFICIAL_TRANSPORT_VALIDATION
# engineering evidence (see persistent-data/transport_validation/README.md).
# Deliberately a SEPARATE tree from CALIBRATION_ROOT -- campaign/model-fitting
# discovery in this module is driven entirely by GLOBAL_SEQUENCE_PATH (the
# fixed 102-entry manifest) and LEDGER_PATH (populated only via explicit
# campaign_sequence_index-keyed update_slot_state() calls); nothing here globs
# or scans persistent-data/ for attempts, so this directory can never be
# silently picked up as campaign evidence, fitting-support, or a calibration
# repeat, regardless of what an official-looking experiment id it might carry.
TRANSPORT_VALIDATION_ROOT = PROJECT_ROOT / "persistent-data/transport_validation"

LEDGER_STATE_NOT_ATTEMPTED = "NOT_ATTEMPTED"
LEDGER_STATE_VALID = "VALID"
LEDGER_STATE_INVALID_RETRY_REQUIRED = "INVALID_RETRY_REQUIRED"

ROUTING_INSUFFICIENT_SAMPLE = "INSUFFICIENT_ROUTING_SAMPLE"
ROUTING_STRUCTURAL_FAILURE = "STRUCTURAL_ROUTING_FAILURE"
ROUTING_HEALTHY = "HEALTHY_VARIANCE"
ROUTING_PINNING_SUSPECTED = "ROUTING_PINNING_SUSPECTED"

# ---------------------------------------------------------------------------
# CORR-37 (Phase 15B.3B): two-axis validity semantics -- Axis A
# (execution_structural_validity) vs Axis B (steady_state_fit_eligibility).
# See experiments/manifests/rl_dynamics_calibration_v2.yaml.
# ---------------------------------------------------------------------------

EXECUTION_VALID = "VALID"
EXECUTION_INVALID = "INVALID"

FIT_RUN_ELIGIBLE = "STEADY_STATE_FIT_ELIGIBLE"
FIT_RUN_PARTIAL = "STEADY_STATE_FIT_PARTIAL"
FIT_RUN_INELIGIBLE = "STEADY_STATE_FIT_INELIGIBLE"

FIT_POINT_ELIGIBLE_NONE = "NONE"
FIT_POINT_INELIGIBLE_HEALTH_INSTABILITY = "LOAD_INDUCED_HEALTH_INSTABILITY"
FIT_POINT_INELIGIBLE_TARGET_UNOBSERVABLE = "APPLICATION_TARGET_UNOBSERVABLE_UNDER_LOAD"

CELL_STABLE_SUPPORT = "STABLE_CALIBRATION_SUPPORT"
CELL_OUTSIDE_SUPPORT = "OUTSIDE_STABLE_CALIBRATION_SUPPORT"
CELL_SUPPORT_PENDING = "SUPPORT_DETERMINATION_PENDING_REMAINING_REPEATS"

MIN_FIT_ELIGIBLE_POINTS_FOR_SUFFICIENT_REPEAT = 2  # of 4 decision points

# ---------------------------------------------------------------------------
# Phase 15B.3L.1: derived, reporting-only SLA-attainment diagnostics.
#
# Frozen source values (unchanged by this phase, only READ and reused here):
#   configs/experiment.yaml: sla.p95_latency_ms=300, sla.error_rate=0.01
#   configs/reward.yaml:     sla.p95_target_ms=300 (SYNC'd to experiment.yaml)
#
# Two DIFFERENT, both already-frozen latency SOURCES are compared against
# this same 300ms numeric threshold for two DIFFERENT purposes -- callers
# must be explicit about which one they are passing in:
#   - CLIENT p95 (traffic-generator-observed, `client_latency_stats.p95_seconds`):
#     the CORR-18 Sec.9 Phase-11C HPA-tuning candidate-selection SLA
#     ("SLA met <=> client p95 < 300ms").
#   - SERVER p95 (Prometheus histogram, `latency_p95_seconds` /
#     `demo_app_request_duration_seconds_bucket`): the metric the RL reward's
#     own SLA term (configs/reward.yaml sla.p95_target_ms) and the dynamics-
#     calibration protocol (CORR-33 sec.29: "client p95 is never a policy
#     input and never silently substitutes for server p95 in model
#     selection") actually operate on.
#
# `compute_sla_assessment()` computes the same latency/success/overall
# verdict shape for either source -- it never decides which one a report
# should use; the caller documents that. This is a REPORTING-ONLY
# diagnostic: it must never be conflated with, or substituted for,
# `execution_structural_validity`/`steady_state_fit_eligibility` -- SLA
# violation is explicitly listed as a non-invalidating performance outcome
# (`valid_performance_outcomes_not_grounds_for_invalidity: very_high_latency`,
# experiments/manifests/rl_dynamics_calibration_v1.yaml) and is never itself
# an Axis-B fit-exclusion criterion (`assess_decision_point_eligibility`
# only ever excludes on an unobservable sample or a Ready-count mismatch).
# ---------------------------------------------------------------------------

SLA_P95_LATENCY_TARGET_SECONDS = 0.300
SLA_SUCCESS_RATE_THRESHOLD = 0.99


def compute_sla_assessment(
    p95_latency_seconds: Optional[float],
    success_rate: Optional[float],
    p95_target_seconds: float = SLA_P95_LATENCY_TARGET_SECONDS,
    success_rate_threshold: float = SLA_SUCCESS_RATE_THRESHOLD,
) -> Dict[str, Any]:
    """Derive the frozen-threshold SLA verdict for one (p95, success_rate)
    pair. `p95_latency_seconds` must be a SINGLE metric the caller has
    already chosen (client or server, see module docstring above) --
    this function does not know or care which. Returns `None` for a
    verdict whenever its required input is missing, never fabricating a
    pass/fail from an absent measurement."""
    latency_sla_met = None if p95_latency_seconds is None else p95_latency_seconds < p95_target_seconds
    success_sla_met = None if success_rate is None else success_rate >= success_rate_threshold
    overall_sla_met = None if (latency_sla_met is None or success_sla_met is None) else (latency_sla_met and success_sla_met)
    return {
        "p95_latency_seconds": p95_latency_seconds,
        "p95_target_seconds": p95_target_seconds,
        "success_rate": success_rate,
        "success_rate_threshold": success_rate_threshold,
        "latency_sla_met": latency_sla_met,
        "success_sla_met": success_sla_met,
        "overall_sla_met": overall_sla_met,
    }


TUNNEL_TERMINATION_ALREADY_EXITED = "already_exited_before_cleanup_ran"
TUNNEL_TERMINATION_EXPLICITLY_TERMINATED = "explicitly_terminated_by_cleanup"


def classify_tunnel_termination_action(already_dead_before_cleanup: bool) -> str:
    """Phase 15B.3M.1: names which of the two possible cleanup histories
    happened for a background subprocess (the official load-generator
    tunnel, in practice) -- distinguishing 'cleanup found it already dead'
    from 'cleanup explicitly terminated a still-running process'. These are
    different facts: the former means the process died on its own at some
    unknown earlier point (evidence of a possible mid-run failure); the
    latter means it was alive until cleanup deliberately stopped it (no
    evidence of an unplanned death, though this alone does not prove it was
    functioning correctly the whole time -- see `compute_sla_assessment`'s
    own module docstring for the analogous "don't over-claim from a boolean"
    discipline)."""
    return TUNNEL_TERMINATION_ALREADY_EXITED if already_dead_before_cleanup else TUNNEL_TERMINATION_EXPLICITLY_TERMINATED


TRANSPORT_TYPE_DOCKER_NETWORK_TCP_RELAY = "docker_network_tcp_relay"
RELAY_IMAGE = "alpine/socat"


def build_relay_docker_run_command(
    container_name: str,
    docker_network: str,
    host_port: Optional[int],
    container_port: int,
    node_ip: str,
    nodeport: int,
    image: str = RELAY_IMAGE,
) -> List[str]:
    """Phase 15B.3N.3 (CORR-40, Protocol V5): deterministic construction of
    the official `docker run` command for the raw-TCP relay -- the frozen
    V5 workload transport (`host -> Docker-published loopback port -> raw
    TCP relay container on the Minikube Docker network -> Minikube node
    IP:demo-app-loadtest NodePort -> Kubernetes Service dataplane -> pods`).
    Never HTTP-aware, never retries, never pools connections, never load-
    balances -- `socat TCP-LISTEN:<port>,fork,reuseaddr TCP:<node_ip>:
    <nodeport>` maps exactly one inbound TCP connection to exactly one
    outbound TCP connection, so Kubernetes (via the NodePort/Service
    dataplane) remains solely responsible for backend selection.

    `--restart=no` is explicit and non-negotiable: no supervisor, no
    automatic relaunch, no Docker restart policy may hide a relay failure
    -- CORR-39's FAILED classification must remain reachable. The host
    port is published on `127.0.0.1` ONLY, never `0.0.0.0`. When
    `host_port` is `None`, Docker's own ephemeral-port allocation is used
    (`-p 127.0.0.1::<container_port>`) rather than a manually-guessed free
    port; the caller must resolve the actual assigned port afterward via
    `docker port`/`docker inspect` (structural discovery, never assumed).
    """
    # `alpine/socat`'s own image ENTRYPOINT is already `socat` -- the
    # container command is only the two address arguments, never a
    # redundant leading "socat" (which `socat` itself would otherwise
    # misparse as a third address and refuse to start: "exactly 2
    # addresses required (there are 3)", caught by the Phase 15B.3N.3
    # non-official integration smoke test before this was ever used for
    # an official attempt).
    publish = f"127.0.0.1:{host_port}:{container_port}" if host_port is not None else f"127.0.0.1::{container_port}"
    return [
        "docker", "run", "-d",
        "--name", container_name,
        "--network", docker_network,
        "--restart", "no",
        "-p", publish,
        image,
        f"TCP-LISTEN:{container_port},fork,reuseaddr", f"TCP:{node_ip}:{nodeport}",
    ]


def build_relay_target_url(host_port: int, host_ip: str = "127.0.0.1") -> str:
    """Phase 15B.3N.3: the traffic generator's `--target-url` for the V5
    transport -- always the loopback address the relay's host port was
    published on, never the Minikube node IP or NodePort directly (the Mac
    host cannot route to those at all under the Docker driver -- Phase
    15B.3N.1/15B.3N.2 proved this empirically)."""
    return f"http://{host_ip}:{host_port}"


# ---------------------------------------------------------------------------
# Phase 15B.3M.2: CORR-39 resolved contract for official external
# workload-route integrity. `offered==sent`/`dropped_overload==0` prove only
# that the generator attempted the frozen schedule against its own
# configured destination -- they say nothing about whether that destination
# actually delivered the request into the intended Kubernetes workload path
# (host -> Minikube-service tunnel -> demo-app-loadtest Service -> pods).
# This distinguishes GENERATOR SCHEDULE EXECUTION INTEGRITY (already proven
# by those two checks) from OFFICIAL WORKLOAD-ROUTE INTEGRITY (governed here).
# ---------------------------------------------------------------------------

ROUTE_INTEGRITY_CONFIRMED = "CONFIRMED"
ROUTE_INTEGRITY_FAILED = "FAILED"
ROUTE_INTEGRITY_UNVERIFIED = "UNVERIFIED"

INVALID_OFFICIAL_WORKLOAD_ROUTE_INTEGRITY_UNVERIFIED = "official_workload_route_integrity_unverified"

# Below this fraction of the traffic duration, a zero-success interval is
# treated as an ordinary, non-material performance blip (e.g. a brief
# reconnection storm) -- never itself grounds to question route integrity.
DEFAULT_ROUTE_INTEGRITY_MATERIALITY_FRACTION = 0.25

# Phase 15B.3M.4: the temporal resolution at which route-integrity evidence
# is evaluated within a material interval -- matches the existing tunnel/
# listener/endpoint observer cadence and the metrics/counter collector
# cadence, so no new active monitoring cadence is introduced, and the
# partition is fully reproducible offline from already-persisted files.
ROUTE_INTEGRITY_EVIDENCE_WINDOW_SECONDS = 10.0

# Phase 15B.3P.2 (CORR-43, Protocol V8): a deterministic, exhaustively
# (not Monte-Carlo) derived safety threshold for `partition_into_evidence_
# windows()`'s optional `min_window_seconds` merge behavior -- see
# docs/experimental_design_corrections.md CORR-43 for the full derivation.
# For a periodic sampler with worst-case realized period T, a window of
# width r is GUARANTEED (over every possible sampler/window phase offset,
# proven by exact formula floor(r/T) and confirmed by exhaustive fine-grid
# enumeration, not sampling) to contain at least k interior samples iff
# r >= k*T. `DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8 = 3.0`
# is a conservative, measured-with-margin bound on the V7/CORR-42 dataplane
# sampler's REAL inter-sample gap (2.0s nominal sleep +
# `sample_dataplane_ingress_rules()` command duration; Phase 15B.3P.1
# measured a live mean of 0.191s and a worst observed 0.306s over 7 real
# invocations; Phase 15B.3P.2's own live validation re-measured this with a
# larger sample -- see the phase's report -- and both remain comfortably
# under the ~1.0s margin this constant assumes). Two thresholds follow:
#   - 2*T = the minimum window width that guarantees >=2 interior samples
#     (sufficient for one confirming pair, ASSUMING no identity/regime
#     transition occurs inside the window);
#   - 3*T = the minimum window width that guarantees >=3 interior samples,
#     which is what is actually needed to remain robust to a SINGLE
#     identity/regime transition landing anywhere inside the window (a
#     transition invalidates exactly one adjacent sample pair; with 3
#     samples, at least one of the two adjacent pairs is unaffected no
#     matter where the transition falls, without relying on CORR-42's
#     reactive resample also completing before the window's own end).
# `MIN_EVIDENCE_WINDOW_SECONDS_V8` uses the STRICTER (larger) 3*T bound, so
# a window built to satisfy it is deterministically safe against BOTH
# failure classes attempt4 exhibited (the pure short-tail count shortage at
# `[175,179)`, and -- independent of and in addition to CORR-42's own
# reactive-resample mitigation -- the identity-transition-coincidence class
# exemplified by `[115,125)`) -- CONDITIONAL on the inter-sample gap
# actually staying at or below T.
#
# Phase 15B.3P.3 audit amendment (pre-first-use -- V8 has not yet been used
# by any OFFICIAL attempt, so this is amended in place per project
# convention rather than requiring a new correction/version): the above
# `floor(r/T)` derivation is a sound, exhaustively-verified mathematical
# IMPLICATION -- IF every real inter-sample gap is <= T, THEN a window of
# width >= k*T contains >= k interior samples, for any phase. What this
# constant does NOT claim, and never did claim in the implementation
# itself, is that T=3.0s is a hard, OS/Docker-enforced REAL-TIME bound. It
# is not: `_sample_dataplane_ingress_periodically()` sleeps a fixed
# `DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7` (2.0s) AFTER each
# sample's `docker exec` calls complete (not against an absolute monotonic
# deadline), so the realized gap is `2.0s + command_duration`, and
# `sample_dataplane_ingress_rules()`'s own two sequential `docker exec`
# calls each carry a `timeout=5` -- meaning a pathologically slow but
# still-successful command could in principle push a single gap well
# above 3.0s, and a Python process invoking Docker on macOS is not a
# hard-real-time system in the first place. `T=3.0` is therefore an
# empirically well-supported (Phase 15B.3P.1: n=7, mean 0.191s, max
# 0.306s; Phase 15B.3P.2: n=24, mean 0.258s, max 0.644s; Phase 15B.3P.3:
# see this phase's own live M1-M4 measurements) but NOT hard-guaranteed
# assumption. This project therefore does not claim "a guaranteed maximum
# inter-sample gap" as a real-time property. What IS true, and load-bearing
# instead: every real inter-sample gap is now explicitly RECORDED
# (`actual_gap_seconds_from_previous_sample`, `acquisition_gap_exceeded_
# assumed_bound` -- see `summarize_dataplane_acquisition_gaps()`) for
# post-hoc audit, and `compute_dataplane_ingress_evidence_windows()` can
# NEVER fabricate a confirming pair regardless of gap size -- an unusually
# large gap can only ever result in FEWER (or zero) confirming pairs being
# available, correctly and honestly reflected as `UNVERIFIED`, never as a
# false `CONFIRMED`. The window-partitioning correction itself (merging a
# short trailing remainder) remains fully justified independent of this
# distinction: it removes a KNOWN GEOMETRIC pathology (an
# arithmetic-remainder-driven short window) that existed regardless of how
# tightly T could ever be bounded.
DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8 = 3.0
MIN_EVIDENCE_WINDOW_SECONDS_FOR_COUNT_GUARANTEE_V8 = 2 * DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8
MIN_EVIDENCE_WINDOW_SECONDS_V8 = 3 * DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8


def find_all_zero_success_runs(rows: Sequence[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """From per-second traffic-summary rows (each with numeric-coercible
    `elapsed_seconds`, `sent_requests`, `success_2xx`), find EVERY maximal
    run of consecutive seconds where `sent_requests > 0` and
    `success_2xx == 0` -- i.e. every total-delivery-failure interval, with
    no assumption about WHY (application, Service/dataplane, or external
    tunnel), and no assumption that only the single longest run matters
    (Phase 15B.3M.4: a run can contain more than one independently-material
    zero-success interval; the frozen materiality rule is applied per
    interval, never only to whichever happens to be longest). Returns a
    list of `(start_elapsed, end_elapsed)` tuples in encounter order; empty
    if no second ever has zero success while requests were sent. Rows are
    assumed pre-sorted by `elapsed_seconds`, one row per second (matching
    `loadgen/traffic_generator.py`'s own summary cadence)."""
    runs: List[Tuple[float, float]] = []
    run_start: Optional[float] = None
    run_end: Optional[float] = None
    for row in rows:
        sent = int(row["sent_requests"])
        success = int(row["success_2xx"])
        elapsed = float(row["elapsed_seconds"])
        if sent > 0 and success == 0:
            if run_start is None:
                run_start = elapsed
            run_end = elapsed
        else:
            if run_start is not None:
                runs.append((run_start, run_end))
            run_start = None
            run_end = None
    if run_start is not None:
        runs.append((run_start, run_end))
    return runs


def longest_zero_success_run(rows: Sequence[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    """The single longest of `find_all_zero_success_runs()`'s results, or
    `None` if there are none. Retained as a convenience/reporting function
    -- attempt-level route-integrity classification (Phase 15B.3M.4) must
    consult ALL materially-qualifying runs, never only this one."""
    runs = find_all_zero_success_runs(rows)
    if not runs:
        return None
    return max(runs, key=lambda r: r[1] - r[0])


def is_interval_material(
    interval_start: float,
    interval_end: float,
    traffic_duration_seconds: float,
    scored_window: Tuple[float, float],
    materiality_fraction_threshold: float = DEFAULT_ROUTE_INTEGRITY_MATERIALITY_FRACTION,
) -> bool:
    """CORR-39 sec.5's materiality gate, applied to ONE candidate interval
    (Phase 15B.3M.4: every independently-material interval is evaluated,
    not only the longest). All three required: the interval has positive
    duration, it overlaps `scored_window=(lo, hi)`, and its duration is at
    least `materiality_fraction_threshold` of the total traffic duration."""
    duration = interval_end - interval_start
    if duration <= 0 or not traffic_duration_seconds:
        return False
    overlaps_scored_window = interval_start < scored_window[1] and interval_end >= scored_window[0]
    fraction = duration / traffic_duration_seconds
    return overlaps_scored_window and fraction >= materiality_fraction_threshold


def partition_into_evidence_windows(
    interval_start: float,
    interval_end: float,
    window_seconds: float = None,
    min_window_seconds: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Phase 15B.3M.4: deterministic, offline-reproducible half-open
    `[start, end)` evidence windows of `window_seconds` (default
    `ROUTE_INTEGRITY_EVIDENCE_WINDOW_SECONDS`), aligned to `interval_start`
    itself (never to an absolute wall-clock boundary) and covering
    `[interval_start, interval_end)` exactly. A shorter final window is
    included, never dropped, when the interval length is not an exact
    multiple of `window_seconds`. 10s matches the existing tunnel/listener/
    endpoint observer cadence and the metrics/counter collector cadence, so
    no new active monitoring cadence is introduced.

    Phase 15B.3P.2 (CORR-43, Protocol V8): `min_window_seconds`, when given
    (non-`None`), changes ONLY how a non-exact-multiple remainder is
    handled -- it never changes any FULL window, never drops or duplicates
    any covered time, and never reorders windows. When the trailing
    (final) window produced by the loop above would be strictly shorter
    than `min_window_seconds` AND at least one other (full) window already
    exists, the trailing window is merged into the immediately preceding
    window instead of being kept as its own separate window -- extending
    that preceding window's end to the interval's own end, and removing
    the separate short trailing window. This never applies when the
    interval fits entirely within a single window (there is no preceding
    window to merge into -- this remains the sole irreducible short-whole-
    interval case; see CORR-43 sec.C), and never applies when the
    interval length is an exact multiple of `window_seconds` (no
    remainder exists to merge). When `min_window_seconds` is `None` (the
    default), this reproduces the EXACT pre-CORR-43 (through Protocol V7)
    behavior byte-for-byte -- no historical call site, and no historical
    attempt's already-persisted `evidence_windows`, are affected by this
    parameter's existence. `union(windows) == [interval_start,
    interval_end)` and windows remain pairwise non-overlapping and
    chronologically ordered in both the default and the merged case (see
    `TestPartitionIntoEvidenceWindowsCorr43Phase15B3P2` for the
    corresponding invariant proofs)."""
    if window_seconds is None:
        window_seconds = ROUTE_INTEGRITY_EVIDENCE_WINDOW_SECONDS
    windows: List[Tuple[float, float]] = []
    t = interval_start
    while t < interval_end:
        t_end = min(t + window_seconds, interval_end)
        windows.append((t, t_end))
        t = t_end
    if min_window_seconds is not None and len(windows) >= 2:
        last_start, last_end = windows[-1]
        if (last_end - last_start) < min_window_seconds:
            prev_start, _prev_end = windows[-2]
            windows[-2] = (prev_start, last_end)
            windows.pop()
    return windows


def classify_route_integrity_window(
    delivered_request_evidence: bool,
    tunnel_process_alive: Optional[bool] = None,
    local_listener_present: Optional[bool] = None,
    nodeport_dataplane_ingress_evidence: Optional[bool] = None,
) -> Dict[str, str]:
    """Phase 15B.3M.4: per-window (NOT attempt-level) CONFIRMED/FAILED/
    UNVERIFIED classification, scoped to a single
    `ROUTE_INTEGRITY_EVIDENCE_WINDOW_SECONDS`-wide evidence window inside a
    material zero-success interval. This is the temporal core of the
    hardening: the same three-way evidentiary rule Phase 15B.3M.3
    established at the interval level now applies PER WINDOW, so that one
    window's positive evidence can never stand in for another window's
    absence of evidence.

      - `tunnel_process_alive is False`, or `local_listener_present is
        False` (direct evidence a required harness-side component was
        down/absent DURING THIS WINDOW) -> `FAILED`. This precedence is
        UNCHANGED by Phase 15B.3O.1/CORR-41 below: a confirmed harness-path
        failure is always decisive, regardless of what any other evidence
        source shows.
      - otherwise, `delivered_request_evidence` truthy (positive proof a
        request crossed the harness boundary WITHIN THIS SAME WINDOW) OR
        `nodeport_dataplane_ingress_evidence` truthy (Phase 15B.3O.1/
        CORR-41: positive proof workload traffic reached the Kubernetes
        Service/NodePort dataplane WITHIN THIS SAME WINDOW, read directly
        from kube-proxy's own dataplane counters -- see
        `compute_dataplane_ingress_evidence_windows()`) -> `CONFIRMED`.
        CORR-41's entire point: this source is available even when the
        application has zero Ready endpoints or never responds, so a
        genuine backend/application collapse no longer forces UNVERIFIED
        the way it did for campaign_sequence_index=12 attempt3 -- backend
        availability remains a SEPARATE question, not a precondition for
        route-health evidence.
      - otherwise -> `UNVERIFIED`. Component-level health being healthy in
        this window is supporting evidence only, never sufficient proof of
        forwarding for this specific window. Omitting
        `nodeport_dataplane_ingress_evidence` (its default, `None`)
        reproduces the exact pre-CORR-41 (V4/V5) behavior of this function
        -- no historical call site or historical classification is
        affected.
    """
    if tunnel_process_alive is False:
        return {
            "window_route_status": ROUTE_INTEGRITY_FAILED,
            "reason": "tunnel/helper process confirmed not alive during this window",
        }
    if local_listener_present is False:
        return {
            "window_route_status": ROUTE_INTEGRITY_FAILED,
            "reason": "expected local tunnel listener confirmed absent during this window",
        }
    route_evidence_sources: List[str] = []
    if delivered_request_evidence:
        route_evidence_sources.append("delivered_request_evidence")
    if nodeport_dataplane_ingress_evidence:
        route_evidence_sources.append("nodeport_dataplane_ingress_evidence")
    if route_evidence_sources:
        return {
            "window_route_status": ROUTE_INTEGRITY_CONFIRMED,
            "reason": (
                "positive route evidence observed within this window from: " + ", ".join(route_evidence_sources)
            ),
        }
    return {
        "window_route_status": ROUTE_INTEGRITY_UNVERIFIED,
        "reason": (
            "no confirmed harness-side failure in this window but also no positive delivered-request "
            "evidence within it -- component-level health is supporting evidence only, never sufficient "
            "proof of forwarding for this specific window"
        ),
    }


def classify_route_integrity(
    material_intervals: Sequence[Tuple[float, float]] = (),
    evidence_windows: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """CORR-39 resolved contract (Phase 15B.3M.2), PROOF-SUFFICIENCY
    hardened (Phase 15B.3M.3), TEMPORAL-COVERAGE hardened (Phase 15B.3M.4).

    Route integrity is `CONFIRMED` whenever there is no MATERIAL
    zero-success interval at all (`material_intervals` empty) -- ordinary
    application-level failures never by themselves put route integrity in
    question.

    When one or more material intervals exist, this function no longer
    reasons about "the interval" as a single evidentiary unit (Phase
    15B.3M.3's residual gap, closed here): each material interval is
    partitioned into `ROUTE_INTEGRITY_EVIDENCE_WINDOW_SECONDS`-wide
    evidence windows (`partition_into_evidence_windows()`), each window is
    independently classified (`classify_route_integrity_window()`), and
    `evidence_windows` is the FLATTENED list of every such window's result
    across every material interval. Attempt-level aggregation:

      - `FAILED` if ANY constituent window is `FAILED` -- a single
        confirmed harness-route failure anywhere is decisive and is never
        downgraded by other windows being healthy.
      - otherwise `CONFIRMED` ONLY if EVERY constituent window across
        EVERY material interval is `CONFIRMED` (full temporal coverage).
        One delivered request somewhere inside a long material interval no
        longer confirms the whole interval -- this is precisely the
        index12-shaped regression this phase closes: "CONFIRMED at t1 does
        not imply CONFIRMED at t2" for a route that may fail partway
        through and never recover.
      - otherwise `UNVERIFIED` -- no window is confirmed-failed, but at
        least one window lacks positive delivered-request evidence. Some
        positive evidence somewhere is never permitted to upgrade a later
        unverified window.

    Returns a dict (never a bare string/boolean): `classification`,
    `reason`, `material_intervals`, `evidence_windows` (echoed back
    verbatim so the classification is never divorced from the windowed
    evidence that produced it), `confirmed_window_count`,
    `unverified_window_count`, `failed_window_count`, and
    `full_temporal_coverage` (True iff every window is `CONFIRMED`).
    """
    if not material_intervals:
        return {
            "classification": ROUTE_INTEGRITY_CONFIRMED,
            "reason": "no material zero-success delivery interval occurred",
            "material_intervals": [],
            "evidence_windows": [],
            "confirmed_window_count": 0,
            "unverified_window_count": 0,
            "failed_window_count": 0,
            "full_temporal_coverage": True,
        }

    total = len(evidence_windows)
    confirmed_count = sum(1 for w in evidence_windows if w.get("window_route_status") == ROUTE_INTEGRITY_CONFIRMED)
    unverified_count = sum(1 for w in evidence_windows if w.get("window_route_status") == ROUTE_INTEGRITY_UNVERIFIED)
    failed_count = sum(1 for w in evidence_windows if w.get("window_route_status") == ROUTE_INTEGRITY_FAILED)
    full_temporal_coverage = total > 0 and confirmed_count == total

    if failed_count > 0:
        classification, reason = (
            ROUTE_INTEGRITY_FAILED,
            f"{failed_count} of {total} evidence window(s) across the material interval(s) show a confirmed "
            "harness-route failure (tunnel process or local listener) -- a single failed window is sufficient "
            "for attempt-level FAILED, and no other window's health downgrades or overrides it",
        )
    elif unverified_count > 0:
        classification, reason = (
            ROUTE_INTEGRITY_UNVERIFIED,
            f"{unverified_count} of {total} evidence window(s) across the material interval(s) have no confirmed "
            "harness-side failure but also no positive delivered-request evidence -- CONFIRMED requires full "
            "temporal coverage (every window CONFIRMED); positive delivery evidence in some other window never "
            "retroactively confirms these",
        )
    else:
        classification, reason = (
            ROUTE_INTEGRITY_CONFIRMED,
            f"all {total} evidence window(s) across every material interval show positive delivered-request "
            "evidence with no confirmed harness-route failure -- full temporal coverage",
        )

    return {
        "classification": classification,
        "reason": reason,
        "material_intervals": list(material_intervals),
        "evidence_windows": list(evidence_windows),
        "confirmed_window_count": confirmed_count,
        "unverified_window_count": unverified_count,
        "failed_window_count": failed_count,
        "full_temporal_coverage": full_temporal_coverage,
    }


# =============================================================================
# Phase 15B.3O.1 / CORR-41: Kubernetes-dataplane (kube-proxy NodePort/Service)
# workload-ingress evidence -- an evidence source INDEPENDENT of application
# availability, closing the methodology gap that campaign_sequence_index=12
# attempt3 exposed: CORR-39's only positive evidence sources (HTTP responses,
# per-pod application counters) both require the application itself to be
# alive and observable, so a genuine application/backend collapse is
# indistinguishable from a genuine harness/route failure -- both produce
# UNVERIFIED. This section adds a THIRD, endpoint-independent positive
# source read directly from the Minikube node's kube-proxy dataplane
# (iptables/nf_tables), so "traffic reached the Kubernetes Service/NodePort
# dataplane" can be proven even when the backend has zero Ready endpoints or
# never responds. It does NOT change the traffic path (still Protocol V5's
# Docker-network raw TCP relay -> NodePort) and does NOT weaken CORR-39 --
# it adds one more decisive-positive source alongside the existing two.
# =============================================================================

DATAPLANE_INGRESS_ENDPOINT_BACKED_COMMENT = "autoscaler-app/demo-app-loadtest:http"
DATAPLANE_INGRESS_NO_ENDPOINTS_COMMENT = "autoscaler-app/demo-app-loadtest:http has no endpoints"

DATAPLANE_REGIME_ENDPOINT_BACKED = "endpoint_backed"
DATAPLANE_REGIME_NO_ENDPOINTS = "no_endpoints"
DATAPLANE_REGIME_ABSENT = "absent"

# Phase 15B.3O.1 non-official validation (see
# persistent-data/transport_validation/protocol_v6_route_evidence/) measured
# ~0.05-0.14s per `docker exec minikube iptables-save -t nat -c` invocation.
# 3s is faster than the existing 10s route-integrity evidence window (so
# every window can contain a fully-interior sample pair) while keeping
# observer duty-cycle to a few percent of one core on the 2-core Minikube VM.
# FROZEN HISTORICAL VALUE -- this is exactly what attempts 1-4 (V6/CORR-41)
# used and must never be edited; Phase 15B.3P.1/CORR-42/Protocol V7 found
# that the REAL (not nominal) inter-sample gap this constant produces is
# materially larger (empirically ~3.3-4.3s, mean ~3.6s, on attempt4's own
# data -- `sleep(3.0)` plus real `docker exec` command latency plus thread
# scheduling), which a formal timing-geometry analysis showed can leave a
# full 10s route window with as few as 2 interior samples (not the 3+ this
# constant's own justification assumed) and can make ANY trailing window
# narrower than roughly 2x the real cadence structurally unobservable
# regardless of dataplane health -- see
# persistent-data/transport_validation/protocol_v7_temporal_coverage/ and
# docs/experimental_design_corrections.md CORR-42 for the full analysis.
DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS = 3.0

# Phase 15B.3P.1 (CORR-42, Protocol V7): the corrected, GOING-FORWARD
# periodic cadence -- a modest, quantitatively-justified reduction (not an
# arbitrary jump to 1s), evaluated as the fallback-only lever per the
# governing instructions' own preference ordering. It materially improves,
# but does not by itself guarantee, full interior-sample coverage; it is
# deliberately combined with the two REACTIVE mechanisms below rather than
# relied upon alone. Historical attempts (1-4, V6/CORR-41) are entirely
# unaffected -- this constant is never applied retroactively.
DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7 = 2.0

# Phase 15B.3P.1 (CORR-42, Protocol V7): after detecting that a new sample's
# `regime`/`rule_identity` differs from the immediately preceding sample
# (kube-proxy reprogrammed the Service -- a pod restart, an endpoint-count
# transition), wait this long and take ONE immediate reactive follow-up
# sample under the NEW identity, rather than waiting a full periodic cadence
# period. This directly targets the "reset landed inside a sparse window"
# blind-spot class (see CORR-42 Part C/D of the timing analysis): it cannot
# fabricate evidence for the pre-transition identity, but it maximizes the
# chance that a second, same-identity, fully-interior sample exists soon
# after the transition -- so a legitimate (never fabricated) confirming pair
# can still form within whatever window the transition happened to land in,
# if the new identity's own traffic continues.
DATAPLANE_INGRESS_TRANSITION_RESAMPLE_DELAY_SECONDS = 1.0


def should_trigger_immediate_resample(
    previous_sample: Optional[Dict[str, Any]],
    current_sample: Dict[str, Any],
) -> bool:
    """Phase 15B.3P.1 (CORR-42): pure decision function -- True exactly when
    `current_sample`'s `regime` or `rule_identity` differs from
    `previous_sample`'s (a detected kube-proxy reprogramming event), or when
    there is no previous sample to compare against (first-ever sample is
    never a "transition" -- returns False). An `observation_error` on either
    sample is treated as insufficient information to declare a transition
    (returns False) -- this function only ever requests MORE evidence, so
    erring toward "no transition detected" here costs nothing beyond one
    fewer reactive sample; it never suppresses evidence collection.
    """
    if previous_sample is None:
        return False
    if previous_sample.get("observation_error") or current_sample.get("observation_error"):
        return False
    if previous_sample.get("regime") != current_sample.get("regime"):
        return True
    return previous_sample.get("rule_identity") != current_sample.get("rule_identity")


def summarize_dataplane_acquisition_gaps(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Phase 15B.3P.3: pure summary statistics over a dataplane sample
    stream's `actual_gap_seconds_from_previous_sample` field (recorded by
    `run_dynamics_calibration_slot.py`'s `_take_and_record_dataplane_sample()`).
    This is observability only -- it never feeds into
    `classify_route_integrity_window()`/`classify_route_integrity()`, and
    it deliberately distinguishes an OBSERVED maximum (a fact about this
    specific finite run) from any claim of a GUARANTEED maximum (which
    would require a hard, OS/Docker-enforced real-time bound this project
    does not claim to have -- see CORR-43/V8's amended rationale,
    docs/experimental_design_corrections.md). Returns `n=0` and all other
    fields `None` if fewer than one gap is available (e.g. a single-sample
    stream, or all samples marked `observation_error`)."""
    gaps = sorted(
        s["actual_gap_seconds_from_previous_sample"]
        for s in samples
        if s.get("actual_gap_seconds_from_previous_sample") is not None
    )
    n = len(gaps)
    if n == 0:
        return {"n": 0, "min": None, "median": None, "mean": None, "p95": None, "p99": None, "max": None}

    def _percentile(sorted_values: List[float], pct: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        rank = pct * (len(sorted_values) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(sorted_values) - 1)
        frac = rank - lo
        return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac

    return {
        "n": n,
        "min": gaps[0],
        "median": _percentile(gaps, 0.5),
        "mean": sum(gaps) / n,
        "p95": _percentile(gaps, 0.95),
        "p99": _percentile(gaps, 0.99),
        "max": gaps[-1],
        "observed_maximum_is_not_a_guaranteed_maximum": True,
    }


_IPTABLES_SAVE_RULE_RE = re.compile(r"^\[(\d+):(\d+)\]\s+-A\s+(\S+)\s+(.*)$")


def parse_dataplane_ingress_rules(
    nat_table_iptables_save_text: str,
    filter_table_iptables_save_text: str,
    endpoint_backed_comment: str = DATAPLANE_INGRESS_ENDPOINT_BACKED_COMMENT,
    no_endpoints_comment: str = DATAPLANE_INGRESS_NO_ENDPOINTS_COMMENT,
) -> Dict[str, Any]:
    """Phase 15B.3O.1/CORR-41: pure parse of two `iptables-save -c` dumps
    (nat table, filter table -- exactly as produced by `docker exec minikube
    iptables-save -t nat -c` / `-t filter -c`) into the Service-specific
    dataplane-ingress evidence for `demo-app-loadtest`.

    Two mutually-exclusive, kube-proxy-programmed regimes were empirically
    confirmed live on this project's Minikube/kube-proxy-iptables(nf_tables)
    environment (see the NON_OFFICIAL_ROUTE_EVIDENCE_VALIDATION artifact):

      - `endpoint_backed`: one or more `nat` table rules of the form
        `-A KUBE-SEP-<hash> ... -j DNAT --to-destination <podIP>:<port>`
        carrying EXACTLY `endpoint_backed_comment` (never the "has no
        endpoints" variant). kube-proxy creates ONE such chain PER Ready
        endpoint, sharing the same comment -- ALL matching rules are summed
        (empirically verified: 2 endpoints, 10 requests, 7+3=10). This
        regime only exists while at least one Ready endpoint exists.
      - `no_endpoints`: a single `filter` table rule
        `-A KUBE-EXTERNAL-SERVICES ... --dport <nodeport> -j REJECT
        --reject-with icmp-port-unreachable` carrying EXACTLY
        `no_endpoints_comment`, programmed by kube-proxy ONLY while the
        Service has zero Ready endpoints. This is the evidence source that
        answers the phase's central question when the backend is down.

    If NEITHER regime's rule is found (chain renamed, comment text changed
    upstream, or a transient kube-proxy resync mid-sample), `regime` is
    `DATAPLANE_REGIME_ABSENT` -- an OBSERVATION gap, never asserted as
    negative route evidence (see `compute_dataplane_ingress_evidence_windows`
    and sec.24 "observer failure is not route failure").

    `rule_identity` is the sorted tuple of exact chain names backing the
    current regime -- NOT stable across pod recreation (empirically
    confirmed: a scale-to-0/scale-to-1 cycle changed
    `KUBE-SEP-ENIF66D2G3466LGQ` to a brand-new `KUBE-SEP-OGXRNUUKR756PIKN`
    for the same Service). Callers MUST treat any change in `rule_identity`
    (or in `regime`) between two samples as a baseline reset, never as a
    negative delta -- see `compute_dataplane_ingress_evidence_windows`.
    """
    endpoint_backed_rules: List[Dict[str, Any]] = []
    for line in nat_table_iptables_save_text.splitlines():
        m = _IPTABLES_SAVE_RULE_RE.match(line.strip())
        if not m:
            continue
        packets, bytes_, chain, rest = m.groups()
        if f'--comment "{endpoint_backed_comment}"' not in rest:
            continue
        if "-j DNAT" not in rest:
            continue  # exclude the sibling KUBE-MARK-MASQ rule sharing the same comment
        endpoint_backed_rules.append(
            {"table": "nat", "chain": chain, "packet_count": int(packets), "byte_count": int(bytes_), "raw_line": line.strip()}
        )

    no_endpoints_rule: Optional[Dict[str, Any]] = None
    for line in filter_table_iptables_save_text.splitlines():
        m = _IPTABLES_SAVE_RULE_RE.match(line.strip())
        if not m:
            continue
        packets, bytes_, chain, rest = m.groups()
        if f'--comment "{no_endpoints_comment}"' not in rest:
            continue
        if "-j REJECT" not in rest:
            continue
        no_endpoints_rule = {
            "table": "filter", "chain": chain, "packet_count": int(packets), "byte_count": int(bytes_), "raw_line": line.strip(),
        }
        break  # exactly one such rule is ever programmed per Service

    if endpoint_backed_rules:
        regime = DATAPLANE_REGIME_ENDPOINT_BACKED
        rules = endpoint_backed_rules
    elif no_endpoints_rule is not None:
        regime = DATAPLANE_REGIME_NO_ENDPOINTS
        rules = [no_endpoints_rule]
    else:
        regime = DATAPLANE_REGIME_ABSENT
        rules = []

    return {
        "regime": regime,
        "rules": rules,
        "rule_identity": tuple(sorted(r["chain"] for r in rules)),
        "total_packet_count": sum(r["packet_count"] for r in rules) if rules else None,
        "total_byte_count": sum(r["byte_count"] for r in rules) if rules else None,
    }


def compute_dataplane_ingress_evidence_windows(
    window_start_elapsed: float,
    window_end_elapsed: float,
    traffic_start_epoch: Optional[float] = None,
    dataplane_samples: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """Phase 15B.3O.1/CORR-41: the windowed evidence computation for the
    Kubernetes-dataplane ingress source, mirroring
    `compute_delivered_request_evidence_windows()`'s window-local,
    reset-safe, non-duplicating attribution discipline.

    Each element of `dataplane_samples` is one `parse_dataplane_ingress_rules()`
    result plus `elapsed_seconds_since_traffic_start` (float) and optionally
    `observation_error` (truthy means the sampler itself failed this sample
    -- e.g. a `docker exec` timeout -- which is NEVER treated as route
    FAILED, only as reduced evidence availability, sec.24).

    Consecutive samples (sorted by elapsed time) form a candidate evidence
    interval `(t1, t2]`, credited to this window only when ALL hold:
      - neither sample has `observation_error` set;
      - both samples' `regime` AND `rule_identity` are IDENTICAL (an
        identity/regime change means kube-proxy reprogrammed the Service
        between samples -- e.g. a pod restart or an endpoint-count
        transition -- so the counters are not comparable across it; this is
        rejected as a RESET, never as a negative delta, and a fresh
        baseline begins at the later sample);
      - `total_packet_count` did not decrease (a decrease with unchanged
        identity would itself indicate an un-signalled counter reset --
        also rejected as a RESET, defensively, even though no such case was
        observed live);
      - the ENTIRE interval `(t1, t2]` is inside this window's bounds (a
        sample straddling the window boundary is rejected as
        CROSS_WINDOW and credited to neither window, exactly like the
        existing per-pod application-counter rule).

    A strictly positive packet-count delta on an accepted interval is
    decisive dataplane-ingress evidence for this window -- regardless of
    which regime it occurred in. `dataplane_backend_regimes_with_evidence`
    records which regime(s) produced that evidence, so a window CONFIRMED
    purely via a `no_endpoints`-regime interval is distinguishable in the
    persisted timeline from one confirmed via `endpoint_backed` traffic --
    this is the endpoint/context separation sec.17 requires: route health
    and backend availability remain two different questions even though
    both are visible in the same artifact.
    """
    samples = sorted(dataplane_samples, key=lambda s: float(s.get("elapsed_seconds_since_traffic_start", 0.0)))
    evidence = False
    intervals_used: List[Dict[str, Any]] = []
    intervals_rejected_cross_window: List[Dict[str, Any]] = []
    intervals_rejected_reset: List[Dict[str, Any]] = []
    observer_error_present = False
    regimes_with_evidence: List[str] = []

    epoch_start = (traffic_start_epoch + window_start_elapsed) if traffic_start_epoch is not None else None

    for s1, s2 in zip(samples, samples[1:]):
        if s1.get("observation_error") or s2.get("observation_error"):
            observer_error_present = True
            continue
        e1 = float(s1["elapsed_seconds_since_traffic_start"])
        e2 = float(s2["elapsed_seconds_since_traffic_start"])
        if e2 <= window_start_elapsed or e1 >= window_end_elapsed:
            continue  # this interval does not touch this window at all

        record = {
            "t1_elapsed": e1, "t2_elapsed": e2,
            "regime_t1": s1.get("regime"), "regime_t2": s2.get("regime"),
            "rule_identity_t1": s1.get("rule_identity"), "rule_identity_t2": s2.get("rule_identity"),
            "packet_count_t1": s1.get("total_packet_count"), "packet_count_t2": s2.get("total_packet_count"),
        }
        if epoch_start is not None:
            record["t1_epoch"] = epoch_start - window_start_elapsed + e1
            record["t2_epoch"] = epoch_start - window_start_elapsed + e2

        fully_inside_window = window_start_elapsed <= e1 < window_end_elapsed and window_start_elapsed <= e2 < window_end_elapsed
        if not fully_inside_window:
            intervals_rejected_cross_window.append(record)
            continue

        identity_changed = s1.get("regime") != s2.get("regime") or s1.get("rule_identity") != s2.get("rule_identity")
        p1, p2 = s1.get("total_packet_count"), s2.get("total_packet_count")
        counter_decreased = p1 is not None and p2 is not None and p2 < p1
        if identity_changed or counter_decreased:
            intervals_rejected_reset.append(record)
            continue

        intervals_used.append(record)
        if p1 is not None and p2 is not None and p2 > p1:
            evidence = True
            regime_here = s2.get("regime")
            if regime_here and regime_here not in regimes_with_evidence:
                regimes_with_evidence.append(regime_here)

    return {
        "window_start_elapsed": window_start_elapsed,
        "window_end_elapsed": window_end_elapsed,
        "nodeport_dataplane_ingress_evidence": evidence,
        "dataplane_counter_intervals_used": intervals_used,
        "dataplane_counter_intervals_rejected_cross_window": intervals_rejected_cross_window,
        "dataplane_counter_intervals_rejected_reset": intervals_rejected_reset,
        "dataplane_observer_error_present": observer_error_present,
        "dataplane_backend_regimes_with_evidence": regimes_with_evidence,
    }


def parse_endpoints_ready_count(endpoints_json: Dict[str, Any]) -> Dict[str, Any]:
    """Pure parse of a `kubectl get endpoints <svc> -o json` (core v1
    Endpoints) object into ready/not-ready address counts and the
    associated pod-reference names, where present. Never invents a ready
    address that is not present in `subsets[].addresses`."""
    subsets = endpoints_json.get("subsets") or []
    ready_addresses: List[str] = []
    not_ready_addresses: List[str] = []
    for subset in subsets:
        for addr in subset.get("addresses", []) or []:
            target_ref = addr.get("targetRef", {}) or {}
            ready_addresses.append(target_ref.get("name") or addr.get("ip", ""))
        for addr in subset.get("notReadyAddresses", []) or []:
            target_ref = addr.get("targetRef", {}) or {}
            not_ready_addresses.append(target_ref.get("name") or addr.get("ip", ""))
    return {
        "ready_count": len(ready_addresses),
        "not_ready_count": len(not_ready_addresses),
        "ready_pod_names": ready_addresses,
        "not_ready_pod_names": not_ready_addresses,
    }


def compute_delivered_request_evidence(
    interval_start_epoch: float,
    interval_end_epoch: float,
    metrics_rows: Sequence[Dict[str, Any]] = (),
    counter_rows: Sequence[Dict[str, Any]] = (),
) -> Optional[bool]:
    """Phase 15B.3M.3: the single STRONGEST evidence a material
    zero-client-success interval's route integrity can be `CONFIRMED` on --
    positive proof that at least one request actually crossed the harness
    boundary into the intended Kubernetes/application path during
    `[interval_start_epoch, interval_end_epoch]` (wall-clock, so it can be
    checked against metrics/counter rows sourced from different
    collector-process clocks). Two independent sources, either sufficient:

      - `metrics_rows`: any row with `timestamp_epoch` inside the window
        and a positive, non-empty `request_rate_rps` (the server-observed
        Prometheus rate -- only non-zero if the app actually counted
        requests; 2xx is never required, this reflects total request rate).
      - `counter_rows`: any pod whose raw per-pod counter value increased
        between two samples both inside the window (a coarse, reset-naive
        check deliberately -- a false negative here from an unhandled
        counter reset only means this function correctly declines to
        assert delivery evidence it cannot support, never the reverse).

    Returns `True` (positive evidence found), `False` (rows exist inside
    the window from at least one source but show no delivery), or `None`
    (neither source has ANY sample inside the window at all -- evidence is
    genuinely absent, never fabricated as a negative finding).

    Phase 15B.3M.5 field-name fix: `counter_rows` (loaded from
    `per_pod_request_counters.csv` by `scripts/per_pod_counter_collector.py`,
    `CSV_FIELDNAMES = [..., "pod", "raw_counter_value"]`) keys the pod
    identity as `pod`, not `pod_name` -- this function originally read
    `row.get("pod_name")`, which is never present in the real CSV, so the
    counter-increment source silently never fired in practice (every row
    was skipped by the `pod is None` guard below). This is a pre-first-use
    implementation-correctness fix: this function has never governed a
    live classification (superseded by the windowed sibling below before
    any live attempt used it), so it is corrected in place."""
    any_row_in_window = False
    for row in metrics_rows:
        epoch = float(row.get("timestamp_epoch", "nan") or "nan")
        if not (interval_start_epoch <= epoch <= interval_end_epoch):
            continue
        any_row_in_window = True
        rps_raw = row.get("request_rate_rps")
        if rps_raw not in (None, ""):
            try:
                if float(rps_raw) > 0:
                    return True
            except (TypeError, ValueError):
                pass

    samples_by_pod: Dict[str, List[Tuple[float, float]]] = {}
    for row in counter_rows:
        epoch = float(row.get("timestamp_epoch", "nan") or "nan")
        if not (interval_start_epoch <= epoch <= interval_end_epoch):
            continue
        any_row_in_window = True
        pod = row.get("pod")
        value = row.get("raw_counter_value")
        if pod is None or value is None:
            continue
        samples_by_pod.setdefault(pod, []).append((epoch, float(value)))
    for samples in samples_by_pod.values():
        samples.sort(key=lambda s: s[0])
        for (_, v1), (_, v2) in zip(samples, samples[1:]):
            if v2 > v1:
                return True

    return False if any_row_in_window else None


def compute_delivered_request_evidence_windows(
    window_start_elapsed: float,
    window_end_elapsed: float,
    traffic_start_epoch: Optional[float] = None,
    requests_rows: Sequence[Dict[str, Any]] = (),
    metrics_rows: Sequence[Dict[str, Any]] = (),
    counter_rows: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """Phase 15B.3M.4: the WINDOWED sibling of `compute_delivered_request_evidence()`.
    Phase 15B.3M.5 (window-local delivery-attribution hardening): closed a
    residual gap where an OBSERVATION timestamp inside a window was being
    treated as an EVENT-ATTRIBUTION timestamp for that same window, which
    is not always true. Decisive `delivered_request_evidence` now comes
    from exactly TWO window-local sources:

      - **HTTP response, attributed by DISPATCH time** (`requests_rows`,
        `requests.csv`): a request whose `actual_dispatch_elapsed_seconds`
        -- the moment the generator actually initiated the request attempt
        (`time.monotonic()`, captured as the first statement inside
        `loadgen/traffic_generator.py:_dispatch_single_request()`, before
        URL construction or the `httpx` call itself) -- falls inside
        `[window_start_elapsed, window_end_elapsed)`, AND that request
        eventually produced a genuine HTTP response outcome
        (`success_2xx`/`client_error_4xx`/`server_error_5xx`/`http_*`).
        `completion_elapsed_seconds` is recorded for provenance only and
        is NEVER consulted for window membership: a request dispatched
        in a PRIOR window whose response merely happens to complete inside
        THIS window must not be credited to this window (attribution
        belongs to the window in which the route-entry attempt began, not
        the window in which its response happened to arrive). `timeout`
        and `connection_error` remain non-proof.
      - **Raw per-pod counter increment** (`counter_rows`,
        `per_pod_request_counters.csv`): a positive delta between two
        consecutive samples for the SAME pod whose entire observation
        interval `(t1, t2]` is fully contained in this window's epoch
        bounds. An interval that only partially overlaps this window (one
        bounding sample inside, one outside) is recorded as a REJECTED
        cross-window interval for this window and is never credited here
        -- conservative and deliberate, so a single increment is never
        duplicated or smeared across two windows; if it cannot be safely
        localized to one window, it counts as evidence for none.

    `metrics_rows` (server-side `metrics.csv`, `request_rate_rps` ==
    `sum(rate(demo_app_requests_total{...}[1m]))` -- a 60-SECOND ROLLING
    lookback, confirmed by direct read of `collectors/metrics_collector.py`)
    is recorded as `rolling_server_rps_observed` for SCIENTIFIC CONTEXT
    ONLY. A positive sample timestamped inside this 10-second window
    reflects up to 60 seconds of PRIOR request history -- it is NOT proof
    that a request specifically entered the route during this window, so
    it is NEVER included in `delivered_request_evidence`
    (`rolling_server_rps_supporting_only` is always `True`, so this
    non-decisiveness is explicit and auditable in the persisted timeline
    rather than silently dropped).

    Always returns a concrete dict (never `None`): an evidence window is a
    synthetic partition of an already-material interval, not a separate
    observation attempt, so the absence of any positive signal in a window
    is `delivered_request_evidence=False`, not "unknown"."""
    http_response_delivery_evidence = False
    http_response_request_indices: List[Any] = []
    http_response_dispatch_times: List[float] = []
    http_response_completion_times: List[Optional[float]] = []
    for row in requests_rows:
        try:
            dispatch = float(row.get("actual_dispatch_elapsed_seconds", "nan"))
        except (TypeError, ValueError):
            continue
        if not (window_start_elapsed <= dispatch < window_end_elapsed):
            continue
        outcome = str(row.get("outcome") or "")
        if outcome in ("success_2xx", "client_error_4xx", "server_error_5xx") or outcome.startswith("http_"):
            http_response_delivery_evidence = True
            http_response_request_indices.append(row.get("request_index"))
            http_response_dispatch_times.append(dispatch)
            completion_raw = row.get("completion_elapsed_seconds")
            try:
                http_response_completion_times.append(
                    float(completion_raw) if completion_raw not in (None, "") else None
                )
            except (TypeError, ValueError):
                http_response_completion_times.append(None)

    rolling_server_rps_observed = False
    counter_delivery_evidence = False
    counter_intervals_used: List[Dict[str, Any]] = []
    counter_intervals_rejected_as_cross_window: List[Dict[str, Any]] = []
    if traffic_start_epoch is not None:
        epoch_start = traffic_start_epoch + window_start_elapsed
        epoch_end = traffic_start_epoch + window_end_elapsed

        for row in metrics_rows:
            epoch = float(row.get("timestamp_epoch", "nan") or "nan")
            if not (epoch_start <= epoch < epoch_end):
                continue
            rps_raw = row.get("request_rate_rps")
            if rps_raw not in (None, ""):
                try:
                    if float(rps_raw) > 0:
                        rolling_server_rps_observed = True
                        break
                except (TypeError, ValueError):
                    pass

        samples_by_pod: Dict[str, List[Tuple[float, float]]] = {}
        for row in counter_rows:
            epoch = float(row.get("timestamp_epoch", "nan") or "nan")
            pod = row.get("pod")
            value = row.get("raw_counter_value")
            if pod is None or value is None:
                continue
            samples_by_pod.setdefault(pod, []).append((epoch, float(value)))
        for pod, samples in samples_by_pod.items():
            samples.sort(key=lambda s: s[0])
            for (e1, v1), (e2, v2) in zip(samples, samples[1:]):
                if e2 < epoch_start or e1 >= epoch_end:
                    continue  # this interval does not touch this window at all
                interval_record = {"pod": pod, "t1_epoch": e1, "t2_epoch": e2, "v1": v1, "v2": v2}
                if epoch_start <= e1 < epoch_end and epoch_start <= e2 < epoch_end:
                    counter_intervals_used.append(interval_record)
                    if v2 > v1:
                        counter_delivery_evidence = True
                else:
                    counter_intervals_rejected_as_cross_window.append(interval_record)

    delivery_evidence_sources: List[str] = []
    if http_response_delivery_evidence:
        delivery_evidence_sources.append("http_response")
    if counter_delivery_evidence:
        delivery_evidence_sources.append("counter_increment")

    return {
        "window_start_elapsed": window_start_elapsed,
        "window_end_elapsed": window_end_elapsed,
        "http_response_delivery_evidence": http_response_delivery_evidence,
        "http_response_request_indices": http_response_request_indices,
        "http_response_dispatch_times": http_response_dispatch_times,
        "http_response_completion_times": http_response_completion_times,
        "counter_delivery_evidence": counter_delivery_evidence,
        "counter_intervals_used": counter_intervals_used,
        "counter_intervals_rejected_as_cross_window": counter_intervals_rejected_as_cross_window,
        "rolling_server_rps_observed": rolling_server_rps_observed,
        "rolling_server_rps_supporting_only": True,
        "delivered_request_evidence": http_response_delivery_evidence or counter_delivery_evidence,
        "delivery_evidence_sources": delivery_evidence_sources,
    }


def aggregate_observer_evidence_for_window(
    window_start_elapsed: float,
    window_end_elapsed: float,
    periodic_samples: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """Phase 15B.3M.4: maps the harness's already-captured ~10s periodic
    tunnel/listener/endpoint observer samples (each keyed by its own
    `elapsed_seconds_since_traffic_start`) into whichever SINGLE evidence
    window contains that sample's timestamp -- never smeared across the
    whole material interval the way Phase 15B.3M.3's `_throughout` fields
    were.

    A window with NO observer sample inside it reports `None`/unknown for
    every health field -- genuinely absent evidence is never fabricated as
    healthy. A window with one or more samples reports `True` only if
    every sample within it agrees the component was healthy, `False` if
    ANY sample within it reports the component unhealthy/absent (a single
    bad sample is decisive), and `None` if samples exist but never
    actually reported that field.
    """
    samples_in_window = [
        s
        for s in periodic_samples
        if window_start_elapsed <= float(s.get("elapsed_seconds_since_traffic_start", -1.0)) < window_end_elapsed
    ]
    if not samples_in_window:
        return {
            "sample_count": 0,
            "tunnel_process_alive": None,
            "local_listener_present": None,
            "loadtest_ready_endpoint_count": None,
            "endpoint_query_error": None,
        }

    def _aggregate_bool_field(field: str) -> Optional[bool]:
        values = [s.get(field) for s in samples_in_window if s.get(field) is not None]
        if not values:
            return None
        if any(v is False for v in values):
            return False
        return all(v is True for v in values)

    ready_counts: List[int] = []
    query_errors: List[str] = []
    for s in samples_in_window:
        endpoints = s.get("loadtest_endpoints")
        if endpoints is None:
            err = s.get("endpoint_query_error")
            if err:
                query_errors.append(str(err))
            continue
        ready_counts.append(parse_endpoints_ready_count(endpoints).get("ready_count", 0))

    return {
        "sample_count": len(samples_in_window),
        "tunnel_process_alive": _aggregate_bool_field("tunnel_process_alive"),
        "local_listener_present": _aggregate_bool_field("local_listener_present"),
        "loadtest_ready_endpoint_count": (min(ready_counts) if ready_counts else None),
        "endpoint_query_error": ("; ".join(query_errors) if query_errors else None),
    }


# ---------------------------------------------------------------------------
# Phase 15B.3C.1: HPA restoration-wait timeout derived from the FROZEN HPA
# scale-down mechanics (k8s/hpa/hpa.yaml behavior.scaleDown, CORR-26) -- not
# an arbitrary convenience value. Fixes a real Phase 15B.3C finding: a fixed
# 180s wait was insufficient for a 4-replica-origin convergence.
# ---------------------------------------------------------------------------
HPA_SCALE_DOWN_STABILIZATION_SECONDS = 120  # k8s/hpa/hpa.yaml: behavior.scaleDown.stabilizationWindowSeconds
HPA_SCALE_DOWN_STEP_INTERVAL_SECONDS = 60  # k8s/hpa/hpa.yaml: behavior.scaleDown.policies[0].periodSeconds (removes <=1 pod per interval)
HPA_OBSERVER_SYNC_SAFETY_MARGIN_SECONDS = 30  # ~2x kube-controller-manager's default --horizontal-pod-autoscaler-sync-period (15s), plus polling/propagation overhead


# ---------------------------------------------------------------------------
# Frozen artifact loaders
# ---------------------------------------------------------------------------


def load_manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schedule_registry() -> dict:
    with open(SCHEDULE_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_global_sequence() -> List[dict]:
    with open(GLOBAL_SEQUENCE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_of(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def build_protocol_provenance() -> Dict[str, Any]:
    """Phase 15B.3E (CORR-38 provenance hardening), extended Phase 15B.3M.2
    (CORR-39), extended Phase 15B.3N.3 (CORR-40, Protocol V5 transport
    transition), extended Phase 15B.3O.1 (CORR-41, Protocol V6
    route-integrity evidence hardening), extended Phase 15B.3P.1 (CORR-42,
    Protocol V7 temporal dataplane evidence ACQUISITION hardening),
    extended Phase 15B.3P.2 (CORR-43, Protocol V8 deterministic
    short-trailing-window evidence-WINDOW-PARTITIONING hardening), extended
    Phase 15B.5C (CORR-44, replica-transition route-materiality/scored-
    window OPERATIONALIZATION), extended Phase 15B.5E (CORR-45, replica-
    transition scale-down lag TARGET-STATE OPERATIONALIZATION): every NEW
    attempt must unambiguously record which protocol layer governs it.

    V8/CORR-43 changes ONLY the evidence-WINDOW-PARTITIONING layer
    (`partition_into_evidence_windows()`'s optional `min_window_seconds`
    merge behavior, driven by `MIN_EVIDENCE_WINDOW_SECONDS_V8`) -- it does
    NOT touch `classify_route_integrity_window()`, `classify_route_
    integrity()`, or `compute_dataplane_ingress_evidence_windows()` (the
    CORR-41 evidence-CLASSIFICATION semantics, `route_integrity_protocol_
    version` stays 6), and it does NOT touch the CORR-42 dataplane-evidence
    ACQUISITION schedule itself (`dataplane_acquisition_protocol_version`
    stays 2 -- the sampling cadence, reactive-resample, and anchor-sample
    mechanisms are entirely unchanged). A new, independent `window_
    partitioning_protocol_version` field (1 -> 2) tracks this
    partitioning-only change so it is never conflated with either a
    classification change or an acquisition change. Nor does it touch the
    transport mechanism frozen by CORR-40 (`transport_protocol_version`
    stays 5), nor the pre-existing `base_matrix_protocol_version` (still
    1), `execution_validity_protocol_version` (still 2, CORR-37), or
    `routing_relevance_protocol_version` (still 3, CORR-38). Applies
    PROSPECTIVELY ONLY: campaign_sequence_index=12 attempts 1-4 (all
    pre-CORR-43) are never reclassified or re-partitioned by this change --
    their stored `metadata.json` and already-persisted `evidence_windows`
    retain whatever provenance and window boundaries were live at the time
    they executed (attempt3: v5/CORR-40; attempt4: v6/CORR-41).

    CORR-44/Phase 15B.5C changes NEITHER the classification rules
    (`route_integrity_protocol_version` stays 6 -- `classify_route_
    integrity_window()`/`classify_route_integrity()`/`is_interval_
    material()` are all byte-for-byte unchanged) NOR any steady-state
    parameter (`[60,180)`/45s-of-180s is completely untouched for
    `fitting`/`interpolation_validation` slots) -- it only specifies, for
    the first time, WHICH already-generic `is_interval_material()`
    parameters (`scored_window`, `materiality_fraction_threshold`) apply to
    a `run_kind=replica_transition` run, which had never been given a
    value for either before Phase 15B.5B's own ad hoc, explicitly-
    disclosed-as-unfrozen implementation choice. No new protocol_version
    field is introduced for this -- see `replica_transition_route_scored_
    window()`/`REPLICA_TRANSITION_MATERIALITY_FRACTION`'s own docstrings.

    CORR-45/Phase 15B.5E fixes a false-positive in
    `compute_transition_replica_lags()`'s scale-down `ready`/`available`
    lag computation (an unrelated readiness/availability loss on a pod
    that had not yet been removed could previously satisfy `<= target`
    independently of whether `current_replicas` had itself reached the
    target -- see that function's own docstring for the full account) and
    adds trajectory-gap/lag-uncertainty measurement metadata. It changes
    NO protocol_version field: no classification/acquisition/partitioning/
    transport mechanism changed, and `replica_transition` had governed
    exactly one official attempt (index30, scale-up only) before this fix
    -- a pre-first-use hardening of the scale-down path specifically,
    exactly analogous to CORR-39 sec.8/9/10's own precedent. Index30
    itself is NOT reclassified or recomputed by this correction -- it
    already used the scale-up path, which this fix does not change the
    behavior of (see the governing Phase 15B.5E report's own regression
    tests).

    CORR-46/Phase 15B.5H (index33 attempt1's structural-INVALID
    classification exposed a real execution-harness robustness gap, not a
    scientific/methodology one): a single transient `kubectl`/`docker`
    `subprocess.TimeoutExpired` inside the pod-readiness collector
    subprocess, the replica-transition trajectory sampler thread, or the
    route-integrity/tunnel sampler thread previously either crashed the
    collector subprocess outright (non-zero exit, tripping the existing
    `collector_or_generator_structural_failure` gate) or silently killed a
    background sampler thread for the rest of the attempt (no further
    samples, no visible error). Isolated sampling timeouts are now caught
    at their point of origin and persisted as an explicit recoverable
    observation error (never a fabricated value), and the affected
    collector/sampler continues to its next scheduled iteration. This
    changes NO scientific/methodology field: `route_integrity_protocol_
    version` stays 6, `dataplane_acquisition_protocol_version` stays 2,
    `window_partitioning_protocol_version` stays 2, `transport_protocol_
    version` stays 5 -- no classification rule, materiality window,
    lag-target-state definition, or evidence-window-partitioning behavior
    changed; only observer/collector error-handling robustness changed, so
    no protocol_version field is bumped (exactly analogous to how CORR-44/
    CORR-45 themselves introduced no protocol_version bump). The
    pre-existing `collector_or_generator_structural_failure` structural-
    validity gate is explicitly UNCHANGED and still fires on a genuine
    (unrecovered) collector/generator failure -- this correction narrows
    what counts as "genuine failure" to exclude a single recoverable
    sampling-command timeout, it does not weaken or remove the gate
    itself. Index32 remains VALID, unchanged. Index33 attempt1 remains
    INVALID, unchanged and immutable -- this correction is prospective
    only; a future index33 attempt2 (if separately authorized) would be
    the first attempt to run under this hardening.

    CORR-47/Phase 15B.6B (pre-first-use workload-transition
    operationalization -- Phase 15B.6A found `run_kind=workload_
    transition` had no execution branch at all, before any official
    attempt of that kind ever ran): freezes, for the first time, the same
    class of previously-unspecified details CORR-44/45 already froze for
    `replica_transition` -- `scientific_role=WORKLOAD_TRANSITION_
    TRAJECTORY_VALIDATION` (explicitly excluded from steady-state fitting/
    interpolation-validation dataset construction), a workload-transition-
    specific materiality rule (`workload_transition_route_scored_window()`
    /`WORKLOAD_TRANSITION_MATERIALITY_FRACTION`, reusing the SAME
    duration-relative 0.25 fraction already established by CORR-39/CORR-44
    -- confirmed by source review that `is_interval_material()` was always
    generic, never replica-count-specific), and a dedicated `execute_
    workload_transition_slot()` execution path reusing the identical V5
    transport/collector/CORR-46-hardened infrastructure already proven by
    `replica_transition`. Unlike `replica_transitions`, the manifest names
    NO `measured_fields` for `workload_transitions` at all -- rather than
    inventing metric names or settling thresholds, this correction
    persists a single honest `WORKLOAD_TRANSITION_DERIVED_METRICS_STATUS =
    NOT_FORMALLY_OPERATIONALIZED` marker and preserves the full raw
    trajectory (target/realized RPS, latency, requests-in-progress, CPU,
    memory, fixed-replica state) for future analysis. Changes NO
    scientific field of `replica_transition`, `fitting`, or
    `interpolation_validation`: `route_integrity_protocol_version` stays
    6, `dataplane_acquisition_protocol_version` stays 2, `window_
    partitioning_protocol_version` stays 2, `transport_protocol_version`
    stays 5 -- this correction only extends already-generic, already-
    proven mechanisms to a run kind that had never yet used them, so no
    protocol_version field is bumped (exactly analogous to CORR-44/45/46's
    own precedent). No historical official attempt is reclassified; the
    global campaign sequence, its schedules, and their hashes are
    completely unchanged.

    CORR-48/Phase 15B.6D (index38's own first-official-use audit found a
    real implementation defect, not a scientific one): `build_workload_
    transition_trajectory_rows()` (CORR-47) had no upper-bound filter on
    `elapsed_seconds`, so the metrics collector's own deliberate post-
    traffic observation buffer (`total_traffic_seconds + 30` seconds --
    the same cleanup-robustness margin already used by `replica_
    transition`) was silently admitted into the derived SCIENTIFIC
    trajectory artifact and mislabeled `post_step` (index38's own 27-row
    file included 3 rows at elapsed 240.0/250.0/260.01, all at or past the
    official 240s traffic boundary). The function now requires a
    `total_traffic_seconds` parameter and excludes any row failing
    `0.0 <= elapsed_seconds < total_traffic_seconds` -- the same half-open
    interval CORR-47 already uses for its own route-materiality scored
    window, so the two now agree exactly. Raw `metrics.csv` is never
    altered; only the derived, non-authoritative diagnostic trajectory is
    now boundary-restricted. Direct code-path tracing confirmed this
    buggy row set was NEVER consulted by generator totals, latency
    statistics, success rate, zero-success-interval detection, CORR-47
    materiality, Protocol V8, route classification, `execution_
    structural_validity`, or SLA -- all independently derived from
    `traffic_summary.csv`/`requests.csv`/`pod_readiness_history.csv`/
    `loadtest_route_integrity_timeline.json`. The one consumer of the
    buggy row set, `fixed_replica_invariant`, was independently
    re-evaluated over the corrected row set for index38 and produced a
    byte-identical result -- no retroactive correction to index38's own
    already-persisted metadata was needed or made. Index38 remains
    permanently `VALID`, its own official `workload_transition_
    timeline.csv` (27 rows) is unchanged and immutable; a separate,
    clearly-labeled read-only reanalysis (24 rows) was produced for
    reference only. This correction is prospective only.
    """
    return {
        "active_calibration_protocol_version": 8,
        "active_calibration_manifest": "experiments/manifests/rl_dynamics_calibration_v8.yaml",
        "base_matrix_protocol_version": 1,
        "execution_validity_protocol_version": 2,
        "routing_relevance_protocol_version": 3,
        "route_integrity_protocol_version": 6,
        "transport_protocol_version": 5,
        "dataplane_acquisition_protocol_version": 2,
        "window_partitioning_protocol_version": 2,
        "active_correction_reference": "CORR-48",
        "prior_correction_reference": "CORR-47",
        "provenance_hashes_sha256": {
            "v3_manifest": _sha256_of(MANIFEST_V3_PATH),
            "v3_protocol_doc": _sha256_of(PROTOCOL_DOC_V3_PATH),
            "v4_manifest": _sha256_of(MANIFEST_V4_PATH),
            "v4_protocol_doc": _sha256_of(PROTOCOL_DOC_V4_PATH),
            "v5_manifest": _sha256_of(MANIFEST_V5_PATH),
            "v5_protocol_doc": _sha256_of(PROTOCOL_DOC_V5_PATH),
            "v6_manifest": _sha256_of(MANIFEST_V6_PATH),
            "v6_protocol_doc": _sha256_of(PROTOCOL_DOC_V6_PATH),
            "v7_manifest": _sha256_of(MANIFEST_V7_PATH),
            "v7_protocol_doc": _sha256_of(PROTOCOL_DOC_V7_PATH),
            "v8_manifest": _sha256_of(MANIFEST_V8_PATH),
            "v8_protocol_doc": _sha256_of(PROTOCOL_DOC_V8_PATH),
            "global_campaign_sequence": _sha256_of(GLOBAL_SEQUENCE_PATH),
        },
    }


DEFAULT_CPU_REQUEST_CORES_PER_POD = 0.1  # 100m, frozen application_resources.cpu_request_millicores (v1 manifest)
DEFAULT_CPU_LIMIT_CORES_PER_POD = 0.5  # 500m, frozen application_resources.cpu_limit_millicores (v1 manifest)


def cpu_normalization(
    aggregate_cpu_cores: float,
    current_replicas: int,
    cpu_request_cores_per_pod: float = DEFAULT_CPU_REQUEST_CORES_PER_POD,
    cpu_limit_cores_per_pod: float = DEFAULT_CPU_LIMIT_CORES_PER_POD,
) -> Dict[str, Any]:
    """Phase 15B.3F.1: the ONE reusable CPU-normalization helper -- closes a
    reporting/narrative-only bug in Phase 15B.3D/15B.3E/15B.3F's own report
    prose, which compared a MULTI-POD aggregate CPU reading directly against
    a SINGLE pod's request/limit (e.g. "0.53 cores... against the 0.5-core
    limit" for a 2-replica cell), instead of first scaling the denominator
    by `current_replicas` as this project's own already-frozen RL feature
    engineering rule requires (`docs/rl_mdp_spec.md` sec.4, CORR-27 v2:
    `cpu_per_replica_frac_request = (cpu_cores / ready_replicas) / 0.100`).
    This function generalizes that same request-relative-fraction logic
    into an explicit, reusable, directly-testable form, and adds the
    analogous limit-relative fraction and the raw aggregate denominators
    themselves, since report prose needs those spelled out to avoid the
    exact single-pod-denominator mistake this phase corrects.

    Returns a dict with all four CPU quantities this project's own reports
    must not conflate:
      - `aggregate_cpu_cores` (A): the raw, UNCHANGED empirical dynamics
        output -- never altered by this function.
      - `mean_cpu_cores_per_current_replica` (B):
        `aggregate_cpu_cores / current_replicas`.
      - `hpa_comparable_cpu_percent` (C): `aggregate_cpu_cores /
        (current_replicas * cpu_request_cores_per_pod) * 100` -- nominally
        comparable to HPA's own `averageUtilization` (per CORR-27 v2's own
        "nominally comparable... not literally identical" caveat, which
        still applies in full here: Metrics Server sampling/staleness,
        eligible-pod selection, and kube-controller-manager's own
        reconciliation/tolerance rules are not reproduced by this
        arithmetic identity).
      - `aggregate_cpu_limit_percent` (D): `aggregate_cpu_cores /
        (current_replicas * cpu_limit_cores_per_pod) * 100` -- how much of
        the pods' own CPU LIMIT (never HPA's target) is in use in aggregate.

    Raises ValueError on `current_replicas <= 0`, negative
    `aggregate_cpu_cores`, or non-positive request/limit-per-pod values --
    never silently produces a division-by-zero or a negative-CPU artifact.
    """
    if current_replicas <= 0:
        raise ValueError(f"current_replicas must be > 0, got {current_replicas}")
    if aggregate_cpu_cores < 0:
        raise ValueError(f"aggregate_cpu_cores must be >= 0, got {aggregate_cpu_cores}")
    if cpu_request_cores_per_pod <= 0:
        raise ValueError(f"cpu_request_cores_per_pod must be > 0, got {cpu_request_cores_per_pod}")
    if cpu_limit_cores_per_pod <= 0:
        raise ValueError(f"cpu_limit_cores_per_pod must be > 0, got {cpu_limit_cores_per_pod}")

    aggregate_request_cores = current_replicas * cpu_request_cores_per_pod
    aggregate_limit_cores = current_replicas * cpu_limit_cores_per_pod
    mean_cpu_cores_per_current_replica = aggregate_cpu_cores / current_replicas
    hpa_comparable_cpu_percent = (aggregate_cpu_cores / aggregate_request_cores) * 100.0
    aggregate_cpu_limit_percent = (aggregate_cpu_cores / aggregate_limit_cores) * 100.0

    return {
        "aggregate_cpu_cores": aggregate_cpu_cores,
        "current_replicas": current_replicas,
        "aggregate_request_cores": aggregate_request_cores,
        "aggregate_limit_cores": aggregate_limit_cores,
        "mean_cpu_cores_per_current_replica": mean_cpu_cores_per_current_replica,
        "hpa_comparable_cpu_percent": hpa_comparable_cpu_percent,
        "aggregate_cpu_limit_percent": aggregate_cpu_limit_percent,
    }


def hpa_target_equivalent_aggregate_cpu_cores(
    current_replicas: int,
    hpa_target_average_utilization_percent: float = 260.0,
    cpu_request_cores_per_pod: float = DEFAULT_CPU_REQUEST_CORES_PER_POD,
) -> float:
    """Phase 15B.3F.1: the aggregate CPU (in cores) that would correspond to
    the frozen HPA target (`averageUtilization=260`) at a given replica
    count -- a REFERENCE relationship only, per CORR-27 v2's own
    "nominally comparable, not literally identical" caveat. Never implies
    HPA actually would or did scale merely because one instantaneous
    aggregate sample exceeds this value -- real HPA behavior additionally
    depends on Metrics Server sampling/staleness, eligible-pod selection,
    HPA's own reconciliation cadence, and its behavior/tolerance rules,
    none of which this arithmetic identity reproduces.
    """
    if current_replicas <= 0:
        raise ValueError(f"current_replicas must be > 0, got {current_replicas}")
    return current_replicas * cpu_request_cores_per_pod * (hpa_target_average_utilization_percent / 100.0)


# ---------------------------------------------------------------------------
# Slot resolution -- the ONLY scientific-parameter entry point the harness
# is allowed to use. No CLI flag anywhere overrides any field returned here.
# ---------------------------------------------------------------------------


def resolve_planned_slot(campaign_sequence_index: int, sequence: Optional[List[dict]] = None) -> dict:
    """Resolve the one authoritative planned-slot record for an index.

    Raises ValueError if the index is out of the frozen [1, len(sequence)]
    range or is not found -- this is the harness's sole defense against
    ever attempting an unplanned/invented run.
    """
    if sequence is None:
        sequence = load_global_sequence()
    for entry in sequence:
        if entry["campaign_sequence_index"] == campaign_sequence_index:
            return dict(entry)
    raise ValueError(
        f"campaign_sequence_index={campaign_sequence_index} not found in the frozen "
        f"global sequence (valid range: 1..{len(sequence)})"
    )


def verify_schedule_hash(slot: dict, registry: Optional[dict] = None) -> str:
    """Independently verify the slot's schedule hash against THREE sources:
    the global-sequence entry itself, the schedule registry, and the actual
    on-disk file -- CORR-32/34 require all three to agree before any traffic.

    Returns the verified hash. Raises RuntimeError on any mismatch or a
    missing file, describing exactly which source disagreed.
    """
    if registry is None:
        registry = load_schedule_registry()

    schedule_path = PROJECT_ROOT / slot["schedule_path"]
    if not schedule_path.exists():
        raise RuntimeError(f"Schedule file does not exist: {schedule_path}")

    on_disk_hash = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    sequence_hash = slot["schedule_sha256"]

    registry_entry = registry.get(slot["cell_or_transition_id"])
    if registry_entry is None:
        raise RuntimeError(f"No schedule_registry.json entry for cell_or_transition_id={slot['cell_or_transition_id']!r}")
    registry_hash = registry_entry["schedule_sha256"]

    if not (on_disk_hash == sequence_hash == registry_hash):
        raise RuntimeError(
            "Schedule hash mismatch -- refusing to send traffic. "
            f"on_disk={on_disk_hash} sequence={sequence_hash} registry={registry_hash}"
        )
    return on_disk_hash


# ---------------------------------------------------------------------------
# Attempt identity
# ---------------------------------------------------------------------------


def utc_timestamp_id(when: Optional[datetime] = None) -> str:
    """The repository's established UTC experiment-ID timestamp convention,
    e.g. 20260908T102529Z (matches existing persistent-data/traffic/* IDs)."""
    when = when or datetime.now(timezone.utc)
    return when.strftime("%Y%m%dT%H%M%SZ")


def make_attempt_id(planned_slot_id: str, attempt_number: int, when: Optional[datetime] = None) -> str:
    return f"{planned_slot_id}_attempt{attempt_number}_{utc_timestamp_id(when)}"


# ---------------------------------------------------------------------------
# Campaign ledger (CORR-33 sec.12 / CORR-34 sec.35 spec) -- a derived
# convenience index; per-attempt metadata remains the source of truth.
# ---------------------------------------------------------------------------


def init_ledger(sequence: Sequence[dict]) -> dict:
    slots = {}
    for entry in sequence:
        idx = entry["campaign_sequence_index"]
        slots[str(idx)] = {
            "campaign_sequence_index": idx,
            "planned_slot_id": entry["planned_slot_id"],
            "state": LEDGER_STATE_NOT_ATTEMPTED,
            "all_attempt_ids": [],
            "valid_attempt_id": None,
        }
    return {"slots": slots}


def load_ledger(path: Path = LEDGER_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_ledger(ledger: dict, path: Path = LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".ledger_tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def ensure_ledger(sequence: Sequence[dict], path: Path = LEDGER_PATH) -> dict:
    """Load the ledger if it exists; otherwise initialize (all NOT_ATTEMPTED)
    and persist it. Never regenerates an existing ledger's contents."""
    existing = load_ledger(path)
    if existing is not None:
        return existing
    ledger = init_ledger(sequence)
    atomic_write_ledger(ledger, path)
    return ledger


def update_slot_state(
    ledger: dict,
    campaign_sequence_index: int,
    state: str,
    attempt_id: Optional[str] = None,
    mark_valid: bool = False,
) -> dict:
    if state not in (LEDGER_STATE_NOT_ATTEMPTED, LEDGER_STATE_VALID, LEDGER_STATE_INVALID_RETRY_REQUIRED):
        raise ValueError(f"unknown ledger state: {state!r}")
    key = str(campaign_sequence_index)
    if key not in ledger["slots"]:
        raise KeyError(f"campaign_sequence_index={campaign_sequence_index} not present in ledger")
    slot = ledger["slots"][key]
    slot["state"] = state
    if attempt_id is not None and attempt_id not in slot["all_attempt_ids"]:
        slot["all_attempt_ids"].append(attempt_id)
    if mark_valid:
        if attempt_id is None:
            raise ValueError("mark_valid=True requires an attempt_id")
        slot["valid_attempt_id"] = attempt_id
    else:
        # valid_attempt_id must always reflect "the currently valid attempt,
        # or None" -- a transition away from VALID (e.g. a corrected
        # INVALID_RETRY_REQUIRED determination) must clear a stale pointer
        # left over from an earlier mark_valid=True call, never leave it
        # dangling on an attempt no longer considered valid.
        slot["valid_attempt_id"] = None
    return ledger


def next_resumable_index(ledger: dict) -> Optional[int]:
    """The lowest campaign_sequence_index with no VALID attempt yet -- the
    ONE frozen resumption rule (CORR-34 sec.34). Never picks an "easier" run."""
    not_valid = [
        int(k) for k, v in ledger["slots"].items() if v["state"] != LEDGER_STATE_VALID
    ]
    return min(not_valid) if not_valid else None


def reconstruct_ledger_from_attempts(sequence: Sequence[dict], attempts: Sequence[dict]) -> dict:
    """Prove the ledger is a derived convenience index: rebuild it purely
    from a list of per-attempt metadata records (each with
    campaign_sequence_index, attempt_id, structural_validity in
    {"VALID","INVALID"}), with no other state.
    """
    ledger = init_ledger(sequence)
    for attempt in attempts:
        idx = attempt["campaign_sequence_index"]
        is_valid = attempt["structural_validity"] == "VALID"
        state = LEDGER_STATE_VALID if is_valid else LEDGER_STATE_INVALID_RETRY_REQUIRED
        update_slot_state(ledger, idx, state, attempt_id=attempt["attempt_id"], mark_valid=is_valid)
    return ledger


# ---------------------------------------------------------------------------
# CORR-35 exposure-weighted routing reconstruction and classification
# ---------------------------------------------------------------------------


def reconstruct_counter_total(samples: Sequence[Tuple[float, float]]) -> Tuple[float, int]:
    """Reset-aware positive-delta reconstruction of a raw Counter's total
    delta across an ordered list of (timestamp, value) samples.

    A negative diff is a counter reset: it contributes 0 (never a fabricated
    or subtracted negative amount), and the running total simply continues
    accumulating from the post-reset sample going forward. Returns
    (total_delta, reset_count).
    """
    ordered = sorted(samples, key=lambda s: s[0])
    total = 0.0
    reset_count = 0
    for (_, prev_v), (_, cur_v) in zip(ordered, ordered[1:]):
        diff = cur_v - prev_v
        if diff >= 0:
            total += diff
        else:
            reset_count += 1
    return total, reset_count


EXCLUDED_INCOMPLETE_POD_COUNTER_OBSERVABILITY = "EXCLUDED_INCOMPLETE_POD_COUNTER_OBSERVABILITY"
EXCLUDED_OTHER_NO_RELEVANT_PODS = "EXCLUDED_OTHER_NO_RELEVANT_PODS"


def compute_expected_and_actual(
    boundary_timestamps: Sequence[float],
    ready_pod_set_at: Dict[float, Sequence[str]],
    pod_raw_samples: Dict[str, Sequence[Tuple[float, float]]],
    relevant_pod_set_at: Optional[Dict[float, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """CORR-35's interval-level, exposure-weighted expected routing
    opportunity, plus each pod's independently-reconstructed actual count,
    over the CORRECT identical interval domain, gated by COMPLETE per-pod
    joint observability (Phase 15B.3C.2, closing a remaining loophole
    found after the Phase 15B.3C.1 domain-partition fix).

    Every interval `[t_j, t_j1)` resolves to exactly one of four
    categories:

      - `ALLOCATABLE` (`N_ready_j > 0`, every relevant pod jointly
        observed): the interval aggregate is computed from EVERY relevant
        pod's counter delta; Ready-set membership determines only who
        SHARES the resulting expected credit. `expected` and the
        classification-domain `actual` are BOTH accumulated ONLY over
        this domain.
      - `ZERO_READY` (`N_ready_j == 0`, every relevant pod jointly
        observed): real traffic with no Ready pod to receive expected
        credit -- tracked separately (`zero_ready_actual`/
        `zero_ready_traffic_total`), a health/service-routing diagnostic,
        never routing-fairness evidence.
      - `EXCLUDED_INCOMPLETE_POD_COUNTER_OBSERVABILITY`: at least one
        RELEVANT pod (see below) lacks a counter sample at one or both of
        this interval's boundaries. The interval is dropped ENTIRELY --
        never partially reconstructed from whichever pods happen to have
        complete data. Zero conservation error on an incomplete subset is
        not sufficient evidence of trustworthiness (Phase 15B.3C.1 fixed
        expected-vs-actual domain matching, but a further loophole
        remained: BOTH could still be computed from an incomplete pod
        subset while trivially conserving against each other).
      - `EXCLUDED_OTHER_NO_RELEVANT_PODS`: no pod is relevant to this
        interval at all (e.g. between two schedule/collector artifacts
        with no pod activity) -- vanishingly rare in practice, kept for
        completeness rather than silently dropped without a reason.

    A pod is RELEVANT to interval `[t_j, t_j1)` iff it appears in
    `relevant_pod_set_at[t_j]` or `relevant_pod_set_at[t_j1]` (the
    authoritative Kubernetes-API pod-presence history -- ANY status,
    Ready or not, sourced from `pod_readiness_history.csv`, never merely
    `ready_pod_set_at`, since a NotReady-but-observable pod can still
    legitimately carry real traffic) OR it has a counter sample at either
    boundary (counter-series identity, a defensive fallback for the rare
    case the two independently-polled collectors disagree on an instant).
    A pod not yet created (no readiness presence and no counter sample
    anywhere before this interval) or already deleted (none after) is
    correctly NOT relevant and never forces an exclusion -- only a pod
    that plausibly existed and could have served traffic, but is missing
    a boundary sample, does.

    `relevant_pod_set_at` is optional for pure counter-series-only callers
    (falls back to counter-series identity alone) -- real orchestrator
    usage always supplies it from the full readiness history.

    Phase 15B.3C's implementation bug (this function's OLDER docstring):
    `actual` was accumulated unconditionally across BOTH ALLOCATABLE and
    ZERO_READY domains, comparing `expected` and `actual` over different
    domains (fixed in Phase 15B.3C.1). Phase 15B.3C.2 closes the further
    loophole where an interval could still be silently reconstructed from
    an INCOMPLETE pod subset even after that fix (an implementation-
    conformance tightening, not a new methodology rule -- CORR-35/37 never
    permitted partial-pod-population fairness reconstruction; see CORR-38
    policy discussion in the Phase 15B.3C.2 report).

    `boundary_timestamps` must be sorted ascending and aligned with entries
    in `ready_pod_set_at`/`relevant_pod_set_at` and with sample timestamps
    in `pod_raw_samples` (the collectors poll at the same cadence, so this
    holds by construction for real data, after cross-process epoch
    alignment).
    """
    boundaries = sorted(boundary_timestamps)
    all_pods = sorted(pod_raw_samples.keys())
    expected: Dict[str, float] = {p: 0.0 for p in all_pods}
    actual: Dict[str, float] = {p: 0.0 for p in all_pods}
    zero_ready_actual: Dict[str, float] = {p: 0.0 for p in all_pods}
    reset_counts: Dict[str, int] = {p: 0 for p in all_pods}

    samples_by_pod_ts = {p: dict(pod_raw_samples[p]) for p in all_pods}

    allocatable_aggregate_total = 0.0
    zero_ready_traffic_total = 0.0
    candidate_interval_count = 0
    allocatable_interval_count = 0
    zero_ready_interval_count = 0
    excluded_incomplete_observability_count = 0
    excluded_interval_reasons: List[Dict[str, Any]] = []

    for t_j, t_j1 in zip(boundaries, boundaries[1:]):
        candidate_interval_count += 1

        relevant = set()
        if relevant_pod_set_at is not None:
            relevant |= set(relevant_pod_set_at.get(t_j, []))
            relevant |= set(relevant_pod_set_at.get(t_j1, []))
        relevant |= {p for p in all_pods if t_j in samples_by_pod_ts[p] or t_j1 in samples_by_pod_ts[p]}

        if not relevant:
            excluded_interval_reasons.append(
                {"interval_start": t_j, "interval_end": t_j1, "reason": EXCLUDED_OTHER_NO_RELEVANT_PODS}
            )
            continue

        missing = sorted(p for p in relevant if not (t_j in samples_by_pod_ts.get(p, {}) and t_j1 in samples_by_pod_ts.get(p, {})))
        if missing:
            excluded_incomplete_observability_count += 1
            excluded_interval_reasons.append(
                {
                    "interval_start": t_j,
                    "interval_end": t_j1,
                    "reason": EXCLUDED_INCOMPLETE_POD_COUNTER_OBSERVABILITY,
                    "missing_pods": missing,
                }
            )
            continue

        observable = sorted(relevant)
        sum_j = sum(samples_by_pod_ts[p][t_j] for p in observable)
        sum_j1 = sum(samples_by_pod_ts[p][t_j1] for p in observable)
        agg_delta, _ = reconstruct_counter_total([(t_j, sum_j), (t_j1, sum_j1)])

        per_pod_delta: Dict[str, float] = {}
        for p in observable:
            d, r = reconstruct_counter_total([(t_j, samples_by_pod_ts[p][t_j]), (t_j1, samples_by_pod_ts[p][t_j1])])
            per_pod_delta[p] = d
            reset_counts[p] = reset_counts.get(p, 0) + r

        ready_set = list(ready_pod_set_at.get(t_j, []))
        n_ready = len(ready_set)

        if n_ready == 0:
            zero_ready_interval_count += 1
            zero_ready_traffic_total += agg_delta
            for p in observable:
                zero_ready_actual[p] = zero_ready_actual.get(p, 0.0) + per_pod_delta[p]
            continue

        allocatable_interval_count += 1
        allocatable_aggregate_total += agg_delta
        for p in observable:
            actual[p] = actual.get(p, 0.0) + per_pod_delta[p]
        increment = agg_delta / n_ready
        for p in ready_set:
            expected[p] = expected.get(p, 0.0) + increment

    whole_joint_observable_aggregate_total = allocatable_aggregate_total + zero_ready_traffic_total

    return {
        "expected": expected,
        "actual": actual,
        "zero_ready_actual": zero_ready_actual,
        "counter_reset_counts": reset_counts,
        "allocatable_aggregate_total": allocatable_aggregate_total,
        "zero_ready_traffic_total": zero_ready_traffic_total,
        "whole_joint_observable_aggregate_total": whole_joint_observable_aggregate_total,
        "expected_conservation_error": allocatable_aggregate_total - sum(expected.values()),
        "actual_conservation_error": allocatable_aggregate_total - sum(actual.values()),
        "candidate_interval_count": candidate_interval_count,
        "allocatable_interval_count": allocatable_interval_count,
        "zero_ready_interval_count": zero_ready_interval_count,
        "excluded_incomplete_observability_count": excluded_incomplete_observability_count,
        "excluded_interval_reasons": excluded_interval_reasons,
        "complete_pod_observability_pass": excluded_incomplete_observability_count == 0,
        # Phase 15B.3J.1 (index9 evidence closure): `expected_conservation_error`/
        # `actual_conservation_error` are a sum over the ALLOCATABLE domain --
        # when that domain is empty (`allocatable_interval_count == 0`, e.g.
        # every candidate interval excluded for incomplete pod counter
        # observability), the sum-over-zero-elements is numerically 0.0 even
        # though NO conservation arithmetic was actually exercised. That 0.0
        # must never be read as "conservation was tested and passed" on its
        # own -- these two fields make the domain size and assessability
        # explicit so a bare 0.0 can't be misinterpreted as a substantive
        # pass. They do not change any arithmetic above; they are read-only
        # derived facts about `allocatable_interval_count`.
        "retained_interval_count": allocatable_interval_count,
        "conservation_assessable": allocatable_interval_count > 0,
        "expected_conservation_error_numeric": allocatable_aggregate_total - sum(expected.values()),
        "actual_conservation_error_numeric": allocatable_aggregate_total - sum(actual.values()),
        # Backward-compatible aliases (kept as the primary "did this pass
        # conservation" signal consulted by the orchestrator/tests):
        "conservation_error": allocatable_aggregate_total - sum(actual.values()),
        "aggregate_total": allocatable_aggregate_total,
        "unattributed_total": zero_ready_traffic_total,
    }


def classify_routing(
    expected: Dict[str, float],
    actual: Dict[str, float],
    min_expected: float = 50.0,
    min_assessable: int = 2,
) -> Dict[str, Any]:
    """CORR-35's exhaustive 4-tier classifier, checked in the exact frozen
    precedence order: INSUFFICIENT -> STRUCTURAL_FAILURE -> HEALTHY -> SUSPECTED.
    """
    assessable = sorted(p for p in expected if expected[p] >= min_expected)
    if len(assessable) < min_assessable:
        return {
            "classification": ROUTING_INSUFFICIENT_SAMPLE,
            "assessable_pods": assessable,
            "ratios": {},
            "routing_ratio_cv": None,
            "reason": f"only {len(assessable)} assessable pod(s), need >= {min_assessable}",
        }

    ratios = {p: (actual.get(p, 0.0) / expected[p]) for p in assessable}
    mean = sum(ratios.values()) / len(ratios)
    variance = sum((r - mean) ** 2 for r in ratios.values()) / len(ratios)
    cv = (math.sqrt(variance) / mean) if mean > 0 else float("inf")

    starved = [p for p, r in ratios.items() if r < 0.01]
    if starved:
        return {
            "classification": ROUTING_STRUCTURAL_FAILURE,
            "assessable_pods": assessable,
            "ratios": ratios,
            "routing_ratio_cv": cv,
            "reason": f"pod(s) {starved} received <1% of expected routing opportunity",
        }

    if all(0.5 <= r <= 1.5 for r in ratios.values()) and cv <= 0.30:
        return {
            "classification": ROUTING_HEALTHY,
            "assessable_pods": assessable,
            "ratios": ratios,
            "routing_ratio_cv": cv,
            "reason": "every assessable pod within [0.5,1.5]x expected and CV<=0.30",
        }

    return {
        "classification": ROUTING_PINNING_SUSPECTED,
        "assessable_pods": assessable,
        "ratios": ratios,
        "routing_ratio_cv": cv,
        "reason": "sufficiently sampled, not structural, not within healthy bounds",
    }


# ---------------------------------------------------------------------------
# Decision-aligned extraction (never a future sample)
# ---------------------------------------------------------------------------


def select_decision_sample(
    target_elapsed_seconds: float,
    samples: Sequence[Dict[str, Any]],
    staleness_threshold_seconds: float = 30.0,
    elapsed_key: str = "elapsed_seconds",
    validity_key: Optional[str] = "sample_valid",
) -> Optional[Dict[str, Any]]:
    """The latest VALID sample at-or-before target_elapsed_seconds, within
    the frozen staleness tolerance. Never returns a sample after the
    target, and never returns an invalid sample (`validity_key` truthy
    check, e.g. metrics_collector's own `sample_valid` column) -- the
    frozen protocol's extraction rule is explicitly "latest VALID
    observation", not merely "latest observation" (docs/rl_dynamics_calibration_protocol.md
    sec.9 / experiments/manifests/rl_dynamics_calibration_v1.yaml
    steady_state_timing.extraction_rule).

    Pass validity_key=None to disable the validity filter (e.g. for input
    data that has no such column).
    """
    candidates = [s for s in samples if s[elapsed_key] <= target_elapsed_seconds]
    if validity_key is not None:
        def _is_valid(s: Dict[str, Any]) -> bool:
            v = s.get(validity_key)
            return v is True or (isinstance(v, str) and v.strip().lower() == "true")

        candidates = [s for s in candidates if _is_valid(s)]
    if not candidates:
        return None
    best = max(candidates, key=lambda s: s[elapsed_key])
    if target_elapsed_seconds - best[elapsed_key] > staleness_threshold_seconds:
        return None
    return best


# ---------------------------------------------------------------------------
# Generator structural checks
# ---------------------------------------------------------------------------


def check_offered_equals_sent(traffic_metadata: dict) -> Tuple[bool, str]:
    offered = traffic_metadata["total_offered"]
    sent = traffic_metadata["total_sent"]
    if offered != sent:
        return False, f"offered ({offered}) != sent ({sent})"
    return True, "offered == sent"


def check_no_dropped_overload(traffic_metadata: dict) -> Tuple[bool, str]:
    dropped = traffic_metadata.get("outcome_totals", {}).get("dropped_overload", 0)
    if dropped:
        return False, f"{dropped} request(s) dropped_overload -- harness-capacity structural defect"
    return True, "no dropped_overload outcomes"


def check_no_external_contamination_during_run(readiness_metadata: dict) -> Tuple[bool, str]:
    """Phase 15B.3E: preflight alone only checks ONCE, before mutation, that
    no unrelated Docker-Desktop/Kind container is running -- Docker Desktop
    could in principle restart one of those containers mid-campaign (as
    happened before Phase 15B.3D's live run even started). The pod-
    readiness collector's continuous contamination watch (recorded in
    `cluster/external_container_contamination_history.csv`, summarized in
    `pod_readiness_metadata.json`) is consulted here to enforce, for the
    FULL duration of formal stabilization + traffic, the SAME already-
    frozen structural principle preflight checks once
    (`unrelated_kind_cluster_contamination`, inherited unchanged since V1)
    -- never silently accepting measurements collected under new host
    contamination that arose after preflight passed.
    """
    ticks = readiness_metadata.get("external_contamination_detected_ticks_elapsed_seconds", [])
    if ticks:
        return False, f"unrelated_kind_cluster_contamination: unrelated container(s) became RUNNING during the run at elapsed_seconds={ticks}"
    return True, "no unrelated Kind/Docker-Desktop-Kubernetes container became RUNNING during the run"


# ---------------------------------------------------------------------------
# Raw collector output parsers (pure -- the collector scripts do the I/O and
# call these to turn one poll's raw JSON into row dicts / a pod->value map)
# ---------------------------------------------------------------------------


def parse_pod_readiness_snapshot(
    pods_json: dict,
    timestamp_utc: str,
    timestamp_epoch: float,
    elapsed_seconds: float,
) -> List[Dict[str, Any]]:
    """Turn one `kubectl get pods -o json` snapshot into per-pod readiness
    history rows. Never invents a Ready condition that is not present."""
    rows: List[Dict[str, Any]] = []
    for item in pods_json.get("items", []):
        metadata = item.get("metadata", {})
        status = item.get("status", {})
        conditions = status.get("conditions", []) or []
        ready_condition = next((c for c in conditions if c.get("type") == "Ready"), None)
        container_statuses = status.get("containerStatuses", []) or []
        restart_count = sum(cs.get("restartCount", 0) for cs in container_statuses)
        rows.append(
            {
                "timestamp_utc": timestamp_utc,
                "timestamp_epoch": timestamp_epoch,
                "elapsed_seconds": elapsed_seconds,
                "pod_name": metadata.get("name", ""),
                "pod_uid": metadata.get("uid", ""),
                "phase": status.get("phase", ""),
                "ready_status": ready_condition.get("status", "") if ready_condition else "",
                "ready_last_transition_time": ready_condition.get("lastTransitionTime", "") if ready_condition else "",
                "deletion_timestamp": metadata.get("deletionTimestamp", "") or "",
                "restart_count": restart_count,
            }
        )
    return rows


def is_pod_ready_row(row: Dict[str, Any]) -> bool:
    return row.get("ready_status") == "True"


def parse_pod_container_state_snapshot(
    pods_json: dict,
    timestamp_utc: str,
    timestamp_epoch: float,
    elapsed_seconds: float,
) -> List[Dict[str, Any]]:
    """Phase 15B.3C evidence-preservation hardening: turn one
    `kubectl get pods -o json` snapshot into per-CONTAINER state history
    rows (restartCount, current state, lastState/termination reason/exit
    code, startedAt/finishedAt, containerID). Never fabricates absent
    termination information -- missing fields are empty strings, not
    invented values."""
    rows: List[Dict[str, Any]] = []
    for item in pods_json.get("items", []):
        metadata = item.get("metadata", {})
        status = item.get("status", {})
        for cs in status.get("containerStatuses", []) or []:
            state = cs.get("state", {}) or {}
            last_state = cs.get("lastState", {}) or {}
            state_kind = next(iter(state.keys()), "")
            last_state_kind = next(iter(last_state.keys()), "")
            last_terminated = last_state.get("terminated", {}) or {}
            rows.append(
                {
                    "timestamp_utc": timestamp_utc,
                    "timestamp_epoch": timestamp_epoch,
                    "elapsed_seconds": elapsed_seconds,
                    "pod_name": metadata.get("name", ""),
                    "container_name": cs.get("name", ""),
                    "container_id": cs.get("containerID", "") or "",
                    "restart_count": cs.get("restartCount", 0),
                    "current_state": state_kind,
                    "last_state": last_state_kind,
                    "last_termination_reason": last_terminated.get("reason", ""),
                    "last_termination_exit_code": last_terminated.get("exitCode", ""),
                    "last_termination_started_at": last_terminated.get("startedAt", ""),
                    "last_termination_finished_at": last_terminated.get("finishedAt", ""),
                }
            )
    return rows


def normalize_kubernetes_events(events_json: dict) -> List[Dict[str, Any]]:
    """Turn a raw `kubectl get events -o json` payload into a flat,
    normalized row list. Read-only transformation -- never issues any
    Kubernetes call itself."""
    rows: List[Dict[str, Any]] = []
    for item in events_json.get("items", []):
        involved = item.get("involvedObject", {}) or {}
        rows.append(
            {
                "first_timestamp": item.get("firstTimestamp") or item.get("eventTime") or "",
                "last_timestamp": item.get("lastTimestamp") or item.get("eventTime") or "",
                "count": item.get("count", 1),
                "type": item.get("type", ""),
                "reason": item.get("reason", ""),
                "involved_object_kind": involved.get("kind", ""),
                "involved_object_name": involved.get("name", ""),
                "involved_object_uid": involved.get("uid", ""),
                "message": item.get("message", ""),
            }
        )
    return rows


def restarted_pods_between(before_rows: Sequence[Dict[str, Any]], after_rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Compare two parse_pod_container_state_snapshot-style row sets
    (or parse_pod_readiness_snapshot rows, which also carry restart_count)
    and return pod names whose restart_count increased."""
    before_by_pod: Dict[str, int] = {}
    for r in before_rows:
        pod = r["pod_name"]
        before_by_pod[pod] = max(before_by_pod.get(pod, 0), int(r["restart_count"]))
    restarted = []
    after_by_pod: Dict[str, int] = {}
    for r in after_rows:
        pod = r["pod_name"]
        after_by_pod[pod] = max(after_by_pod.get(pod, 0), int(r["restart_count"]))
    for pod, after_count in after_by_pod.items():
        if after_count > before_by_pod.get(pod, 0):
            restarted.append(pod)
    return sorted(restarted)


# ---------------------------------------------------------------------------
# Per-pod restart-evidence classification (Phase 15B.3J.1, index9 evidence
# closure): `restarted_pods_between`/`restarted_pods_during_run` are, by
# design, Category-A-only (a completed, directly-observed restart_count
# increment). A pod can receive a kubelet-issued, liveness-probe-attributed
# `Killing` action whose COMPLETION is never directly observed within the
# collection window -- that is real, non-fabricated evidence of a restart
# having been INITIATED, distinct both from a confirmed completion and from
# no restart evidence at all. This taxonomy makes that distinction explicit
# and testable, without changing `restarted_pods_between`'s own narrow,
# already-frozen contract.
# ---------------------------------------------------------------------------

RESTART_COMPLETED_AND_OBSERVED = "RESTART_COMPLETED_AND_OBSERVED"
RESTART_INITIATED_BY_KUBELET = "RESTART_INITIATED_BY_KUBELET"
NO_RESTART_EVIDENCE = "NO_RESTART_EVIDENCE"


def classify_pod_restart_evidence(
    pod_uid: str,
    baseline_restart_count: int,
    final_restart_count: int,
    killing_events: Sequence[Dict[str, Any]],
    run_start_utc: str,
    run_end_utc: str,
) -> str:
    """Classify one pod's restart evidence for a single run into exactly
    one of three states:

      - `RESTART_COMPLETED_AND_OBSERVED`: `final_restart_count >
        baseline_restart_count` -- a directly-observed restart_count
        increment is the strongest possible evidence and takes precedence
        regardless of anything in `killing_events` (a pod's completed
        restart is real even if the corresponding kubelet event was never
        captured or already scrolled out of the events API).
      - `RESTART_INITIATED_BY_KUBELET`: no restart_count increase was
        observed, but at least one `killing_events` entry is a genuine,
        this-run, liveness-probe-attributed kill of THIS pod (see filters
        below).
      - `NO_RESTART_EVIDENCE`: neither of the above.

    `killing_events` entries are expected to be `normalize_kubernetes_events`-
    shaped rows, carrying `reason`, `involved_object_uid`, `message`, and
    `first_timestamp` (ISO-8601 UTC, directly string-comparable). An event
    is admitted as this-run evidence only if ALL of:
      1. `reason == "Killing"` (excludes Unhealthy probe-failure warnings,
         which alone never imply a kill/restart actually happened).
      2. `involved_object_uid == pod_uid` (never attributes an event to
         the wrong pod object merely because two pods share a name across
         create/delete cycles -- UID is the canonical identity, per
         Protocol V3/CORR-38).
      3. `"liveness probe" in message.lower()` (excludes a plain
         scale-down/graceful-termination `Killing` event, which carries a
         different message and is never a liveness-triggered restart).
      4. `run_start_utc <= first_timestamp <= run_end_utc` -- Kubernetes
         merges repeated identical events into ONE object carrying
         `firstTimestamp`/`lastTimestamp`/`count` spanning the object's
         ENTIRE observed history. An event whose `first_timestamp` predates
         this run's own start is contaminated by earlier, unrelated runs
         against the same long-lived pod object and must be excluded --
         even though the SAME merged object's `last_timestamp` might
         genuinely fall inside this run (the merged object does not carry
         enough information to split that apart, so it is conservatively
         excluded from Category B rather than guessed at).
    """
    if final_restart_count > baseline_restart_count:
        return RESTART_COMPLETED_AND_OBSERVED
    for ev in killing_events:
        if ev.get("reason") != "Killing":
            continue
        if ev.get("involved_object_uid") != pod_uid:
            continue
        message = (ev.get("message") or "").lower()
        if "liveness probe" not in message:
            continue
        first = ev.get("first_timestamp")
        if first is not None and run_start_utc <= first <= run_end_utc:
            return RESTART_INITIATED_BY_KUBELET
    return NO_RESTART_EVIDENCE


def parse_prometheus_vector_by_pod(response_json: dict, pod_label: str = "pod") -> Dict[str, float]:
    """Turn one instant-query vector response into {pod_name: raw_value}.

    Each series of demo_app_requests_total carries its own `pod` label
    (one Counter per pod, per-pod-exposed metrics) -- this is a plain
    per-series extraction, never an aggregation.
    """
    if response_json.get("status") != "success":
        return {}
    result = response_json.get("data", {}).get("result", [])
    out: Dict[str, float] = {}
    for series in result:
        pod = series.get("metric", {}).get(pod_label)
        if not pod:
            continue
        raw_val = series.get("value", [None, None])[1]
        if raw_val is None:
            continue
        try:
            out[pod] = float(raw_val)
        except (TypeError, ValueError):
            continue
    return out


def detect_pod_series_gap(previous_pod_names: Sequence[str], current_pod_names: Sequence[str]) -> List[str]:
    """Phase 15B.3C.2 (future-collector hardening, optional): given the set
    of pod names returned by the counter collector's PREVIOUS poll and its
    CURRENT poll, return the pod names present previously but missing now
    -- an explicit, self-relative scrape-gap signal recorded at collection
    time, independent of any later cross-referencing against the readiness
    collector's own (differently-clocked) snapshots. Does not itself
    decide relevance/exclusion (that remains `compute_expected_and_actual`'s
    job) -- purely a diagnostic aid for future runs to make gaps visible
    immediately rather than only discoverable in post-hoc forensic audits.
    """
    return sorted(set(previous_pod_names) - set(current_pod_names))


def compute_missing_known_pods(
    known_pods: Sequence[Dict[str, Any]],
    prometheus_pod_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Phase 15B.3C.3 (CORR-38 sec.C): a SNAPSHOT-based missing-pod-series
    check -- unlike `detect_pod_series_gap` (which only detects a pod
    PREVIOUSLY returned by Prometheus that has since vanished),
    this compares the CURRENT tick's authoritative Kubernetes-known pod
    population (`known_pods`, e.g. from `parse_pod_readiness_snapshot()`)
    directly against the CURRENT tick's Prometheus-returned pod names.
    This correctly detects a pod absent from the very FIRST tick, or
    absent for the ENTIRE run -- cases the previous-tick-relative check
    could never catch, exactly the failure mode observed for `6lld4` in
    index2 (Phase 15B.3C.2 sec.8). Returns one entry per Kubernetes-known
    pod whose name is not present in `prometheus_pod_names`, carrying
    enough context (uid, phase, ready_status) to distinguish "pod exists
    but Prometheus never saw it" from other conditions without
    fabricating a counter value.
    """
    returned = set(prometheus_pod_names)
    missing = []
    for pod in known_pods:
        if pod.get("pod_name") not in returned:
            missing.append(
                {
                    "pod_uid": pod.get("pod_uid", ""),
                    "pod_name": pod.get("pod_name", ""),
                    "phase": pod.get("phase", ""),
                    "ready_status": pod.get("ready_status", ""),
                    "deletion_timestamp": pod.get("deletion_timestamp", ""),
                }
            )
    return missing


def detect_pod_identity_change(previous_uid_by_name: Dict[str, str], current_uid_by_name: Dict[str, str]) -> List[str]:
    """Phase 15B.3C.3 (CORR-38 sec.A, identity_consistency_check): Kubernetes
    pod UID is the canonical pod identity; pod NAME is only the
    human-readable join label used because Prometheus's own `pod` metric
    label carries no UID. Returns the names present in BOTH mappings whose
    UID differs between them -- such a name must never be silently treated
    as the same object when joining counter samples across boundaries (a
    replaced pod object reusing a name, however unlikely given Kubernetes'
    random-suffix naming, must not have its counter series merged with its
    predecessor's).
    """
    changed = []
    for name, uid in current_uid_by_name.items():
        prev_uid = previous_uid_by_name.get(name)
        if prev_uid is not None and prev_uid != uid:
            changed.append(name)
    return sorted(changed)


EXTERNAL_CONTAMINATION_NAME_PREFIXES = ("desktop-", "kind-")


def classify_external_container_contamination(container_names: Sequence[str]) -> List[str]:
    """Phase 15B.3E: the single, shared definition of an "unrelated Kind/
    Docker-Desktop-Kubernetes container" -- Docker Desktop's own built-in
    Kubernetes feature auto-starts a 4-node "kind"-based cluster
    (`desktop-control-plane`, `desktop-worker`, `desktop-worker2`,
    `desktop-worker3`) plus support containers (`kind-cloud-provider`,
    `kind-registry-mirror`) whenever Docker Desktop launches with
    Kubernetes enabled -- entirely unrelated to this project's own
    `minikube` container, and a genuine source of CPU contention plus a
    frozen structural-invalidity criterion (`unrelated_kind_cluster_contamination`,
    inherited unchanged since V1) if present during an official attempt.
    Used both by `preflight_checks()` (a one-shot check before mutation)
    and by the live contamination observer (a continuous watch during
    stabilization/traffic, since Docker Desktop could in principle restart
    these containers mid-campaign) -- one definition, never duplicated.
    """
    return sorted(n for n in container_names if any(n.startswith(p) for p in EXTERNAL_CONTAMINATION_NAME_PREFIXES))


# ---------------------------------------------------------------------------
# CORR-37: Axis B -- per-decision-point and per-run fit-eligibility
# ---------------------------------------------------------------------------


def assess_decision_point_eligibility(
    sample: Optional[Dict[str, Any]],
    ready_count_at_target: int,
    intended_replicas: int,
) -> Tuple[bool, str]:
    """Frozen CORR-37 rule for ONE decision-aligned point.

    `sample` is the result of `select_decision_sample` (None if
    unobservable within the frozen staleness tolerance). `ready_count_at_target`
    must come from the authoritative per-pod Kubernetes Ready-condition
    history at (or nearest to) the target timestamp -- never from a
    Prometheus/kube-state-metrics aggregate. Never imputes a missing value.
    """
    if sample is None:
        return False, FIT_POINT_INELIGIBLE_TARGET_UNOBSERVABLE
    if ready_count_at_target != intended_replicas:
        return False, FIT_POINT_INELIGIBLE_HEALTH_INSTABILITY
    return True, FIT_POINT_ELIGIBLE_NONE


def run_level_fit_classification(fit_eligible_count: int, total_points: int = 4) -> str:
    if fit_eligible_count == total_points:
        return FIT_RUN_ELIGIBLE
    if fit_eligible_count == 0:
        return FIT_RUN_INELIGIBLE
    return FIT_RUN_PARTIAL


def is_repeat_sufficient_for_cell_support(fit_eligible_count: int) -> bool:
    return fit_eligible_count >= MIN_FIT_ELIGIBLE_POINTS_FOR_SUFFICIENT_REPEAT


def group_attempt_ids_by_cell(attempts: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Phase 15B.3K: the single, shared grouping any current or future
    model-fitting dataset builder MUST use for per-cell repeat counting --
    attempts are grouped strictly by their `cell_or_transition_id` string,
    verbatim as the frozen matrix/global sequence defines it.

    A merged cell (e.g. `dyncal_r8_L2_L5`, which the matrix intentionally
    deduplicated from what would otherwise have been two separate
    `dyncal_r8_L2`/`dyncal_r8_L5` cells that coincide at replicas=8 --
    see `experiments/manifests/rl_dynamics_calibration_v1.yaml`'s
    `deduplicated_cells`) has exactly ONE `cell_or_transition_id` string in
    every attempt's metadata. Grouping by that string therefore
    automatically and structurally cannot split or duplicate it into two
    per-role observations -- there is no "dyncal_r8_L2"/"dyncal_r8_L5" key
    for anything to be split into, by construction of the frozen matrix.
    This function does not decide sufficiency itself (see
    `is_repeat_sufficient_for_cell_support`/`cell_stable_support`); it only
    performs the grouping those functions' caller must first do correctly.
    """
    groups: Dict[str, List[str]] = {}
    for a in attempts:
        cell = a["cell_or_transition_id"]
        groups.setdefault(cell, []).append(a["attempt_id"])
    return groups


def cell_stable_support(sufficient_repeat_flags: Sequence[bool], repeat_type: str) -> str:
    """Frozen CORR-37 cell-level support rule, precommitted BEFORE
    observing repeats 2/3 of any cell: >=2 of 3 fitting repeats, or >=1 of
    2 validation repeats, must be `is_repeat_sufficient_for_cell_support`.

    Callers MUST pass the COMPLETE set of the cell's official repeats
    (i.e. all 3 fitting or both 2 validation repeats already executed) --
    calling this with a partial repeat set (e.g. only repeat 1 of a
    3-repeat cell) risks a PREMATURE conclusion. Use
    `cell_support_status_given_partial_repeats` while repeats remain
    outstanding.
    """
    if repeat_type not in ("fitting", "validation"):
        raise ValueError(f"unknown repeat_type: {repeat_type!r}")
    required = 2 if repeat_type == "fitting" else 1
    sufficient_count = sum(1 for f in sufficient_repeat_flags if f)
    return CELL_STABLE_SUPPORT if sufficient_count >= required else CELL_OUTSIDE_SUPPORT


def cell_support_status_given_partial_repeats(
    sufficient_repeat_flags: Sequence[bool],
    total_repeats_in_cell: int,
    repeat_type: str,
) -> str:
    """CORR-37: the mathematically honest status while a cell's repeats
    are still in progress -- never a premature STABLE/OUTSIDE call.

    `sufficient_repeat_flags` covers only the repeats EXECUTED so far
    (length <= total_repeats_in_cell). Returns:
      - CELL_STABLE_SUPPORT if the required count is already reached
        (decidable regardless of any remaining repeats),
      - CELL_OUTSIDE_SUPPORT if even every remaining repeat being
        sufficient could not reach the required count (decidable now),
      - CELL_SUPPORT_PENDING otherwise (only decidable once more repeats
        are observed).
    """
    if repeat_type not in ("fitting", "validation"):
        raise ValueError(f"unknown repeat_type: {repeat_type!r}")
    required = 2 if repeat_type == "fitting" else 1
    sufficient_count = sum(1 for f in sufficient_repeat_flags if f)
    remaining = total_repeats_in_cell - len(sufficient_repeat_flags)
    if sufficient_count >= required:
        return CELL_STABLE_SUPPORT
    if sufficient_count + remaining < required:
        return CELL_OUTSIDE_SUPPORT
    return CELL_SUPPORT_PENDING


def restoration_wait_seconds(starting_replicas: int, target_replicas: int = 1) -> float:
    """Phase 15B.3C.1: a conservative, deterministic upper bound on how
    long natural HPA scale-down convergence can take after HPA is
    restored, derived from the FROZEN scale-down mechanics
    (`k8s/hpa/hpa.yaml`, CORR-26) -- never an arbitrary convenience value,
    and never a direct-scale shortcut that would bypass HPA's own
    behavior. Must safely cover the worst case (`maxReplicas=8` -> 1).
    """
    if starting_replicas < target_replicas:
        raise ValueError(f"starting_replicas ({starting_replicas}) must be >= target_replicas ({target_replicas})")
    steps = starting_replicas - target_replicas
    return (
        HPA_SCALE_DOWN_STABILIZATION_SECONDS
        + steps * HPA_SCALE_DOWN_STEP_INTERVAL_SECONDS
        + HPA_OBSERVER_SYNC_SAFETY_MARGIN_SECONDS
    )


def ledger_state_for_execution_validity(execution_structural_validity: str) -> str:
    """CORR-37 ledger_semantics_v2: the ledger reflects Axis A ONLY.
    Fit-eligibility (Axis B) never influences ledger state."""
    if execution_structural_validity == EXECUTION_VALID:
        return LEDGER_STATE_VALID
    if execution_structural_validity == EXECUTION_INVALID:
        return LEDGER_STATE_INVALID_RETRY_REQUIRED
    raise ValueError(f"unknown execution_structural_validity: {execution_structural_validity!r}")


# ---------------------------------------------------------------------------
# Phase 15B.5B: replica-transition EXECUTION support.
#
# The scientific design for `replica_transition` slots was already frozen
# (manifest["replica_transitions"]: pairs, workload_rule, timing,
# measured_fields, schedules) -- Phase 15B.5A found that NO execution
# implementation existed for it (the harness only ever supported the
# single-scale steady-state flow, keyed on a single `slot["replicas"]`
# value that transition slots never carry). This section adds pure,
# unit-testable EXECUTION helpers only; it does not alter the frozen
# design in any way (see `resolve_transition_timing_plan`'s own docstring).
# ---------------------------------------------------------------------------

RUN_KIND_FITTING = "fitting"
RUN_KIND_INTERPOLATION_VALIDATION = "interpolation_validation"
RUN_KIND_REPLICA_TRANSITION = "replica_transition"
RUN_KIND_WORKLOAD_TRANSITION = "workload_transition"

TRANSITION_DIRECTION_SCALE_UP = "scale_up"
TRANSITION_DIRECTION_SCALE_DOWN = "scale_down"

TRANSITION_SCIENTIFIC_ROLE = "TRANSITION_TRAJECTORY_VALIDATION"

NOT_FORMALLY_OPERATIONALIZED = "NOT_FORMALLY_OPERATIONALIZED"

# The manifest's own `replica_transitions.measured_fields` names all seven
# of these, but only the three replica-count lags below have an
# unambiguous, already-frozen-by-construction definition: their start/end
# conditions are objectively the trigger timestamp and a Kubernetes-API-
# observed replica count crossing the transition's own target -- no
# arbitrary band/tolerance is required to define "reached". The other four
# (cpu/latency/in_progress "settling time", request_rate_realization_lag)
# are NAMED in the manifest but have NO frozen settling threshold/band/
# window defined anywhere in this project's manifest or protocol
# documentation (confirmed by repo-wide search, Phase 15B.5A/15B.5B).
# Inventing an arbitrary tolerance for them here would silently create a
# NEW scientific definition inside an execution-implementation task, which
# this project's own governing instructions explicitly forbid -- see
# `build_transition_metrics_summary`.
TRANSITION_MEASURED_FIELDS_WITH_FROZEN_DEFINITION = (
    "desired_to_current_lag_seconds",
    "desired_to_ready_lag_seconds",
    "desired_to_available_lag_seconds",
)
TRANSITION_MEASURED_FIELDS_NOT_FORMALLY_OPERATIONALIZED = (
    "cpu_settling_time_seconds",
    "latency_settling_time_seconds",
    "in_progress_settling_time_seconds",
    "request_rate_realization_lag_seconds",
)


def is_replica_transition_slot(slot: dict) -> bool:
    return slot.get("run_kind") == RUN_KIND_REPLICA_TRANSITION


# ---------------------------------------------------------------------------
# CORR-44 (Phase 15B.5C): replica-transition route-materiality/scored-window
# OPERATIONALIZATION -- freezes, before any official replica_transition
# attempt, WHICH values of `is_interval_material()`'s already-generic
# `scored_window`/`materiality_fraction_threshold` parameters apply to a
# `run_kind=replica_transition` run. `is_interval_material()` itself is
# UNCHANGED (it was already duration-relative by construction -- see its
# own docstring and `DEFAULT_ROUTE_INTEGRITY_MATERIALITY_FRACTION`'s), and
# steady-state's own [60,180)/45s-of-180s values (unaffected by this
# section) are untouched.
#
# Source-of-truth finding (Phase 15B.5C sec.4-6 of its own report): the
# original CORR-39 sec.5 materiality gate was ALREADY defined as a
# FRACTION of total traffic duration, not an absolute 45-second value --
# `DEFAULT_ROUTE_INTEGRITY_MATERIALITY_FRACTION = 0.25` and
# `is_interval_material()`'s own `fraction = duration /
# traffic_duration_seconds` prove this directly from the code itself, and
# CORR-39 sec.7's own prose says so explicitly: "a materiality criterion
# (zero-success interval, overlapping the scored window, covering >=25% of
# total traffic duration)". 45 seconds is simply `0.25 x 180s` for
# steady-state's OWN 180-second traffic duration -- it was never frozen as
# an absolute, duration-independent invariant anywhere in this project's
# documentation. This directly resolves the choice in favor of the
# duration-relative interpretation.
# ---------------------------------------------------------------------------

# CORR-44: replica-transition runs reuse the SAME 25% fraction steady-state
# already uses -- a deliberately named alias (not a new numeric value) so a
# future reader never has to wonder whether transitions use a DIFFERENT
# fraction than steady-state. They do not.
REPLICA_TRANSITION_MATERIALITY_FRACTION = DEFAULT_ROUTE_INTEGRITY_MATERIALITY_FRACTION


def replica_transition_route_scored_window(total_traffic_seconds: float) -> Tuple[float, float]:
    """CORR-44: for a `replica_transition` run, the ENTIRE traffic duration
    is the route-integrity scored window -- `[0, total_traffic_seconds)`,
    deliberately INCLUDING the pre-trigger
    `pre_transition_traffic_steady_seconds` baseline.

    This is NOT the steady-state flow's `[60,180)` pattern applied
    mechanically to a longer duration. Steady-state's own first 60 seconds
    is named `warmup_window_seconds` in the manifest and is explicitly
    excluded from scoring because it is discard-only preparation traffic.
    A replica-transition run's own first 60 seconds carries a materially
    DIFFERENT, explicitly-named manifest role:
    `pre_transition_traffic_steady_seconds` -- genuine, load-bearing
    baseline traffic that establishes the trajectory the transition is
    measured against, never described anywhere as discardable warmup. If
    the harness could not actually deliver that baseline traffic, the
    transition experiment built on top of it is not trustworthy merely
    because post-trigger delivery later recovers -- so a route-integrity
    failure during `[0,60)` must remain visible, not silently excluded
    from scoring the way steady-state's own warmup is.
    """
    return (0.0, float(total_traffic_seconds))


def replica_transition_materiality_threshold_seconds(
    total_traffic_seconds: float,
    fraction: float = REPLICA_TRANSITION_MATERIALITY_FRACTION,
) -> float:
    """CORR-44: the absolute material-interval-duration threshold implied
    by the frozen fraction, for a given transition traffic duration --
    pure arithmetic, exposed as its own named function (rather than left
    implicit inside a bare `fraction * duration` at each call site) so the
    frozen 240s-run number (`0.25 x 240 = 60.0`) is directly testable and
    citable on its own. For the frozen official 240-second transition
    schedule this evaluates to exactly 60.0 seconds -- NOT the
    steady-state 45.0s value, which remains specific to 180-second
    steady-state runs and is entirely unaffected by this function's
    existence."""
    return fraction * float(total_traffic_seconds)


def transition_direction(replicas_before: int, replicas_after: int) -> str:
    """Generic direction resolution -- supports both scale-up (e.g. 5->6)
    and scale-down (e.g. 2->1) directional slots, exactly as the frozen
    `directional_types` list in the manifest enumerates both directions of
    every pair."""
    if replicas_after == replicas_before:
        raise ValueError(f"replicas_before ({replicas_before}) == replicas_after ({replicas_after}); not a transition")
    return TRANSITION_DIRECTION_SCALE_UP if replicas_after > replicas_before else TRANSITION_DIRECTION_SCALE_DOWN


def resolve_transition_timing_plan(manifest: dict) -> Dict[str, float]:
    """Pure passthrough of the already-frozen
    `manifest["replica_transitions"]["timing"]` block -- never invents,
    overrides, or re-derives a value. Raises `KeyError` loudly if the
    manifest's own structure is ever missing an expected key, rather than
    silently defaulting one in."""
    timing = manifest["replica_transitions"]["timing"]
    return {
        "pre_run_replica_stabilization_seconds": float(timing["pre_run_replica_stabilization_seconds"]),
        "pre_transition_traffic_steady_seconds": float(timing["pre_transition_traffic_steady_seconds"]),
        "trigger_at_traffic_elapsed_seconds": float(timing["trigger_at_traffic_elapsed_seconds"]),
        "post_transition_observation_seconds": float(timing["post_transition_observation_seconds"]),
        "total_traffic_seconds": float(timing["total_traffic_seconds"]),
        "total_seconds_per_run": float(timing["total_seconds_per_run"]),
    }


def initial_replica_count_for_restoration(slot: dict) -> int:
    """Generalizes the single-replica-target restoration-wait computation
    (`restoration_wait_seconds`) to slots carrying `replicas_before`/
    `replicas_after` instead of a single `replicas` value: uses the HIGHER
    of the two -- the true worst-case starting point HPA scale-down
    restoration must cover, regardless of whether the transition itself
    completed by the time restoration begins. Steady-state slots (which
    carry `replicas`) are completely unaffected -- this is a superset
    helper, not a replacement for the existing steady-state code path."""
    if slot.get("replicas") is not None:
        return int(slot["replicas"])
    return max(int(slot["replicas_before"]), int(slot["replicas_after"]))


def count_schedule_data_rows(schedule_path: Path) -> int:
    """Pure count of data rows (excluding the header) in a schedule CSV --
    used to report `expected_requests` directly from the schedule file
    itself, never assumed from the offered-RPS x duration label alone."""
    with open(schedule_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


# ---------------------------------------------------------------------------
# CORR-47 (Phase 15B.6B): workload-transition pre-first-use operationalization.
#
# Phase 15B.6A found the manifest's own `workload_transitions` section
# (frozen since v1) had never actually been read by the execution harness:
# `RUN_KIND_WORKLOAD_TRANSITION` was a bare string constant with no
# corresponding `is_..._slot()` gate, no dedicated execution path, and
# `dry_run_report()`/`main()` silently fell through to the STEADY-STATE
# body, producing a factually wrong 180s/warmup-scored-window report for a
# run whose own frozen manifest section specifies a 240s/60s-step plan.
# This section closes that gap using the IDENTICAL pattern already
# established for `replica_transition` (CORR-44/CORR-45/CORR-46) -- no new
# architecture invented, only the workload-transition-specific instance of
# an already-proven pattern.
# ---------------------------------------------------------------------------

WORKLOAD_TRANSITION_SCIENTIFIC_ROLE = "WORKLOAD_TRANSITION_TRAJECTORY_VALIDATION"


def is_workload_transition_slot(slot: dict) -> bool:
    return slot.get("run_kind") == RUN_KIND_WORKLOAD_TRANSITION


def resolve_workload_transition_timing_plan(manifest: dict) -> Dict[str, float]:
    """Pure passthrough of the already-frozen
    `manifest["workload_transitions"]["timing"]` block -- mirrors
    `resolve_transition_timing_plan()`'s own pattern exactly. This block
    has existed, frozen, since v1 of the manifest; nothing in the
    execution harness ever read it before this correction (Phase 15B.6A
    finding). Raises `KeyError` loudly if the manifest's own structure is
    ever missing an expected key, rather than silently defaulting one in."""
    timing = manifest["workload_transitions"]["timing"]
    return {
        "pre_run_replica_stabilization_seconds": float(timing["pre_run_replica_stabilization_seconds"]),
        "pre_step_traffic_seconds": float(timing["pre_step_traffic_seconds"]),
        "trigger_at_traffic_elapsed_seconds": float(timing["trigger_at_traffic_elapsed_seconds"]),
        "post_step_observation_seconds": float(timing["post_step_observation_seconds"]),
        "total_traffic_seconds": float(timing["total_traffic_seconds"]),
        "total_seconds_per_run": float(timing["total_seconds_per_run"]),
    }


# CORR-47: reuses the SAME 0.25 duration-relative fraction already used by
# steady-state (CORR-39) and replica-transition (CORR-44) -- confirmed by
# direct inspection that `is_interval_material()` was ALWAYS a generic,
# duration-relative gate (`fraction = duration / traffic_duration_seconds`)
# never specifically justified only for replica-count transitions; CORR-39
# sec.7's own prose already stated the rule as "a materiality criterion
# ... covering >=25% of total traffic duration" in general terms. A
# workload-transition-specific correction is frozen here (rather than
# silently inheriting `REPLICA_TRANSITION_MATERIALITY_FRACTION`) so a
# future reader never has to guess whether this run kind's materiality was
# ever actually reviewed, per this correction's own governing task.
WORKLOAD_TRANSITION_MATERIALITY_FRACTION = DEFAULT_ROUTE_INTEGRITY_MATERIALITY_FRACTION


def workload_transition_route_scored_window(total_traffic_seconds: float) -> Tuple[float, float]:
    """CORR-47: for a `workload_transition` run, the ENTIRE traffic
    duration is the route-integrity scored window -- `[0,
    total_traffic_seconds)`, deliberately INCLUDING the pre-step
    `pre_step_traffic_seconds` baseline, for the same reason CORR-44
    already established for `replica_transition`'s own pre-trigger
    baseline (restated here, not silently inherited): the manifest names
    this baseline `pre_step_traffic_seconds`, genuine load-bearing
    experimental traffic that establishes the trajectory the workload step
    is measured against -- never described anywhere as discardable warmup,
    unlike steady-state's own `warmup_window_seconds`. A route-integrity
    failure during `[0,60)` must remain visible, not silently excluded
    from scoring."""
    return (0.0, float(total_traffic_seconds))


def workload_transition_materiality_threshold_seconds(
    total_traffic_seconds: float,
    fraction: float = WORKLOAD_TRANSITION_MATERIALITY_FRACTION,
) -> float:
    """CORR-47: pure arithmetic, mirrors `replica_transition_materiality_
    threshold_seconds()`'s own pattern. For the frozen official 240-second
    workload-transition schedule this evaluates to exactly 60.0 seconds --
    the same numeric value as replica-transition's own 240s-run threshold,
    because both run kinds share the same frozen 240s `total_traffic_
    seconds` and the same duration-relative 0.25 fraction, not because one
    was copied from the other."""
    return fraction * float(total_traffic_seconds)


# CORR-47 sec.5: the manifest's `workload_transitions` section (unlike
# `replica_transitions`) names NO `measured_fields` at all -- there is
# nothing to mark `NOT_FORMALLY_OPERATIONALIZED` on a per-field basis
# without first inventing field names this correction's own governing task
# explicitly forbids inventing. A single blanket status is persisted
# instead, honestly describing this state.
WORKLOAD_TRANSITION_DERIVED_METRICS_STATUS = NOT_FORMALLY_OPERATIONALIZED
WORKLOAD_TRANSITION_DERIVED_METRICS_STATUS_REASON = (
    "The manifest's workload_transitions section names no measured_fields at all "
    "(unlike replica_transitions, which names seven) -- there is no frozen derived "
    "response-lag/settling-time metric name or operational definition for this run "
    "kind anywhere in this project's manifest or protocol documentation (confirmed "
    "by repo-wide search, Phase 15B.6A/15B.6B). Inventing a metric name or an "
    "arbitrary settling threshold/band here would silently create a NEW scientific "
    "definition inside an execution-implementation task, which this project's own "
    "governing instructions explicitly forbid. The raw trajectory (target RPS, "
    "realized request rate, latency, requests-in-progress, CPU, memory, fixed-"
    "replica state) is fully preserved in this attempt's own artifacts, in case a "
    "future phase formally operationalizes a specific derived metric."
)


def workload_transition_phase(elapsed_seconds: float, trigger_at_traffic_elapsed_seconds: float) -> str:
    """Pure classification of one elapsed-time sample into the workload
    step's two phases, matching the schedule's own `pre_step`/`post_step`
    `phase_name` column exactly (CORR-47 sec.9: the workload step is
    schedule-owned -- this function only reproduces, for streams that do
    not themselves carry a `phase_name` column [e.g. `metrics.csv`], the
    same phase boundary the schedule already encodes)."""
    return "pre_step" if elapsed_seconds < trigger_at_traffic_elapsed_seconds else "post_step"


def build_workload_transition_trajectory_rows(
    metrics_rows: Sequence[Dict[str, Any]],
    trigger_at_traffic_elapsed_seconds: float,
    rps_before: float,
    rps_after: float,
    total_traffic_seconds: float,
) -> List[Dict[str, Any]]:
    """CORR-47 sec.13 (boundary-hardened by CORR-48, Phase 15B.6D):
    assembles the workload-transition trajectory artifact by ALIGNING
    already-collected raw streams rather than adding a new expensive live
    Kubernetes/Prometheus poller -- `metrics.csv` (produced by the same
    `collectors/metrics_collector.py` invocation already used, unchanged,
    by every other run kind) already carries every raw signal this run
    kind's trajectory needs in one place: `request_rate_rps`,
    `latency_p50/p95/p99_seconds`, `requests_in_progress`, `cpu_cores`,
    `memory_bytes`, and `replicas_current/desired/ready/available` (needed
    for the fixed-replica invariant, sec.12 -- never a NEW replica-count
    trigger, since none is commanded for this run kind). This function is
    pure and unit-testable without any live cluster: it only reads
    already-loaded row dicts and appends `elapsed_from_workload_step`/
    `phase`/`target_rps` columns, exactly reproducing the schedule's own
    `pre_step`/`post_step` phase boundary (`workload_transition_phase()`).
    Missing/non-numeric raw values are left as `None` -- never fabricated.

    CORR-48 (index38's own first-official-use audit found this): the
    collector processes are deliberately sized for `total_traffic_seconds
    + 30` seconds (observation/cleanup robustness margin) and their raw
    `metrics.csv` LEGITIMATELY contains samples past official traffic end
    -- that raw file is never truncated or altered here. Without this
    function's own `0.0 <= elapsed_from_traffic_start < total_traffic_
    seconds` filter, those post-traffic buffer samples were previously
    admitted into the OFFICIAL scientific trajectory and mislabeled
    `post_step` (index38's own first attempt showed exactly this: 3 of 27
    rows at elapsed 240.0/250.0/260.01, all >= the 240s official traffic
    boundary, incorrectly carrying `phase=post_step`/`target_rps=32.0`).
    The half-open interval excludes a sample landing exactly at
    `total_traffic_seconds` itself -- it belongs to the post-traffic
    buffer, not the closing instant of official traffic."""
    rows: List[Dict[str, Any]] = []
    for row in metrics_rows:
        elapsed = float(row["elapsed_seconds"])
        if not (0.0 <= elapsed < total_traffic_seconds):
            continue
        phase = workload_transition_phase(elapsed, trigger_at_traffic_elapsed_seconds)
        target_rps = rps_before if phase == "pre_step" else rps_after

        def _f(key: str) -> Optional[float]:
            raw = row.get(key)
            if raw in (None, ""):
                return None
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None

        rows.append(
            {
                "elapsed_from_traffic_start": elapsed,
                "elapsed_from_workload_step": round(elapsed - trigger_at_traffic_elapsed_seconds, 3),
                "phase": phase,
                "target_rps": target_rps,
                "request_rate_rps": _f("request_rate_rps"),
                "latency_p50_seconds": _f("latency_p50_seconds"),
                "latency_p95_seconds": _f("latency_p95_seconds"),
                "latency_p99_seconds": _f("latency_p99_seconds"),
                "requests_in_progress": _f("requests_in_progress"),
                "cpu_cores": _f("cpu_cores"),
                "memory_bytes": _f("memory_bytes"),
                "replicas_desired": _f("replicas_desired"),
                "replicas_current": _f("replicas_current"),
                "replicas_ready": _f("replicas_ready"),
                "replicas_available": _f("replicas_available"),
            }
        )
    return rows


def evaluate_fixed_replica_invariant(
    trajectory_rows: Sequence[Dict[str, Any]],
    fixed_replicas: int,
) -> Dict[str, Any]:
    """CORR-47 sec.12: honestly reports whether the fixed-replica count
    held throughout a workload-transition run, WITHOUT inventing a new
    structural-invalidity rule -- the governing task's own instruction is
    explicit: 'report that honestly and evaluate using existing
    structural-validity principles rather than hiding it. Do not invent an
    automatic invalidity rule unless already supported by project
    methodology.' This function only classifies; callers decide (using
    already-existing, unrelated rules such as `hpa_unexpectedly_active`)
    whether any observed drift is itself independently invalidating."""
    desired_values = [r["replicas_desired"] for r in trajectory_rows if r["replicas_desired"] is not None]
    current_values = [r["replicas_current"] for r in trajectory_rows if r["replicas_current"] is not None]
    drift_samples = [
        r for r in trajectory_rows
        if r["replicas_desired"] is not None and r["replicas_desired"] != fixed_replicas
    ]
    return {
        "fixed_replicas_expected": fixed_replicas,
        "desired_replicas_observed_distinct_values": sorted(set(desired_values)),
        "current_replicas_observed_distinct_values": sorted(set(current_values)),
        "fixed_replica_invariant_holds": len(drift_samples) == 0,
        "drift_sample_count": len(drift_samples),
        "drift_first_elapsed_from_traffic_start": drift_samples[0]["elapsed_from_traffic_start"] if drift_samples else None,
    }


def first_post_step_request_elapsed_seconds(requests_rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    """CORR-47 sec.14: purely DESCRIPTIVE evidence -- the earliest
    generator-reported `planned_elapsed_seconds` among requests whose
    `phase_name == 'post_step'`, i.e. the schedule's own first
    post-step-phase request. This is NEVER reinterpreted as a derived
    application-response lag (there is no frozen operational definition
    for one, sec.5) -- it only answers "when did the schedule's own step
    begin issuing requests," which is knowable directly from the
    generator's own request-plan data, independent of any application
    response."""
    candidates = [
        float(r["planned_elapsed_seconds"])
        for r in requests_rows
        if r.get("phase_name") == "post_step" and r.get("planned_elapsed_seconds") not in (None, "")
    ]
    return min(candidates) if candidates else None


def compute_transition_replica_lags(
    trigger_monotonic: float,
    timeline_samples: Sequence[Dict[str, Any]],
    target_replicas: int,
    direction: str,
) -> Dict[str, Optional[float]]:
    """Phase 15B.5B, hardened Phase 15B.5E (CORR-45): the three replica-
    count-trajectory lags whose endpoints are objectively defined by the
    trigger timestamp and by a Kubernetes-API-observed replica count
    crossing the transition's own target -- NOT an invented scientific
    threshold, since "current/ready/available replicas reached the frozen
    target replica count" requires no arbitrary band/tolerance to define.
    This is the governing Phase 15B.5B task's own explicitly-authorized
    fallback definition, used only because no conflicting frozen
    definition exists anywhere in this project's manifest/protocol
    documentation (confirmed by repo-wide search, Phase 15B.5A/15B.5B).

    CORR-45 (Phase 15B.5E) FALSE-POSITIVE FIX: the original Phase 15B.5B
    implementation checked `ready_replicas`/`available_replicas` against
    `target_replicas` COMPLETELY INDEPENDENTLY of whether
    `current_replicas` had itself already reached the target. For a
    scale-DOWN transition (e.g. 2->1), this meant an UNRELATED readiness
    loss on a pod that had not yet been removed (`current_replicas` still
    2, `ready_replicas` dropped to 1 because ONE of the two surviving pods
    became unready) would satisfy `ready_replicas <= 1` and be
    misreported as "the 2->1 scale-down has completed and its surviving
    pod is Ready" -- an artificially short, FALSE lag for a transition
    that had not actually happened yet. `desired_to_ready_lag_seconds`/
    `desired_to_available_lag_seconds` now ALWAYS additionally require
    `current_replicas` to have already reached `target_replicas` at that
    same sample -- i.e. the replica-COUNT transition itself must be
    observed complete before readiness/availability at the target count
    can be credited. `desired_to_current_lag_seconds` is unaffected by
    this fix (it was already checking `current_replicas` alone).

    Direction-specific "reached" conditions (frozen, Phase 15B.5E sec.3):
      scale-up   (target > pre-trigger count): `value >= target_replicas`
                 (Kubernetes may transiently overshoot en route; `>=` is
                 the correct "reached-or-passed" test for an increasing
                 target).
      scale-down (target < pre-trigger count): `value == target_replicas`
                 EXACTLY -- a `<=` test would consider an INTERMEDIATE
                 replica count (e.g. observing 0 mid-restart on the way to
                 a target of 1) as having "reached" the final target,
                 which is not what a scale-down TO a specific count means.

    `timeline_samples` rows must each carry `sample_monotonic` plus
    `current_replicas`/`ready_replicas`/`available_replicas` (ints, or
    `None` if genuinely unobserved at that sample -- never fabricated).
    Only samples at or after `trigger_monotonic` are considered. Returns
    `None` for any lag whose target condition is never observed to be
    satisfied within the given samples -- a short/incomplete observation
    window must reduce evidence, never manufacture it.

    IMPORTANT SEMANTIC NOTE (Phase 15B.5E sec.4): every returned lag is a
    FIRST-OBSERVED TRAJECTORY LAG, not an exact physical Kubernetes-event
    time -- `timeline_samples` is produced by a periodic (nominally 2.0s,
    never guaranteed) poll, so the true underlying state transition could
    have occurred anywhere in the gap between the previous sample and the
    one credited here. Callers/reports must describe this as "first
    observed at +Xs", never as "reached at exactly +Xs", unless an
    independent, authoritative Kubernetes API/event timestamp is also
    available and cited.
    """
    if direction not in (TRANSITION_DIRECTION_SCALE_UP, TRANSITION_DIRECTION_SCALE_DOWN):
        raise ValueError(f"unknown transition direction: {direction!r}")
    is_scale_down = direction == TRANSITION_DIRECTION_SCALE_DOWN
    after_trigger = sorted(
        (s for s in timeline_samples if s.get("sample_monotonic") is not None and s["sample_monotonic"] >= trigger_monotonic),
        key=lambda s: s["sample_monotonic"],
    )

    def _current_reached(value: Optional[int]) -> bool:
        if value is None:
            return False
        return (value == target_replicas) if is_scale_down else (value >= target_replicas)

    def _first_lag(field: Optional[str]) -> Optional[float]:
        """`field=None` computes the `current_replicas`-only lag. A
        non-`None` field ADDITIONALLY requires `current_replicas` to have
        reached the target at that SAME sample (the CORR-45 fix) before
        that field's own target condition is credited."""
        for sample in after_trigger:
            current_value = sample.get("current_replicas")
            if field is None:
                target_value = current_value
            else:
                if not _current_reached(current_value):
                    continue  # CORR-45: replica-count transition itself not yet observed complete
                target_value = sample.get(field)
            if target_value is None:
                continue
            reached = (target_value == target_replicas) if is_scale_down else (target_value >= target_replicas)
            if reached:
                return round(sample["sample_monotonic"] - trigger_monotonic, 3)
        return None

    return {
        "desired_to_current_lag_seconds": _first_lag(None),
        "desired_to_ready_lag_seconds": _first_lag("ready_replicas"),
        "desired_to_available_lag_seconds": _first_lag("available_replicas"),
    }


def compute_transition_replica_lag_uncertainty(
    trigger_monotonic: float,
    timeline_samples: Sequence[Dict[str, Any]],
    target_replicas: int,
    direction: str,
) -> Dict[str, Dict[str, Optional[float]]]:
    """CORR-45 (Phase 15B.5E): per-lag measurement-granularity metadata,
    complementing (never replacing) `compute_transition_replica_lags()`'s
    existing first-observed values. For each of the three lags, reports
    the elapsed-from-trigger time of the sample immediately BEFORE the
    first satisfying sample (`None` if the very first post-trigger sample
    already satisfies it, or if no post-trigger samples exist at all
    before it), the first satisfying sample's own elapsed-from-trigger
    time (duplicating `compute_transition_replica_lags()`'s own value,
    for convenience), and the width of the observation interval between
    them -- i.e. how much sampling-cadence slack surrounds the reported
    lag. Never fabricates a previous-sample time when none exists."""
    if direction not in (TRANSITION_DIRECTION_SCALE_UP, TRANSITION_DIRECTION_SCALE_DOWN):
        raise ValueError(f"unknown transition direction: {direction!r}")
    is_scale_down = direction == TRANSITION_DIRECTION_SCALE_DOWN
    after_trigger = sorted(
        (s for s in timeline_samples if s.get("sample_monotonic") is not None and s["sample_monotonic"] >= trigger_monotonic),
        key=lambda s: s["sample_monotonic"],
    )

    def _current_reached(value: Optional[int]) -> bool:
        if value is None:
            return False
        return (value == target_replicas) if is_scale_down else (value >= target_replicas)

    def _uncertainty_for(field: Optional[str]) -> Dict[str, Optional[float]]:
        for i, sample in enumerate(after_trigger):
            current_value = sample.get("current_replicas")
            if field is None:
                target_value = current_value
            else:
                if not _current_reached(current_value):
                    continue
                target_value = sample.get(field)
            if target_value is None:
                continue
            reached = (target_value == target_replicas) if is_scale_down else (target_value >= target_replicas)
            if reached:
                first_elapsed = round(sample["sample_monotonic"] - trigger_monotonic, 3)
                previous_elapsed = round(after_trigger[i - 1]["sample_monotonic"] - trigger_monotonic, 3) if i > 0 else None
                width = round(first_elapsed - previous_elapsed, 3) if previous_elapsed is not None else None
                return {
                    "previous_sample_elapsed_from_trigger": previous_elapsed,
                    "first_target_sample_elapsed_from_trigger": first_elapsed,
                    "observation_interval_width_seconds": width,
                }
        return {
            "previous_sample_elapsed_from_trigger": None,
            "first_target_sample_elapsed_from_trigger": None,
            "observation_interval_width_seconds": None,
        }

    return {
        "desired_to_current_lag_seconds": _uncertainty_for(None),
        "desired_to_ready_lag_seconds": _uncertainty_for("ready_replicas"),
        "desired_to_available_lag_seconds": _uncertainty_for("available_replicas"),
    }


def summarize_replica_transition_trajectory_gaps(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """CORR-45 (Phase 15B.5E): pure summary statistics over a replica-
    transition trajectory sample stream's ACTUAL inter-sample gaps
    (consecutive differences of `elapsed_from_traffic_start`), mirroring
    `summarize_dataplane_acquisition_gaps()`'s own pattern and rationale
    -- a nominal 2.0s sampling interval
    (`DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7`, reused for the
    trajectory sampler too) is NOT a guaranteed acquisition cadence; index30's
    own live trajectory (65 samples over ~247.1s, averaging ~3.86s/sample
    against a 2.0s nominal configuration) is direct empirical proof of
    this. Pure observability -- never feeds into any lag/classification
    computation; a large gap can only ever reduce the temporal resolution
    at which a lag was actually observed (reflected in
    `compute_transition_replica_lag_uncertainty()`'s own
    `observation_interval_width_seconds`), never fabricate a false one.
    Returns `n=0` and all other fields `None` if fewer than two samples
    are available (no gap can be computed from 0 or 1 samples)."""
    elapsed = sorted(
        s["elapsed_from_traffic_start"]
        for s in samples
        if s.get("elapsed_from_traffic_start") is not None
    )
    gaps = [round(elapsed[i] - elapsed[i - 1], 4) for i in range(1, len(elapsed))]
    n = len(gaps)
    if n == 0:
        return {
            "trajectory_sample_count": len(elapsed),
            "gap_count": 0,
            "gap_min_seconds": None,
            "gap_median_seconds": None,
            "gap_mean_seconds": None,
            "gap_p95_seconds": None,
            "gap_p99_seconds": None,
            "gap_max_seconds": None,
            "count_gap_gt_3_seconds": 0,
            "nominal_sampling_interval_seconds": DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7,
            "observed_sampling_interval_is_not_guaranteed": True,
        }

    def _percentile(sorted_values: List[float], pct: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        rank = pct * (len(sorted_values) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(sorted_values) - 1)
        frac = rank - lo
        return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac

    sorted_gaps = sorted(gaps)
    return {
        "trajectory_sample_count": len(elapsed),
        "gap_count": n,
        "gap_min_seconds": sorted_gaps[0],
        "gap_median_seconds": _percentile(sorted_gaps, 0.5),
        "gap_mean_seconds": sum(sorted_gaps) / n,
        "gap_p95_seconds": _percentile(sorted_gaps, 0.95),
        "gap_p99_seconds": _percentile(sorted_gaps, 0.99),
        "gap_max_seconds": sorted_gaps[-1],
        "count_gap_gt_3_seconds": sum(1 for g in sorted_gaps if g > 3.0),
        "nominal_sampling_interval_seconds": DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7,
        "observed_sampling_interval_is_not_guaranteed": True,
    }


def build_transition_metrics_summary(
    trigger_monotonic: float,
    timeline_samples: Sequence[Dict[str, Any]],
    target_replicas: int,
    direction: str,
) -> Dict[str, Any]:
    """Assembles the manifest's full `measured_fields` list: the three
    objectively-defined replica lags (computed via
    `compute_transition_replica_lags`, CORR-45-hardened against the
    scale-down false-positive described in that function's own
    docstring), plus the four fields that have NO frozen settling-
    threshold definition anywhere in this project's documentation,
    explicitly marked `NOT_FORMALLY_OPERATIONALIZED` rather than silently
    invented -- see this module's own
    `TRANSITION_MEASURED_FIELDS_NOT_FORMALLY_OPERATIONALIZED` docstring.
    Raw time series for these four remain fully preserved in the
    attempt's own `replica_transition_timeline.csv`/`metrics.csv`/
    `requests.csv` artifacts -- nothing is discarded, only the FORMAL
    settling-lag computation is withheld pending a frozen definition.

    CORR-45 additions: `lag_measurement_semantic` states explicitly that
    every lag is a FIRST-OBSERVED TRAJECTORY LAG, never an exact physical
    event time; `lag_uncertainty` carries each lag's own observation-
    interval-width metadata (`compute_transition_replica_lag_uncertainty`);
    `trajectory_gap_statistics` carries the sampling-quality metadata from
    `summarize_replica_transition_trajectory_gaps`."""
    lags = compute_transition_replica_lags(trigger_monotonic, timeline_samples, target_replicas, direction)
    result: Dict[str, Any] = dict(lags)
    for field in TRANSITION_MEASURED_FIELDS_NOT_FORMALLY_OPERATIONALIZED:
        result[field] = NOT_FORMALLY_OPERATIONALIZED
    result["not_formally_operationalized_reason"] = (
        "No settling threshold/band/window is defined for these fields anywhere in the "
        "project's frozen manifest or protocol documentation (confirmed by repo-wide search, "
        "Phase 15B.5A/15B.5B); raw CPU/latency/in-progress/request-rate time series remain "
        "fully preserved in this attempt's own artifacts for future analysis once such a "
        "definition is frozen."
    )
    result["lag_measurement_semantic"] = (
        "FIRST_OBSERVED_TRAJECTORY_LAG -- each non-null lag above is the elapsed-from-trigger "
        "time of the FIRST sampled observation satisfying its target condition, not an exact "
        "physical Kubernetes-event timestamp. The true underlying state transition could have "
        "occurred anywhere between the previous sample and this one -- see "
        "`lag_uncertainty` for each lag's own observation-interval width. Report this as "
        "'first observed at +Xs', never as 'reached at exactly +Xs', unless independently "
        "corroborated by an authoritative Kubernetes API/event timestamp (CORR-45, Phase 15B.5E)."
    )
    result["lag_uncertainty"] = compute_transition_replica_lag_uncertainty(trigger_monotonic, timeline_samples, target_replicas, direction)
    result["trajectory_gap_statistics"] = summarize_replica_transition_trajectory_gaps(timeline_samples)
    return result
