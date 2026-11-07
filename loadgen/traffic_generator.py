"""Host-side open-loop traffic generator for Kubernetes autoscaling experiments.

Dispatches requests according to deterministic or Poisson planned arrival schedules,
classifies individual request outcomes, enforces bounded client concurrency with
generator-side overload dropping, and writes authoritative persistent logs to
persistent-data/traffic/<experiment-id>/.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml

# V2: mirrors app/src/workload.py's VALID_RESOURCE_MODES. Duplicated
# deliberately rather than imported -- the load generator and the demo app
# are separate deployables (loadgen runs host-side, app runs in-cluster)
# with no shared dependency today, and this is a 3-value literal tuple, not
# logic worth coupling the two processes over.
VALID_RESOURCE_MODES = ("cpu", "memory", "mixed")

# V2 Phase 1.1: mirrors app/src/workload.py's DEFAULT_MEMORY_HOLD_MS. A
# legacy profile never sets memory_hold_ms, so this default only matters
# once resource_mode/memory_mb already make a profile non-default (see
# _dispatch_single_request's conditional URL construction below).
DEFAULT_MEMORY_HOLD_MS = 500


class ProfileParseError(Exception):
    """Raised when traffic profile YAML cannot be parsed."""
    pass


class PlannedRequest:
    """Represents a planned request arrival in the authoritative schedule."""

    def __init__(
        self,
        request_index: int,
        planned_elapsed_seconds: float,
        phase_name: str,
        phase_type: str,
        target_rps: float,
        configured_intensity: int,
        endpoint: str = "/work",
    ) -> None:
        self.request_index = request_index
        self.planned_elapsed_seconds = planned_elapsed_seconds
        self.phase_name = phase_name
        self.phase_type = phase_type
        self.target_rps = target_rps
        self.configured_intensity = configured_intensity
        self.endpoint = endpoint

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_index": self.request_index,
            "planned_elapsed_seconds": f"{self.planned_elapsed_seconds:.6f}",
            "phase_name": self.phase_name,
            "phase_type": self.phase_type,
            "target_rps": f"{self.target_rps:.4f}",
            "configured_intensity": self.configured_intensity,
            "endpoint": self.endpoint,
        }


class Phase:
    """Base class for traffic profile phases."""

    def __init__(self, name: str, phase_type: str, duration_seconds: float) -> None:
        self.name = name
        self.phase_type = phase_type
        self.duration_seconds = float(duration_seconds)

    def get_target_rps(self, t_rel: float) -> float:
        raise NotImplementedError


class ConstantPhase(Phase):
    def __init__(self, name: str, duration_seconds: float, target_rps: float) -> None:
        super().__init__(name, "constant", duration_seconds)
        self.target_rps = float(target_rps)

    def get_target_rps(self, t_rel: float) -> float:
        return self.target_rps


class RampPhase(Phase):
    def __init__(self, name: str, duration_seconds: float, start_rps: float, end_rps: float) -> None:
        super().__init__(name, "ramp", duration_seconds)
        self.start_rps = float(start_rps)
        self.end_rps = float(end_rps)

    def get_target_rps(self, t_rel: float) -> float:
        if self.duration_seconds <= 0:
            return self.end_rps
        ratio = min(1.0, max(0.0, t_rel / self.duration_seconds))
        return self.start_rps + ratio * (self.end_rps - self.start_rps)


class StepPhase(Phase):
    def __init__(self, name: str, duration_seconds: float, steps: List[Dict[str, Any]]) -> None:
        super().__init__(name, "step", duration_seconds)
        self.steps = sorted(
            [(float(s.get("at_second", 0)), float(s.get("rps", 1.0))) for s in steps],
            key=lambda x: x[0],
        )

    def get_target_rps(self, t_rel: float) -> float:
        current_rps = self.steps[0][1] if self.steps else 1.0
        for at_sec, rps in self.steps:
            if at_sec <= t_rel:
                current_rps = rps
            else:
                break
        return current_rps


class BurstPhase(Phase):
    def __init__(
        self,
        name: str,
        duration_seconds: float,
        baseline_rps: float,
        burst_rps: float,
        burst_duration_seconds: float,
        inter_burst_interval_seconds: float,
    ) -> None:
        super().__init__(name, "burst", duration_seconds)
        self.baseline_rps = float(baseline_rps)
        self.burst_rps = float(burst_rps)
        self.burst_duration_seconds = float(burst_duration_seconds)
        self.inter_burst_interval_seconds = float(inter_burst_interval_seconds)

    def get_target_rps(self, t_rel: float) -> float:
        cycle_len = self.burst_duration_seconds + self.inter_burst_interval_seconds
        if cycle_len <= 0:
            return self.burst_rps
        t_in_cycle = t_rel % cycle_len
        if t_in_cycle < self.burst_duration_seconds:
            return self.burst_rps
        return self.baseline_rps


class PeriodicPhase(Phase):
    def __init__(
        self,
        name: str,
        duration_seconds: float,
        mean_rps: float,
        amplitude_rps: float,
        period_seconds: float,
        noise_std_rps: float = 0.0,
    ) -> None:
        super().__init__(name, "periodic", duration_seconds)
        self.mean_rps = float(mean_rps)
        self.amplitude_rps = float(amplitude_rps)
        self.period_seconds = float(period_seconds)
        self.noise_std_rps = float(noise_std_rps)

    def get_target_rps(self, t_rel: float) -> float:
        if self.period_seconds <= 0:
            return self.mean_rps
        sin_val = math.sin(2.0 * math.pi * t_rel / self.period_seconds)
        rate = self.mean_rps + self.amplitude_rps * sin_val
        return max(0.1, rate)


class TrafficProfile:
    """Unified representation of a multi-phase traffic workload."""

    def __init__(
        self,
        profile_name: str,
        description: str,
        seed: int,
        endpoint: str,
        intensity: int,
        phases: List[Phase],
        resource_mode: str = "cpu",
        memory_mb: int = 0,
        memory_hold_ms: int = DEFAULT_MEMORY_HOLD_MS,
        memory_safe_max_in_flight: Optional[int] = None,
    ) -> None:
        self.profile_name = profile_name
        self.description = description
        self.seed = seed
        self.endpoint = endpoint
        self.intensity = intensity
        self.phases = phases
        self.total_duration = sum(p.duration_seconds for p in phases)
        # V2 Phase 2.1 PART 3: an OPTIONAL, profile-specific concurrency
        # ceiling that applies ONLY to memory/mixed-mode profiles (see
        # TrafficGenerator.__init__'s `effective_max_in_flight`). The
        # generator's own general `--max-in-flight` (default 50) is a
        # CPU-era concurrency safety valve unrelated to memory sizing; at
        # MAX_WORK_MEMORY_MB=16 (app/src/workload.py), a true 50-way
        # concurrent memory/mixed burst would reach ~800MB, far exceeding
        # the pod's 256Mi limit (V2 Phase 1.1's disclosed trade-off). This
        # field lets a V2 memory/mixed profile declare its OWN safe
        # concurrency bound WITHOUT lowering `--max-in-flight` globally
        # (which would also needlessly constrain CPU-only profiles). None
        # (the default) means "no additional V2-specific cap" -- every
        # pre-V2 and CPU-only profile is completely unaffected.
        self.memory_safe_max_in_flight = memory_safe_max_in_flight
        # V2 additions: whole-profile resource dimension, forwarded as extra
        # /work query params (see _dispatch_single_request) ONLY when the
        # profile is genuinely non-default. Defaults ("cpu", 0, DEFAULT_
        # MEMORY_HOLD_MS) are IDENTICAL to V1's only behavior, so any
        # existing profile that omits these fields is completely unaffected
        # -- not just numerically, but at the literal dispatched URL level
        # (V2 Phase 1.1 legacy-compatibility requirement).
        self.resource_mode = resource_mode
        self.memory_mb = memory_mb
        self.memory_hold_ms = memory_hold_ms

    def get_phase_and_target_rps(self, elapsed: float) -> Tuple[float, str, str]:
        """Determine target RPS and active phase at any elapsed experiment time."""
        if not self.phases:
            return 1.0, "default", "constant"

        if elapsed >= self.total_duration:
            last = self.phases[-1]
            return last.get_target_rps(last.duration_seconds), last.name, last.phase_type

        acc = 0.0
        for p in self.phases:
            if elapsed < acc + p.duration_seconds:
                t_rel = elapsed - acc
                return max(0.1, p.get_target_rps(t_rel)), p.name, p.phase_type
            acc += p.duration_seconds

        last = self.phases[-1]
        return max(0.1, last.get_target_rps(last.duration_seconds)), last.name, last.phase_type


def parse_traffic_profile(yaml_path: Path) -> TrafficProfile:
    """Parse unified or legacy traffic profile YAML into a TrafficProfile instance."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"Traffic profile not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    profile_name = data.get("profile", yaml_path.stem)
    description = data.get("description", "")
    seed = int(data.get("seed", 42))
    endpoint = data.get("endpoint", "/work")
    intensity = int(data.get("intensity", 3))
    # V2: optional, additive fields. Omitted entirely by every existing
    # (pre-V2) profile file, so `.get(..., "cpu")`/`.get(..., 0)` reproduce
    # V1 behavior exactly for them -- validated, never silently coerced.
    resource_mode = data.get("resource_mode", "cpu")
    if resource_mode not in VALID_RESOURCE_MODES:
        raise ProfileParseError(
            f"Invalid resource_mode {resource_mode!r} in {yaml_path}; must be one of {VALID_RESOURCE_MODES}"
        )
    memory_mb = int(data.get("memory_mb", 0))
    if memory_mb < 0:
        raise ProfileParseError(f"memory_mb must be >= 0 in {yaml_path}, got {memory_mb}")
    memory_hold_ms = int(data.get("memory_hold_ms", DEFAULT_MEMORY_HOLD_MS))
    if memory_hold_ms < 0:
        raise ProfileParseError(f"memory_hold_ms must be >= 0 in {yaml_path}, got {memory_hold_ms}")
    memory_safe_max_in_flight = data.get("memory_safe_max_in_flight")
    if memory_safe_max_in_flight is not None:
        memory_safe_max_in_flight = int(memory_safe_max_in_flight)
        if memory_safe_max_in_flight <= 0:
            raise ProfileParseError(
                f"memory_safe_max_in_flight must be > 0 in {yaml_path}, got {memory_safe_max_in_flight}"
            )

    phases: List[Phase] = []

    if "phases" in data and isinstance(data["phases"], list):
        # Unified phase schema
        for idx, p_data in enumerate(data["phases"]):
            p_type = p_data.get("type", "constant")
            p_name = p_data.get("name", f"phase_{idx}_{p_type}")
            duration = float(p_data.get("duration_seconds", 60.0))

            if p_type == "constant":
                rps = float(p_data.get("target_rps", p_data.get("rps", 10.0)))
                phases.append(ConstantPhase(p_name, duration, rps))
            elif p_type == "ramp":
                start_rps = float(p_data.get("start_rps", 5.0))
                end_rps = float(p_data.get("end_rps", 20.0))
                phases.append(RampPhase(p_name, duration, start_rps, end_rps))
            elif p_type == "step":
                steps = p_data.get("steps", [])
                if not steps and "rps" in p_data:
                    steps = [{"at_second": 0, "rps": p_data["rps"]}]
                phases.append(StepPhase(p_name, duration, steps))
            elif p_type == "burst":
                phases.append(
                    BurstPhase(
                        name=p_name,
                        duration_seconds=duration,
                        baseline_rps=float(p_data.get("baseline_rps", 5.0)),
                        burst_rps=float(p_data.get("burst_rps", 50.0)),
                        burst_duration_seconds=float(p_data.get("burst_duration_seconds", 20.0)),
                        inter_burst_interval_seconds=float(p_data.get("inter_burst_interval_seconds", 60.0)),
                    )
                )
            elif p_type == "periodic":
                phases.append(
                    PeriodicPhase(
                        name=p_name,
                        duration_seconds=duration,
                        mean_rps=float(p_data.get("mean_rps", 20.0)),
                        amplitude_rps=float(p_data.get("amplitude_rps", 10.0)),
                        period_seconds=float(p_data.get("period_seconds", 120.0)),
                        noise_std_rps=float(p_data.get("noise_std_rps", 0.0)),
                    )
                )
            else:
                raise ProfileParseError(f"Unsupported phase type '{p_type}' in {yaml_path}")
    elif data.get("profile") == "constant" or "target_rps" in data:
        # Legacy constant
        dur = float(data.get("duration_seconds", 600.0))
        rps = float(data.get("target_rps", 20.0))
        phases.append(ConstantPhase("constant", dur, rps))
    elif "steps" in data:
        # Phase16Z fix: legacy flat "step" schema (configs/traffic_profiles/
        # step.yaml) -- top-level `steps: [{at_second, rps}, ...]` plus a
        # top-level `duration_seconds` for the phase's total span. Converted
        # into the SAME StepPhase already used by the unified `phases:`
        # schema, with no change to the intended timing.
        dur = float(data.get("duration_seconds", 60.0))
        phases.append(StepPhase("step", dur, data["steps"]))
    elif "baseline_rps" in data and "burst_rps" in data:
        # Phase16Z fix: legacy flat "burst" schema (configs/traffic_profiles/
        # burst.yaml) -- top-level baseline/burst/duration/interval fields,
        # a single BurstPhase spanning the full declared duration_seconds
        # (BurstPhase.get_target_rps() already cycles baseline/burst
        # internally from elapsed time modulo the cycle length, so no
        # explicit burst count is needed -- `num_bursts` in the YAML is
        # descriptive metadata only, unused here exactly as it is unused by
        # the unified-schema BurstPhase branch above).
        dur = float(data.get("duration_seconds", 60.0))
        phases.append(
            BurstPhase(
                name="burst",
                duration_seconds=dur,
                baseline_rps=float(data.get("baseline_rps", 5.0)),
                burst_rps=float(data.get("burst_rps", 50.0)),
                burst_duration_seconds=float(data.get("burst_duration_seconds", 20.0)),
                inter_burst_interval_seconds=float(data.get("inter_burst_interval_seconds", 60.0)),
            )
        )
    elif "ramp_up_start_rps" in data:
        # Phase16Z fix: legacy flat "ramp" schema (configs/traffic_profiles/
        # ramp.yaml) -- top-level ramp-up/hold/ramp-down parameters,
        # translated into three phases using the SAME RampPhase/ConstantPhase
        # classes already used by the unified schema (no new phase type
        # introduced). NOTE (disclosed, not silently reconciled): this
        # file's own three sub-durations (`ramp_up_seconds` +
        # `hold_seconds` + `ramp_down_seconds`) sum to less than its
        # top-level `duration_seconds` field for ramp.yaml specifically
        # (400+200+400=1000s vs. a declared 1200s) -- a pre-existing
        # inconsistency in the frozen YAML's own content, not introduced or
        # corrected here. The EFFECTIVE total duration is the sum of the
        # phases actually described (1000s for ramp.yaml), matching
        # TrafficProfile.total_duration's own definition (sum of phase
        # durations) -- see task_outputs/phase_16z_six_profile_semantic_
        # validation.md for the full disclosure.
        ramp_up_seconds = float(data.get("ramp_up_seconds", 60.0))
        hold_seconds = float(data.get("hold_seconds", 0.0))
        ramp_down_seconds = float(data.get("ramp_down_seconds", 60.0))
        phases.append(RampPhase(
            "ramp_up", ramp_up_seconds,
            float(data.get("ramp_up_start_rps", 5.0)), float(data.get("ramp_up_peak_rps", 20.0)),
        ))
        if hold_seconds > 0:
            phases.append(ConstantPhase("hold", hold_seconds, float(data.get("ramp_up_peak_rps", 20.0))))
        phases.append(RampPhase(
            "ramp_down", ramp_down_seconds,
            float(data.get("ramp_up_peak_rps", 20.0)), float(data.get("ramp_down_end_rps", 5.0)),
        ))
    elif "mean_rps" in data and "amplitude_rps" in data:
        # Phase16Z fix: legacy flat "periodic" schema (configs/traffic_profiles/
        # periodic.yaml) -- top-level sinusoidal parameters, a single
        # PeriodicPhase spanning the full declared duration_seconds, using
        # the SAME PeriodicPhase class already used by the unified schema.
        dur = float(data.get("duration_seconds", 60.0))
        phases.append(
            PeriodicPhase(
                name="periodic",
                duration_seconds=dur,
                mean_rps=float(data.get("mean_rps", 20.0)),
                amplitude_rps=float(data.get("amplitude_rps", 10.0)),
                period_seconds=float(data.get("period_seconds", 120.0)),
                noise_std_rps=float(data.get("noise_std_rps", 0.0)),
            )
        )
    else:
        # Fallback single constant phase
        dur = float(data.get("duration_seconds", 60.0))
        phases.append(ConstantPhase("default", dur, 2.0))

    return TrafficProfile(
        profile_name=profile_name,
        description=description,
        seed=seed,
        endpoint=endpoint,
        intensity=intensity,
        phases=phases,
        resource_mode=resource_mode,
        memory_mb=memory_mb,
        memory_hold_ms=memory_hold_ms,
        memory_safe_max_in_flight=memory_safe_max_in_flight,
    )


def _sha256_file(path: Path) -> str:
    """Compute a SHA-256 hex digest of a file's contents for schedule integrity checks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ScheduleValidationError(Exception):
    """Raised when a persisted schedule file is malformed or structurally invalid."""
    pass


def load_planned_schedule(csv_path: Path) -> List[PlannedRequest]:
    """Load a previously persisted traffic_schedule.csv back into PlannedRequest objects.

    This is the exact-replay counterpart to generate_arrival_schedule(): it never
    re-rolls randomness and never alters planned arrival timestamps, phase names,
    intensity, or target RPS metadata -- it reproduces the schedule exactly as
    persisted. Used so multiple HPA-target candidates (Phase 11C) can be driven
    by the bit-identical planned arrival schedule rather than independently
    regenerated schedules that merely share a seed.

    Raises ScheduleValidationError on any structural problem, rather than
    silently repairing or dropping planned arrivals.
    """
    if not csv_path.exists():
        raise ScheduleValidationError(f"Schedule file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ScheduleValidationError(f"Schedule file is empty or has no header: {csv_path}")
        missing = [c for c in SCHEDULE_CSV_FIELDNAMES if c not in reader.fieldnames]
        if missing:
            raise ScheduleValidationError(
                f"Schedule file {csv_path} is missing required column(s): {missing}"
            )
        rows = list(reader)

    if not rows:
        raise ScheduleValidationError(f"Schedule file {csv_path} contains a header but no rows.")

    schedule: List[PlannedRequest] = []
    seen_indices = set()
    prev_elapsed = -1.0

    for line_no, row in enumerate(rows, start=2):  # header is line 1
        try:
            request_index = int(row["request_index"])
            planned_elapsed_seconds = float(row["planned_elapsed_seconds"])
            phase_name = row["phase_name"]
            phase_type = row["phase_type"]
            target_rps = float(row["target_rps"])
            configured_intensity = int(row["configured_intensity"])
            endpoint = row["endpoint"]
        except (KeyError, ValueError, TypeError) as exc:
            raise ScheduleValidationError(
                f"Schedule file {csv_path}, line {line_no}: malformed row ({exc})"
            ) from exc

        if request_index in seen_indices:
            raise ScheduleValidationError(
                f"Schedule file {csv_path}, line {line_no}: duplicate request_index {request_index}"
            )
        seen_indices.add(request_index)

        if planned_elapsed_seconds < prev_elapsed:
            raise ScheduleValidationError(
                f"Schedule file {csv_path}, line {line_no}: planned_elapsed_seconds "
                f"{planned_elapsed_seconds} is not monotonically non-decreasing "
                f"(previous was {prev_elapsed})"
            )
        prev_elapsed = planned_elapsed_seconds

        schedule.append(
            PlannedRequest(
                request_index=request_index,
                planned_elapsed_seconds=planned_elapsed_seconds,
                phase_name=phase_name,
                phase_type=phase_type,
                target_rps=target_rps,
                configured_intensity=configured_intensity,
                endpoint=endpoint,
            )
        )

    return schedule


def generate_arrival_schedule(
    profile: TrafficProfile,
    seed: int,
    arrival_process: str = "poisson",
) -> List[PlannedRequest]:
    """Generate a deterministic planned arrival schedule.

    Outputs PlannedRequest records bounded strictly by profile.total_duration.
    Two runs with identical inputs produce identical planned elapsed timestamps.
    """
    if arrival_process not in ("poisson", "deterministic", "even"):
        raise ValueError(f"Unknown arrival process '{arrival_process}'. Expected 'poisson' or 'deterministic'.")

    schedule: List[PlannedRequest] = []
    total_duration = profile.total_duration

    if arrival_process in ("deterministic", "even"):
        t = 0.0
        idx = 0
        while t < total_duration:
            t_rounded = round(t, 6)
            if t_rounded >= total_duration:
                break
            target_rps, p_name, p_type = profile.get_phase_and_target_rps(t)
            rate = max(0.01, target_rps)
            schedule.append(
                PlannedRequest(
                    request_index=idx,
                    planned_elapsed_seconds=t_rounded,
                    phase_name=p_name,
                    phase_type=p_type,
                    target_rps=round(rate, 4),
                    configured_intensity=profile.intensity,
                    endpoint=profile.endpoint,
                )
            )
            idx += 1
            interval = 1.0 / rate
            t += interval

    elif arrival_process == "poisson":
        rng = random.Random(seed)
        t = 0.0
        idx = 0

        # Draw first arrival interval
        target_rps, _, _ = profile.get_phase_and_target_rps(0.0)
        rate = max(0.01, target_rps)
        u = max(1e-12, min(1.0 - 1e-12, rng.random()))
        interval = -math.log(1.0 - u) / rate
        t += interval

        while t < total_duration:
            t_rounded = round(t, 6)
            if t_rounded >= total_duration:
                break
            target_rps, p_name, p_type = profile.get_phase_and_target_rps(t)
            rate = max(0.01, target_rps)
            schedule.append(
                PlannedRequest(
                    request_index=idx,
                    planned_elapsed_seconds=t_rounded,
                    phase_name=p_name,
                    phase_type=p_type,
                    target_rps=round(rate, 4),
                    configured_intensity=profile.intensity,
                    endpoint=profile.endpoint,
                )
            )
            idx += 1
            u = max(1e-12, min(1.0 - 1e-12, rng.random()))
            interval = -math.log(1.0 - u) / rate
            t += interval

    return schedule


class CompletedRequest:
    """Record of an individual request dispatch and outcome."""

    def __init__(
        self,
        request_index: int,
        planned_elapsed_seconds: float,
        actual_dispatch_elapsed_seconds: Optional[float],
        dispatch_lag_seconds: Optional[float],
        completion_elapsed_seconds: Optional[float],
        configured_intensity: int,
        phase_name: str,
        target_rps: float,
        outcome: str,
        http_status: Optional[int],
        client_latency_seconds: Optional[float],
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        self.request_index = request_index
        self.planned_elapsed_seconds = planned_elapsed_seconds
        self.actual_dispatch_elapsed_seconds = actual_dispatch_elapsed_seconds
        self.dispatch_lag_seconds = dispatch_lag_seconds
        self.completion_elapsed_seconds = completion_elapsed_seconds
        self.configured_intensity = configured_intensity
        self.phase_name = phase_name
        self.target_rps = target_rps
        self.outcome = outcome
        self.http_status = http_status
        self.client_latency_seconds = client_latency_seconds
        # Phase 15B.3M.2 (CORR-39 resolution): diagnostic-only fields, never
        # consulted for outcome classification or timeout behavior -- purely
        # so a future connection_error/timeout is no longer scientifically
        # unattributable. Sanitized/truncated: never expected to carry
        # credentials (this generator only ever talks to its own configured
        # local target URL), but truncation bounds any pathological message.
        self.error_type = error_type
        self.error_message = (error_message[:200] if error_message else error_message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_index": self.request_index,
            "planned_elapsed_seconds": f"{self.planned_elapsed_seconds:.6f}",
            "actual_dispatch_elapsed_seconds": (
                f"{self.actual_dispatch_elapsed_seconds:.6f}" if self.actual_dispatch_elapsed_seconds is not None else ""
            ),
            "dispatch_lag_seconds": (
                f"{self.dispatch_lag_seconds:.6f}" if self.dispatch_lag_seconds is not None else ""
            ),
            "completion_elapsed_seconds": (
                f"{self.completion_elapsed_seconds:.6f}" if self.completion_elapsed_seconds is not None else ""
            ),
            "configured_intensity": self.configured_intensity,
            "phase_name": self.phase_name,
            "target_rps": f"{self.target_rps:.4f}",
            "outcome": self.outcome,
            "http_status": str(self.http_status) if self.http_status is not None else "",
            "client_latency_seconds": (
                f"{self.client_latency_seconds:.6f}" if self.client_latency_seconds is not None else ""
            ),
            "error_type": self.error_type or "",
            "error_message": self.error_message or "",
        }


REQUESTS_CSV_FIELDNAMES = [
    "request_index",
    "planned_elapsed_seconds",
    "actual_dispatch_elapsed_seconds",
    "dispatch_lag_seconds",
    "completion_elapsed_seconds",
    "configured_intensity",
    "phase_name",
    "target_rps",
    "outcome",
    "http_status",
    "client_latency_seconds",
    "error_type",
    "error_message",
]

# V2 note: resource_mode/memory_mb are deliberately NOT added to
# PlannedRequest/this schedule's persisted columns. They are a
# whole-profile setting (TrafficProfile.resource_mode/.memory_mb), not a
# per-request arrival-schedule field, and this CSV's exact column set is
# load-bearing for the frozen V1 schedule-hash verification pipeline
# (traffic_orchestrator.py / load_planned_schedule's exact-replay check) --
# changing it here would risk invalidating that unrelated, frozen contract.
SCHEDULE_CSV_FIELDNAMES = [
    "request_index",
    "planned_elapsed_seconds",
    "phase_name",
    "phase_type",
    "target_rps",
    "configured_intensity",
    "endpoint",
]

SUMMARY_CSV_FIELDNAMES = [
    "elapsed_seconds",
    "phase_name",
    "target_rps",
    "offered_requests",
    "sent_requests",
    "success_2xx",
    "client_error_4xx",
    "server_error_5xx",
    "timeout",
    "connection_error",
    "dropped_overload",
    "achieved_send_rps",
]


class TrafficGenerator:
    """Asynchronous open-loop traffic generator."""

    def __init__(
        self,
        experiment_id: str,
        profile_path: str,
        target_url: str = "http://localhost:8080",
        arrival_process: str = "poisson",
        seed: Optional[int] = None,
        max_in_flight: int = 50,
        timeout_seconds: float = 5.0,
        output_base_dir: str = "persistent-data/traffic",
        project_root: Optional[Path] = None,
        replay_schedule_path: Optional[str] = None,
        connection_mode: str = "keepalive",
    ) -> None:
        self.experiment_id = experiment_id
        self.target_url = target_url.rstrip("/")
        self.arrival_process = arrival_process
        self.max_in_flight = max_in_flight
        self.timeout_seconds = timeout_seconds
        self.replay_schedule_path = replay_schedule_path

        if connection_mode not in ("keepalive", "fresh"):
            raise ValueError(f"Unknown connection_mode '{connection_mode}'. Expected 'keepalive' or 'fresh'.")
        self.connection_mode = connection_mode

        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = project_root

        self.profile = parse_traffic_profile(self.project_root / profile_path)
        self.seed = seed if seed is not None else self.profile.seed

        # V2 Phase 2.1 PART 3: a memory/mixed profile's own declared safety
        # ceiling (if any) NEVER raises the general `--max-in-flight`
        # concurrency bound -- it can only tighten it, and only for THIS
        # profile. A CPU-only profile (memory_safe_max_in_flight=None, the
        # default) is completely unaffected: effective_max_in_flight ==
        # max_in_flight exactly, byte-for-byte the pre-V2 behavior.
        if (
            self.profile.resource_mode in ("memory", "mixed")
            and self.profile.memory_safe_max_in_flight is not None
        ):
            self.effective_max_in_flight = min(self.max_in_flight, self.profile.memory_safe_max_in_flight)
        else:
            self.effective_max_in_flight = self.max_in_flight

        self.output_dir = self.project_root / output_base_dir / self.experiment_id
        self.schedule_path = self.output_dir / "traffic_schedule.csv"
        self.requests_path = self.output_dir / "requests.csv"
        self.summary_path = self.output_dir / "traffic_summary.csv"
        self.metadata_path = self.output_dir / "metadata.json"

        # Runtime state
        self.stop_requested = False
        self.in_flight_count = 0
        self.completed_requests: List[CompletedRequest] = []
        self.planned_schedule: List[PlannedRequest] = []
        self.start_mono: Optional[float] = None
        self.start_utc: Optional[str] = None
        self.end_mono: Optional[float] = None
        self.end_utc: Optional[str] = None

    def setup_directory_and_schedule(self) -> None:
        """Create output directory and persist planned schedule before sending traffic.

        If replay_schedule_path was provided, the exact persisted schedule is
        loaded and reused verbatim (never re-rolled) -- see load_planned_schedule().
        Otherwise a fresh schedule is generated from the profile as before.
        """
        if self.output_dir.exists():
            raise FileExistsError(
                f"Traffic output directory already exists: {self.output_dir}. "
                "Refusing to overwrite existing experiment data."
            )
        self.output_dir.mkdir(parents=True, exist_ok=False)

        if self.replay_schedule_path is not None:
            source_path = Path(self.replay_schedule_path)
            if not source_path.is_absolute():
                source_path = self.project_root / source_path
            self.planned_schedule = load_planned_schedule(source_path)
        else:
            self.planned_schedule = generate_arrival_schedule(
                self.profile,
                seed=self.seed,
                arrival_process=self.arrival_process,
            )

        with open(self.schedule_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SCHEDULE_CSV_FIELDNAMES)
            writer.writeheader()
            for req in self.planned_schedule:
                writer.writerow(req.to_dict())
            f.flush()

    def _signal_handler(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        sys.stderr.write(f"\nReceived signal {sig_name}. Stopping traffic generator gracefully...\n")
        self.stop_requested = True

    async def _dispatch_single_request(
        self,
        client: httpx.AsyncClient,
        req: PlannedRequest,
        start_mono: float,
        requests_file: Any,
    ) -> None:
        """Dispatch an individual HTTP request and record outcome."""
        actual_dispatch_mono = time.monotonic()
        actual_dispatch_elapsed = round(actual_dispatch_mono - start_mono, 6)
        dispatch_lag = round(actual_dispatch_elapsed - req.planned_elapsed_seconds, 6)

        # V2 Phase 1.1 CORRECTION: resource_mode/memory_mb/memory_hold_ms are
        # whole-profile settings (self.profile), not per-request
        # (PlannedRequest/the persisted schedule CSV are intentionally
        # untouched -- see module-level note near SCHEDULE_CSV_FIELDNAMES).
        # Phase 1 always appended these three query params, reasoning that
        # an unmodified V1 profile would send request semantics IDENTICAL to
        # before (since the server's own defaults match). That is numerically
        # true but was found to be an unacceptable regression against the
        # stricter Phase 1.1 legacy-compatibility requirement: a legacy
        # profile's dispatched URL must remain byte-identical to V1
        # (`?intensity=N` only), with no V2-only query parameters appended
        # at all unless the profile is genuinely non-default. Appending is
        # now conditional on the profile actually requesting V2 behavior.
        url = f"{self.target_url}{req.endpoint}?intensity={req.configured_intensity}"
        if self.profile.resource_mode != "cpu" or self.profile.memory_mb > 0:
            url += (
                f"&resource_mode={self.profile.resource_mode}&memory_mb={self.profile.memory_mb}"
                f"&memory_hold_ms={self.profile.memory_hold_ms}"
            )

        outcome = "unknown"
        status_code: Optional[int] = None
        latency: Optional[float] = None
        completion_elapsed: Optional[float] = None
        error_type: Optional[str] = None
        error_message: Optional[str] = None

        try:
            resp = await client.get(url, timeout=self.timeout_seconds)
            completion_mono = time.monotonic()
            latency = round(completion_mono - actual_dispatch_mono, 6)
            completion_elapsed = round(completion_mono - start_mono, 6)
            status_code = resp.status_code

            if 200 <= status_code < 300:
                outcome = "success_2xx"
            elif 400 <= status_code < 500:
                outcome = "client_error_4xx"
            elif 500 <= status_code < 600:
                outcome = "server_error_5xx"
            else:
                outcome = f"http_{status_code}"

        except httpx.TimeoutException as exc:
            completion_mono = time.monotonic()
            latency = self.timeout_seconds
            completion_elapsed = round(completion_mono - start_mono, 6)
            outcome = "timeout"
            error_type = type(exc).__name__
            error_message = str(exc)
        except asyncio.CancelledError as exc:
            completion_mono = time.monotonic()
            latency = round(completion_mono - actual_dispatch_mono, 6)
            completion_elapsed = round(completion_mono - start_mono, 6)
            outcome = "timeout"
            error_type = type(exc).__name__
            error_message = str(exc)
        except (httpx.RequestError, Exception) as exc:
            completion_mono = time.monotonic()
            completion_elapsed = round(completion_mono - start_mono, 6)
            error_type = type(exc).__name__
            error_message = str(exc)
            outcome = "connection_error"
        finally:
            self.in_flight_count = max(0, self.in_flight_count - 1)
            completed = CompletedRequest(
                request_index=req.request_index,
                planned_elapsed_seconds=req.planned_elapsed_seconds,
                actual_dispatch_elapsed_seconds=actual_dispatch_elapsed,
                dispatch_lag_seconds=dispatch_lag,
                completion_elapsed_seconds=completion_elapsed,
                configured_intensity=req.configured_intensity,
                phase_name=req.phase_name,
                target_rps=req.target_rps,
                outcome=outcome,
                http_status=status_code,
                client_latency_seconds=latency,
                error_type=error_type,
                error_message=error_message,
            )
            self.completed_requests.append(completed)

            # Write row to requests.csv
            row_dict = completed.to_dict()
            writer = csv.DictWriter(requests_file, fieldnames=REQUESTS_CSV_FIELDNAMES)
            writer.writerow(row_dict)
            requests_file.flush()

    def _record_dropped_overload(self, req: PlannedRequest, start_mono: float, requests_file: Any) -> None:
        """Record a scheduled arrival dropped due to local concurrency limits."""
        actual_mono = time.monotonic()
        actual_elapsed = round(actual_mono - start_mono, 6)
        dispatch_lag = round(actual_elapsed - req.planned_elapsed_seconds, 6)

        completed = CompletedRequest(
            request_index=req.request_index,
            planned_elapsed_seconds=req.planned_elapsed_seconds,
            actual_dispatch_elapsed_seconds=actual_elapsed,
            dispatch_lag_seconds=dispatch_lag,
            completion_elapsed_seconds=None,
            configured_intensity=req.configured_intensity,
            phase_name=req.phase_name,
            target_rps=req.target_rps,
            outcome="dropped_overload",
            http_status=None,
            client_latency_seconds=None,
        )
        self.completed_requests.append(completed)

        writer = csv.DictWriter(requests_file, fieldnames=REQUESTS_CSV_FIELDNAMES)
        writer.writerow(completed.to_dict())
        requests_file.flush()

    def _write_traffic_summary(self) -> None:
        """Generate 1-second interval traffic_summary.csv from completed requests."""
        max_dispatch = max(
            [r.actual_dispatch_elapsed_seconds for r in self.completed_requests if r.actual_dispatch_elapsed_seconds is not None]
            or [0.0]
        )
        total_seconds = max(
            int(math.ceil(self.profile.total_duration)),
            int(math.ceil(max_dispatch)),
        )
        summary_rows: List[Dict[str, Any]] = []

        # Bin requests by integer second of arrival / dispatch
        for sec in range(total_seconds):
            target_rps, phase_name, _ = self.profile.get_phase_and_target_rps(float(sec))

            sec_offered = sum(
                1 for r in self.planned_schedule if sec <= r.planned_elapsed_seconds < sec + 1
            )
            sec_reqs = [
                r
                for r in self.completed_requests
                if r.actual_dispatch_elapsed_seconds is not None and sec <= r.actual_dispatch_elapsed_seconds < sec + 1
            ]

            sent = sum(1 for r in sec_reqs if r.outcome != "dropped_overload")
            s2xx = sum(1 for r in sec_reqs if r.outcome == "success_2xx")
            c4xx = sum(1 for r in sec_reqs if r.outcome == "client_error_4xx")
            s5xx = sum(1 for r in sec_reqs if r.outcome == "server_error_5xx")
            t_out = sum(1 for r in sec_reqs if r.outcome == "timeout")
            c_err = sum(1 for r in sec_reqs if r.outcome == "connection_error")
            d_overload = sum(1 for r in sec_reqs if r.outcome == "dropped_overload")

            summary_rows.append({
                "elapsed_seconds": sec,
                "phase_name": phase_name,
                "target_rps": f"{target_rps:.4f}",
                "offered_requests": sec_offered,
                "sent_requests": sent,
                "success_2xx": s2xx,
                "client_error_4xx": c4xx,
                "server_error_5xx": s5xx,
                "timeout": t_out,
                "connection_error": c_err,
                "dropped_overload": d_overload,
                "achieved_send_rps": f"{sent:.2f}",
            })

        with open(self.summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDNAMES)
            writer.writeheader()
            for s_row in summary_rows:
                writer.writerow(s_row)
            f.flush()

    def _write_metadata(self) -> None:
        """Write metadata.json atomically upon run completion."""
        actual_runtime = round(self.end_mono - self.start_mono, 2) if self.end_mono and self.start_mono else 0.0
        schedule_end = (
            round(self.planned_schedule[-1].planned_elapsed_seconds, 6)
            if self.planned_schedule
            else 0.0
        )

        outcome_counts = {
            "success_2xx": sum(1 for r in self.completed_requests if r.outcome == "success_2xx"),
            "client_error_4xx": sum(1 for r in self.completed_requests if r.outcome == "client_error_4xx"),
            "server_error_5xx": sum(1 for r in self.completed_requests if r.outcome == "server_error_5xx"),
            "timeout": sum(1 for r in self.completed_requests if r.outcome == "timeout"),
            "connection_error": sum(1 for r in self.completed_requests if r.outcome == "connection_error"),
            "dropped_overload": sum(1 for r in self.completed_requests if r.outcome == "dropped_overload"),
        }

        # Dispatch lag stats
        lags = [r.dispatch_lag_seconds for r in self.completed_requests if r.dispatch_lag_seconds is not None]
        lag_stats = {
            "mean_seconds": round(sum(lags) / len(lags), 6) if lags else 0.0,
            "max_seconds": round(max(lags), 6) if lags else 0.0,
            "min_seconds": round(min(lags), 6) if lags else 0.0,
        }

        # Latency stats
        latencies = [
            r.client_latency_seconds for r in self.completed_requests if r.client_latency_seconds is not None
        ]
        latencies.sort()
        lat_stats = {
            "mean_seconds": round(sum(latencies) / len(latencies), 6) if latencies else None,
            "p50_seconds": round(latencies[int(len(latencies) * 0.50)], 6) if latencies else None,
            "p95_seconds": round(latencies[int(len(latencies) * 0.95)], 6) if latencies else None,
            "p99_seconds": round(latencies[int(len(latencies) * 0.99)], 6) if latencies else None,
        }

        is_smoke = "smoke" in self.profile.profile_name.lower() or "test" in self.profile.profile_name.lower()

        metadata = {
            "experiment_id": self.experiment_id,
            "profile": self.profile.profile_name,
            "seed": self.seed,
            "arrival_process": self.arrival_process,
            "configured_intensity": self.profile.intensity,
            "target_url": self.target_url,
            "requested_duration_seconds": self.profile.total_duration,
            "schedule_end_elapsed_seconds": schedule_end,
            "actual_runtime_seconds": actual_runtime,
            "actual_duration_seconds": actual_runtime,
            "duration_seconds": actual_runtime,
            "duration_semantics": {
                "requested_duration_seconds": "Configured total duration of the workload profile.",
                "schedule_end_elapsed_seconds": "Planned arrival elapsed time of the final scheduled request.",
                "actual_runtime_seconds": "Total wall-clock execution time of the generator including profile completion and bounded grace period.",
                "duration_seconds": "Alias to actual_runtime_seconds (retained for backward compatibility).",
                "actual_duration_seconds": "Alias to actual_runtime_seconds (retained for backward compatibility).",
            },
            "timeout_seconds": self.timeout_seconds,
            "max_in_flight": self.max_in_flight,
            "effective_max_in_flight": self.effective_max_in_flight,
            "connection_mode": self.connection_mode,
            "total_offered": len(self.planned_schedule),
            "total_sent": sum(1 for r in self.completed_requests if r.outcome != "dropped_overload"),
            "outcome_totals": outcome_counts,
            "dispatch_lag_stats": lag_stats,
            "client_latency_stats": lat_stats,
            "schedule_generation_info": {
                "arrival_process": self.arrival_process,
                "seed": self.seed,
                "total_planned_requests": len(self.planned_schedule),
                "replayed_from_schedule": self.replay_schedule_path,
                "schedule_sha256": _sha256_file(self.schedule_path),
            },
            "test_non_benchmark": is_smoke,
            "start_time_utc": self.start_utc,
            "end_time_utc": self.end_utc,
        }

        tmp_path = self.metadata_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(self.metadata_path)

    async def run_async(self) -> None:
        """Asynchronous execution loop for the open-loop dispatcher."""
        self.setup_directory_and_schedule()

        orig_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        orig_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)

        self.start_mono = time.monotonic()
        self.start_utc = datetime.now(timezone.utc).isoformat()
        profile_deadline_mono = self.start_mono + self.profile.total_duration

        print(f"Traffic generator started for experiment '{self.experiment_id}'")
        print(f"Profile: {self.profile.profile_name} | Seed: {self.seed} | Process: {self.arrival_process}")
        print(f"Target: {self.target_url}{self.profile.endpoint}?intensity={self.profile.intensity}")
        print(f"Total planned requests: {len(self.planned_schedule)} | Duration: {self.profile.total_duration}s")
        print(f"Max in-flight concurrency bound: {self.effective_max_in_flight} (general cap: {self.max_in_flight})")

        requests_file = open(self.requests_path, "w", encoding="utf-8", newline="")
        writer = csv.DictWriter(requests_file, fieldnames=REQUESTS_CSV_FIELDNAMES)
        writer.writeheader()
        requests_file.flush()

        active_tasks: set = set()

        # connection_mode "fresh" (Phase 11C.4B) sets max_keepalive_connections=0
        # so httpx keeps no idle connections in its pool -- every request opens a
        # new TCP connection and closes it after the response, instead of reusing
        # a long-lived keep-alive connection. This matters specifically when the
        # target is a genuine Kubernetes Service (kube-proxy load-balances at
        # NEW-connection time, not per-request): with keep-alive, a handful of
        # long-lived connections can pin nearly all traffic to whichever pod(s)
        # kube-proxy happened to route those few connections to, starving other
        # Ready replicas regardless of how well the Service itself is load
        # balancing. "keepalive" (the default) preserves this project's original,
        # historical behavior exactly -- Phase-10 calibration and any other prior
        # artifact was collected with this default and is unaffected by this flag
        # existing, since nothing changes unless "fresh" is explicitly requested.
        max_keepalive = 0 if self.connection_mode == "fresh" else max(10, self.max_in_flight)
        limits = httpx.Limits(
            max_connections=max(10, self.max_in_flight * 2),
            max_keepalive_connections=max_keepalive,
        )

        try:
            async with httpx.AsyncClient(limits=limits) as client:
                for req in self.planned_schedule:
                    if self.stop_requested:
                        break

                    # Enforce profile boundary: do not schedule requests at or past profile duration
                    if req.planned_elapsed_seconds >= self.profile.total_duration:
                        break

                    req_deadline_mono = self.start_mono + req.planned_elapsed_seconds

                    # If scheduled arrival is beyond profile deadline, stop scheduling
                    if req_deadline_mono >= profile_deadline_mono:
                        break

                    # CORR-52: membership of this request in the offered workload is decided
                    # ONCE, by the two fixed-value checks above (mirroring the schedule-
                    # generation-time membership test CORR-12 already established as
                    # authoritative -- see docs/experimental_design_corrections.md CORR-11/
                    # CORR-12 and phase_16l3_duration_semantics_evidence_audit.md). This
                    # single gate below re-applies that same "has the window already
                    # closed" question exactly once, using a live clock read, to preserve
                    # the pre-existing "no catch-up burst after a genuine stall" behavior:
                    # if cumulative elapsed time (from all prior requests, or a genuine host
                    # stall) has already exhausted the profile window by the time we reach
                    # this request, it is correctly abandoned here, with nothing dispatched.
                    if time.monotonic() >= profile_deadline_mono:
                        break

                    # Converge toward this request's own (already membership-approved)
                    # deadline. CORR-51 removed one race here (a redundant second read
                    # immediately after loop exit); CORR-52 removes the deeper one: a
                    # RECURRING profile-deadline recheck inside this same convergence loop
                    # cannot be more precise than ordinary asyncio.sleep()/OS scheduling
                    # overshoot (empirically 99 microseconds to several milliseconds), so it
                    # would silently revoke an already-approved dispatch whenever a
                    # schedule's final row lands within that same margin of the boundary --
                    # exactly the i2bval_r2_V1/r1_V2/r4_V1 case. Once membership is approved
                    # (the gate above), only an explicit stop_requested aborts dispatch; the
                    # loop's only remaining job is to wait for this request's own deadline.
                    while not self.stop_requested:
                        now = time.monotonic()
                        rem = req_deadline_mono - now
                        if rem <= 0.0005:
                            break
                        await asyncio.sleep(min(0.05, rem))

                    if self.stop_requested:
                        break

                    # Check bounded concurrency (V2 Phase 2.1: the profile's
                    # own effective_max_in_flight, never looser than the
                    # general --max-in-flight bound -- see __init__).
                    if self.in_flight_count >= self.effective_max_in_flight:
                        # Concurrency limit reached: drop overload at generator side
                        self._record_dropped_overload(req, self.start_mono, requests_file)
                    else:
                        self.in_flight_count += 1
                        task = asyncio.create_task(
                            self._dispatch_single_request(client, req, self.start_mono, requests_file)
                        )
                        active_tasks.add(task)
                        task.add_done_callback(active_tasks.discard)

                # After final planned arrival dispatch, keep lifecycle open until profile deadline if time remains
                while not self.stop_requested:
                    rem = profile_deadline_mono - time.monotonic()
                    if rem <= 0.0005:
                        break
                    await asyncio.sleep(min(0.05, rem))

                # Profile deadline reached: stop scheduling new requests (loop already finished).
                # Allow already-issued in-flight requests a bounded grace period governed by request timeout.
                if active_tasks and not self.stop_requested:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*active_tasks, return_exceptions=True),
                            timeout=self.timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        for t in list(active_tasks):
                            if not t.done():
                                t.cancel()
                        # Ensure cancelled tasks finalize their records before files are closed
                        await asyncio.gather(*active_tasks, return_exceptions=True)

        finally:
            requests_file.close()
            signal.signal(signal.SIGINT, orig_sigint)
            signal.signal(signal.SIGTERM, orig_sigterm)

            self.end_mono = time.monotonic()
            self.end_utc = datetime.now(timezone.utc).isoformat()

            self._write_traffic_summary()
            self._write_metadata()

            print(
                f"Traffic generator completed: offered={len(self.planned_schedule)}, "
                f"completed={len(self.completed_requests)}, "
                f"duration={round(self.end_mono - self.start_mono, 2)}s"
            )

    def run(self) -> None:
        """Synchronous entry point."""
        asyncio.run(self.run_async())


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Host-side open-loop traffic generator for Kubernetes autoscaling experiments"
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        required=True,
        help="Path to traffic profile YAML file",
    )
    parser.add_argument(
        "--experiment-id",
        "--run-id",
        dest="experiment_id",
        required=True,
        help="Unique identifier for experiment run",
    )
    parser.add_argument(
        "--target-url",
        dest="target_url",
        default="http://localhost:8080",
        help="Base URL of application service (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--arrival-process",
        dest="arrival_process",
        choices=["poisson", "deterministic", "even"],
        default="poisson",
        help="Arrival process: 'poisson' (default) or 'deterministic'",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="Random seed for arrival schedule generation",
    )
    parser.add_argument(
        "--max-in-flight",
        dest="max_in_flight",
        type=int,
        default=50,
        help="Maximum concurrent in-flight HTTP requests allowed (default: 50)",
    )
    parser.add_argument(
        "--timeout",
        dest="timeout",
        type=float,
        default=5.0,
        help="Client request timeout in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--config",
        dest="config",
        default="configs/experiment.yaml",
        help="Path to central experiment.yaml config",
    )
    parser.add_argument(
        "--connection-mode",
        dest="connection_mode",
        choices=["keepalive", "fresh"],
        default="keepalive",
        help=(
            "'keepalive' (default): preserve historical httpx keep-alive connection "
            "reuse behavior, unchanged from Phase 10 onward. 'fresh': open a new TCP "
            "connection per request (max_keepalive_connections=0) so a genuine "
            "Kubernetes Service's per-connection load balancing is exercised on "
            "every request -- required when targeting a multi-replica Service "
            "instead of a single-pod-pinned kubectl port-forward (Phase 11C.4B)."
        ),
    )
    parser.add_argument(
        "--replay-schedule",
        dest="replay_schedule",
        default=None,
        help=(
            "Path to a previously persisted traffic_schedule.csv to replay verbatim "
            "instead of generating a new schedule. Used to drive multiple experiment "
            "runs (e.g. HPA-target candidates) with the bit-identical planned "
            "arrival schedule. Never re-rolls randomness; fails clearly on a "
            "malformed or structurally invalid schedule file."
        ),
    )
    return parser.parse_args(args)


def main() -> None:
    parsed = parse_args()
    try:
        generator = TrafficGenerator(
            experiment_id=parsed.experiment_id,
            profile_path=parsed.profile,
            target_url=parsed.target_url,
            arrival_process=parsed.arrival_process,
            seed=parsed.seed,
            max_in_flight=parsed.max_in_flight,
            timeout_seconds=parsed.timeout,
            replay_schedule_path=parsed.replay_schedule,
            connection_mode=parsed.connection_mode,
        )
        generator.run()
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
