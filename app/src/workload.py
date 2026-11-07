# app/src/workload.py
# Controlled CPU/memory work simulation for the autoscaler benchmark.
# Each /work request calls perform_work(intensity, resource_mode, memory_mb),
# which dispatches to a deterministic CPU loop (perform_cpu_work, unchanged
# from V1) and/or a bounded, deterministic memory workload (V2 addition),
# giving a reliable way to drive HPA/PPO observations without a real
# business workload.

import math
import mmap
import os
import time

# Maximum allowed work duration per request — safety cap set via environment variable.
_MAX_WORK_DURATION = float(os.getenv("MAX_WORK_DURATION_SECONDS", "10"))

# V2 Phase 1.1 SAFETY CORRECTION: the pod's own resource.requests/limits
# (k8s/app/deployment.yaml) are memory: requests=64Mi, limits=256Mi. Phase
# 1's original default (256) was the ENTIRE pod memory limit for a SINGLE
# request -- unsafe, because memory_mb is a PER-REQUEST bound and multiple
# /work requests execute concurrently (FastAPI dispatches sync `def` routes,
# including this one, onto a worker-thread pool). The load generator's own
# actual concurrency ceiling is DEFAULT_MAX_IN_FLIGHT=50
# (scripts/traffic_orchestrator.py); a true worst case of 50 simultaneous
# memory-mode requests at the old 256MB bound would be ~12.8GB, instantly
# OOM-killing the pod many times over.
#
# New conservative default: reserving ~60MiB for the baseline Python/
# FastAPI/uvicorn/prometheus_client process footprint (not separately
# measured here -- see Phase 1.1 audit report), ~196MiB remains as budget.
# Dividing by the generator's actual max_in_flight=50 gives ~4MB/request
# for FULL safety even at that literal worst case; this default is set
# slightly above that (16MB) as a deliberate, DISCLOSED trade-off favoring
# memory-pressure observability for the normal/expected low-concurrency V2
# smoke profiles (where realistic concurrent in-flight is far below 50) --
# see the V2 Phase 1.1 report for the explicit risk this accepts (a true
# 50-concurrent burst at 16MB would reach ~800MB, exceeding the pod limit).
# Deployments that genuinely need the full 50-way concurrency ceiling WHILE
# running a memory/mixed V2 profile should additionally cap the load
# generator's own --max-in-flight for that run.
MAX_WORK_MEMORY_MB = int(os.getenv("MAX_WORK_MEMORY_MB", "16"))

# V2 Phase 1.1: a touch-then-immediately-return allocation is too transient
# to be sampled -- Prometheus scrapes every 10s (k8s/monitoring/values.yaml)
# and the PPO control interval is 30s (configs/experiment.yaml), both far
# longer than the microsecond-scale cost of touching a few MB. Holding the
# touched buffer resident for a short, bounded, configurable duration makes
# the resulting memory pressure continuously visible for as long as
# requests keep arriving faster than they're released (Little's-law
# overlap), without ever holding any single request's memory indefinitely.
MAX_MEMORY_HOLD_MS = int(os.getenv("MAX_MEMORY_HOLD_MS", "5000"))
DEFAULT_MEMORY_HOLD_MS = int(os.getenv("DEFAULT_MEMORY_HOLD_MS", "500"))

VALID_RESOURCE_MODES = ("cpu", "memory", "mixed")
_PAGE_SIZE_BYTES = 4096  # touch stride: forces each page to actually fault in, not just reserve virtual address space


def perform_cpu_work(intensity: int) -> dict:
    """Run a CPU-bound computation loop scaled by intensity.

    Args:
        intensity: Work level from 1 (lightest) to 10 (heaviest).
                   Values outside the range are clipped.

    Returns:
        dict with computation result and elapsed time for the response body.
    """
    intensity = max(1, min(10, intensity))
    # Number of iterations scales quadratically with intensity so that
    # increasing intensity produces a clear, measurable difference in CPU time.
    iterations = 20_000 * (intensity ** 2)

    start = time.monotonic()
    total = 0.0

    for i in range(1, iterations + 1):
        total += math.sqrt(float(i)) * math.log(float(i + 1))
        # Respect the safety cap — stop early if the request is taking too long.
        if i % 50_000 == 0:
            if time.monotonic() - start > _MAX_WORK_DURATION:
                break

    elapsed_ms = (time.monotonic() - start) * 1000.0
    return {
        "intensity": intensity,
        "iterations_completed": i,
        "elapsed_ms": round(elapsed_ms, 2),
        "result": round(total, 4),
    }


def perform_memory_work(memory_mb: int, hold_ms: int = DEFAULT_MEMORY_HOLD_MS) -> dict:
    """Allocate and touch `memory_mb` megabytes, hold them resident for
    `hold_ms` milliseconds (so the allocation is actually long enough to be
    sampled by Prometheus/Metrics-Server -- see MAX_MEMORY_HOLD_MS above),
    then release them.

    Args:
        memory_mb: Requested memory size in MB. Clamped to
                   [0, MAX_WORK_MEMORY_MB] (clip, never reject, matching
                   `perform_cpu_work`'s intensity convention). 0 performs
                   no memory work at all (hold_ms is then irrelevant).
        hold_ms: How long to hold the touched buffer resident before
                 releasing it. Clamped to [0, MAX_MEMORY_HOLD_MS] -- never
                 unbounded, so a single request can never hold memory
                 indefinitely.

    Returns:
        dict with requested/touched memory (MB), the hold duration applied,
        and elapsed time.

    V2 Phase 2.2 PART 1 CORRECTION: a plain `bytearray` is allocated via
    CPython's object allocator, which for large buffers delegates to the
    C library's `malloc`/`free` -- but `free()` does NOT guarantee the
    underlying pages are returned to the OS immediately. Under concurrent
    multi-threaded load (FastAPI dispatches sync routes, including this
    one, onto a worker-thread pool), glibc gives each thread its OWN
    arena, and a freed allocation in one thread's arena is typically
    retained by that arena (not munmap'd) rather than handed back to the
    kernel, even though CPython's refcounting already freed the Python
    object correctly. This was CONFIRMED live (V2 Phase 2.1 PART 4 smoke
    test): after a concurrent memory-mode burst ended, pod RSS stayed at
    ~253Mi indefinitely instead of returning toward the ~35-38Mi idle
    baseline -- a real, observed leak-like symptom despite the Python
    object itself being correctly collected.
    anonymous `mmap.mmap(-1, ...)` maps pages DIRECTLY from the kernel,
    bypassing glibc's arena allocator entirely; `.close()` issues an
    explicit `munmap()`, which unconditionally returns those exact pages
    to the OS the moment this function releases them -- regardless of any
    allocator-level retention heuristics. The `finally` block guarantees
    this release happens even if touching pages or sleeping raises.
    """
    memory_mb = max(0, min(MAX_WORK_MEMORY_MB, memory_mb))
    hold_ms = max(0, min(MAX_MEMORY_HOLD_MS, hold_ms))
    if memory_mb == 0:
        return {"memory_mb_requested": 0, "memory_mb_touched": 0, "memory_hold_ms": 0, "memory_elapsed_ms": 0.0}

    start = time.monotonic()
    num_bytes = memory_mb * 1024 * 1024
    # Anonymous, process-private mapping -- never backed by a file, never
    # shared with another process. Pages aren't resident (RSS) until written.
    mapping = mmap.mmap(-1, num_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    try:
        for offset in range(0, num_bytes, _PAGE_SIZE_BYTES):
            mapping[offset] = 1  # one write per page -- forces it resident, so RSS actually reflects memory_mb
        if hold_ms > 0:
            time.sleep(hold_ms / 1000.0)  # mapping stays resident for the whole sleep
    finally:
        # Explicit munmap -- runs even on an exception from the touch loop
        # or the sleep, so this mapping can NEVER outlive the request that
        # created it, and the OS reclaims these exact pages immediately
        # (no reliance on glibc arena trimming, gc.collect(), or any other
        # indirect/best-effort mechanism).
        mapping.close()
    elapsed_ms = (time.monotonic() - start) * 1000.0

    return {
        "memory_mb_requested": memory_mb,
        "memory_mb_touched": memory_mb,
        "memory_hold_ms": hold_ms,
        "memory_elapsed_ms": round(elapsed_ms, 2),
    }


def perform_work(
    intensity: int, resource_mode: str = "cpu", memory_mb: int = 0, memory_hold_ms: int = DEFAULT_MEMORY_HOLD_MS,
) -> dict:
    """V2 unified `/work` dispatcher. `resource_mode="cpu"` (the default)
    with `memory_mb=0` is BYTE-FOR-BYTE backward compatible with calling
    `perform_cpu_work(intensity)` directly -- every V1 response field is
    present with the same value; V2 only ever ADDS fields, never renames or
    removes one, so existing V1 response-shape assertions keep passing.

    resource_mode:
        "cpu"    -- V1 behavior: CPU work only, memory_mb/memory_hold_ms ignored.
        "memory" -- memory work only; CPU work is skipped entirely (0
                    iterations) so memory pressure is never diluted by
                    competing CPU load.
        "mixed"  -- both the full requested CPU work AND the requested
                    memory work, independently.
    """
    if resource_mode not in VALID_RESOURCE_MODES:
        raise ValueError(f"invalid resource_mode {resource_mode!r}; must be one of {VALID_RESOURCE_MODES}")

    if resource_mode == "memory":
        cpu_result = {"intensity": max(1, min(10, intensity)), "iterations_completed": 0, "elapsed_ms": 0.0, "result": 0.0}
    else:
        cpu_result = perform_cpu_work(intensity)

    memory_result = perform_memory_work(memory_mb, hold_ms=memory_hold_ms) if resource_mode in ("memory", "mixed") else {
        "memory_mb_requested": 0, "memory_mb_touched": 0, "memory_hold_ms": 0, "memory_elapsed_ms": 0.0,
    }

    return {
        "resource_mode": resource_mode,
        **cpu_result,
        **memory_result,
    }
