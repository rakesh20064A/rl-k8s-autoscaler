# app/src/workload.py
# Controlled CPU work simulation for the autoscaler benchmark.
# Each /work request calls perform_cpu_work(intensity) which runs a
# deterministic mathematical loop that consumes CPU proportional to intensity.
# This gives a reliable way to drive HPA CPU thresholds without a real business workload.

import math
import os
import time


# Maximum allowed work duration per request — safety cap set via environment variable.
_MAX_WORK_DURATION = float(os.getenv("MAX_WORK_DURATION_SECONDS", "10"))


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
