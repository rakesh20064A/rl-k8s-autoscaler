# app/src/main.py
# FastAPI demo application for the RL autoscaler benchmark.
# Endpoints:
#   GET /          - simple root response
#   GET /health    - lightweight readiness/liveness probe (no CPU work)
#   GET /work      - CPU-bound workload endpoint; drives HPA and RL observations
#   GET /metrics   - Prometheus metrics (scraped by kube-prometheus-stack)

import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager


from fastapi import FastAPI, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

# Support both running as a standalone container/module (WORKDIR /app)
# and importing from repository root (app.src.main)
try:
    from metrics import (
        LAST_WORK_INTENSITY,
        REQUEST_COUNT,
        REQUEST_LATENCY,
        REQUESTS_IN_PROGRESS,
    )
    from workload import perform_cpu_work
except ImportError:
    from app.src.metrics import (
        LAST_WORK_INTENSITY,
        REQUEST_COUNT,
        REQUEST_LATENCY,
        REQUESTS_IN_PROGRESS,
    )
    from app.src.workload import perform_cpu_work

# Logging setup: structured single-line format, level from environment variable.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("demo_app")

# Whether to emit a log line for every /health probe (noisy under frequent probing)
LOG_HEALTH_REQUESTS = os.getenv("LOG_HEALTH_REQUESTS", "false").lower() == "true"


# Lifespan handler manages startup and graceful shutdown logging.
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("demo_app starting")
    yield
    logger.info("demo_app shutting down")


app = FastAPI(
    title="RL Autoscaler Demo App",
    description="Lightweight workload service for the K8s autoscaler benchmark.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)


@app.get("/")
def root():
    return {"service": "demo-app", "status": "running"}


@app.get("/health")
async def health():
    """Lightweight probe endpoint — must always return quickly.

    Deliberately `async def`, not `def`: a sync FastAPI handler is offloaded
    to Starlette's shared AnyIO worker-thread pool (default 40 tokens,
    shared by every sync route in this app, including CPU-bound `/work`).
    Under heavy `/work` concurrency that pool can be fully occupied,
    queuing `/health` behind CPU-bound work and exceeding the Kubernetes
    probe timeout (Index32 diagnosis). An `async def` handler with no
    blocking call runs directly on the event loop and never requests a
    worker-thread token, so it cannot be starved by `/work`."""
    if LOG_HEALTH_REQUESTS:
        logger.debug("health probe")
    return {"status": "ok"}


@app.get("/work")
def work(
    intensity: int = Query(default=3, ge=1, le=10, description="CPU work level 1-10")
):
    """Performs controlled CPU work and returns timing results.

    The intensity parameter drives how much CPU is consumed.
    Traffic generators should vary this to simulate different workload types.
    """
    route = "/work"
    REQUESTS_IN_PROGRESS.labels(route=route).inc()
    start = time.monotonic()
    status_code = "200"

    try:
        result = perform_cpu_work(intensity)
        LAST_WORK_INTENSITY.set(intensity)
        logger.debug(
            "work request completed",
            extra={"intensity": intensity, "elapsed_ms": result["elapsed_ms"]},
        )
        return result
    except Exception:
        status_code = "500"
        logger.exception("work request failed")
        raise
    finally:
        elapsed = time.monotonic() - start
        REQUEST_COUNT.labels(route=route, status_code=status_code).inc()
        REQUEST_LATENCY.labels(route=route).observe(elapsed)
        REQUESTS_IN_PROGRESS.labels(route=route).dec()


@app.get("/metrics")
def metrics():
    """Exposes Prometheus metrics in the text/plain format expected by the scraper."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# Allow graceful shutdown on SIGTERM (Kubernetes sends SIGTERM before SIGKILL).
def _shutdown(signum, frame):
    logger.info("Received signal %d, initiating graceful shutdown", signum)
    sys.exit(0)


signal.signal(signal.SIGTERM, _shutdown)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        log_level=LOG_LEVEL.lower(),
        access_log=LOG_HEALTH_REQUESTS,
    )
