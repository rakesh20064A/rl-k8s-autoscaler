# Part 07 — Application (app/src/)

All FROM_CODE (read from source, not runtime-verified this session — the
live pod could not be `kubectl exec`'d into due to API-server
unreachability; see 04_live_cluster.md).

## Endpoints (`app/src/main.py`)

1. `GET /` (main.py:71-73) — `root()`, no params, returns
   `{"service": "demo-app", "status": "running"}`.
2. `GET /health` (main.py:76-90) — `async def health()`, no params, returns
   `{"status": "ok"}`. Deliberately `async def` (not sync): a sync route is
   offloaded to Starlette's shared AnyIO worker-thread pool (default 40
   tokens) shared with `/work`; under heavy `/work` concurrency that pool
   can be exhausted, starving `/health` behind CPU-bound work past the k8s
   probe timeout (in-code "Index32 diagnosis" comment). `async def` with no
   blocking call runs on the event loop directly, never consuming a
   worker-thread token — this was the already-applied mitigation
   extensively investigated (and found insufficient alone under heavy load)
   in Phase 4B-R2/R3.
3. `GET /work` (main.py:93-141) — `work(intensity, resource_mode, memory_mb,
   memory_hold_ms)`:
   - `intensity: int = Query(default=3, ge=1, le=10)`.
   - `resource_mode: Literal["cpu","memory","mixed"] = Query(default="cpu")`.
   - `memory_mb: int = Query(default=0, ge=0, le=MAX_WORK_MEMORY_MB)`
     (default cap 16, from workload.py).
   - `memory_hold_ms: int = Query(default=DEFAULT_MEMORY_HOLD_MS, ge=0,
     le=MAX_MEMORY_HOLD_MS)` (defaults 500/5000).
   - Calls `perform_work(...)`; sets `LAST_WORK_INTENSITY` gauge; on
     exception logs and re-raises (→500). Wrapped with Prometheus
     instrumentation (`REQUESTS_IN_PROGRESS`, `REQUEST_COUNT`,
     `REQUEST_LATENCY`).
4. `GET /metrics` (main.py:144-148) — `generate_latest()`,
   `CONTENT_TYPE_LATEST`.

**Server startup**: the `if __name__=="__main__"` block
(`uvicorn.run("main:app", host="0.0.0.0", port=8080, ...)`, main.py:160-169)
is **never actually executed** — the real Dockerfile `CMD` is
`python -m uvicorn main:app --host 0.0.0.0 --port 8080` (no `--workers`
flag → 1 worker/process, single uvicorn event loop). Signal handling:
`signal.signal(signal.SIGTERM, _shutdown)` registered at module import time
(main.py:151-157); `_shutdown` logs and `sys.exit(0)`.

## `app/src/workload.py`

Module constants and env-var overrides:
`_MAX_WORK_DURATION = float(os.getenv("MAX_WORK_DURATION_SECONDS", "10"))`;
`MAX_WORK_MEMORY_MB = int(os.getenv("MAX_WORK_MEMORY_MB", "16"))`;
`MAX_MEMORY_HOLD_MS = int(os.getenv("MAX_MEMORY_HOLD_MS", "5000"))`;
`DEFAULT_MEMORY_HOLD_MS = int(os.getenv("DEFAULT_MEMORY_HOLD_MS", "500"))`;
`_PAGE_SIZE_BYTES = 4096` (no env override).

`perform_cpu_work(intensity)`: clips to [1,10]; `iterations = 20_000 *
intensity**2` (quadratic); loop `total += sqrt(i) * log(i+1)`; checks the
10s safety cap every 50,000 iterations.

`perform_memory_work(memory_mb, hold_ms)`: clamps both args; allocates via
**anonymous `mmap.mmap(-1, num_bytes, MAP_PRIVATE|MAP_ANONYMOUS)`**
(explicitly NOT a bytearray — a documented correction from an earlier
approach, since glibc's per-thread arena allocator doesn't reliably
`munmap()` freed bytearray memory under concurrent load, observed live as
pod RSS stuck at ~253Mi instead of returning to ~35-38Mi idle); touches one
byte per 4096-byte page to force residency; `time.sleep(hold_ms/1000)`;
explicit `mapping.close()` in a `finally` block (guaranteed `munmap()` even
on exception).

`perform_work(intensity, resource_mode, memory_mb, memory_hold_ms)` — mode
dispatch: `"cpu"` runs only CPU work; `"memory"` **skips CPU work entirely**
(hardcoded zero-iteration stub, so memory pressure is never diluted by
competing CPU load); `"mixed"` runs both independently and fully.
`resource_mode="cpu"` + `memory_mb=0` is byte-for-byte identical to the
original V1 CPU-only behavior.

## `app/src/metrics.py` — all 4 Prometheus metrics

| Name | Type | Labels | Buckets |
|---|---|---|---|
| `demo_app_requests_total` | Counter | `route`, `status_code` | — |
| `demo_app_request_duration_seconds` | Histogram | `route` | 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0 |
| `demo_app_requests_in_progress` | Gauge | `route` | — |
| `demo_app_last_work_intensity` | Gauge | none | — |

**No per-pod/per-request attribution metric or field exists anywhere** —
none of these metrics or the `/work` response body carry a pod identity.

## `app/Dockerfile`

Multi-stage: `python:3.10-slim` builder (pip install to `/install`) →
`python:3.10-slim` runtime (copies `/install`→`/usr/local`, `src/`→`/app`).
Non-root user (`appuser`/`appgroup`). `EXPOSE 8080`. Env vars baked into the
image: `PYTHONUNBUFFERED=1`, `LOG_LEVEL=INFO`, `LOG_HEALTH_REQUESTS=false`,
`MAX_WORK_DURATION_SECONDS=10` — note `MAX_WORK_MEMORY_MB`/
`MAX_MEMORY_HOLD_MS`/`DEFAULT_MEMORY_HOLD_MS` are **not** set in the
Dockerfile, so they always use workload.py's hardcoded Python defaults
(16/5000/500) regardless of any ConfigMap. Final `CMD`:
`["python","-m","uvicorn","main:app","--host","0.0.0.0","--port","8080"]`.
