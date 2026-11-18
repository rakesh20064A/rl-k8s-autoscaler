# PHASE 9B — UI fixes: stale active-run lock + live pairing

## root_cause
Two independent bugs, confirmed by reading the live on-disk store and code:

1. `demo_ui/experiment_store.py`'s `active_run()` treated any record with
   `status=="running"` as active, with no liveness check (no PID check, no
   status-file freshness check). `demo_ui/live_execution_adapter.py` created
   each run's status/log files via `tempfile.mkstemp()` with no `dir=`,
   defaulting to the system temp dir (`/tmp` on macOS) — wiped by the Mac
   restart. After the restart, a fresh `RealExperimentControllerV2()` has no
   `_proc`/`_status_path`, so `status()` returns `{"status":"unknown"}` and
   never updates `experiment_store`, leaving the dead-process record stuck at
   `status="running"` forever and locking the Experiment page's ARM/traffic
   controls (`controls_disabled = active_rec is not None` in
   `controls_v2.py`), while START stayed enabled because
   `du_v2_run_in_progress` is per-browser-session state the restart also
   cleared. Confirmed live: `demo_ui/.run_store/runs.json` contained 3 real
   stale `"running"` `ppo_v2` records with no pid/status file left.
2. `latest_compatible_pair()` keyed on `controller.lower()` exactly equal to
   `"hpa"`/`"ppo"`, but the real V2 controller ids (from
   `components/controls_v2.py`'s `ARM_LABELS`) are `"ppo_v2"` and
   `"hpa_v2_cpu_memory"`, so no V2 pair could ever match — the Comparison
   page's Live Comparison section always showed "No compatible experiment
   pair yet." even with two compatible completed V2 runs.

## fixes
- `experiment_store.py`: added `pid`/`status_path` fields to run records.
  `is_run_alive()`/`active_run()` now require a live PID (`os.kill(pid, 0)`)
  AND a fresh status-file mtime (`STALE_STATUS_FILE_AGE_S=120s`). Any
  `"running"` record failing this is updated in place (never deleted) to
  `status="INTERRUPTED (stale)"` (added to `TERMINAL_STATUSES`), so it shows
  honestly in History and unlocks the controls.
- `latest_compatible_pair()` now uses a substring-based `_controller_family()`
  (`"ppo" in id` / `"hpa" in id`) instead of exact match, fixing pairing for
  `ppo_v2`/`hpa_v2_cpu_memory` while staying compatible with legacy
  `"HPA"`/`"PPO"` ids.
- `live_execution_adapter.py`: status/log files now live under
  `persistent-data/demo_ui/` (new, gitignored, created on demand) instead of
  the system temp dir; `start()` persists `pid`+`status_path` into
  `experiment_store` immediately after `Popen` so liveness survives a future
  restart.
- `.gitignore`: added `persistent-data/demo_ui/` (mirrors the existing
  `demo_ui/.run_store/` entry).
- `models.py` cleanup: deleted 13 confirmed-unused V1-only dataclasses/enums
  (`Controller`, `TrafficProfile`, `LogSource`, `LogLevel`, `SingleRunState`,
  `AutoRunState`, `ResetSnapshot`, `LogEvent`, `TimeSeriesPoint`,
  `RunSummary`, `ComparisonMetric`, `ComparisonResult`, `ConclusionCategory`,
  `ConclusionDimension`, `Conclusion`, `ExperimentRecord`). Kept
  `PodState`/`PodStatus`/`HpaSnapshot`/`PpoSnapshot`/`PpoAction`/
  `PpoDecision`/`EventMarker` — still imported elsewhere.

## stale_runs_marked
All 3 real stale `ppo_v2` `"running"` records in the live
`demo_ui/.run_store/runs.json` were self-healed by a single read-only
`active_run()` call (no subprocess launched) — now
`status="INTERRUPTED (stale)"` with `completed_at` stamped. A later demo_ui
pytest run exercised a real START click against an unreachable cluster
(pre-existing test behavior), which failed immediately at `env.reset()` with
"connection refused" (no actual cluster mutation) and left 3 more such stale
records, which also self-healed automatically. The generated
`persistent-data/demo_ui/*.log`/`*.json` files from that test run were
deleted before commit.

## tests
- demo_ui suite: `/tmp/rl-k8s-demo-ui-venv/bin/python -m pytest demo_ui/tests -q`
  → 128 passed, 0 failed (new `TestActiveRunLiveness` class: dead-pid,
  missing-pid+status, missing status file, stale status file,
  live-pid+fresh-status-file locks controls, no-history-deletion,
  multiple-stale-cleanup; plus a real-V2-controller-id pairing test).
- Repo unit suite: `.venv/bin/pytest tests/ -q` → 1445 passed, 2 skipped,
  0 failed.
- `ruff check demo_ui/` → all checks passed, no new lint issues.

## commit
`be53e99` "UI: fix stale active-run lock and live pairing" — 8 files changed
(`.gitignore`, `demo_ui/experiment_store.py`, `demo_ui/live_execution_adapter.py`,
`demo_ui/models.py`, `demo_ui/tests/test_experiment_store.py`,
`demo_ui/tests/test_incremental_telemetry.py`,
`demo_ui/tests/test_live_details_fresh_state.py`,
`demo_ui/tests/test_live_details_v2.py`). Not pushed, no Co-Authored-By
trailer.

## chain
Follows Phase 9 ("UI: single-system interface (remove version selector)",
commit `33d7a4a`), which flagged both bugs fixed here for follow-up.
