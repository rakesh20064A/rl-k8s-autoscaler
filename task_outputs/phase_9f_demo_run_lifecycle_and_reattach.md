# PHASE 9F — Demo runs: robust lifecycle, UI re-attach, cleanup, recover

## goal
Permanently fix the recurring "LIVE run status: UNKNOWN, all controls
disabled forever" bug: a live PPO demo run (PID 20102, profile
`demo_constant_cpu_normal`) was started from the Streamlit UI and
completed successfully, but the UI was restarted mid-run, and the new UI
instance could never re-attach to it. Separately, a stale record had
survived a host restart in the past and locked the controls forever. This
had recurred across multiple "Phase 9" iterations; this phase was meant
to be the permanent fix for the whole lifecycle, not another patch.

## root cause
Two independent bugs compounded:

1. **`experiment_store.active_run()` never consulted the run's own status
   file's content.** It only checked (a) whether the stored pid was alive
   and (b) whether the status file's mtime was "fresh" (<120s). When a
   record's pid was dead and its status file was stale-by-mtime,
   `active_run()` unconditionally rewrote it to `"INTERRUPTED (stale)"` —
   even when the status file plainly said `{"status": "completed", ...,
   "summary": {..., "n_samples": 16}}`. This is exactly what happened to
   run `4d4edc8274e6`: PID 20102 had long since exited successfully, but
   the store record still said `status="running"`, and the OLD code would
   have thrown away its real "completed" outcome the next time anyone
   looked.
2. **`RealExperimentControllerV2` (the object that actually drives the
   status panel/log output/STOP) lived in `st.session_state` and had no
   way to re-hydrate itself from disk.** A fresh instance (no `_proc`, no
   `run_id`) created after a Streamlit restart returned `{"status":
   "unknown"}` from `.status()` unconditionally — this is the literal
   "UNKNOWN" the user saw. Separately, the control-strip's
   RUNNING-badge/START-vs-STOP toggle (`components/controls_v2.py`) was
   driven by a session-scoped flag (`du_v2_run_in_progress`), not by the
   same `experiment_store`-backed signal that correctly disabled the
   selects — so a reattached session's selects were locked but STOP was
   never shown.

## fix
**`demo_ui/experiment_store.py`** (single source of truth, moved to
`persistent-data/demo_ui/runs.json`, migrated in place from the old
`demo_ui/.run_store/runs.json` on first import):
- `_reconcile_running()` (backing both `active_run()` and a new public
  `reconcile()`) now checks a running record's OWN status file for a
  terminal outcome (`completed`/`error`/`stopped`/`refused`) BEFORE any
  liveness check, and copies that real outcome in — this is the direct
  fix for the `4d4edc8274e6` scenario. Only when no terminal outcome is
  recorded does it fall through to the pid/heartbeat/expected-duration
  liveness check, and only then mark `STALE_STATUS`, now with a
  human-readable reason.
- `is_run_alive()` gained a third check: elapsed time must not exceed
  `expected_duration_s + 300s` (a new field, populated at start from the
  profile's own declared duration).
- New `pgid`/`log_path`/`expected_duration_s` fields on every run record
  (task sec.1).
- `latest_compatible_pair()` now requires `status == "completed"` exactly
  for a pairing candidate (not just "terminal") — a real gap this
  phase's own regression test caught: `STALE_STATUS` was already in
  `TERMINAL_STATUSES` (for History), so an interrupted run could
  previously be offered as "the" result for its controller family.
  `completed_runs()`/History is unaffected (still shows every terminal
  status).

**`demo_ui/live_execution_adapter.py`**:
- New `RealExperimentControllerV2.attach(rec)`: re-hydrates `run_id`/
  `_status_path`/`_log_path` from a persisted record with no `_proc`
  required — this is the re-attach fix.
- `stop()` rewritten to resolve pid/pgid from the STORE record first
  (works for a run this instance never launched), SIGTERM the process
  group, wait up to 30s, SIGKILL as a last resort. Fixed a genuine latent
  bug found while testing this: the wait loop used to check
  `_pid_alive()` even when this instance held the real `Popen` handle —
  a terminated-but-unreaped child is a zombie whose pid still answers
  `kill(pid, 0)`, so every same-session STOP used to spin the full 30s
  wait before falling through to `self._proc.wait()`, which is what
  actually reaps it. Now uses `.poll()` (which also reaps) whenever
  `_proc` is held, and the pid check only for a truly re-attached run.
- `start()` now stores `pgid` (using `self._proc.pid` directly, NOT a
  racy `os.getpgid()` call immediately after `Popen()` — the child's own
  `setsid()` runs independently of the parent and might not have
  completed yet, which could otherwise read back the PARENT's pgid) and
  `expected_duration_s` (read from the profile file, best-effort).

**`demo_ui/components/controls_v2.py`**: the RUNNING badge / START-vs-STOP
toggle now derives from `active_rec` (same signal that disables the
selects) whenever live, not from the `du_v2_run_in_progress` session key
alone.

**`demo_ui/views/experiment.py`**: `_render_live_status_panel` (the
`st.fragment`) now calls `experiment_store.reconcile()` every tick,
re-attaches `real_ctrl` to whatever the store says is active (or falls
back to the latest v2 run so a just-finished outcome still shows), and
triggers a live-cluster idle-state check (`ensure_canonical_idle_state`)
exactly when reconciliation finds a run newly stale with no process left
alive. Re-attach is also done synchronously before handling a STOP click
(defensive — Streamlit's own execution order means the fragment already
attaches before STOP is clickable, but costs nothing to guarantee). New
RECOVER button, enabled only when no runner process is alive.

**`demo_ui/views/live_details_v2.py`**: fixed a REAL crash this phase's
own fix exposed — a genuine completed run's `ppo_decision.
action_probabilities` is a raw index-ordered list (`autoscaler.src.
rl_env.types.ACTION_NAMES` order), never a name-keyed dict; the panel
called `.items()` on it unconditionally. (Previously unreachable because
no real completed-with-samples v2 run had ever survived long enough to
be shown here — exactly the bug this phase fixes.) Also broadened
`ACTION_COLORS` to cover both the real runner's `_ONE`-suffixed names and
the DEMO fixtures' shorter names.

**`scripts/run_demo_episode_v2.py`**: terminal-status ordering fixed to
match task sec.3 exactly — (a) the run summary is written first (status
stays `"running"`/stage `"FINISHING"` so RE-ATTACH can show the full
outcome while the UI's controls stay locked), (b) cleanup
(`restore_canonical_idle_state`/`clear_lock`) runs, (c) the TERMINAL
status is written LAST, with the cleanup result embedded, so a run is
never shown as COMPLETED before the cluster is actually back to idle.
Every terminal payload now carries a `partial` flag on its summary (only
`False` for a genuine full completion). An unanticipated exception (none
of the specific `except` clauses) now still gets a real terminal "error"
status instead of leaving the status file stuck mid-run. Added a SIGINT
handler (same as the existing SIGTERM one) and an `atexit` safety net.
Decided the existing status-file-rewrite-per-decision (`EPISODE_RUNNING`,
rewritten every ~30s decision step with cumulative samples) already
satisfies "incremental, readable by RE-ATTACH before completion" — no
separate samples JSONL wiring added (the adapter never passed
`--samples-jsonl` to begin with; the inline approach is the smaller,
sufficient change).

**`scripts/v2_live_cluster_state.py`**: added
`no_runner_process_alive()` (fails closed), `cluster_is_canonical_idle()`,
and `ensure_canonical_idle_state()` — shared RECOVER logic used by both
the UI and the new terminal script.

**`scripts/demo_recover.py`** (new): terminal equivalent of the UI's
RECOVER button — `experiment_store.reconcile()` then, only if no runner
is alive, `ensure_canonical_idle_state()`.

## a real production side effect found and fixed mid-phase
While running the full `demo_ui/tests` suite repeatedly during this
phase, a REAL subprocess run was discovered live against the real
cluster (PID 68483, same `ppo_v2`/`demo_constant_cpu_normal` combination).
Root cause: `demo_ui/tests/test_app_smoke.py::
test_start_button_does_not_crash_single_run` clicked the FIRST button on
the Experiment page without ever switching off the page's default "Live
cluster" data source — in an environment where the trained model +
epsilon calibration files already exist (as they do here), that click
drove the real `RealExperimentControllerV2.start()`, launching
`scripts/run_demo_episode_v2.py` in the scientific `.venv` with real
kubectl access. This was pre-existing (not introduced by this phase's
changes) but newly noticed here. Fixed by switching that one test to
Sample data (offline) mode first, matching every other START-click test
in the suite (`test_arm_dropdown.py`, `test_single_system_ui.py`,
`test_demo_traffic_catalog_ui.py` all already did this correctly). The
in-flight real run was left to finish naturally (its own unconditional
cleanup is exactly what this phase hardened) rather than force-killed;
verified afterward that it completed cleanly and the cluster returned to
canonical idle state on its own.

## tests
New/updated, all fakes except where noted:
- `demo_ui/tests/test_experiment_store.py`: regression test reproducing
  the exact `4d4edc8274e6` scenario (dead pid + stale-by-mtime status
  file whose OWN content says `"completed"` → must surface COMPLETED,
  never `STALE_STATUS`), `reconcile()` reporting, `refused`-status
  surfacing, a genuinely-dead-with-no-terminal-file case still marked
  stale with a reason, expected-duration liveness, store
  location/migration, and the `latest_compatible_pair()` pairing-gap
  regression above.
- `demo_ui/tests/test_live_execution_adapter_v2.py`: `attach()`
  hydration, STOP via a re-attached (no-`_proc`) run using a real
  disposable `sleep` subprocess + pgid, and the same-session
  zombie/`.poll()` STOP-latency regression.
- `demo_ui/tests/test_ui_reattach.py` (new): end-to-end `AppTest`
  coverage — fresh session re-attaches to a genuinely active run (real
  stage shown, STOP button present, STOP actually works), fresh session
  after the run finished while the UI was closed (COMPLETED shown,
  controls unlocked — the UI-level version of the regression above), and
  a host-restart/missing-status-file case unlocking controls.
- `demo_ui/tests/test_app_smoke.py`: fixed per the side-effect above.
- `tests/unit/test_run_demo_episode_v2.py`: terminal-status-after-cleanup
  ordering (peeked at mid-cleanup via a fake `restore_canonical_idle_state`),
  unexpected-exception still produces a terminal error with partial
  summary, stopped-run summary marked partial, and `main()`'s SIGINT/
  atexit registration.
- `tests/unit/test_v2_live_cluster_state.py` (new): the 3 new RECOVER
  helpers, fails-closed on a `pgrep` failure, never mutates while a
  runner is alive.
- `tests/unit/test_demo_recover.py` (new): the script's `run_recover()`
  and `main()`.
- Verified: `scripts/generate_demo_profiles.profile_slug`/
  `parse_profile_name` round-trip correctly for all 36 pattern x
  resource x load combinations including Bursts/CPU/High
  (`demo_bursts_cpu_high`) — already covered by
  `demo_ui/tests/test_demo_traffic_catalog_ui.py`; the PID 20102 run's
  `demo_constant_cpu_normal` profile was simply what was selected (the
  defaults), not a mapping bug.

- `/tmp/rl-k8s-demo-ui-venv/bin/python -m pytest demo_ui/tests -q` → 272
  passed.
- `.venv/bin/python -m pytest demo_ui/tests tests/unit -q` → 2025 passed,
  13 skipped (streamlit-dependent files, by design, under the scientific
  venv).
- `.venv/bin/ruff check` on every changed/new file → all checks passed.

## recover (step 9)
Confirmed no `run_demo_episode_v2.py` process alive, then ran
`.venv/bin/python scripts/demo_recover.py` for real:
```json
{
  "active_run": null, "newly_terminal": [], "newly_stale": [],
  "runner_process_alive": false,
  "cluster_cleanup": {"ran": false, "reason": "cluster already in the canonical idle state"}
}
```
Final cluster state: only `demo-app-hpa-v2` HPA present (2/8 replicas,
2 current), controller lock `none`, `demo-app` deployment 2/2 ready —
already canonical idle (the real PID 68483 run's own cleanup, hardened
by this phase, had already restored it).

## commit
`9c991a8` "Demo runs: robust lifecycle, UI re-attach, cleanup, recover" —
15 files changed (1485 insertions, 101 deletions): `demo_ui/
experiment_store.py`, `demo_ui/live_execution_adapter.py`, `demo_ui/
components/controls_v2.py`, `demo_ui/views/experiment.py`, `demo_ui/
views/live_details_v2.py`, `scripts/run_demo_episode_v2.py`, `scripts/
v2_live_cluster_state.py`, new `scripts/demo_recover.py`, and their
tests (including 3 new test files). Not pushed, no Co-Authored-By
trailer.

## chain
Builds directly on Phase 9E (`e3d8996`, idle-HPA-removal + unconditional
cleanup for `ppo_v2`) and Phase 9B (`demo-ui-stale-lock-and-pairing-fix`)
— this phase closes the remaining gap both left: the STORE/UI layer never
consulted a run's own status file as ground truth, and the UI's own
in-memory controller object had no way to resume tracking a run across a
restart at all.
