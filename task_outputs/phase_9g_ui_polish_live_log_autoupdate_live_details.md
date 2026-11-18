# PHASE 9G — UI polish: live log, auto-update, clearer Live Details

## goal
Phase 9F fixed the demo run lifecycle (store reconciliation, re-attach,
STOP, RECOVER) but left several UI-polish issues observed on the real
`demo_bursts_cpu_high` PPO/HPA run pair it produced: the Experiment page
kept showing STOP after a run completed (needed a manual refresh), the
"Recent log output" panel stayed empty until the run finished, Live
Details showed both an "HPA not available" and a "PPO not available"
panel simultaneously, and stored run records had `arm=None` despite
`controller` being set correctly. This phase fixes all of these plus adds
a Live Details run selector, without starting any new live run against
the real cluster.

## root cause (STOP button persisting after completion)
`demo_ui/components/controls_v2.py::render_controls_v2()` computed
`already_running = controls_disabled or st.session_state.get(
"du_v2_run_in_progress", False)`. `controls_disabled` (derived from
`experiment_store.active_run()`, the Phase 9F signal) correctly flipped
back to `False` the moment a live run's own status file recorded a
terminal outcome. But `du_v2_run_in_progress` was a session_state flag
set to `True` once, at START-click time (`views/experiment.py`), and was
only ever reset back to `False` by an explicit STOP-button click — never
when a run finished on its own. Because of the `or`, `already_running`
stayed `True` forever after any live run completed naturally, even on a
full-app rerun that had already re-synced everything else (the selects,
the status text, the store record). This is exactly the same class of
bug Phase 9F fixed for the selects' disabled state but missed for the
button label itself.

**Fix:** for the LIVE path (`is_live=True`), `already_running` is now
derived ONLY from `controls_disabled` (the store-backed signal) — the
session_state flag is never consulted. For DEMO mode (`is_live=False`,
no persisted record exists for a loaded sample fixture), the session key
is still the only available signal and is used exactly as before.

## changes

**1. Live log, runner side (`scripts/run_demo_episode_v2.py`, logging
only — no behavior change):** found that the runner wrote almost nothing
to stdout during a run (only a final JSON summary print, or a stderr
error line) — the log file the UI tails was effectively empty until
completion. Added a `_log()` helper (`print(..., flush=True)` with a
`HH:MM:SS` prefix) and wired it in at every meaningful milestone:
preflight result (OK/FAILED with reason), idle-HPA removal / controller
lock acquisition, "Traffic started: Pattern / Resource / Load (duration
s)", one line per decision (`Decision i/N: ACTION | replicas a→b | X
req/s | p95 Y ms | CPU/replica | mem/replica | tie rule`) or per HPA step
(`Step i/N: HPA desired D, current C | CPU X%/260% | memory Y%/120% |
p95 Z ms`) via a new `_log_decision()`, traffic-stopped/cleanup-starting,
cleanup result, and the final `status=...` line. Every line is flushed
immediately (on top of the existing `-u`/`PYTHONUNBUFFERED=1`), so a
tailing reader sees them as they happen, not batched at exit.

**2. Live log, UI side (`demo_ui/views/experiment.py`):** the "Recent log
output" expander already tailed the log file via `real_ctrl.
recent_log_lines()` inside the 3s-polling fragment — the actual gap was
the runner writing nothing (fixed above). Bumped the tail length from the
default 20 to the spec's ~30 lines (`recent_log_lines(n=30)`); the
expander already opens by default while `real_ctrl.is_running`.

**3. Auto-update / STOP→START root cause (see above) plus "stop polling
once terminal":** `st.fragment(run_every=...)` can't have its interval
changed once applied as a decorator, so both `_render_live_status_panel`
(Experiment) and the new Live Details rendering are now wrapped
FUNCTIONALLY at the call site — `st.fragment(run_every=X)(fn)(args)` —
with `X = 3` (Experiment) / `4` (Live Details) while a run is active and
`None` otherwise. The interval is recomputed fresh on every full-app
rerun from the same store-backed signal the button uses, so polling
genuinely stops (no more fragment reruns at all) once a run goes
terminal, and resumes automatically the next time a run becomes active.

**4. Live Details (`demo_ui/views/live_details_v2.py`,
`demo_ui/views/live_details.py`):** added a run selector
(`render_run_selector()`/`run_selector_label()`) at the top of Live
Details — defaults to the active run if one exists, else the latest
completed run; label format exactly `"PPO · Bursts / CPU / High · 18 Nov
12:40 · completed"` (controller family via `display_arm_label()`,
traffic via a new shared `_traffic_label()` helper, day-month-time with
no leading zero on the day, lowercase status). No selector widget at all
when the store is empty — `render_run_banner(None, ...)` shows "No
experiment has been run yet." plus a new `st.info("Start a run on the
Experiment page")`. `render_decision_or_hpa_panel()` now branches on
`experiment_store._controller_family()` (substring match, same logic as
Comparison pairing and the `arm` backfill) instead of an exact
`== "ppo_v2"` check, and shows exactly one panel. The actual source of
the "HPA not available"/"PPO not available" pair showing simultaneously
turned out to be a SEPARATE section in `live_details.py`'s
`_render_current_cluster_state()` ("Current Cluster State" — the live
cluster's state right now, independent of any run) which unconditionally
rendered an "HPA" sub-panel (often `NOT_AVAILABLE` whenever a PPO run had
the cluster, since PPO requires zero HPAs) directly alongside a
hardcoded "PPO: Not available — no live PPO decision feed wired up"
line. Removed both; that section now shows the real HPA snapshot only
when one happens to be present, with no permanent "PPO" placeholder (the
run-specific decision panel already covers PPO, per-run, above). Added a
one-line caption under each of the 4 telemetry charts (e.g. "p95
latency; dashed line = 1s target."). Labels are "PPO"/"HPA" throughout
(via `display_arm_label()`), never the raw controller id.

**5. Store `arm` field (`demo_ui/experiment_store.py`):** added
`_arm_label()` (same `_controller_family()` substring logic, uppercased)
and wrote it in `start_run()` (at creation) and `update_run()` (whenever
`controller` is (re)set and the caller didn't already pass `arm`
explicitly), so `arm` can never drift out of sync with `controller`. Added
a one-time `_backfill_arm_field_once()` migration, run at import time
(after `_controller_family()` is defined) against the real
`persistent-data/demo_ui/runs.json` — confirmed it correctly backfilled
the real production records from the Phase 9F run pair (`92c6609f5684`
→ `arm="PPO"`, the following `79ade9f37acb` → `arm="HPA"`), mutating only
the record's own fields, never `summary`/samples content.

**6. Comparison:** `demo_ui/views/comparison.py` already hardcodes
"HPA"/"PPO" labels tied directly to `pair["hpa"]`/`pair["ppo"]` (itself
unchanged, controller-family-based pairing from Phase 9F) — verified this
renders correctly with `arm` now populated; no code change needed there.
Added a regression test confirming both paired records carry the correct
`arm` value.

## root_cause_stop_button
See above (same text, for the terse report format): `already_running =
controls_disabled or session_state["du_v2_run_in_progress"]` — the
session flag was set once at START-click and only ever cleared by an
explicit STOP click, never by a run finishing on its own, so it stayed
`True` forever after natural completion even though `controls_disabled`
had correctly flipped back to `False`. Fixed by deriving `already_running`
ONLY from `controls_disabled` for the LIVE path.

## tests
New/updated (all fakes except real disposable `sleep` subprocesses for
pid/pgid liveness, same style as Phase 9F):
- `demo_ui/tests/test_ui_reattach.py`:
  `TestLiveLogTailingDuringAnActiveRun` (log lines visible while status is
  still non-terminal, never only after completion) and
  `TestButtonFlipsToStartAfterCompletionWithoutManualRerun` (the key
  regression test — seeds an active run with the session's
  `du_v2_run_in_progress` flag set as a real START click would, completes
  the run via its own status file exactly as the runner's cleanup leaves
  it, reruns the SAME `AppTest` session with no fresh `AppTest.from_file`
  call, and asserts START shows, STOP does not). Verified this test fails
  against the pre-fix `or`-based logic and passes against the fix.
- `demo_ui/tests/test_experiment_store.py`: `TestArmFieldBackfill` (
  `start_run`/`update_run` write `arm`; one-time backfill fills existing
  `None` records from `controller` including a forward-compatible
  future-controller-id case, leaves already-set/unrecognized records
  alone, never touches `summary`; idempotent) and
  `test_pair_members_both_have_the_arm_field_populated` under
  `TestCompatiblePairing`.
- `demo_ui/tests/test_live_details_v2.py`: `TestRunSelectorLabelFormat`
  (exact label string for a 36-catalog PPO/completed and a
  legacy-pattern HPA/running record), `TestRunSelectorDefaultsAndNoRedundantPanels`
  (selector defaults to the active run; no "no live PPO decision feed"
  text for either a completed PPO or completed HPA run; no selector
  widget at all when the store is empty).
- `tests/unit/test_run_demo_episode_v2.py`: `TestLiveLogLines` (decision
  lines for `ppo_v2` and step lines for `hpa_v2_cpu_memory` appear via
  `capsys`, in the expected format, strictly BEFORE the run-finished
  line — i.e. incremental, not batched; preflight-failure line carries
  the real reason; `_traffic_description()` for both a catalog and a
  legacy profile name).

Results:
- `/tmp/rl-k8s-demo-ui-venv/bin/python -m pytest demo_ui/tests -q` → 286
  passed (was 272 before this phase).
- `.venv/bin/python -m pytest demo_ui/tests tests/unit -q` → 2036 passed,
  13 skipped (streamlit-dependent files, by design, under the scientific
  venv).
- `.venv/bin/ruff check` on every changed/new file → all checks passed.

## commit
"UI polish: live log, auto-update, clearer Live Details" — 10 files
changed: `demo_ui/components/controls_v2.py`, `demo_ui/
experiment_store.py`, `demo_ui/views/experiment.py`, `demo_ui/views/
live_details.py`, `demo_ui/views/live_details_v2.py`, `scripts/
run_demo_episode_v2.py`, and their tests (`demo_ui/tests/
test_experiment_store.py`, `demo_ui/tests/test_live_details_v2.py`,
`demo_ui/tests/test_ui_reattach.py`, `tests/unit/
test_run_demo_episode_v2.py`). Not pushed, no attribution trailer (per
this task's explicit instruction, overriding the usual convention).

## chain
Builds directly on Phase 9F (`9c991a8`) — this phase polishes the UI
surface of the lifecycle Phase 9F made correct underneath: the store's
`active_run()`/`reconcile()` signal was already right, but the button
label, the log panel, and the Live Details page hadn't all been wired to
actually reflect it.
