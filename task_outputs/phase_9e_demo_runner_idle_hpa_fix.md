# PHASE 9E — Demo runner: remove idle HPA before PPO run, always restore

## goal
Fix a live-only bug: starting a live PPO demo run via
`scripts/run_demo_episode_v2.py` failed immediately with `"cannot start
controller 'ppo': conflicting HPA object(s) present in cluster:
['demo-app-hpa-v2']"`. `demo-app-hpa-v2` present is the CANONICAL IDLE
state, not a conflict — PPO is a pure-Python controller that requires
zero HPA objects present, so the idle HPA must be deleted before PPO
starts, not treated as a blocking precondition. `scripts/
evaluate_v2_campaign.py` already solved exactly this for its own live
runs (CORR-63/CORR-64); the fix was never carried over to the separate
demo runner.

## root cause
`run_demo_episode_v2.py`'s `check_training_protection()` correctly allows
`demo-app-hpa-v2` through (it only refuses on an HPA *other* than
`demo-app-hpa-v2`). But nothing in the script ever deleted
`demo-app-hpa-v2` before calling `run_ppo_v2_eval_episode()`, whose
`acquire_controller(CONTROLLER_PPO, ...)` call (in
`autoscaler/src/rl_env/controller_exclusivity.py`) fails closed on ANY
HPA object for the PPO controller. `evaluate_v2_campaign.py`'s own
`build_live_context().ppo_episode` wrapper already deletes the HPA
immediately before acquiring the PPO controller, and its
`restore_canonical_idle_state()` reapplies `demo-app-hpa-v2` at 2
replicas afterward regardless of which arm ran — neither piece existed
in the demo runner. Separately, the demo runner had no SIGTERM handler
at all, so `demo_ui.live_execution_adapter.RealExperimentControllerV2.
stop()` (which sends SIGTERM to the whole process group) killed the
process with zero cluster cleanup.

## fix
- New `scripts/v2_live_cluster_state.py` (shared, minimal): one function,
  `remove_hpa_before_ppo_start()`, factored out of the pattern
  `evaluate_v2_campaign.py`'s `ppo_episode` wrapper already uses inline —
  idempotent delete of any HPA present, immediately before a `ppo_v2` run
  acquires the controller.
- `scripts/run_demo_episode_v2.py`:
  - Imports and REUSES `restore_canonical_idle_state` directly from
    `scripts/evaluate_v2_campaign.py` (not reimplemented) — one source of
    truth for "what the canonical idle cluster state is and how to reach
    it" across both live V2 entry points.
  - After the existing protection check passes, if `arm == "ppo_v2"`:
    calls `remove_hpa_before_ppo_start()` before the episode starts (the
    `hpa_v2_cpu_memory` arm is unchanged — it already applies its own
    HPA via its own `finally`).
  - The episode run is now wrapped in an outer `try/finally` whose
    `finally` is UNCONDITIONAL: it always calls
    `restore_canonical_idle_state()` (reapplies `demo-app-hpa-v2` at 2
    replicas, deletes any stray non-canonical HPA) and `clear_lock()`,
    on success, on any exception, and on an explicit STOP. Cleanup
    failures are caught and recorded, never left to crash the process or
    mask the original exception (mirrors `evaluate_v2_campaign.
    _safe_cleanup()`'s rule).
  - Added a `SIGTERM` handler (installed in `main()`) that raises a new
    `DemoStopRequested` exception instead of letting the OS kill the
    process with no cleanup — this routes an explicit STOP through the
    exact same `finally` cleanup path as any other exception. The
    protection-refusal path is unaffected: it still returns/raises
    before any cluster mutation, so a refused run still does zero cleanup
    and zero mutation.
  - The "unexpected HPA present → refuse to start" guard is untouched —
    still refuses before any mutation (regression-tested).

Note: `scripts/evaluate_v2_campaign.py` itself was intentionally left
unmodified (out of this phase's allowed scope) — its own inline
HPA-delete-before-PPO logic in `build_live_context()` still duplicates
what `v2_live_cluster_state.remove_hpa_before_ppo_start()` now does for
the demo runner; a follow-up could point it at the same shared helper.

## tests
`tests/unit/test_run_demo_episode_v2.py` (new test class,
`TestPpoRunRemovesIdleHpaFirstAndAlwaysRestoresIdleState`, all fakes — no
real cluster/kubectl/subprocess access):
- PPO run with `demo-app-hpa-v2` present deletes it, then starts (no
  refusal), in the order remove → run → cleanup.
- Successful PPO run's cleanup reapplies the HPA and releases the lock.
- An injected exception mid-run still runs cleanup before propagating.
- An injected `DemoStopRequested` (simulating the SIGTERM handler firing)
  still runs cleanup, writes `status: "stopped"`, then propagates.
- The `hpa_v2_cpu_memory` arm never calls the PPO pre-start helper, but
  cleanup still runs.
- Regression: an HPA other than `demo-app-hpa-v2` present still refuses
  to start, with zero mutation and zero cleanup calls.

- `.venv/bin/pytest tests/ -q` → 1940 passed, 2 skipped, 0 failed.
- `/tmp/rl-k8s-demo-ui-venv/bin/python -m pytest demo_ui/tests -q` → 252
  passed, 0 failed (demo_ui untouched by this phase; run to confirm
  nothing broke).
- `.venv/bin/ruff check scripts/ demo_ui/` → all checks passed, no new
  lint issues.

## commit
`e3d8996` "Demo runner: remove idle HPA before PPO run, always restore"
— 3 files changed (`scripts/run_demo_episode_v2.py`, new
`scripts/v2_live_cluster_state.py`, `tests/unit/
test_run_demo_episode_v2.py`). Not pushed, no Co-Authored-By trailer.

## chain
Mirrors the CORR-63/CORR-64 fix already made in
`scripts/evaluate_v2_campaign.py` (see
`docs/experimental_design_corrections.md`) for its own live evaluation
campaign runs, now carried over to the separate demo runner
(`scripts/run_demo_episode_v2.py`) that drives the Streamlit UI's V2 Live
view.
