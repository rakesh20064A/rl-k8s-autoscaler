# Part 13 — Demo UI

VERIFIED/FROM_CODE via research agent this session.

## Structure

`demo_ui/`: `README.md`, `app.py` (Streamlit entrypoint), `comparison_engine.py`,
`conclusion_engine.py`, `experiment_store.py` (untracked), `fixtures.py` +
`fixtures/`, `live_adapter.py` (old, dry-run-only, V1), `live_execution_adapter.py`
(untracked, real subprocess launcher), `live_providers.py`, `models.py`,
`providers.py`, `requirements-ui.txt`, `state_machine.py`, `theme.py`,
`components/` (charts, controller_status, controls, layout, logs,
metric_cards, pods, provenance), `views/` (comparison, experiment, history,
live_details), `tests/` (19 files), `.streamlit/config.toml`,
`.run_store/runs.json` (gitignored).

## Entry point and "Start" mechanics

`app.py` is a plain Streamlit router (Experiment/Live Details/Comparison/
History pages) with an explicit "HARD BOUNDARY" docstring comment stating
this module must never itself start a port-forward/kubectl/traffic-
generator/PPO/HPA call directly.

Real "Start" (LIVE + Manual mode only, in `views/experiment.py`) calls
`live_execution_adapter.RealExperimentController.start()`, which:
1. Validates `.venv/bin/python` exists — **currently FAILS** (see
   06_python_env.md; raises `RuntimeError(...run scripts/bootstrap.sh
   first)`).
2. Launches `scripts/run_demo_episode.py` as a subprocess (own process
   group, `start_new_session=True`, so STOP can `killpg` it and its
   traffic-generator child together).
3. `run_demo_episode.py` reuses V1's already-tested
   `evaluate_ppo_vs_hpa.py`'s `run_ppo_episode`/`run_hpa_episode` against
   one of the 6 V1 `TRAINING_PROFILES` or the demo-only
   `demo_smoke_test` profile — never the frozen V1 holdout (firewalled via
   `rl_live_common.assert_training_eligible`).
4. Progress polled via a small JSON status file; a `st.fragment(run_every=3)`
   auto-refreshing panel reads it without re-triggering START.
DEMO mode (and LIVE+Auto, not yet wired) instead drives pure-offline
`SingleControllerRun`/`AutoComparisonRun` state machines via
`FixtureExperimentController` — no subprocess, no real infrastructure.

## V1-only / not V2-aware — confirmed

`grep -rn "rl_v2|environment_v2|train_ppo_final_v2|train_ppo_live_v2|
config_v2|reward_v2" demo_ui/ --include="*.py"` (excluding tests): **zero
matches**. `run_demo_episode.py`'s default model path is
`models/ppo/ppo_model.zip` (the V1 model; `models/ppo_v2/` is empty).
**The demo UI has no knowledge of V2 at all** — its LIVE execution path
only exercises V1 evaluation mechanics, V1 profiles, and the V1 model.

## Uncommitted UI files (purpose)

- `experiment_store.py` — single JSON-file-backed source of truth for real
  LIVE run records, shared by Experiment/Live Details/History/Comparison;
  deliberately not `st.session_state` (doesn't survive restarts/navigation).
- `live_execution_adapter.py` — the one module allowed to launch a real
  subprocess, kept separate from the dry-run-only `live_adapter.py` by
  design (enforced by a dedicated test asserting no real-mutation path
  exists in the old module's source).
- 8 new test files covering: empty-state rendering before any real run,
  widget-state persistence across navigation, `experiment_store`'s own
  lifecycle functions, Experiment-as-sole-control-plane integration,
  fixture lazy opt-in (Comparison/History render nothing until an explicit
  "Load demo fixture data" click), incremental telemetry streaming (not
  only at episode completion), fresh-session empty-state rendering, and
  single-data-point chart regression coverage.

## TODO/FIXME within demo_ui

Only 3 hits, all in `demo_ui/providers.py` (lines 11, 65, 82) — describing
a NOT-YET-BUILT future live `ExperimentController`/`ResultStore`
implementation that would reuse "the project's existing detached-supervisor
pattern" — now partially superseded by the newer
`live_execution_adapter.py`/`experiment_store.py` work (which implements a
real, if currently-broken-by-missing-`.venv`, live path). No other
TODO/FIXME/XXX/known-bug markers found anywhere else in `demo_ui/`.
