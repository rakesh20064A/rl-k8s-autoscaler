# demo_ui

Additive, offline demo/UI layer for the RL-vs-HPA autoscaler project. Fully
isolated from the governed scientific core (`scripts/train_ppo_live_session.py`,
`scripts/traffic_orchestrator.py`, PPO reward/observation/action logic,
Kubernetes/HPA manifests). Nothing in this package imports from, invokes, or
modifies any of those.

## Status: offline UI/component development only

This phase ships:

- Pure, offline, fully-unit-tested logic: `models.py`, `state_machine.py`,
  `comparison_engine.py`, `conclusion_engine.py`, `fixtures.py`.
- A data-provider abstraction (`providers.py`) with offline fixture
  implementations only (`Fixture*`). No live Kubernetes/Prometheus/PPO/HPA
  call exists anywhere in this package.
- Streamlit/Plotly UI code (`app.py`, `components/`, `views/`) that requires
  `streamlit`/`plotly`, which are not installed in this repo's scientific
  `.venv` by design (see Future dependency requirement below).

## Future dependency requirement

Before running the UI, add to the project's dependency manifest and install
into `.venv` (not done automatically by this phase, per explicit
instruction):

```
streamlit>=1.37
plotly>=5.20
```

Then launch, **only when no live rollout/training process is active**, from
the repo root with `demo_ui/` on `PYTHONPATH` so a forced-light theme is
honored regardless of the viewer's OS/browser dark-mode preference
(`demo_ui/.streamlit/config.toml` is only auto-discovered when `demo_ui/` is
the current working directory or via this env var):

```
STREAMLIT_THEME_BASE=light PYTHONPATH=. /path/to/ui-venv/bin/streamlit run demo_ui/app.py
```

or, equivalently, with `demo_ui/` as the working directory:

```
cd demo_ui && PYTHONPATH=.. /path/to/ui-venv/bin/streamlit run app.py
```

## Running tests

The offline logic is fully testable without Streamlit/Plotly:

```
.venv/bin/python -m pytest demo_ui/tests -q
```

## Architecture

```
demo_ui/
├── app.py                 Streamlit entrypoint (wires Fixture* providers)
├── theme.py                Palette + CSS (lab-instrumentation visual language)
├── models.py                Pure dataclasses/enums shared by everything
├── state_machine.py         SingleControllerRun (Auto OFF) / AutoComparisonRun (Auto ON)
├── comparison_engine.py     Pure HPA-vs-PPO metric comparison
├── conclusion_engine.py     Deterministic, evidence-traceable conclusion text
├── fixtures.py               Deterministic demo data generators
├── providers.py              Protocol interfaces + Fixture* implementations
├── components/               Streamlit building blocks (controls, cards, charts, logs, pods, ...)
├── views/                     Streamlit pages (experiment, live_details, comparison, history) --
│                              named `views/`, not `pages/`, so Streamlit's auto-detected native
│                              multipage navigation never activates (it triggers on any sibling
│                              `pages/` directory next to the entrypoint script)
├── .streamlit/config.toml     Forces a light base theme (fixes dark-mode text-contrast bugs)
└── tests/                     Offline unit tests (no Streamlit/Plotly import required)
```

## Data provider abstraction (sec.18)

`providers.py` defines four `Protocol`s: `MetricsProvider`, `ClusterProvider`,
`ExperimentController`, `ResultStore`. UI components depend only on these
protocols, never on a concrete implementation. This phase wires in:

- `FixtureMetricsProvider`
- `FixtureClusterProvider`
- `FixtureExperimentController`
- `FixtureResultStore`

A future phase can implement `PrometheusMetricsProvider`,
`KubernetesClusterProvider`, a live `ExperimentController` (reusing the
project's existing detached-supervisor pattern — tracked Prometheus
port-forward, tracked caffeinate, HPA removed only after dependencies are
verified healthy, PPO launched via `start_new_session=True`, signal-safe
cleanup), and a real `ResultStore` reading/writing the
`experiments/<id>/{hpa,ppo}/...` layout — **without changing any UI
component**, by swapping the provider passed into `app.py`/the pages.

## Hard safety boundary (this phase)

No file in this package performs, at any point:

- `kubectl` / Docker / Minikube subprocess calls
- binding or connecting to localhost port 9090
- starting/stopping the traffic generator
- starting/stopping PPO training
- creating/deleting/scaling the HPA or Deployment
- writing to `persistent-data/`, episode ledgers, or checkpoints

All data is either pure in-memory computation or `demo_ui/fixtures.py`
deterministic generators, clearly labeled `is_demo_data=True` throughout the
UI (`OFFLINE UI DEVELOPMENT MODE` banner, `DEMO DATA` captions on fixture
schedule hashes and conclusions).
