# RL-Based Adaptive Auto-Scaling for Kubernetes: A Comparative Study Against HPA

BTech capstone project evaluating whether a reinforcement-learning (PPO)
controller — observing 11 multi-resource signals (CPU, memory, request
rate, latency, replica state, and their short-horizon trends) — can match
or improve on Kubernetes' built-in Horizontal Pod Autoscaler (HPA) for
scaling a live, containerized FastAPI workload on a real Minikube cluster.

Three controllers are compared head-to-head on identical, held-out traffic:

| Arm | Controller | Signal |
|---|---|---|
| **PPO (multi-metric, 11 signals)** | direct `kubectl scale`, PPO policy | CPU, memory, request rate, latency, replica state + trends |
| **HPA CPU+Memory (resource baseline)** | `k8s/hpa/hpa_v2_cpu_memory.yaml` | CPU and memory utilization |
| **HPA CPU-only (Kubernetes default)** | `k8s/hpa/hpa_v2_cpu_only_min2.yaml` | CPU utilization only |

## V1 → V2 history, in brief

The project's first iteration (V1) trained a CPU-only-observation PPO
policy and froze it at a reduced training budget
(`session_019`, 5,120 live timesteps) after live-cluster wall-clock
constraints cut training short; its final evaluation produced only 1 of
3 required valid PPO repeats, so no formal PPO-vs-HPA verdict could be
drawn (see `docs/viva_results_summary.md`).

V2 is a from-scratch redesign, not a patch: a new 11-feature
multi-resource observation space (`configs/rl_v2.yaml`,
`docs/rl_mdp_spec.md`), a 7-component reward (`configs/reward_v2.yaml`),
a safety-aware tie-break rule for near-tied policy outputs (CORR-58,
`autoscaler/src/rl_env/action_selection.py`), a raised replica floor of 2
(CORR-55, after two V1-era zero-ready-replica failures at a floor of 1),
and a third arm (`HPA CPU-only`, CORR-62) added purely for comparison
against the Kubernetes-default configuration. V2 training completed its
full amended 48-session budget and its evaluation campaign reached all 27
pre-registered valid repeats (see Training summary and Results summary
below). Every V2 design decision and mid-course correction is recorded,
append-only, in `docs/experimental_design_corrections.md` (CORR-1 through
CORR-65).

## Architecture

```
Traffic Generator (loadgen/) --> demo-app (app/) --> Prometheus (k8s/monitoring/)
                                       ^                        |
                                       |                        v
                 HPA (k8s/hpa/)  <--- OR --->  PPO Agent (autoscaler/src/rl_env/)
                                       |
                                 Kubernetes Deployment (k8s/app/)
```

Full diagram: `docs/figures/diagram_system_architecture.{svg,pdf,png}`
(Minikube cluster, Metrics Server at a 15s resolution, Prometheus,
host-side load generator/relay, and the PPO agent's
observation-builder → policy → safety-layer → executor pipeline, gated by
a controller-exclusivity lock so the PPO agent and either HPA arm are
never both active on the same Deployment at once). The RL decision loop
itself (observation → policy → safety gating → Kubernetes API → reward)
is diagrammed separately in `docs/figures/diagram_rl_control_loop.*`, and
the end-to-end experiment pipeline (training → pre-registered evaluation
→ exploratory stress test) in `docs/figures/diagram_experiment_pipeline.*`.

## Repo structure

- **`app/`** — the benchmark FastAPI application (`/work` CPU-bound
  endpoint, `/health`, Prometheus `/metrics`).
- **`loadgen/`** — deterministic, seeded traffic generator producing
  reproducible request schedules.
- **`autoscaler/src/rl_env/`** — the Gymnasium-compatible RL environment:
  observation/action/reward (V1 and V2), safety-aware action selection,
  controller-exclusivity locking, the live Kubernetes backend.
- **`scripts/`** — V2 training (`train_ppo_final_v2.py`), evaluation and
  stress-test campaign orchestration (`evaluate_v2_campaign.py`),
  analysis (`analyze_v2_evaluation.py`), and figure/table generation
  (`generate_v2_phase6e_figures.py`, `generate_v2_paper_tables.py`,
  `paper_style.py`), plus cluster bootstrap/deploy helpers.
- **`k8s/`** — Kubernetes manifests: `app/` (Deployment/Service),
  `hpa/` (all HPA variants, V1 and V2), `monitoring/`
  (kube-prometheus-stack values), `namespaces/`, `logging/`,
  `metrics-server/`, `rl-agent/`, `storage/`.
- **`configs/`** — frozen experiment/reward/RL configuration
  (`rl.yaml`/`rl_v2.yaml`, `reward_v2.yaml`, `experiment.yaml`) and
  traffic profiles, including the held-out pre-registered evaluation
  workload (`traffic_profiles/v2_eval/`) and the exploratory stress
  workload (`traffic_profiles/v2_stress/`), each firewalled from the
  training traffic allowlist.
- **`docs/`** — the frozen MDP specification (`rl_mdp_spec.md`), the
  append-only design-correction log (`experimental_design_corrections.md`,
  CORR-1–CORR-65), the evaluation pre-registration
  (`evaluation_v2_preregistration.md`), the viva (V1) results summary,
  publication `figures/` and `tables/` (see Links below), and
  `archive/` (superseded figure drafts, kept for provenance, not part of
  the current figure set).
- **`demo_ui/`** — a Streamlit instrumentation console (demo/fixture
  mode by default; see `demo_ui/README.md`).
- **`collectors/`** — the Prometheus client and metrics-collection
  harness.
- **`persistent-data/`** — all raw experiment evidence: training session
  telemetry (`rl_training_v2/final_min2/sessions/`), the freeze manifest
  and its two amendments, evaluation and stress-test run ledgers
  (`evaluation_v2/eval_final/`, `evaluation_v2/stress_explore/`).
- **`results/`** — curated V1 evaluation evidence (ledgers, figures,
  provenance) kept for historical comparison.
- **`task_outputs/`** — frozen V1 pre-registered evaluation protocol
  documents, kept at this exact path because the final V1 checkpoint's
  metadata hash-pins them by this relative path; also the per-task
  output log for this capstone's working sessions.
- **`tests/`** — the unit test suite (`tests/unit/`).
- **`models/`** — trained PPO checkpoints (V1 and V2).
- **`experiments/`**, **`dashboards/`** — supporting experiment configs
  and Grafana dashboard definitions.
- **`requirements/`** — scientific-venv dependency pins (`base.txt`,
  `rl.txt`, `dev.txt`); `app/requirements.txt` and
  `demo_ui/requirements-ui.txt` are separate, smaller dependency sets for
  the benchmark app and the demo UI respectively.

## Setup

- Python 3.10 (see `.python-version`)
- Minikube (Docker driver recommended on macOS), `kubectl`, `helm`
- Docker Desktop (or equivalent) for the Minikube Docker driver

Scientific venv (training/evaluation/analysis):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt -r requirements/rl.txt
# pip install -r requirements/dev.txt   # ruff/lint, only if developing
cp .env.example .env   # adjust if needed
```

Demo UI venv (separate, smaller dependency set):

```bash
python3 -m venv demo_ui/.venv
source demo_ui/.venv/bin/activate
pip install -r demo_ui/requirements-ui.txt
```

## How to run

**Cluster prepare**

```bash
minikube start --driver=docker
kubectl apply -f k8s/namespaces/
# install kube-prometheus-stack per k8s/monitoring/values.yaml
bash scripts/deploy_app.sh      # builds + deploys the benchmark app
```

`kube-prometheus-stack` (Prometheus + Grafana) must be reachable before
training, evaluation, or stress-test runs — both the RL environment and
every HPA arm read Metrics Server/Prometheus. `dashboards/grafana/` holds
the project's dashboard definitions.

**Training** (live cluster; writes to `persistent-data/rl_training_v2/`):

```bash
.venv/bin/python scripts/train_ppo_final_v2.py --dry-run          # preflight only
.venv/bin/python scripts/train_ppo_final_v2.py --execute-campaign --i-confirm-live-cluster-interaction
```

**Evaluation** (pre-registered `eval_final` campaign, 27 runs):

```bash
.venv/bin/python scripts/evaluate_v2_campaign.py --plan eval --dry-run
.venv/bin/python scripts/evaluate_v2_campaign.py --plan eval --execute --i-confirm-live-cluster-interaction
```

**Exploratory stress test** (`stress_explore` campaign, 9 runs,
descriptive-only, CORR-65 — never replaces the pre-registered evaluation
above):

```bash
.venv/bin/python scripts/evaluate_v2_campaign.py --plan stress --dry-run
.venv/bin/python scripts/evaluate_v2_campaign.py --plan stress --execute --i-confirm-live-cluster-interaction
```

**Analysis / figures / tables** (read-only against committed
`persistent-data/`, no cluster needed):

```bash
.venv/bin/python scripts/analyze_v2_evaluation.py evaluation
.venv/bin/python scripts/analyze_v2_evaluation.py training-curve
.venv/bin/python scripts/generate_v2_phase6e_figures.py   # docs/figures/
.venv/bin/python scripts/generate_v2_paper_tables.py      # docs/tables/
```

**UI demo** (fixture/offline mode by default, no cluster needed):

```bash
cd demo_ui && streamlit run app.py
```

**Tests**:

```bash
.venv/bin/pytest tests/ -q
.venv/bin/ruff check .
```

## Training summary

Final training campaign: **48 analyzed sessions** (256 steps each via
`step_telemetry.jsonl`, 30s decision interval) = **12,288 cumulative
live timesteps**, governed by `persistent-data/rl_training_v2/final_min2/FREEZE_MANIFEST.json`
and amendments `FREEZE_MANIFEST_AMENDMENT_001.json`/`_002.json`. Two
training attempts (sessions 4 and 38) failed and were retried under the
project's tooling-failure/retry discipline; these are visible as dashed
markers in every `docs/figures/training_*.png` figure and are listed
explicitly in `docs/tables/training_summary.md`.

Learning indicators across the 48 sessions (`docs/tables/training_summary.md`):

| indicator | first session | last session |
|---|---|---|
| entropy loss | -1.0984 | -0.2792 |
| explained variance | 0.0125 | 0.7993 |

Entropy loss falls and explained variance rises toward training's end,
consistent with the policy becoming less random and the value function
increasingly fitting returns. See `docs/figures/training_entropy.png`,
`training_explained_variance.png`, `training_value_loss.png`,
`training_approx_kl.png`, `training_mean_step_reward.png`,
`training_reward_components.png`, `training_action_mix.png`,
`training_mean_replicas.png` for the full set.

PPO hyperparameters: `configs/rl_v2.yaml` (`docs/tables/T3_ppo_hyperparameters.md`).
Observation features: `docs/rl_mdp_spec.md` (`docs/tables/T1_observation_features.md`).
Reward components: `configs/reward_v2.yaml` (`docs/tables/T2_reward_components.md`).

## Results summary

**Pre-registered evaluation** (`eval_final`, `docs/evaluation_v2_preregistration.md`,
CORR-60): 27 of 27 planned runs VALID (9 per arm, 3 repeats per profile ×
arm cell across `eval_cpu_step`, `eval_memory_ramp`, `eval_mixed_burst`).
Per the pre-registration's own decision rule (an arm is "better" on a
metric for a profile only if better in **all 3 of 3 repeats**; no
aggregate winner across metrics or profiles):

- **SLA-violation-rate@1000ms**: no decidable winner, any arm vs. any arm,
  in any profile — every arm sits at or near 0 at this workload intensity.
- **replica-minutes (cost)**: no single arm is cheapest in every profile.
  `HPA CPU-only` is cheapest in 2 of 3 profiles and ties on the third;
  `PPO`'s cost ranking is profile-dependent (cheaper than
  `HPA CPU+Memory` on two profiles, more expensive than both HPA arms on
  `eval_cpu_step`).
- No safety termination occurred in any of the 27 valid runs.

Full per-profile tables: `docs/tables/eval_final_primary_metrics.md`,
`eval_final_secondary_metrics.md`; full narrative:
`persistent-data/traffic/v2_phase6a_results_report.md` (sections 1-7).

**Exploratory stress test** (`stress_explore`, CORR-65) — descriptive
only, 1 repeat per profile × arm, reported **separately from and never
merged with** the pre-registered results above. At training-level peak
loads (up to 70 RPS), the primary SLA metric becomes clearly nonzero and
differentiates arms, unlike `eval_final`'s near-floor values:

- `stress_cpu_burst`: PPO has the lowest cost (84.00 vs. 123.00/130.00
  replica-minutes) and lowest mean p95 latency (1567.65 vs.
  1703.08-1820.30 ms), with an SLA-violation rate similar to (not
  meaningfully better than) both HPA arms (~0.50-0.53 for all 3 arms).
- `stress_memory_ramp`: no SLA breaches for any arm; PPO's cost is
  mid-range (56.00, between HPA CPU-only's 40.00 and HPA CPU+memory's
  78.00).
- `stress_mixed_burst`: PPO has the *highest* SLA-violation rate of the 3
  arms (0.20 vs. 0.00/0.10) at mid-range cost — the opposite ordering
  from `stress_cpu_burst`.
- **No single arm is uniformly better across all 3 stress profiles.**

Full narrative, including the per-profile replica-ceiling comparison and
the 975ms histogram-bucket-artefact caveat: `persistent-data/traffic/v2_phase6a_results_report.md`
(section 8). Full descriptive table: `docs/tables/stress_explore_descriptive_metrics.md`.
Figures: `docs/figures/stress_*.png` (every caption/title says
"exploratory").

## Limitations

- **Light evaluation loads / ceiling effect**: `eval_final`'s 3 held-out
  profiles peak at 18-20 RPS, well below training-level peaks (up to 70
  RPS). The primary SLA metric sits at or near its floor (~0) for every
  arm at this intensity, so it does not discriminate between controllers
  under `eval_final` — the exploratory stress test (above) was added
  post-hoc specifically to probe heavier loads, but remains descriptive
  (1 repeat) and does not substitute for a pre-registered, 3-repeat
  evaluation at higher load.
- **1-repeat stress test**: every `stress_explore` number is a single
  observation, not a mean over repeats, and cannot support the
  pre-registered better-in-all-3-repeats decision rule.
- **Histogram-bucket artefact**: a recurring p95 latency value of exactly
  975ms (seen in several `eval_memory_ramp`/`eval_mixed_burst`/stress
  cells) is a Prometheus histogram-bucket interpolation artefact, not a
  precisely measured value — the true p95 lies somewhere in the 0.5-1.0s
  bucket range.
- **Single-node local cluster**: all training, evaluation, and stress
  runs executed on a single-node Minikube cluster (Docker driver), not a
  multi-node production-scale cluster.
- **Two failed-and-retried training sessions**: sessions 4 and 38 of the
  48-session training campaign failed and were retried under the
  project's tooling-failure/retry discipline (marked explicitly in every
  `docs/figures/training_*.png` figure and in `docs/tables/training_summary.md`);
  the 48 analyzed sessions are the successful retries, not a silently
  dropped or averaged-over failure.
- `reaction_time_s` is reported as `n/a` throughout the evaluation summary
  — the analysis script found no usable signal for it in this normalized
  series schema; it is not fabricated or imputed.

## Reproducibility

- **Freeze manifest**: `persistent-data/rl_training_v2/final_min2/FREEZE_MANIFEST.json`
  hash-pins every file governing the final V2 training/evaluation
  contract at the point training began, amended twice
  (`FREEZE_MANIFEST_AMENDMENT_001.json`, `_002.json`) for narrowly scoped,
  pre-registered changes only.
- **Pre-registration**: `docs/evaluation_v2_preregistration.md` /
  `persistent-data/rl_training_v2/final_min2/EVALUATION_PREREGISTRATION.json`
  (CORR-60) fixes the arms, profiles, metrics, and decision rule *before*
  any evaluation run.
- **Correction log**: `docs/experimental_design_corrections.md` is an
  append-only record of every design decision and mid-course correction
  (CORR-1 through CORR-65), each with Problem/Decision/Scope/Scientific
  Impact/Preserved Invariants/Non-Retroactivity sections — nothing in
  this project is corrected silently.
- Every run (training, evaluation, stress) is recorded in an append-only
  JSONL ledger under `persistent-data/`; failed/retried attempts are
  preserved, never deleted or edited in place.

## Figures, diagrams, and tables

- `docs/figures/FIGURES.md` — full index of every published figure and
  diagram (file, caption, data source, key takeaway, suggested paper
  section): 42 one-plot-per-file figures (workloads, training, evaluation,
  stress, discussion) and 4 Graphviz diagrams (system architecture, RL
  control loop, experiment pipeline, safety-aware tie-break flow).
- `docs/tables/` — paper-ready tables in CSV, Markdown, and LaTeX:
  observation features (T1), reward components (T2), PPO hyperparameters
  (T3), experimental design (T4a-c), evaluation primary/secondary
  metrics, stress-test descriptive metrics, and the training summary.
