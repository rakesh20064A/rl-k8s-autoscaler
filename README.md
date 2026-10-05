# RL-Based Adaptive Auto-Scaling for Kubernetes: A Comparative Study Against HPA

BTech capstone project evaluating whether a reinforcement-learning (PPO)
controller can match or improve on Kubernetes' built-in Horizontal Pod
Autoscaler (HPA) for scaling a live, containerized workload.

## Objective

Train a PPO agent, live against a real Minikube cluster, to make
replica-scaling decisions for a benchmark application — then compare it
against the cluster's canonical HPA on an identical, held-out traffic
workload, under a pre-registered evaluation protocol.

## Architecture

```
Traffic Generator (loadgen/) --> demo-app (app/) --> Prometheus/Grafana (k8s/monitoring/)
                                       ^                        |
                                       |                        v
                         HPA (k8s/hpa/)  <--- OR --->  PPO Agent (autoscaler/)
                                       |
                                 Kubernetes Deployment (k8s/app/)
```

- **`app/`** — the benchmark FastAPI application (`/work` CPU-bound endpoint,
  `/health`, Prometheus `/metrics`).
- **`loadgen/`** — deterministic, seeded traffic generator producing
  reproducible request schedules.
- **`autoscaler/src/rl_env/`** — the Gymnasium-compatible RL environment
  (observation/action/reward, live Kubernetes backend).
- **`scripts/`** — live training runner (`train_ppo_live_session.py`),
  traffic orchestration, final evaluation harness
  (`evaluate_ppo_vs_hpa.py`), and result analysis/report generation.
- **`k8s/`** — Kubernetes manifests (app Deployment/Service, canonical HPA,
  kube-prometheus-stack monitoring values).
- **`demo_ui/`** — a Streamlit instrumentation console (demo/fixture mode;
  see `demo_ui/README.md`).
- **`collectors/`** — the Prometheus client and metrics-collection harness.
- **`configs/`** — frozen experiment/reward/RL configuration and traffic
  profiles (including the final held-out evaluation workload).
- **`docs/`** — the frozen MDP specification and design-correction record.
- **`task_outputs/`** — the frozen evaluation protocol documents and the
  final Index39/40 results (metrics, tables, figures, report drafts).
- **`persistent-data/`** — the final trained checkpoint (`session_019`) and
  the final evaluation evidence/ledgers only (see Limitations).

## PPO vs. HPA, in brief

- **HPA**: reactive, threshold-based (scales on CPU utilization relative to
  a 260% target — the project's long-standing canonical configuration).
- **PPO**: learns a policy over an 11-dimensional observation (CPU,
  request rate, latency, replica state, short-horizon history) and a
  3-way discrete action (scale down / hold / scale up), trained live
  against the real cluster with a reward balancing SLA performance against
  replica cost.

## Setup

- Python 3.10 (see `.python-version`)
- Minikube (Docker driver recommended on macOS)
- `kubectl`, `helm` (for kube-prometheus-stack)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt -r requirements/rl.txt
cp .env.example .env   # adjust if needed
```

## Deployment

```bash
minikube start --driver=docker
kubectl apply -f k8s/namespaces/
# install kube-prometheus-stack per k8s/monitoring/values.yaml
bash scripts/deploy_app.sh      # builds + deploys the benchmark app
kubectl apply -f k8s/hpa/hpa.yaml
```

Prometheus/Grafana (`k8s/monitoring/`) provide the metrics the RL
environment and the HPA both depend on; `dashboards/grafana/` holds the
project's Grafana dashboard definitions.

## Training and evaluation overview

Live PPO training is driven by `scripts/train_ppo_live_session.py`
(checkpoint-per-rollout, 256 steps/rollout, 30s decision interval).
Final evaluation replays a single frozen, hash-pinned, never-previously-
inspected held-out workload (`configs/traffic_profiles/iteration_rl_final_holdout_v1.yaml`)
against both controllers via `scripts/evaluate_ppo_vs_hpa.py`, following
the pre-registered protocol in `task_outputs/phase_16u_direct_ppo_protocol.yaml`.
Analysis and report generation are in `scripts/analyze_final_evaluation.py`
and `scripts/generate_index40_report_package.py`.

## Final experiment status

Training was frozen at checkpoint `session_019` (5120 cumulative live
timesteps, 20 PPO updates) — short of the original 12288-step design
budget, due to live-cluster wall-clock constraints within the capstone
timeline (see `task_outputs/index40_viva_results_summary.md`).

Final evaluation executed the pre-registered sequence
PPO1 → HPA1 → PPO2 → HPA2 → PPO3 → HPA3 on the identical held-out
workload:

| | Valid full-duration repeats |
|---|---|
| **HPA** | 3 / 3 |
| **PPO** | 1 / 3 (PPO1 and PPO2 terminated under the frozen safety rule: 4 consecutive decision steps with an unreachable telemetry/application target) |

Because the pre-registered decision rule requires **3** valid
full-duration repeats per controller, **the formal n=3 comparison rule is
not applicable, and no formal winner is declared.**

Descriptively: HPA completed all three repeats with lower P95 latency; the
one full-duration PPO repeat used substantially fewer replicas but
recorded higher latency and a higher SLA-violation rate. Full detail,
figures, and the exact Section-10 metrics are in `task_outputs/index39_*`
and `task_outputs/index40_*`.

**This project does not claim PPO outperformed HPA, nor the reverse.**

## Limitations

- PPO's valid sample size (n=1) cannot support the pre-registered decision
  rule or any statistical claim.
- Training used a single live seed and a reduced budget relative to the
  original design target.
- This repository includes only the final checkpoint (`session_019`) and
  curated final-evaluation evidence — the full historical training/
  calibration provenance (dozens of intermediate checkpoints, raw traffic
  logs, phase-by-phase audit reports) exists in the project's full working
  history but is intentionally excluded here to keep the public repository
  concise. See `task_outputs/index40_results_chapter_draft.md` for the
  full write-up.

## Running the important components

```bash
# Unit tests (no live cluster required for most)
pytest tests/ -q

# Demo UI (fixture/offline mode by default)
pip install -r demo_ui/requirements-ui.txt   # separate from the scientific venv
cd demo_ui && streamlit run app.py

# Offline analysis / report regeneration (reads only committed evidence)
python scripts/analyze_final_evaluation.py
python scripts/generate_index40_report_package.py
```

Live training/evaluation (`--execute` paths) require a running Minikube
cluster and are not run as part of this README's quick-start.
