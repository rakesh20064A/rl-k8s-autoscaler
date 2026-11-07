# Part 05 — Repository and Git

All VERIFIED directly this session (via a dedicated research agent, commands
re-stated here) unless marked FROM_CODE.

## Basic facts

- HEAD: `1d74164263969fcaf5ddf93fb026fa6d783034b0`, branch `main`.
- Remote: `origin` → `https://github.com/rakesh20064A/rl-k8s-autoscaler.git`
  (fetch+push) — note the remote name has no `-github-clean` suffix even
  though the local directory does; this is the GitHub-published clean copy.
- Full commit log (only 2 commits exist in this repo's history):
  ```
  1d74164 2026-10-05 Finalize project structure and validated results
  491bb94 2026-10-05 Initial release: RL-based Kubernetes autoscaling capstone
  ```
- `git stash list`: empty.
- **Is any V2 work committed?** NO — every V2-related file (all of
  `autoscaler/src/rl_env/*_v2.py`, `action_selection.py`,
  `safe_inference.py`, `safe_ppo_controller.py`, `decision_record_v2.py`,
  `controller_exclusivity.py`, `training_contract_v2.py`,
  `training_schedule_v2.py`, every V2 script, every V2 test,
  `configs/rl_v2.yaml`, `configs/reward_v2.yaml`,
  `k8s/hpa/hpa_v2_cpu_memory.yaml`, all of `configs/traffic_profiles/v2/`,
  and all V2 training/pilot data under `persistent-data/`) is **untracked**
  — it exists only on disk, never committed. The entire multi-week V2
  effort (Phases 1 through 4B-R4F) is uncommitted work.

## `git status --porcelain` — full (16 modified, ~68 untracked)

16 modified files (all V1-area or shared files, each touched additively to
support V2 without altering V1 semantics): `.gitignore`, `app/src/main.py`,
`app/src/workload.py`, `autoscaler/src/rl_env/environment.py`,
`autoscaler/src/rl_env/live_backend.py`,
`autoscaler/src/rl_env/training_wrapper.py`,
`demo_ui/components/charts.py`, `demo_ui/components/controls.py`,
`demo_ui/tests/test_app_smoke.py`, `demo_ui/theme.py`,
`demo_ui/views/{comparison,experiment,history,live_details}.py`,
`k8s/app/deployment.yaml`, `loadgen/traffic_generator.py`,
`tests/unit/test_rl_env_live_backend.py`.

Per-file one-sentence description of each modification (from diff content):

| File | Change |
|---|---|
| `.gitignore` | Adds `persistent-data/controller/` and `demo_ui/.run_store/` to ignore list. |
| `app/src/main.py` | Extends `/work` with `resource_mode`/`memory_mb`/`memory_hold_ms` params, dispatching to new `perform_work()`; V1-compatible by default. |
| `app/src/workload.py` | Adds bounded/released memory-pressure workload machinery (`perform_memory_work`, mmap-based, explicit release) alongside the untouched original `perform_cpu_work`. |
| `autoscaler/src/rl_env/environment.py` | Adds an `is_reversal` flag to the step-info dict (purely additive, for V2's reward wrapper to read). |
| `autoscaler/src/rl_env/live_backend.py` | Adds `list_hpa_names()` to detect coexistence of `demo-app-hpa`/`demo-app-hpa-v2`. |
| `autoscaler/src/rl_env/training_wrapper.py` | Adds `STATUS_PILOT_SAFETY_INTERRUPTED` + `log_pilot_safety_interruption()` so an external SB3 safety-callback stop still gets a ledger row. |
| `demo_ui/components/charts.py` | Adds empty-state chart rendering; switches to `lines+markers` so single-point series render. |
| `demo_ui/components/controls.py` | Decouples widget state from session-state keys (fixes a reset-on-navigation bug); adds "active-run authority". |
| `demo_ui/tests/test_app_smoke.py` | Updates a widget key reference to match the controls.py rename. |
| `demo_ui/theme.py` | Broadens a CSS color-forcing rule for radio-option text visibility. |
| `demo_ui/views/comparison.py` | Adds a real "Live Comparison" section from `experiment_store`, keeping the old fixture comparison as clearly-labeled DEMO content. |
| `demo_ui/views/experiment.py` | Wires LIVE+Manual START/STOP to a real subprocess via `live_execution_adapter.py`. |
| `demo_ui/views/history.py` | Adds a "Live Experiment Runs" table from `experiment_store`. |
| `demo_ui/views/live_details.py` | Makes the real `experiment_store` run the primary content; fixture content moved to a collapsed expander. |
| `k8s/app/deployment.yaml` | Adds a documentation-only comment recording the Phase 4B-R3 TCP-liveness experiment and its revert — **no functional spec change**. |
| `loadgen/traffic_generator.py` | Adds optional V2 profile fields (`resource_mode`, `memory_mb`, `memory_hold_ms`, `memory_safe_max_in_flight`), fully backward compatible. |
| `tests/unit/test_rl_env_live_backend.py` | Adds coverage for the new `list_hpa_names()` method. |

Untracked files grouped by area: V2 autoscaler core (10 files under
`autoscaler/src/rl_env/`), V2 configs (4 files/dirs), V2 scripts (7 files),
V2 tests (27 files), demo_ui additions (`experiment_store.py`,
`live_execution_adapter.py`, 8 new test files), and the entire
`persistent-data/` directory (shows as one untracked entry even though most
of its subpaths are individually `.gitignore`d — `persistent-data/
rl_training_v2/` specifically is NOT covered by any ignore rule, which is
why it surfaces).

## `.gitignore` (VERIFIED, full read)

Excludes Python/Node/IDE artifacts, `.venv/`/`venv/`/`env/`/`ml-env/`,
secrets (`.env*` except `.env.example`, `*.pem/*.key/*.crt`,
`credentials*`, `secrets*`), and a long list of generated bulk experiment
directories under `persistent-data/` (`logs/`, `backups/`, `traffic/`,
`traffic_archive/`, `metrics/`, several `dynamics_calibration*`,
`transport_validation/`, `hpa/`, `hpa_mapping/`, `models/`, and — newly
added — `controller/`), plus `demo_ui/.run_store/`. **Not ignored**:
`persistent-data/rl_training_v2/` (all pilot/final training data) — this is
intentional and is why that tree is visible/trackable as untracked content.

## Directory tree (depth 3, summarized)

Top-level: `.env.example`, `.gitignore`, `.python-version` (`3.10.14`),
`Makefile`, `README.md`, `app/`, `autoscaler/`, `collectors/`, `configs/`,
`dashboards/grafana/`, `demo_ui/`, `docs/`, `experiments/manifests/`,
`k8s/`, `loadgen/`, `models/`, `persistent-data/`, `pytest.ini`,
`requirements/`, `results/`, `scripts/`, `task_outputs/`, `tests/`.

## Prior audit reports under `persistent-data/traffic/` and `task_outputs/`

`persistent-data/traffic/*.md` (top level): `v2_phase4br4d_route_diagnostic_preflight.md`
(20,366B, Nov 3 13:15), `v2_phase4br4e_baseline_readiness_audit.md`
(18,491B, Nov 3 13:24), `v2_phase4br4f_multimetric_controller_optimization_audit.md`
(45,672B, Nov 3 13:45). Plus
`v2_phase4br4c_true_service_fixed_two_validation/postrun_forensic_audit.md`
(10,927B, Nov 3 12:13) inside its own evidence subdirectory.

`task_outputs/*` (all dated Oct 5, the frozen Phase 16U artifacts):
`phase_16u_completion_report_and_verdict.md`, `phase_16u_direct_ppo_protocol.yaml`,
`phase_16u_manifest.json`, `phase_16u_ppo_hyperparameters.json`,
`phase_16u_reward_specification.yaml`, `phase_16u_safety_and_fallback_protocol.yaml`,
`phase_16u_training_and_evaluation_workload_split.json`.

### Detailed summaries of the three 2026-11-03 reports (R4C/R4D/R4E)

**R4C post-run forensic audit**: a fixed-two-replica, no-HPA, no-PPO
validation test (launched 2026-10-12, audited 21 days later on 2026-11-03)
used a `minikube service --url` SSH tunnel. The generator itself completed
normally (600.02s actual vs. 600s requested, full 12,021-row schedule).
But a dramatic, permanent "connectivity cliff" appears at t=328s: clean
~32% success traffic through t=327s, then `success_2xx` drops to exactly
zero and `connection_error` dominates every second for the remaining 272s
(45% of the run) with zero recovery. The audit could NOT determine whether
this reflects a tunnel-infrastructure failure or a genuine simultaneous
dual-pod outage — no Kubernetes events/pod history/Prometheus data survived
21 days, and the request-level CSV schema has no per-pod identity field at
all. **Verdict: `FIXED2_VALIDATION_INCONCLUSIVE`** (execution completed
normally; scientific validation inconclusive, not a pass).

**R4D route-diagnostic preflight**: started/verified the existing Minikube
profile (no new cluster), patched Metrics Server to 15s resolution
(verified live, 3 advancing timestamp samples), and investigated the R4C
ambiguity without re-running anything live. Found the live cluster
non-canonical (2 replicas, no HPA, both pods showing extra restarts
partly attributable to the 21-day Minikube-stop/resume cycle). Re-hashed
and confirmed all 4 R4C evidence files byte-identical (verdict preserved).
Documented that `minikube service --url` uses an SSH tunnel to the
Service's ClusterIP — a different mechanism from the project's own
validated V5 `socat`-relay-to-NodePort approach, which has never shown this
failure mode in any real pilot/campaign session. Proposed (did not
implement) a future isolated diagnostic using the V5 relay against a
disposable test Deployment with continuous durable state capture.
**Verdict: `ROUTE_DIAGNOSTIC_PREFLIGHT_INCONCLUSIVE`** (Metrics Server work
done/verified; overall blocked by non-canonical cluster state).

**R4E baseline readiness audit**: definitively established that despite its
"v2" filename, `k8s/hpa/hpa_v2_cpu_memory.yaml` (`demo-app-hpa-v2`) is NOT
what live training automation actually restores — `LiveKubernetesBackend`
defaults to and `train_ppo_final_v2.py`'s preflight explicitly checks for
`demo-app-hpa` (`k8s/hpa/hpa.yaml`). Confirmed all 3 governing manifest
hashes unchanged. Using `kubectl get events`, found direct, conclusive
evidence that the pods' most recent restart was a kubelet `SandboxChanged`
event tied to the Minikube stop/start cycle, not the application (though
cumulative restart counts 5/8 still include real restarts from the
original R4C test). Confirmed via `kubectl exec` (no tunnel needed) that
Prometheus is genuinely, actively scraping both app pods, and a live
per-pod CPU query returns distinguishable values per pod. Documented the
exact V5 route end to end. **Verdict: `BASELINE_READINESS_BLOCKED`** — every
diagnostic question answered with direct evidence, but live state does not
match canonical and must block further live diagnostics until an
explicitly authorized restoration.

**Status as of THIS snapshot (2026-11-07)**: none of the restorations R4D/
R4E recommended have been applied — see 04_live_cluster.md, the drift is
unchanged (still 2 replicas, still no HPA) and the Metrics Server 15s patch
has additionally been lost (reverted to 60s) by a subsequent Minikube
restart.
