# Part 08 — Configuration Values and k8s Manifests

All FROM_CODE. Manifest hashes VERIFIED directly this session (match every
previously-recorded value exactly — no drift on any governing manifest).

## configs/experiment.yaml ("single source of truth")

`kubernetes`: app_namespace=autoscaler-app, monitoring_namespace=monitoring,
deployment_name=demo-app, service_name=demo-app,
configmap_name=demo-app-config. `scaling`: min_replicas=1, max_replicas=8,
initial_replicas=1, cooldown_seconds=60. `prometheus`:
url="http://localhost:9090", scrape_interval_seconds=10,
staleness_threshold_seconds=30. `collection`: sampling_interval_seconds=10,
output_dir=persistent-data. `sla`: p95_latency_ms=300, error_rate=0.01.
`rl`: control_interval_seconds=30, mode=shadow,
model_path=persistent-data/models/ppo_current/model.zip,
time_since_scale_cap_seconds=300. `traffic`: default_seed=42,
workload_endpoint=/work, default_intensity=3. `paths`: persistent_data,
datasets, experiments, dashboards, scripts.

Read by: `autoscaler/src/rl_env/config.py::load_env_config_from_yaml()`
(`scaling.*` → `config.py:85-87,139`; `scaling.cooldown_seconds` →
`GatingConfig.scale_down_cooldown_seconds`, `config.py:115`;
`rl.control_interval_seconds` → `EnvConfig.decision_interval_s`,
`config.py:137`). Also `--config` default for
`loadgen/traffic_generator.py`/`collectors/metrics_collector.py`.

## configs/app.yaml / configs/hpa.yaml

Manually-maintained "design source" documents (`app.yaml`: image/resources/
probes/termination-grace/env, matching `k8s/app/*.yaml` byte-for-byte;
`hpa.yaml`: target_cpu_utilization_percent=260, min=1/max=8, scale-up/down
stabilization windows — extensively documenting the CORR-24/CORR-26 HPA-
tuning history). **No Python code was found directly loading either file
by path** — their values are manually mirrored into `k8s/` manifests; this
is a documented discrepancy (Part 15), not a bug per se.

## configs/rl.yaml — PPO + MDP contract v2 (FROZEN, V1)

`rl_contract_version: 2`. `action_space`: n_actions=3, action_map=
{0:-1, 1:0, 2:+1}. `ppo`: learning_rate=3e-4, n_steps=2048, batch_size=64,
n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
vf_coef=0.5, max_grad_norm=0.5, policy=MlpPolicy, net_arch=[128,128].
`training`: total_timesteps=500000, eval_freq=10000, n_eval_episodes=5,
seeds=[42,7,123]. `observation_v2`: 11-entry ordered list (see
09_rl_contract.md for the full per-feature table). `history.
ewma_span_steps=3`. Read by `config.py::load_env_config_from_yaml()`
(`config.py:50-143`, validates `rl_contract_version==2`).

## configs/rl_v2.yaml (V2)

Byte-identical to rl.yaml for shared keys (`rl_contract_version=2`
deliberately unchanged, not a contract bump), **plus** additive-only V2
blocks: `safe_action.near_tie_epsilon: 0.05` (explicitly marked
`PROVISIONAL default -- NOT a validated scientific value`),
`safe_action.epsilon_status: provisional_unvalidated`;
`controller_generation: v2`, `workload_contract: multi_resource_v2`,
`inference_policy: safe_near_tie_hold`,
`reward_status: v2_designed_pending_training_validation`. Read by
`config_v2.py::load_env_config_v2_from_yaml()` (validates
`EXPECTED_CONTROLLER_GENERATION`/`EXPECTED_WORKLOAD_CONTRACT`, raises on
mismatch). `safe_action.*` is explicitly NOT read by the V1 loader.

## configs/reward.yaml (V1, FROZEN) / configs/reward_v2.yaml (V2)

V1 weights: w_sla=1.0, w_reliability=5.0, w_resource=1.0, w_action=0.1,
w_reversal=0.5, w_boundary=0.2, w_efficiency=0.3; sla.p95_target_ms=300.

V2 weights (redesigned formula, same `reward_contract_version: 2` for
loader compat): w_reliability=6.0, w_sla=4.0, w_pressure=2.0, w_waste=0.3,
w_action=0.1, w_reversal=0.5, w_boundary=0.2; sla.p95_target_ms=300;
pressure: cpu_safe_high=3.0/cpu_hard_limit=5.0,
memory_safe_high=1.2/memory_hard_limit=4.0; waste:
cpu_waste_threshold=0.3/memory_waste_threshold=0.7. See 09_rl_contract.md
for the full formula and the two confirmed dead-config-key defects
(`min_replicas`/`max_replicas` and the per-cost multipliers are never
actually read by the loader).

## configs/traffic_profiles/ — V1 root profiles

| File | duration_s | shape | intensity | seed |
|---|---|---|---|---|
| constant.yaml | 600 | flat 20 rps | 3 | 42 |
| step.yaml | 900 | 5→50→5→30 rps steps | 3 | 42 |
| burst.yaml | 1200 | baseline=5, burst=70×7, dur=30s, inter=120s | 4 | 42 |
| ramp.yaml | 1200 | 3→60 over 400s, hold 200s, →3 over 400s | 3 | 42 |
| periodic.yaml | 1800 | sinusoid mean=20, amp=15, period=300s | 3 | 42 |
| mixed.yaml | 2400 | 8-phase concat (primary comparison scenario) | 3 | 42 |
| smoke_test.yaml / smoke_test_15s.yaml / demo_smoke_test.yaml | 25/15/40 | flat, tiny | 3 | 42 | test/demo only |
| unseen_eval.yaml | 2100 | 8-phase | 4 | 99 | excluded from formal comparison |
| iteration_rl_final_holdout_v1.yaml | 1500 | 6-phase | 3 | 777 | frozen final PPO-vs-HPA holdout |

V1 allowlists (`scripts/rl_live_common.py:34-36`):
`TRAINING_PROFILES = (constant, step, burst, ramp, periodic, mixed)`;
`FINAL_HOLDOUT_PROFILE = iteration_rl_final_holdout_v1`;
`EXCLUDED_FROM_FORMAL_COMPARISON = (unseen_eval,)`. Enforced by
`assert_training_eligible()`/`assert_final_evaluation_eligible()`.

## configs/traffic_profiles/v2/ — V2 profiles

| File | duration_s | shape | intensity | resource_mode | memory_mb | memory_safe_max_in_flight |
|---|---|---|---|---|---|---|
| constant_cpu.yaml | 600 | flat 20 rps | 3 | cpu | 0 | — |
| constant_memory.yaml | 600 | flat 20 rps | 3 | memory | 16 | 8 |
| constant_mixed.yaml | 600 | flat 20 rps | 3 | mixed | 16 | 8 |
| ramp_mixed.yaml | 1200 | same shape as ramp.yaml | 3 | mixed | 16 | 8 |
| burst_mixed.yaml | 1200 | same shape as burst.yaml | 4 | mixed | 16 | 8 |
| memory_cap_smoke_test.yaml | 60 | flat 30 rps | 1 | memory | 16 | 8 |

V2 allowlist (`autoscaler/src/rl_env/training_contract_v2.py:52-58`):
`V2_TRAINING_PROFILE_ALLOWLIST = (constant_cpu, constant_memory,
constant_mixed, ramp_mixed, burst_mixed)` — exactly the 5 non-smoke-test
profiles; `memory_cap_smoke_test.yaml` explicitly excluded.

## k8s/ manifests — hashes (VERIFIED, all unchanged from prior phases)

| File | SHA-256 |
|---|---|
| k8s/app/deployment.yaml | `bca1e8af2febebaf3272399a63b1ad4a2aeb3440994009528d14f57de4fd92d8` |
| k8s/app/service.yaml | `3e20a2fd30ce20b905ab6b6a6023a652c05683e2e9b3de496df3f2ae22d190c7` |
| k8s/app/service-loadtest.yaml | `1bab41f9d451d939ac8c00415cf874a202c00593adcf9820e00ba9586ff446fa` |
| k8s/app/service-monitor.yaml | `27d9e081dbce90a09bb329746b84b16f062907866467d13c4d7608b527fb30b5` |
| k8s/app/configmap.yaml | `bf871811b38677b1bf7a90312c2ae3043d83c31f45f9809d3b9404cdc44a17db` |
| k8s/hpa/hpa.yaml | `809550767890ef3dc50bd9d93f3ca2d1cad2b0dc26881dca0a75a31fa0c5d7a1` |
| k8s/hpa/hpa_v2_cpu_memory.yaml | `a64c6614dfd64217df691cbd707063637bf100c09083ad9b0854b65f292b649a` |
| k8s/monitoring/grafana-dashboard.yaml | `a3133827be1fe0ea4d720102b39aab119be60761fae4434ff6848bff992fcf04` |
| k8s/monitoring/values.yaml | `5233398b3c8518494c41283424ae2fb2094db74880cf16ef24f051c954f1c033` |
| k8s/namespaces/app-namespace.yaml | `cbae93e6c53db0b56ac5e619bc898a9242ac211d83cc7e4bdca7df4169d86df0` |

**k8s/app/deployment.yaml**: replicas=1 (canonical), terminationGrace=30s,
prometheus scrape annotations; image `rl-autoscaler-demo-app:local`,
`Never` pull; resources 100m/64Mi req, 500m/256Mi limit; readiness/liveness
HTTP `/health` as documented above (TCP-liveness experiment documented as
rejected-and-reverted in a comment); podAntiAffinity preferred weight=50.

**k8s/app/service.yaml**: ClusterIP, selector `app=demo-app`, port 80→8080,
label `purpose: primary-metrics-source` (the only Service carrying this
label — CORR-22 fix against duplicate scraping).

**k8s/app/service-loadtest.yaml**: NodePort, same selector,
`sessionAffinity: None`, label `purpose: experiment-load-generator-access`
(deliberately distinct label) — exists because `kubectl port-forward` to a
Service pins to one backend pod (CORR-21).

**k8s/app/service-monitor.yaml**: `selector.matchLabels: {app: demo-app,
purpose: primary-metrics-source}` (AND of both, CORR-22), interval=10s.

**k8s/hpa/hpa.yaml**: `demo-app-hpa`, min=1/max=8, CPU averageUtilization=
260 (frozen, CORR-26), scaleUp window=15s/+2 pods, scaleDown
stabilization=120s/-1 pod/60s.

**k8s/hpa/hpa_v2_cpu_memory.yaml**: `demo-app-hpa-v2` (separate object
name), same scaleTargetRef/min/max, TWO metrics (CPU 260% unchanged +
memory 120% live-calibrated), same behavior block.

**k8s/monitoring/values.yaml** (Helm values, kube-prometheus-stack 82.9.0):
`prometheus.prometheusSpec`: scrapeInterval=10s, evaluationInterval=10s,
**retention=6h, retentionSize=2GB** (explains why R4C's 21-day-old
Prometheus data could never have survived — far short of 21 days
regardless of the cluster restart), requests cpu=100m/mem=256Mi, limits
cpu=500m/mem=512Mi. `grafana`: persistence.enabled=false. `alertmanager`/
`nodeExporter`: disabled.
