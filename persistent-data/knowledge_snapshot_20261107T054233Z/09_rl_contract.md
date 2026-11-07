# Part 09 — RL Contract (V1 and V2)

All FROM_CODE, drawn from this conversation's own direct, line-cited audit
work in Phase 4B-R4F (2026-11-03) plus the Part 08 config inventory, and
the real empirical computation performed in that same phase. Not
re-verified live this session (environment/reward/action-selection logic
is pure Python, independent of cluster state, and none of these source
files showed as modified in Part 05's git diff since R4F).

## Observation contract — 11-element vector (`autoscaler/src/rl_env/observation.py`)

Pipeline: `app/src/metrics.py` → Prometheus (10s scrape) → instant PromQL
query (issued once per 30s PPO decision step) →
`collectors/metrics_collector.py::PrometheusClient.query()` (NaN/Inf/absent
→ `(None, None)`, never fabricated) → `live_backend.py::_snapshot_metrics()`
→ `observation.py::determine_validity()` + `per_replica_ratios()` +
`build_observation()` → 11-float vector → PPO's `MlpPolicy`.

| idx | feature | source PromQL (window) | agg. | norm | dtype/range |
|---|---|---|---|---|---|
| 0 | cpu_per_replica | `sum(rate(container_cpu_usage_seconds_total{...}[1m]))÷ready÷0.1` | fleet AVERAGE, not true per-pod | `[0,5]→[0,1]` | float |
| 1 | cpu_per_replica_delta | EWMA(span=3) of #0 | derived | `[-5,5]→[-1,1]` | float |
| 2 | memory_per_replica | `sum(container_memory_working_set_bytes{...})÷ready÷64MiB` (instant) | fleet AVERAGE | `[0,4]→[0,1]` | float |
| 3 | in_progress_per_replica | `sum(demo_app_requests_in_progress{...})÷ready` (instant) | fleet AVERAGE | `[0,20]→[0,1]` | float |
| 4 | request_rate_norm | `sum(rate(demo_app_requests_total{...}[1m]))` | cluster | `[0,100]→[0,1]` | float |
| 5 | request_rate_delta | EWMA of #4 | derived | `[-100,100]→[-1,1]` | float |
| 6 | p95_latency_norm | `histogram_quantile(0.95, sum(rate(..._bucket[1m])) by (le))` | cluster | `[0,2000ms]→[0,1]` | float |
| 7 | p95_latency_delta | EWMA of #6 | derived | `[-2000,2000]→[-1,1]` | float |
| 8 | error_rate | `sum(rate(...,status_code=~"5.."[1m]))÷request_rate`, safe-zero if rate=0 | cluster ratio | already [0,1] | float |
| 9 | ready_replica_norm | Deployment object directly (CORR-25, not Prometheus) | n/a | `[min,max]→[0,1]` | float |
| 10 | seconds_since_last_scale_norm | controller-internal timer | n/a | `[0,300s]→[0,1]` | float |

Shape/dtype: `(11,)` float32 Gym Box space. Training and inference share
exactly ONE `build_observation()` implementation (no parallel/duplicated
normalization path). **Memory is independently represented from CPU**
(separate PromQL metric, separate index, separate reward-pressure
sub-term) — confirmed, never inferred from CPU. **Error rate is safe at
zero requests** — explicit `request_rate ? error_rate/request_rate : 0.0`
guard. **Missing/NaN/Inf cannot silently become valid** —
`collectors/metrics_collector.py` converts NaN/+Inf/-Inf PromQL results to
`(None, None)` BEFORE they reach the observation builder, which then
correctly fails `determine_validity()`'s `OVERRIDE_STALE_OR_MISSING_METRIC`
check.

**Known design limitation (not a bug)**: features 0/2/3 are fleet-wide
AVERAGES (`sum(...)÷ready_replicas`), architecturally different from the
safety layer's own true-per-pod-max metric
(`get_max_per_pod_memory_mib()`, `max(sum by (pod) (...))`). PPO's
observation cannot see "one hot pod among several idle ones" — only the
fleet average.

**Window-overlap note**: cpu/request-rate/error-rate/latency all use 1-
minute `rate()` windows against a 30-second control interval — consecutive
observations overlap 50%, a deliberate smoothing choice never explicitly
ablated/justified in the repo as such.

## Actions

`ACTION_MAP = {0: -1, 1: 0, 2: +1}` (`autoscaler/src/rl_env/types.py:17`),
`ACTION_NAMES = {0: "SCALE_DOWN_ONE", 1: "HOLD", 2: "SCALE_UP_ONE"}`.
Replica bounds: min=1, max=8 (both V1 and V2). Step size: always exactly
±1 replica per decision (never larger).

## Gating/safety (`autoscaler/src/rl_env/gating.py`, shared V1/V2)

`compute_effective_action()` precedence: (1) structural/safety invalidity
→ HOLD before gating is even invoked; (2) replica-bound clamp
(`OVERRIDE_REPLICA_BOUND_CLAMP`) if the resulting count would exit [1,8];
(3) pending-scale gate (`OVERRIDE_PENDING_SCALE_GATE`) if
`current_desired != current_ready`; (4) scale-up has NO additional cooldown
beyond pending-scale gating; (5) scale-down cooldown
(`OVERRIDE_SCALE_DOWN_COOLDOWN`, 60s, only for `requested_delta<0`).

`PilotSafetyCallback` (`scripts/v2_pilot_safety.py`): per-pod memory
threshold **230MiB**, sustained-sample requirement **2 consecutive
breaches** (via `max(sum by (pod)(container_memory_working_set_bytes))`,
never aggregate). Zero-ready-replica: **immediate** trip (not sustained-
gated), but only if the pod list itself is non-empty (an empty list from a
kubectl query failure does NOT itself trip this specific check — a
fail-open edge case flagged, not a confirmed bug, in Phase 4B-R4F).
Pod-restart detection: compares each currently-live pod's restartCount
against a baseline captured once at session start, keyed by pod name — a
brand-new pod (not in baseline) gets its own current count as baseline, so
it can never false-positive merely for being new.

Drain retry (`scripts/v2_traffic_orchestrator.py`): 75s total budget, 5s
fixed backoff, retries ONLY `DrainQueryError` (transient), never
`DrainTimeoutError` (a genuine "never drained" signal propagates
immediately). Never interprets an unavailable Prometheus as zero in-flight.

## Episode lifecycle

Decision interval: 30s (`configs/experiment.yaml:
rl.control_interval_seconds`). Profile duration varies per-profile (600-
2400s V1, 600-1200s V2). V2 deterministic rotation
(`training_schedule_v2.py::build_training_profile_rotation`): a persistent
seeded `random.Random(seed)` instance reshuffles a fresh copy of the
5-profile allowlist every cycle, guaranteeing exactly one occurrence of
each profile per 5-episode window, reproducible given `(seed, n_episodes)`.
Invalid-step limit / max-consecutive-invalid-steps: governed by
`max_consecutive_invalid_steps` in `EnvConfig`/`EnvConfigV2` (value not
re-derived this session; unchanged from the frozen MDP spec per
`docs/rl_mdp_spec.md`).

## Reward — V1 (`reward.py`, frozen) vs. V2 (`reward_v2.py`)

V1 weights: w_sla=1.0, w_reliability=5.0, w_resource=1.0, w_action=0.1,
w_reversal=0.5, w_boundary=0.2, w_efficiency=0.3; SLA target p95=300ms.

**V2 — 7 components** (`reward_v2.py`, formulas and weights confirmed via
Phase 4B-R4F's two-agent audit):

| component | formula | weight | range |
|---|---|---|---|
| reliability | `w·clip01(error_rate)` | 6.0 | [0,6.0] |
| sla | `w·clip01(((p95-300)/300)⁺)²` | 4.0 | [0,4.0] |
| pressure | `w·max(pressure_cpu, pressure_mem)`, each `clip01((x-safe_high)/(hard_limit-safe_high))²` (cpu: 3.0/5.0, mem: 1.2/4.0) | 2.0 | [0,2.0] |
| waste | `w` if idle-and-healthy gate passes, else 0 | 0.3 | {0,0.3} |
| action | `w` if requested_delta≠0 and override∈{NONE,PENDING_SCALE_GATE,SCALE_DOWN_COOLDOWN} | 0.1 | {0,0.1} |
| reversal | `w` if effective_delta≠0 and is_reversal | 0.5 | {0,0.5} |
| boundary | `w` if override==REPLICA_BOUND_CLAMP | 0.2 | {0,0.2} |

`total = -sum(components)`, range `[-13.1, 0]` (no explicit final clip,
only sub-term clips). SLA/pressure/waste/reliability are driven entirely
by the realized OUTCOME (post-gating), never the requested delta — only
action/reversal/boundary distinguish requested-vs-effective, by deliberate
design (confirmed: a pending-scale/cooldown-suppressed request still pays
the small action cost; a bound-clamped one pays boundary instead, mutually
exclusive with action cost). **Two confirmed dead-config-key defects**
(P2, non-blocking): `reward_v2.yaml`'s `min_replicas`/`max_replicas` are
never read by the loader (dataclass defaults 1/8 always used); the
`per_action_cost`/`per_reversal_cost`/`per_boundary_violation_cost=1.0`
multipliers are documented but never read (the actual cost is simply the
weight itself).

## PPO hyperparameters

V1 (`configs/rl.yaml`/`task_outputs/phase_16u_ppo_hyperparameters.json`,
frozen): learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
max_grad_norm=0.5, policy=MlpPolicy, net_arch=[128,128],
total_timesteps=500000 (V1's full budget — NOT the same as V2's).

V2 (same hyperparameter VALUES, different budget): n_steps=256 (not 2048 —
V2 uses a much smaller per-session rollout), total budget
**5120 timesteps = 20 sessions × 256**, training_seed=42, policy
architecture identical [128,128] MlpPolicy, device not pinned explicitly
(SB3 default, CPU on this machine per absence of a GPU/CUDA mention
anywhere in the V2 scripts).

## Action selection / near-tie policy

`argmax`/`model.predict(deterministic=True)` occurs ONLY in the frozen V1
evaluation script (`scripts/evaluate_ppo_vs_hpa.py:449`, plain
`model.predict(obs, deterministic=True)`, no tie-break wrapper — correctly
untouched). The V2 near-tie policy
(`autoscaler/src/rl_env/action_selection.py::select_safe_deterministic_action`)
implements exactly: find every action within `epsilon` of the max
probability; if more than one qualifies, choose HOLD; else choose the sole
max. `epsilon` default `DEFAULT_NEAR_TIE_EPSILON = 0.05`, status
`"provisional_unvalidated"` (recorded per-checkpoint in
`V2ModelMetadata.safe_inference_epsilon_status`, `training_contract_v2.py`).
Wired into `safe_inference.py::infer_safe_action()` → every failure mode
(model None, non-finite observation, extraction failure, malformed
probabilities) collapses to `HOLD`/`OVERRIDE_POLICY_INFERENCE_FAILURE`,
never SCALE_DOWN. Called by `safe_ppo_controller.py::run_safe_ppo_step()`,
whose ONLY production call site is
`scripts/evaluate_ppo_vs_hpa_v2.py:98` — **does `evaluate_ppo_vs_hpa_v2.py`
use the safe selector?** YES. **Confirmed DEFECT** (from Phase 4B-R4F,
unresolved): that same script (and `safe_ppo_controller.py` underneath it)
actually constructs and steps the **V1** `KubernetesAutoscalerEnv`/
`configs/rl.yaml`, not V2's — so if ever executed it would score with V1's
reward while being labeled V2. Never yet executed (`--execute` never
invoked in this project's history).

**Does training ever use the near-tie selector?** NO, confirmed by direct
grep (none of `train_ppo_live.py`, `train_ppo_live_session.py`,
`train_ppo_final_v2.py`, `train_ppo_live_v2.py` import
`action_selection`/`safe_inference`/`safe_ppo_controller`) AND
test-enforced (`tests/unit/test_training_excludes_safe_tie_override.py`
AST-parses the training scripts to assert this structurally).

**Critical empirical finding (computed directly in Phase 4B-R4F, real data,
not reconstructed)**: using the actual `session_002` checkpoint
(`sha256 c721144f...`) against the actual 768 real `normalized_observation`
vectors recorded during training: margin (top1-top2) min=0.0000,
max=0.0654, mean=0.0233, median=0.0202. **91.0% of all 768 real decisions
would be forced to HOLD at epsilon=0.05.** Plain-argmax distribution:
HOLD=634 (82.6%), SCALE_UP=124 (16.1%), SCALE_DOWN=10 (1.3%). This is
strong evidence epsilon=0.05 must NOT be adopted as final before a
post-training recalibration.

## Contract versioning

`rl_contract_version: 2` in BOTH `configs/rl.yaml` and `configs/rl_v2.yaml`
is a loader-schema version (the 11-element observation format), NOT the
same thing as "project Phase V2" — it was deliberately kept unchanged for
V2 since the observation/action mechanics are byte-identical between V1
and V2 (only the reward formula differs). Checkpoint/schema compatibility:
`checkpoint_state.verify_resume_provenance()` compares a
`governing_hashes_snapshot` (sha256 of `environment_v2.py`, `reward_v2.py`,
`observation.py`, `configs/rl_v2.yaml`, `configs/reward_v2.yaml`) between
sessions, raising `CheckpointProvenanceError` on any drift — confirmed
unchanged since Phase 4A (no git diff on any of these 5 files per Part 05).
SB3's own `PPO.load()` additionally provides architecture-level protection
(would fail to reconstruct the first linear layer on a dimensionality
mismatch).
