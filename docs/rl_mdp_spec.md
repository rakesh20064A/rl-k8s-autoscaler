# RL Autoscaler MDP Contract — v2 (FROZEN)

**Status**: FROZEN by Phase 15A.1, superseding v1 (Phase 15A). This is the single
authoritative specification of the RL decision problem (observation, action, reward, timing,
safety, episode semantics). The Gymnasium environment, live controller, and PPO training code
that will be built in later phases must implement this contract exactly — do not silently
diverge from it in code. Any future change requires a new, explicit correction entry
(mirroring `docs/experimental_design_corrections.md`'s CORR-N convention) and a version bump
below.

```
rl_contract_version: 2
reward_contract_version: 2
frozen_in_phase: 15A.1
frozen_date_utc: 2026-09-09
supersedes: rl_contract_version 1 (Phase 15A) -- see CORR-27 and the Change Log at the end
            of this document for exactly what changed and why.
```

This document does not contain narrative rationale — see
`task_outputs/phase_15a1_rl_mdp_contract_integrity_corrections.md` for the full discussion,
evidence, and alternatives considered. This file is the terse, implementation-facing contract.

---

## 0. Implementation status (as of this freeze)

**Still zero RL code exists** (unchanged since Phase 15A). `autoscaler/src/`,
`autoscaler/tests/`, `training/`, `k8s/rl-agent/` are empty directories. No Gymnasium `Env`,
no PPO training script, no trained model, no simulator, no live controller.
`experiments/run.py` (referenced by `scripts/run_rl_experiment.sh --controller rl`) does not
exist. `gymnasium`, `stable-baselines3`, `torch` are pinned in `requirements/rl.txt` but are
**not installed** in `.venv` and are **not imported anywhere** in the codebase. This document
specifies the target contract for the implementation that has not yet begun. This phase
(15A.1) made only specification/config/test corrections — no environment code was written.

---

## 1. Decision interval

```
RL_DECISION_INTERVAL_SECONDS = 30
```

Unchanged from v1. Authoritative value in `configs/experiment.yaml: rl.control_interval_seconds`.
Must be used identically by the offline environment, live controller, training, and evaluation.

## 2. Observation timing rule

At decision step with timestamp `t`:
- Only Prometheus samples with `timestamp <= t` may be used.
- The "current step value" for each raw metric is the mean of all Prometheus samples whose
  timestamp falls in `(t - RL_DECISION_INTERVAL_SECONDS, t]` (nominally 3 samples at the
  10s scrape cadence; fewer if scrapes were delayed).
- A sample is "stale" if the most recent available sample for a required metric is older
  than `configs/experiment.yaml: prometheus.staleness_threshold_seconds` (30s) before `t`.
  Stale ⇒ that metric is treated as missing for this step (§9/§13 safety fallback).
- No traffic-schedule or future-timestep information is ever exposed to the observation.

## 3. Raw metrics used (Prometheus only, per CORR-4/CORR-9)

Unchanged from v1. All from `collectors/metrics_collector.py`'s existing PromQL query set
(`app_ns=autoscaler-app`, `dep_name=demo-app`, `route=/work`) — no new PromQL introduced:

| Raw metric | PromQL | Missing/None semantics |
|---|---|---|
| `request_rate_rps` | `sum(rate(demo_app_requests_total{route="/work"}[1m]))` | null if target down or query error; never invented as 0 |
| `error_rate_rps` (5xx only) | `sum(rate(demo_app_requests_total{route="/work",status_code=~"5.."}[1m]))` | 0.0 if `request_rate_rps` is a real (non-null) value and this series is absent without error; null if `request_rate_rps` is null |
| `latency_p95_seconds` | `histogram_quantile(0.95, sum(rate(demo_app_request_duration_seconds_bucket{route="/work"}[1m])) by (le))` | see §6 (v2-corrected zero-traffic semantics) |
| `requests_in_progress` | `sum(demo_app_requests_in_progress{route="/work"})` | 0.0 is a genuine reading (gauge) |
| `cpu_cores` | `sum(rate(container_cpu_usage_seconds_total{pod=~"demo-app-.*"}[1m]))` | null on query failure |
| `memory_bytes` | `sum(container_memory_working_set_bytes{pod=~"demo-app-.*"})` | null on query failure |
| `replicas_ready` (`deployment_status_ready_replicas`) | `kube_deployment_status_replicas_ready{deployment="demo-app"}` | null on query failure |
| `replicas_current` (`deployment_status_replicas`) — **new in v2, used only by the resource reward term, §11** | `kube_deployment_status_replicas{deployment="demo-app"}` | null on query failure |

**Live-controller-observable reliability is 5xx-only.** A live RL controller has no access to
client-side transport outcomes (`timeout`, `connection_error`, `dropped_overload`) that only
exist in the traffic generator's own `requests.csv` — that file is offline evaluation-only
telemetry never available to a running policy. This is a strictly weaker reliability signal
than the one used for HPA-tuning analysis (CORR-25/CORR-26) and must never be conflated with
it in any report comparing RL and HPA.

## 4. CPU/memory/in-progress ambiguity resolution (per-Ready-replica, not aggregate)

```
cpu_per_replica_frac_request     = (cpu_cores / ready_replicas) / 0.100        # 0.100 cores = 100m CPU request
memory_per_replica_frac_request  = (memory_bytes / ready_replicas) / 67108864  # 64 MiB request, in bytes
in_progress_per_replica          = requests_in_progress / ready_replicas
```

**[CORR-27 wording correction, v2]** `cpu_per_replica_frac_request` is a **nominally
comparable CPU-request-relative utilization proxy** to Kubernetes HPA's own
`averageUtilization` (both express aggregate pod CPU as a fraction of the 100m CPU request,
CORR-20 §1) — it is **not** literally identical to HPA's internal utilization signal. HPA's
actual computed value additionally depends on Metrics Server's own sampling/staleness
behavior, its eligible-pod selection rules, its handling of pods without a usable metric yet,
not-yet-Ready-pod exclusion, the HPA controller's own reconciliation timing, and conservative
recalculation rules internal to `kube-controller-manager` — none of which this RL feature
reproduces. The feature itself is unchanged from v1; only the claim of exact equivalence is
withdrawn.

**`ready_replicas == 0` fallback** (should not occur in normal operation since
`min_replicas=1`, but must not divide by zero if it transiently does): if `ready_replicas == 0`
for a step, none of the three per-replica ratios above can be computed — treat all three as
missing for that step (not zero, not infinity) and trigger the §13 safety fallback (HOLD),
with `action_override_reason="stale_or_missing_metric"` (§13). This is distinct from, and
takes priority over, ordinary metric staleness.

## 5. History/trend representation

Unchanged mechanism from v1. For `cpu_per_replica_frac_request`, `request_rate_rps`, and
`latency_p95_seconds` only (the three signals most directly linked to proactive SLA
avoidance):

```
ewma_t = alpha * step_value_t + (1 - alpha) * ewma_(t-1)      # alpha = 2 / (N+1), N = 3 steps (90s smoothing span)
delta_t = step_value_t - ewma_(t-1)                            # momentum: how far this step deviates from the recent trend
```

`ewma_0` (first decision step of an episode) is initialized to `step_value_0` (delta = 0 at
episode start). **[v2 addition]** The EWMA update for `latency_p95_seconds` is only performed
using a **valid** `step_value_t` — either a genuinely-observed latency, or the `no_traffic`
semantic `0ms` value (§6) — both are legitimate values and update the EWMA normally. If the
step is structurally invalid (positive traffic but latency unavailable, §6), `ewma_t` is
**not** updated at all for that step (it is carried forward unchanged from `ewma_(t-1)`, the
same "hold, never fabricate" rule already applied to other missing features, §7).

## 6. Zero-traffic vs. missing-latency semantics — [CORR-27 correction, v2]

Prometheus's latency histogram legitimately has no observations when there is no traffic —
this is not a telemetry failure and must not be treated as one:

- If `request_rate_rps == 0` (a genuine, non-null zero) **and** the latency histogram has no
  observations in the window: `latency_p95_seconds` is defined as `0.0` (semantic, not a
  measurement), `latency_observation_state = "no_traffic"`. This value participates normally
  in normalization (§7 index 6/7) and in the EWMA (§5) — it is a real, valid observation of
  "nothing happened," not a fabricated placeholder.
- If `request_rate_rps > 0` **but** `latency_p95_seconds` is unavailable or stale: this step
  is **structurally invalid** (`latency_observation_state = "invalid"`) and invokes the
  deterministic safety HOLD (§13), `action_override_reason="stale_or_missing_metric"`. The
  midpoint-of-range fallback (§7) never applies to this case — an invalid step is handled
  entirely through §13's HOLD/invalid-step machinery, not through feature-level interpolation.
- Otherwise (`request_rate_rps > 0` and latency available): `latency_observation_state = "observed"`.

This v2 rule replaces v1's undifferentiated "missing ⇒ hold previous value or midpoint" rule
specifically for latency — v1 did not distinguish "no traffic" from "telemetry failure," which
could have caused a legitimate idle period to look identical to a genuine metrics outage.

## 7. Frozen observation vector (v2) — exact order, same 11 elements as v1

**[CORR-27 correction, v2]** Only the CPU/CPU-delta raw clip range changed (§8). No feature
was added or removed and no index changed position; the semantic-version bump is required
because normalization and missing-data semantics changed underneath several entries (§4, §6),
not because the vector's shape changed.

| idx | name | raw source | raw clip range | transform | final range | temporal |
|--:|---|---|---|---|---|---|
| 0 | `cpu_per_replica` | §4 | **[0, 5.0]** (5.0 = limit/request physical ceiling, CORR-9/Phase-10; was `[0,4.0]` in v1) | `clip(x,0,5)/5` | [0,1] | current step |
| 1 | `cpu_per_replica_delta` | §5 | **[-5.0, 5.0]** (was `[-4.0,4.0]` in v1) | `clip(delta,-5,5)/5` | [-1,1] | trend (90s EWMA) |
| 2 | `memory_per_replica` | §4 | [0, 4.0] (limit/request=4.0 exactly; unchanged) | `clip(x,0,4)/4` | [0,1] | current step |
| 3 | `in_progress_per_replica` | §4 | [0, 20.0] (unchanged) | `clip(x,0,20)/20` | [0,1] | current step |
| 4 | `request_rate_norm` | `request_rate_rps` | [0, 100.0] RPS (unchanged; re-justified without held-out data, §12) | `clip(x,0,100)/100` | [0,1] | current step |
| 5 | `request_rate_delta` | §5 | [-100.0, 100.0] (unchanged) | `clip(delta,-100,100)/100` | [-1,1] | trend (90s EWMA) |
| 6 | `p95_latency_norm` | `latency_p95_seconds` | [0, 2000] ms (unchanged); `0` if `latency_observation_state=="no_traffic"` (§6) | `clip(x_ms,0,2000)/2000` | [0,1] | current step |
| 7 | `p95_latency_delta` | §5 | [-2000, 2000] ms (unchanged) | `clip(delta_ms,-2000,2000)/2000` | [-1,1] | trend (90s EWMA) |
| 8 | `error_rate` | `error_rate_rps / request_rate_rps` | [0,1] (already a fraction; 0.0 if `request_rate_rps==0`; unchanged) | `clip(x,0,1)` | [0,1] | current step |
| 9 | `ready_replica_norm` | `replicas_ready` | [1,8] (unchanged) | `(clip(x,1,8)-1)/7` | [0,1] | current step |
| 10 | `seconds_since_last_scale_norm` | controller-tracked | [0, 300] s (unchanged); counts seconds since the last **effective** scaling action (§10) | `clip(x,0,300)/300` | [0,1] | current step |

`observation_space = gymnasium.spaces.Box(low=[0,-1,0,0,0,-1,0,-1,0,0,0], high=[1]*11, shape=(11,), dtype=float32)` — unchanged shape/bounds from v1.

**Missing-data behavior (indices 0,1,2,3,4,5,7 — index 6 follows §6 instead)**: if a raw input
is unavailable for the current step, that feature is held at its previous step's *normalized*
value (never fabricated as 0.0 or clipped-max) and `action_override_reason="stale_or_missing_metric"`
is recorded for the step (§13). At the very first step of an episode with no prior value, the
feature defaults to its raw range's midpoint and the step is flagged. Index 8 (`error_rate`)
follows its own null rule (§3). Indices 9–10 are controller-tracked state, never missing.

This ordering is part of the model contract, stable across training, saved PPO models, and
inference. Any further change to order/shape requires another `rl_contract_version` bump.

## 8. CPU normalization correction — [CORR-27, v2]

v1 clipped `cpu_per_replica` at `4.0`× the CPU request. This was **too low**: the true
physical ceiling is `limit/request = 500m/100m = 5.0`×, and Phase-10's single-replica
saturation calibration (10 RPS) recorded a **steady-state mean** of `0.463` cores = **4.63×**
request (with individual samples reaching the hard `0.500`-core limit = exactly `5.0×`) —
both above v1's `4.0×` clip, meaning v1 would have silently truncated genuine severe-pressure
readings before they ever reached the policy. **v2 freezes the raw clip range at `[0.0, 5.0]`**
for both `cpu_per_replica` (index 0) and its delta (index 1, symmetric `[-5.0,5.0]`), so the
observation space's most severe end now spans the entire physically possible range, not an
arbitrary truncation below it.

## 9. Action encoding — exact, immutable (unchanged from v1)

```
0 = SCALE_DOWN_ONE   (requested_replica_delta = -1)
1 = HOLD             (requested_replica_delta =  0)
2 = SCALE_UP_ONE     (requested_replica_delta = +1)
```

`action_space = gymnasium.spaces.Discrete(3)`. Matches `configs/rl.yaml: action_space.action_map`
verbatim — **unchanged by v2**, per the task's explicit instruction that the action encoding
may remain unchanged even though the contract version bumps.

**Boundary handling**: `effective_replica_delta = clip(current_desired + requested_replica_delta, 1, 8) - current_desired`.
The action itself never violates `[1,8]` at the infrastructure level. `requested_replica_delta`
and `effective_replica_delta` are always recorded separately (§14 transition schema).

## 10. Desired vs. effective capacity (semantics the future simulator/controller must support)

Unchanged from v1. Four distinct, separately tracked quantities per CORR-25's evidence
hierarchy: RL controller's own last requested value (analogue of `hpa_status_desired_replicas`),
`deployment_spec_desired_replicas` (what was actually written to the Deployment),
`deployment_status_replicas` ("current" — pods that exist, used by the v2 resource reward
term, §11), `deployment_status_ready_replicas`/`available` (physical serving capacity, used by
all per-replica pressure ratios and by pending-scale gating, §12). A scale-up request does not
teleport Ready replicas — Ready lags the desired-replica mutation by ~10s empirically
(readinessProbe `initialDelaySeconds=5`+`periodSeconds=5` plus scheduling/container-start,
CORR-25). A scale-down drops `current`/`ready`/`available` immediately, together with the
desired-replica decrease (CORR-25 §4). Not implemented (simulated) in this phase — frozen as a
requirement for the environment that will be built later.

## 11. Resource-reward metric: current replicas, not Ready replicas — [CORR-27, v2]

Phase 11C.4P / CORR-26 selected the HPA baseline primarily using **Deployment current
replica-seconds** (Ready replica-seconds were explicitly secondary corroboration only, per
that phase's own methodology). A newly-created-but-not-yet-Ready pod already consumes real
compute resources (it is scheduled, pulling/running its image, being probed) even though it is
not yet serving traffic. Conflating "resources consumed" with "capacity currently serving" is
a category error the v1 reward made by using `ready_replicas` for the resource term. **v2
freezes**:

```
resource_fraction = (deployment_current_replicas - 1) / 7      # NOT (ready_replicas - 1) / 7
```

clipped to `[0,1]`, using the new `replicas_current`/`deployment_status_replicas` raw metric
(§3). **Ready replicas continue to be used exclusively for**: the observation vector's
per-replica CPU/memory/in-progress normalization (§4) and `ready_replica_norm` (§7 index 9),
and pending-scale gating (§12) — i.e., for *serving-capacity* questions. **Current replicas**
are used exclusively for the *resource-consumption* reward term. These two concepts are never
conflated again.

## 12. Cooldown / action-gating rule — asymmetric, [CORR-27 correction, v2]

v1 froze a **symmetric** 60s minimum spacing between all effective actions. This was wrong: it
could unnecessarily delay an urgently-needed scale-up during a severe, fast-rising workload
increase, and it was in tension with the ±1 action space's own responsiveness rationale. **v2
freezes an asymmetric rule**, still deliberately not a copy of HPA's own stabilization windows:

1. **Pending-scale gating (unchanged from v1, applies to both directions)**: if
   `deployment_spec_desired_replicas != deployment_status_ready_replicas` at decision time `t`
   (a previous capacity transition has not yet been fully realized), any non-HOLD policy
   action is converted to an *effective* HOLD for this step
   (`requested_replica_delta` still recorded verbatim; `effective_replica_delta=0`;
   `action_override_reason="pending_scale_gate"`). This prevents "repeated +1 requests while a
   pod is already starting" for scale-ups, and prevents compounding scale-downs before the
   Deployment has settled.
2. **SCALE_UP: no additional cooldown.** If observations are valid (§13) and no transition is
   pending (rule 1 does not apply), a `SCALE_UP_ONE` action is **effective at the very next
   normal 30-second decision point** — there is no extra 60s scale-up-specific cooldown.
   Necessary, repeated scale-ups may therefore occur once per decision interval, gated only by
   pending-scale readiness, not by an arbitrary timer. This directly addresses responsiveness
   to a severe or accelerating load increase.
3. **SCALE_DOWN: 60-second minimum spacing (unchanged numeric value from v1, now
   direction-specific instead of symmetric)**: a `SCALE_DOWN_ONE` action requires both (a) no
   transition pending, and (b) at least `configs/experiment.yaml: scaling.cooldown_seconds`
   (60s = 2 decision intervals) since the most recent **effective** scaling action (up or
   down). A scale-down requested before 60s have elapsed is converted to effective HOLD,
   `action_override_reason="scale_down_cooldown"`. This remains conservative about capacity
   removal without constraining responsiveness to rising load.
4. **Emergency reversal is explicitly permitted**: a `SCALE_UP_ONE` must **not** be blocked
   merely because a `SCALE_DOWN_ONE` occurred 30 seconds earlier, provided observations are
   valid and no transition is pending — rule 3's 60s spacing applies only to *another*
   scale-down, never to a scale-up that follows a recent scale-down. The reward's reversal
   penalty (§17) may still apply to discourage this pattern if it represents genuine
   oscillation — but the **deterministic guardrail never blocks a needed scale-up**, per the
   task's explicit priority ordering (responsiveness to rising load over scale-down-cooldown
   symmetry).

**Full-traverse timing, stated precisely relative to the first effective action (not episode
start)**: once the **first effective `SCALE_UP_ONE`** occurs at some decision time `t0`, and
assuming every subsequent step's observations remain valid and no telemetry failure gates the
decision, a fully-committed policy could in principle request `SCALE_UP_ONE` again at
`t0 + 30s` — but whether that step's `effective_replica_delta` is actually nonzero depends on
whether `deployment_status_ready_replicas` has caught up to `deployment_spec_desired_replicas`
by then (rule 1). Empirically (CORR-25) that catch-up takes ~10s, comfortably inside one 30s
decision interval, so in the common case a 1→8 traverse would proceed at roughly one effective
`+1` every 30s once begun — i.e. **roughly 210 seconds after the first effective scale-up**,
not 210 seconds from episode start, and not guaranteed if readiness is ever slower than one
decision interval (in which case the relevant step's action is gated per rule 1 and the
traverse takes correspondingly longer). This is a description of expected timing under
`SCALE_UP`'s no-extra-cooldown rule, not a hard bound the environment enforces.

## 13. Invalid-step counting and deterministic safety guardrails

**[CORR-27 correction, v2]** v1's "more than 3 consecutive invalid steps" was ambiguous about
the exact trigger point and reset rule. v2 freezes:

- `consecutive_invalid_steps` starts at 0 at episode reset.
- Any decision step where `sample_valid==False` (CORR-9's own definition — target down,
  essential telemetry missing, or a query error) increments `consecutive_invalid_steps` by 1
  and produces an effective HOLD (this step's reward is handled per §18, not §17's normal
  formula).
- **A single fully-valid decision step resets `consecutive_invalid_steps` to 0** — validity is
  not "sticky" across a recovered step.
- Invalid steps 1–3 (i.e. `consecutive_invalid_steps` reaching 1, 2, or 3): deterministic HOLD,
  episode continues.
- The **4th consecutive invalid decision step** (`consecutive_invalid_steps` reaching 4) sets
  `terminated=True` — with a 30s decision interval this corresponds to **120 seconds of
  persistent decision-time telemetry failure**.
- **Zero Ready replicas for more than 120 seconds** remains a separate, explicit termination
  condition (distinct from the telemetry-invalidity counter, since a Deployment could in
  principle report perfectly valid — but zero — Ready replicas, e.g. a genuine crash-loop,
  which is a different failure mode than "we can't observe the state at all").

| Condition | Deterministic behavior | `action_override_reason` |
|---|---|---|
| `requested_replica_delta` would take replicas outside `[1,8]` | Clamp; effective delta reduced/zeroed | `replica_bound_clamp` |
| Pending capacity transition (§12 rule 1) | Effective HOLD | `pending_scale_gate` |
| Scale-down requested within 60s of the last effective action (§12 rule 3) | Effective HOLD | `scale_down_cooldown` |
| Required metric stale/missing, or `ready_replicas==0` (§4/§7) | Effective HOLD; `consecutive_invalid_steps += 1` | `stale_or_missing_metric` |
| Prometheus unreachable | Effective HOLD; `consecutive_invalid_steps += 1` | `prometheus_unavailable` |
| Policy inference raises an exception | Effective HOLD; error logged | `policy_inference_failure` |
| Kubernetes API patch call fails | Effective HOLD; no blind retry within the step | `k8s_api_failure` |
| A previous step's K8s patch is still in flight when the next decision is due | Apply only the latest decision; never double-apply | (not a per-step override reason; a controller-level invariant) |
| A live HPA object targeting the same Deployment is detected at controller start | Refuse to start (fail closed) | n/a (pre-episode check) |
| None of the above | Normal operation | `none` |

In every failure case the deterministic fallback is **HOLD** (preserve current desired
replicas) — never an arbitrary reset and never blind retry-until-success.

## 14. Transition record schema — v2 additions

**[CORR-27 addition]** Superset of v1's field list, with new fields to distinguish the
corrected v2 semantics:

`episode_id, step_index, decision_timestamp_utc, raw_observation{name:value}, normalized_observation[11], action_index, semantic_action, requested_replica_delta, effective_replica_delta, deployment_desired_replicas_before, deployment_desired_replicas_after, deployment_current_replicas_before, deployment_current_replicas_after, ready_replicas_before, ready_replicas_after, available_replicas_before, available_replicas_after, reward_total, reward_components{sla, reliability, resource, action, reversal, boundary, efficiency_bonus}, terminated, truncated, action_override_reason, bound_clamped, pending_scale, consecutive_invalid_steps, reward_validity, latency_observation_state, sample_valid, observation_age_seconds`.

`bound_clamped` and `pending_scale` are booleans derived from (but kept alongside, for query
convenience) `action_override_reason` (`bound_clamped = (action_override_reason ==
"replica_bound_clamp")`; `pending_scale = (action_override_reason == "pending_scale_gate")`).
`reward_validity` is `"full"` for a normally-computed step or `"partial"` for a step
overridden per §18.

## 15. Frozen v2 reward formula

```
reward_t = -R_sla - R_reliability - R_resource - R_action - R_reversal - R_boundary + R_efficiency
```

| Component | Formula | Coefficient | Max magnitude | v2 change |
|---|---|--:|--:|---|
| `R_sla` | `w_sla * clip(excess^2, 0, 4.0)`, `excess = max(0, p95_ms-300)/300` | `w_sla=1.0` | 4.0 | unchanged |
| `R_reliability` | `w_reliability * error_rate` (§3, 5xx-only, live-observable) | `w_reliability=5.0` | 5.0 | unchanged |
| `R_resource` | `w_resource * (deployment_current_replicas-1)/7` | `w_resource=1.0` | 1.0 | **uses `current`, not `ready` (§11)** |
| `R_action` | `w_action * 1{policy requested a non-HOLD action on a valid step}` (§17) | `w_action=0.1` | 0.1 | **based on requested, not effective — see §17** |
| `R_reversal` | `w_reversal * 1{direction reversal, computed from EFFECTIVE actions only, within reversal_window_seconds}` | `w_reversal=0.5` | 0.5 | clarified: effective-only |
| `R_boundary` | `w_boundary * 1{action_override_reason == "replica_bound_clamp"}` | `w_boundary=0.2` | 0.2 | **narrowed from "requested != effective" to bound-clamp only (§16)** |
| `R_efficiency` | `w_efficiency * 1{cpu_per_replica in [1.0,3.0] AND p95_ms<300 AND error_rate==0}` | `w_efficiency=0.3` | +0.3 | **band upper bound 3.0, re-derived independent of HPA (§19)** |

`reversal_window_seconds = 120` (4 decision intervals; unchanged from v1).

**Expected per-step magnitude** (documented, not trained — ranges shift slightly from v1 only
where the resource/efficiency changes matter): healthy/efficient HOLD step ≈ +0.30; mild SLA
violation (p95≈350ms, 2 current replicas) ≈ -0.17; severe SLA violation (p95≈1500ms, some
errors) ≈ -4.6 to -5.6 (bounded, never unbounded); an unnecessary scale action while already
healthy ≈ -0.1 to -0.6 depending on reversal. No single component can be numerically dwarfed
or dwarf another — the resource term alone can never justify tolerating a severe SLA breach.

## 16. Boundary vs. other overrides — [CORR-27 correction, v2]

v1 defined the boundary penalty as firing whenever `requested_replica_delta !=
effective_replica_delta` — too broad, since that mismatch can also arise from pending-scale
gating, scale-down cooldown, or any telemetry-failure HOLD, none of which represent the policy
requesting a physically impossible replica count. **v2 freezes**: `R_boundary` fires **only**
when `action_override_reason == "replica_bound_clamp"` (i.e. the requested action would have
taken replicas outside `[1,8]`). It never fires for `pending_scale_gate`, `scale_down_cooldown`,
or any of the telemetry-failure reasons — those are legitimate deterministic timing/safety
decisions, not the policy asking for something impossible.

## 17. Action-penalty and reversal-penalty semantics — [CORR-27 correction, v2]

- `R_action` is evaluated on **`requested_replica_delta`**, not `effective_replica_delta`,
  but **only on a step where `sample_valid==True`** (a step the policy could meaningfully
  evaluate). Concretely: `R_action` fires when the policy requested a non-HOLD action **and**
  `action_override_reason` is one of `{"none", "replica_bound_clamp", "pending_scale_gate",
  "scale_down_cooldown"}` (all of which reflect the policy actually making a decision from
  valid data, whether or not that decision was realized) — **and does not fire** when
  `action_override_reason` is a telemetry-failure reason (`stale_or_missing_metric`,
  `prometheus_unavailable`, `policy_inference_failure`, `k8s_api_failure`), since in those
  cases the policy could not meaningfully evaluate the situation (§18).
- **No double-counting**: when `action_override_reason=="replica_bound_clamp"`, only
  `R_boundary` fires for that step's request-vs-effective mismatch; `R_action` does **not**
  additionally fire for the same event (the boundary penalty already captures "the policy
  asked for something and it didn't happen because of a bound," and stacking the generic churn
  cost on top would double-penalize one decision). For `pending_scale_gate` and
  `scale_down_cooldown`, `R_action` **does** fire (a legitimate, in-bounds request that was
  deterministically deferred still reflects the policy attempting to churn) but `R_boundary`
  does not (no bound was violated).
- `R_reversal` is computed **only from effective (non-zero `effective_replica_delta`) scaling
  actions** — a requested-but-gated/clamped action is never treated as establishing or
  reversing a "direction" for oscillation-tracking purposes, since nothing physically changed.

## 18. Reward behavior on invalid/overridden steps — [CORR-27, v2]

When a step's `action_override_reason` is one of the four telemetry-failure reasons
(`stale_or_missing_metric`, `prometheus_unavailable`, `policy_inference_failure`,
`k8s_api_failure`), the reward for that step is **not** computed from held/fabricated
observations. `reward_validity="partial"` and:

- `R_sla = 0` (no real p95 reading to penalize against)
- `R_reliability = 0` unless a genuine, valid `error_rate` observation exists independently of
  the failure (rare, but possible if only e.g. the latency histogram query failed while the
  error-rate query succeeded — each raw metric's own success/failure is tracked independently,
  §3)
- `R_action = 0`, `R_reversal = 0`, `R_boundary = 0` (§17 — the policy's action was not a real,
  fully-evaluated decision)
- `R_efficiency = 0` (no bonus without a genuine healthy-state observation)
- `R_resource` **may still be computed** if `deployment_current_replicas` was independently
  and successfully obtained for this step (it is a separate Kubernetes/kube-state-metrics
  query from the application-level metrics that may have failed, §3) — resource consumption is
  a real, ongoing physical fact independent of whether Prometheus's *application* telemetry is
  healthy.

This prevents the training algorithm from ever being trained against invented latency/error
values during a telemetry fault, while still not pretending that ongoing infrastructure
resource consumption stopped just because observability temporarily did.

## 19. Efficiency-band re-derivation, independent of the HPA baseline — [CORR-27, v2]

v1 set the efficiency bonus's CPU band upper bound to `2.6`× request specifically because the
HPA baseline selected in CORR-26 is `averageUtilization=260`. This coupled the RL reward
directly to the outcome of an unrelated controller's hyperparameter search. **v2 re-derives
the upper bound from application-only, no-HPA calibration evidence** — specifically Phase
11C.4D's controlled, HPA-deleted, fixed-1-replica, final-path (genuine Service load-balancing,
`connection_mode=fresh`) Metrics-API CPU map (CORR-23 §4):

| RPS | Metrics-API steady CPU (median of 3 repeats) | as fraction of 100m request | client p95 (all levels 3–7 RPS) |
|--:|--:|--:|--:|
| 5.5 | 279.5m | 2.80x | ≤~135ms |
| **6.0** | **297.4m** | **2.97x ≈ 3.0x** | ≤~135ms |
| 7.0 | 338.8m | 3.39x | ≤~135ms |

At 6 RPS (single replica, no HPA), CPU load is ≈3.0× the CPU request while client p95 remains
comfortably under the 300ms SLA (never exceeding ~135ms at any tested level from 3–7 RPS,
CORR-23 §4) — i.e. 3.0× request is application-verified as a **safe, still-SLA-compliant**
operating point, entirely independent of any HPA candidate-selection outcome. **v2 freezes the
efficiency band at `[1.0, 3.0]`× request** — deliberately *not* `2.6` (the HPA value), to make
the independence concrete and auditable: if v2's band happened to still read `2.6`, one could
not distinguish "independently re-derived and coincidentally similar" from "not actually
decoupled." `3.0 != 2.6` demonstrates the decoupling directly. The bonus term itself
(`w_efficiency=0.3`) was judged still useful (not redundant with the resource term, since
resource alone cannot distinguish "efficiently loaded" from "idle" — only the CPU-band
membership does) and is retained rather than removed.

## 20. Episode contract, including v2 first-decision semantics — [CORR-27 addition]

- One episode = one full traffic-profile execution, `t=0` to that profile's total duration
  (variable length across profiles).
- **Reset sequence (frozen, precise)**:
  1. Deployment scaled to `initial_replicas=1` (`configs/experiment.yaml`).
  2. Episode does not proceed until `ready_replicas==1` is confirmed.
  3. A fixed 60-second pre-episode stabilization window elapses (mirrors the HPA-tuning
     convention, for HPA/RL episode-framing parity). Prometheus samples collected during this
     window are used **only** to initialize each trend feature's EWMA (§5) — never treated as
     part of the scored episode itself.
  4. The traffic profile's own schedule starts at episode `t=0` immediately after
     stabilization ends.
  5. The policy's first decision (`step_index=0`) occurs at `t=0` and may use **only**
     observations derived from samples at or before `t=0` (i.e., the stabilization-window
     data, per step 3 — the new workload has not yet had time to generate any evidence of its
     own). Consequently the first decision will typically, though not necessarily, be `HOLD`,
     since the pre-episode state was already stable at 1 replica.
  6. The next decision (`step_index=1`, `t=30s`) may use only data observed through `t=30s`.
  7. No policy action may ever see traffic samples generated after its own decision timestamp
     — this is a hard invariant carried over unchanged from v1's observation-timing rule (§2).
- `truncated=True` when the traffic profile's scheduled duration completes (normal, expected
  time-limit end — not a failure).
- `terminated=True` per the exact counter rule in §13 (4th consecutive invalid step, or zero
  Ready replicas for >120s).
- Replicas are never reset mid-trajectory.

## 21. Reward is not a policy input — [CORR-27 clarification]

Reward is required for **offline** purposes only: simulator/environment training, validation
during training (`configs/rl.yaml: training.eval_freq`), and optional post-hoc reward
telemetry/auditing from the transition log (§14). **A deployed inference controller does not
receive or need the reward value in order to select an action** — at inference time, the
policy network maps `observation -> action` directly; reward is never computed or consulted as
part of live action selection. (v1's prose did not explicitly claim otherwise, but also never
stated this distinction — it is made explicit here to prevent any future implementation from
assuming the live controller needs a reward-computation path at all.)

## 22. Train / validation / held-out data status — [CORR-27 correction, v2]

- **`configs/traffic_profiles/unseen_eval.yaml` is NO LONGER ELIGIBLE AS THE SOLE FORMAL FINAL
  HELD-OUT EVALUATION PROFILE.** Marked:
  `NOT_ELIGIBLE_AS_THE_SOLE_FORMAL_FINAL_HELD_OUT_EVALUATION_PROFILE`.
  **Reason**: Phase 15A inspected this file's contents (its burst magnitude, 80 RPS) and used
  that information as part of the justification for `request_rate_norm`'s raw clip bound
  (100 RPS). This is a genuine, if narrow, information leak from a nominally held-out
  evaluation workload into a state-normalization design decision — the precommitted rule
  (task instructions, this phase and Phase 15A's own) is that held-out evaluation traffic must
  never influence state-feature tuning, reward tuning, PPO tuning, environment calibration, or
  model selection. §7/§12 of this document re-justify the 100 RPS bound using **only**
  non-held-out information (development-candidate profile peaks: `burst`=70, `ramp`=60,
  `step`=50 RPS — none of which is `unseen_eval.yaml` — giving ~30 RPS of headroom above the
  highest known development profile without reference to the held-out file's own magnitude).
  The file itself is **not deleted, not rewritten, not run** — it is preserved exactly as-is;
  only its *epistemic status* as a clean, never-inspected final held-out workload is revoked.
- **Required future action**: a dedicated workload-freeze phase, occurring **after** this v2
  contract is implemented and after training/validation workload families are frozen, must
  define a **new** formal held-out evaluation workload (a new file, new seed, new shape —
  distinct from anything inspected during state/reward design). Once frozen, that new file
  must never again influence any further model/reward/state/hyperparameter selection. This
  phase does **not** create or inspect that replacement file.
- **Candidate, NOT yet frozen** (unchanged from v1): `burst.yaml`, `constant.yaml`,
  `mixed.yaml`, `periodic.yaml`, `ramp.yaml`, `step.yaml` — YAML designs only, never executed.
  A future phase must select training vs. validation profiles among these (or new ones),
  freeze+hash their schedules, and confirm zero overlap with whatever new held-out workload is
  eventually defined.
- Never used for RL training: `configs/traffic_profiles/hpa_tuning/*` and
  `configs/traffic_profiles/hpa_mapping/*` (HPA-specific artifacts, CORR-17–26) — read-only
  reference only.

## 23. PPO / Gymnasium compatibility

Unchanged from v1. `observation_space`: `Box(shape=(11,), dtype=float32)`, bounds per §7.
`action_space`: `Discrete(3)`, encoding per §9. Reward: scalar `float` per step, offline-only
(§21). Episode interface: standard Gymnasium 5-tuple (`obs, reward, terminated, truncated,
info`). Maps directly to Stable-Baselines3 `PPO(policy="MlpPolicy", ...)` as already
configured in `configs/rl.yaml: ppo` (unchanged).

## 24. Reward-validity decomposition, made fully explicit — [CORR-30 clarification, no version change]

**This section clarifies semantics that §15/§17/§18 already implied but never fully spelled
out as an explicit matrix; it introduces no new numeric value, coefficient, or eligibility
rule.** §17 already conditions `R_action`/`R_reversal`/`R_boundary` purely on
`action_override_reason`/`effective_delta`/`is_reversal` — all decision-time quantities — and
§18's zeroing rule is scoped to steps where `action_override_reason` is itself a
telemetry-failure reason (a decision-time condition). Neither section ever conditioned
`R_action`/`R_reversal`/`R_boundary` on the validity of the *resulting* state `s_(t+1)`. An
implementation (Phase 15B.1) collapsed decision-state and outcome-state validity into one
joint flag and, as a result, incorrectly zeroed an already-determined, valid policy action's
action/reversal/boundary consequences whenever the *outcome* observation alone failed. CORR-30
(Phase 15B.1A) corrects the implementation to match this document's own already-authoritative
semantics. The exact four-case matrix, now made fully explicit:

| Case | `s_t` (decision) | `s_(t+1)` (outcome) | `R_sla`/`R_reliability`/`R_efficiency` | `R_action`/`R_reversal`/`R_boundary` | `reward_validity` |
|---|:--:|:--:|---|---|---|
| A | valid | valid | computed normally | computed normally | `full` |
| B | **invalid** | valid | `sla`/`efficiency`=0; `reliability` only if independently valid | all zero (no real decision was evaluated) | `partial` |
| C | valid | **invalid** | `sla`/`efficiency`=0; `reliability` only if independently valid | **preserved** — computed exactly as in Case A, since they depend only on decision-time quantities | `partial` |
| D | **invalid** | **invalid** | same as Case B | same as Case B | `partial` |

`R_resource` may be computed in any case (including B/C/D) whenever `deployment_current_replicas`
was independently obtained for the relevant step — it is a separate Kubernetes/kube-state-metrics
signal from whichever application-level metric may have failed (§18, unchanged). The
governing invariant, stated explicitly for the first time here: **an invalid outcome
observation must never retroactively erase a valid policy action's already-determined
action/reversal/boundary consequences** — doing so would create a perverse incentive structure
in which a policy could effectively "avoid" a churn/reversal/boundary cost merely because
telemetry happened to fail on the very next tick, which has nothing to do with the quality of
the decision itself.

## 25. Episode-start control age — [CORR-30 freeze, no version change]

This document's §20 froze the reset *sequence* (Deployment→1 replica, confirm Ready, 60s
stabilization, episode `t=0`) but never froze the numeric value `seconds_since_last_scale_norm`
(observation index 10) reads at the very first decision. This was a genuine gap, not a
previously-frozen value being changed. **Frozen now**: the logical reset-to-`initial_replicas`
event is treated as an effective scaling action that occurred exactly
`pre_episode_stabilization_s` (60s) before episode `t=0` — i.e. at episode `t=0`,
`seconds_since_last_effective_scale = 60s`, normalized `60/300 = 0.2` under the frozen cap —
regardless of whatever control history (or lack thereof) existed before this reset. This
value then grows normally (`t=30s` with no further scaling ⇒ `90s`, etc.) exactly as if a real
scaling action had occurred at logical time `-60s`. This prevents episode-start state from
depending on arbitrary prior history and gives a deterministic, reproducible value usable
identically by the mechanics backend and any future live backend (which would actually wait
out the 60s stabilization window in real time, arriving at the same logical value). The three
EWMA trend features (indices 1, 5, 7) are separately and independently frozen at exactly `0.0`
at the first decision (§5 — first-valid-value initialization, unchanged), since no in-episode
workload trend exists yet. **Note**: this 60s-elapsed control age does not, by itself, permit a
`SCALE_DOWN` at episode start to bypass the scale-down cooldown check in any meaningful way,
because a `SCALE_DOWN` request when already at `initial_replicas=min_replicas=1` resolves as
`replica_bound_clamp` (§9/§13's precedence: bound-clamp is checked before pending-scale gating
and before the cooldown check) — the cooldown being already-satisfied is irrelevant once the
request is out of bounds.

---

## Change log

- **v1** (Phase 15A, 2026-09-09): initial freeze. Superseded the informal, partially-specified
  `observation_v1` draft in `configs/rl.yaml`.
- **v2** (Phase 15A.1, 2026-09-09, CORR-27): corrected 7 specification-integrity issues found
  reviewing v1 before any environment code was written: (1) CPU/CPU-delta raw clip widened
  `4.0→5.0`× request to match the true physical ceiling and Phase-10 saturation evidence
  (§8); (2) withdrew the claim that the CPU feature is "identical to" HPA's own utilization
  signal, replaced with "nominally comparable proxy" (§4); (3) documented held-out-evaluation
  leakage from Phase 15A's use of `unseen_eval.yaml`'s magnitude in normalization design, and
  revoked that file's eligibility as the sole formal final held-out workload (§22); (4) replaced
  symmetric 60s action-gating with an asymmetric rule (no extra cooldown on necessary
  scale-ups, 60s cooldown retained only for scale-downs, emergency-reversal explicitly
  permitted) (§12); (5) switched the resource reward term from Ready to current replicas, to
  match the CORR-26 HPA-selection methodology's own primary efficiency metric (§11); (6)
  decoupled the efficiency-bonus CPU band from the HPA-selected 260% target, re-deriving
  `3.0`× request from Phase 11C.4D's own no-HPA calibration mapping (§19); (7) narrowed the
  boundary penalty to bound-clamps only, separated it from pending/cooldown/telemetry-failure
  overrides, clarified action/reversal-penalty semantics precisely to avoid double-counting,
  froze zero-traffic-vs-missing-latency semantics, froze exact first-decision and
  invalid-step-termination rules, and clarified reward is never a live-inference input (§6,
  §13, §16–18, §20–21). See CORR-27 for the full narrative and
  `task_outputs/phase_15a1_rl_mdp_contract_integrity_corrections.md` for the complete report.
- **v2 clarification, no version change** (Phase 15B.1A, 2026-09-09, CORR-30): while
  implementing the environment (Phase 15B.1), a defect was found where the code collapsed
  decision-state (`s_t`) and outcome-state (`s_(t+1)`) validity into one joint flag, incorrectly
  zeroing `R_action`/`R_reversal`/`R_boundary` whenever the outcome alone was invalid — contrary
  to §17's already-authoritative wording, which conditions those three components purely on
  decision-time quantities. §24 makes the resulting four-case validity matrix fully explicit
  (no coefficient, formula, or eligibility rule changed) and §25 freezes the previously-unfrozen
  `seconds_since_last_scale_norm` value at episode start (`60/300=0.2`, treating the reset-to-
  `initial_replicas` event as an effective action 60s before `t=0`) — filling a genuine
  specification gap, not altering a previously-frozen value. `rl_contract_version`/
  `reward_contract_version` remain `2`. See CORR-30 and
  `task_outputs/phase_15b1a_reward_validity_and_reset_timing_correction.md` for the complete report.
