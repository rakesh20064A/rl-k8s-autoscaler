# PHASE 9H — Demo pipeline: fix traffic path, state, UI (verified live)

## goal
Phase 9F/9G hardened the demo run lifecycle (store reconciliation,
re-attach, STOP, RECOVER, live log, Live Details run selector), but the
latest PPO demo run (`demo_bursts_cpu_high`) showed 0 req/s on every
decision, "SCALE_UP_ONE | replicas 3→3", a huge JSON blob dumped into the
log at the end, STOP/selects stuck after COMPLETED, a static Live
Details disclaimer instead of a real Prometheus check, and Comparison
showing "HPA run: not available · PPO run: not available" for a workload
both arms had actually run. This phase root-causes and fixes all of
these, adds a traffic guard, retroactively invalidates every pre-existing
zero-traffic run, and verifies the fix for real against the live
cluster (two real demo runs, PPO and HPA, on a new short smoke profile).

## root cause 1 — 0 req/s on every decision
`scripts/run_demo_episode_v2.py` never instantiated ANY traffic
orchestrator. It logged `"Traffic started: ..."` as a purely cosmetic
string, then ran `eval_v2_episodes.py`'s per-arm episode loop directly —
which is **observe-only** (reads Prometheus/pod-health/the Deployment
every decision interval but never starts a generator process). The
working evaluation/training paths (`scripts/evaluate_v2_campaign.py`'s
`V2EvalTrafficOrchestrator`, `scripts/v2_traffic_orchestrator.
V2TrafficOrchestrator`) always call `ctx.orchestrator.start_episode_
traffic()` (the generic `scripts/traffic_orchestrator.TrafficOrchestrator`
machinery: V5 Docker relay route + `python -m loadgen.traffic_generator`
subprocess) *before* their episode function — the demo runner never did.
Verified directly: scanning the real store, every pre-existing demo run
(`demo_constant_cpu_normal`, `demo_bursts_cpu_high`, ...) has
`request_rate_rps` zero/absent in 100% of its samples (see "invalidated
runs" below) — systemic, not one-off.

**Fix** (`scripts/run_demo_episode_v2.py`): added `resolve_demo_profile_
metadata()` + `V2DemoTrafficOrchestrator(to.TrafficOrchestrator)` —
same architecture as `V2EvalTrafficOrchestrator`/`V2TrafficOrchestrator`:
every generic piece (route lifecycle, generator process lifecycle,
drain, provenance) is inherited unchanged; only `start_episode_traffic`
is overridden to resolve against `DEMO_PROFILE_ALLOWLIST`/
`DEMO_PROFILE_DIR`. `run_demo_episode_v2()` now calls `orchestrator.
start_episode_traffic()` before the episode loop and `stop_episode_
traffic()`/`drain_server_side_requests()`/`force_cleanup()` after —
reusing the already-tested generator/relay machinery, never a new
traffic path.

## root cause 2 — no guard against blind zero-traffic runs
Nothing confirmed traffic was actually arriving before letting the
episode run its full duration. **Fix**: `confirm_traffic_arriving()` —
polls the SAME `request_rate_rps` PromQL `live_backend.py` uses
(`sum(rate(demo_app_requests_total{namespace="autoscaler-app",
route="/work"}[1m]))`) and the generator's own liveness, for up to 90s
(5s poll interval, >= 0.5 req/s to confirm). On failure, raises
`ToolingFailureError("traffic not reaching the app: <reason>")` —
routed through the exact same cleanup/terminal-status path every other
tooling failure uses (status `error`, idle state restored, controller
lock released).

## root cause 3 — before→after replica log bug ("3→3")
`decision_record_v2.py`'s `current_replicas` field is itself populated
(inside a frozen training-contract helper, out of this phase's scope)
from `info["deployment_current_replicas_after"]` — the SAME post-step
value `desired_replicas` also comes from. Trusting it for "before"
always printed the post-action value twice. **Fix** (entirely on the
`run_demo_episode_v2.py` side, per the task's own guidance): `on_sample()`
tracks each decision's own real resulting replica count
(`EpisodeSample.desired_replicas`, a genuine post-action value) in a
closure, seeded with the episode's configured `initial_replicas`, and
passes the PREVIOUS decision's result in as `before_replicas` to
`_log_decision()`. Verified live: PPO smoke run printed
`Decision 1/4: HOLD | replicas 2→5` (real initial→real post-action), then
`5→5`, `5→5`, `5→5` for the HOLD decisions that followed — never a
duplicated value.

## root cause 4 — JSON blob dumped into the log
`main()`'s final `print(json.dumps(summary, default=str))` included the
FULL `samples` list (every `EpisodeSample`, including observation
vectors) — and this script's stdout is redirected verbatim into the
run's human-readable log file by `live_execution_adapter.py`. **Fix**:
strips `samples` from the printed summary, replaced with `n_samples`
(a count). The per-decision log lines (`_log_decision`) and the samples
JSONL file are unaffected — this was the only dump site.

## root cause 5 — Comparison showing "not available" for both arms
Two separate bugs, found while fixing the above:
1. Zero traffic (root cause 1) meant no run ever had real data to pair.
2. **Independent of (1)**: `run_demo_episode_v2.py`'s summary never
   populated `workload_hash`/`workload_seed`/`workload_duration_seconds` —
   `experiment_store._same_workload()` requires ALL of
   `profile_name`/`workload_hash`/`workload_seed`/`workload_duration_s`
   to be non-None and equal, so pairing could never succeed for ANY run,
   ever, even with real traffic. **Fix**: the summary now includes
   `workload_hash`/`workload_seed`/`workload_duration_seconds` from the
   same `ProfileMetadata` the traffic orchestrator resolves (one
   source of truth for the profile's identity).
3. A third, pre-existing, unrelated bug found during live verification:
   `demo_ui/views/comparison.py`'s overlay charts read `extract_series()`
   with field names `"replicas_desired"`/`"latency_p95_ms"` — neither is
   a real key in the sample schema (`desired_replicas`/`p95_latency_ms`),
   so the charts were always empty even for a correctly-paired run. Fixed
   in passing (2-line fix, same file already in scope for this phase).

Verified live: after the fix, Comparison shows `"Comparing workload:
Demo Smoke Bursts Cpu"` / `"HPA run: <id> · PPO run: <id>"` / `"HPA
result available · PPO result available"`, and both overlay charts carry
real non-empty series.

## root cause 6 — STOP/selects stuck after COMPLETED
`demo_ui/views/experiment.py`'s `_render_live_status_panel()` gated the
one-shot full-page `st.rerun()` (which re-derives `active_rec` at the
top of `render()` and unlocks the control strip) on `real_ctrl.
is_running`. But `real_ctrl.is_running` is recomputed from `active_rec_
now` (the SAME tick's reconciliation result) just a few lines earlier in
the SAME function call — so in the common case where `reconcile()` is the
one that first discovers a status-file transition to terminal (it runs
before `real_ctrl.status()` ever gets a chance to), `is_running` is
ALREADY `False` by the time the `st.rerun()` gate is checked, in that
same tick — the gate almost never fires, leaving the OUTER control strip
(rendered before this fragment, from a now-stale `active_rec`) stuck on
STOP/disabled-selects until some unrelated full-page rerun happens to
occur. **Fix**: replaced the `real_ctrl.is_running`-gated trigger with a
one-shot debounce keyed on `f"{run_id}:{status_label}"` in
`session_state` — fires exactly once per (run_id, terminal status)
transition regardless of which code path discovers it first, never loops.
Button/select locking itself was already correctly store-derived
(`controls_disabled = active_rec is not None`, Phase 9F/9G) — only the
rerun TRIGGER was the stale-flag bug.

## root cause 7 — static Live Details disclaimer
`demo_ui/views/live_details.py` printed a fixed caption ("Requires an
already-running, externally-managed Prometheus port-forward...") instead
of checking. **Fix**: `PrometheusMetricsProvider.check_prometheus_
reachable()` (new, calls the already-tested `PrometheusClient.check_
readiness()` against `/-/ready`) is called first; when reachable, a
one-line confirmation replaces the disclaimer; when not, an `st.warning`
names the EXACT command (`kubectl port-forward -n monitoring svc/
kube-prom-stack-kube-prome-prometheus 9090:9090`). Also added a
`Controller lock` cell to the metric strip (reads `controller_
exclusivity.read_lock()`, read-only).

## sec.3 — invalidating existing bad runs
Added `experiment_store.INVALID_NO_TRAFFIC` (`"invalid_no_traffic"`) —
TERMINAL (included in `TERMINAL_STATUSES`, so History/the run selector
still show it) but excluded from Comparison pairing (`latest_compatible_
pair()` already required `status == "completed"`, unchanged — this new
status simply never qualifies) and never the Live Details default
selection (`render_run_selector()` now skips it when picking a default,
still fully selectable). `is_zero_traffic_run()`/`invalidate_zero_
traffic_runs()` scan every terminal run's own samples (every
`request_rate_rps` zero/absent throughout == invalid; no samples at all
== not applicable, e.g. a refusal) — never deletes a record or any file.

Run against the real `persistent-data/demo_ui/runs.json`: **9 runs
invalidated** (`4d4edc8274e6`, `66675ce8db65`, `c09f2ed5db4d`,
`bb704507ecf3`, `a4503cb95533`, `92c6609f5684`, `79ade9f37acb`,
`c50d1625fd47`, `75d31d5a8c90` — every pre-existing `demo_constant_
cpu_normal`/`demo_bursts_cpu_high` completed/error run). Re-running is a
no-op (idempotent, verified); all 33 store records and their samples are
still present afterward.

## new smoke profile (sec.7)
`configs/traffic_profiles/v2_demo/demo_smoke_bursts_cpu.yaml` — 120s,
burst 8→40 req/s, CPU-focused, same schema as `demo_bursts_cpu_high.
yaml`. Allowlisted additively in `run_demo_episode_v2.py`'s own
`DEMO_SMOKE_TEST_PROFILE_NAMES` (never touches `scripts/generate_demo_
profiles.py`, outside this phase's file scope) — not reachable from the
UI's own Pattern/Resource/Load selects, only by passing the name
directly (used for the live verification below).

## real end-to-end verification (sec.7) — against the live cluster
Preflight confirmed: Prometheus ready, canonical idle state (only
`demo-app-hpa-v2`, 2 replicas, lock `none`). Both runs launched via
`demo_ui.live_execution_adapter.RealExperimentControllerV2.start()` —
the EXACT controller the Streamlit button uses — with `model_path` from
the real `demo_ui/config_v2.yaml` (session_047's final checkpoint).

**PPO run** (`3391262457b9`): traffic confirmed at 1.6 req/s within the
guard window; log shows `Decision 1/4: HOLD | replicas 2→5`, `2/4: HOLD |
5→5`, `3/4: HOLD | 5→5`, `4/4: HOLD | 5→5`; status `completed`.
Prometheus cross-check (independent range query over the run's real
window): peak 14.68 req/s, mean 5.46 req/s. p95 latency > 0 on decisions
1-3 (2358/465 ms visible; decision 4 reads 0 as traffic had already
ebbed to 0 at that sample). Cleanup restored `demo-app-hpa-v2` at 2
replicas, lock `none`.

**HPA run** (`8d9545091172`): traffic confirmed at 1.5 req/s; log shows
`Step 1/4: HPA desired 4, current 4`, `2/4: desired 6, current 6`
(scaled up under real CPU 195%/173% load), `3/4`/`4/4: desired 6, current
6` (CPU easing to 101%/30% as the burst ended); status `completed`.
Prometheus cross-check: peak 15.86 req/s, mean 7.11 req/s. p95 latency
3624/3194/2084/247 ms (> 0, trending down as traffic ebbed). Cleanup
restored `demo-app-hpa-v2` at 2 replicas, lock `none`.

Both runs' summaries carry the IDENTICAL `workload_hash`
(`f06160f575...`), `workload_seed` (424242), `workload_duration_seconds`
(120.0) — the same profile file, confirming the sec.5 pairing fix.

**Cluster state after both runs**: `kubectl get hpa -A` → only
`demo-app-hpa-v2`, 2/2 replicas; `kubectl get pods -A` (non-kube-system)
→ unchanged `demo-app` pods, restart counts still 2/2 each (identical to
the pre-session baseline — zero NEW restarts from either run); no
`rl-traffic-relay-*` Docker containers; no `loadgen`/`run_demo_episode_
v2.py`/`socat` processes left running.

**Pod restart investigation**: `kubectl describe pod` on both `demo-app`
pods shows their `restartCount=2` events predate this session entirely
(unchanged across the whole phase). During the live smoke runs' real
burst traffic, both pods logged NEW transient `Unhealthy` (liveness/
readiness probe timeout) warning events — expected under real CPU
pressure from a 40 req/s burst plus PPO/HPA scaling activity — but
`restartCount` never incremented; these did not escalate into actual
restarts.

**UI verification** (Streamlit `AppTest`, against the REAL store these
two runs just wrote to, not a mock): Experiment page shows `du_v2_start_
button` (no `du_v2_stop_button`), all 4 selects (`du_v2_arm_widget`/
`_pattern_widget`/`_resource_widget`/`_load_widget`) `disabled=False`.
Live Details run selector lists both smoke runs with the correct labels
(`"HPA · Demo Smoke Bursts Cpu · ... · completed"` /
`"PPO · Demo Smoke Bursts Cpu · ... · completed"`) and the correct panel
per arm (HPA targets / PPO decision explanation). Comparison correctly
pairs them (`"HPA result available · PPO result available"`), default
selection is the latest genuinely-valid completed run (never an
`invalid_no_traffic` one).

One iteration note: the first PPO attempt (`f889937137f7`) failed with
`"failed to load PPO model at .../models/ppo_v2/ppo_model_v2_min2.zip"` —
a driver-script mistake (omitted `model_path`, falling back to a
nonexistent default), not a bug in the fix; traffic was still correctly
confirmed (1.4 req/s) before the model-load failure, and cleanup
correctly restored idle state. Left as a real `error` record (not
invalidated — it has no samples, so `is_zero_traffic_run()` correctly
does not apply).

## tests
- `tests/unit/test_run_demo_episode_v2.py`: added an autouse fixture
  patching `run_demo_episode_v2`'s own default arguments (not the
  module-level factory name — a bound default is captured at def-time) to
  a fully offline fake orchestrator/Prometheus reading for every
  pre-existing test (none of them should touch Docker/network just
  because they call the function); new tests for the traffic guard
  (fails closed on zero confirmed rate, fails closed on generator exit,
  confirms + stops + drains + force-cleans on success), before→after
  replica correctness across TWO genuinely different replica counts, and
  no-sample-dump on `main()`'s stdout.
- `demo_ui/tests/test_experiment_store.py`: `is_zero_traffic_run`/
  `invalidate_zero_traffic_runs` (marks only zero-traffic runs, never a
  running record, idempotent, never deletes), `invalid_no_traffic` is
  terminal but never a pairing candidate.
- `demo_ui/tests/test_live_details_v2.py`: the run selector never
  defaults to an `invalid_no_traffic` run (still selectable).
- `demo_ui/tests/test_live_details_fresh_state.py`: the old static
  disclaimer text never appears; the actionable port-forward command
  shows when unreachable; no warning when reachable.
- `demo_ui/tests/test_live_providers.py`: `check_prometheus_reachable()`
  true/false cases.
- `demo_ui/tests/test_comparison_live_pairing.py` (new): overlay charts
  use the real field names (source guard), `extract_series()` returns
  real data for a completed run, `_latest_current_system_run()` skips
  `invalid_no_traffic`.

`/tmp/rl-k8s-demo-ui-venv/bin/python -m pytest demo_ui/tests -q` → 305
passed. `.venv/bin/python -m pytest tests/unit -q` → 1965 passed, 2
skipped (pre-existing). `ruff check` on every changed/new file → clean.

## files changed
`scripts/run_demo_episode_v2.py`, `demo_ui/experiment_store.py`,
`demo_ui/live_providers.py`, `demo_ui/views/experiment.py`,
`demo_ui/views/live_details.py`, `demo_ui/views/live_details_v2.py`,
`demo_ui/views/comparison.py`,
`configs/traffic_profiles/v2_demo/demo_smoke_bursts_cpu.yaml` (new),
plus their tests (`tests/unit/test_run_demo_episode_v2.py`,
`demo_ui/tests/test_experiment_store.py`, `test_live_details_v2.py`,
`test_live_details_fresh_state.py`, `test_live_providers.py`,
`test_comparison_live_pairing.py` (new)).
