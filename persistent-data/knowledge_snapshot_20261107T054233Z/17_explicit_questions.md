# Part 17 — Explicit Questions (with evidence)

**1. Current git HEAD/branch; is any V2 work committed? Exactly what is
uncommitted?**
HEAD `1d74164263969fcaf5ddf93fb026fa6d783034b0`, branch `main`. **NO V2
work is committed** — every V2 source/config/script/test file and all V2
training data is untracked. 16 modified V1-area files (listed in full in
05_git_repo.md) plus ~68 untracked files/dirs. VERIFIED.

**2. Exact V2 reward weights and `rl_v2.yaml` values.**
See 08_configs_and_manifests.md / 09_rl_contract.md full tables. Weights:
w_reliability=6.0, w_sla=4.0, w_pressure=2.0, w_waste=0.3, w_action=0.1,
w_reversal=0.5, w_boundary=0.2. `rl_v2.yaml` additive block:
`safe_action.near_tie_epsilon=0.05` (provisional_unvalidated),
`controller_generation=v2`, `workload_contract=multi_resource_v2`. FROM_CODE.

**3. V1 and V2 observation normalisation constants.**
Identical between V1 and V2 (V2 reuses V1's `observation.py` unchanged):
cpu [0,5], memory [0,4], in_progress [0,20], request_rate [0,100],
p95_latency [0,2000ms], EWMA span=3 steps, seconds_since_scale cap=300s.
See 09_rl_contract.md full table. FROM_CODE.

**4. Was Phase 4A instrumentation (per-update SB3 metrics,
normalized_observation) actually implemented and does data exist?**
**YES, confirmed directly this session**: `persistent-data/rl_training_v2/
final/ppo_update_metrics.jsonl` exists with 4 real rows (approx_kl,
clip_fraction, entropy_loss, explained_variance, learning_rate, loss,
policy_gradient_loss, value_loss — all populated with real numeric
values); `final/sessions/session_002/step_telemetry.jsonl`'s rows DO
contain a populated `normalized_observation` field (11-element real
vector). VERIFIED.

**5. How memory is allocated and released in `/work` memory mode.**
Anonymous `mmap.mmap(-1, num_bytes, MAP_PRIVATE|MAP_ANONYMOUS)`, one-byte
touch per 4096-byte page, `time.sleep(hold_ms/1000)`, explicit
`mapping.close()` (→`munmap()`) in a `finally` block — NOT a bytearray
(a documented correction from an earlier approach that leaked RSS under
concurrent load via glibc arena retention). FROM_CODE,
`app/src/workload.py`.

**6. Is the safe near-tie selector wired into any live inference path?**
YES, into `scripts/evaluate_ppo_vs_hpa_v2.py` (via `safe_ppo_controller.py`
→ `safe_inference.py` → `action_selection.py`) — but that script has
**never been executed** (`--execute` never invoked), and it has a confirmed
defect: it actually steps the V1 environment, not V2's. Training NEVER
uses the selector (confirmed by grep + a dedicated structural test).
FROM_CODE.

**7. Which HPA does each V2 script restore and check?**
**`demo-app-hpa` (V1 CPU-only) in every single case** —
`LiveKubernetesBackend`'s default `hpa_manifest_path`, and every V2
preflight check, hardcode this name. `demo-app-hpa-v2` has never been
restored/applied live by any script. FROM_CODE, Part 11.

**8. Current Metrics Server `--metric-resolution` and observed cadence.**
**60s** (VERIFIED, live args captured before API-server unreachability) —
the Phase 4B-R4D 15s patch did not survive a subsequent Minikube restart.
Cadence re-verification (5 timestamped samples): **UNKNOWN** (API server
became unreachable before this could be attempted this session).

**9. Docker Desktop CPU/memory/swap/disk allocation and Minikube
CPU/memory.**
Docker Desktop VM: 4 CPUs, 4096MiB memory, 65536MiB disk (Swap/Rosetta
keys not found in this version's settings schema — UNKNOWN). Minikube
container (inside that VM): 2 CPUs (`NanoCpus=2000000000`), 3072MiB
(`Memory=3221225472`) — matches the `minikube` profile's own declared
config exactly. VERIFIED.

**10. Current Deployment replicas, HPA presence, lock state, pod
restarts.**
replicas=2 (not canonical 1), HPA absent, controller lock=`none`, pods
`demo-app-6579f6d9b-96fjb` (6 restarts) / `-jhdpm` (9 restarts), both with
a restart timestamped at the same moment as the rest of the cluster's pods
(consistent with a recent Minikube stop/start, not new application
instability). VERIFIED (captured before API-server unreachability).

**11. Does Prometheus expose per-pod request counts for demo-app?**
In principle yes (`demo_app_requests_total` carries no pod label itself,
but Prometheus's own scrape relabeling attaches the `pod` label
automatically to every target-scraped series, confirmed live-working for
CPU in Phase 4B-R4E) — **NOT re-verified live this session** (API server
unreachable). No per-REQUEST attribution exists anywhere regardless (Part
10). UNKNOWN (live) / FROM_CODE (capability exists in principle).

**12. Uvicorn worker/thread count and whether CPU work can starve
`/health`.**
1 worker (no `--workers` flag in the Dockerfile CMD), single event loop.
`/health` is deliberately `async def` (runs on the event loop, never
requests an AnyIO worker-thread-pool token) specifically because `/work`'s
sync CPU-bound handler can exhaust that shared 40-token pool — this
mitigation is already applied, and was found (Phase 4B-R2/R3) to be
necessary but NOT sufficient alone under sustained heavy CPU load (GIL
contention / Linux CFS throttling of the whole container can still delay
the event loop itself). FROM_CODE + prior live evidence.

**13. Does the R4F report exist?**
YES — `persistent-data/traffic/v2_phase4br4f_multimetric_controller_
optimization_audit.md` (45,672 bytes, Nov 3 13:45). VERIFIED, `ls -la`.

**14. Does anything in the clean repo depend on the old repo path?**
YES — **`.venv` itself is a symlink into the old repo** (now broken, old
repo deleted), and `tests/unit/test_traffic_generator.py` hardcodes 3
absolute old-repo paths. No other tracked source/config file references
it. VERIFIED (Part 06/15).

**15. Is the 768-step V2 checkpoint compatible with the current code
contracts?**
YES — `verify_resume_provenance()`'s governing-hash check (covering
`environment_v2.py`, `reward_v2.py`, `observation.py`, `rl_v2.yaml`,
`reward_v2.yaml`) would still pass, since none of these 5 files show as
modified in the current git diff since `session_002` was created. A
resume remains scientifically valid from `session_002`, subject to the
live-cluster and root-cause blockers described in Part 12/15. FROM_CODE +
VERIFIED (git diff).
