# Part 15 — Discrepancies (no fixes, report only)

1. **Live Deployment replicas (2) vs. manifest (1)** — `k8s/app/
   deployment.yaml` declares `replicas: 1`; live `spec.replicas: 2`.
   Known since Phase 4B-R4C (2026-10-12), still unresolved 4 days later.
2. **Live HPA absent vs. canonical `demo-app-hpa` expected** — suspended
   for the R4C test, never restored. Same age as #1.
3. **Metrics Server resolution**: Phase 4B-R4D set it to 15s via a live
   `kubectl patch` with no persistence mechanism; a subsequent Minikube
   restart (observed this session) reverted it to the addon default 60s.
   **Newly confirmed this session**: the 15s value is not durable across
   any Minikube restart unless `scripts/configure_metrics_server.sh` (or
   equivalent) is re-run every time.
4. **`rl-autoscaler-demo-app:local` image ID changed** since Phase 4B-R3
   (`da89...` → `75911ce139a5`) — the image was rebuilt at some point; the
   in-pod source hash could not be re-verified against current repo source
   this session (API server unreachable).
5. **`.venv` is a symlink into the now-deleted old repository**
   (`~/Capstone/rl-k8s-autoscaler/.venv`) — confirmed broken; the old repo
   directory no longer exists at all on this machine. No working
   project-dedicated Python environment currently exists anywhere.
6. **`tests/unit/test_traffic_generator.py` hardcodes 3 absolute paths**
   to the now-deleted old repo (lines 1285, 1324, 1415) — these specific
   tests will fail or silently misresolve.
7. **Installed global-Python package versions diverge from every pinned
   requirements file** (numpy, pandas, torch, uvicorn, PyYAML, requests all
   at different versions than `requirements/*.txt` specify), and the
   scientific stack (stable-baselines3, gymnasium, kubernetes client,
   httpx, fastapi, prometheus-client, pytest, ruff) is **entirely absent**
   from every reachable interpreter on this machine.
8. **`configs/app.yaml`/`configs/hpa.yaml` are manually-mirrored "design
   source" files** with no automated Python loader found referencing them
   by path — their values happen to match the corresponding `k8s/`
   manifests, maintained by hand, not by any sync script.
9. **`configs/rl_dynamics.yaml` points to a subdirectory
   (`configs/traffic_profiles/rl_dynamics_calibration/`) that does not
   exist** in the current file tree.
10. **`k8s/hpa/hpa_v2_cpu_memory.yaml` ("demo-app-hpa-v2") is never
    actually restored or applied by any V2 training/evaluation script** —
    only the frozen V1 `demo-app-hpa` is ever live-restored, despite the
    V2 filename/naming suggesting otherwise. (Part 11.)
11. **`scripts/evaluate_ppo_vs_hpa_v2.py` and `safe_ppo_controller.py`
    actually construct the V1 `KubernetesAutoscalerEnv`/`configs/rl.yaml`**,
    not the V2 equivalents — a confirmed defect from Phase 4B-R4F, never
    executed, never fixed. (Part 09.)
12. **Pilot telemetry schema lacks `normalized_observation`**; final-
    campaign telemetry has it — a real, expected schema difference since
    the instrumentation was added only in Phase 4A, after all pilot
    sessions completed. Not a bug, but a genuine cross-dataset
    incompatibility if the two were ever compared feature-for-feature.
13. **Two dead config keys in `configs/reward_v2.yaml`** (`min_replicas`/
    `max_replicas` and the three `per_*_cost` multipliers) are documented
    but never actually read by the loader. (Part 09.)
14. **Pilot sessions 000/002 lack metadata/model/training_metrics files**
    that sessions 001/003 have — consistent with those two having failed
    before a checkpoint write completed (not re-derived in depth this
    session; consistent with this conversation's own prior history).
15. **Node allocatable (4 CPU/4GB) vs. actual cgroup cap (2 CPU/3GB)** —
    the CORR-1 mismatch, now independently re-confirmed LIVE this session
    with a direct demonstration of its consequence (sustained apiserver
    unreachability under real CPU contention, absent any training/traffic
    workload). (Part 03/04.)
16. **Prometheus retention (6h/2GB) explains why 21-day-old R4C evidence
    could never have survived** regardless of the cluster being stopped —
    this was previously left as an open question in the R4C audit and is
    now resolved by this session's config inventory (Part 08).
