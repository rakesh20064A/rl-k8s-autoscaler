# V2 Phase 4B-KS — Complete Project Knowledge & Current-Instance Snapshot

Performed: 2026-11-07, read-only. All facts labeled VERIFIED (observed
directly this session), FROM_CODE (read from source, not runtime-verified),
or UNKNOWN (could not be established — reason stated) throughout the files
below.

## Files in this snapshot

| File | Contents |
|---|---|
| 01_host.md | macOS/chip/RAM/disk/power/pyenv |
| 02_docker.md | Docker Desktop version, resources, containers, images |
| 03_minikube_k8s.md | Minikube profile/config, CORR-1 mismatch (live-confirmed) |
| 04_live_cluster.md | Full live cluster state; API server went unreachable mid-session |
| 05_git_repo.md | Git state, full diff descriptions, prior audit report summaries |
| 06_python_env.md | **`.venv` is a broken symlink into the deleted old repo** |
| 07_application.md | app/src/ endpoints, workload CPU/memory mechanics, metrics |
| 08_configs_and_manifests.md | Every configs/ key + every k8s/ manifest hash |
| 09_rl_contract.md | Full observation/action/reward/PPO/tie-break contract, V1 vs V2 |
| 10_traffic_routing_monitoring.md | Traffic generator, V5 route, Prometheus URL consistency |
| 11_hpa_controller_exclusivity.md | Which HPA is actually restored/checked by each script |
| 12_training_checkpoints_results.md | Pilot + final campaign state, resume compatibility |
| 13_demo_ui.md | UI structure, V1-only confirmed, uncommitted UI files |
| 14_scripts_docs_todos.md | Every script classified, 48 CORR entries, TODO grep |
| 15_discrepancies.md | 16 discrepancies, no fixes |
| 16_tests.md | **BLOCKED — no usable test environment exists anywhere** |
| 17_explicit_questions.md | All 15 required questions answered with evidence |
| raw/ | Full raw command output backing every summary above |

## Top 10 unknowns / risks (ranked)

1. **No working Python environment exists anywhere on this machine** —
   `.venv` is a symlink into the now-deleted old repo; pytest/ruff/the
   entire scientific stack are absent from every reachable interpreter.
   `scripts/bootstrap.sh` must be re-run before any test or training work.
2. **Live cluster remains non-canonical** (2 replicas, HPA absent) —
   unresolved since Phase 4B-R4C (2026-10-12), still true 4 days later.
3. **Metrics Server resolution reverted to 60s** — the Phase 4B-R4D 15s
   patch was never durable and was lost on a subsequent Minikube restart.
4. **Live, reproduced CORR-1 consequence**: the `minikube` container
   sustained 229-483% CPU against its 200% cgroup cap for several minutes
   this session, with the API server becoming unresponsive, absent any
   training/traffic workload — a structural host-resource risk, not just a
   historical finding.
5. **`rl-autoscaler-demo-app:local` image ID changed** since Phase 4B-R3 —
   unverified whether current repo source still matches the running image.
6. **R4C's central question remains `INCONCLUSIVE`**: does the V5/ad-hoc
   traffic path reliably reach both pods under 2-replica load? Not
   resolved by this snapshot (out of scope — read-only).
7. **Confirmed defect**: `evaluate_ppo_vs_hpa_v2.py`/`safe_ppo_controller.py`
   actually run the V1 environment, not V2's — never executed, never fixed.
8. **`demo-app-hpa-v2` has never been live-restored or tested** by any
   script — the Option-A fairness comparison has no runnable path today.
9. **epsilon=0.05 would force HOLD on 91% of real trained-policy
   decisions** (empirically computed in Phase 4B-R4F) — must not be
   adopted before post-training recalibration.
10. **`tests/unit/test_traffic_generator.py` hardcodes 3 paths to the
    deleted old repo** — will fail once a test environment exists again.

## Final report

```text
snapshot_dir = persistent-data/knowledge_snapshot_20261107T054233Z/
files_written = 00_INDEX.md, 01_host.md, 02_docker.md, 03_minikube_k8s.md,
  04_live_cluster.md, 05_git_repo.md, 06_python_env.md, 07_application.md,
  08_configs_and_manifests.md, 09_rl_contract.md,
  10_traffic_routing_monitoring.md, 11_hpa_controller_exclusivity.md,
  12_training_checkpoints_results.md, 13_demo_ui.md,
  14_scripts_docs_todos.md, 15_discrepancies.md, 16_tests.md,
  17_explicit_questions.md, raw/*.txt (6 files)
live_cluster_reachable = PARTIAL (reachable for ~first 2-3 minutes of this
  session, then sustained unreachable for 5+ minutes under real CPU
  contention on the Minikube container — not fixed, not caused by this
  session)
unknown_count = 10 (five-metrics-API-sample cadence; in-pod image hash
  re-verification; monitoring-namespace/helm/Prometheus live detail;
  per-pod request-count live re-verification; Docker Desktop Swap/Rosetta
  settings; pytest/ruff pass-fail counts; a few narrower items noted
  inline in their respective part files)
top_10_unknowns_or_risks = see above
r4f_report_exists = YES
code_or_config_modified = NO
kubernetes_resources_modified = NO
docker_or_minikube_state_modified = NO
training_or_traffic_started = NO
commit_or_push = NO
status = SNAPSHOT_COMPLETE
```

**Final verdict: `KNOWLEDGE_SNAPSHOT_COMPLETE_WITH_UNKNOWNS`**
(Part 04/16 items blocked by a genuinely unreachable API server and a
genuinely absent Python environment — both are themselves significant,
honestly-reported findings, not gaps in effort.)
