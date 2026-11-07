# Part 14 — Scripts, Docs, and Open Items

FROM_CODE. Full script-by-script table in `raw/` (agent output); summarized
here.

## scripts/ classification

**V1**: `evaluate_ppo_vs_hpa.py`, `train_ppo_live.py`,
`train_ppo_live_session.py`, `run_dynamics_calibration_slot.py`,
`run_hpa260_v5_revalidation_trial.py`, `run_bridge_trial.py`,
`materialize_rl_final_holdout_workload.py`, `analyze_final_evaluation.py`,
`generate_index40_report_package.py`.

**V2**: `evaluate_ppo_vs_hpa_v2.py`, `train_ppo_final_v2.py`,
`train_ppo_live_v2.py`, `v2_pilot_safety.py`, `v2_traffic_orchestrator.py`,
`v2_training_instrumentation.py`.

**Demo**: `run_demo_episode.py` (isolated DEMO-ONLY execution adapter for
the Streamlit Experiment page; explicitly NOT part of the scientific
pipeline per its own docstring).

**Shared-utility**: `traffic_orchestrator.py`, `rl_live_common.py`,
`dynamics_calibration_lib.py`, `generate_dashboard_configmap.py`,
`integrate_replica_trajectory.py`, `per_pod_counter_collector.py`,
`pod_readiness_collector.py`, `smoke_test_ppo_sb3_compat.py`, plus
`loadgen/traffic_generator.py` and `collectors/{metrics_collector,
hpa_status_collector}.py`.

**Historical** (Phase 10-15 calibration/tuning, frozen artifacts of prior
investigation phases, not part of ongoing V1/V2 work): `aggregate_hpa_
tuning.py`, `compile_stage1_calibration.py`, `generate_dynamics_calibration_
schedules.py`, `generate_global_campaign_sequence.py`, `generate_tuning_
schedule.py`, `hpa_phase_entry_snapshot.py`, `metrics_api_cpu_sampler.py`,
`summarize_calibration_level.py`.

Shell scripts (`.sh`): `bootstrap.sh` (creates the venv — needed now that
`.venv` is broken), `start_minikube.sh` ("never auto-deletes an existing
cluster" — conservative for an 8GB Mac), `deploy_app.sh`, `install_monitoring.sh`,
`configure_metrics_server.sh` (sets the Minikube addon's
`--metric-resolution` — this is the mechanism that SHOULD be re-run to
restore the lost 15s setting, per Part 04's finding, though not executed
this read-only phase), `health_check.sh`, `cleanup_runtime_only.sh`,
`backup_results.sh`, plus V1/historical experiment-runner scripts.

## docs/ — CORR index

`docs/experimental_design_corrections.md`: **48 CORR entries** (CORR-1
through CORR-48), spanning Phase 6A through Phase 15B.6D — the full project
correction history. Full one-line list captured in `raw/` (agent output);
most relevant to current open questions: CORR-1 (Minikube resource-
accounting mismatch — re-confirmed live this session, Part 03), CORR-21
(`kubectl port-forward` doesn't load-balance — directly explains the R4C/R4D
tunnel ambiguity), CORR-22 (duplicate Prometheus scraping eliminated),
CORR-25 (replica-trajectory integration uses current, not Ready, replicas),
CORR-26 (260% HPA target frozen), CORR-27 (RL MDP contract V2: CPU-clip,
HPA-coupling, gating, held-out-leakage corrections), CORR-28 (RL dependency
compatibility freeze — the SB3/gymnasium/torch pin rationale), CORR-30
(reward-validity decomposition + episode-reset control age).

`docs/rl_mdp_spec.md`: "RL Autoscaler MDP Contract — v2 (FROZEN)", 25
numbered sections — the project's single authoritative, version-frozen MDP
contract (observation/action/reward/episode/PPO-compatibility definitions).
Any change to the observation vector, action map, or reward formula
requires a version bump and an explicit CORR entry.

Other docs: 8 versioned `rl_dynamics_calibration_protocol*.md` files
(iterative post-CORR-37 protocol revisions), `viva_results_summary.md`
(4,782B — name suggests a thesis/viva defense results summary, not
examined in depth this pass).

## TODO/FIXME/XXX/HACK/provisional grep (repo-wide)

**No literal TODO/FIXME/XXX/HACK/NOT_IMPLEMENTED found** except 2 genuine
TODOs in `demo_ui/providers.py` (see Part 13). All other "provisional"
hits are deliberate SCIENTIFIC-STATUS markers, not implementation stubs —
concentrated entirely around `safe_action.near_tie_epsilon=0.05`
(`configs/rl_v2.yaml`, `action_selection.py`, `training_contract_v2.py`,
`scripts/evaluate_ppo_vs_hpa_v2.py`, and multiple tests that explicitly
assert this provisional status is itself a tested invariant, not an
oversight) plus one resolved-provisional note in
`k8s/hpa/hpa_v2_cpu_memory.yaml` (the memory target was provisional before
being live-calibrated to 120%). No `FIXME`/`XXX`/`HACK`/"not yet
implemented" found anywhere in the searched tree.
