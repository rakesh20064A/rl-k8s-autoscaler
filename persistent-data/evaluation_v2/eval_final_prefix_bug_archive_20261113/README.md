# Archived: pre-fix eval_final attempts (V2 Phase 5D)

Two implementation bugs in `scripts/evaluate_v2_campaign.py`'s first-ever
live execution (never tested against a real cluster before this phase):

1. `V2EvalTrafficOrchestrator` was constructed without a real `http_get`
   callable (defaulted to `None`), crashing the whole process inside
   `verify_route_alive()`.
2. `restore_canonical_idle_state()` called
   `KubectlRunner.wait_for_replica_state(2, timeout=...)` with the wrong
   keyword (`timeout` instead of `timeout_seconds`), crashing cleanup.
3. (Found after fixing 1+2): the `ppo_v2` arm's `acquire_controller(CONTROLLER_PPO, ...)`
   failed closed because `demo-app-hpa-v2` was still applied (left behind
   by the prior run's own canonical-idle-state cleanup) -- the ppo_v2 arm
   wrapper now deletes it first, exactly as `run_final_training_campaign()`
   does for live training.

None of these 3 attempts (run_index 0, `eval_cpu_step__repeat0__ppo_v2`,
attempts 1-2 here, plus one earlier fully-crashed attempt with no ledger
row at all) ever actually started real traffic or a real PPO episode --
zero scientific content, pure software-bug failures in brand-new code.
Archived (not deleted) per this project's own "never delete evidence"
discipline, then cleared from the active ledger so run_index 0 restarts
cleanly under the fixed code.
Attempt 3 (after the HPA-delete fix) failed both tries with 'generator failed to start', but the EXACT SAME code path (build_live_context + start_episode_traffic) succeeded when run manually/standalone immediately afterward -- consistent with a transient Docker/relay-container resource hiccup from rapid attempt churn, not a reproducible code bug. Archived, not deleted.
