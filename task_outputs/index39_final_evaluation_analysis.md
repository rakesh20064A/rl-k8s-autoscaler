# Index39 — Final Evaluation Analysis

OFFLINE analysis only. No live execution occurred to produce this document.

## 1. Training completion

Frozen at `session_019`: 5120 cumulative timesteps, 20 PPO updates, training seed 42.
`session_020` does not exist (verified).

## 2. Final evaluation protocol

Formal sequence executed: PPO1 → HPA1 → PPO2 → HPA2 → PPO3 → HPA3, replaying the
identical hash-pinned final holdout workload (`308f4f90ea3cea97f1927cec97efeaa652c4424e9fe5bcea421358f3425e3409`, seed 777, 1500s)
for every episode, per `phase_16u_direct_ppo_protocol.yaml` Section 10 /
`phase_16u_training_and_evaluation_workload_split.json`.

## 3. Episode outcome classification

| Episode | Controller | Repeat | Status | Steps (obs/nom) | Mean Replicas | Max Replicas | Mean P95 (ms) | SLA Viol. Rate | Resource Proxy |
|---|---|---|---|---|---|---|---|---|---|
| HPA1 | HPA | 1 | completed_full_duration | 50/50 | 5.06 | 7 | 1592.0 | 0.720 | 0.609 |
| HPA2 | HPA | 2 | completed_full_duration | 50/50 | 5.35 | 8 | 986.7 | 0.540 | 0.649 |
| HPA3 | HPA | 3 | completed_full_duration | 50/50 | 5.29 | 8 | 689.4 | 0.520 | 0.626 |
| PPO3 | PPO | 3 | completed_full_duration | 50/50 | 1.20 | 4 | 2298.5 | 0.920 | 0.034 |
| PPO1 | PPO | 1 | terminated_safety | 23/50 | 0.75 | 1 | 1925.0 | 0.826 | 0.000 |
| PPO2 | PPO | 2 | terminated_safety | 20/50 | 0.67 | 1 | 2024.3 | 0.700 | 0.000 |

## 4. Frozen Section-10 metrics

Computed exactly per the frozen definitions (`sla_violation_rate`, `error_rate_mean`,
`resource_proxy_mean`, `reaction_time_seconds`, `action_churn_count`) — see
`task_outputs/index39_final_evaluation_metrics.json` for full per-episode values,
including partial-duration PPO1/PPO2 (reported for completeness, never mixed into
full-duration aggregates).

## 5. Valid full-duration comparison table (HPA1, HPA2, HPA3, PPO3 only)

| HPA1 | HPA | 1 | completed_full_duration | 50/50 | 5.06 | 7 | 1592.0 | 0.720 | 0.609 |
| HPA2 | HPA | 2 | completed_full_duration | 50/50 | 5.35 | 8 | 986.7 | 0.540 | 0.649 |
| HPA3 | HPA | 3 | completed_full_duration | 50/50 | 5.29 | 8 | 689.4 | 0.520 | 0.626 |
| PPO3 | PPO | 3 | completed_full_duration | 50/50 | 1.20 | 4 | 2298.5 | 0.920 | 0.034 |

## 6. Safety outcomes

| PPO1 | PPO | 1 | terminated_safety | 23/50 | 0.75 | 1 | 1925.0 | 0.826 | 0.000 |
| PPO2 | PPO | 2 | terminated_safety | 20/50 | 0.67 | 1 | 2024.3 | 0.700 | 0.000 |

Both PPO1 and PPO2 terminated via the frozen `consecutive_invalid_steps_limit_reached`
rule (4 consecutive decision steps with Prometheus target unreachable) — a legitimate,
pre-registered safety outcome, not an implementation defect.

## 7. Infrastructure failure provenance

HPA repeat2, physical attempt1: status=`tooling_failed`. Traffic generator completed its
full 1500s run
(`traffic_completed=True`), but metrics finalization failed
(`metrics_complete=False`) during post-workload Prometheus drain,
concurrent with observed `NodeNotReady` events. Preserved at
`persistent-data/evaluation/final_session019/episode_04_hpa_repeat2_physical_attempt1_FAILED.json`.
**Excluded from scientific aggregation** — not a scientific HPA result.
HPA repeat2 physical attempt2 (after infrastructure recovery) completed the full
duration and is the scientific HPA2 result used throughout this analysis.

## 8. Descriptive resource-vs-latency findings

Across the three valid full-duration HPA repeats, mean replicas ranged
5.06–5.35
(peaking at 8 replicas), with mean P95 latency
689–1592 ms.

The single valid PPO full-duration repeat (PPO3) used substantially fewer replicas
(mean 1.20, max 4) than any HPA repeat, but recorded a
higher mean P95 latency (2298 ms) than all three HPA repeats. This is a
**single-repeat observation**, not a statistically supported trend.

PPO's two safety-terminated repeats (PPO1, PPO2) are themselves evidence of a
robustness limitation under this specific held-out workload: the policy (or the
surrounding live infrastructure under PPO's control pattern) encountered sustained
telemetry/target unavailability severe enough to trigger the frozen safety
termination rule in 2 of 3 attempts, while HPA's control pattern did not trigger
the equivalent failure mode in any of its 3 valid attempts (one HPA attempt did
fail, but from infrastructure/tooling causes unrelated to HPA's own control
actions, not a safety-rule termination).

## 9. Formal decision-rule status

**NOT APPLICABLE.** The frozen Section 10 decision rule requires n=3 valid
full-duration repeats per controller. HPA achieved 3/3; PPO achieved 1/3.

## 10. Limitations

- PPO's valid full-duration sample size (n=1) cannot support the frozen
  decision rule, any inferential statistic, or a directional conclusion.
- PPO1/PPO2's partial-duration data (24/50 and 20/50 decision steps
  respectively) are reported for transparency but are not comparable to a
  full 1500s exposure and are excluded from the primary comparison.
- HPA2 physical attempt1 is excluded entirely (no scientific result).
- Whether PPO's two safety terminations reflect the learned policy's
  behavior, the live infrastructure's independent instability during this
  evaluation window, or an interaction between the two cannot be
  disentangled from this evidence alone.

## 11. Final scientifically defensible conclusion

See `task_outputs/index39_final_scientific_conclusion.md`.
