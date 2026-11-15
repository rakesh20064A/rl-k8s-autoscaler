| profile | arm | n_repeats | sla_violation_rate_1000ms | replica_minutes |
|---|---|---|---|---|
| eval_cpu_step | HPA CPU+Memory (resource baseline) | 3 | 0.0000 +/- 0.0000 | 60.5000 +/- 0.8660 |
| eval_cpu_step | HPA CPU-only (Kubernetes default) | 3 | 0.0000 +/- 0.0000 | 59.3333 +/- 0.2887 |
| eval_cpu_step | PPO (multi-metric, 11 signals) | 3 | 0.0083 +/- 0.0144 | 72.6667 +/- 0.2887 |
| eval_memory_ramp | HPA CPU+Memory (resource baseline) | 3 | 0.0000 +/- 0.0000 | 75.1667 +/- 1.6073 |
| eval_memory_ramp | HPA CPU-only (Kubernetes default) | 3 | 0.0000 +/- 0.0000 | 40.0000 +/- 0.0000 |
| eval_memory_ramp | PPO (multi-metric, 11 signals) | 3 | 0.0000 +/- 0.0000 | 56.1667 +/- 0.2887 |
| eval_mixed_burst | HPA CPU+Memory (resource baseline) | 3 | 0.0000 +/- 0.0000 | 74.6667 +/- 5.1316 |
| eval_mixed_burst | HPA CPU-only (Kubernetes default) | 3 | 0.0000 +/- 0.0000 | 40.0000 +/- 0.0000 |
| eval_mixed_burst | PPO (multi-metric, 11 signals) | 3 | 0.0000 +/- 0.0000 | 58.3333 +/- 2.3094 |
