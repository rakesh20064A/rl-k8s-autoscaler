| profile | arm | n_repeats | exploratory | sla_violation_rate_1000ms | replica_minutes | mean_p95_latency_ms | median_p95_latency_ms | error_rate | max_per_pod_memory_mib | scaling_actions_count | reversals_count | reaction_time_s | pod_restarts_count | invalid_samples_count | sla_violation_rate_300ms_reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stress_cpu_burst | HPA CPU+Memory (resource baseline) | 1 | yes | 0.5263 | 123.0000 | 1820.3034 | 1371.6415 | 0.0000 | 43.0664 | 0.0000 | 0.0000 | n/a | 0.0000 | 2.0000 | 0.6053 |
| stress_cpu_burst | HPA CPU-only (Kubernetes default) | 1 | yes | 0.5250 | 130.0000 | 1703.0794 | 2304.0822 | 0.0000 | 43.9414 | 0.0000 | 0.0000 | n/a | 0.0000 | 0.0000 | 0.6250 |
| stress_cpu_burst | PPO (multi-metric, 11 signals) | 1 | yes | 0.5000 | 84.0000 | 1567.6466 | 1263.5172 | 0.0000 | 44.6680 | 3.0000 | 0.0000 | n/a | 0.0000 | 0.0000 | 0.7750 |
| stress_memory_ramp | HPA CPU+Memory (resource baseline) | 1 | yes | 0.0000 | 78.0000 | 975.0000 | 975.0000 | 0.0000 | 119.4609 | 0.0000 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.0000 |
| stress_memory_ramp | HPA CPU-only (Kubernetes default) | 1 | yes | 0.0000 | 40.0000 | 975.0000 | 975.0000 | 0.0000 | 152.3359 | 0.0000 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.0000 |
| stress_memory_ramp | PPO (multi-metric, 11 signals) | 1 | yes | 0.0000 | 56.0000 | 950.6250 | 975.0000 | 0.0000 | 120.0742 | 1.0000 | 0.0000 | n/a | 0.0000 | 0.0000 | 0.9750 |
| stress_mixed_burst | HPA CPU+Memory (resource baseline) | 1 | yes | 0.0000 | 76.0000 | 980.3134 | 978.8153 | 0.0000 | 88.1680 | 0.0000 | 0.0000 | n/a | 0.0000 | 1.0000 | 1.0000 |
| stress_mixed_burst | HPA CPU-only (Kubernetes default) | 1 | yes | 0.1000 | 62.5000 | 1041.4198 | 982.3931 | 0.0000 | 103.7578 | 0.0000 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.0000 |
| stress_mixed_burst | PPO (multi-metric, 11 signals) | 1 | yes | 0.2000 | 66.5000 | 1091.7229 | 983.7270 | 0.0000 | 87.7266 | 2.0000 | 0.0000 | n/a | 0.0000 | 0.0000 | 0.9750 |
