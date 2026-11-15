| arm | profiles_run_on | role |
|---|---|---|
| PPO (multi-metric, 11 signals) | all 3 eval + all 3 stress | RL policy under test |
| HPA CPU+Memory (resource baseline) | all 3 eval + all 3 stress | resource-metric baseline (CPU+memory) |
| HPA CPU-only (Kubernetes default) | all 3 eval + all 3 stress | Kubernetes-default baseline (CPU only) |
