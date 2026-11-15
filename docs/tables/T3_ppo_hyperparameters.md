| parameter | value |
|---|---|
| learning_rate | 0.0003 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| policy | MlpPolicy |
| net_arch | [128, 128] |
| training_budget | 48 sessions x 256 steps/session = 12288 total timesteps (final_min2 campaign) |
