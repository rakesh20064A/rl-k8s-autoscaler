# Index40 Final Results Table

| Controller | Repeat | Outcome | Full duration | Valid primary comparison | SLA violation rate | Error rate mean | Resource proxy mean | Reaction time seconds | Action churn | Mean P95 latency ms | Mean replicas | Max replicas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PPO | 1 | terminated_safety | False | False | 0.8260869565217391 | 0.0 | 0.0 | None | 0 | 1925.0388921302472 | 0.75 | 1 |
| HPA | 1 | completed_full_duration | True | True | 0.72 | 0.0 | 0.6085714285714289 | 75.0 | 10 | 1591.9543710431828 | 5.0588235294117645 | 7 |
| PPO | 2 | terminated_safety | False | False | 0.7 | 0.0 | 0.0 | None | 0 | 2024.280275762718 | 0.6666666666666666 | 1 |
| HPA | 2 | completed_full_duration | True | True | 0.54 | 0.0 | 0.648571428571429 | 10.0 | 14 | 986.7128173402114 | 5.352941176470588 | 8 |
| PPO | 3 | completed_full_duration | True | True | 0.92 | 0.0 | 0.03428571428571428 | 1350.0 | 3 | 2298.4703264780524 | 1.196078431372549 | 4 |
| HPA | 3 | completed_full_duration | True | True | 0.52 | 0.0 | 0.6257142857142859 | 77.14285714285714 | 15 | 689.4460420854738 | 5.294117647058823 | 8 |
