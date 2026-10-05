# Final Results Chapter (Draft)

## 1. Experimental setup

A PPO policy was trained live against a Kubernetes cluster and compared
against the cluster's canonical Horizontal Pod Autoscaler (HPA) on an
identical, hash-pinned, previously-unseen held-out traffic workload.

## 2. Training completion

Training was frozen at checkpoint `session_019`: 5120 cumulative
environment timesteps and 20 PPO updates, using training seed 42.

## 3. Final holdout protocol

The final evaluation replayed the single frozen held-out workload
(`iteration_rl_final_holdout_v1`, SHA-256 `308f4f90ea3cea97f1927cec97efeaa652c4424e9fe5bcea421358f3425e3409`,
seed 777, 1500 seconds) across six episodes in the pre-registered order
PPO1 → HPA1 → PPO2 → HPA2 → PPO3 → HPA3, with the two controllers never
active simultaneously.

## 4. Episode outcome summary

HPA completed all three of its repeats to full duration. PPO completed
one of its three repeats (repeat 3) to full duration; repeats 1 and 2
terminated early under the frozen safety rule (4 consecutive decision
steps with the application/Prometheus target unreachable). A separate
HPA repeat-2 attempt (physical attempt 1) failed for infrastructure
reasons (a Prometheus connection failure during post-workload metric
drain, concurrent with observed cluster node instability) and was
excluded entirely from the scientific record; HPA repeat 2 was
successfully completed on a second physical attempt after the
infrastructure issue resolved.

## 5. HPA results

Across its three full-duration repeats, HPA's mean P95 latency ranged
from 689 ms to 1592 ms, with mean replica
counts between 5.06 and 5.35 (peaking at
8 replicas) and SLA violation rates between
0.52 and 0.72.

## 6. PPO results

PPO's single valid full-duration repeat (repeat 3) used a mean of
1.20 replicas (maximum 4), substantially fewer
than any HPA repeat, but recorded a higher mean P95 latency
(2298 ms) and a higher SLA violation rate (0.92)
than any HPA repeat.

## 7. Resource-latency trade-off

The single PPO full-duration repeat is consistent with a resource/latency
trade-off relative to HPA: markedly lower resource usage alongside
markedly higher latency and SLA-violation rate. This observation is based
on a single repeat and is not statistically generalizable.

## 8. Safety termination findings

PPO repeats 1 and 2 terminated after 4 consecutive decision steps with
an unreachable telemetry/application target — a pre-registered safety
rule, not an implementation defect. This is itself a relevant finding:
under this specific held-out workload and live-cluster conditions, two
of three PPO evaluation attempts could not be sustained to completion,
while HPA's control pattern did not trigger this failure mode in any of
its three valid attempts.

## 9. Section-10 metric interpretation

All five frozen Section-10 metrics (`sla_violation_rate`, `error_rate_mean`,
`resource_proxy_mean`, `reaction_time_seconds`, `action_churn_count`) were
computed identically for every episode from the same recorded per-step
samples; `action_churn_count` is diagnostic only and not part of the
win/loss decision rule.

## 10. Formal comparison-rule status

The pre-registered decision rule requires three valid full-duration
repeats per controller. HPA achieved 3/3; PPO achieved 1/3. **The formal
n=3 decision rule is therefore not applicable, and no formal comparative
winner is declared.**

## 11. Limitations

PPO's valid sample size (n=1) cannot support the frozen decision rule or
any statistical claim. PPO1/PPO2's partial data are reported for
transparency only and are excluded from the primary comparison. The
excluded HPA2 infrastructure-failure attempt reflects live-cluster
conditions during the evaluation window, not a scientific result.

## 12. Final conclusion

HPA demonstrated more consistent completion and lower latency in the
held-out evaluation, while the single full-duration PPO run used
substantially fewer replicas. Two PPO repeats terminated under the
predefined safety rule, so the planned three-repeat primary comparison
could not be completed and no formal winner was declared.
