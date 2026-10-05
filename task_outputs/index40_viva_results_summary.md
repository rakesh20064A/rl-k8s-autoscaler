# Viva / Demo Results Summary (Index40)

## Why PPO was chosen
PPO (Proximal Policy Optimization) is a widely-used, stable on-policy RL
algorithm well suited to a discrete, low-dimensional action space like
replica scaling decisions, and is a standard baseline choice for
autoscaling research.

## What PPO observed
An 11-dimensional normalized observation vector covering CPU/replica,
request-rate, latency, and short-horizon history (EWMA) features, sourced
live from Prometheus and the Kubernetes Deployment/HPA objects.

## Actions
`Discrete(3)`: SCALE_DOWN_ONE (-1), HOLD (0), SCALE_UP_ONE (+1), applied
every 30-second decision interval, bounded to the replica range [1, 8].

## Reward objective
A multi-component reward balancing SLA/latency performance against
replica-resource cost (the same `resource_proxy` quantity reused in the
Section-10 evaluation metric), with penalties for scaling reversals —
see `phase_16u_reward_specification.yaml` for the exact frozen formula.

## Training budget actually completed
5120 environment timesteps (20 PPO updates), live against the real
cluster.

## Why only 5120 instead of the original 12288-step design budget
The original 12288-step design budget (~118.6 hours of pure live decision
time, excluding episode-reset overhead) proved infeasible to complete
within the capstone timeline on a single-machine, single-cluster
development environment subject to real infrastructure interruptions
(network changes, transient node instability). Training was honestly
frozen at the live progress actually achieved (5120 steps) rather than
silently extending the deadline or fabricating additional training.

## Final holdout design
A single, newly-materialized, hash-pinned traffic workload
(`iteration_rl_final_holdout_v1`, seed 777, 1500s, peak 65 RPS) was
deliberately designed to combine workload shapes not seen together in any
training profile, and was never inspected or executed before both PPO
training and the evaluation protocol were frozen.

## Why HPA target = 260%
260% CPU utilization (relative to a 100m request) is the project's
long-standing canonical HPA configuration (established well before this
RL work began), chosen so HPA's natural scale-up threshold falls in a
comparable operating range to the environment's reward design.

## What safety termination means
A frozen, pre-registered rule: if the live application/Prometheus target
is unreachable for 4 consecutive 30-second decision steps (120 seconds),
the episode terminates and the system is restored to a safe state. This
protects against evaluating (or training) on meaningless data during a
genuine infrastructure outage — it is not an implementation bug.

## Why HPA2 (attempt 1) was retried
HPA repeat 2's first physical attempt failed for a clear infrastructure
reason (a Prometheus connection failure during post-workload metric
drain, concurrent with observed Kubernetes node instability) — the
traffic workload itself had already completed in full. This is a pure
tooling failure with no scientific outcome to be "outcome-aware" about,
so a second physical attempt at the same logical repeat was permitted
and executed once the infrastructure recovered and was independently
verified stable.

## Why PPO1/PPO2 were NOT retried
PPO repeats 1 and 2 terminated via the frozen, pre-registered SAFETY rule,
which is itself a legitimate, meaningful scientific outcome (not a tooling
failure) — retrying it because the outcome was unfavorable would be
outcome-aware retry, which the project's evaluation protocol explicitly
prohibits.

## Main result
HPA completed 3/3 held-out repeats with consistently lower latency and
higher resource usage; PPO completed 1/3 repeats, using substantially
fewer replicas in that one repeat but with higher latency and a higher
SLA-violation rate. Because PPO did not achieve the required three valid
repeats, the pre-registered formal comparison rule could not be applied,
and **no formal winner is declared**.

## Limitations
Single trained checkpoint, single evaluation pass, PPO n=1 valid sample
(no statistical power), live-infrastructure variability during evaluation,
and a substantially reduced training budget relative to the original
design target.

## Future improvements
Complete the originally-designed training budget; obtain three valid
full-duration PPO repeats (reusing the same safety-termination handling,
simply re-attempting the logical repeat under the SAME retry discipline
already used for tooling failures, if and only if the protocol is amended
to permit it); investigate the root cause of PPO's telemetry-unavailability
safety terminations specifically under this held-out workload; and
evaluate robustness to infrastructure instability as its own explicit,
measured research question.
