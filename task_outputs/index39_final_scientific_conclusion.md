# Index39 — Final Scientific Conclusion

The predefined three-repeat PPO-vs-HPA primary comparison could not be
completed as designed because two of the three PPO evaluation repeats
terminated under the frozen safety rule. HPA completed all three held-out
repeats, whereas PPO completed one full repeat. Therefore, the frozen n=3
decision rule is not applicable, and no formal comparative winner is
declared.

## What can be said descriptively

- HPA completed 3/3 held-out repeats of the final evaluation workload.
- PPO completed 1/3 full repeats; the other two terminated early under the
  frozen, pre-registered safety rule (4 consecutive decision steps with the
  Prometheus/application target unreachable).
- In its one full-duration repeat, PPO used markedly fewer replicas
  (mean ~1.20, max 4) than HPA did in any of its three repeats (mean
  ~5.06–5.35, max 7–8).
- HPA achieved lower mean P95 latency in all three of its repeats than PPO
  did in its one full repeat.
- PPO's own safety terminations are themselves a relevant finding: under
  this specific held-out workload and live-cluster conditions, the
  evaluation could not sustain two of three attempts to full duration,
  which is evidence of a robustness limitation in this configuration,
  independent of any latency/resource comparison.

## What cannot be said

- No claim of statistical significance is made or supported (sample sizes
  of n=1 and n=3 do not support one, and the frozen protocol itself
  disclaims inferential testing even at the originally intended n=3).
- No formal "PPO wins," "HPA wins," or "mixed/no-clear-winner under the
  n=3 rule" conclusion is drawn, because the required comparable sample
  count (3 valid full-duration repeats per controller) was not achieved
  for PPO.
- This is not evidence that PPO is categorically worse or better than HPA
  in general -- it reflects this one evaluation run, under this one
  held-out workload, on this one trained checkpoint (`session_019`,
  5120 timesteps), with this specific live-infrastructure behavior during
  the evaluation window.

## Formal status

**NO FORMAL WINNER DECLARED.** The comparison is reported as incomplete
under its own pre-registered decision rule.
