# PHASE 9I — Comparison: compute metrics from run samples

## goal
Phase 9H (commit `a7912cb`) fixed the Comparison page's live-pairing
overlay CHARTS (a field-name mismatch meant `extract_series()` always
returned empty series) but left a sibling bug in the metrics TABLE below
those charts: for the current latest valid completed PPO/HPA pair on the
same profile, both charts render correctly while every single row of the
metrics table shows "—". This phase root-causes and fixes that, and adds
the missing full coverage of per-arm metrics the task specifies.

## root cause
`demo_ui/views/comparison.py`'s `_render_live_comparison()` built the
table from `_METRIC_LABELS` by reading `hpa_summary.get(key)` /
`ppo_summary.get(key)` for keys like `"mean_replicas"`, `"max_replicas"`,
`"scaling_events"`, `"mean_latency_p95_ms"`, `"success_rate_rps"`,
`"error_rate_rps"` directly off `run["summary"]`. But
`experiment_store.status_file_fields_for_store()` (the ONE place that
maps a run's status-file payload onto what gets persisted into
`run["summary"]`) never writes any such top-level field — `summary` only
ever carries `episode_id`/`arm`/`profile`/`n_samples`/`demo_only`/
`partial`/`workload_hash`/`workload_seed`/`workload_duration_seconds`/
`samples`. So every `summary.get(key)` call was unconditionally `None`,
for every run, regardless of how much real per-decision data the run
actually produced — the same class of bug Phase 9H fixed for the charts
(wrong field names), except here it was worse: the table never read
`samples` at all, it read a summary-level field that was simply never
populated by anything.

Verified against the real store (`persistent-data/demo_ui/runs.json`,
36 records) and its real samples: the latest genuinely-paired, same-
workload, `status == "completed"` v2 HPA/PPO pair is `aea60a66ed80`
(HPA, `hpa_v2_cpu_memory`) / `7838c5a5b76d` (PPO, `ppo_v2`), both profile
`demo_bursts_cpu_normal`, 16 samples each. Each sample's real schema
(verified directly from the file, not guessed):
`decision_index`, `t_s`/`time_s`, `p95_latency_ms`, `error_rate`,
`request_rate_rps`, `desired_replicas`, `ready_replicas`,
`max_per_pod_memory_mib`, `pod_restart_counts` (a dict of pod name ->
restart count), `hpa_status`/`ppo_decision` (arm-specific), plus
observation vectors. None of the six keys the table used to read exist
anywhere in `run["summary"]`.

## fix
Replaced the summary-field lookup with `compute_run_metrics(run)` —
computes every metric directly from `run["summary"]["samples"]`, the
SAME data `experiment_store.extract_series()` already correctly reads
(post Phase 9H) for the overlay charts above the table:

- decisions: `len(samples)`
- mean / max replicas: mean/max of `desired_replicas`
- replica-minutes: each decision's `desired_replicas` held for the
  interval to the next decision (`t_s`/`time_s` delta, minutes); the
  final decision's interval is assumed equal to the previous one
  (uniform decision cadence) since there is no further decision to
  bound it
- scaling actions: count of consecutive `desired_replicas` changes
- mean / median / max P95 latency (ms): from `p95_latency_ms`
- share of decisions with P95 >= 1s: `% of p95_latency_ms >= 1000`
- mean / peak request rate (req/s): from `request_rate_rps`
- error rate (%): mean of `error_rate` * 100
- max per-pod memory (MiB): max of `max_per_pod_memory_mib`
- pod restarts: max, across decisions, of `sum(pod_restart_counts.values())`
  (pod identities churn across decisions as pods are replaced, so the
  per-decision SUM — not a single pod's count — is tracked, and the max
  across all decisions captures restarts of pods no longer present in
  the final decision's dict)

A metric is `None` ("—") only when there are genuinely zero samples, or
that field is absent from every sample in a run that does have samples
— never because of a field-name bug. `_fmt_value()` renders units
(`ms`/`%`/`req/s`/`MiB`) and sensible rounding (0 decimals for counts,
1-2 for rates/latency/percentages). For cost/latency-type rows
(everything except the two purely descriptive request-rate rows and the
decision count), the lower (better) of the two arms' values is wrapped
in `<b>` — bold text only, never a colour fill, per the task's own
constraint. One line is added under the table: "Demo runs illustrate
behaviour; they are not evaluation evidence."

## real values for the current latest valid pair
`aea60a66ed80` (HPA) vs `7838c5a5b76d` (PPO), profile
`demo_bursts_cpu_normal`, both 16 decisions:

| Metric | HPA | PPO |
|---|---|---|
| Number of decisions | 16 | 16 |
| Mean replicas | 3.75 | **3.13** |
| Max replicas | 4 | 4 |
| Replica-minutes | 30.0 | **25.0** |
| Scaling actions | 3 | **2** |
| Mean P95 latency | 297.3 ms | **208.2 ms** |
| Median P95 latency | 221.5 ms | 224.0 ms (HPA lower) |
| Max P95 latency | 1493.2 ms | **380.4 ms** |
| Decisions with P95 >= 1s | 6.3% | **0.0%** |
| Mean request rate | 6.31 req/s | 6.08 req/s (descriptive, not highlighted) |
| Peak request rate | 12.42 req/s | 12.42 req/s (descriptive, not highlighted) |
| Error rate | 0.00% | 0.00% (tie, nothing bolded) |
| Max per-pod memory | 43.8 MiB | **39.8 MiB** |
| Pod restarts | 2 | 2 (tie, nothing bolded) |

No dashes anywhere in this real pair's row set — every metric the task
lists computes a real number.

## tests
`demo_ui/tests/test_comparison_metrics_table.py` (new):
- fixtures shaped EXACTLY like the real sample schema (field names
  pulled from the actual `persistent-data/demo_ui/runs.json` samples
  above, not guessed), with hand-computed expected values for every
  metric category (replicas/replica-minutes/scaling actions, latency
  mean/median/max/share>=1s, request rate mean/peak, error rate, max
  memory, pod restarts) — `TestComputeRunMetricsFromRealShapedSamples`.
- `TestZeroDecisionsRunShowsDashesNotFieldNameBug`: a run with no samples
  (or `None`) shows `decisions == 0` and every other metric `None`
  ("—"); `_fmt_value()` renders "—" only for `None`.
- `TestLowerIsBetterBolding`: descriptive rows (request rate, decision
  count) carry `lower_is_better is None` (no highlight either direction);
  a faster synthetic arm produces a strictly lower latency metric.
- `TestComparisonTableRendersAgainstTheRealStore`: reads the REAL
  `persistent-data/demo_ui/runs.json` (read-only, never written),
  resolves the real latest valid completed pair via
  `comparison._latest_current_system_run()` /
  `experiment_store.latest_compatible_pair()`, asserts every metric for
  both arms is non-`None`, then runs a Streamlit `AppTest` against the
  real app/store and confirms the Comparison page's rendered HTML
  contains the `du-cmp-table` table, the new "not evaluation evidence"
  caption, the real HPA mean-replicas value as text, and no em-dash
  inside the table.

`/tmp/rl-k8s-demo-ui-venv/bin/python -m pytest demo_ui/tests -q` → 317
passed (12 new). `.venv/bin/python -m pytest tests/unit -q` → 1965
passed, 2 skipped (pre-existing, unrelated to this change).
`.venv/bin/ruff check` on `demo_ui/views/comparison.py` and
`demo_ui/tests/test_comparison_metrics_table.py` → clean (two initial
findings — an unused `json` import and an unused `st_testing` binding in
the new test file — fixed before commit).

## files changed
`demo_ui/views/comparison.py` (metrics computed from samples, bolding,
new caption), `demo_ui/tests/test_comparison_metrics_table.py` (new).
