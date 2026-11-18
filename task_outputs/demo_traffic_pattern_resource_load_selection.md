# Demo traffic: pattern × resource × load selection

## Summary

Replaced the Experiment page's single "Demo traffic" select with three
independent selects -- **Traffic pattern** (Constant, Step, Ramp, Bursts,
Periodic (wave), Ramp + bursts), **Resource type** (CPU, Memory, Mixed),
**Load level** (Normal, High) -- whose combination picks one of 36
newly generated, deterministic demo traffic profiles.

## Profile generator (`scripts/generate_demo_profiles.py`)

- New script, deterministic, produces 36 YAML profiles into
  `configs/traffic_profiles/v2_demo/`, named
  `demo_<pattern>_<resource>_<load>.yaml`.
- Every profile: 480s total duration (= 16 × the project's 30s decision
  interval, confirmed against `scripts/eval_v2_episodes.py`'s
  `decision_interval_s=30.0` default and the existing 3 `demo_*_short`
  profiles' own 480s/16-decision convention).
- Deterministic seeds 7101-7136, one per file. Assignment order:
  patterns outer, resources middle, loads inner
  (`index = pattern_idx*6 + resource_idx*2 + load_idx`, `seed = 7101 + index`).
- Load-level envelope (confirmed against existing profiles, not invented):
  - Normal: peak ~18 RPS (<= 20 limit), intensity 3.
  - High, CPU/Mixed: peak 65 RPS (<= 70 limit), intensity 4 -- the same
    peak already validated live by the training `burst_mixed` profile and
    the `stress_cpu_burst`/`stress_mixed_burst` exploratory profiles.
  - High, Memory: peak 40 RPS (<= 45 limit), intensity 4 -- the same peak
    already validated live by `stress_memory_ramp.yaml`.
  - Memory/Mixed profiles (both load levels) set `memory_mb: 16` /
    `memory_safe_max_in_flight: 8`, matching every other V2 memory/mixed
    profile in the repo.
- "Ramp + bursts" is a genuine composite: a `ramp` phase (200s) followed
  by a `burst` phase (280s), using the existing unified multi-phase
  schema (`phases:` list) -- no new phase type introduced.
- Every file is validated before being written (parses via
  `loadgen.traffic_generator.parse_traffic_profile`, exact 480s duration,
  non-degenerate peak RPS, peak within its load level's limit); the
  script refuses to write a file that fails validation.
- The original 3 `demo_*_short` profiles were left in place, untouched
  (`LEGACY_DEMO_PROFILE_NAMES`) -- additive, not replaced or aliased.

## Firewall

- `tests/unit/test_v2_demo_profile_catalog.py` asserts the 36-profile
  catalog is disjoint from `V2_TRAINING_PROFILE_ALLOWLIST`,
  `EVAL_PROFILE_ALLOWLIST`, and `STRESS_PROFILE_ALLOWLIST`, and that the
  live training schedule rotation never selects any of them.
- `scripts/run_demo_episode_v2.py`'s `DEMO_PROFILE_ALLOWLIST` was
  extended (union, not replacement) with the 36 catalog names; its
  `resolve_demo_profile_path()` refuses any training/eval/stress profile
  name (covered by dedicated tests).

## UI changes (`demo_ui/`)

- `components/controls_v2.py`: 3 new selects (pattern/resource/load)
  replace the old single "Demo Traffic" select, reusing the existing
  active-run-locking (`controls_disabled`) logic. Returns
  `(arm, profile_name, pattern, resource, load, start_clicked, stop_clicked)`.
- `profile_preview.py` (new): computes the request-rate-over-time curve
  for a profile directly from the real parsed YAML.
- `views/experiment.py`: new "Traffic Preview" section below the
  controls -- a Plotly line chart of the selected profile's actual
  request rate, a one-line templated-by-resource-type description, and
  (when Load level is High) the note "High load: up to 70 req/s, like
  the stress test". The "DEMO ONLY" label remains visible.
- `fixtures_v2.py`: Sample-data fixtures are now derived from the actual
  profile file's RPS curve (`rps_curve_from_profile`) instead of a
  hand-written per-profile curve -- generalizes to all 36+3 profiles with
  no new hardcoded cases.
- `views/live_details_v2.py`: run banner / telemetry strip now show the
  parsed pattern/resource/load (e.g. "Bursts / Memory / High") instead of
  a raw profile-name title-case string, for any profile the catalog can
  parse back; legacy `demo_*_short` records fall back to the old format.
- `views/live_details.py`: its own "Sample data (offline)" dashboard also
  offers the same 3 selects now (previously a single select).
- `demo_ui/requirements-ui.txt`: added `httpx==0.28.1` (transitively
  required by `loadgen.traffic_generator`, now imported by the UI for the
  preview chart and fixture derivation).

## Tests

- New `tests/unit/test_v2_demo_profile_catalog.py` (488 cases): catalog
  shape, seed assignment, per-file parse/duration/non-degenerate/peak
  checks, training/eval/stress firewall, demo-runner acceptance.
- New `demo_ui/tests/test_demo_traffic_catalog_ui.py` (109 cases): 3-select
  → filename mapping for all 36 combinations (driven through the real
  Streamlit app, not just the pure function), Sample-data rendering for
  all 36 combinations, and a grep-style check for "V1"/"V2"/"Generation"
  wording in the new/changed UI source files.
- Updated `tests/unit/test_run_demo_episode_v2.py` and
  `demo_ui/tests/test_live_details_fresh_state.py` for the new/extended
  allowlist and the 3-select key set.

## Verification

- `/tmp/rl-k8s-demo-ui-venv/bin/python -m pytest demo_ui/tests -q`: 237 passed.
- `.venv/bin/pytest tests/ -q`: 1934 passed, 2 skipped.
- `ruff check demo_ui/ scripts/`: all checks passed.

## Commit

`24f2b27` "Demo traffic: pattern × resource × load selection" (not pushed).
