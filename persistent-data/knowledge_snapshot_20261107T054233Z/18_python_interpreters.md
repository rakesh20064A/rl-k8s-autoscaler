# Part 18 — Python Interpreter Inventory & R4F Report Check

All VERIFIED directly, 2026-11-07, read-only.

## Pin-list correction

The user-supplied pin list is accurate against the actual requirements
files EXCEPT: `requests` is pinned at `2.32.3` in `requirements/base.txt`
but `2.34.2` in `demo_ui/requirements-ui.txt` (two different pins for the
same package, by design — base.txt is the scientific stack, UI reqs is
separate); `PyYAML` is `6.0.2` in base.txt but `6.0.3` in UI reqs,
likewise. No other discrepancy found.

## Every interpreter found

| # | Path | Realpath | Version | Arch |
|---|---|---|---|---|
| 1 | `~/.pyenv/shims/python[3]` → | `~/.pyenv/versions/3.10.14/bin/python3.10` | 3.10.14 | arm64 |
| 2 | `~/.pyenv/versions/3.10.14/envs/ml-env/bin/python` | same (pyenv virtualenv) | 3.10.14 | arm64 |
| 3 | `~/miniconda3/bin/python` (conda `base` env — the only conda env; `conda env list` shows no project-specific env) | same | 3.13.5 | arm64 |
| 4 | `/opt/homebrew/bin/python3.14` | `/opt/homebrew/opt/python@3.14/bin/python3.14` | 3.14.6 | arm64 |
| 5 | `/usr/local/bin/python3.12` / `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12` | same file | 3.12.0 | **x86_64** (Intel build, running under Rosetta on this arm64 Mac — this is WHY its installed `torch` fails to import: `dlopen ... incompatible architecture (have 'x86_64', need ... arm64)`) |
| 6 | `/usr/bin/python3` | `/Library/Developer/CommandLineTools/usr/bin/python3` | 3.9.6 | arm64 |

`pipx`: not installed. `mamba`: not found. No other virtualenvs found
(`find ~ -maxdepth 5 -name pyvenv.cfg`, excluding caches, returned only an
unrelated `~/.config/gcloud/virtenv/pyvenv.cfg`). Exhaustive filesystem
search for `stable_baselines3` anywhere under `/Users/rakeshdandamudi`
(excluding Library/Caches and Trash): **zero results** — confirming no
hidden/forgotten venv contains it either.

## Package check — interpreters with torch/gymnasium/sb3 (3 qualify, all via torch only)

**#1 `~/.pyenv/versions/3.10.14/bin/python3.10`** (closest version/arch
match to the project's own `.python-version`):

| Package | Installed | Pinned | Status |
|---|---|---|---|
| stable-baselines3 | **MISSING** | 2.6.0 | MISSING |
| gymnasium | **MISSING** | 1.1.1 | MISSING |
| torch | 2.9.1 | 2.5.1 | DIFFERENT |
| numpy | 2.2.6 | 1.26.4 | DIFFERENT |
| pandas | 2.3.3 | 2.2.3 | DIFFERENT |
| kubernetes | **MISSING** | 31.0.0 | MISSING |
| prometheus-api-client | **MISSING** | 0.5.5 | MISSING |
| requests | 2.34.2 | 2.32.3 | DIFFERENT |
| PyYAML | 6.0.3 | 6.0.2 | DIFFERENT |
| fastapi | **MISSING** | 0.115.6 | MISSING |
| uvicorn | 0.54.0 | 0.32.1 | DIFFERENT |
| prometheus-client | **MISSING** | 0.21.1 | MISSING |
| python-dotenv | **MISSING** | 1.0.1 | MISSING |
| tensorboard | **MISSING** | 2.18.0 | MISSING |
| scikit-learn | **MISSING** | 1.6.1 | MISSING |
| matplotlib | 3.10.7 | 3.10.9 | DIFFERENT |
| httpx | **MISSING** | 0.28.1 | MISSING |
| pytest | **MISSING** | 8.3.4 | MISSING |
| pytest-cov | **MISSING** | 6.0.0 | MISSING |
| ruff | **MISSING** | 0.9.2 | MISSING |
| mypy | **MISSING** | 1.14.1 | MISSING |
| locust | **MISSING** | 2.33.1 | MISSING |
| streamlit | 1.65.0 | 1.65.0 | MATCH |
| plotly | 7.1.0 | 7.1.0 | MATCH |

**#2 `~/.pyenv/versions/3.10.14/envs/ml-env/bin/python`**: has `torch==2.8.0`,
`scikit-learn==1.7.2`, `tensorboard==2.16.2`, `prometheus_client==0.23.1`,
`fastapi==0.104.1`, `httpx==0.28.1`, `python-dotenv==1.0.0`,
`PyYAML==6.0.2` (MATCH), `pandas==2.1.3` — but **`stable_baselines3` and
`gymnasium` are both MISSING**, `kubernetes` client is MISSING,
`prometheus-api-client` is MISSING, `pytest`/`ruff`/`mypy`/`locust` all
MISSING. This is the "richest" non-qualifying environment found (clearly a
general-purpose ML/notebook environment, not this project's dedicated
venv), but it still fails on the two single most decisive packages.

**#3 `~/miniconda3/bin/python`** (conda `base`, requires
`KMP_DUPLICATE_LIB_OK=TRUE` to even import `torch` without an OpenMP
runtime-conflict crash): `torch==2.8.0`, `scikit-learn==1.7.2`,
`tensorboard==2.20.0`, `httpx==0.28.1`, `python-dotenv==1.1.0`,
`PyYAML==6.0.2` (MATCH) — but **`stable_baselines3` and `gymnasium` both
MISSING**, same as #2.

## Interpreters with NONE of stable_baselines3/gymnasium/torch

- `/opt/homebrew/bin/python3.14` — all three missing.
- `/usr/local/bin/python3.12` — sb3/gymnasium missing; torch present but
  fails to import (x86_64/arm64 architecture mismatch, see table above).
- `/usr/bin/python3` (3.9.6, Apple's Command Line Tools Python) — all
  three missing.

## Conclusion

**No interpreter on this machine has `stable_baselines3` installed at
all** — not one. This is the single disqualifying fact: every other
package gap is secondary. **NO_COMPLETE_ENV_FOUND.**

## Functional check (Part 5)

**Could not be performed for ANY interpreter** — loading
`models/ppo/ppo_model.zip` or
`persistent-data/rl_training_v2/final/sessions/session_002/ppo_model.zip`
requires `PPO.load()` from the `stable_baselines3` package itself, which
is absent from every interpreter found. Reported honestly as blocked, not
attempted via a disallowed workaround (no `pip install` was run, per the
strict read-only mandate).

`pytest --collect-only -q`: **could not be run on any interpreter**
(`pytest` is absent from all 6 interpreters checked, including the 3
torch-qualifying ones). `ruff --version`: **not found anywhere** — no
`ruff` Python module on any interpreter, and no standalone `ruff` binary
exists on the machine (`which ruff`, `/opt/homebrew/bin/ruff`,
`/usr/local/bin/ruff`, `brew list ruff`, and a filesystem-wide `find`
all came back empty/absent).

## R4F report check

- `ls -la`: 45,672 bytes, 2026-11-03 13:45.
- `shasum -a 256`:
  `6680d062ded918b8bbc851f22c7ea4c24b87c59319853cae19a68c22509ec582`.
- `wc -l`: 748 lines.
- Headings: 1 title + 14 numbered `##` sections (Executive verdict through
  "Explicit list of items that should remain unchanged"), with sub-headings
  under sections 2-4, 7, and 11 (full list captured in `raw/`).
- **Executive verdict** (verbatim): `V2_R4F_AUDIT_COMPLETE_WITH_BLOCKERS` —
  summarizing that the observation pipeline and tie-break policy are both
  already well-implemented and fully tested, but two confirmed defects
  remain (the V1-environment coupling in the V2 evaluation path; the
  non-canonical live cluster), plus the empirical finding that
  epsilon=0.05 would force HOLD on 91% of the 768 real trained-policy
  decisions.
- **Prioritised roadmap (Section 11, verbatim)**: **P0 = none** (nothing in
  this audit blocks training mechanics themselves; the real P0 blocker is
  the separate, already-known live-cluster availability question from
  Phases 4B-R2 through R4E). **P1** = (1) fix the V1/V2 environment
  coupling defect, (2) build the missing V2-HPA restoration/episode-runner
  path, (3) do not adopt epsilon=0.05 as final — recalibrate after training
  completes. **P2** = all other Section-9 items. **REJECT** = Option B
  (Prometheus Adapter), new scenario profiles beyond the one recommended,
  and any reward-weight retuning not backed by a confirmed defect.

---

```text
best_interpreter_path = NONE (no interpreter has stable_baselines3)
best_interpreter_realpath = ~/.pyenv/versions/3.10.14/bin/python3.10 (closest version/arch match, still incomplete)
python_version_arch = 3.10.14 / arm64
missing_packages = stable-baselines3, gymnasium, kubernetes, prometheus-api-client, fastapi, prometheus-client, python-dotenv, tensorboard, scikit-learn, httpx, pytest, pytest-cov, ruff, mypy, locust (on the closest-match interpreter; stable-baselines3/gymnasium are missing from EVERY interpreter checked, with no exception)
version_drift_packages = torch (2.9.1 vs 2.5.1), numpy (2.2.6 vs 1.26.4), pandas (2.3.3 vs 2.2.3), requests (2.34.2 vs 2.32.3), PyYAML (6.0.3 vs 6.0.2), uvicorn (0.54.0 vs 0.32.1), matplotlib (3.10.7 vs 3.10.9)
v1_model_loads = NO (stable_baselines3 unavailable on every interpreter; PPO.load() could not be attempted anywhere)
v2_session002_model_loads = NO (same reason)
pytest_collect_count = UNKNOWN (pytest not installed on any interpreter found)
ruff_version = UNKNOWN (ruff not installed anywhere, no standalone binary found)
r4f_report_exists = YES
r4f_sha256 = 6680d062ded918b8bbc851f22c7ea4c24b87c59319853cae19a68c22509ec582
r4f_verdict = V2_R4F_AUDIT_COMPLETE_WITH_BLOCKERS
files_modified = NO
packages_installed = NO
venv_or_symlink_changed = NO
kubernetes_touched = NO
status = INVENTORY_COMPLETE
```

**Final verdict: `NO_COMPLETE_ENV_FOUND`**
