# Part 06 — Python Environments

## CRITICAL FINDING: `.venv` is a broken symlink into the deleted old repo

**VERIFIED directly** (independently confirmed by both this session's own
earlier check and a dedicated research agent):

```
$ ls -la .venv
lrwxr-xr-x  1 rakeshdandamudi  staff  55 Oct  6 18:09 .venv -> /Users/rakeshdandamudi/Capstone/rl-k8s-autoscaler/.venv
$ ls -ld /Users/rakeshdandamudi/Capstone/rl-k8s-autoscaler
ls: /Users/rakeshdandamudi/Capstone/rl-k8s-autoscaler: No such file or directory
```

The clean repo's `.venv` has, since this project's creation, been a
**symlink pointing INTO the forbidden old repository**
(`~/Capstone/rl-k8s-autoscaler`), not an independent virtual environment of
its own. That old repository **no longer exists on this machine at all**
(confirmed `ls -ld` on the bare directory path) — something removed it
entirely between Phase 4B-R4F (2026-11-03, when `.venv/bin/python` was last
used successfully) and now. **This was not done by this session** — no
`rm`/delete command targeting that path was ever issued in this
conversation's history. As a direct consequence, `.venv/bin/python` is now
completely nonexistent, and **no working, package-complete Python
environment for this project currently exists on this machine.**

This also means: every prior phase's `.venv/bin/python` invocation
(Phase 4A through 4B-R4F) was, at the filesystem level, reading interpreter/
package files that physically lived inside the forbidden old repository's
directory tree, even though the resolved interpreter path/architecture/
package versions were independently verified correct at the time. This is
disclosed here as a retrospective finding, not something to fix in this
read-only phase.

**Risk surface directly caused by this**: `demo_ui/live_execution_adapter.py`
hardcodes `VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"` and
raises `RuntimeError("...run scripts/bootstrap.sh first")` if missing —
**the UI's real LIVE-execution path is currently non-functional.**
`demo_ui/README.md`'s own instructions (`.venv/bin/python -m pytest
demo_ui/tests -q`) are likewise currently non-functional.

## No other repo-tracked old-path references found, except one real test risk

Grep across all tracked source/config files for the literal old-repo path
(without `-github-clean`) found **no hits** except:
- `tests/unit/test_traffic_generator.py` (lines 1285, 1324, 1415) —
  hardcodes `project_root = Path("/Users/rakeshdandamudi/Capstone/rl-k8s-autoscaler")`
  to locate calibration CSVs and for `sys.path.insert`. **These specific
  test methods will fail or silently resolve a nonexistent directory now
  that the old repo is gone.** `DEFECT`, confirmed, not fixed (read-only
  phase).

`.python-version` = `3.10.14`. No `pyproject.toml`/`setup.cfg` found;
`pytest.ini` exists but has no old-path reference.

## Currently reachable Python (pyenv global 3.10.14) — package audit

Since `.venv` is unusable, the only reachable interpreter is the pyenv
global `3.10.14` (`~/.pyenv/versions/3.10.14/bin/python3`, arm64). Package
check (`pip freeze`):

| Package | Installed | Pinned (repo) | Status |
|---|---|---|---|
| numpy | 2.2.6 | `requirements/base.txt`: 1.26.4 | mismatch (major) |
| pandas | 2.3.3 | 2.2.3 | mismatch (minor) |
| torch | 2.9.1 | `requirements/rl.txt`: 2.5.1 | mismatch |
| PyYAML | 6.0.3 | base.txt: 6.0.2 / UI reqs: 6.0.3 | partial match |
| streamlit | 1.65.0 | UI reqs: 1.65.0 | MATCH |
| plotly | 7.1.0 | UI reqs: 7.1.0 | MATCH |
| uvicorn | 0.54.0 | 0.32.1 | mismatch |
| requests | 2.34.2 | base.txt: 2.32.3 / UI reqs: 2.34.2 | partial match |
| stable-baselines3 | **NOT INSTALLED** | rl.txt: 2.6.0 | MISSING |
| gymnasium | **NOT INSTALLED** | rl.txt: 1.1.1 | MISSING |
| kubernetes (py client) | **NOT INSTALLED** | base.txt: 31.0.0 | MISSING |
| httpx | **NOT INSTALLED** | dev.txt: 0.28.1 | MISSING |
| fastapi | **NOT INSTALLED** | 0.115.6 | MISSING |
| prometheus-client | **NOT INSTALLED** | 0.21.1 | MISSING |
| pytest | **NOT INSTALLED** | dev.txt: 8.3.4 | MISSING |
| ruff | **NOT INSTALLED** | dev.txt: 0.9.2 | MISSING |

**Conclusion**: the globally-reachable interpreter satisfies only the
demo-UI's own lightweight requirements (streamlit/plotly/pyyaml/requests)
— not the project's scientific stack at all. Also checked the separate
pyenv virtualenv `ml-env`
(`~/.pyenv/versions/3.10.14/envs/ml-env`): also missing
`stable_baselines3`. **No usable training/test/lint environment currently
exists anywhere on this machine** — confirmed directly this session
(`pytest`/`ruff` both report `No module named ...` under every interpreter
checked). `scripts/bootstrap.sh` would need to be re-run from scratch to
recreate a working, project-dedicated `.venv` (not done — read-only phase).

## Every requirements file (FROM_CODE, full contents)

- `app/requirements.txt`: fastapi==0.115.6, uvicorn[standard]==0.32.1,
  prometheus-client==0.21.1.
- `requirements/base.txt`: kubernetes==31.0.0, prometheus-api-client==0.5.5,
  requests==2.32.3, PyYAML==6.0.2, numpy==1.26.4, pandas==2.2.3,
  fastapi==0.115.6, uvicorn[standard]==0.32.1, prometheus-client==0.21.1,
  python-dotenv==1.0.1.
- `requirements/rl.txt` (extends base.txt): stable-baselines3==2.6.0,
  gymnasium==1.1.1, torch==2.5.1, tensorboard==2.18.0, scikit-learn==1.6.1,
  matplotlib==3.10.9. Contains an inline CORR-28 comment documenting why
  this exact SB3/gymnasium/torch combination was chosen.
- `requirements/dev.txt` (extends base.txt): locust==2.33.1, ruff==0.9.2,
  pytest==8.3.4, pytest-cov==6.0.0, mypy==1.14.1, httpx==0.28.1.
- `demo_ui/requirements-ui.txt`: streamlit==1.65.0, plotly==7.1.0,
  requests==2.34.2, PyYAML==6.0.3 — explicitly documented to be installed
  into a SEPARATE venv, never the scientific `.venv`.

## Which environment the UI vs. training scripts expect

`demo_ui/README.md`: the UI's own rendering code is meant to run in its
OWN lightweight venv (`requirements-ui.txt`). But `demo_ui/
live_execution_adapter.py` (the real LIVE-mode execution path, new/
untracked) explicitly launches its subprocess via `.venv/bin/python` (the
scientific venv, since `scripts/run_demo_episode.py` imports
`stable_baselines3`/the Kubernetes client, which the lightweight UI venv
deliberately lacks) — so the UI straddles two environments, and its real-
execution half is currently non-functional per the `.venv` breakage above.
All training/evaluation scripts (`train_ppo_*`, `evaluate_ppo_*`) expect
the full scientific `.venv` with `requirements/rl.txt` installed.
