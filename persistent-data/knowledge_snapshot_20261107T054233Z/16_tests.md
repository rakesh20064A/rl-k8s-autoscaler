# Part 16 — Tests (offline only)

**BLOCKED — VERIFIED this session.** No usable Python environment exists
anywhere on this machine right now (see 06_python_env.md: `.venv` is a
broken symlink into the deleted old repository; the global pyenv
interpreter and the separate `ml-env` pyenv virtualenv both lack `pytest`,
`ruff`, and the entire scientific stack).

Commands attempted and their exact failures (VERIFIED):
```
$ .venv/bin/python -m pytest --collect-only -q
(eval):5: no such file or directory: .venv/bin/python

$ ~/.pyenv/versions/3.10.14/bin/python -m pytest --collect-only -q
/Users/.../python3.10.14/bin/python: No module named pytest

$ ~/.pyenv/versions/3.10.14/bin/python -m ruff --version
No module named ruff
$ which ruff
ruff not found
```

**`pytest --collect-only -q` test count: UNKNOWN** (could not run — no
`pytest` installation exists anywhere this session located).
**Full suite pass/fail/error counts: UNKNOWN** (same reason).
**ruff result: UNKNOWN** (same reason).

This is a genuine, newly-discovered blocker, not a live-cluster-dependent
test being correctly skipped per the task's own instructions — this is a
TOTAL absence of any runnable test/lint environment. Per this phase's
strict read-only mandate, `scripts/bootstrap.sh` was NOT run to recreate
`.venv` (that would be an environment-modifying action, explicitly out of
scope for this snapshot). **Recreating `.venv` from scratch via
`scripts/bootstrap.sh` is a prerequisite for running ANY test in this
project going forward**, and should be the first action of any
implementation phase that follows this snapshot.
