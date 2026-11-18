# PHASE 9D — UI: arm selection as a dropdown (PPO / HPA)

## goal
On the Experiment page (both "Live cluster" and "Sample data (offline)"
modes), replace the arm radio buttons with an `st.selectbox` styled
identically to the Traffic pattern / Resource type / Load level selects
in the same control row, drop the "(CPU+memory)" suffix from the arm
label, move that detail into a `help=` tooltip, and make sure the short
"PPO"/"HPA" labels (not any longer display form) are what Live Details,
History, and Comparison show for an arm — all as a pure UI/label change
with no change to the internal arm ids or how they're consumed downstream.

## investigation
`demo_ui/components/controls_v2.py` already centralizes everything arm-
related: `ARM_LABELS` (the one source of the arm's display string),
`display_arm_label()` (the one formatting function, used by
`live_details_v2.py`, `live_details.py`, `history.py`, and
`views/experiment.py`), and `render_controls_v2()` — the single control-
strip builder both "Live cluster" and "Sample data (offline)" modes on
the Experiment page call (`views/experiment.py`'s `render()` calls it
once, before branching on `is_live`), so there was no separate Sample-
mode control path to keep in sync. The old arm control was an
`st.radio(..., horizontal=True)` inside its own `st.columns(...)` cell
(`c_arm`), already the first column of the same row as the pattern/
resource/load selects — so no layout/column restructuring was needed,
only swapping the widget type and its options source.
`views/comparison.py` (the Comparison page) already renders literal
`"HPA"`/`"PPO"` strings in its chart legends/table, never through
`display_arm_label()`, so it required no change.

## changes
- `demo_ui/components/controls_v2.py`:
  - `ARM_LABELS["hpa_v2_cpu_memory"]` changed from `"HPA (CPU+memory)"`
    to `"HPA"` — the single source `display_arm_label()` and the control
    both read from, so this alone fixes the label everywhere it's shown
    (Live Details header/card, History table, Experiment page banners).
  - Added `ARM_HELP = "HPA uses CPU (260%) and memory (120%) targets,
    2-8 replicas"`, the one place that CPU/memory-target detail now
    lives.
  - The arm control in `render_controls_v2()` changed from
    `st.radio(..., horizontal=True)` to `st.selectbox(..., help=ARM_HELP)`,
    same key (`du_v2_arm_widget`), same session-state field
    (`du_v2_selected_arm`), same `controls_disabled` lock already shared
    with the pattern/resource/load selects, same `_LABEL_TO_ARM` mapping
    feeding the exact same internal arm id (`ppo_v2` / `hpa_v2_cpu_memory`)
    through the same downstream path (`render()` in `experiment.py` →
    `RealExperimentControllerV2.start()` / `build_fixture_v2_run()`).
    Default remains `"ppo_v2"` (unchanged `setdefault`).
- `demo_ui/tests/test_arm_dropdown.py` (new): label/mapping unit checks,
  dropdown-vs-radio widget-type check, default-is-PPO check, options-are-
  exactly-PPO/HPA check, help-tooltip-text check, PPO→`ppo_v2` and
  HPA→`hpa_v2_cpu_memory` mapping checks driven through the real
  Experiment page in Sample data (offline) mode, disabled-while-active-run
  check (Live cluster mode, reusing the same stale-lock test pattern from
  Phase 9B) vs. enabled-when-idle, same-row-in-sample-mode check, and
  short-label checks on History / Live Details / Comparison (including an
  explicit "`CPU+memory` must never appear" assertion on all three).

## tests
- demo_ui suite: `/tmp/rl-k8s-demo-ui-venv/bin/python -m pytest demo_ui/tests -q`
  → 252 passed, 0 failed (was 237 before this phase's new test file).
- Repo unit suite: `.venv/bin/pytest tests/ -q` → 1934 passed, 2 skipped,
  0 failed.
- `.venv/bin/ruff check demo_ui/` → all checks passed, no new lint issues.

## commit
`861df7d` "UI: arm selection as dropdown (PPO / HPA)" — 2 files changed
(`demo_ui/components/controls_v2.py`, new
`demo_ui/tests/test_arm_dropdown.py`). Not pushed, no Co-Authored-By
trailer.

## chain
Follows Phase 9B (`be53e99`, stale-lock/pairing fix) and the demo-traffic
3-select phase (`24f2b27`), both on top of Phase 9's single-system
interface (`33d7a4a`) that introduced `display_arm_label()`.
