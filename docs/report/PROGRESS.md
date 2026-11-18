# Capstone report build progress

## Status: COMPLETE (compiles cleanly, in page-count range) -- single-system, official-format rewrite

This session performed a full structural rewrite of the report that was
previously marked COMPLETE (see "Prior COMPLETE state" below for what that
session left). The report now describes ONE system only (no V1/V2,
generation, or redesign framing anywhere outside verbatim code listings /
real repo filenames -- see "Residual V-string exceptions" below), follows
the official VIT-AP format (docs/report/ref/official_format.pdf) for front
matter order, title page, and Chapter 1/2 section structure, uses
continuous arabic page numbering (no roman numerals anywhere), has
hand-built TOC/List of Tables/List of Figures as actual tables (not
\tableofcontents-generated), adds a Standards section, a Biodata chapter,
and a new Demo Interface section (Chapter 4) describing the current
demo_ui/ code with real screenshots.

## What changed this session (resuming from "COMPLETE, 75pp, roman+arabic")

- **Numbering scheme**: `\pagenumbering{roman}` for front matter + restart
  at `\pagenumbering{arabic}` for the body -- REMOVED. Now: title page is
  the only unnumbered page (`titlepage` environment's own `\thispagestyle{empty}`),
  then `\pagenumbering{arabic}` is called once, right before the Certificate
  page, which resets the counter to 1 -- Certificate is page 1, and
  numbering is continuous arabic straight through Biodata at the end. See
  `main.tex`.
- **Front matter order**: now exactly Title page (unnumbered) -> Certificate
  -> Acknowledgements -> Abstract -> Table of Contents -> List of Tables ->
  List of Figures -> List of Abbreviations -> Chapter 1 onward, matching
  the spec and `official_format.pdf`'s own order.
- **TOC / List of Tables / List of Figures**: rebuilt as explicit
  `longtable`s (`chapters/00_toc.tex`, `00_list_of_tables.tex`,
  `00_list_of_figures.tex`) with columns matching `official_format.pdf`
  exactly (S.No./Chapter/Title/Page Number for the TOC; No./Title/Page No.
  for the two lists), every page number a `\pageref{}` so it stays correct
  under recompilation. `\tableofcontents`/`\listoffigures`/`\listoftables`
  macros are no longer invoked (removed from `main.tex`); `tocloft` package
  is still loaded but unused (harmless).
- **Single-system rewrite**: removed entirely -- the "V1 Results" section
  and its two figures (`results/figures/replicas_over_time_full_runs.png`,
  `p95_latency_over_time_full_runs.png`), "V1 Failure Analysis", the
  "V1 Reward Function" section/equation, the "V1 to V2 Redesign" section
  and its comparison table (`tab:v1-v2-comparison`), every "V1"/"V2"/
  "generation"/"redesign" mention in prose, captions, table cells, and the
  List of Abbreviations (V1/V2 rows dropped). Every design decision
  (observation space, replica floor=2, conditional waste term, 1000ms SLA
  target, safety-aware tie-break, two HPA baselines, 12,288-step budget)
  is now justified on its own technical merits with CORR-log citations,
  not by contrast with a discarded design. The two zero-ready-replica
  training stops (sessions 4, 38) are now framed as live evidence *for*
  the floor=2 design choice, not as a V1-era failure mode recurring.
  Abstract, Acknowledgements ("two-generation redesign" phrase), and
  Introduction rewritten accordingly. CPU-only HPA arm's in-report label
  changed from "(Kubernetes default)" to "(CPU baseline)" everywhere,
  including inside the regenerated chart legends (see below) -- it uses
  the same 260% target and 2-8 bounds as every other arm, never called a
  stock/default configuration.
- **Plain names**: `eval_final`/`stress_explore` -> "held-out evaluation"/
  "stress test" throughout prose, table headers, and figure captions;
  `eval_cpu_step`/`eval_memory_ramp`/`eval_mixed_burst` and their
  `stress_*` counterparts -> "CPU step"/"memory ramp"/"mixed bursts" (and
  "CPU bursts" for the stress CPU profile). Two data-table source files
  were renamed off their `eval_final`-bearing names:
  `tables/eval_final_primary_metrics.tex` ->
  `tables/held_out_evaluation_primary_metrics.tex`;
  `tables/eval_final_secondary_metrics.tex` and
  `tables/stress_explore_descriptive_metrics.tex` were deleted (superseded
  by hand-written, trimmed appendix tables -- see below).
- **Official structure**: Chapter 1 now has exactly Objectives / Background
  and Literature Survey / Problem Statement / Motivation / Basic Concepts /
  Existing System and Its Limitations / Proposed Approach / Organization
  of the Report. Chapter 2 retitled "PPO-BASED MULTI-METRIC AUTOSCALING
  SYSTEM FOR KUBERNETES" and reorganized under Proposed System / Working
  Methodology / **Standards (NEW)** / System Details -> Software, Hardware
  / Evaluation Protocol / Demonstration Interface Design, matching
  `official_format.pdf`'s own 2.1-2.4.2 numbering exactly where it
  specifies one (2.3 Standards, 2.4.1 Software, 2.4.2 Hardware). Chapter 3
  section 3.3 is literally titled "List of Components and Their Cost" per
  spec. Chapter 5 retitled "CONCLUSION \& FUTURE WORKS". Chapter 6
  Appendix, Chapter 7 References unchanged in role. **New Chapter 8
  Biodata** (`chapters/08_biodata.tex`) after References: one page per
  student (forced `\newpage`), photo-placeholder box (no real photos
  exist), Name, "Mobile Number: [to fill]", real VIT-AP e-mail, "Permanent
  Address: [to fill]", in the order given in the task spec.
- **Standards section (NEW)**: nine short paragraphs, each on a standard
  actually exercised by this repo's own code: Kubernetes `autoscaling/v2`
  HPA API; OCI/Docker image format; Prometheus exposition format and
  PromQL; HTTP/1.1; YAML and JSON (RFC 8259); ISO 8601 UTC timestamps;
  SHA-256 (FIPS 180-4, freeze manifest); Gymnasium environment API; IEEE
  citation style.
- **Software/Hardware tables**: already had real, correctly pinned
  versions from a prior session; this session added the missing `httpx`
  (demo UI, transitive via `loadgen.traffic_generator`, pinned 0.28.1) and
  verified every other pinned version against the real requirements files
  (`requirements/base.txt`, `requirements/rl.txt`,
  `demo_ui/requirements-ui.txt`) via a dedicated research pass -- all
  confirmed correct, no changes needed to the existing numbers.
- **Chapter 4 reorder**: now Experimental Setup; Training Campaign;
  Held-Out Evaluation; Exploratory Stress Test; **Demonstration Interface
  (NEW)**; Comparative Analysis Against Related Work; Discussion;
  Limitations -- matching the spec's required order.
- **Demo Interface section (NEW, Chapter 4)**: written from a dedicated
  research pass that read the *current* `demo_ui/` source fresh (post this
  session's own earlier 4 UI-hardening phases, commits up to 86353bd),
  not from any older description. Covers: the five pages (Experiment,
  Live Details, Comparison, Training, History); the Experiment page's four
  dropdowns with their *exact* option strings (Traffic pattern literally
  includes "Periodic (wave)" and "Ramp + bursts", not the shorter forms a
  casual description might assume); the "Live cluster"/"Sample data
  (offline)" toggle and why offline mode exists; the 36 demo profiles
  (verified against `configs/traffic_profiles/v2_demo/` and
  `scripts/generate_demo_profiles.py` -- 36 is exactly right: 6 patterns x
  3 resources x 2 loads, 8 minutes/480s each, Normal~18 RPS all resources,
  High~65 RPS CPU/Mixed and ~40 RPS Memory, all confirmed against real
  generated profile files, not assumed); Live Details' four charts and the
  PPO decision-explanation panel's exact fields (including the literal
  tie-rule caption strings); the Comparison page's real computed-metrics
  list (confirmed against `comparison.py`'s `_METRIC_ROWS` post-fix) and
  its exact UI caption "Demo runs illustrate behaviour; they are not
  evaluation evidence." (quoted verbatim, semicolon and British spelling
  preserved); the safety guards (controller-lock/training/evaluation/
  unexpected-HPA refusal, `INTERRUPTED (stale)` status, `invalid_no_traffic`
  status, the manual RECOVER control). Six screenshots
  (`docs/report/ref/ui/ui_1.png`-`ui_6.png`) included with proper figure
  captions. **The task prompt's own guessed ui_5/ui_6 mapping was WRONG and
  was corrected**: by visually inspecting all six images, ui_5 is actually
  the **Comparison** page (sidebar shows "Comparison" selected, HPA/PPO
  overlaid legend) and ui_6 is actually the **Training** page (sidebar
  shows "Training" selected, "Sessions completed: 48/48"), the reverse of
  the prompt's assumption -- the report's figure order/captions use the
  verified real content, not the prompt's guess.
- **Fact corrections**:
  - Explained variance: already correct in the prior session's text
    (0.0125 first update; mean 0.703 over last five updates/sessions
    43-47; peak 0.799) -- independently reconfirmed this session by
    computing directly from
    `persistent-data/rl_training_v2/final_min2/ppo_update_metrics.jsonl`
    (48 rows; first=0.012548..., last-5 mean=0.703274..., peak=0.799341...
    at the very last/47th update). No change needed.
  - Grafana disabled during training AND evaluation: the prior session's
    report only stated this for evaluation. This session found direct,
    dated confirmation that Grafana was also at 0 replicas during the
    training-campaign launches themselves
    (`persistent-data/traffic/v2_phase4e_launch_report.md`,
    `v2_phase4g_launch_report.md`: "Grafana already 0" / "Grafana 0") --
    now stated as a confirmed fact for both campaigns in Chapter 4
    (Training Campaign section and Limitations), not just evaluation.
  - Externally-killed evaluation re-run / host restart during evaluation:
    still **[CHECK]**, same as the prior session left it, but refined with
    a real, closer finding -- a dedicated research pass found that the
    held-out evaluation campaign's *first* live execution did have one
    attempt that crashed before completion with no recorded ledger row,
    documented in
    `persistent-data/evaluation_v2/eval_final_prefix_bug_archive_20261113/README.md`,
    caused by two *internal software defects* (an unset HTTP client
    reference; a wrong keyword argument in the cluster-restore function),
    not by an external kill signal -- both were fixed and the campaign
    restarted cleanly under corrected code (the 27 runs reported
    throughout the chapter). This is disclosed in Chapter 4 Section 4.3
    as a confirmed fact, while the externally-killed-signal and
    evaluation-time-host-restart possibilities remain explicitly
    **[CHECK]** (unconfirmed) exactly as the prior session left them --
    the only host-restart record anywhere in the repo is still the
    earlier, unrelated Phase 4B-R4C pre-training diagnostic event.
  - Correction-log dividing line: "CORR-55 onward are final-system-only"
    was independently re-verified against the full log
    (`docs/experimental_design_corrections.md`) this session and confirmed
    accurate -- CORR-55 through CORR-65 are unambiguously final-system
    design/evaluation entries; CORR-1-26 are pre-RL calibration, CORR-27-48
    are the foundational/original MDP contract and a since-superseded
    dynamics-calibration side-track (predates any V1/V2 generational
    split -- the "V2" in CORR-27's title is a *spec-draft* version number,
    unrelated to the repo's later file-suffix split), and CORR-49-54 exist
    only as code-comment references, never written up as entries in the
    `.md` file at all (a pre-existing documentation gap, noted in the
    appendix text rather than silently assumed away).
- **CORR appendix table**: the old full CORR-1-through-65 longtable
  (every entry, including the pre-V2/pre-RL calibration ones) is replaced
  by a short, plain-language table of only the 11 final-system entries,
  CORR-55 through CORR-65.
- **Tables/figures readability (no landscape, no tiny tables)**: the two
  `\begin{landscape}` `\tiny` appendix tables (13-16 columns each) are
  GONE -- replaced with two portrait `\small` tables, rows grouped by
  profile (`\multicolumn` sub-headers, not repeated profile names), short
  arm labels (HPA C+M / HPA CPU / PPO) with a legend line, only the columns
  the spec names (mean p95, SLA@1000, SLA@300, max per-pod memory, scaling
  actions); error rate, pod restarts, and reversals (all exactly 0 in
  every run of both campaigns) are called out as a sentence instead of
  three all-zero columns. `\usepackage{rotating}`/`pdflscape}` are still
  loaded in `main.tex` but no longer used anywhere (harmless). Eight
  single-panel training-curve figures were regrouped into two 2x2
  subfigure floats; three eval box plots into one 3-up subfigure float;
  two stress box plots into one 2-up float; several same-campaign
  over-time figures paired into 2-up floats (CPU-step replicas+p95;
  memory+mixed-bursts p95; PPO policy-map+tie-rule-frequency; stress
  replicas-over-time+replica-minutes) -- both for the spec's own
  "group near-identical plots as subfigures" instruction and for page
  budget. Every box-plot caption states "log-scaled y-axis" explicitly.
- **Figure/diagram regeneration** (read-only against `persistent-data/`
  and `configs/`, writes only under `docs/figures/`, no code file edited
  on disk): two real content bugs were found baked into pre-rendered
  figure files as literal text, which a plain `.tex` edit cannot fix:
  (1) `docs/figures/diagram_experiment_pipeline.dot` literally had "V1"
  and "V2 design" boxes drawn into the pipeline diagram -- this Graphviz
  source was edited directly (it is a report diagram source, not
  scientific code/data) to a single training -> held-out-evaluation ->
  stress-test pipeline, and the PDF/PNG/SVG were regenerated with `dot`;
  (2) several matplotlib-generated chart PDFs (replica/latency-over-time
  figures, the cost-vs-latency scatter) had "HPA CPU-only (Kubernetes
  default)" and "eval_final (filled)"/"stress_explore (hollow)" baked into
  their legends as literal vector text, sourced from a label dict in
  `scripts/paper_style.py` and a hardcoded legend string in
  `scripts/generate_v2_phase6e_figures.py`. Per the task's explicit
  instruction not to touch code, **no script file on disk was edited**;
  instead, a throwaway runner (in the scratchpad, not committed) imported
  these already-existing, repo-approved figure-generation functions and
  monkeypatched the one label dict entry and the legend string *at
  runtime only*, then called the exact same plotting functions the repo
  already uses, which write only to `docs/figures/` (as documented in
  `generate_v2_phase6e_figures.py`'s own docstring) -- this is disclosed
  here plainly as the one place this session's work touched something
  outside `docs/report/`, and it was a label-string-only fix, not a change
  to any score, metric, or finding.
- **Page count**: recomputed after the numbering-scheme change (not
  reused from before it) -- final count is **75 pages**, at the top edge
  of the 65-75 target range, achieved via: subfigure grouping (above),
  shrinking demo-screenshot and diagram widths, singlespacing + tighter
  itemsep in the References chapter (26 entries now fit in 2 pages instead
  of 3), and removing one appendix code listing ("Trainer Session Loop")
  whose content was already fully covered in Chapter 2 prose (Session/
  Checkpoint/Resume Design, Freeze Manifest) -- an abridged exception
  block in the "Evaluation Runner Core" listing and the input-validation
  block in the "Safety-Aware Tie-Break Selector" listing were also
  shortened for the same reason (prose elsewhere already covers it).
- **Compile**: `cd docs/report && tectonic -X compile main.tex --outdir=build`.
  Clean: 0 fatal errors, 0 "??" anywhere in the compiled PDF text, no
  undefined-reference or multiply-defined-label warnings, largest overfull
  `\hbox` is 36.9pt (well under the ~50pt guidance). One persistent,
  harmless warning: "Object @page.1 already defined" from `xdvipdfmx`
  (a hyperref/tectonic destination-naming quirk, not a content or
  page-numbering defect -- the actual rendered page numbers were spot-
  checked against the TOC/List of Tables/List of Figures `\pageref`
  entries, including the title page being unnumbered and the Certificate
  page showing "1", and all match).
- Final PDF copied to `docs/report/Capstone_Report.pdf` (75 pages).

## Residual "V1"/"V2"/"Kubernetes default" grep hits (all reviewed, all
## justified exceptions, NOT violations of the one-system-only framing)

Running `grep -niE "v1|v2|generation|redesign|eval_final|stress_explore|kubernetes.?default"`
over the compiled PDF text and over the `.tex` sources outside
`\begin{lstlisting}...\end{lstlisting}` blocks returns only:
1. **Real, on-disk repo filenames/object names** cited in prose for
   documentation accuracy, never renamed because doing so would
   misdescribe the actual artifact: `reward_v2.py`, `environment_v2.py`,
   `rl_v2.yaml`, `reward_v2.yaml`, `hpa_v2_cpu_memory.yaml`,
   `hpa_v2_cpu_only_min2.yaml`, the Kubernetes objects
   `demo-app-hpa-v2`/`demo-app-hpa-v2-cpu`, the real data path
   `persistent-data/rl_training_v2/final_min2/ppo_update_metrics.jsonl`
   (this exact path is also the one the task prompt itself names for the
   explained-variance computation), and the real config directory
   `configs/traffic_profiles/v2_demo/`. None of these describe "two
   generations" of the project -- they are just the real names of files
   this project actually has.
2. **The literal Kubernetes API string `autoscaling/v2`** in the Standards
   section and the two HPA-manifest listings -- this is the Kubernetes
   project's own API-version identifier, required by the task's own
   instruction to document "Kubernetes autoscaling/v2 HPA API" as a
   standard.
3. **Listing captions and YAML content inside `\begin{lstlisting}` blocks**
   (e.g. `RewardConfigV2`, `compute_reward_v2`) -- explicitly permitted by
   the task spec's own wording ("except inside verbatim code listings
   where identifiers like RewardConfigV2 are fine").
No hit anywhere asserts, implies, or narrates two versions, phases,
generations, or a redesign of the project.

## Open [CHECK] items (carried over / refined this session)

1. **Externally-killed evaluation or training re-run**: not confirmed.
   Closest located events: the training process at session 4 self-
   terminated on the recorded `zero_ready_replicas` condition (not an
   external kill); one held-out-evaluation attempt crashed on an internal
   software defect (not an external kill) before the campaign restarted
   cleanly. No external-kill-signal record was found anywhere in the
   repository for either campaign.
2. **Host restart during the evaluation campaign specifically**: not
   confirmed. The only host-restart record anywhere in the repository
   remains the earlier, unrelated Phase 4B-R4C pre-training diagnostic
   event, not `eval_final`/`stress_explore`'s own ledgers or reports.

Both are disclosed honestly as unverified in the report text (Chapter 4,
Section 4.3 and the Limitations list) rather than asserted as fact.

## Commits

Committed to `main` in per-chapter/per-concern chunks (see `git log --
oneline -- docs/report/`), no push, no Co-Authored-By trailer, matching
this repo's existing convention for report-build commits.
