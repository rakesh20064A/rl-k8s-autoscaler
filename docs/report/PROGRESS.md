# Capstone report build progress

## Status: COMPLETE (compiles cleanly, in page-count range)

- Front matter: title page, certificate, acknowledgements, abstract, TOC,
  list of figures, list of tables, list of abbreviations — DONE.
- Chapter 1 Introduction — DONE.
- Chapter 2 Proposed System and Methodology — DONE.
- Chapter 3 Cost Analysis — DONE.
- Chapter 4 Results and Discussion — DONE.
- Chapter 5 Conclusion and Future Work — DONE.
- Chapter 6 Appendix — DONE.
- Chapter 7 References — DONE.

## Build

- Compiler: tectonic (installed via `brew install tectonic`).
- Build command: `cd docs/report && tectonic -X compile main.tex --outdir=build`.
- Final PDF copied to `docs/report/Capstone_Report.pdf`.
- Final page count: **75 pages** (front matter roman i-x, body arabic 1-65),
  within the 65-75 target range.
- Per-chapter page ranges (physical pages, arabic numbering restarts at 1
  after front matter): front matter 1-10 (10pp); Ch1 Introduction 11-20
  (10pp); Ch2 Methodology 21-37 (17pp); Ch3 Cost Analysis 38-39 (2pp);
  Ch4 Results and Discussion 40-57 (18pp); Ch5 Conclusion 58-59 (2pp);
  Ch6 Appendix 60-72 (13pp); Ch7 References 73-75 (3pp).
- Compile log is clean: 0 fatal errors, 0 unresolved "??" cross-references,
  0 overfull \hbox warnings wider than 50pt (the few remaining overfull
  boxes are all under ~30pt, in the List of Figures/Tables and a couple of
  table cells, and were left as acceptable per the task's own guidance on
  minor warnings).
- Key fixes applied during the build/polish pass (for anyone resuming or
  rebuilding): removed `amssymb` (conflicts with `newtxmath`'s `\Bbbk`);
  changed `\newcommand{\cite}`->`\renewcommand{\cite}` (report class
  already defines `\cite`); dropped `language=yaml` from appendix
  `lstlisting`s (the `listings` package has no built-in YAML language);
  added `\allowbreak` after `/` and `\_` document-wide (via a redefinition
  of `\_`) and switched several wide tables to `p{}`-wrapped columns or
  `\resizebox{\linewidth}{!}{...}` to eliminate margin overruns; tightened
  figure/table/float spacing and switched figures from `[H]` to `[htbp]`
  placement, reduced `04_results.tex` figure widths, single-spaced code
  listings and the CORR-log longtable, and reduced `\tocdepth` to 1, to
  bring the original 87-page draft down into the 65-75pp target range.

## Open [CHECK] items (intentionally left, per explicit instruction)

1. **Externally-killed training re-run**: not confirmed in repo docs.
   Closest located event: the Phase 5E report's discovery that the
   training process (PID 26490) had disappeared, which the campaign's own
   metadata attributes to a self-terminating `zero_ready_replicas` stop at
   session 4, not a confirmed external kill signal. Flagged as [CHECK] in
   Chapter 4 ("The Two Zero-Ready-Pod Stops" and the Limitations list).
2. **Host restart during evaluation**: not confirmed in repo docs. Found a
   host-restart-related data-loss note in an unrelated, earlier
   pre-training diagnostic validation run (Phase 4B-R4C), not the final
   `eval_final`/`stress_explore` evaluation campaigns. Flagged as [CHECK]
   in Chapter 4 Limitations.

Both are disclosed honestly as unverified in the report text rather than
asserted as fact; left as-is per explicit coordinator instruction not to
pursue further.

## V1 figures note

`docs/archive/figures_v1/` is confirmed genuinely empty (no git history
ever populated it). The actual V1 evaluation figures were located at
`results/figures/*.png` (added in the project's first commit) and two of
them are embedded in Chapter 4 (`replicas_over_time_full_runs.png`,
`p95_latency_over_time_full_runs.png`).

## Commits

Committed to `main` in per-chapter chunks (see git log), no push, no
Co-Authored-By trailer.
