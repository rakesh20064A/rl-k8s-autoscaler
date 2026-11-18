Date: 2026-11-18

Prompt summary: Build `docs/presentation/Capstone_Final_Review.pptx` (plus a PDF export) from the official `docs/presentation/ref/Capstone_PPT_Template.pptx` template using python-pptx in a separate venv, pulling every number/figure/table from `docs/report/`, `docs/figures/`, `docs/tables/`, and `README.md`, describing only the final system (no V1/V2 language), with template layouts/fonts/logo/slide-numbers preserved, speaker notes on every slide, and a visual + text/notes verification pass before committing.

## Output

- `docs/presentation/Capstone_Final_Review.pptx` (33 slides)
- `docs/presentation/Capstone_Final_Review.pdf` (rendered export, same 33 slides)
- `.gitignore` updated to exclude `docs/presentation/ref/` and `docs/report/ref/*.pdf`
- Committed to `main` as "Final review presentation" (no attribution trailer, per repo instructions)

## Method

1. Installed `python-pptx` into `/tmp/ppt-venv` (separate from the project's scientific `.venv`), plus `pillow` for aspect-ratio-preserving image placement.
2. Read the template's actual slide/layout/placeholder structure via python-pptx before building anything (16 slides, 12 slide layouts; content slides 3-14 use the "Blank" layout with four hand-placed shapes per slide: chapter heading, sub-heading/body textbox, slide-number field, VIT-AP logo picture).
3. Read every content source end-to-end: `README.md`; `docs/report/chapters/01_introduction.tex`, `02_methodology.tex`, `03_cost_analysis.tex`, `05_conclusion.tex`, `07_references.tex`; `docs/tables/*.md` (T1-T4, eval_final_primary/secondary_metrics, stress_explore_descriptive_metrics, training_summary); `docs/figures/FIGURES.md`; and the HPA manifests under `k8s/hpa/` (confirmed the 260% CPU target and 2-8 replica bounds are identical across all three arms, including the CPU-only arm).
4. Wrote one build script (python-pptx + raw OOXML/lxml manipulation for bullet/table styling) that:
   - Edits slides 1 (title), 2 (outline, left verbatim), and single-slide sections (3, 4, 5, 7, 9, 11) in place, replacing placeholder text only, never moving/resizing template shapes.
   - Duplicates slides for sections that need more than one slide (6, 8, 10, 12, 13, 14, 15) via a `clone_slide()` helper that deep-copies every shape, including re-relating the embedded logo picture's image part (not just the XML) so the logo stays intact on every duplicate.
   - Uses one pristine, never-edited clone (`fresh_slide()`) as the source for every duplicated slide, instead of chaining clones off already-edited slides — an earlier version of the script cloned from already-content-laden slides and leaked stale captions/pictures onto later slides; this was caught during the visual render pass and fixed by always cloning from the pristine source.
   - Builds tables (python-pptx `add_table`, manually styled: white fill, no banding, bold headers, 14-16pt, arm-name cells coloured by the fixed PPO/HPA-CPU+Memory/HPA-CPU-only palette) and figures (aspect-ratio-preserving placement, 1-line captions) directly from the real `docs/figures/*.png` files and `docs/tables/*.md` numbers.
   - Builds 4 UI-screenshot placeholder slides (Experiment page, Live Details PPO, Live Details HPA, Comparison/Training) as explicit "[UI screenshot to be added]" boxes sized/positioned like a real screenshot, per the constraint that those screenshots don't exist in this repo yet.
   - Writes 4-8 sentence speaker notes on every one of the 33 slides.
   - Re-orders the final slide sequence via the presentation's `sldIdLst` (clones are appended at the end by python-pptx; the script tracks desired order by `slide_id` and rebuilds the list once at the end).
5. Rendered to PDF with `soffice --headless --convert-to pdf`, to PNG per slide with `pdftoppm`, and visually read every one of the 33 slide images (two full passes: one caught the stale-caption cloning bug, the second confirmed all 33 are clean — logo visible, no overflow/overlap, figures readable, captions correct).
6. Extracted all slide text, table-cell text, and speaker notes via python-pptx and grepped case-insensitively for "v1", "v2", "generation", "redesign", "eval_final", "stress_explore", "kubernetes default" — 0 hits.
7. Re-rendered the final PDF into `docs/presentation/` (the one committed) and added `docs/presentation/ref/` and `docs/report/ref/*.pdf` to `.gitignore` (neither was previously covered).

## Content decisions worth recording

- Training-budget numbers (48 sessions x 256 steps/session = 12,288 timesteps) were cross-checked against `docs/tables/training_summary.md`'s `n_sessions_analyzed=48` and `docs/tables/T3_ppo_hyperparameters.md`'s `training_budget` field, and against the README — all three agree.
- All metrics tables (held-out evaluation, stress test) use the exact mean values from `docs/tables/eval_final_primary_metrics.md` and `docs/tables/stress_explore_descriptive_metrics.md`, rounded for slide readability; arm names are the plain story names ("PPO (multi-metric, 11 signals)", "HPA CPU+Memory", "HPA CPU-only"), never the internal table/campaign names.
- Bibliography: 14 of the report's 27 references were selected (Borg/Kubernetes, the three autoscaling-survey papers, RL/PPO/GAE fundamentals, Gymnasium, Stable-Baselines3, the three RL-for-resource-management papers most directly comparable, FIRM, and the three reliable-RL-evaluation/pre-registration papers), split 7+7 across two slides at 14pt.
- SDP Id and the supervisor's exact title were filled in per the task's literal text; `SDP Id: [to fill]` is the one remaining open item.

## [to fill] items

- `SDP Id:` on the title slide (left as `[to fill]`, per instruction).
- The 4 UI-screenshot placeholder slides in section 13 (Experiment page, Live Details - PPO, Live Details - HPA, Comparison/Training) are placeholders only — the user will supply the real screenshots separately; no screenshot was generated or faked.

## Verification results

- Render check: all 33 slides visually inspected (two passes); one real bug found and fixed (stale captions/figures leaking onto cloned slides from non-pristine clone sources); final pass clean.
- Text/notes grep for forbidden terms: 0 hits.
- Slide-number fields (`<a:fld type="slidenum">`) are genuine PowerPoint/LibreOffice auto-computed fields, not literal text, so they resolve to the correct position automatically regardless of slide reordering — confirmed by rendering (positions 8, 9, etc. showed "8", "9" correctly). On content slides the template's own number color (`B5A989` on a dark circle) is low-contrast by the template's own original design (confirmed against the unmodified template's own render) — not something introduced or alterable by this build.
