# Paper 1: section map and completeness (15 November 2026)

How each section of the final paper (ACM `acmart`) maps to the course's
required content, where its draft text already lives, and what is missing.
Status: **Drafted** = prose exists in `docs/` and needs only editing into the
template; **Partial** = some prose or numbers exist, pieces missing;
**Not started** = nothing beyond notes.

## Required content → where it lands

| Course requirement | Paper section(s) | Status |
| --- | --- | --- |
| Describe the model, implementation, assumptions and experimental setup | Method | **Mostly drafted.** Assumptions are scattered across five docs and need one consolidated subsection |
| Explain the V&V methods and why they are appropriate | Method (V&V subsection) | **Partial.** The methods are documented; the "why appropriate" argument is not written |
| Present and interpret model experiments and V&V results | Results | **Mostly drafted** for point-estimate results; `pooled_edu` pending, coverage not run, convergence trace not written up, no consolidated table |
| Discuss limitations, uncertainty, generalizability, future work | Discussion and Conclusion | **Partial.** Limitations are measured and drafted; uncertainty depends on coverage; generalizability and future work exist only as notes |
| Cite all sources, datasets, code and resources | References | **Partial.** 24 verified BibTeX entries for the matching literature; dataset, software and synthesis-method entries not yet collected in one `.bib` |

## Section by section

| Section | Status (≈ %) | Draft text already in | Missing |
| --- | --- | --- | --- |
| **Abstract** | Not started (0%) | n/a | Write last, once `pooled_edu` and coverage settle |
| **Introduction**: motivation, purpose, research question, V&V goals | Not started (≈ 10%) | Proposal motivation **[check: not in repo]**; plan's "why appropriate" short version; BPM PAY as evidence that uncontrolled attributes are the normal case | Rewrite around the findings: research question (when does a thin, heterogeneous origin need its own estimate?), V&V goals, contributions as reframed by the matching check |
| **Background**: prior work and context | Partial (≈ 35%) | `statistical_matching_reading.md` (related-work paragraph drafted; Beckman, Guo & Bhat, Ye 2009 checked against full text); `darien-microdata-literature` search notes | Beckman → Ye lineage for synthesis; max entropy (Wu 2018); small-area estimation link; Darién context (SNM counts, BPM, Pinzón-Espinosa 2026); non-probability sampling (Elliott & Valliant 2017; Meng 2018). Framing reads (La et al. 2025; Bijak ch. 4–5) not done |
| **Method: model design and implementation** | Drafted (≈ 80%) | `sprint2_simulation_design.md` (truth, selection, inputs), `sprint3_ipf_designs.md` (pooled/separate IPF, departures 1–4), `maxent_design.md`, `sprint4_pooled_edu.md`, the hierarchical block in the plan | A hierarchical design write-up as a doc (it exists only in the plan); a design-overview table (each design = which conditional independence); IPF spec with notation and stopping rule |
| **Method: assumptions** | Partial (≈ 50%) | "Assumptions added" lists in Sprint 2, Sprint 3 and max-entropy docs; external-validity draft in Sprint 2 doc | Consolidate into one prose subsection; add the hierarchical/`pooled_edu` assumptions and ACS-as-sending-population |
| **Method: experimental setup** | Drafted (≈ 80%) | Sprint 2 doc (five γ, calibration targets), Sprint 3 and Sprint 4 crossover docs (sizes, replicates, two metrics), run log | Fix stale male target (0.715 → 0.745); one table of all runs and parameters |
| **Method: V&V methods and why appropriate** | Partial (≈ 40%) | Verification tables in README and each doc; plan's justification sketch (no enumerated reference → manufacture a truth; convergence contested in the literature → verify it; coverage holds reported uncertainty to account) | Write the justification; name the three V&V techniques explicitly (code verification, convergence verification, validation against planted truth, plus coverage); map Rässler's four validity levels onto margins / SRMSE / MI |
| **Results: experiments** | Mostly drafted (≈ 65%) | Sprint 3 doc (pooled vs separate, failures, crossover fine run), Sprint 4 crossover doc (heterogeneity curve, ρ = −0.92, slope ≈ −2.3), plan's hierarchical findings and max-entropy MI results; two figures in `docs/figures/` | `pooled_edu` results; consolidated design × metric × origin × γ table; figure polish; stand-alone captions |
| **Results: V&V** | Partial (≈ 50%) | Test tables (code verification), max-entropy MI = 0 confirmation, UNHCR/SNM reality and plausibility checks with numbers | Coverage of intervals; convergence trace vs Ye et al. (2022) |
| **Discussion and Conclusion** | Partial (≈ 30%) | Direction-of-borrowing headline; measured limitations (education gap, age shape); external-validity draft; journal-version idea (Fréchet bounds); Paper 2 bridge (BPM roles) | Interpretation for other corridors (generalizability); uncertainty discussion (needs coverage); future-work list; conclusion |
| **References** | Partial (≈ 40%) | `statistical_matching.bib` (24 entries, verification table); "References to add" in `maxent_design.md` | One master `.bib`; datasets (ACS PUMS via Census Bureau; UNHCR 2023 BPM in its official form; SNM Cuadros 002/003); software with versions; synthesis methods (Auld et al., Wu 2018 author list flag); open flags (Marella & Pfeffermann venue, Beckman journal pages, two `[citation]` marks) |
| **Appendix** | Partial (≈ 30%) | Selection function in full (Sprint 2 doc), UNHCR/SNM aggregates (plan Sprint 4), test tables | Run-log summary; per-origin supplementary tables; hand-computed verification case; aggregates-only UNHCR table formatted |

The percentages are rough judgements of how much final prose exists, not
measurements.

## Critical path

1. `pooled_edu` results (running now).
2. Decide on coverage: run it, or cut it with a stated reason. It is the one
   course-relevant V&V technique with nothing yet, and the Discussion's
   uncertainty paragraph depends on it.
3. Convergence trace write-up (small; the plan calls it the item not to cut).
4. Consolidated results table, then Method assumptions and the V&V
   justification.
5. Background, Introduction, Discussion, Conclusion, Abstract, in that order.
6. Master `.bib` and assembly in `acmart`.
