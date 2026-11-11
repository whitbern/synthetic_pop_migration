# Crossover against heterogeneity, and on education alone (Sprint 4 draft)

Extends the Sprint 3 crossover (`docs/sprint3_ipf_designs.md`) in three
ways:

1. **A second thin group.** Cuba is added next to Cameroon, because the
   separate design failed for Cuba at every γ, even with 44–51 sample
   members.
2. **A second metric.** The crossover is located twice: on adult SRMSE of
   the sex × age band × education table, as before, and on the TVD of the
   adult education distribution alone.
3. **A heterogeneity axis.** Each crossover is plotted against a direct
   measure of how far the group is from the pool.

Code: `src/synthpop/crossover.py` (`origins` and `metrics` config keys,
`plot_heterogeneity`) and `src/synthpop/scoring.py` (`heterogeneity`).
Config: the `crossover` block of `configs/sprint2.json`. Output:
`data/sprint4_crossover/`.

## Heterogeneity measures

Both are computed in the truth (the flow) at each γ, on adults:

- **Pooled-limit TVD** (the x-axis). Run pooled IPF with the *whole truth*
  as the seed, as if the sample were perfect and infinite: the pool's
  sex × age × education table fitted to the group's own sex and age margins.
  Then take the TVD between that and the group's true adult table. This is
  the bias pooled estimation keeps however large the sample, so it is the
  quantity the crossover should depend on. Separate estimation pays variance
  to remove it. `test_pooled_limit_is_what_pooled_ipf_gives_with_the_truth_as_seed`
  checks that the function equals that IPF run's error.
- **Education TVD vs pool.** TVD between the group's adult education
  distribution and all adults'. This is simpler, and it ignores the sex and
  age structure.

**Prediction.** The crossover *n* falls as pooled-limit TVD rises: the
further a group is from the pool, the sooner its own sample pays off. The
Sprint 3 fine run already showed this ordering for Cameroon across γ, using
pooled SRMSE at small *n* as a stand-in.

## Results

*[ACS run: crossover table by origin × metric × γ, pooled-limit TVD,
`crossover_vs_heterogeneity.png`, and the per-origin crossover plots.]*

## Reproducibility

Cameroon's random stream is unchanged from Sprint 3, so its SRMSE results
here reproduce the fine run at the shared sizes. Cuba gets its own stream.
Failure rates and empty control cells are reported as before. The separate
design is never patched.
