# Statistical matching: reading note and positioning (after Sprint 4)

Reading note for the related-work and discussion sections. It answers the
three questions set in the plan (*The statistical matching check*): has the
matching literature already measured what conditional independence costs, can
the uncontrolled joint be bounded instead of assumed, and do the two
literatures cite each other. Every reference was checked against a publisher,
repository or index page; the status of each is in the table at the end, and
BibTeX for `acmart` is in `docs/statistical_matching.bib`.

## The setup in matching terms

Statistical matching combines two samples of one population that share some
variables but never observe the variables of interest together: file A holds
(X, Y), file B holds (X, Z), and the target is the joint of (X, Y, Z)
[@dorazio2006book; @rassler2002book]. The joint of Y and Z given X is not
identified. The default is the **conditional independence assumption (CIA)**,
Y ⊥ Z | X, which picks one point out of the set of joints consistent with the
two files; it cannot be tested from A and B alone [@dorazio2006book;
@rassler2004; @leulescu2013]. The alternative the field developed is to report
the whole set (the *uncertainty space*) and measure its size
[@dorazio2006jos; @conti2012; @conti2017].

Our study is not the canonical A/B layout, because the seed observes origin,
age, sex and education together. The mapping that matters is per origin:

- **Population margins** play file B: origin by age and by sex, no education.
- **The seed** plays file A, but for a thin origin it is a handful of people,
  so the directly observed (origin, education | age, sex) joint is mostly
  noise or empty. What the seed knows well is the *pooled* education
  distribution given age and sex, dominated by Venezuela.
- What is unidentified from "margins + pooled seed" is exactly the
  education-by-origin association within age-sex cells. Each design resolves
  it with a different conditional independence:

| Design | Assumption it imposes on the uncontrolled attribute |
| --- | --- |
| Pooled IPF | E ⊥ O given (A, S): the CIA with origin as Y, education as Z, age-sex as X |
| Pooled max entropy | E ⊥ (O, A, S) within each age stratum |
| Separate max entropy | E ⊥ (A, S) given O within each age stratum: a CIA in the other direction |
| Separate IPF | No CIA; uses the origin's own seed, so noise and empty cells instead of bias |
| Hierarchical (pool target) | Shrinks the origin's seed toward the pooled-IPF CIA point by κ |
| Hierarchical (own target) | Shrinks toward the separate-max-entropy CIA point by κ |

So the designs compared in Paper 1 are, in matching language, **different
CIA points plus two ways of interpolating between a CIA point and direct
data**. That is the sentence for the method section.

A second non-identified component has no counterpart in the classical setup:
the planted SES selection (γ) makes the seed's education distribution differ
from the population's *even where the seed is large*. The classical theory
assumes both files are representative. Marella and Pfeffermann extend
matching to informative sampling and non-ignorable nonresponse
[@marella_pfeffermann_nr], but they model the response mechanism
parametrically; nothing in the margins can detect γ, which is why calibration
is exact at every γ while the university share moves by about ten points.

### What the book's appendices support (D'Orazio, Di Zio and Scanu 2006, App. A–D)

Checked 2026-11-15 against Appendices A–D only (book pp. 205–222); the main
chapters were not available. Page numbers are the book's.

- **Non-identification, formally.** Appendix A gives Rubin's (1974)
  factorization for three variables where X1 and X2 are never observed
  together and X3 is observed whenever X1 is: θ_{1|3} is estimable from
  complete cases, θ_{2,3} with missing-data methods, and the association of
  X1 and X2 given X3, θ_{1,2|3}, "is inestimable from the observed data set"
  (pp. 207–208). The authors note the factorization is "particularly
  helpful, in particular for the statistical matching problem" (p. 208).
  This is the citable statement behind "the joint of Y and Z given X is not
  identified" above; with origin as X1, education as X2 and age–sex as X3,
  θ_{1,2|3} is exactly the origin-by-education association within age-sex
  cells that the designs disagree about.
- **The CIA as a loglinear model.** Appendix B writes conditional
  independence of X1 and X2 given X3 as the loglinear model with only the
  λ13 and λ23 interactions (p. 212), and gives IPF as the ML fit, starting
  from a table "with an interaction structure no more complex than the fixed
  loglinear model" (pp. 212–213). Read the other way, this is our design
  point: IPF keeps the starting table's interactions, so **the seed chooses
  the model**. A pooled seed has no origin–education interaction, so pooled
  IPF fits the CIA model E ⊥ O | (A, S); a per-origin seed fits whatever
  interactions that origin's few records happen to show. Use this to state
  the table in *The setup in matching terms* in loglinear language.
- **Why γ is a different problem (missingness mechanism).** Appendix A
  defines MCAR, MAR and MNAR and says inference from the observed part alone
  is appropriate only under MCAR or MAR (plus distinctness: "ignorable")
  (p. 206). Our planted selection depends on SES, which is correlated with
  education, the variable the margins do not carry; relative to the
  population the seed is therefore selected non-ignorably on the attribute of
  interest. That is the formal reason calibration to margins cannot detect or
  correct γ (last paragraph of *The setup in matching terms*;
  [@marella_pfeffermann_nr]).
- **Calibration guarantees only the controlled totals.** Appendix D:
  calibration (GREG, raking-type) weights are the design weights changed
  "as little as possible" so that the weighted sample reproduces known
  auxiliary totals exactly (pp. 221–222); the model only affects efficiency
  and the guarantees are design-based (p. 221), which presumes known,
  positive inclusion probabilities (p. 220). Our seed under γ has neither
  known inclusion probabilities nor a probability design, so "calibration is
  exact at every γ" is a property of the method, not evidence about
  education. One sentence in the methods section.
- **Distance functions (App. C).** Gower distance and "donation classes"
  (pp. 216–217) are the matching analogue of Beckman et al.'s D(p, c) with a
  0–1 loss (exact match within a cell). Not needed for Paper 1 beyond that
  link.

**Claims in this note that only the main chapters can confirm:** the
A/B-file definition of the matching problem and the CIA statement with its
untestability (cited to the book in *The setup in matching terms* and in the
related-work draft); Singh et al. (1993) as cited in the book (Q1); and
anything about the uncertainty space, Fréchet bounds or logical constraints
if cited to the book rather than to [@dorazio2006jos]. Until the chapters
arrive, cite Appendix A (pp. 207–208) for non-identification and the JOS
paper for the uncertainty space.

## Q1. Has the field already quantified what the CIA costs?

**Yes, in general form.** The cost of matching under a false CIA has been
measured since Rodgers' evaluation of early merge files [@rodgers1984]. In
Rässler's simulations [@rassler2004, Table 2], nearest-neighbour matching,
which always imposes the CIA, estimates a true correlation of 0.80 as 0.58 or
0.46 with **zero** interval coverage when the CIA fails, and is unbiased when
it holds. D'Orazio, Di Zio and Scanu's real-data example puts 0.0597 of mass
on a cell whose true value is 0.0009, and puts positive mass on structural
zeros [@dorazio2006jos]. Singh et al. (1993, cited in [@dorazio2006book]) show
by simulation that auxiliary information reduces the error; not read here.

Rässler's four levels of validity line up with our battery and give its
vocabulary [@rassler2004]: preserving marginal distributions (level 4: our
exact calibration), the joint distribution (level 2: SRMSE), and the
correlation structure (level 3: mutual information). Her key point is that
"in the typical fusion situation only the fourth level can be controlled".
That is our calibration result stated in advance: every design hits the
margins, and that check is blind to which CIA it made.

**What stays ours.** The general cost is known, so the paper should not claim
the first measurement of it. What is new: (a) the setting is population
synthesis, where the CIA is made silently by the algorithm rather than chosen
by an analyst; (b) the "file A" is a thin, non-probability seed with planted,
non-ignorable selection, so CIA error and selection error can be separated;
(c) the groups are heterogeneous and unbalanced (one origin ≈ 80% of the
seed), which is what makes pooling expensive; and (d) the crossover sample
size at which the direct data beat the CIA point, measured against planted
truth. Reframe the claim from "we measure the cost of independence" to "we
measure, in a synthesis setting with a known answer, which conditional
independence each design makes and what it costs, and when an origin's own
data are worth more than the assumption".

## Q2. Can the uncontrolled joint be bounded instead of assumed?

**Yes, partly, and cheaply.** For categorical data the conditional Fréchet
bounds are [@dorazio2006jos; StatMatch `Frechet.bounds.cat`]

    max(0, p(y|x) + p(z|x) − 1)  ≤  p(y, z | x)  ≤  min(p(y|x), p(z|x)),

averaged over x, and the size of the uncertainty space is summarized by the
average bound width [@dorazio2006jos; @conti2012]. With origin as Y and
education as Z given an age-sex cell, write π = P(O = o | a, s) from the
margins and p = P(E = e | a, s) from the pooled seed. Dividing by π gives the
bounds on the quantity the designs disagree about:

    max(0, (p − (1 − π)) / π)  ≤  P(E = e | a, s, o)  ≤  min(1, p / π).

This explains the hierarchical results in one line. For the dominant origin
(π ≈ 0.8), with p = 0.30 the bounds are [0.125, 0.375]: pooled data and
margins alone already pin Venezuela down, because Venezuela *is* most of the
pool. For a thin origin (π ≈ 0.05) the same inputs give [0, 1]: the pool says
nothing about it, and every design's answer for Haiti, Cuba, Ecuador or
Cameroon comes from its assumption or its own few people. The same formula
with age as Y and education as Z, given origin, bounds the age-education
association that separate maximum entropy sets to zero.

Two limits. The bounds cover only the CIA component: they take the seed's
pooled education shares as correct, so they cannot cover the γ selection
error, which needs an assumption about selection (a sensitivity analysis over
γ, or the model-based route of [@marella_pfeffermann_nr]). And logical
constraints narrow them [@dorazio2006jos; @conti2012]: for example, a
structural zero such as no university degree under 18.

**Decision.** As the plan said, this is a fourth design for the journal
version: report an interval for the survey-only attribute per origin, narrowed
by constraints, with the CIA designs as points inside it. Not retrofitted into
the semester paper. One cheap use there: the bound widths per origin are a
diagnostic that predicts where pooling is safe (narrow, dominant origin) and
where it is pure assumption (vacuous, thin origins). That may be worth a
sentence in the discussion even without computing them.

## Q3. Do the two literatures cite each other?

**Hardly, as far as this check reached.** The synthesis reviews and method
papers checked do not frame the core synthesis step as matching:

- Chapuis, Taillandier and Drogoul's review [@chapuis2022] does not contain
  "statistical matching" or "data fusion" (verbatim check of about 40% of the
  text, the rest via an extracted summary). It does note, as a property of
  independent sampling, that "all unknown relationships between attributes are
  statistically independent in the generated synthetic population", which is
  the CIA under another name and without the literature behind it.
- Yaméogo et al. [@yameogo2021] mention data fusion only as future work.
- Beckman, Baggerly and McKay [@beckman1996] (checked 2026-11-15 against the
  full text of the Los Alamos TRANSIMS/TMIP report version, *Creating
  Synthetic Baseline Populations*; page numbers below are the report's, see
  the verification table) **cite no matching or data-fusion work**: the
  reference list (pp. 18–19) is travel-behaviour papers, the 1990 census
  documentation, Deming and Stephan (1940), Ireland and Kullback (1968),
  Little and Wu (1991) and Breiman et al. (1984). But they do **not** make the
  assumption silently. They assume that "all of the census tracts ... that
  contribute to a PUMA have the same correlation structure", measured by odds
  ratios (p. 5), and say that "without this or similar assumptions the
  estimation of odds ratios for individual tables is intractable" (p. 8).
  That is the non-identification, stated, and resolved by a homogeneity
  assumption, without the matching vocabulary. See *Beckman et al. as a
  precedent* below for what this means for us.
- Guo and Bhat [@guo_bhat2007] and Ye et al. [@ye2009] (checked 2026-11-15
  against full texts: the Guo–Bhat author manuscript and the Ye et al. TRB
  2009 conference paper; page numbers are those documents' own) also **cite
  no matching or data-fusion work** (reference lists: Guo–Bhat pp. 20–21,
  Ye et al. pp. 18–19). Both frame their contribution as extending Beckman
  et al. to control person-level as well as household-level margins
  (Guo–Bhat pp. 1, 7–8; Ye et al. pp. 4–5), not as anything about the
  uncontrolled attributes. Guo and Bhat describe the IPF step as one that
  "preserves the correlation structure observed in the sample households"
  of the seed area (p. 4), i.e. the Beckman assumption restated as a
  property of the algorithm rather than an assumption; Ye et al. do not
  discuss it.
- With these three in full text and Chapuis et al. partially, the claim can
  now be made for **the founding synthesis-method papers we examined**
  (Beckman, Guo–Bhat, Ye) and the review: none cites matching. Keep the
  scope qualifier; it is not a census of the field.

Where the literatures do touch is applied pipelines that use matching as a
**separate attribute-attachment step after synthesis**: Namazi-Rad et al.
build a synthetic population from census tables and then attach wealth from
the Bangladesh DHS by nearest-neighbour matching [@namazirad2017]; Hörl and
Balac attach travel-survey activity chains to a synthetic Paris population
[@horl2021] (that they call this step statistical matching was not confirmed
from the full text); and the microsimulation literature discusses the CIA
directly when imputing expenditure into income data [@decoster2020]. None of
these treats the uncontrolled attribute in the seed-based synthesis step
itself as a matching problem, nor reports its uncertainty space.

**So the bridge paragraph is warranted, with one correction**: synthesis
practice makes the CIA inside IPF or maximum entropy without connecting it to
the matching literature or its uncertainty results, and mostly validates
against margins, which (Rässler's level 4) cannot test it. It is not true that
the founding paper makes the assumption *silently*: Beckman et al. state it
(as homogeneity of odds ratios across tracts) and justify it as necessary.

### Beckman et al. as a precedent

What the full text shows, with report page numbers:

- **Seed and margins.** Tract (or block-group) margins from STF-3A; the seed
  is the multiway table built from the PUMS of the PUMA containing the tract
  (p. 2, p. 5). IPF estimates "maintain the
  same odds ratios as those in the sample table in the absence of any
  marginal information to the contrary", citing Little and Wu (1991) (p. 6).
  Whole PUMS households are then drawn in proportion to the fitted table,
  with a 0–1 distance so only exact matches are drawn (pp. 11–12); multiple
  draws are suggested to carry synthesis uncertainty (p. 12).
- **The assumption is our pooled-IPF assumption.** Every tract in a PUMA is
  fitted from the same PUMA seed, so tracts inherit the PUMA's odds ratios.
  Map tract to origin and PUMA to the pooled seed: for an attribute with no
  margin (our education), equal odds ratios across groups reduces to
  E ⊥ O | (A, S), the pooled-IPF row of the table above. Their "statistically
  correct" variant, a joint (m+1)-way fit with tract as a dimension whose
  combined odds ratios equal the PUMS (pp. 8–10), keeps the same homogeneity
  assumption; it "only marginally outperforms" fitting tracts one at a time
  (p. 16). Neither variant has a separate (tract-own) seed, because no tract
  microdata exist.
- **Attributes only in the sample.** Attributes outside the fitted table
  ride along on the drawn PUMS households. Their validation targets exactly
  those: household size, which STF-3A has but the fit does not use (p. 13),
  and the joint of household size and vehicles (pp. 13–15). They also tried
  imputing household size from the PUMS with CART and found it added
  variability but no improvement (pp. 15–16).
- **Planted-truth validation with heterogeneity.** Their second validation
  treats 22 San Francisco Bay Area PUMAs as pseudo-tracts and a 5% "super
  PUMS" drawn from all 22 as the seed (pp. 13–14): the same design as ours,
  with origin in place of tract. The worst errors are the three City of San
  Francisco PUMAs, known to differ (fewer vehicles), where vehicles for small
  households are overestimated and single-person no-vehicle households
  "greatly underestimated"; they were added "to see if a few heterogeneous
  census tracts in a PUMA would invalidate the procedure" (p. 14). That is
  our pooled-design bias for thin, dissimilar origins, observed in 1996 on
  three units. They then argue real tracts are more homogeneous than their
  pseudo-tracts (p. 14). Our setting is the case where that argument fails:
  origins are not more homogeneous than the pool, and the thin ones are the
  ones of interest.
- **Zero cells.** IPF keeps sample zeros (609 of 11,760 family cells nonzero,
  p. 15); adding 0.01 to empty cells "badly biases" the margins and is "not
  recommended" (p. 16). Relevant to our separate-IPF failures and to the
  hierarchical design, which fills empty cells by borrowing instead of by a
  constant.
- **Independence baseline.** Multiplying the margins (full independence) and
  drawing from the PUMS "badly biases" the margins and the household
  size × vehicles joint; "not recommended" (pp. 16–17). This is the
  independence baseline our maximum-entropy designs sit close to.

### What the two follow-ups add (Guo and Bhat 2007; Ye et al. 2009)

- **Zero cells, two answers.** Guo and Bhat treat cells empty in the seed
  but present in the margins ("incorrect zero cell values", p. 7) by capping
  IPF iterations, coarsening categories, or reporting where they occur
  (pp. 7, 9). In Dallas–Fort Worth, 388 of 3,372 block groups failed at
  household level and 151 at person level, all from small groups absent from
  the PUMA sample (p. 14). Their validation then **drops** those block groups
  and evaluates 62 Tarrant County block groups without the problem (p. 16):
  the same survivor bias we flag for separate IPF. Ye et al. instead
  **borrow** the zero cells' probabilities from the region-wide PUMS, capped
  at 1 / (households in the PUMA) because the zero itself is evidence the
  group is rare locally, and rescale the non-zero cells to sum to one
  (pp. 11–12). That is pooled borrowing for empty cells only, with an ad hoc
  cap in place of our κ: the closest precedent for the hierarchical design
  (which shrinks every cell, not just the empty ones, and estimates the
  amount by empirical Bayes).
- **Validation is against the fitted targets, not a truth.** Guo and Bhat
  score the synthetic population against the IPF-estimated tables (pp.
  16–19); Ye et al. against the "observed IPF-generated" person table with
  a χ² test that omits zero cells (pp. 14–16). Neither can see an error in
  the uncontrolled association, because the reference is itself the CIA
  point. Beckman et al.'s pseudo-tract design remains the only planted-truth
  check among the three.
- **Both misreport Beckman on zero cells.** Ye et al. write that Beckman et
  al. "recommended adding an arbitrarily small value (e.g., 0.01) to zero
  cells" (p. 11); Beckman et al. say tweaking "badly biases" the margins and
  "is not recommended" (Beckman p. 16). Guo and Bhat say Beckman compared
  tweaking against a maximum-iteration approach (p. 7); Beckman compared it
  against the untweaked basic algorithm (Beckman pp. 15–16). Cite Beckman
  directly on this point.

How to use it in the paper: cite Beckman et al. as the origin of the
assumption we test (same seed odds ratios for every subgroup sharing a seed),
and as the first planted-truth check of it. Our contribution relative to them
is to make the subgroup the object of interest, vary its size and its
distance from the pool, and name the assumption as a CIA so the matching
results (uncertainty space, bounds) apply.

Not in Beckman, contrary to what one might expect: no discussion of selection
or non-representative seeds (although they cite Little and Wu, whose title is
about target and sampled populations differing), and no reference to the
conditional independence assumption or to matching. Little and Wu (1991) is
worth reading for the γ (selection) component in its own right
[@little_wu1991].

## Positioning the Sprint 4 findings

1. **Shared empirical-Bayes κ collapses to pooling.** In matching terms the
   pooled distribution is the CIA point, and the data inform the
   origin-by-education association only where π is large (Q2). A shared κ is
   estimated mostly from the origin whose bounds are tight and which matches
   the pool by construction, so it says "pooling is fine" for the origin
   where pooling cannot be wrong. This is the matching non-identification
   showing up inside an estimator. It also echoes the Bayesian matching
   literature: when a parameter is not identified, the posterior follows the
   prior [@rassler2002book; @fosdick2016], and a hyperparameter estimated from
   the pool inherits the pool.
2. **Per-origin κ lands near the oracle.** This is a composite estimator in
   the small-area sense: the origin's direct estimate weighted against a
   synthetic (CIA) one by its own information. It is the closest our design
   comes to the auxiliary-information route [@fosdick2016], in which outside
   information picks a point inside the uncertainty space instead of the CIA.
3. **Direction of borrowing matters more than amount.** Matching already has
   this result in a different form: the choice of common variables X decides
   how large the uncertainty space is, and StatMatch's `Fbwidths.by.x` ranks
   candidate X sets by bound width. Borrowing across origins and borrowing
   across cells within an origin are two different CIAs (E ⊥ O | A, S versus
   E ⊥ A, S | O); κ only moves along the line from the seed to whichever one
   you picked. Picking the wrong CIA cannot be repaired by tuning κ. This is
   the cleanest link to the field and should be stated as such.
4. **Separate max entropy removes the bias but keeps no age–education
   association.** This is Rässler's level 3 failure, by construction: it is a
   CIA design, just with origin as the conditioning variable. Its MI of
   exactly zero is the analogue of D'Orazio et al.'s CIA estimate putting mass
   where the truth has none.

## Related-work paragraph (draft)

> Adding an attribute that no population margin controls is, in survey
> statistics, a statistical matching problem: one source carries the
> attribute, another carries the controls, and their joint is not identified
> [@dorazio2006book; @rassler2002book]. The standard resolution, conditional
> independence given the common variables, cannot be tested from the sources,
> and its cost when it fails is well documented [@rodgers1984; @rassler2004;
> @dorazio2006jos]. The field's alternative is to report the set of joints
> consistent with the data, through Fréchet bounds narrowed by logical
> constraints and auxiliary information [@dorazio2006jos; @conti2012;
> @fosdick2016]. Population synthesis has used matching to attach survey
> attributes after the fact [@namazirad2017; @horl2021], but the synthesis
> methods themselves make the same independence assumption inside the
> fitting step. Beckman et al. state it, as subgroups sharing the seed's odds
> ratios, and justify it as necessary for tractability [@beckman1996]; the
> reviews describe it as a property of independent sampling
> [@chapuis2022]; neither connects it to the matching results on what it
> costs or how to bound it. Later methods kept this structure, adding
> person-level controls and handling empty seed cells by coarsening,
> dropping or borrowing regional probabilities [@guo_bhat2007; @ye2009],
> and validated against the fitted tables rather than a known truth. We make
> that connection: each design we compare
> corresponds to a different conditional independence, and the planted truth
> lets us measure what each costs for subgroups that differ from the pool,
> the case Beckman et al. observed in three heterogeneous pseudo-tracts and
> set aside as rare.

## Verification status of references

| Key | Verified against | Note |
| --- | --- | --- |
| dorazio2006book | Appendices A–D full text (Wiley Online Library, pp. 205–222); Booktopia listing | John Wiley & Sons Ltd 2006, ISBN 0-470-02353-8 (= 978-0-470-02353-2), book DOI 10.1002/0470023554 from the appendix DOIs. **Main chapters not read**; see the appendices section for which claims still need them |
| dorazio2006jos | Full-text PDF (ISTAT) | JOS 22(1):137–157 |
| rassler2002book | Springer page | LNS 168, DOI 10.1007/978-1-4613-0053-3 |
| rassler2004 | Full-text PDF | Austrian J. Stat. 33(1&2):153–171 |
| conti2012 | Full-text PDF (SCB) | JOS 28(1):69–88 |
| conti2016 | RePEc | JASA 111(516):1715–1725, DOI 10.1080/01621459.2015.1112803 |
| conti2017 | RePEc handle | Commun. Stat. Theory Methods 46(2):967–994; title from search index, not read |
| fosdick2016 | Project Euclid URL and arXiv 1506.05886 | AoAS 10(4), DOI 10.1214/16-AOAS925; page range not confirmed |
| endres2019 | RePEc | JOS 35(3):599–624, DOI 10.2478/jos-2019-0025 |
| rubin1986 | RePEc | JBES 4(1):87–94 (NRC reference list says 86–94; RePEc used) |
| rodgers1984 | NRC (1991) reference list | JBES 2:91–102; issue not confirmed |
| leulescu2013 | Eurostat PDF cover | Methodologies and Working Papers, doi:10.2785/44822 |
| marella_pfeffermann_nr | Southampton ePrints PDF | Title and authors only; **venue and year not confirmed**, find before citing |
| chapuis2022 | JASSS page | 25(2):6, DOI 10.18564/jasss.4762 |
| beckman1996 | Full text of the Los Alamos TRANSIMS/TMIP report version (uploaded scan, 19 pp. + tables/figures); journal citation from RePEc | TR-A 30(6):415–429. Page numbers cited in this note are the **report's**; map to journal pages before quoting. DOI 10.1016/0965-8564(96)00004-3 not confirmed (Crossref blocked here) |
| guo_bhat2007 | Full text of author manuscript (uploaded); journal citation from Ye et al.'s reference list and repository/index titles | TRR 2014:92–101. Page numbers in this note are the manuscript's. DOI 10.3141/2014-12 not confirmed |
| ye2009 | Full text of the TRB 88th Annual Meeting paper (uploaded, "submitted for presentation only"); TRID record 881554 | Conference paper, not peer-reviewed journal; Xin Ye (not Peijun Ye). Paper number not confirmed |
| little_wu1991 | Title from library catalogue; volume and pages from Beckman et al.'s reference list | JASA 86:87–95; Beckman's list says "known marginals", the published title says "known margins"; issue/DOI not confirmed |
| yameogo2021 | JASSS page | 24(2):5, DOI 10.18564/jasss.4482 |
| namazirad2017 | UOW repository | CEUS 63:3–14; DOI not shown |
| horl2021 | ETH Research Collection, Wikipedia citation | TR-C 130:103291, DOI 10.1016/j.trc.2021.103291 |
| decoster2020 | IJM page | IJM 13(3):70–94, DOI 10.34196/ijm.00226 |
| jauslin2023 | arXiv page | DOI 10.1016/j.jspi.2022.12.003; venue inferred as J. Stat. Plan. Inference from the DOI, confirm |

Not used, because they could not be verified this session: Kadane (1978;
the NRC list gives the Treasury title "Statistical problems of merged data
files", and the 2001 JOS reprint was not confirmed), Moriarity and Scheuren
(2001), Okner (1972, NRC list gives *Annals of Economic and Social
Measurement* 1:325–362), Manski's partial-identification work, and Singh et
al. (1993). Wu et al. (2018) still carries its "[verify author list]" flag in
`docs/maxent_design.md`; the ACM page did not load here.
