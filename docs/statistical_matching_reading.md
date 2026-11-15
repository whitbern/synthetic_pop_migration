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
- Beckman, Baggerly and McKay (1996), Guo and Bhat (2007) and Ye et al.
  (2009) were **not** checked in full text (paywalled). The claim "the
  synthesis-method literature does not cite matching" needs that check before
  it goes in the paper; until then write "the synthesis reviews we examined".

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

**So the bridge paragraph is warranted**: synthesis practice makes the CIA
silently, inside IPF or maximum entropy, and then validates against margins,
which (Rässler's level 4) cannot test it.

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
> methods themselves, and their reviews [@chapuis2022], make the same
> independence assumption inside the fitting step without naming it, and
> validate against margins that cannot test it. We make the assumption
> explicit: each design we compare corresponds to a different conditional
> independence, and the planted truth lets us measure what each costs.

## Verification status of references

| Key | Verified against | Note |
| --- | --- | --- |
| dorazio2006book | Booktopia listing (ISBN 9780470023532, 2006, 270 pp.) | Publisher Wiley, Chichester, per the StatMatch documentation |
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
