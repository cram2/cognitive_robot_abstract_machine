# Mutagenesis: relational circuit against flat-table trees

## What this experiment is for

The Tracy clutter-picking experiment compares a relational causal circuit with a flat
joint probability tree on a robot's own recorded attempts. This experiment puts the
same relational pipeline, and the same `cause`/`causes_effect` EQL queries, to a
standard relational-learning benchmark instead: the CTU Mutagenesis dataset
(https://relational.fel.cvut.cz/dataset/Mutagenesis), 188 nitroaromatic molecules with
their atoms and bonds and whether each tested mutagenic. Against it stand the three
flat tables a flat learner can build from such data, and the questions are chosen so
that where the pipelines agree, disagree and refuse is itself the result.

- The **relational circuit** is an RSPN fitted on the molecules' relational structure:
  one circuit over the molecule's own attributes and its aggregation counts, one
  template over an atom, one over a bond. For every query it grounds itself into a
  circuit over the queried molecule, atoms and bonds and registers that circuit as a
  `CausalCircuit`.
- The **propositional tree** is a joint probability tree (JPT) on the molecule's own
  attributes and the same five counts, the classic propositional summary of a
  relational example.
- The **unrolled tree** is a JPT on a table that also carries every atom's and bond's
  attributes under the part's position, padded with an absent marker past a molecule's
  last part.
- The **scalars-only tree** is a JPT on the molecule's own attributes alone, what a flat
  learner sees without the relational feature extraction.

All three trees are registered as a `CausalCircuit` the same way the relational circuit
is, and a **regression adjustment** on the propositional table, the textbook backdoor
estimator with a logistic regression in place of a circuit, stands beside them as a
reference that is not a circuit at all. `results.md` holds the comparison;
`causal_query_results.md` is the earlier, single-model run of the branching-atom
question on the relational circuit alone.

## The domain

`domain.py` holds the classes every other module works on.

| class | what it is |
|---|---|
| `MutagenesisMolecule` | one molecule: the `ind1` structural indicator, `logp`, `lumo`, whether it is `mutagenic`, and its `atoms` and `bonds` |
| `MutagenesisAtom` | one atom as an exchangeable part: its element, atom-type code, partial charge and how many bonds it takes part in |
| `MutagenesisBond` | one bond as an exchangeable part: its type |
| `MutagenesisMoleculeAggregations` | the counts the relational model derives over the parts: all atoms, chlorine atoms and branching atoms over the atoms, double bonds and aromatic bonds over the bonds |

`dataset.py` fetches the molecules from the CTU database, hands them to the shared
comparison as an `ExampleDataset`, summarises how often they are mutagenic by the
indicator, the branching-atom count, the aromatic-bond count and the atom count, and
generates synthetic molecules of the same shape for tests that must run without
network access. `domain.py` also describes the molecule to the shared comparison as a
`RelationalDomain`: which class is the example, which fields are its exchangeable
parts, and which attribute is the effect.

## Why three flat tables

A recorded clutter has a fixed number of neighbours, so the Tracy flat table unrolls
one block of columns per neighbour. A molecule has between 14 and 40 atoms and no
canonical order (the dataset lists heavy atoms first and hydrogens last, but nothing
ties a position to an identity), so a flat learner has three choices, and each is a
pipeline here:

- keep only the scalars, and lose every question about the parts;
- add the aggregation counts, which is exactly the table the relational circuit's
  class-level circuit is fitted on, so on those columns the two are the same tree;
- unroll the parts by position and pad, which makes a column mean whatever part a
  molecule happens to list there.

Every query lists one atom and one bond with all their attributes open. For the
relational circuit that is what makes grounding retain the molecule's counts as
variables (a query with an empty atom list is a molecule with no atoms, whose counts
are zero). A flat table ignores a part a query merely lists and refuses a query that
constrains a column it does not have: sets one of its attributes, or marks it as cause,
confounder or effect.

## The pipelines

The pipelines, the studies and the report are not this experiment's own: they live in
the shared `experiments.causal_reasoning.comparison` package and work on any relational
example a `RelationalDomain` describes, so that every dataset compared this way is
compared the same way. This package supplies the domain, the data, the questions and
the words.

| file | what it holds |
|---|---|
| `domain.py` | the molecule, its parts and their aggregation counts, and `molecule_domain()`, the molecule as the shared comparison sees it |
| `dataset.py` | fetching the molecules, the synthetic generator, and the mutagenicity summaries the report opens with |
| `queries.py` | the question catalogue, each question one EQL query that reads the same for every pipeline |
| `run_pipeline.py` | the whole comparison end to end: the `Experiment` handed to the shared runner, and the report's prose |
| `causal_query.py` | the earlier single-model run of the branching-atom question |

And in the shared package:

| file | what it holds |
|---|---|
| `domain.py` | `RelationalDomain`, what an example and its exchangeable parts are; `ExampleView`, how much of an example a likelihood is taken over |
| `flat_table.py` | `Schema`, how EQL names every attribute; `FlatTable`, the examples as one row each in one of the three `TableLayout`s |
| `pipelines.py` | `CausalQueryPipeline` and its three implementations: `RelationalPipeline`, `HybridPipeline`, and `FlatTablePipeline` once per layout |
| `baselines.py` | regression adjustment on the propositional table |
| `neural_baseline.py` | deep set adjustment, a permutation invariant encoder over the parts with the same g-computation on top |
| `queries.py` | what every question is made of: `CausalQueryCase`, `Confounder`, and the open-part queries |
| `dataset.py` | `ExampleDataset`: splitting, reordering the parts, and the effect's rate |
| `evaluation.py` | asking every question to every pipeline and recording what came of it (`evaluate`), then the studies: `permutation_study` reorders every example's parts and asks the part questions again, `split_study` repeats the comparison over several random splits, `learning_curve` fits on growing shares of the examples, `ground_truth_study` scores every pipeline against a `KnownTruth`, `monte_carlo_study` follows the relational circuit's answers as grounding draws more samples, `scaling_study` measures cost against the number of parts |
| `report.py` | rendering the comparison and the studies as Markdown, around the `ReportText` an experiment writes |
| `run.py` | `Experiment`, `RunSettings` and `run`, the studies in order with the report written after each |

**One model per cause.** Backdoor adjustment needs the circuit to be
support-deterministic over the cause: no sum unit may mix branches that overlap on it.
A fit guarantees that by stratifying its training rows on the cause's exact value.
Each pipeline keeps one plain model for everything that is not a causal query, such as
scoring held-out molecules, and fits one further model per cause variable the first
time it is asked about that cause. A cause on an atom attribute stratifies the atom
template in the relational pipeline, or that position's column in the unrolled tree;
the other tables have no column for it.

**The questions.**

1. *Branching atoms, aromatic bonds and double bonds cause mutagenicity*, each asked
   three times: adjusting for the `ind1` indicator, which marks the fused-ring
   molecules that are both large and mostly mutagenic; adjusting for the number of
   atoms, which a larger molecule has more of along with more of every other kind of
   atom and bond; and adjusting for both. The cause is a count over the exchangeable
   parts, and what adjusting changes is a result in itself.
2. *The indicator causes mutagenicity*, once adjusting for `logp`, which every
   pipeline has, and once for the branching-atom count, which the scalars-only tree
   does not. The cause is an attribute of the molecule itself.
3. *The indicator causes an atom to be carbon*, and *the branching-atom count causes
   an atom to be terminal* (to have a single bond). The cause is molecule-level, the
   effect one atom's own attribute.
4. *An atom's element causes it to be terminal*. Cause and effect both live on one
   atom.

Every pipeline that has the columns answers 1 and 2 identically. The relational
circuit and the unrolled tree answer 3 and 4, about different atoms: an exchangeable
one and whichever the molecule lists at that position.

Every answer is read per region of the cause with the number of training molecules
the region holds, a Wilson interval on the adjusted probability, and two summaries
that do not depend on an argmax over sparse regions: the *trend*, Spearman's rank
correlation between a numeric cause and the adjusted probability over the supported
regions, and the *contrast*, the adjusted probability at the highest supported region
minus at the lowest, with Newcombe's interval. A region holding fewer molecules than
the support threshold is marked and left out of every summary.

**The studies.** A single split cannot tell the pipelines apart, so several things are
measured around it. Reordering every molecule's atoms and bonds at random, twenty times
over, and asking the atom questions again shows what an answer about "atom 0" is
worth: nothing about a relational circuit can depend on the order, while an unrolled
table's column holds a different atom of every molecule afterwards; the dataset's own
order is the baseline every reordering is measured against. Following the relational
circuit's answers as grounding draws more and more Monte-Carlo samples shows how many
it takes for them to settle. Fitting on a growing share of the molecules shows how much
data each pipeline needs to explain a whole molecule, atoms and bonds included, which
only the relational circuit and the unrolled tree can score at all. Repeating the
comparison over several random splits gives the spread of every number, and is
optional (`--splits N`).

## What the results show

Numbers from `results.md`: one 150/38 split with seed 0, twenty random orderings of
the atoms and bonds, a support threshold of ten training molecules per cause region.

- **Where every pipeline has the columns, every pipeline answers alike.** On the
  eleven molecule-level questions the relational circuit and the propositional tree
  give the same numbers to the third decimal, because on those columns they are the
  same tree fitted on the same rows, and the unrolled tree is within a few hundredths
  of them. Mutagenicity rises from 0.13 at ten branching atoms to 1 at seventeen and
  above (contrast 0.87, interval [0.49, 0.97]), from 0.27 at six aromatic bonds to 1
  at seventeen and above (0.73, [0.47, 0.86]), and by 0.26 from two to four double
  bonds ([0.06, 0.40]); the indicator raises it from 0.31 to 0.95 adjusting for `logp`
  and from 0.42 to 0.94 adjusting for the branching count. The scalars-only tree
  answers one of the fourteen questions: without the counts a flat learner cannot even
  pose the rest. Regression adjustment, the estimator that is not a circuit, agrees on
  which setting is best but flattens every contrast (0.58 against 0.82 for branching
  atoms adjusting for the indicator, 0.05 against 0.73 for aromatic bonds adjusting
  for both confounders), which is what a logistic model does to a relation that is a
  step.
- **Adjusting changes the answer inside a region, not the trend.** At thirteen
  branching atoms the naive rate is 0.73, 0.82 adjusting for the indicator, 0.80 for
  the atom count and 0.70 for both; at eleven aromatic bonds it is 0.33 naive and 0.65
  adjusting for the indicator. The trend stays between 0.90 and 1.00 and the contrast
  moves by at most 0.05 under any adjustment, so the backdoor adjustment is doing
  something, and what it does not do is change what the question is about.
- **The relational circuit answers every question about an atom, and its answers are
  the atom counts.** The indicator makes an exchangeable atom carbon with probability
  0.55 against 0.43 without it; the branching-atom count has no effect on whether an
  atom is terminal (0.23 to 0.53 across the supported regions, trend -0.05); and the
  atom's element decides it entirely: hydrogen and chlorine are terminal with
  probability 1.00, oxygen 0.94, carbon and nitrogen 0.00, which is the valence table.
  The last question, whose cause and effect both live on one atom, is one the earlier
  version of this pipeline refused; it is answered now because a cause region is
  counted once however the support writes it.
- **The unrolled tree's answers about atoms are answers about the listing order.**
  In the dataset's own order it finds atom 0 carbon with probability 1.00 whether or
  not the indicator is set (the CTU listing puts a carbon first in every molecule),
  and refuses both terminal questions because no molecule's first atom is terminal.
  Over twenty reorderings its most effective setting moves in 95% of them on the
  carbon question and its adjusted probabilities range by up to 0.65 there and 1.00
  on the terminal question; the relational circuit's answers do not move by 1e-9,
  because an exchangeable atom has no position.
- **Only the relational circuit explains whole molecules.** On the held-out molecules
  both cover, its mean whole-molecule log-likelihood is -40.8 against the unrolled
  tree's -91.3, and it covers more of them (73.7% against 44.7%); reordering the atoms
  drops the unrolled tree's coverage to 5% and its likelihood by 56 nats, and leaves
  the relational circuit's unchanged. The learning curve shows why: the relational
  circuit pools every atom of every training molecule into one template, so at a
  fifth of the data it already covers 30% of the held-out molecules at -54 nats,
  where the unrolled tree, with one row per molecule over about 200 columns, covers
  10% at -69 and, as it covers more, scores worse.
- **Grounding needs no more than fifty samples.** The counts a query leaves open take
  few distinct values, and grounding integrates over the distinct values it has
  drawn, so from fifty samples on every answer is the same to the third decimal.
- **Cost.** The relational circuit is 4,663 nodes and 44 seconds of fitting, the
  propositional tree 3,009 and 10 seconds, the unrolled tree 72,243 and 83 seconds.
  Once fitted, the trees and the circuit take about the same time per question (7 to
  9 seconds), most of it in the backdoor adjustment over the atom count's many
  regions; the scalars-only tree answers in 0.1 seconds, and answers almost nothing.

What the comparison does not show: that the relational circuit gives better causal
answers on molecule-level questions than a flat tree given the same counts. It cannot,
because on those columns it is that tree. What it shows is that the questions a flat
learner can pose, and the answers it gives about parts, depend on how the parts were
written down, and that the relational circuit is the one model here for which they do
not.

## Running it

```bash
# fit, score and question every pipeline, reorder the atoms twenty times, follow the
# grounding samples and measure the learning curve; needs network access to the CTU
# database and the experiments ORM interface
python scripts/regenerate_all_orm.py
python -m experiments.causal_reasoning.mutagenesis.run_pipeline

# a quicker look: three orderings; add splits with --splits N
python -m experiments.causal_reasoning.mutagenesis.run_pipeline --orderings 3
```

The report is written out again after every study, so a run stopped part-way leaves
what it has finished in `results.md`.

The tests under `test/experiments_test/causal_reasoning_test/test_mutagenesis` run the pipelines and the
studies on the synthetic molecules, so they need no network access.
