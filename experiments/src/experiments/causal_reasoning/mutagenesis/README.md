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
  attributes and the same four counts, the classic propositional summary of a
  relational example.
- The **unrolled tree** is a JPT on a table that also carries every atom's and bond's
  attributes under the part's position, padded with an absent marker past a molecule's
  last part.
- The **scalars-only tree** is a JPT on the molecule's own attributes alone, what a flat
  learner sees without the relational feature extraction.

All three trees are registered as a `CausalCircuit` the same way the relational circuit
is. `results.md` holds the comparison; `causal_query_results.md` is the earlier,
single-model run of the branching-atom question on the relational circuit alone.

## The domain

`domain.py` holds the classes every other module works on.

| class | what it is |
|---|---|
| `MutagenesisMolecule` | one molecule: the `ind1` structural indicator, `logp`, `lumo`, whether it is `mutagenic`, and its `atoms` and `bonds` |
| `MutagenesisAtom` | one atom as an exchangeable part: its element, atom-type code, partial charge and how many bonds it takes part in |
| `MutagenesisBond` | one bond as an exchangeable part: its type |
| `MutagenesisMoleculeAggregations` | the counts the relational model derives over the parts: chlorine atoms and branching atoms over the atoms, double bonds and aromatic bonds over the bonds |

`dataset.py` fetches the molecules from the CTU database, holds them as a
`MutagenesisDataset` with a train/test split and mutagenic rates grouped by any key,
and generates synthetic molecules of the same shape for tests that must run without
network access.

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

| file | what it holds |
|---|---|
| `flat_table.py` | `MoleculeSchema`, how EQL names every attribute; `FlatTable`, the molecules as one row each in one of the three `TableLayout`s; `MoleculeView`, how much of a molecule a likelihood is taken over |
| `pipelines.py` | `CausalQueryPipeline` and its two implementations, `RelationalPipeline` and `FlatTablePipeline`, the latter once per layout |
| `queries.py` | the question catalogue, each question one EQL query that reads the same for every pipeline |
| `evaluation.py` | asking every question to every pipeline and recording what came of it (`evaluate`), then the three studies: `permutation_study` reorders every molecule's atoms and bonds and asks the atom questions again, `split_study` repeats the comparison over several random splits, `learning_curve` fits on growing shares of the molecules |
| `report.py` | rendering the comparison and the studies as Markdown |
| `run_pipeline.py` | the whole comparison end to end |
| `causal_query.py` | the earlier single-model run of the branching-atom question |

**One model per cause.** Backdoor adjustment needs the circuit to be
support-deterministic over the cause: no sum unit may mix branches that overlap on it.
A fit guarantees that by stratifying its training rows on the cause's exact value.
Each pipeline keeps one plain model for everything that is not a causal query, such as
scoring held-out molecules, and fits one further model per cause variable the first
time it is asked about that cause. A cause on an atom attribute stratifies the atom
template in the relational pipeline, or that position's column in the unrolled tree;
the other tables have no column for it.

**The questions.**

1. *Branching atoms, aromatic bonds and double bonds cause mutagenicity*, each
   adjusting for the `ind1` indicator, which marks the fused-ring molecules that are
   both large and mostly mutagenic. The cause is a count over the exchangeable parts.
2. *The indicator causes mutagenicity*, once adjusting for `logp`, which every
   pipeline has, and once for the branching-atom count, which the scalars-only tree
   does not. The cause is an attribute of the molecule itself.
3. *The indicator causes an atom to be carbon*, and *the branching-atom count causes
   an atom to be terminal* (to have a single bond). The cause is molecule-level, the
   effect one atom's own attribute.
4. *An atom's element causes it to be terminal*. Cause and effect both live on one
   atom.

Every pipeline that has the columns answers 1 and 2 identically. The relational
circuit and the unrolled tree answer 3, about different atoms: an exchangeable one and
whichever the molecule lists first. Only the unrolled tree answers 4; the relational
circuit refuses it, see below.

**The studies.** A single split cannot tell the pipelines apart, so three things are
measured around it. Reordering every molecule's atoms and bonds at random and asking
the atom questions again shows what an answer about "atom 0" is worth: nothing about
a relational circuit can depend on the order, while an unrolled table's column holds a
different atom of every molecule afterwards. Repeating the comparison over several
random splits gives the spread of every number. Fitting on a growing share of the
molecules shows how much data each pipeline needs to explain a whole molecule, atoms
and bonds included, which only the relational circuit and the unrolled tree can score
at all.

## What the results show

Numbers from `results.md`: one 150/38 split with seed 0 for the questions and
timings, five splits for the spreads, three random atom orderings.

- **Where every pipeline has the columns, every pipeline answers alike.** On the five
  molecule-level questions the relational circuit, the propositional tree and the
  unrolled tree agree to the third decimal: mutagenicity rises from 0 at seven
  branching atoms to 1 at seventeen and above, from 0.24 at six aromatic bonds to 1 at
  fourteen and above, and the indicator raises it from 0.31 to 0.95 adjusting for
  `logp` and from 0.37 to 0.94 adjusting for the branching count. This is expected:
  the relational circuit's class circuit and the propositional tree are the same tree
  on the same eight columns, and Monte-Carlo grounding with 2000 samples reproduces
  its regions. The scalars-only tree answers one of the eight questions; without the
  counts a flat learner cannot even pose the rest. Over five splits the best regions
  are stable except for the double-bond question, whose top region is a count held by
  one or two molecules, a caveat on reading "most effective" off tiny regions.
- **The unrolled tree's answers about atoms are answers about the listing order.**
  In the dataset's own order it finds atom 0 carbon with probability 1.00 whether or
  not the indicator is set (the CTU listing puts a carbon first in every molecule) and
  cannot find a terminal atom 0 at all. Reordering the atoms at random three times
  flips its most effective setting on both atom questions and moves its adjusted
  probabilities by up to 1.00; the relational circuit's answers do not move at all
  (0.52 against 0.44 carbon, 0.51 against 0.34 terminal, matching the raw atom
  counts), because an exchangeable atom has no position.
- **Only the relational circuit explains whole molecules.** On the held-out molecules
  both cover, its mean whole-molecule log-likelihood is 15.4 against the unrolled
  tree's -132.7, about 150 nats per molecule, and it covers more of them (60.5%
  against 55.3%); reordering the atoms drops the unrolled tree's coverage to 10 to
  18% and its likelihood by a further 20 nats, and leaves the relational circuit's
  unchanged. The learning curve shows why: the relational circuit pools every atom of
  every training molecule into one template, so its whole-molecule likelihood climbs
  from -17 at a fifth of the data to +4 at four fifths, while the unrolled tree, with
  one row per molecule over about 200 columns, stays between -183 and -132.
- **Extra columns do not buy the flat trees anything on the shared columns.** The
  unrolled tree's marginal on the scalars and counts is worse than the propositional
  tree's (-8.55 against -8.24; -8.67 against -8.34 over five splits), and the
  scalars-only tree is marginally the best on the scalars alone (-3.08 against -3.15
  over five splits). It is also the most expensive model here: 15,541 nodes and 66
  seconds of fitting against the relational circuit's 7,507 nodes and 27 seconds.
- **The atom-level cause is refused by all.** The relational circuit, grounding with
  the counts left open, mixes one copy of the atom template per sampled count, and
  those copies overlap on the element without being identical (a copy for a molecule
  with no chlorine has no chlorine atom, a copy for one with some has), so the
  grounded circuit is not support-deterministic over the element and verification
  rejects it. The propositional and scalars-only trees have no column for it. The
  unrolled tree has the column but, in the dataset's order, no terminal atom 0 to
  read an effect off; under random orderings it answers, and the answer moves.
- **Cost.** Once fitted, the trees answer in 0.04 to 0.4 seconds and the relational
  circuit in 2.3, the difference being the Monte-Carlo grounding.

What the comparison does not show: that the relational circuit gives better causal
answers on molecule-level questions than a flat tree given the same counts. It cannot,
because on those columns it is that tree. What it shows is that the questions a flat
learner can pose, and the answers it gives about parts, depend on how the parts were
written down, and that the relational circuit is the one model here for which they do
not.

## Running it

```bash
# fit, score and question every pipeline, reorder the atoms three times, repeat over
# five splits and measure the learning curve; needs network access to the CTU
# database and the experiments ORM interface, and takes about half an hour
python scripts/regenerate_all_orm.py
python -m experiments.causal_reasoning.mutagenesis.run_pipeline

# a quicker look: one ordering, two splits
python -m experiments.causal_reasoning.mutagenesis.run_pipeline --orderings 1 --splits 2
```

The tests under `test/causal_reasoning_test/test_mutagenesis` run the pipelines and the
studies on the synthetic molecules, so they need no network access.
