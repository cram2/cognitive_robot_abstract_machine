# Mutagenesis: relational circuit against flat-table trees

The CTU Mutagenesis dataset records 188 nitroaromatic molecules, each as its own attributes (the `ind1` structural indicator, `logp`, `lumo`, and whether it tested mutagenic) with one exchangeable part per atom (element, atom-type code, partial charge, number of bonds) and one per bond (its type). A molecule has between 14 and 40 atoms, and its atoms have no canonical order; the dataset lists heavy atoms first and hydrogens last, but nothing ties position to identity.

Four pipelines were fitted on the same molecules and asked the same `cause`/`causes_effect` EQL queries:

- **relational circuit**: a relational probabilistic circuit fitted on the molecules' relational structure, one circuit over the molecule's own attributes and its aggregation counts (atoms, chlorine atoms, branching atoms, double bonds, aromatic bonds), one template over an atom's attributes and one over a bond's, grounded per query into a circuit over exactly the queried molecule, atoms and bonds and registered as a causal circuit;
- **propositional tree**: a joint probability tree fitted on the molecules flattened into one table of the molecule's own attributes and the same five counts, the classic propositional summary of a relational example, registered as a causal circuit the same way;
- **unrolled tree**: the same tree on a table that also carries every atom's and bond's attributes under the part's position, padded with an absent marker past a molecule's last part, so that a column means whatever part a molecule happens to list at that position;
- **scalars-only tree**: the same tree on the molecule's own attributes alone, what a flat learner sees without the relational feature extraction.

Every flat tree answers a query by backdoor adjustment on a table column; the relational circuit does the same on the variable of a grounded circuit. In both, the model is stratified so it is support-deterministic over the cause, the effect's probability is read off every region of the cause, and any variable the query marks as a confounder is summed out of that reading. Every query lists one atom and one bond with all their attributes open, which is what makes grounding retain the molecule's counts as variables; a flat table ignores parts a query says nothing about and refuses a query that constrains a column it does not have.

## Setup

- molecules: 188 (150 to fit on, 38 held out)
- molecules where the molecule is mutagenic: 66.5%
- fewest training rows per leaf, as a share of the rows fitted on: 0.05 in a cause-specific model, 0.15 in the plain model that scores held-out molecules
- split seed: 0
- fewest training molecules a cause region may hold for its effect to be read as an answer: 10; a region below that is marked † in the tables and takes no part in any summary

## How often the molecule is mutagenic

The molecules themselves, before any model: the share where the molecule is mutagenic, grouped by the `ind1` indicator, by how many branching atoms (atoms with three or four bonds, the ring-fusion and branch points of the molecular graph) the molecule has, by how many of its bonds are aromatic, and by how many atoms it holds at all. This is the signal the models are asked to explain.

| ind1 | molecules | effect |
|---|---|---|
| False | 85 | 30.6% |
| True | 103 | 96.1% |

| branching atoms | molecules | effect |
|---|---|---|
| 7 | 7 | 0.0% |
| 8 | 12 | 33.3% |
| 9 | 12 | 16.7% |
| 10 | 13 | 15.4% |
| 11 | 7 | 14.3% |
| 12 | 5 | 40.0% |
| 13 | 16 | 68.8% |
| 14 | 21 | 61.9% |
| 15 | 20 | 85.0% |
| 16 | 12 | 83.3% |
| 17 | 19 | 100.0% |
| 18 | 14 | 100.0% |
| 19 | 7 | 100.0% |
| 20 | 1 | 100.0% |
| 21 | 15 | 100.0% |
| 22 | 4 | 100.0% |
| 24 | 2 | 100.0% |
| 25 | 1 | 100.0% |

| aromatic bonds | molecules | effect |
|---|---|---|
| 5 | 1 | 0.0% |
| 6 | 32 | 21.9% |
| 10 | 11 | 27.3% |
| 11 | 17 | 35.3% |
| 12 | 60 | 70.0% |
| 14 | 3 | 100.0% |
| 15 | 4 | 100.0% |
| 16 | 7 | 100.0% |
| 17 | 16 | 100.0% |
| 18 | 1 | 100.0% |
| 19 | 21 | 100.0% |
| 21 | 1 | 100.0% |
| 22 | 1 | 100.0% |
| 24 | 10 | 100.0% |
| 26 | 2 | 100.0% |
| 30 | 1 | 100.0% |

| atoms | molecules | effect |
|---|---|---|
| 14 | 7 | 0.0% |
| 16 | 8 | 50.0% |
| 17 | 5 | 40.0% |
| 18 | 8 | 0.0% |
| 19 | 8 | 25.0% |
| 20 | 9 | 0.0% |
| 21 | 4 | 25.0% |
| 22 | 7 | 71.4% |
| 23 | 4 | 25.0% |
| 24 | 14 | 64.3% |
| 25 | 6 | 83.3% |
| 26 | 17 | 70.6% |
| 27 | 5 | 60.0% |
| 28 | 20 | 95.0% |
| 29 | 7 | 85.7% |
| 30 | 20 | 90.0% |
| 31 | 2 | 50.0% |
| 32 | 12 | 100.0% |
| 34 | 13 | 100.0% |
| 36 | 1 | 100.0% |
| 38 | 7 | 100.0% |
| 40 | 4 | 100.0% |


## Which questions each pipeline can answer

One row per question, one column per pipeline. An answered cell says, in words, which setting of the cause makes the effect most likely after adjustment and how likely, against the least favourable setting, over the regions that hold enough training molecules to be read; a refused cell says why the pipeline could not answer at all.

| question | relational circuit | propositional tree | unrolled tree | scalars-only tree | regression adjustment | neural adjustment |
|---|---|---|---|---|---|---|
| How many branching atoms cause a molecule to be mutagenic, adjusting for the ind1 indicator? | answered: with 18 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 10 branching atoms it is only 0.18. | answered: with 18 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 10 branching atoms it is only 0.18. | answered: with 21 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 10 branching atoms it is only 0.18. | refused: the fitted table has no column for the queried variables. | answered: with 21 branching atoms, the molecule is mutagenic with probability 0.97, the highest of any setting; with 10 branching atoms it is only 0.39. | answered: with 21 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 10 branching atoms it is only 0.15. |
| How many branching atoms cause a molecule to be mutagenic, adjusting for the number of atoms? | answered: with 21 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 10 branching atoms it is only 0.13. | answered: with 21 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 10 branching atoms it is only 0.13. | answered: with 18 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 10 branching atoms it is only 0.13. | refused: the fitted table has no column for the queried variables. | answered: with 21 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 10 branching atoms it is only 0.12. | answered: with 21 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 10 branching atoms it is only 0.13. |
| How many branching atoms cause a molecule to be mutagenic, adjusting for the ind1 indicator and the number of atoms? | answered: with 21 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 10 branching atoms it is only 0.13. | answered: with 21 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 10 branching atoms it is only 0.13. | answered: with 18 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 10 branching atoms it is only 0.13. | refused: the fitted table has no column for the queried variables. | answered: with 21 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 10 branching atoms it is only 0.12. | answered: with 21 branching atoms, the molecule is mutagenic with probability 0.99, the highest of any setting; with 10 branching atoms it is only 0.16. |
| How many aromatic bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator? | answered: with 17 aromatic bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 6 aromatic bonds it is only 0.24. | answered: with 17 aromatic bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 6 aromatic bonds it is only 0.24. | answered: with 17 aromatic bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 6 aromatic bonds it is only 0.24. | refused: the fitted table has no column for the queried variables. | answered: with 19 aromatic bonds, the molecule is mutagenic with probability 0.93, the highest of any setting; with 6 aromatic bonds it is only 0.38. | answered: with 19 aromatic bonds, the molecule is mutagenic with probability 0.94, the highest of any setting; with 11 aromatic bonds it is only 0.64. |
| How many aromatic bonds cause a molecule to be mutagenic, adjusting for the number of atoms? | answered: with 19 aromatic bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 6 aromatic bonds it is only 0.27. | answered: with 19 aromatic bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 6 aromatic bonds it is only 0.27. | answered: with 19 aromatic bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 6 aromatic bonds it is only 0.30. | refused: the fitted table has no column for the queried variables. | answered: with 19 aromatic bonds, the molecule is mutagenic with probability 0.86, the highest of any setting; with 6 aromatic bonds it is only 0.46. | answered: with 19 aromatic bonds, the molecule is mutagenic with probability 0.95, the highest of any setting; with 11 aromatic bonds it is only 0.59. |
| How many aromatic bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator and the number of atoms? | answered: with 17 aromatic bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 6 aromatic bonds it is only 0.27. | answered: with 19 aromatic bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 6 aromatic bonds it is only 0.27. | answered: with 19 aromatic bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 6 aromatic bonds it is only 0.30. | refused: the fitted table has no column for the queried variables. | answered: with 19 aromatic bonds, the molecule is mutagenic with probability 0.69, the highest of any setting; with 6 aromatic bonds it is only 0.64. | answered: with 19 aromatic bonds, the molecule is mutagenic with probability 0.86, the highest of any setting; with 12 aromatic bonds it is only 0.64. |
| How many double bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator? | answered: with 4 double bonds, the molecule is mutagenic with probability 0.79, the highest of any setting; with 2 double bonds it is only 0.54. | answered: with 4 double bonds, the molecule is mutagenic with probability 0.79, the highest of any setting; with 2 double bonds it is only 0.54. | answered: with 4 double bonds, the molecule is mutagenic with probability 0.79, the highest of any setting; with 2 double bonds it is only 0.54. | refused: the fitted table has no column for the queried variables. | answered: with 4 double bonds, the molecule is mutagenic with probability 0.74, the highest of any setting; with 2 double bonds it is only 0.56. | answered: with 4 double bonds, the molecule is mutagenic with probability 0.75, the highest of any setting; with 2 double bonds it is only 0.60. |
| How many double bonds cause a molecule to be mutagenic, adjusting for the number of atoms? | answered: with 4 double bonds, the molecule is mutagenic with probability 0.85, the highest of any setting; with 2 double bonds it is only 0.57. | answered: with 4 double bonds, the molecule is mutagenic with probability 0.85, the highest of any setting; with 2 double bonds it is only 0.57. | answered: with 4 double bonds, the molecule is mutagenic with probability 0.85, the highest of any setting; with 2 double bonds it is only 0.60. | refused: the fitted table has no column for the queried variables. | answered: with 4 double bonds, the molecule is mutagenic with probability 0.70, the highest of any setting; with 2 double bonds it is only 0.63. | answered: with 4 double bonds, the molecule is mutagenic with probability 0.66, the highest of any setting; with 3 double bonds it is only 0.64. |
| How many double bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator and the number of atoms? | answered: with 4 double bonds, the molecule is mutagenic with probability 0.82, the highest of any setting; with 2 double bonds it is only 0.58. | answered: with 4 double bonds, the molecule is mutagenic with probability 0.82, the highest of any setting; with 2 double bonds it is only 0.58. | answered: with 4 double bonds, the molecule is mutagenic with probability 0.82, the highest of any setting; with 2 double bonds it is only 0.60. | refused: the fitted table has no column for the queried variables. | answered: with 4 double bonds, the molecule is mutagenic with probability 0.72, the highest of any setting; with 2 double bonds it is only 0.59. | answered: with 4 double bonds, the molecule is mutagenic with probability 0.98, the highest of any setting; with 2 double bonds it is only 0.97. |
| Does the ind1 indicator cause a molecule to be mutagenic, adjusting for its hydrophobicity (logp)? | answered: with ind1 = True, the molecule is mutagenic with probability 0.95, the highest of any setting; with ind1 = False it is only 0.31. | answered: with ind1 = True, the molecule is mutagenic with probability 0.95, the highest of any setting; with ind1 = False it is only 0.31. | answered: with ind1 = True, the molecule is mutagenic with probability 0.95, the highest of any setting; with ind1 = False it is only 0.31. | answered: with ind1 = True, the molecule is mutagenic with probability 0.97, the highest of any setting; with ind1 = False it is only 0.31. | answered: with ind1 = True, the molecule is mutagenic with probability 0.90, the highest of any setting; with ind1 = False it is only 0.40. | answered: with ind1 = False, the molecule is mutagenic with probability 0.98, the highest of any setting; with ind1 = True it is only 0.97. |
| Does the ind1 indicator cause a molecule to be mutagenic, adjusting for its branching-atom count? | answered: with ind1 = True, the molecule is mutagenic with probability 0.94, the highest of any setting; with ind1 = False it is only 0.42. | answered: with ind1 = True, the molecule is mutagenic with probability 0.94, the highest of any setting; with ind1 = False it is only 0.42. | answered: with ind1 = True, the molecule is mutagenic with probability 0.95, the highest of any setting; with ind1 = False it is only 0.37. | refused: the fitted table has no column for the queried variables. | answered: with ind1 = True, the molecule is mutagenic with probability 0.79, the highest of any setting; with ind1 = False it is only 0.57. | answered: with ind1 = False, the molecule is mutagenic with probability 0.98, the highest of any setting; with ind1 = True it is only 0.97. |
| Does the ind1 indicator cause atom 0 of a molecule to be carbon? | answered: with ind1 = True, atom 0 is carbon with probability 0.55, the highest of any setting; with ind1 = False it is only 0.43. | refused: the fitted table has no column for the queried variables. | answered: with ind1 = False, atom 0 is carbon with probability 1.00, the highest of any setting; with ind1 = True it is only 1.00. | refused: the fitted table has no column for the queried variables. | refused: the fitted table has no column for the queried variables. | answered: with ind1 = False, atom 0 is carbon with probability 0.49, the highest of any setting; with ind1 = True it is only 0.49. |
| How many branching atoms cause atom 0 of a molecule to be terminal, with a single bond? | answered: with 10 branching atoms, atom 0 is terminal with probability 0.45, the highest of any setting; with 13 branching atoms it is only 0.23. | refused: the fitted table has no column for the queried variables. | refused: the effect has zero probability under every cause region. | refused: the fitted table has no column for the queried variables. | refused: the fitted table has no column for the queried variables. | answered: with 22 branching atoms, atom 0 is terminal with probability 0.44, the highest of any setting; with 7 branching atoms it is only 0.36. |
| Does the element of atom 0 of a molecule cause it to be terminal, with a single bond? | answered: with atom 0 being of element cl, atom 0 is terminal with probability 1.00, the highest of any setting; with atom 0 being of element c it is only 0.00. | refused: the fitted table has no column for the queried variables. | refused: the effect has zero probability under every cause region. | refused: the fitted table has no column for the queried variables. | refused: the fitted table has no column for the queried variables. | answered: with atom 0 being of element cl, atom 0 is terminal with probability 0.76, the highest of any setting; with atom 0 being of element c it is only 0.29. |

A question about counts needs the counts: the scalars-only tree refuses it. A question whose effect is one atom's own attribute needs the atoms: the propositional tree refuses it, the unrolled tree answers it about whatever atom the molecules list at that position, and the relational circuit answers it about an exchangeable atom. What an answer about "atom 0" is worth is what the reordering below measures.

## Trend and contrast

The most effective setting is an argmax over up to twenty sparse regions and moves with the split. Two summaries that do not: *trend* is Spearman's rank correlation between the cause's value and the adjusted probability over the supported regions, for a numeric cause; *contrast* is the adjusted probability at the highest supported region minus at the lowest (for a symbolic cause, at the most effective minus at the least), with Newcombe's interval from the Wilson intervals of the two regions' support.

| question | relational circuit, trend | relational circuit, contrast | propositional tree, trend | propositional tree, contrast | unrolled tree, trend | unrolled tree, contrast | scalars-only tree, trend | scalars-only tree, contrast | regression adjustment, trend | regression adjustment, contrast | neural adjustment, trend | neural adjustment, contrast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| branching_atom_count_causes_mutagenicity_adjusting_indicator_1 | 0.90 | 0.82 [0.44, 0.95] (10 → 21) | 0.92 | 0.82 [0.44, 0.95] (10 → 21) | 0.93 | 0.82 [0.44, 0.95] (10 → 21) | - | - | 1.00 | 0.58 [0.20, 0.80] (10 → 21) | 1.00 | 0.85 [0.47, 0.96] (10 → 21) |
| branching_atom_count_causes_mutagenicity_adjusting_atom_count | 0.93 | 0.87 [0.49, 0.97] (10 → 21) | 0.93 | 0.87 [0.49, 0.97] (10 → 21) | 0.90 | 0.87 [0.49, 0.97] (10 → 21) | - | - | 1.00 | 0.88 [0.50, 0.97] (10 → 21) | 1.00 | 0.87 [0.49, 0.97] (10 → 21) |
| branching_atom_count_causes_mutagenicity_adjusting_indicator_1_and_atom_count | 0.97 | 0.87 [0.49, 0.97] (10 → 21) | 0.98 | 0.87 [0.49, 0.97] (10 → 21) | 0.95 | 0.87 [0.49, 0.97] (10 → 21) | - | - | 1.00 | 0.88 [0.50, 0.97] (10 → 21) | 1.00 | 0.83 [0.45, 0.95] (10 → 21) |
| aromatic_bond_count_causes_mutagenicity_adjusting_indicator_1 | 0.90 | 0.76 [0.50, 0.89] (6 → 19) | 0.97 | 0.76 [0.50, 0.89] (6 → 19) | 0.90 | 0.76 [0.50, 0.89] (6 → 19) | - | - | 1.00 | 0.55 [0.27, 0.72] (6 → 19) | 0.70 | 0.13 [-0.10, 0.33] (6 → 19) |
| aromatic_bond_count_causes_mutagenicity_adjusting_atom_count | 1.00 | 0.73 [0.47, 0.86] (6 → 19) | 1.00 | 0.73 [0.47, 0.86] (6 → 19) | 1.00 | 0.70 [0.44, 0.84] (6 → 19) | - | - | 1.00 | 0.41 [0.12, 0.61] (6 → 19) | 0.70 | 0.25 [0.00, 0.45] (6 → 19) |
| aromatic_bond_count_causes_mutagenicity_adjusting_indicator_1_and_atom_count | 0.90 | 0.73 [0.47, 0.86] (6 → 19) | 0.97 | 0.73 [0.47, 0.86] (6 → 19) | 1.00 | 0.70 [0.44, 0.84] (6 → 19) | - | - | 1.00 | 0.05 [-0.23, 0.31] (6 → 19) | 0.60 | 0.11 [-0.14, 0.33] (6 → 19) |
| double_bond_count_causes_mutagenicity_adjusting_indicator_1 | 1.00 | 0.26 [0.06, 0.40] (2 → 4) | 1.00 | 0.26 [0.06, 0.40] (2 → 4) | 1.00 | 0.26 [0.06, 0.40] (2 → 4) | - | - | 1.00 | 0.18 [-0.02, 0.34] (2 → 4) | 1.00 | 0.16 [-0.04, 0.31] (2 → 4) |
| double_bond_count_causes_mutagenicity_adjusting_atom_count | 1.00 | 0.28 [0.10, 0.42] (2 → 4) | 1.00 | 0.28 [0.10, 0.42] (2 → 4) | 1.00 | 0.25 [0.06, 0.38] (2 → 4) | - | - | 1.00 | 0.07 [-0.12, 0.24] (2 → 4) | 0.50 | 0.01 [-0.19, 0.18] (2 → 4) |
| double_bond_count_causes_mutagenicity_adjusting_indicator_1_and_atom_count | 1.00 | 0.24 [0.05, 0.38] (2 → 4) | 1.00 | 0.24 [0.05, 0.38] (2 → 4) | 1.00 | 0.22 [0.03, 0.36] (2 → 4) | - | - | 1.00 | 0.13 [-0.07, 0.29] (2 → 4) | 1.00 | 0.01 [-0.11, 0.07] (2 → 4) |
| indicator_causes_mutagenicity_adjusting_logp | - | 0.64 [0.51, 0.74] (False → True) | - | 0.64 [0.51, 0.74] (False → True) | - | 0.64 [0.51, 0.74] (False → True) | - | 0.66 [0.52, 0.76] (False → True) | - | 0.50 [0.35, 0.62] (False → True) | - | 0.00 [-0.07, 0.07] (True → False) |
| indicator_causes_mutagenicity_adjusting_branching_atom_count | - | 0.52 [0.38, 0.63] (False → True) | - | 0.52 [0.38, 0.63] (False → True) | - | 0.58 [0.44, 0.69] (False → True) | - | - | - | 0.22 [0.07, 0.36] (False → True) | - | 0.00 [-0.07, 0.07] (True → False) |
| indicator_causes_carbon_atom_0 | - | 0.12 [-0.04, 0.27] (False → True) | - | - | - | 0.00 [-0.05, 0.04] (True → False) | - | - | - | - | - | 0.00 [-0.03, 0.03] (True → False) |
| branching_atom_count_causes_terminal_atom_0 | -0.05 | -0.04 [-0.39, 0.32] (10 → 21) | - | - | - | - | - | - | - | - | 0.97 | 0.09 [-0.09, 0.27] (7 → 25) |
| element_causes_terminal_atom_0 | - | 1.00 [0.86, 1.00] (c → cl) | - | - | - | - | - | - | - | - | - | 0.46 [0.26, 0.59] (c → cl) |

## What adjusting for changes

The same count question under each set of confounders it was asked with, read off the relational circuit. *n* is how many training molecules hold that value of the cause; † marks a region below the support threshold.

### branching_atom_count

| cause region | n | naive | adjusted for the ind1 indicator | adjusted for the number of atoms | adjusted for the ind1 indicator and the number of atoms |
|---|---|---|---|---|---|
| 7 † | 7 | 0.000 | 0.000 | 0.000 | 0.000 |
| 8 † | 8 | 0.375 | 0.375 | 0.176 | 0.176 |
| 9 † | 9 | 0.111 | 0.111 | 0.063 | 0.063 |
| 10 | 11 | 0.182 | 0.182 | 0.127 | 0.127 |
| 11 † | 5 | 0.200 | 0.200 | 0.107 | 0.115 |
| 12 † | 4 | 0.250 | 0.250 | 0.286 | 0.300 |
| 13 | 11 | 0.727 | 0.818 | 0.796 | 0.703 |
| 14 | 17 | 0.588 | 0.629 | 0.500 | 0.638 |
| 15 | 17 | 0.824 | 0.806 | 0.684 | 0.839 |
| 16 | 12 | 0.833 | 0.849 | 0.893 | 0.904 |
| 17 | 14 | 1.000 | 1.000 | 1.000 | 1.000 |
| 18 | 12 | 1.000 | 1.000 | 1.000 | 1.000 |
| 19 † | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| 20 † | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| 21 | 12 | 1.000 | 1.000 | 1.000 | 1.000 |
| 22 † | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| 24 † | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| 25 † | 1 | 1.000 | 1.000 | 1.000 | 1.000 |

### aromatic_bond_count

| cause region | n | naive | adjusted for the ind1 indicator | adjusted for the number of atoms | adjusted for the ind1 indicator and the number of atoms |
|---|---|---|---|---|---|
| 5 † | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| 6 | 25 | 0.240 | 0.240 | 0.269 | 0.269 |
| 10 † | 8 | 0.250 | 0.250 | 0.328 | 0.114 |
| 11 | 15 | 0.333 | 0.651 | 0.374 | 0.387 |
| 12 | 52 | 0.712 | 0.693 | 0.736 | 0.808 |
| 14 † | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| 15 † | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| 16 † | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| 17 | 12 | 1.000 | 1.000 | 1.000 | 1.000 |
| 18 † | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| 19 | 18 | 1.000 | 1.000 | 1.000 | 1.000 |
| 22 † | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| 24 † | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| 26 † | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| 30 † | 1 | 1.000 | 1.000 | 1.000 | 1.000 |

### double_bond_count

| cause region | n | naive | adjusted for the ind1 indicator | adjusted for the number of atoms | adjusted for the ind1 indicator and the number of atoms |
|---|---|---|---|---|---|
| 2 | 86 | 0.547 | 0.537 | 0.571 | 0.582 |
| 3 | 13 | 0.692 | 0.741 | 0.734 | 0.740 |
| 4 | 33 | 0.848 | 0.794 | 0.852 | 0.820 |
| 5 † | 2 | 0.500 | 0.547 | 0.868 | 0.860 |
| 6 † | 9 | 0.889 | 0.924 | 0.910 | 0.948 |
| 7 † | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 † | 5 | 0.800 | 0.887 | 0.842 | 0.906 |
| 9 † | 1 | 1.000 | 1.000 | 1.000 | 1.000 |


## Fit and likelihood

What each pipeline cost. *Models fitted* counts the plain model plus one support-deterministic model per distinct cause the questions asked about and the pipeline could fit; *training seconds* and the *nodes*/*edges* of every fitted circuit are summed over them, which for the relational circuit includes the part templates.

| pipeline | models fitted | training seconds | nodes | edges |
|---|---|---|---|---|
| relational circuit | 6 | 44.57 | 4663 | 4645 |
| propositional tree | 5 | 10.76 | 3009 | 3004 |
| unrolled tree | 6 | 83.32 | 72243 | 72237 |
| scalars-only tree | 2 | 0.03 | 232 | 230 |
| regression adjustment | 12 | 2.04 | 0 | 0 |
| neural adjustment | 15 | 1.97 | 0 | 0 |

How well each explains molecules it never saw, on three views of one molecule: its own scalars, which every pipeline models; its scalars and counts; and the whole molecule, parts included, which only the pipelines that model the parts can score. The relational circuit scores a whole molecule as its class circuit over the scalars and counts times each part template over one part given the counts; the unrolled tree scores it as one row. *Held-out coverage* is the share of held-out molecules that lie inside the plain model's support at all, since a tree's leaves span only the value ranges they were fitted on, and a whole molecule is covered only if every one of its parts is. The *mean log-likelihood* is over the covered molecules only; the last column restricts it to the molecules every pipeline in the table covers, so the numbers are over the same rows.

### scalars

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|
| relational circuit | 92.1% | -3.05 | -3.05 |
| propositional tree | 92.1% | -3.05 | -3.05 |
| unrolled tree | 92.1% | -3.08 | -3.08 |
| scalars-only tree | 92.1% | -2.77 | -2.77 |

### scalars and counts

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|
| relational circuit | 84.2% | -8.95 | -9.01 |
| propositional tree | 84.2% | -8.95 | -9.01 |
| unrolled tree | 81.6% | -9.38 | -9.38 |

### whole molecule

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|
| relational circuit | 73.7% | -46.85 | -40.84 |
| unrolled tree | 44.7% | -91.47 | -91.26 |

## Seconds per question

Wall-clock time from asking to the answer or the refusal. The *first ask* of a cause includes fitting that cause's own support-deterministic model; *asked again* repeats the question with every model fitted, so only grounding (for the relational circuit), verification and backdoor adjustment remain. A refusal is fast when it is a schema check; a relational answer draws Monte-Carlo samples for every count the query leaves open and grounds one part template per sampled value, which is where its time goes.

| question | relational circuit, first ask | relational circuit, asked again | propositional tree, first ask | propositional tree, asked again | unrolled tree, first ask | unrolled tree, asked again | scalars-only tree, first ask | scalars-only tree, asked again | regression adjustment, first ask | regression adjustment, asked again | neural adjustment, first ask | neural adjustment, asked again |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| branching_atom_count_causes_mutagenicity_adjusting_indicator_1 | 9.07 | 3.17 | 3.82 | 1.90 | 35.08 | 3.81 | 0.01 | 0.00 | 0.02 | - | 0.19 | - |
| branching_atom_count_causes_mutagenicity_adjusting_atom_count | 12.70 | 12.88 | 14.05 | 13.88 | 19.42 | 19.31 | 0.00 | 0.00 | 0.02 | - | 0.15 | - |
| branching_atom_count_causes_mutagenicity_adjusting_indicator_1_and_atom_count | 25.38 | 25.33 | 28.39 | 28.46 | 35.35 | 36.27 | 0.00 | 0.01 | 0.03 | - | 0.12 | - |
| aromatic_bond_count_causes_mutagenicity_adjusting_indicator_1 | 8.61 | 2.15 | 3.66 | 1.02 | 27.74 | 3.44 | 0.00 | 0.00 | 0.02 | - | 0.33 | - |
| aromatic_bond_count_causes_mutagenicity_adjusting_atom_count | 8.41 | 9.43 | 7.29 | 7.50 | 9.96 | 11.41 | 0.00 | 0.00 | 0.03 | - | 0.33 | - |
| aromatic_bond_count_causes_mutagenicity_adjusting_indicator_1_and_atom_count | 16.26 | 16.76 | 14.65 | 14.73 | 21.03 | 20.91 | 0.01 | 0.00 | 0.02 | - | 0.33 | - |
| double_bond_count_causes_mutagenicity_adjusting_indicator_1 | 8.70 | 1.49 | 2.92 | 0.47 | 16.32 | 2.34 | 0.00 | 0.00 | 0.02 | - | 0.32 | - |
| double_bond_count_causes_mutagenicity_adjusting_atom_count | 5.07 | 5.43 | 2.87 | 3.24 | 4.55 | 4.50 | 0.00 | 0.00 | 0.02 | - | 0.32 | - |
| double_bond_count_causes_mutagenicity_adjusting_indicator_1_and_atom_count | 7.99 | 8.11 | 5.54 | 5.51 | 7.53 | 8.35 | 0.00 | 0.00 | 0.02 | - | 0.02 | - |
| indicator_causes_mutagenicity_adjusting_logp | 8.79 | 1.27 | 2.05 | 0.16 | 6.95 | 0.59 | 0.18 | 0.15 | 0.01 | - | 0.02 | - |
| indicator_causes_mutagenicity_adjusting_branching_atom_count | 1.93 | 2.16 | 0.47 | 0.51 | 0.86 | 0.97 | 0.00 | 0.01 | 0.01 | - | 0.02 | - |
| indicator_causes_carbon_atom_0 | 2.71 | 2.77 | 0.00 | 0.01 | 0.49 | 0.57 | 0.00 | 0.00 | 0.01 | - | 0.87 | - |
| branching_atom_count_causes_terminal_atom_0 | 2.63 | 2.15 | 0.00 | 0.00 | 2.26 | 3.45 | 0.00 | 0.00 | 0.01 | - | 0.96 | - |
| element_causes_terminal_atom_0 | 18.97 | 11.99 | 0.00 | 0.00 | 5.05 | 0.35 | 0.00 | 0.00 | 0.01 | - | 0.78 | - |

## Does the order of the parts matter?

Every molecule's parts were put in a random order, 20 times over, and each time the pipelines that model the parts were refitted on the same split and asked the questions about parts again; the parts in the order the dataset lists them is the baseline every reordering is measured against. A relational circuit treats the atoms as exchangeable, so nothing about it can depend on the order; an unrolled table's column `atoms[0]` holds a different atom of every molecule after each reordering. Per question and pipeline: how many reorderings were answered; over the cause regions every answered ordering distinguishes, the mean standard deviation and the widest range of the adjusted probability; the share of reorderings whose most effective region is not the dataset-order one; and the share whose trend changed sign.

| question | pipeline | reorderings answered | mean sd of adjusted P(effect) | widest range | argmax moved | trend sign flipped |
|---|---|---|---|---|---|---|
| indicator_causes_carbon_atom_0 | relational circuit | 20 of 20 | 0.000 | 0.00 | 0.0% | - |
| branching_atom_count_causes_terminal_atom_0 | relational circuit | 20 of 20 | 0.000 | 0.00 | 0.0% | 0.0% |
| element_causes_terminal_atom_0 | relational circuit | 20 of 20 | 0.000 | 0.00 | 0.0% | - |
| indicator_causes_carbon_atom_0 | unrolled tree | 20 of 20 | 0.122 | 0.65 | 95.0% | - |
| branching_atom_count_causes_terminal_atom_0 | unrolled tree | 20 of 20 | 0.218 | 1.00 | - | - |
| element_causes_terminal_atom_0 | unrolled tree | 20 of 20 | 0.014 | 0.23 | - | - |

The whole-molecule likelihood of the same held-out molecules with the parts in the order the dataset lists them, and over the reorderings. The dataset's order is the order the molecules were drawn in, heavy atoms first and hydrogens last, so a column that addresses an atom by position addresses an element more often than chance, but never the same atom. *Largest drop* is how far below the dataset-order likelihood the worst reordering took each pipeline.

| pipeline | dataset order, coverage / mean log-likelihood | reorderings, coverage / mean log-likelihood (mean ± sd) | largest drop |
|---|---|---|---|
| relational circuit | 73.7% / -46.85 | 0.737 ± 0.000 / -46.85 ± 0.00 | 0.00 |
| unrolled tree | 44.7% / -91.47 | 0.049 ± 0.034 / -137.80 ± 8.08 | 56.50 |

## Over several splits

The comparison repeated over 5 random splits (seeds 0, 1, 2, 3, 4), mean ± standard deviation. The likelihoods are over the molecules every pipeline modelling the view covers.

### scalars

| pipeline | held-out coverage | mean log-likelihood (covered by all) |
|---|---|---|
| relational circuit | 0.932 ± 0.036 | -3.05 ± 0.18 |
| propositional tree | 0.932 ± 0.036 | -3.05 ± 0.18 |
| unrolled tree | 0.932 ± 0.036 | -3.15 ± 0.22 |
| scalars-only tree | 0.889 ± 0.061 | -2.92 ± 0.17 |

### scalars and counts

| pipeline | held-out coverage | mean log-likelihood (covered by all) |
|---|---|---|
| relational circuit | 0.763 ± 0.076 | -9.02 ± 0.32 |
| propositional tree | 0.763 ± 0.076 | -9.02 ± 0.32 |
| unrolled tree | 0.795 ± 0.073 | -9.41 ± 0.30 |

### whole molecule

| pipeline | held-out coverage | mean log-likelihood (covered by all) |
|---|---|---|
| relational circuit | 0.653 ± 0.125 | -52.16 ± 6.00 |
| unrolled tree | 0.426 ± 0.110 | -85.80 ± 4.76 |

Per question, how many splits each pipeline answered, and the mean ± standard deviation over the splits of its trend and of its contrast:

| question | relational circuit, answered | relational circuit, trend | relational circuit, contrast | propositional tree, answered | propositional tree, trend | propositional tree, contrast | unrolled tree, answered | unrolled tree, trend | unrolled tree, contrast | scalars-only tree, answered | scalars-only tree, trend | scalars-only tree, contrast | regression adjustment, answered | regression adjustment, trend | regression adjustment, contrast | neural adjustment, answered | neural adjustment, trend | neural adjustment, contrast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| branching_atom_count_causes_mutagenicity_adjusting_indicator_1 | 5 of 5 | 0.96 ± 0.04 | 0.77 ± 0.11 | 5 of 5 | 0.94 ± 0.02 | 0.77 ± 0.11 | 5 of 5 | 0.94 ± 0.02 | 0.77 ± 0.11 | 0 of 5 | - | - | 5 of 5 | 1.00 ± 0.00 | 0.63 ± 0.10 | 5 of 5 | 1.00 ± 0.00 | 0.90 ± 0.04 |
| branching_atom_count_causes_mutagenicity_adjusting_atom_count | 5 of 5 | 0.94 ± 0.02 | 0.86 ± 0.06 | 5 of 5 | 0.93 ± 0.02 | 0.86 ± 0.06 | 5 of 5 | 0.90 ± 0.03 | 0.86 ± 0.06 | 0 of 5 | - | - | 5 of 5 | 1.00 ± 0.00 | 0.92 ± 0.04 | 5 of 5 | 1.00 ± 0.00 | 0.92 ± 0.04 |
| branching_atom_count_causes_mutagenicity_adjusting_indicator_1_and_atom_count | 5 of 5 | 0.96 ± 0.02 | 0.86 ± 0.06 | 5 of 5 | 0.94 ± 0.05 | 0.86 ± 0.06 | 5 of 5 | 0.95 ± 0.02 | 0.86 ± 0.06 | 0 of 5 | - | - | 5 of 5 | 1.00 ± 0.00 | 0.93 ± 0.04 | 5 of 5 | 1.00 ± 0.00 | 0.88 ± 0.05 |
| aromatic_bond_count_causes_mutagenicity_adjusting_indicator_1 | 5 of 5 | 0.89 ± 0.05 | 0.77 ± 0.05 | 5 of 5 | 0.94 ± 0.04 | 0.77 ± 0.05 | 5 of 5 | 0.94 ± 0.07 | 0.77 ± 0.05 | 0 of 5 | - | - | 5 of 5 | 1.00 ± 0.00 | 0.58 ± 0.10 | 5 of 5 | 0.82 ± 0.15 | 0.12 ± 0.07 |
| aromatic_bond_count_causes_mutagenicity_adjusting_atom_count | 5 of 5 | 0.90 ± 0.06 | 0.71 ± 0.08 | 5 of 5 | 0.97 ± 0.04 | 0.71 ± 0.08 | 5 of 5 | 0.95 ± 0.07 | 0.72 ± 0.07 | 0 of 5 | - | - | 5 of 5 | 1.00 ± 0.00 | 0.50 ± 0.07 | 5 of 5 | 0.85 ± 0.13 | 0.38 ± 0.21 |
| aromatic_bond_count_causes_mutagenicity_adjusting_indicator_1_and_atom_count | 5 of 5 | 0.88 ± 0.07 | 0.71 ± 0.08 | 5 of 5 | 0.96 ± 0.04 | 0.71 ± 0.08 | 5 of 5 | 0.92 ± 0.08 | 0.72 ± 0.07 | 0 of 5 | - | - | 5 of 5 | 1.00 ± 0.00 | 0.13 ± 0.06 | 5 of 5 | 0.49 ± 0.76 | 0.22 ± 0.23 |
| double_bond_count_causes_mutagenicity_adjusting_indicator_1 | 5 of 5 | 0.96 ± 0.08 | 0.34 ± 0.07 | 5 of 5 | 0.96 ± 0.08 | 0.34 ± 0.07 | 5 of 5 | 0.96 ± 0.08 | 0.34 ± 0.07 | 0 of 5 | - | - | 5 of 5 | 1.00 ± 0.00 | 0.27 ± 0.08 | 5 of 5 | 1.00 ± 0.00 | 0.15 ± 0.14 |
| double_bond_count_causes_mutagenicity_adjusting_atom_count | 5 of 5 | 1.00 ± 0.00 | 0.32 ± 0.04 | 5 of 5 | 1.00 ± 0.00 | 0.33 ± 0.04 | 5 of 5 | 1.00 ± 0.00 | 0.31 ± 0.04 | 0 of 5 | - | - | 5 of 5 | 1.00 ± 0.00 | 0.14 ± 0.09 | 5 of 5 | 0.86 ± 0.20 | 0.12 ± 0.08 |
| double_bond_count_causes_mutagenicity_adjusting_indicator_1_and_atom_count | 5 of 5 | 1.00 ± 0.00 | 0.33 ± 0.07 | 5 of 5 | 1.00 ± 0.00 | 0.33 ± 0.06 | 5 of 5 | 1.00 ± 0.00 | 0.32 ± 0.06 | 0 of 5 | - | - | 5 of 5 | 1.00 ± 0.00 | 0.21 ± 0.08 | 5 of 5 | 0.60 ± 0.80 | 0.13 ± 0.16 |
| indicator_causes_mutagenicity_adjusting_logp | 5 of 5 | - | 0.64 ± 0.02 | 5 of 5 | - | 0.64 ± 0.02 | 5 of 5 | - | 0.64 ± 0.02 | 5 of 5 | - | 0.66 ± 0.02 | 5 of 5 | - | 0.48 ± 0.02 | 5 of 5 | - | 0.18 ± 0.15 |
| indicator_causes_mutagenicity_adjusting_branching_atom_count | 5 of 5 | - | 0.54 ± 0.04 | 5 of 5 | - | 0.54 ± 0.04 | 5 of 5 | - | 0.58 ± 0.03 | 0 of 5 | - | - | 5 of 5 | - | 0.22 ± 0.05 | 5 of 5 | - | 0.13 ± 0.12 |
| indicator_causes_carbon_atom_0 | 5 of 5 | - | 0.08 ± 0.03 | 0 of 5 | - | - | 5 of 5 | - | 0.00 ± 0.00 | 0 of 5 | - | - | 0 of 5 | - | - | 5 of 5 | - | 0.00 ± 0.00 |
| branching_atom_count_causes_terminal_atom_0 | 5 of 5 | -0.22 ± 0.25 | -0.08 ± 0.05 | 0 of 5 | - | - | 0 of 5 | - | - | 0 of 5 | - | - | 0 of 5 | - | - | 5 of 5 | 0.99 ± 0.01 | 0.15 ± 0.07 |
| element_causes_terminal_atom_0 | 5 of 5 | - | 1.00 ± 0.00 | 0 of 5 | - | - | 0 of 5 | - | - | 0 of 5 | - | - | 0 of 5 | - | - | 5 of 5 | - | 0.44 ± 0.17 |

## How much training data it takes

Every pipeline's plain model fitted on a growing share of the molecules and scored on the same held-out fifth, over 3 splits, mean ± standard deviation of the held-out coverage and of the mean log-likelihood over the covered molecules. The relational circuit's templates pool every part of every training molecule, where the unrolled tree sees one row per molecule. Every atom and every bond of every training molecule goes into the templates.

### scalars and counts

| training share | relational circuit, coverage | relational circuit, mean log-likelihood | propositional tree, coverage | propositional tree, mean log-likelihood | unrolled tree, coverage | unrolled tree, mean log-likelihood |
|---|---|---|---|---|---|---|
| 20.0% | 0.377 ± 0.174 | -7.55 ± 0.91 | 0.377 ± 0.174 | -7.55 ± 0.91 | 0.447 ± 0.057 | -8.28 ± 0.12 |
| 40.0% | 0.632 ± 0.064 | -9.02 ± 0.11 | 0.632 ± 0.064 | -9.02 ± 0.11 | 0.649 ± 0.033 | -9.12 ± 0.08 |
| 60.0% | 0.754 ± 0.033 | -8.91 ± 0.13 | 0.754 ± 0.033 | -8.91 ± 0.13 | 0.746 ± 0.025 | -9.44 ± 0.04 |
| 80.0% | 0.825 ± 0.012 | -9.07 ± 0.23 | 0.825 ± 0.012 | -9.07 ± 0.23 | 0.851 ± 0.033 | -9.35 ± 0.11 |

### whole molecule

| training share | relational circuit, coverage | relational circuit, mean log-likelihood | unrolled tree, coverage | unrolled tree, mean log-likelihood |
|---|---|---|---|---|
| 20.0% | 0.298 ± 0.118 | -54.46 ± 6.55 | 0.105 ± 0.037 | -69.15 ± 11.76 |
| 40.0% | 0.535 ± 0.054 | -43.81 ± 6.10 | 0.246 ± 0.033 | -69.10 ± 5.14 |
| 60.0% | 0.702 ± 0.045 | -49.31 ± 4.38 | 0.351 ± 0.087 | -81.08 ± 5.61 |
| 80.0% | 0.754 ± 0.012 | -53.22 ± 5.21 | 0.482 ± 0.110 | -84.41 ± 5.32 |

## How many grounding samples it takes

Inference on a grounded circuit is exact; grounding itself draws Monte-Carlo samples for every count the query leaves open and mixes one copy of the part templates per sampled value, so marginalising the open counts is a consistent estimate, not an exact sum. The relational circuit was fitted once and asked the same two questions with grounding drawing more and more samples; *deviation* is the largest difference, over the cause regions, from the answer at 32,000 samples, and *settled from* is the smallest number of samples from which every larger one stays within 0.01 of it.

### branching_atom_count_causes_mutagenicity_adjusting_atom_count

Settled from 50 samples.

| samples | answered | deviation from reference | seconds |
|---|---|---|---|
| 50 | answered | 0.000 | 18.8 |
| 200 | answered | 0.000 | 12.5 |
| 1,000 | answered | 0.000 | 11.5 |
| 2,000 | answered | 0.000 | 12.0 |
| 8,000 | answered | 0.000 | 12.0 |
| 32,000 | answered | 0.000 | 12.0 |

### branching_atom_count_causes_terminal_atom_0

Settled from 50 samples.

| samples | answered | deviation from reference | seconds |
|---|---|---|---|
| 50 | answered | 0.000 | 2.0 |
| 200 | answered | 0.000 | 2.1 |
| 1,000 | answered | 0.000 | 1.9 |
| 2,000 | answered | 0.000 | 2.0 |
| 8,000 | answered | 0.000 | 2.0 |
| 32,000 | answered | 0.000 | 2.0 |


## What the results show

- The relational circuit answered 14 of 14 questions.
- The propositional tree answered 11 of 14 questions, refusing `indicator_causes_carbon_atom_0` because the fitted table has no column for the queried variables; `branching_atom_count_causes_terminal_atom_0` because the fitted table has no column for the queried variables; `element_causes_terminal_atom_0` because the fitted table has no column for the queried variables.
- The unrolled tree answered 12 of 14 questions, refusing `branching_atom_count_causes_terminal_atom_0` because the effect has zero probability under every cause region; `element_causes_terminal_atom_0` because the effect has zero probability under every cause region.
- The scalars-only tree answered 1 of 14 questions, refusing `branching_atom_count_causes_mutagenicity_adjusting_indicator_1` because the fitted table has no column for the queried variables; `branching_atom_count_causes_mutagenicity_adjusting_atom_count` because the fitted table has no column for the queried variables; `branching_atom_count_causes_mutagenicity_adjusting_indicator_1_and_atom_count` because the fitted table has no column for the queried variables; `aromatic_bond_count_causes_mutagenicity_adjusting_indicator_1` because the fitted table has no column for the queried variables; `aromatic_bond_count_causes_mutagenicity_adjusting_atom_count` because the fitted table has no column for the queried variables; `aromatic_bond_count_causes_mutagenicity_adjusting_indicator_1_and_atom_count` because the fitted table has no column for the queried variables; `double_bond_count_causes_mutagenicity_adjusting_indicator_1` because the fitted table has no column for the queried variables; `double_bond_count_causes_mutagenicity_adjusting_atom_count` because the fitted table has no column for the queried variables; `double_bond_count_causes_mutagenicity_adjusting_indicator_1_and_atom_count` because the fitted table has no column for the queried variables; `indicator_causes_mutagenicity_adjusting_branching_atom_count` because the fitted table has no column for the queried variables; `indicator_causes_carbon_atom_0` because the fitted table has no column for the queried variables; `branching_atom_count_causes_terminal_atom_0` because the fitted table has no column for the queried variables; `element_causes_terminal_atom_0` because the fitted table has no column for the queried variables.
- The regression adjustment answered 11 of 14 questions, refusing `indicator_causes_carbon_atom_0` because the fitted table has no column for the queried variables; `branching_atom_count_causes_terminal_atom_0` because the fitted table has no column for the queried variables; `element_causes_terminal_atom_0` because the fitted table has no column for the queried variables.
- The neural adjustment answered 14 of 14 questions.
- On `branching_atom_count_causes_mutagenicity_adjusting_indicator_1`, the pipelines disagree on the most effective setting: the relational circuit and the propositional tree say 18 branching atoms (1.00, 1.00); the unrolled tree, the regression adjustment and the neural adjustment say 21 branching atoms (1.00, 0.97, 1.00).
- On `branching_atom_count_causes_mutagenicity_adjusting_atom_count`, the pipelines disagree on the most effective setting: the relational circuit, the propositional tree, the regression adjustment and the neural adjustment say 21 branching atoms (1.00, 1.00, 1.00, 1.00); the unrolled tree says 18 branching atoms (1.00).
- On `branching_atom_count_causes_mutagenicity_adjusting_indicator_1_and_atom_count`, the pipelines disagree on the most effective setting: the relational circuit, the propositional tree, the regression adjustment and the neural adjustment say 21 branching atoms (1.00, 1.00, 1.00, 0.99); the unrolled tree says 18 branching atoms (1.00).
- On `aromatic_bond_count_causes_mutagenicity_adjusting_indicator_1`, the pipelines disagree on the most effective setting: the relational circuit, the propositional tree and the unrolled tree say 17 aromatic bonds (1.00, 1.00, 1.00); the regression adjustment and the neural adjustment say 19 aromatic bonds (0.93, 0.94).
- On `aromatic_bond_count_causes_mutagenicity_adjusting_atom_count`, every pipeline that answered finds 19 aromatic bonds the most effective setting (adjusted probabilities: relational circuit 1.00, propositional tree 1.00, unrolled tree 1.00, regression adjustment 0.86, neural adjustment 0.95).
- On `aromatic_bond_count_causes_mutagenicity_adjusting_indicator_1_and_atom_count`, the pipelines disagree on the most effective setting: the relational circuit says 17 aromatic bonds (1.00); the propositional tree, the unrolled tree, the regression adjustment and the neural adjustment say 19 aromatic bonds (1.00, 1.00, 0.69, 0.86).
- On `double_bond_count_causes_mutagenicity_adjusting_indicator_1`, every pipeline that answered finds 4 double bonds the most effective setting (adjusted probabilities: relational circuit 0.79, propositional tree 0.79, unrolled tree 0.79, regression adjustment 0.74, neural adjustment 0.75).
- On `double_bond_count_causes_mutagenicity_adjusting_atom_count`, every pipeline that answered finds 4 double bonds the most effective setting (adjusted probabilities: relational circuit 0.85, propositional tree 0.85, unrolled tree 0.85, regression adjustment 0.70, neural adjustment 0.66).
- On `double_bond_count_causes_mutagenicity_adjusting_indicator_1_and_atom_count`, every pipeline that answered finds 4 double bonds the most effective setting (adjusted probabilities: relational circuit 0.82, propositional tree 0.82, unrolled tree 0.82, regression adjustment 0.72, neural adjustment 0.98).
- On `indicator_causes_mutagenicity_adjusting_logp`, the pipelines disagree on the most effective setting: the relational circuit, the propositional tree, the unrolled tree, the scalars-only tree and the regression adjustment say ind1 = True (0.95, 0.95, 0.95, 0.97, 0.90); the neural adjustment says ind1 = False (0.98).
- On `indicator_causes_mutagenicity_adjusting_branching_atom_count`, the pipelines disagree on the most effective setting: the relational circuit, the propositional tree, the unrolled tree and the regression adjustment say ind1 = True (0.94, 0.94, 0.95, 0.79); the neural adjustment says ind1 = False (0.98).
- On `indicator_causes_carbon_atom_0`, the pipelines disagree on the most effective setting: the relational circuit says ind1 = True (0.55); the unrolled tree and the neural adjustment say ind1 = False (1.00, 0.49).
- On `branching_atom_count_causes_terminal_atom_0`, the pipelines disagree on the most effective setting: the relational circuit says 10 branching atoms (0.45); the neural adjustment says 22 branching atoms (0.44).
- On `element_causes_terminal_atom_0`, every pipeline that answered finds atom 0 being of element cl the most effective setting (adjusted probabilities: relational circuit 1.00, neural adjustment 0.76).
- On the scalars, the scalars-only tree assigns the highest mean log-likelihood (-2.77, against relational circuit -3.05, propositional tree -3.05, unrolled tree -3.08) to the held-out molecules every pipeline covers; coverage: relational circuit 92.1%, propositional tree 92.1%, unrolled tree 92.1%, scalars-only tree 92.1%.
- On the scalars and counts, the relational circuit assigns the highest mean log-likelihood (-9.01, against propositional tree -9.01, unrolled tree -9.38) to the held-out molecules every pipeline covers; coverage: relational circuit 84.2%, propositional tree 84.2%, unrolled tree 81.6%.
- On the whole molecule, the relational circuit assigns the highest mean log-likelihood (-40.84, against unrolled tree -91.26) to the held-out molecules every pipeline covers; coverage: relational circuit 73.7%, unrolled tree 44.7%.
- Over 20 reorderings, the relational circuit's adjusted effect probabilities ranged by up to 0.00 and its most effective region moved in 0.0% of the reorderings; its whole-molecule mean log-likelihood fell by up to 0.00 from the dataset's own order.
- Over 20 reorderings, the unrolled tree's adjusted effect probabilities ranged by up to 1.00 and its most effective region moved in 95.0% of the reorderings; its whole-molecule mean log-likelihood fell by up to 56.50 from the dataset's own order.
- The relational circuit takes 7.51 seconds per answered question on average once its models are fitted.
- The propositional tree takes 7.03 seconds per answered question on average once its models are fitted.
- The unrolled tree takes 9.37 seconds per answered question on average once its models are fitted.
- The scalars-only tree takes 0.15 seconds per answered question on average once its models are fitted.

## How many branching atoms cause a molecule to be mutagenic, adjusting for the ind1 indicator?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.000 | 0.000 | [0.00, 0.35] |
| 8 † | 8 | 0.053 | 0.375 | 0.375 | [0.14, 0.69] |
| 9 † | 9 | 0.060 | 0.111 | 0.111 | [0.02, 0.44] |
| 10 | 11 | 0.073 | 0.182 | 0.182 | [0.05, 0.48] |
| 11 † | 5 | 0.033 | 0.200 | 0.200 | [0.04, 0.62] |
| 12 † | 4 | 0.027 | 0.250 | 0.250 | [0.05, 0.70] |
| 13 | 11 | 0.073 | 0.727 | 0.818 | [0.52, 0.95] |
| 14 | 17 | 0.113 | 0.588 | 0.629 | [0.40, 0.81] |
| 15 | 17 | 0.113 | 0.824 | 0.806 | [0.57, 0.93] |
| 16 | 12 | 0.080 | 0.833 | 0.849 | [0.57, 0.96] |
| 17 | 14 | 0.093 | 1.000 | 1.000 | [0.78, 1.00] |
| 18 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 19 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 20 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 21 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 22 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 24 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 25 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 17: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.000 | 0.000 | [0.00, 0.35] |
| 8 † | 8 | 0.053 | 0.375 | 0.375 | [0.14, 0.69] |
| 9 † | 9 | 0.060 | 0.111 | 0.111 | [0.02, 0.44] |
| 10 | 11 | 0.073 | 0.182 | 0.182 | [0.05, 0.48] |
| 11 † | 5 | 0.033 | 0.200 | 0.200 | [0.04, 0.62] |
| 12 † | 4 | 0.027 | 0.250 | 0.250 | [0.05, 0.70] |
| 13 | 11 | 0.073 | 0.727 | 0.818 | [0.52, 0.95] |
| 14 | 17 | 0.113 | 0.588 | 0.629 | [0.40, 0.81] |
| 15 | 17 | 0.113 | 0.824 | 0.806 | [0.57, 0.93] |
| 16 | 12 | 0.080 | 0.833 | 0.849 | [0.57, 0.96] |
| 17 | 14 | 0.093 | 1.000 | 1.000 | [0.78, 1.00] |
| 18 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 19 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 20 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 21 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 22 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 24 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 25 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 17: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.000 | 0.000 | [0.00, 0.35] |
| 8 † | 8 | 0.053 | 0.375 | 0.375 | [0.14, 0.69] |
| 9 † | 9 | 0.060 | 0.111 | 0.111 | [0.02, 0.44] |
| 10 | 11 | 0.073 | 0.182 | 0.182 | [0.05, 0.48] |
| 11 † | 5 | 0.033 | 0.200 | 0.200 | [0.04, 0.62] |
| 12 † | 4 | 0.027 | 0.250 | 0.250 | [0.05, 0.70] |
| 13 | 11 | 0.073 | 0.727 | 0.818 | [0.52, 0.95] |
| 14 | 17 | 0.113 | 0.588 | 0.629 | [0.40, 0.81] |
| 15 | 17 | 0.113 | 0.824 | 0.806 | [0.57, 0.93] |
| 16 | 12 | 0.080 | 0.833 | 0.849 | [0.57, 0.96] |
| 17 | 14 | 0.093 | 1.000 | 1.000 | [0.78, 1.00] |
| 18 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 19 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 20 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 21 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 22 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 24 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 25 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 17: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.000 | 0.179 | [0.04, 0.55] |
| 8 † | 8 | 0.053 | 0.375 | 0.240 | [0.07, 0.58] |
| 9 † | 9 | 0.060 | 0.111 | 0.312 | [0.11, 0.63] |
| 10 | 11 | 0.073 | 0.182 | 0.393 | [0.17, 0.67] |
| 11 † | 5 | 0.033 | 0.200 | 0.478 | [0.16, 0.82] |
| 12 † | 4 | 0.027 | 0.250 | 0.564 | [0.18, 0.88] |
| 13 | 11 | 0.073 | 0.727 | 0.647 | [0.36, 0.85] |
| 14 | 17 | 0.113 | 0.588 | 0.722 | [0.48, 0.88] |
| 15 | 17 | 0.113 | 0.824 | 0.789 | [0.55, 0.92] |
| 16 | 12 | 0.080 | 0.833 | 0.844 | [0.56, 0.96] |
| 17 | 14 | 0.093 | 1.000 | 0.887 | [0.63, 0.97] |
| 18 | 12 | 0.080 | 1.000 | 0.920 | [0.65, 0.99] |
| 19 † | 4 | 0.027 | 1.000 | 0.944 | [0.46, 1.00] |
| 20 † | 1 | 0.007 | 1.000 | 0.962 | [0.19, 1.00] |
| 21 | 12 | 0.080 | 1.000 | 0.974 | [0.72, 1.00] |
| 22 † | 3 | 0.020 | 1.000 | 0.982 | [0.42, 1.00] |
| 24 † | 2 | 0.013 | 1.000 | 0.992 | [0.34, 1.00] |
| 25 † | 1 | 0.007 | 1.000 | 0.995 | [0.20, 1.00] |

EQL's own `cause` search settles on 21: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.97).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.000 | 0.004 | [0.00, 0.36] |
| 8 † | 8 | 0.053 | 0.375 | 0.041 | [0.00, 0.38] |
| 9 † | 9 | 0.060 | 0.111 | 0.091 | [0.01, 0.41] |
| 10 | 11 | 0.073 | 0.182 | 0.147 | [0.04, 0.44] |
| 11 † | 5 | 0.033 | 0.200 | 0.218 | [0.04, 0.64] |
| 12 † | 4 | 0.027 | 0.250 | 0.297 | [0.06, 0.73] |
| 13 | 11 | 0.073 | 0.727 | 0.406 | [0.18, 0.68] |
| 14 | 17 | 0.113 | 0.588 | 0.539 | [0.32, 0.75] |
| 15 | 17 | 0.113 | 0.824 | 0.680 | [0.44, 0.85] |
| 16 | 12 | 0.080 | 0.833 | 0.807 | [0.52, 0.94] |
| 17 | 14 | 0.093 | 1.000 | 0.878 | [0.62, 0.97] |
| 18 | 12 | 0.080 | 1.000 | 0.928 | [0.66, 0.99] |
| 19 † | 4 | 0.027 | 1.000 | 0.969 | [0.48, 1.00] |
| 20 † | 1 | 0.007 | 1.000 | 0.991 | [0.20, 1.00] |
| 21 | 12 | 0.080 | 1.000 | 0.998 | [0.75, 1.00] |
| 22 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 24 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 25 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 21: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).


## How many branching atoms cause a molecule to be mutagenic, adjusting for the number of atoms?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.000 | 0.000 | [0.00, 0.35] |
| 8 † | 8 | 0.053 | 0.375 | 0.176 | [0.04, 0.52] |
| 9 † | 9 | 0.060 | 0.111 | 0.063 | [0.01, 0.38] |
| 10 | 11 | 0.073 | 0.182 | 0.127 | [0.03, 0.42] |
| 11 † | 5 | 0.033 | 0.200 | 0.107 | [0.01, 0.54] |
| 12 † | 4 | 0.027 | 0.250 | 0.286 | [0.06, 0.72] |
| 13 | 11 | 0.073 | 0.727 | 0.796 | [0.50, 0.94] |
| 14 | 17 | 0.113 | 0.588 | 0.500 | [0.29, 0.71] |
| 15 | 17 | 0.113 | 0.824 | 0.684 | [0.45, 0.85] |
| 16 | 12 | 0.080 | 0.833 | 0.893 | [0.62, 0.98] |
| 17 | 14 | 0.093 | 1.000 | 1.000 | [0.78, 1.00] |
| 18 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 19 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 20 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 21 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 22 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 24 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 25 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 17: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.000 | 0.000 | [0.00, 0.35] |
| 8 † | 8 | 0.053 | 0.375 | 0.176 | [0.04, 0.52] |
| 9 † | 9 | 0.060 | 0.111 | 0.062 | [0.01, 0.38] |
| 10 | 11 | 0.073 | 0.182 | 0.127 | [0.03, 0.42] |
| 11 † | 5 | 0.033 | 0.200 | 0.107 | [0.01, 0.54] |
| 12 † | 4 | 0.027 | 0.250 | 0.286 | [0.06, 0.72] |
| 13 | 11 | 0.073 | 0.727 | 0.796 | [0.50, 0.94] |
| 14 | 17 | 0.113 | 0.588 | 0.500 | [0.29, 0.71] |
| 15 | 17 | 0.113 | 0.824 | 0.684 | [0.45, 0.85] |
| 16 | 12 | 0.080 | 0.833 | 0.893 | [0.62, 0.98] |
| 17 | 14 | 0.093 | 1.000 | 1.000 | [0.78, 1.00] |
| 18 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 19 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 20 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 21 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 22 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 24 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 25 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 17: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.000 | 0.000 | [0.00, 0.35] |
| 8 † | 8 | 0.053 | 0.375 | 0.176 | [0.04, 0.52] |
| 9 † | 9 | 0.060 | 0.111 | 0.062 | [0.01, 0.38] |
| 10 | 11 | 0.073 | 0.182 | 0.127 | [0.03, 0.42] |
| 11 † | 5 | 0.033 | 0.200 | 0.107 | [0.01, 0.54] |
| 12 † | 4 | 0.027 | 0.250 | 0.286 | [0.06, 0.72] |
| 13 | 11 | 0.073 | 0.727 | 0.796 | [0.50, 0.94] |
| 14 | 17 | 0.113 | 0.588 | 0.500 | [0.29, 0.71] |
| 15 | 17 | 0.113 | 0.824 | 0.684 | [0.45, 0.85] |
| 16 | 12 | 0.080 | 0.833 | 0.893 | [0.62, 0.98] |
| 17 | 14 | 0.093 | 1.000 | 1.000 | [0.78, 1.00] |
| 18 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 19 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 20 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 21 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 22 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 24 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 25 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 17: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.000 | 0.005 | [0.00, 0.36] |
| 8 † | 8 | 0.053 | 0.375 | 0.020 | [0.00, 0.35] |
| 9 † | 9 | 0.060 | 0.111 | 0.057 | [0.01, 0.37] |
| 10 | 11 | 0.073 | 0.182 | 0.120 | [0.03, 0.41] |
| 11 † | 5 | 0.033 | 0.200 | 0.204 | [0.04, 0.63] |
| 12 † | 4 | 0.027 | 0.250 | 0.297 | [0.06, 0.73] |
| 13 | 11 | 0.073 | 0.727 | 0.409 | [0.18, 0.68] |
| 14 | 17 | 0.113 | 0.588 | 0.547 | [0.32, 0.75] |
| 15 | 17 | 0.113 | 0.824 | 0.693 | [0.46, 0.86] |
| 16 | 12 | 0.080 | 0.833 | 0.812 | [0.53, 0.94] |
| 17 | 14 | 0.093 | 1.000 | 0.889 | [0.64, 0.97] |
| 18 | 12 | 0.080 | 1.000 | 0.936 | [0.67, 0.99] |
| 19 † | 4 | 0.027 | 1.000 | 0.971 | [0.48, 1.00] |
| 20 † | 1 | 0.007 | 1.000 | 0.991 | [0.20, 1.00] |
| 21 | 12 | 0.080 | 1.000 | 0.998 | [0.75, 1.00] |
| 22 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 24 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 25 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 21: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.000 | 0.004 | [0.00, 0.36] |
| 8 † | 8 | 0.053 | 0.375 | 0.020 | [0.00, 0.35] |
| 9 † | 9 | 0.060 | 0.111 | 0.061 | [0.01, 0.38] |
| 10 | 11 | 0.073 | 0.182 | 0.127 | [0.03, 0.42] |
| 11 † | 5 | 0.033 | 0.200 | 0.207 | [0.04, 0.63] |
| 12 † | 4 | 0.027 | 0.250 | 0.293 | [0.06, 0.73] |
| 13 | 11 | 0.073 | 0.727 | 0.402 | [0.18, 0.68] |
| 14 | 17 | 0.113 | 0.588 | 0.537 | [0.32, 0.74] |
| 15 | 17 | 0.113 | 0.824 | 0.685 | [0.45, 0.85] |
| 16 | 12 | 0.080 | 0.833 | 0.809 | [0.53, 0.94] |
| 17 | 14 | 0.093 | 1.000 | 0.887 | [0.63, 0.97] |
| 18 | 12 | 0.080 | 1.000 | 0.937 | [0.67, 0.99] |
| 19 † | 4 | 0.027 | 1.000 | 0.974 | [0.48, 1.00] |
| 20 † | 1 | 0.007 | 1.000 | 0.994 | [0.20, 1.00] |
| 21 | 12 | 0.080 | 1.000 | 0.999 | [0.76, 1.00] |
| 22 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 24 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 25 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 21: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).


## How many branching atoms cause a molecule to be mutagenic, adjusting for the ind1 indicator and the number of atoms?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.000 | 0.000 | [0.00, 0.35] |
| 8 † | 8 | 0.053 | 0.375 | 0.176 | [0.04, 0.52] |
| 9 † | 9 | 0.060 | 0.111 | 0.063 | [0.01, 0.38] |
| 10 | 11 | 0.073 | 0.182 | 0.127 | [0.03, 0.42] |
| 11 † | 5 | 0.033 | 0.200 | 0.115 | [0.01, 0.55] |
| 12 † | 4 | 0.027 | 0.250 | 0.300 | [0.06, 0.73] |
| 13 | 11 | 0.073 | 0.727 | 0.703 | [0.41, 0.89] |
| 14 | 17 | 0.113 | 0.588 | 0.638 | [0.40, 0.82] |
| 15 | 17 | 0.113 | 0.824 | 0.839 | [0.61, 0.95] |
| 16 | 12 | 0.080 | 0.833 | 0.904 | [0.63, 0.98] |
| 17 | 14 | 0.093 | 1.000 | 1.000 | [0.78, 1.00] |
| 18 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 19 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 20 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 21 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 22 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 24 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 25 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 15: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.84).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.000 | 0.000 | [0.00, 0.35] |
| 8 † | 8 | 0.053 | 0.375 | 0.176 | [0.04, 0.52] |
| 9 † | 9 | 0.060 | 0.111 | 0.062 | [0.01, 0.38] |
| 10 | 11 | 0.073 | 0.182 | 0.127 | [0.03, 0.42] |
| 11 † | 5 | 0.033 | 0.200 | 0.115 | [0.01, 0.55] |
| 12 † | 4 | 0.027 | 0.250 | 0.300 | [0.06, 0.73] |
| 13 | 11 | 0.073 | 0.727 | 0.703 | [0.41, 0.89] |
| 14 | 17 | 0.113 | 0.588 | 0.638 | [0.40, 0.82] |
| 15 | 17 | 0.113 | 0.824 | 0.839 | [0.61, 0.95] |
| 16 | 12 | 0.080 | 0.833 | 0.904 | [0.63, 0.98] |
| 17 | 14 | 0.093 | 1.000 | 1.000 | [0.78, 1.00] |
| 18 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 19 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 20 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 21 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 22 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 24 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 25 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 15: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.84).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.000 | 0.000 | [0.00, 0.35] |
| 8 † | 8 | 0.053 | 0.375 | 0.176 | [0.04, 0.52] |
| 9 † | 9 | 0.060 | 0.111 | 0.062 | [0.01, 0.38] |
| 10 | 11 | 0.073 | 0.182 | 0.127 | [0.03, 0.42] |
| 11 † | 5 | 0.033 | 0.200 | 0.115 | [0.01, 0.55] |
| 12 † | 4 | 0.027 | 0.250 | 0.300 | [0.06, 0.73] |
| 13 | 11 | 0.073 | 0.727 | 0.703 | [0.41, 0.89] |
| 14 | 17 | 0.113 | 0.588 | 0.638 | [0.40, 0.82] |
| 15 | 17 | 0.113 | 0.824 | 0.839 | [0.61, 0.95] |
| 16 | 12 | 0.080 | 0.833 | 0.904 | [0.63, 0.98] |
| 17 | 14 | 0.093 | 1.000 | 1.000 | [0.78, 1.00] |
| 18 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 19 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 20 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 21 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 22 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 24 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 25 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 15: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.84).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.000 | 0.008 | [0.00, 0.36] |
| 8 † | 8 | 0.053 | 0.375 | 0.022 | [0.00, 0.35] |
| 9 † | 9 | 0.060 | 0.111 | 0.055 | [0.01, 0.37] |
| 10 | 11 | 0.073 | 0.182 | 0.116 | [0.02, 0.41] |
| 11 † | 5 | 0.033 | 0.200 | 0.209 | [0.04, 0.63] |
| 12 † | 4 | 0.027 | 0.250 | 0.331 | [0.07, 0.75] |
| 13 | 11 | 0.073 | 0.727 | 0.471 | [0.22, 0.73] |
| 14 | 17 | 0.113 | 0.588 | 0.617 | [0.39, 0.80] |
| 15 | 17 | 0.113 | 0.824 | 0.749 | [0.51, 0.89] |
| 16 | 12 | 0.080 | 0.833 | 0.851 | [0.57, 0.96] |
| 17 | 14 | 0.093 | 1.000 | 0.918 | [0.67, 0.98] |
| 18 | 12 | 0.080 | 1.000 | 0.960 | [0.70, 1.00] |
| 19 † | 4 | 0.027 | 1.000 | 0.984 | [0.49, 1.00] |
| 20 † | 1 | 0.007 | 1.000 | 0.994 | [0.20, 1.00] |
| 21 | 12 | 0.080 | 1.000 | 0.998 | [0.75, 1.00] |
| 22 † | 3 | 0.020 | 1.000 | 0.999 | [0.44, 1.00] |
| 24 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 25 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 21: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.000 | 0.023 | [0.00, 0.38] |
| 8 † | 8 | 0.053 | 0.375 | 0.045 | [0.00, 0.38] |
| 9 † | 9 | 0.060 | 0.111 | 0.086 | [0.01, 0.41] |
| 10 | 11 | 0.073 | 0.182 | 0.156 | [0.04, 0.45] |
| 11 † | 5 | 0.033 | 0.200 | 0.264 | [0.06, 0.67] |
| 12 † | 4 | 0.027 | 0.250 | 0.403 | [0.10, 0.80] |
| 13 | 11 | 0.073 | 0.727 | 0.555 | [0.29, 0.79] |
| 14 | 17 | 0.113 | 0.588 | 0.698 | [0.46, 0.86] |
| 15 | 17 | 0.113 | 0.824 | 0.812 | [0.58, 0.93] |
| 16 | 12 | 0.080 | 0.833 | 0.892 | [0.62, 0.98] |
| 17 | 14 | 0.093 | 1.000 | 0.939 | [0.70, 0.99] |
| 18 | 12 | 0.080 | 1.000 | 0.965 | [0.71, 1.00] |
| 19 † | 4 | 0.027 | 1.000 | 0.978 | [0.49, 1.00] |
| 20 † | 1 | 0.007 | 1.000 | 0.984 | [0.20, 1.00] |
| 21 | 12 | 0.080 | 1.000 | 0.988 | [0.74, 1.00] |
| 22 † | 3 | 0.020 | 1.000 | 0.990 | [0.43, 1.00] |
| 24 † | 2 | 0.013 | 1.000 | 0.993 | [0.34, 1.00] |
| 25 † | 1 | 0.007 | 1.000 | 0.994 | [0.20, 1.00] |

EQL's own `cause` search settles on 21: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.99).


## How many aromatic bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 5 † | 1 | 0.007 | 0.000 | 0.000 | [0.00, 0.79] |
| 6 | 25 | 0.167 | 0.240 | 0.240 | [0.11, 0.43] |
| 10 † | 8 | 0.053 | 0.250 | 0.250 | [0.07, 0.59] |
| 11 | 15 | 0.100 | 0.333 | 0.651 | [0.40, 0.84] |
| 12 | 52 | 0.347 | 0.712 | 0.693 | [0.56, 0.80] |
| 14 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 15 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 16 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 17 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 18 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 19 | 18 | 0.120 | 1.000 | 1.000 | [0.82, 1.00] |
| 22 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 24 † | 6 | 0.040 | 1.000 | 1.000 | [0.61, 1.00] |
| 26 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 30 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 12: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.69).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 5 † | 1 | 0.007 | 0.000 | 0.000 | [0.00, 0.79] |
| 6 | 25 | 0.167 | 0.240 | 0.240 | [0.11, 0.43] |
| 10 † | 8 | 0.053 | 0.250 | 0.250 | [0.07, 0.59] |
| 11 | 15 | 0.100 | 0.333 | 0.651 | [0.40, 0.84] |
| 12 | 52 | 0.347 | 0.712 | 0.693 | [0.56, 0.80] |
| 14 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 15 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 16 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 17 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 18 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 19 | 18 | 0.120 | 1.000 | 1.000 | [0.82, 1.00] |
| 22 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 24 † | 6 | 0.040 | 1.000 | 1.000 | [0.61, 1.00] |
| 26 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 30 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 12: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.69).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 5 † | 1 | 0.007 | 0.000 | 0.000 | [0.00, 0.79] |
| 6 | 25 | 0.167 | 0.240 | 0.240 | [0.11, 0.43] |
| 10 † | 8 | 0.053 | 0.250 | 0.250 | [0.07, 0.59] |
| 11 | 15 | 0.100 | 0.333 | 0.651 | [0.40, 0.84] |
| 12 | 52 | 0.347 | 0.712 | 0.693 | [0.56, 0.80] |
| 14 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 15 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 16 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 17 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 18 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 19 | 18 | 0.120 | 1.000 | 1.000 | [0.82, 1.00] |
| 22 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 24 † | 6 | 0.040 | 1.000 | 1.000 | [0.61, 1.00] |
| 26 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 30 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 12: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.69).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 5 † | 1 | 0.007 | 0.000 | 0.326 | [0.02, 0.90] |
| 6 | 25 | 0.167 | 0.240 | 0.378 | [0.22, 0.57] |
| 10 † | 8 | 0.053 | 0.250 | 0.595 | [0.28, 0.85] |
| 11 | 15 | 0.100 | 0.333 | 0.648 | [0.40, 0.84] |
| 12 | 52 | 0.347 | 0.712 | 0.697 | [0.56, 0.80] |
| 14 † | 1 | 0.007 | 1.000 | 0.786 | [0.13, 0.99] |
| 15 † | 3 | 0.020 | 1.000 | 0.823 | [0.30, 0.98] |
| 16 † | 4 | 0.027 | 1.000 | 0.856 | [0.38, 0.98] |
| 17 | 12 | 0.080 | 1.000 | 0.885 | [0.61, 0.97] |
| 18 † | 1 | 0.007 | 1.000 | 0.908 | [0.17, 1.00] |
| 19 | 18 | 0.120 | 1.000 | 0.928 | [0.72, 0.98] |
| 22 † | 1 | 0.007 | 1.000 | 0.966 | [0.19, 1.00] |
| 24 † | 6 | 0.040 | 1.000 | 0.980 | [0.59, 1.00] |
| 26 † | 2 | 0.013 | 1.000 | 0.988 | [0.33, 1.00] |
| 30 † | 1 | 0.007 | 1.000 | 0.996 | [0.20, 1.00] |

EQL's own `cause` search settles on 19: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.93).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 5 † | 1 | 0.007 | 0.000 | 0.871 | [0.16, 1.00] |
| 6 | 25 | 0.167 | 0.240 | 0.807 | [0.62, 0.92] |
| 10 † | 8 | 0.053 | 0.250 | 0.629 | [0.31, 0.87] |
| 11 | 15 | 0.100 | 0.333 | 0.637 | [0.39, 0.83] |
| 12 | 52 | 0.347 | 0.712 | 0.654 | [0.52, 0.77] |
| 14 † | 1 | 0.007 | 1.000 | 0.723 | [0.11, 0.98] |
| 15 † | 3 | 0.020 | 1.000 | 0.777 | [0.27, 0.97] |
| 16 † | 4 | 0.027 | 1.000 | 0.834 | [0.36, 0.98] |
| 17 | 12 | 0.080 | 1.000 | 0.887 | [0.61, 0.98] |
| 18 † | 1 | 0.007 | 1.000 | 0.920 | [0.18, 1.00] |
| 19 | 18 | 0.120 | 1.000 | 0.939 | [0.74, 0.99] |
| 22 † | 1 | 0.007 | 1.000 | 0.990 | [0.20, 1.00] |
| 24 † | 6 | 0.040 | 1.000 | 0.999 | [0.61, 1.00] |
| 26 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 30 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 19: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.94).


## How many aromatic bonds cause a molecule to be mutagenic, adjusting for the number of atoms?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 5 † | 1 | 0.007 | 0.000 | 0.000 | [0.00, 0.79] |
| 6 | 25 | 0.167 | 0.240 | 0.269 | [0.14, 0.46] |
| 10 † | 8 | 0.053 | 0.250 | 0.328 | [0.11, 0.66] |
| 11 | 15 | 0.100 | 0.333 | 0.374 | [0.18, 0.62] |
| 12 | 52 | 0.347 | 0.712 | 0.736 | [0.60, 0.84] |
| 14 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 15 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 16 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 17 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 18 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 19 | 18 | 0.120 | 1.000 | 1.000 | [0.82, 1.00] |
| 22 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 24 † | 6 | 0.040 | 1.000 | 1.000 | [0.61, 1.00] |
| 26 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 30 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 12: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.74).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 5 † | 1 | 0.007 | 0.000 | 0.000 | [0.00, 0.79] |
| 6 | 25 | 0.167 | 0.240 | 0.269 | [0.14, 0.46] |
| 10 † | 8 | 0.053 | 0.250 | 0.328 | [0.11, 0.66] |
| 11 | 15 | 0.100 | 0.333 | 0.374 | [0.18, 0.62] |
| 12 | 52 | 0.347 | 0.712 | 0.736 | [0.60, 0.84] |
| 14 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 15 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 16 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 17 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 18 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 19 | 18 | 0.120 | 1.000 | 1.000 | [0.82, 1.00] |
| 22 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 24 † | 6 | 0.040 | 1.000 | 1.000 | [0.61, 1.00] |
| 26 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 30 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 12: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.74).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 5 † | 1 | 0.007 | 0.000 | 0.000 | [0.00, 0.79] |
| 6 | 25 | 0.167 | 0.240 | 0.299 | [0.16, 0.50] |
| 10 † | 8 | 0.053 | 0.250 | 0.328 | [0.11, 0.66] |
| 11 | 15 | 0.100 | 0.333 | 0.374 | [0.18, 0.62] |
| 12 | 52 | 0.347 | 0.712 | 0.746 | [0.61, 0.84] |
| 14 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 15 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 16 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 17 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 18 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 19 | 18 | 0.120 | 1.000 | 1.000 | [0.82, 1.00] |
| 22 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 24 † | 6 | 0.040 | 1.000 | 1.000 | [0.61, 1.00] |
| 26 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 30 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 12: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.75).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 5 † | 1 | 0.007 | 0.000 | 0.421 | [0.04, 0.93] |
| 6 | 25 | 0.167 | 0.240 | 0.458 | [0.28, 0.65] |
| 10 † | 8 | 0.053 | 0.250 | 0.607 | [0.29, 0.85] |
| 11 | 15 | 0.100 | 0.333 | 0.642 | [0.39, 0.83] |
| 12 | 52 | 0.347 | 0.712 | 0.676 | [0.54, 0.79] |
| 14 † | 1 | 0.007 | 1.000 | 0.740 | [0.11, 0.98] |
| 15 † | 3 | 0.020 | 1.000 | 0.769 | [0.27, 0.97] |
| 16 † | 4 | 0.027 | 1.000 | 0.796 | [0.33, 0.97] |
| 17 | 12 | 0.080 | 1.000 | 0.821 | [0.54, 0.95] |
| 18 † | 1 | 0.007 | 1.000 | 0.844 | [0.15, 0.99] |
| 19 | 18 | 0.120 | 1.000 | 0.864 | [0.64, 0.96] |
| 22 † | 1 | 0.007 | 1.000 | 0.914 | [0.17, 1.00] |
| 24 † | 6 | 0.040 | 1.000 | 0.939 | [0.54, 1.00] |
| 26 † | 2 | 0.013 | 1.000 | 0.957 | [0.31, 1.00] |
| 30 † | 1 | 0.007 | 1.000 | 0.979 | [0.20, 1.00] |

EQL's own `cause` search settles on 19: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.86).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 5 † | 1 | 0.007 | 0.000 | 0.746 | [0.12, 0.98] |
| 6 | 25 | 0.167 | 0.240 | 0.708 | [0.51, 0.85] |
| 10 † | 8 | 0.053 | 0.250 | 0.550 | [0.25, 0.82] |
| 11 | 15 | 0.100 | 0.333 | 0.589 | [0.35, 0.79] |
| 12 | 52 | 0.347 | 0.712 | 0.688 | [0.55, 0.80] |
| 14 † | 1 | 0.007 | 1.000 | 0.886 | [0.16, 1.00] |
| 15 † | 3 | 0.020 | 1.000 | 0.916 | [0.37, 1.00] |
| 16 † | 4 | 0.027 | 1.000 | 0.928 | [0.44, 1.00] |
| 17 | 12 | 0.080 | 1.000 | 0.938 | [0.67, 0.99] |
| 18 † | 1 | 0.007 | 1.000 | 0.947 | [0.19, 1.00] |
| 19 | 18 | 0.120 | 1.000 | 0.955 | [0.76, 0.99] |
| 22 † | 1 | 0.007 | 1.000 | 0.981 | [0.20, 1.00] |
| 24 † | 6 | 0.040 | 1.000 | 0.992 | [0.60, 1.00] |
| 26 † | 2 | 0.013 | 1.000 | 0.998 | [0.34, 1.00] |
| 30 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 19: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.95).


## How many aromatic bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator and the number of atoms?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 5 † | 1 | 0.007 | 0.000 | 0.000 | [0.00, 0.79] |
| 6 | 25 | 0.167 | 0.240 | 0.269 | [0.14, 0.46] |
| 10 † | 8 | 0.053 | 0.250 | 0.114 | [0.02, 0.46] |
| 11 | 15 | 0.100 | 0.333 | 0.387 | [0.19, 0.63] |
| 12 | 52 | 0.347 | 0.712 | 0.808 | [0.68, 0.89] |
| 14 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 15 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 16 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 17 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 18 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 19 | 18 | 0.120 | 1.000 | 1.000 | [0.82, 1.00] |
| 22 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 24 † | 6 | 0.040 | 1.000 | 1.000 | [0.61, 1.00] |
| 26 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 30 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 12: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.81).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 5 † | 1 | 0.007 | 0.000 | 0.000 | [0.00, 0.79] |
| 6 | 25 | 0.167 | 0.240 | 0.269 | [0.14, 0.46] |
| 10 † | 8 | 0.053 | 0.250 | 0.114 | [0.02, 0.46] |
| 11 | 15 | 0.100 | 0.333 | 0.387 | [0.19, 0.63] |
| 12 | 52 | 0.347 | 0.712 | 0.808 | [0.68, 0.89] |
| 14 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 15 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 16 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 17 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 18 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 19 | 18 | 0.120 | 1.000 | 1.000 | [0.82, 1.00] |
| 22 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 24 † | 6 | 0.040 | 1.000 | 1.000 | [0.61, 1.00] |
| 26 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 30 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 12: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.81).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 5 † | 1 | 0.007 | 0.000 | 0.000 | [0.00, 0.79] |
| 6 | 25 | 0.167 | 0.240 | 0.299 | [0.16, 0.50] |
| 10 † | 8 | 0.053 | 0.250 | 0.114 | [0.02, 0.46] |
| 11 | 15 | 0.100 | 0.333 | 0.387 | [0.19, 0.63] |
| 12 | 52 | 0.347 | 0.712 | 0.800 | [0.67, 0.89] |
| 14 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 15 † | 3 | 0.020 | 1.000 | 1.000 | [0.44, 1.00] |
| 16 † | 4 | 0.027 | 1.000 | 1.000 | [0.51, 1.00] |
| 17 | 12 | 0.080 | 1.000 | 1.000 | [0.76, 1.00] |
| 18 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 19 | 18 | 0.120 | 1.000 | 1.000 | [0.82, 1.00] |
| 22 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 24 † | 6 | 0.040 | 1.000 | 1.000 | [0.61, 1.00] |
| 26 † | 2 | 0.013 | 1.000 | 1.000 | [0.34, 1.00] |
| 30 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 12: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.80).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 5 † | 1 | 0.007 | 0.000 | 0.634 | [0.09, 0.97] |
| 6 | 25 | 0.167 | 0.240 | 0.638 | [0.44, 0.80] |
| 10 † | 8 | 0.053 | 0.250 | 0.655 | [0.33, 0.88] |
| 11 | 15 | 0.100 | 0.333 | 0.659 | [0.41, 0.84] |
| 12 | 52 | 0.347 | 0.712 | 0.663 | [0.53, 0.78] |
| 14 † | 1 | 0.007 | 1.000 | 0.671 | [0.10, 0.98] |
| 15 † | 3 | 0.020 | 1.000 | 0.675 | [0.21, 0.94] |
| 16 † | 4 | 0.027 | 1.000 | 0.679 | [0.25, 0.93] |
| 17 | 12 | 0.080 | 1.000 | 0.684 | [0.41, 0.87] |
| 18 † | 1 | 0.007 | 1.000 | 0.688 | [0.10, 0.98] |
| 19 | 18 | 0.120 | 1.000 | 0.692 | [0.46, 0.85] |
| 22 † | 1 | 0.007 | 1.000 | 0.704 | [0.10, 0.98] |
| 24 † | 6 | 0.040 | 1.000 | 0.711 | [0.33, 0.92] |
| 26 † | 2 | 0.013 | 1.000 | 0.719 | [0.18, 0.97] |
| 30 † | 1 | 0.007 | 1.000 | 0.735 | [0.11, 0.98] |

EQL's own `cause` search settles on 19: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.69).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 5 † | 1 | 0.007 | 0.000 | 0.775 | [0.13, 0.99] |
| 6 | 25 | 0.167 | 0.240 | 0.744 | [0.55, 0.87] |
| 10 † | 8 | 0.053 | 0.250 | 0.666 | [0.34, 0.89] |
| 11 | 15 | 0.100 | 0.333 | 0.653 | [0.40, 0.84] |
| 12 | 52 | 0.347 | 0.712 | 0.644 | [0.51, 0.76] |
| 14 † | 1 | 0.007 | 1.000 | 0.661 | [0.09, 0.97] |
| 15 † | 3 | 0.020 | 1.000 | 0.683 | [0.22, 0.94] |
| 16 † | 4 | 0.027 | 1.000 | 0.725 | [0.28, 0.95] |
| 17 | 12 | 0.080 | 1.000 | 0.776 | [0.49, 0.92] |
| 18 † | 1 | 0.007 | 1.000 | 0.824 | [0.14, 0.99] |
| 19 | 18 | 0.120 | 1.000 | 0.859 | [0.64, 0.95] |
| 22 † | 1 | 0.007 | 1.000 | 0.903 | [0.17, 1.00] |
| 24 † | 6 | 0.040 | 1.000 | 0.914 | [0.51, 0.99] |
| 26 † | 2 | 0.013 | 1.000 | 0.919 | [0.29, 1.00] |
| 30 † | 1 | 0.007 | 1.000 | 0.934 | [0.18, 1.00] |

EQL's own `cause` search settles on 19: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.86).


## How many double bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 2 | 86 | 0.573 | 0.547 | 0.537 | [0.43, 0.64] |
| 3 | 13 | 0.087 | 0.692 | 0.741 | [0.47, 0.90] |
| 4 | 33 | 0.220 | 0.848 | 0.794 | [0.63, 0.90] |
| 5 † | 2 | 0.013 | 0.500 | 0.547 | [0.11, 0.92] |
| 6 † | 9 | 0.060 | 0.889 | 0.924 | [0.61, 0.99] |
| 7 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 8 † | 5 | 0.033 | 0.800 | 0.887 | [0.45, 0.99] |
| 9 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 2: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.54).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 2 | 86 | 0.573 | 0.547 | 0.537 | [0.43, 0.64] |
| 3 | 13 | 0.087 | 0.692 | 0.741 | [0.47, 0.90] |
| 4 | 33 | 0.220 | 0.848 | 0.794 | [0.63, 0.90] |
| 5 † | 2 | 0.013 | 0.500 | 0.547 | [0.11, 0.92] |
| 6 † | 9 | 0.060 | 0.889 | 0.924 | [0.61, 0.99] |
| 7 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 8 † | 5 | 0.033 | 0.800 | 0.887 | [0.45, 0.99] |
| 9 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 2: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.54).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 2 | 86 | 0.573 | 0.547 | 0.538 | [0.43, 0.64] |
| 3 | 13 | 0.087 | 0.692 | 0.741 | [0.47, 0.90] |
| 4 | 33 | 0.220 | 0.848 | 0.794 | [0.63, 0.90] |
| 5 † | 2 | 0.013 | 0.500 | 0.547 | [0.11, 0.92] |
| 6 † | 9 | 0.060 | 0.889 | 0.924 | [0.61, 0.99] |
| 7 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 8 † | 5 | 0.033 | 0.800 | 0.887 | [0.45, 0.99] |
| 9 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 2: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.54).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 2 | 86 | 0.573 | 0.547 | 0.563 | [0.46, 0.66] |
| 3 | 13 | 0.087 | 0.692 | 0.650 | [0.39, 0.85] |
| 4 | 33 | 0.220 | 0.848 | 0.740 | [0.57, 0.86] |
| 5 † | 2 | 0.013 | 0.500 | 0.824 | [0.24, 0.99] |
| 6 † | 9 | 0.060 | 0.889 | 0.891 | [0.57, 0.98] |
| 7 † | 1 | 0.007 | 1.000 | 0.938 | [0.18, 1.00] |
| 8 † | 5 | 0.033 | 0.800 | 0.966 | [0.53, 1.00] |
| 9 † | 1 | 0.007 | 1.000 | 0.982 | [0.20, 1.00] |

EQL's own `cause` search settles on 4: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.74).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 2 | 86 | 0.573 | 0.547 | 0.596 | [0.49, 0.69] |
| 3 | 13 | 0.087 | 0.692 | 0.651 | [0.39, 0.85] |
| 4 | 33 | 0.220 | 0.848 | 0.753 | [0.59, 0.87] |
| 5 † | 2 | 0.013 | 0.500 | 0.858 | [0.25, 0.99] |
| 6 † | 9 | 0.060 | 0.889 | 0.928 | [0.61, 0.99] |
| 7 † | 1 | 0.007 | 1.000 | 0.966 | [0.19, 1.00] |
| 8 † | 5 | 0.033 | 0.800 | 0.985 | [0.55, 1.00] |
| 9 † | 1 | 0.007 | 1.000 | 0.995 | [0.20, 1.00] |

EQL's own `cause` search settles on 4: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.75).


## How many double bonds cause a molecule to be mutagenic, adjusting for the number of atoms?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 2 | 86 | 0.573 | 0.547 | 0.571 | [0.47, 0.67] |
| 3 | 13 | 0.087 | 0.692 | 0.734 | [0.46, 0.90] |
| 4 | 33 | 0.220 | 0.848 | 0.852 | [0.69, 0.94] |
| 5 † | 2 | 0.013 | 0.500 | 0.868 | [0.26, 0.99] |
| 6 † | 9 | 0.060 | 0.889 | 0.910 | [0.59, 0.99] |
| 7 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 8 † | 5 | 0.033 | 0.800 | 0.842 | [0.41, 0.98] |
| 9 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 2: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.57).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 2 | 86 | 0.573 | 0.547 | 0.570 | [0.46, 0.67] |
| 3 | 13 | 0.087 | 0.692 | 0.733 | [0.46, 0.90] |
| 4 | 33 | 0.220 | 0.848 | 0.851 | [0.69, 0.94] |
| 5 † | 2 | 0.013 | 0.500 | 0.867 | [0.26, 0.99] |
| 6 † | 9 | 0.060 | 0.889 | 0.909 | [0.59, 0.99] |
| 7 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 8 † | 5 | 0.033 | 0.800 | 0.843 | [0.41, 0.98] |
| 9 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 2: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.57).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 2 | 86 | 0.573 | 0.547 | 0.603 | [0.50, 0.70] |
| 3 | 13 | 0.087 | 0.692 | 0.733 | [0.46, 0.90] |
| 4 | 33 | 0.220 | 0.848 | 0.851 | [0.69, 0.94] |
| 5 † | 2 | 0.013 | 0.500 | 0.867 | [0.26, 0.99] |
| 6 † | 9 | 0.060 | 0.889 | 0.909 | [0.59, 0.99] |
| 7 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 8 † | 5 | 0.033 | 0.800 | 0.843 | [0.41, 0.98] |
| 9 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 2: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.60).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 2 | 86 | 0.573 | 0.547 | 0.632 | [0.53, 0.73] |
| 3 | 13 | 0.087 | 0.692 | 0.667 | [0.40, 0.86] |
| 4 | 33 | 0.220 | 0.848 | 0.702 | [0.53, 0.83] |
| 5 † | 2 | 0.013 | 0.500 | 0.734 | [0.19, 0.97] |
| 6 † | 9 | 0.060 | 0.889 | 0.765 | [0.44, 0.93] |
| 7 † | 1 | 0.007 | 1.000 | 0.794 | [0.13, 0.99] |
| 8 † | 5 | 0.033 | 0.800 | 0.820 | [0.39, 0.97] |
| 9 † | 1 | 0.007 | 1.000 | 0.845 | [0.15, 0.99] |

EQL's own `cause` search settles on 4: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.70).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 2 | 86 | 0.573 | 0.547 | 0.649 | [0.54, 0.74] |
| 3 | 13 | 0.087 | 0.692 | 0.645 | [0.38, 0.84] |
| 4 | 33 | 0.220 | 0.848 | 0.656 | [0.49, 0.79] |
| 5 † | 2 | 0.013 | 0.500 | 0.690 | [0.17, 0.96] |
| 6 † | 9 | 0.060 | 0.889 | 0.758 | [0.43, 0.93] |
| 7 † | 1 | 0.007 | 1.000 | 0.863 | [0.15, 1.00] |
| 8 † | 5 | 0.033 | 0.800 | 0.942 | [0.50, 1.00] |
| 9 † | 1 | 0.007 | 1.000 | 0.983 | [0.20, 1.00] |

EQL's own `cause` search settles on 4: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.66).


## How many double bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator and the number of atoms?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 2 | 86 | 0.573 | 0.547 | 0.582 | [0.48, 0.68] |
| 3 | 13 | 0.087 | 0.692 | 0.740 | [0.47, 0.90] |
| 4 | 33 | 0.220 | 0.848 | 0.820 | [0.66, 0.92] |
| 5 † | 2 | 0.013 | 0.500 | 0.860 | [0.26, 0.99] |
| 6 † | 9 | 0.060 | 0.889 | 0.948 | [0.63, 0.99] |
| 7 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 8 † | 5 | 0.033 | 0.800 | 0.906 | [0.47, 0.99] |
| 9 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 2: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.58).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 2 | 86 | 0.573 | 0.547 | 0.582 | [0.48, 0.68] |
| 3 | 13 | 0.087 | 0.692 | 0.738 | [0.47, 0.90] |
| 4 | 33 | 0.220 | 0.848 | 0.821 | [0.66, 0.92] |
| 5 † | 2 | 0.013 | 0.500 | 0.857 | [0.25, 0.99] |
| 6 † | 9 | 0.060 | 0.889 | 0.947 | [0.63, 0.99] |
| 7 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 8 † | 5 | 0.033 | 0.800 | 0.906 | [0.47, 0.99] |
| 9 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 2: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.58).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 2 | 86 | 0.573 | 0.547 | 0.599 | [0.49, 0.70] |
| 3 | 13 | 0.087 | 0.692 | 0.738 | [0.47, 0.90] |
| 4 | 33 | 0.220 | 0.848 | 0.821 | [0.66, 0.92] |
| 5 † | 2 | 0.013 | 0.500 | 0.852 | [0.25, 0.99] |
| 6 † | 9 | 0.060 | 0.889 | 0.936 | [0.62, 0.99] |
| 7 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |
| 8 † | 5 | 0.033 | 0.800 | 0.919 | [0.48, 0.99] |
| 9 † | 1 | 0.007 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 2: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.60).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 2 | 86 | 0.573 | 0.547 | 0.593 | [0.49, 0.69] |
| 3 | 13 | 0.087 | 0.692 | 0.657 | [0.39, 0.85] |
| 4 | 33 | 0.220 | 0.848 | 0.720 | [0.55, 0.84] |
| 5 † | 2 | 0.013 | 0.500 | 0.781 | [0.21, 0.98] |
| 6 † | 9 | 0.060 | 0.889 | 0.838 | [0.51, 0.96] |
| 7 † | 1 | 0.007 | 1.000 | 0.886 | [0.16, 1.00] |
| 8 † | 5 | 0.033 | 0.800 | 0.923 | [0.49, 0.99] |
| 9 † | 1 | 0.007 | 1.000 | 0.951 | [0.19, 1.00] |

EQL's own `cause` search settles on 4: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.72).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 2 | 86 | 0.573 | 0.547 | 0.971 | [0.91, 0.99] |
| 3 | 13 | 0.087 | 0.692 | 0.975 | [0.73, 1.00] |
| 4 | 33 | 0.220 | 0.848 | 0.978 | [0.86, 1.00] |
| 5 † | 2 | 0.013 | 0.500 | 0.980 | [0.33, 1.00] |
| 6 † | 9 | 0.060 | 0.889 | 0.983 | [0.68, 1.00] |
| 7 † | 1 | 0.007 | 1.000 | 0.984 | [0.20, 1.00] |
| 8 † | 5 | 0.033 | 0.800 | 0.986 | [0.55, 1.00] |
| 9 † | 1 | 0.007 | 1.000 | 0.987 | [0.20, 1.00] |

EQL's own `cause` search settles on 4: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.98).


## Does the ind1 indicator cause a molecule to be mutagenic, adjusting for its hydrophobicity (logp)?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| False | 68 | 0.453 | 0.309 | 0.309 | [0.21, 0.43] |
| True | 82 | 0.547 | 0.951 | 0.951 | [0.88, 0.98] |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.95).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| False | 68 | 0.453 | 0.309 | 0.309 | [0.21, 0.43] |
| True | 82 | 0.547 | 0.951 | 0.951 | [0.88, 0.98] |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.95).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| False | 68 | 0.453 | 0.309 | 0.309 | [0.21, 0.43] |
| True | 82 | 0.547 | 0.951 | 0.951 | [0.88, 0.98] |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.95).

### scalars-only tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| False | 68 | 0.453 | 0.309 | 0.309 | [0.21, 0.43] |
| True | 82 | 0.547 | 0.951 | 0.967 | [0.90, 0.99] |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.97).

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| False | 68 | 0.453 | 0.309 | 0.403 | [0.29, 0.52] |
| True | 82 | 0.547 | 0.951 | 0.902 | [0.82, 0.95] |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.90).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| False | 68 | 0.453 | 0.309 | 0.975 | [0.91, 0.99] |
| True | 82 | 0.547 | 0.951 | 0.971 | [0.91, 0.99] |

EQL's own `cause` search settles on False: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.98).


## Does the ind1 indicator cause a molecule to be mutagenic, adjusting for its branching-atom count?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| False | 68 | 0.453 | 0.309 | 0.422 | [0.31, 0.54] |
| True | 82 | 0.547 | 0.951 | 0.940 | [0.87, 0.97] |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.94).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| False | 68 | 0.453 | 0.309 | 0.422 | [0.31, 0.54] |
| True | 82 | 0.547 | 0.951 | 0.940 | [0.87, 0.97] |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.94).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| False | 68 | 0.453 | 0.309 | 0.370 | [0.27, 0.49] |
| True | 82 | 0.547 | 0.951 | 0.948 | [0.88, 0.98] |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.95).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| False | 68 | 0.453 | 0.309 | 0.571 | [0.45, 0.68] |
| True | 82 | 0.547 | 0.951 | 0.791 | [0.69, 0.87] |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.79).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| False | 68 | 0.453 | 0.309 | 0.977 | [0.91, 0.99] |
| True | 82 | 0.547 | 0.951 | 0.974 | [0.91, 0.99] |

EQL's own `cause` search settles on False: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.98).


## Does the ind1 indicator cause atom 0 of a molecule to be carbon?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| False | 68 | 0.453 | 0.431 | 0.431 | [0.32, 0.55] |
| True | 82 | 0.547 | 0.548 | 0.548 | [0.44, 0.65] |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.55).

### propositional tree

Refused: the fitted table has no column for the queried variables.

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| False | 68 | 0.453 | 1.000 | 1.000 | [0.95, 1.00] |
| True | 82 | 0.547 | 1.000 | 1.000 | [0.96, 1.00] |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

Refused: the fitted table has no column for the queried variables.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| False | 1472 | 0.375 | 0.429 | 0.486 | [0.46, 0.51] |
| True | 2456 | 0.625 | 0.522 | 0.485 | [0.47, 0.51] |

EQL's own `cause` search settles on False: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.49).


## How many branching atoms cause atom 0 of a molecule to be terminal, with a single bond?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 † | 7 | 0.047 | 0.440 | 0.440 | [0.16, 0.76] |
| 8 † | 8 | 0.053 | 0.526 | 0.526 | [0.23, 0.80] |
| 9 † | 9 | 0.060 | 0.446 | 0.446 | [0.19, 0.73] |
| 10 | 11 | 0.073 | 0.448 | 0.448 | [0.21, 0.72] |
| 11 † | 5 | 0.033 | 0.407 | 0.407 | [0.12, 0.77] |
| 12 † | 4 | 0.027 | 0.315 | 0.315 | [0.07, 0.74] |
| 13 | 11 | 0.073 | 0.229 | 0.229 | [0.07, 0.52] |
| 14 | 17 | 0.113 | 0.366 | 0.366 | [0.18, 0.60] |
| 15 | 17 | 0.113 | 0.418 | 0.418 | [0.22, 0.65] |
| 16 | 12 | 0.080 | 0.429 | 0.429 | [0.20, 0.69] |
| 17 | 14 | 0.093 | 0.395 | 0.395 | [0.19, 0.65] |
| 18 | 12 | 0.080 | 0.366 | 0.366 | [0.16, 0.64] |
| 19 † | 4 | 0.027 | 0.367 | 0.367 | [0.09, 0.78] |
| 20 † | 1 | 0.007 | 0.302 | 0.302 | [0.02, 0.90] |
| 21 | 12 | 0.080 | 0.407 | 0.407 | [0.19, 0.67] |
| 22 † | 3 | 0.020 | 0.444 | 0.444 | [0.10, 0.85] |
| 24 † | 2 | 0.013 | 0.321 | 0.321 | [0.04, 0.84] |
| 25 † | 1 | 0.007 | 0.316 | 0.316 | [0.02, 0.90] |

EQL's own `cause` search settles on 15: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.42).

### propositional tree

Refused: the fitted table has no column for the queried variables.

### unrolled tree

Refused: the effect has zero probability under every cause region.

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

Refused: the fitted table has no column for the queried variables.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 7 | 98 | 0.025 | 0.500 | 0.355 | [0.27, 0.45] |
| 8 | 136 | 0.035 | 0.500 | 0.365 | [0.29, 0.45] |
| 9 | 164 | 0.042 | 0.476 | 0.378 | [0.31, 0.45] |
| 10 | 212 | 0.054 | 0.448 | 0.392 | [0.33, 0.46] |
| 11 | 104 | 0.026 | 0.462 | 0.405 | [0.32, 0.50] |
| 12 | 90 | 0.023 | 0.467 | 0.415 | [0.32, 0.52] |
| 13 | 261 | 0.066 | 0.410 | 0.422 | [0.36, 0.48] |
| 14 | 435 | 0.111 | 0.451 | 0.428 | [0.38, 0.47] |
| 15 | 468 | 0.119 | 0.453 | 0.432 | [0.39, 0.48] |
| 16 | 355 | 0.090 | 0.448 | 0.435 | [0.38, 0.49] |
| 17 | 411 | 0.105 | 0.409 | 0.438 | [0.39, 0.49] |
| 18 | 366 | 0.093 | 0.410 | 0.441 | [0.39, 0.49] |
| 19 | 130 | 0.033 | 0.408 | 0.442 | [0.36, 0.53] |
| 20 | 34 | 0.009 | 0.412 | 0.443 | [0.29, 0.61] |
| 21 | 430 | 0.109 | 0.409 | 0.444 | [0.40, 0.49] |
| 22 | 120 | 0.031 | 0.450 | 0.444 | [0.36, 0.53] |
| 24 | 76 | 0.019 | 0.368 | 0.443 | [0.34, 0.56] |
| 25 | 38 | 0.010 | 0.342 | 0.443 | [0.30, 0.60] |

EQL's own `cause` search settles on 22: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.44).


## Does the element of atom 0 of a molecule cause it to be terminal, with a single bond?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| br † | 2 | 0.001 | 1.000 | 1.000 | [0.34, 1.00] |
| c | 1915 | 0.488 | 0.000 | 0.000 | [0.00, 0.00] |
| cl | 23 | 0.006 | 1.000 | 1.000 | [0.86, 1.00] |
| f † | 1 | 0.002 | 1.000 | 1.000 | [0.21, 1.00] |
| h | 1223 | 0.311 | 1.000 | 1.000 | [1.00, 1.00] |
| i † | 8 | 0.000 | 1.000 | 1.000 | [0.68, 1.00] |
| n | 481 | 0.070 | 0.000 | 0.000 | [0.00, 0.01] |
| o | 275 | 0.122 | 0.942 | 0.942 | [0.91, 0.96] |

EQL's own `cause` search settles on h: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).

### propositional tree

Refused: the fitted table has no column for the queried variables.

### unrolled tree

Refused: the effect has zero probability under every cause region.

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

Refused: the fitted table has no column for the queried variables.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| br † | 2 | 0.001 | 1.000 | 0.541 | [0.11, 0.92] |
| c | 1915 | 0.488 | 0.000 | 0.293 | [0.27, 0.31] |
| cl | 23 | 0.006 | 1.000 | 0.755 | [0.55, 0.89] |
| f † | 8 | 0.002 | 1.000 | 0.602 | [0.29, 0.85] |
| h | 1223 | 0.311 | 1.000 | 0.507 | [0.48, 0.54] |
| i † | 1 | 0.000 | 1.000 | 0.502 | [0.06, 0.95] |
| n | 275 | 0.070 | 0.000 | 0.301 | [0.25, 0.36] |
| o | 481 | 0.122 | 0.942 | 0.595 | [0.55, 0.64] |

EQL's own `cause` search settles on cl: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.76).

