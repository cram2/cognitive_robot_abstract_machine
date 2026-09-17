# Mutagenesis: relational circuit against flat-table trees

The CTU Mutagenesis dataset records 188 nitroaromatic molecules, each as its own attributes (the `ind1` structural indicator, `logp`, `lumo`, and whether it tested mutagenic) with one exchangeable part per atom (element, atom-type code, partial charge, number of bonds) and one per bond (its type). A molecule has between 14 and 40 atoms, and its atoms have no canonical order; the dataset lists heavy atoms first and hydrogens last, but nothing ties position to identity.

Four pipelines were fitted on the same molecules and asked the same `cause`/`causes_effect` EQL queries:

- **relational circuit**: a relational probabilistic circuit fitted on the molecules' relational structure, one circuit over the molecule's own attributes and its aggregation counts (chlorine atoms, branching atoms, double bonds, aromatic bonds), one template over an atom's attributes and one over a bond's, grounded per query into a circuit over exactly the queried molecule, atoms and bonds and registered as a causal circuit;
- **propositional tree**: a joint probability tree fitted on the molecules flattened into one table of the molecule's own attributes and the same four counts, the classic propositional summary of a relational example, registered as a causal circuit the same way;
- **unrolled tree**: the same tree on a table that also carries every atom's and bond's attributes under the part's position, padded with an absent marker past a molecule's last part, so that a column means whatever part a molecule happens to list at that position;
- **scalars-only tree**: the same tree on the molecule's own attributes alone, what a flat learner sees without the relational feature extraction.

Every flat tree answers a query by backdoor adjustment on a table column; the relational circuit does the same on the variable of a grounded circuit. In both, the model is stratified so it is support-deterministic over the cause, the effect's probability is read off every region of the cause, and any variable the query marks as a confounder is summed out of that reading. Every query lists one atom and one bond with all their attributes open, which is what makes grounding retain the molecule's counts as variables; a flat table ignores parts a query says nothing about and refuses a query that constrains a column it does not have.

## Setup

- molecules: 188 (150 to fit on, 38 held out)
- molecules that tested mutagenic: 66.5%
- fewest training rows per leaf: 15 in a cause-specific model, 50 in the plain model that scores held-out molecules
- split seed: 0

## How often a molecule is mutagenic

The molecules themselves, before any model: the share that tested mutagenic, grouped by the `ind1` indicator, by how many branching atoms (atoms with three or four bonds, the ring-fusion and branch points of the molecular graph) the molecule has, and by how many of its bonds are aromatic. This is the signal the models are asked to explain.

| ind1 | molecules | mutagenic |
|---|---|---|
| False | 85 | 30.6% |
| True | 103 | 96.1% |

| branching atoms | molecules | mutagenic |
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

| aromatic bonds | molecules | mutagenic |
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


## Which questions each pipeline can answer

One row per question, one column per pipeline. An answered cell says, in words, which setting of the cause makes the effect most likely after adjustment and how likely, against the least favourable setting; a refused cell says why the pipeline could not answer at all.

| question | relational circuit | propositional tree | unrolled tree | scalars-only tree |
|---|---|---|---|---|
| How many branching atoms cause a molecule to be mutagenic, adjusting for the ind1 indicator? | answered: with 17 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 7 branching atoms it is only 0.00. | answered: with 17 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 7 branching atoms it is only 0.00. | answered: with 17 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 7 branching atoms it is only 0.00. | refused: the fitted table has no column for the queried variables. |
| How many aromatic bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator? | answered: with 14 aromatic bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 5 aromatic bonds it is only 0.00. | answered: with 14 aromatic bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 5 aromatic bonds it is only 0.00. | answered: with 14 aromatic bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 5 aromatic bonds it is only 0.00. | refused: the fitted table has no column for the queried variables. |
| How many double bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator? | answered: with 9 double bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 5 double bonds it is only 0.50. | answered: with 9 double bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 5 double bonds it is only 0.50. | answered: with 9 double bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 5 double bonds it is only 0.50. | refused: the fitted table has no column for the queried variables. |
| Does the ind1 indicator cause a molecule to be mutagenic, adjusting for its hydrophobicity (logp)? | answered: with ind1 = True, the molecule is mutagenic with probability 0.95, the highest of any setting; with ind1 = False it is only 0.31. | answered: with ind1 = True, the molecule is mutagenic with probability 0.95, the highest of any setting; with ind1 = False it is only 0.31. | answered: with ind1 = True, the molecule is mutagenic with probability 0.95, the highest of any setting; with ind1 = False it is only 0.31. | answered: with ind1 = True, the molecule is mutagenic with probability 0.95, the highest of any setting; with ind1 = False it is only 0.31. |
| Does the ind1 indicator cause a molecule to be mutagenic, adjusting for its branching-atom count? | answered: with ind1 = True, the molecule is mutagenic with probability 0.94, the highest of any setting; with ind1 = False it is only 0.37. | answered: with ind1 = True, the molecule is mutagenic with probability 0.94, the highest of any setting; with ind1 = False it is only 0.37. | answered: with ind1 = True, the molecule is mutagenic with probability 0.94, the highest of any setting; with ind1 = False it is only 0.35. | refused: the fitted table has no column for the queried variables. |
| Does the ind1 indicator cause atom 0 of a molecule to be carbon? | answered: with ind1 = True, atom 0 is carbon with probability 0.52, the highest of any setting; with ind1 = False it is only 0.44. | refused: the fitted table has no column for the queried variables. | answered: with ind1 = True, atom 0 is carbon with probability 1.00, the highest of any setting; with ind1 = False it is only 1.00. | refused: the fitted table has no column for the queried variables. |
| How many branching atoms cause atom 0 of a molecule to be terminal, with a single bond? | answered: with 8 branching atoms, atom 0 is terminal with probability 0.51, the highest of any setting; with 25 branching atoms it is only 0.34. | refused: the fitted table has no column for the queried variables. | refused: the effect has zero probability under every cause region. | refused: the fitted table has no column for the queried variables. |
| Does the element of atom 0 of a molecule cause it to be terminal, with a single bond? | refused: the model is not support-deterministic over the cause. | refused: the fitted table has no column for the queried variables. | refused: the effect has zero probability under every cause region. | refused: the fitted table has no column for the queried variables. |

A question about counts needs the counts: the scalars-only tree refuses it. A question whose effect is one atom's own attribute needs the atoms: the propositional tree refuses it, the unrolled tree answers it about whatever atom the molecules list at that position, and the relational circuit answers it about an exchangeable atom. A question whose cause is one atom's own attribute is refused by the relational circuit: grounding with the molecule's counts left open mixes one copy of the atom template per sampled count, and those copies overlap on the atom's element without being identical (a copy for a molecule with no chlorine has no chlorine atom, a copy for one with some has), so the grounded circuit is not support-deterministic over the element. The unrolled tree answers it, since a column is a column; what that answer is worth is what the reordering below measures.

## Fit and likelihood

What each pipeline cost. *Models fitted* counts the plain model plus one support-deterministic model per distinct cause the questions asked about and the pipeline could fit; *training seconds* and the *nodes*/*edges* of every fitted circuit are summed over them, which for the relational circuit includes the atom and bond templates.

| pipeline | models fitted | training seconds | nodes | edges |
|---|---|---|---|---|
| relational circuit | 6 | 26.90 | 7507 | 7489 |
| propositional tree | 5 | 5.28 | 678 | 673 |
| unrolled tree | 6 | 66.37 | 15541 | 15535 |
| scalars-only tree | 2 | 0.01 | 93 | 91 |

How well each explains molecules it never saw, on three views of a molecule: its own scalars, which every pipeline models; its scalars and counts; and the whole molecule, atoms and bonds included, which only the pipelines that model the parts can score. The relational circuit scores a whole molecule as its class circuit over the scalars and counts times each part template over one atom or bond given the counts; the unrolled tree scores it as one row. *Held-out coverage* is the share of held-out molecules that lie inside the plain model's support at all, since a tree's leaves span only the value ranges they were fitted on, and a whole molecule is covered only if every one of its atoms and bonds is. The *mean log-likelihood* is over the covered molecules only; the last column restricts it to the molecules every pipeline in the table covers, so the numbers are over the same rows.

### scalars

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|
| relational circuit | 92.1% | -2.90 | -2.90 |
| propositional tree | 92.1% | -2.90 | -2.90 |
| unrolled tree | 94.7% | -3.15 | -3.13 |
| scalars-only tree | 92.1% | -2.90 | -2.90 |

### scalars and counts

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|
| relational circuit | 89.5% | -8.24 | -8.24 |
| propositional tree | 89.5% | -8.24 | -8.24 |
| unrolled tree | 92.1% | -8.60 | -8.55 |

### whole molecule

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|
| relational circuit | 60.5% | 5.56 | 15.43 |
| unrolled tree | 55.3% | -133.09 | -132.69 |

## Seconds per question

Wall-clock time from asking to the answer or the refusal. The *first ask* of a cause includes fitting that cause's own support-deterministic model; *asked again* repeats the question with every model fitted, so only grounding (for the relational circuit), verification and backdoor adjustment remain. A refusal is fast when it is a schema check; a relational answer draws Monte-Carlo samples for every count the query leaves open and grounds one atom template per sampled value, which is where its time goes.

| question | relational circuit, first ask | relational circuit, asked again | propositional tree, first ask | propositional tree, asked again | unrolled tree, first ask | unrolled tree, asked again | scalars-only tree, first ask | scalars-only tree, asked again |
|---|---|---|---|---|---|---|---|---|
| branching_atom_count_causes_mutagenicity | 5.59 | 1.53 | 1.49 | 0.33 | 25.27 | 0.56 | 0.00 | 0.00 |
| aromatic_bond_count_causes_mutagenicity | 7.38 | 2.64 | 1.30 | 0.26 | 21.80 | 0.50 | 0.00 | 0.00 |
| double_bond_count_causes_mutagenicity | 8.00 | 2.32 | 1.14 | 0.16 | 11.79 | 0.32 | 0.00 | 0.00 |
| indicator_causes_mutagenicity_adjusting_logp | 5.89 | 1.88 | 1.09 | 0.05 | 4.44 | 0.18 | 0.05 | 0.04 |
| indicator_causes_mutagenicity_adjusting_branching_atom_count | 1.77 | 1.66 | 0.16 | 0.15 | 0.29 | 0.81 | 0.00 | 0.00 |
| indicator_causes_carbon_atom_0 | 3.34 | 3.10 | 0.00 | 0.00 | 0.18 | 0.18 | 0.00 | 0.00 |
| branching_atom_count_causes_terminal_atom_0 | 3.44 | 3.20 | 0.00 | 0.00 | 0.37 | 0.37 | 0.00 | 0.00 |
| element_causes_terminal_atom_0 | 4.31 | 0.28 | 0.00 | 0.00 | 2.74 | 0.18 | 0.00 | 0.00 |

## Does the order of the atoms matter?

Every molecule's atoms and bonds were put in a random order, 3 times over, and each time the pipelines that model the parts were refitted on the same split and asked the questions about atoms again. A relational circuit treats the atoms as exchangeable, so nothing about it can depend on the order; an unrolled table's column `atoms[0]` holds a different atom of every molecule after each reordering. *Best regions* lists every most effective cause region found over the orderings; *largest difference* is, over the cause regions every answered ordering distinguishes, the widest gap between orderings in the effect's adjusted probability.

| question | pipeline | orderings answered | best regions | largest difference in adjusted P(effect) |
|---|---|---|---|---|
| indicator_causes_carbon_atom_0 | relational circuit | 3 of 3 | True | 0.00 |
| branching_atom_count_causes_terminal_atom_0 | relational circuit | 3 of 3 | 8 | 0.00 |
| element_causes_terminal_atom_0 | relational circuit | 0 of 3 | - | - |
| indicator_causes_carbon_atom_0 | unrolled tree | 3 of 3 | False, True | 0.22 |
| branching_atom_count_causes_terminal_atom_0 | unrolled tree | 3 of 3 | 19, 25 | 1.00 |
| element_causes_terminal_atom_0 | unrolled tree | 3 of 3 | h | 0.04 |

The whole-molecule likelihood of the same held-out molecules under each ordering:

| pipeline | ordering 0, coverage / mean log-likelihood | ordering 1, coverage / mean log-likelihood | ordering 2, coverage / mean log-likelihood | largest difference |
|---|---|---|---|---|
| relational circuit | 60.5% / 5.56 | 60.5% / 5.56 | 60.5% / 5.56 | 0.00 |
| unrolled tree | 10.5% / -149.40 | 18.4% / -153.13 | 10.5% / -169.96 | 20.57 |

## Over several splits

The comparison repeated over 5 random splits (seeds 0, 1, 2, 3, 4), mean ± standard deviation. The likelihoods are over the molecules every pipeline modelling the view covers.

### scalars

| pipeline | held-out coverage | mean log-likelihood (covered by all) |
|---|---|---|
| relational circuit | 0.926 ± 0.054 | -3.15 ± 0.23 |
| propositional tree | 0.926 ± 0.054 | -3.15 ± 0.23 |
| unrolled tree | 0.953 ± 0.031 | -3.31 ± 0.19 |
| scalars-only tree | 0.921 ± 0.047 | -3.08 ± 0.18 |

### scalars and counts

| pipeline | held-out coverage | mean log-likelihood (covered by all) |
|---|---|---|
| relational circuit | 0.858 ± 0.094 | -8.34 ± 0.22 |
| propositional tree | 0.858 ± 0.094 | -8.34 ± 0.22 |
| unrolled tree | 0.879 ± 0.091 | -8.67 ± 0.26 |

### whole molecule

| pipeline | held-out coverage | mean log-likelihood (covered by all) |
|---|---|---|
| relational circuit | 0.553 ± 0.109 | 9.70 ± 3.93 |
| unrolled tree | 0.484 ± 0.143 | -137.52 ± 6.76 |

Per question, how many splits each pipeline answered and which most effective cause regions it found across them:

| question | relational circuit | propositional tree | unrolled tree | scalars-only tree |
|---|---|---|---|---|
| branching_atom_count_causes_mutagenicity | 5 of 5: 17 | 5 of 5: 17 | 5 of 5: 16, 17 | 0 of 5 |
| aromatic_bond_count_causes_mutagenicity | 5 of 5: 14 | 5 of 5: 14 | 5 of 5: 14 | 0 of 5 |
| double_bond_count_causes_mutagenicity | 5 of 5: 5, 6, 9 | 5 of 5: 5, 6, 9 | 5 of 5: 5, 6, 9 | 0 of 5 |
| indicator_causes_mutagenicity_adjusting_logp | 5 of 5: True | 5 of 5: True | 5 of 5: True | 5 of 5: True |
| indicator_causes_mutagenicity_adjusting_branching_atom_count | 5 of 5: True | 5 of 5: True | 5 of 5: True | 0 of 5 |
| indicator_causes_carbon_atom_0 | 5 of 5: True | 0 of 5 | 5 of 5: False, True | 0 of 5 |
| branching_atom_count_causes_terminal_atom_0 | 5 of 5: 7, 8 | 0 of 5 | 0 of 5 | 0 of 5 |
| element_causes_terminal_atom_0 | 0 of 5 | 0 of 5 | 0 of 5 | 0 of 5 |

## How much training data it takes

Every pipeline's plain model fitted on a growing share of the molecules and scored on the same held-out fifth, over 3 splits, mean ± standard deviation of the held-out coverage and of the mean log-likelihood over the covered molecules. The relational circuit's templates pool every atom of every training molecule, about 26 rows per molecule, where the unrolled tree sees one row per molecule.

### scalars and counts

| training share | relational circuit, coverage | relational circuit, mean log-likelihood | propositional tree, coverage | propositional tree, mean log-likelihood | unrolled tree, coverage | unrolled tree, mean log-likelihood |
|---|---|---|---|---|---|---|
| 20.0% | 0.807 ± 0.045 | -9.89 ± 0.28 | 0.807 ± 0.045 | -9.89 ± 0.28 | 0.807 ± 0.045 | -9.89 ± 0.28 |
| 40.0% | 0.930 ± 0.033 | -10.14 ± 0.30 | 0.930 ± 0.033 | -10.14 ± 0.30 | 0.930 ± 0.033 | -10.14 ± 0.30 |
| 60.0% | 0.912 ± 0.012 | -8.58 ± 0.09 | 0.912 ± 0.012 | -8.58 ± 0.09 | 0.939 ± 0.025 | -8.68 ± 0.21 |
| 80.0% | 0.921 ± 0.021 | -8.20 ± 0.09 | 0.921 ± 0.021 | -8.20 ± 0.09 | 0.939 ± 0.025 | -8.70 ± 0.39 |

### whole molecule

| training share | relational circuit, coverage | relational circuit, mean log-likelihood | unrolled tree, coverage | unrolled tree, mean log-likelihood |
|---|---|---|---|---|
| 20.0% | 0.316 ± 0.043 | -17.45 ± 12.99 | 0.228 ± 0.045 | -182.59 ± 12.82 |
| 40.0% | 0.412 ± 0.025 | -2.57 ± 7.90 | 0.377 ± 0.033 | -174.36 ± 10.04 |
| 60.0% | 0.561 ± 0.012 | -0.25 ± 1.63 | 0.509 ± 0.025 | -132.32 ± 5.57 |
| 80.0% | 0.632 ± 0.057 | 3.77 ± 1.65 | 0.596 ± 0.045 | -136.63 ± 3.85 |

## What the results show

- The relational circuit answered 7 of 8 questions, refusing `element_causes_terminal_atom_0` because the model is not support-deterministic over the cause.
- The propositional tree answered 5 of 8 questions, refusing `indicator_causes_carbon_atom_0` because the fitted table has no column for the queried variables; `branching_atom_count_causes_terminal_atom_0` because the fitted table has no column for the queried variables; `element_causes_terminal_atom_0` because the fitted table has no column for the queried variables.
- The unrolled tree answered 6 of 8 questions, refusing `branching_atom_count_causes_terminal_atom_0` because the effect has zero probability under every cause region; `element_causes_terminal_atom_0` because the effect has zero probability under every cause region.
- The scalars-only tree answered 1 of 8 questions, refusing `branching_atom_count_causes_mutagenicity` because the fitted table has no column for the queried variables; `aromatic_bond_count_causes_mutagenicity` because the fitted table has no column for the queried variables; `double_bond_count_causes_mutagenicity` because the fitted table has no column for the queried variables; `indicator_causes_mutagenicity_adjusting_branching_atom_count` because the fitted table has no column for the queried variables; `indicator_causes_carbon_atom_0` because the fitted table has no column for the queried variables; `branching_atom_count_causes_terminal_atom_0` because the fitted table has no column for the queried variables; `element_causes_terminal_atom_0` because the fitted table has no column for the queried variables.
- On `branching_atom_count_causes_mutagenicity`, every pipeline that answered finds 17 branching atoms the most effective setting (adjusted probabilities: relational circuit 1.00, propositional tree 1.00, unrolled tree 1.00).
- On `aromatic_bond_count_causes_mutagenicity`, every pipeline that answered finds 14 aromatic bonds the most effective setting (adjusted probabilities: relational circuit 1.00, propositional tree 1.00, unrolled tree 1.00).
- On `double_bond_count_causes_mutagenicity`, every pipeline that answered finds 9 double bonds the most effective setting (adjusted probabilities: relational circuit 1.00, propositional tree 1.00, unrolled tree 1.00).
- On `indicator_causes_mutagenicity_adjusting_logp`, every pipeline that answered finds ind1 = True the most effective setting (adjusted probabilities: relational circuit 0.95, propositional tree 0.95, unrolled tree 0.95, scalars-only tree 0.95).
- On `indicator_causes_mutagenicity_adjusting_branching_atom_count`, every pipeline that answered finds ind1 = True the most effective setting (adjusted probabilities: relational circuit 0.94, propositional tree 0.94, unrolled tree 0.94).
- On `indicator_causes_carbon_atom_0`, every pipeline that answered finds ind1 = True the most effective setting (adjusted probabilities: relational circuit 0.52, unrolled tree 1.00).
- On the scalars, the relational circuit assigns the highest mean log-likelihood (-2.90, against propositional tree -2.90, unrolled tree -3.13, scalars-only tree -2.90) to the held-out molecules every pipeline covers; coverage: relational circuit 92.1%, propositional tree 92.1%, unrolled tree 94.7%, scalars-only tree 92.1%.
- On the scalars and counts, the relational circuit assigns the highest mean log-likelihood (-8.24, against propositional tree -8.24, unrolled tree -8.55) to the held-out molecules every pipeline covers; coverage: relational circuit 89.5%, propositional tree 89.5%, unrolled tree 92.1%.
- On the whole molecule, the relational circuit assigns the highest mean log-likelihood (15.43, against unrolled tree -132.69) to the held-out molecules every pipeline covers; coverage: relational circuit 60.5%, unrolled tree 55.3%.
- Reordering the atoms moved the relational circuit's adjusted effect probabilities by up to 0.00 and its whole-molecule mean log-likelihood by 0.00.
- Reordering the atoms moved the unrolled tree's adjusted effect probabilities by up to 1.00 and its whole-molecule mean log-likelihood by 20.57; it changed the most effective setting of `indicator_causes_carbon_atom_0`, `branching_atom_count_causes_terminal_atom_0`.
- The relational circuit takes 2.33 seconds per answered question on average once its models are fitted.
- The propositional tree takes 0.19 seconds per answered question on average once its models are fitted.
- The unrolled tree takes 0.42 seconds per answered question on average once its models are fitted.
- The scalars-only tree takes 0.04 seconds per answered question on average once its models are fitted.

## How many branching atoms cause a molecule to be mutagenic, adjusting for the ind1 indicator?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 7 | 0.047 | 0.000 | 0.000 |
| 8 | 0.053 | 0.375 | 0.375 |
| 9 | 0.060 | 0.111 | 0.111 |
| 10 | 0.073 | 0.182 | 0.182 |
| 11 | 0.033 | 0.200 | 0.200 |
| 12 | 0.027 | 0.250 | 0.250 |
| 13 | 0.073 | 0.727 | 0.727 |
| 14 | 0.113 | 0.588 | 0.588 |
| 15 | 0.113 | 0.824 | 0.824 |
| 16 | 0.080 | 0.833 | 0.833 |
| 17 | 0.093 | 1.000 | 1.000 |
| 18 | 0.080 | 1.000 | 1.000 |
| 19 | 0.027 | 1.000 | 1.000 |
| 20 | 0.007 | 1.000 | 1.000 |
| 21 | 0.080 | 1.000 | 1.000 |
| 22 | 0.020 | 1.000 | 1.000 |
| 24 | 0.013 | 1.000 | 1.000 |
| 25 | 0.007 | 1.000 | 1.000 |

EQL's own `cause` search settles on 17: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).

### propositional tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 7 | 0.047 | 0.000 | 0.000 |
| 8 | 0.053 | 0.375 | 0.375 |
| 9 | 0.060 | 0.111 | 0.111 |
| 10 | 0.073 | 0.182 | 0.182 |
| 11 | 0.033 | 0.200 | 0.200 |
| 12 | 0.027 | 0.250 | 0.250 |
| 13 | 0.073 | 0.727 | 0.727 |
| 14 | 0.113 | 0.588 | 0.588 |
| 15 | 0.113 | 0.824 | 0.824 |
| 16 | 0.080 | 0.833 | 0.833 |
| 17 | 0.093 | 1.000 | 1.000 |
| 18 | 0.080 | 1.000 | 1.000 |
| 19 | 0.027 | 1.000 | 1.000 |
| 20 | 0.007 | 1.000 | 1.000 |
| 21 | 0.080 | 1.000 | 1.000 |
| 22 | 0.020 | 1.000 | 1.000 |
| 24 | 0.013 | 1.000 | 1.000 |
| 25 | 0.007 | 1.000 | 1.000 |

EQL's own `cause` search settles on 17: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).

### unrolled tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 7 | 0.047 | 0.000 | 0.000 |
| 8 | 0.053 | 0.375 | 0.375 |
| 9 | 0.060 | 0.111 | 0.111 |
| 10 | 0.073 | 0.182 | 0.182 |
| 11 | 0.033 | 0.200 | 0.200 |
| 12 | 0.027 | 0.250 | 0.250 |
| 13 | 0.073 | 0.727 | 0.727 |
| 14 | 0.113 | 0.588 | 0.588 |
| 15 | 0.113 | 0.824 | 0.824 |
| 16 | 0.080 | 0.833 | 0.833 |
| 17 | 0.093 | 1.000 | 1.000 |
| 18 | 0.080 | 1.000 | 1.000 |
| 19 | 0.027 | 1.000 | 1.000 |
| 20 | 0.007 | 1.000 | 1.000 |
| 21 | 0.080 | 1.000 | 1.000 |
| 22 | 0.020 | 1.000 | 1.000 |
| 24 | 0.013 | 1.000 | 1.000 |
| 25 | 0.007 | 1.000 | 1.000 |

EQL's own `cause` search settles on 17: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.


## How many aromatic bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 5 | 0.007 | 0.000 | 0.000 |
| 6 | 0.167 | 0.240 | 0.240 |
| 10 | 0.053 | 0.250 | 0.250 |
| 11 | 0.100 | 0.333 | 0.333 |
| 12 | 0.347 | 0.712 | 0.713 |
| 14 | 0.007 | 1.000 | 1.000 |
| 15 | 0.020 | 1.000 | 1.000 |
| 16 | 0.027 | 1.000 | 1.000 |
| 17 | 0.080 | 1.000 | 1.000 |
| 18 | 0.007 | 1.000 | 1.000 |
| 19 | 0.120 | 1.000 | 1.000 |
| 22 | 0.007 | 1.000 | 1.000 |
| 24 | 0.040 | 1.000 | 1.000 |
| 26 | 0.013 | 1.000 | 1.000 |
| 30 | 0.007 | 1.000 | 1.000 |

EQL's own `cause` search settles on 12: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.71).

### propositional tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 5 | 0.007 | 0.000 | 0.000 |
| 6 | 0.167 | 0.240 | 0.240 |
| 10 | 0.053 | 0.250 | 0.250 |
| 11 | 0.100 | 0.333 | 0.333 |
| 12 | 0.347 | 0.712 | 0.713 |
| 14 | 0.007 | 1.000 | 1.000 |
| 15 | 0.020 | 1.000 | 1.000 |
| 16 | 0.027 | 1.000 | 1.000 |
| 17 | 0.080 | 1.000 | 1.000 |
| 18 | 0.007 | 1.000 | 1.000 |
| 19 | 0.120 | 1.000 | 1.000 |
| 22 | 0.007 | 1.000 | 1.000 |
| 24 | 0.040 | 1.000 | 1.000 |
| 26 | 0.013 | 1.000 | 1.000 |
| 30 | 0.007 | 1.000 | 1.000 |

EQL's own `cause` search settles on 12: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.71).

### unrolled tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 5 | 0.007 | 0.000 | 0.000 |
| 6 | 0.167 | 0.240 | 0.240 |
| 10 | 0.053 | 0.250 | 0.250 |
| 11 | 0.100 | 0.333 | 0.333 |
| 12 | 0.347 | 0.712 | 0.711 |
| 14 | 0.007 | 1.000 | 1.000 |
| 15 | 0.020 | 1.000 | 1.000 |
| 16 | 0.027 | 1.000 | 1.000 |
| 17 | 0.080 | 1.000 | 1.000 |
| 18 | 0.007 | 1.000 | 1.000 |
| 19 | 0.120 | 1.000 | 1.000 |
| 22 | 0.007 | 1.000 | 1.000 |
| 24 | 0.040 | 1.000 | 1.000 |
| 26 | 0.013 | 1.000 | 1.000 |
| 30 | 0.007 | 1.000 | 1.000 |

EQL's own `cause` search settles on 12: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.71).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.


## How many double bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 2 | 0.573 | 0.547 | 0.537 |
| 3 | 0.087 | 0.692 | 0.692 |
| 4 | 0.220 | 0.848 | 0.823 |
| 5 | 0.013 | 0.500 | 0.500 |
| 6 | 0.060 | 0.889 | 0.889 |
| 7 | 0.007 | 1.000 | 1.000 |
| 8 | 0.033 | 0.800 | 0.800 |
| 9 | 0.007 | 1.000 | 1.000 |

EQL's own `cause` search settles on 2: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.54).

### propositional tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 2 | 0.573 | 0.547 | 0.537 |
| 3 | 0.087 | 0.692 | 0.692 |
| 4 | 0.220 | 0.848 | 0.823 |
| 5 | 0.013 | 0.500 | 0.500 |
| 6 | 0.060 | 0.889 | 0.889 |
| 7 | 0.007 | 1.000 | 1.000 |
| 8 | 0.033 | 0.800 | 0.800 |
| 9 | 0.007 | 1.000 | 1.000 |

EQL's own `cause` search settles on 2: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.54).

### unrolled tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 2 | 0.573 | 0.547 | 0.540 |
| 3 | 0.087 | 0.692 | 0.692 |
| 4 | 0.220 | 0.848 | 0.823 |
| 5 | 0.013 | 0.500 | 0.500 |
| 6 | 0.060 | 0.889 | 0.889 |
| 7 | 0.007 | 1.000 | 1.000 |
| 8 | 0.033 | 0.800 | 0.800 |
| 9 | 0.007 | 1.000 | 1.000 |

EQL's own `cause` search settles on 2: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.54).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.


## Does the ind1 indicator cause a molecule to be mutagenic, adjusting for its hydrophobicity (logp)?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| False | 0.453 | 0.309 | 0.309 |
| True | 0.547 | 0.951 | 0.951 |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.95).

### propositional tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| False | 0.453 | 0.309 | 0.309 |
| True | 0.547 | 0.951 | 0.951 |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.95).

### unrolled tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| False | 0.453 | 0.309 | 0.309 |
| True | 0.547 | 0.951 | 0.951 |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.95).

### scalars-only tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| False | 0.453 | 0.309 | 0.309 |
| True | 0.547 | 0.951 | 0.951 |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.95).


## Does the ind1 indicator cause a molecule to be mutagenic, adjusting for its branching-atom count?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| False | 0.453 | 0.309 | 0.371 |
| True | 0.547 | 0.951 | 0.938 |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.94).

### propositional tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| False | 0.453 | 0.309 | 0.371 |
| True | 0.547 | 0.951 | 0.938 |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.94).

### unrolled tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| False | 0.453 | 0.309 | 0.348 |
| True | 0.547 | 0.951 | 0.937 |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.94).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.


## Does the ind1 indicator cause atom 0 of a molecule to be carbon?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| False | 0.453 | 0.441 | 0.441 |
| True | 0.547 | 0.522 | 0.522 |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.52).

### propositional tree

Refused: the fitted table has no column for the queried variables.

### unrolled tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| False | 0.453 | 1.000 | 1.000 |
| True | 0.547 | 1.000 | 1.000 |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.


## How many branching atoms cause atom 0 of a molecule to be terminal, with a single bond?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 7 | 0.047 | 0.450 | 0.450 |
| 8 | 0.053 | 0.514 | 0.514 |
| 9 | 0.060 | 0.467 | 0.467 |
| 10 | 0.073 | 0.450 | 0.450 |
| 11 | 0.033 | 0.463 | 0.463 |
| 12 | 0.027 | 0.467 | 0.467 |
| 13 | 0.073 | 0.392 | 0.392 |
| 14 | 0.113 | 0.453 | 0.453 |
| 15 | 0.113 | 0.455 | 0.455 |
| 16 | 0.080 | 0.445 | 0.445 |
| 17 | 0.093 | 0.410 | 0.410 |
| 18 | 0.080 | 0.410 | 0.410 |
| 19 | 0.027 | 0.405 | 0.405 |
| 20 | 0.007 | 0.411 | 0.411 |
| 21 | 0.080 | 0.405 | 0.405 |
| 22 | 0.020 | 0.450 | 0.450 |
| 24 | 0.013 | 0.368 | 0.368 |
| 25 | 0.007 | 0.342 | 0.342 |

EQL's own `cause` search settles on 15: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.46).

### propositional tree

Refused: the fitted table has no column for the queried variables.

### unrolled tree

Refused: the effect has zero probability under every cause region.

### scalars-only tree

Refused: the fitted table has no column for the queried variables.


## Does the element of atom 0 of a molecule cause it to be terminal, with a single bond?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

Refused: the model is not support-deterministic over the cause.

### propositional tree

Refused: the fitted table has no column for the queried variables.

### unrolled tree

Refused: the effect has zero probability under every cause region.

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

