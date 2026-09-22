# Tracy clutter picking: relational circuit against flat-table trees

Tracy's left arm picks one milk carton out of a ten-carton clutter in MuJoCo, holding it by contact friction alone. Every attempt is recorded as a relational scene: the attempt's own attributes (environment, grasp friction, grasp yaw, whether the target came up) and one exchangeable part per neighbouring carton (its position relative to the target, its distance band, which side of the fingers' closing axis it stands on, and how far the pick shoved it). Every recorded attempt has 9 neighbours, and they have no canonical order; the recording lists them in the order the layout was drawn, and nothing ties a position to an identity.

Four pipelines were fitted on the same recorded attempts and asked the same `cause`/`causes_effect` EQL queries:

- **relational circuit**: a relational probabilistic circuit fitted on the attempts' relational structure, one circuit over the attempt's own attributes and its aggregation count (neighbours adjacent to the target), one template over a neighbour's attributes, grounded per query into a circuit over exactly the queried attempt and neighbours and registered as a causal circuit;
- **propositional tree**: a joint probability tree fitted on the attempts flattened into one table of the attempt's own attributes and the same count, the classic propositional summary of a relational example, registered as a causal circuit the same way;
- **unrolled tree**: the same tree on a table that also carries every neighbour's attributes under the neighbour's position, so that a column means whatever neighbour an attempt happens to list at that position;
- **scalars-only tree**: the same tree on the attempt's own attributes alone, what a flat learner sees without the relational feature extraction.

Every flat tree answers a query by backdoor adjustment on a table column; the relational circuit does the same on the variable of a grounded circuit. In both, the model is stratified so it is support-deterministic over the cause, the effect's probability is read off every region of the cause, and any variable the query marks as a confounder is summed out of that reading. A query lists as many neighbours as the clutter it asks about has, with all their attributes open, which is what the relational circuit grounds itself for; a flat table ignores parts a query says nothing about, so it answers a question about a clutter of another size with the numbers it has for the recorded one, and refuses a query that constrains a column it does not have.

## Setup

- attempts: 300 (240 to fit on, 60 held out)
- attempts where the target is lifted: 65.3%
- fewest training rows per leaf, as a share of the rows fitted on: 0.05 in a cause-specific model, 0.15 in the plain model that scores held-out attempts
- split seed: 0
- fewest training attempts a cause region may hold for its effect to be read as an answer: 10; a region below that is marked † in the tables and takes no part in any summary

## How often the target is lifted

The attempts themselves, before any model: the share where the target is lifted, grouped by the environment the clutter stood in, by the grasp's friction coefficient, and by how many neighbours stood adjacent to the target (closer than the fingers' sweep). This is the signal the models are asked to explain.

| environment | attempts | effect |
|---|---|---|
| bin | 158 | 45.6% |
| table | 142 | 87.3% |

| friction coefficient | attempts | effect |
|---|---|---|
| 0.125 | 76 | 0.0% |
| 0.1875 | 78 | 87.2% |
| 0.25 | 79 | 77.2% |
| 0.375 | 24 | 100.0% |
| 0.5 | 17 | 100.0% |
| 0.75 | 26 | 100.0% |

| adjacent neighbours | attempts | effect |
|---|---|---|
| 0 | 130 | 86.9% |
| 1 | 28 | 85.7% |
| 2 | 73 | 45.2% |
| 3 | 48 | 41.7% |
| 4 | 16 | 25.0% |
| 5 | 5 | 40.0% |


## Which questions each pipeline can answer

One row per question, one column per pipeline. An answered cell says, in words, which setting of the cause makes the effect most likely after adjustment and how likely, against the least favourable setting, over the regions that hold enough training attempts to be read; a refused cell says why the pipeline could not answer at all.

| question | relational circuit | propositional tree | unrolled tree | scalars-only tree | regression adjustment | neural adjustment |
|---|---|---|---|---|---|---|
| In a clutter of 9 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment? | answered: with a grasp friction coefficient of 0.375, the target is lifted with probability 1.00, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.00. | answered: with a grasp friction coefficient of 0.375, the target is lifted with probability 1.00, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.00. | answered: with a grasp friction coefficient of 0.375, the target is lifted with probability 1.00, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.00. | answered: with a grasp friction coefficient of 0.375, the target is lifted with probability 1.00, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.00. | answered: with a grasp friction coefficient of 0.75, the target is lifted with probability 0.90, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.59. | answered: with a grasp friction coefficient of 0.75, the target is lifted with probability 1.00, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.28. |
| In a clutter of 9 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment? | answered: with 0 adjacent neighbours, the target is lifted with probability 0.88, the highest of any setting; with 4 adjacent neighbours it is only 0.23. | answered: with 0 adjacent neighbours, the target is lifted with probability 0.88, the highest of any setting; with 4 adjacent neighbours it is only 0.23. | answered: with 0 adjacent neighbours, the target is lifted with probability 0.88, the highest of any setting; with 4 adjacent neighbours it is only 0.23. | refused: the fitted table has no column for the queried variables. | answered: with 0 adjacent neighbours, the target is lifted with probability 0.80, the highest of any setting; with 4 adjacent neighbours it is only 0.43. | answered: with 0 adjacent neighbours, the target is lifted with probability 0.81, the highest of any setting; with 4 adjacent neighbours it is only 0.54. |
| In a clutter of 9 neighbours, how many of them standing adjacent to the target causes it to be lifted, with nothing adjusted for? | answered: with 0 adjacent neighbours, the target is lifted with probability 0.88, the highest of any setting; with 4 adjacent neighbours it is only 0.23. | answered: with 0 adjacent neighbours, the target is lifted with probability 0.88, the highest of any setting; with 4 adjacent neighbours it is only 0.23. | answered: with 0 adjacent neighbours, the target is lifted with probability 0.88, the highest of any setting; with 4 adjacent neighbours it is only 0.23. | refused: the fitted table has no column for the queried variables. | answered: with 0 adjacent neighbours, the target is lifted with probability 0.87, the highest of any setting; with 4 adjacent neighbours it is only 0.23. | answered: with 0 adjacent neighbours, the target is lifted with probability 0.89, the highest of any setting; with 4 adjacent neighbours it is only 0.31. |
| In a clutter of 9 neighbours, does neighbour 0 standing along the fingers' closing axis cause it to be disturbed by the pick? | answered: with neighbour 0 standing along the closing axis, neighbour 0 is disturbed with probability 0.18, the highest of any setting; with neighbour 0 standing across the closing axis it is only 0.03. | refused: the fitted table has no column for the queried variables. | answered: with neighbour 0 standing along the closing axis, neighbour 0 is disturbed with probability 0.22, the highest of any setting; with neighbour 0 standing across the closing axis it is only 0.03. | refused: the fitted table has no column for the queried variables. | refused: the fitted table has no column for the queried variables. | answered: with neighbour 0 standing along the closing axis, neighbour 0 is disturbed with probability 0.12, the highest of any setting; with neighbour 0 standing across the closing axis it is only 0.08. |
| In a clutter of 4 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment? | answered: with a grasp friction coefficient of 0.375, the target is lifted with probability 1.00, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.00. | answered: with a grasp friction coefficient of 0.375, the target is lifted with probability 1.00, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.00. | answered: with a grasp friction coefficient of 0.375, the target is lifted with probability 1.00, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.00. | answered: with a grasp friction coefficient of 0.375, the target is lifted with probability 1.00, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.00. | answered: with a grasp friction coefficient of 0.75, the target is lifted with probability 0.90, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.59. | answered: with a grasp friction coefficient of 0.75, the target is lifted with probability 1.00, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.28. |
| In a clutter of 12 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment? | answered: with 0 adjacent neighbours, the target is lifted with probability 0.88, the highest of any setting; with 4 adjacent neighbours it is only 0.23. | answered: with 0 adjacent neighbours, the target is lifted with probability 0.88, the highest of any setting; with 4 adjacent neighbours it is only 0.23. | answered: with 0 adjacent neighbours, the target is lifted with probability 0.88, the highest of any setting; with 4 adjacent neighbours it is only 0.23. | refused: the fitted table has no column for the queried variables. | answered: with 0 adjacent neighbours, the target is lifted with probability 0.80, the highest of any setting; with 4 adjacent neighbours it is only 0.43. | answered: with 0 adjacent neighbours, the target is lifted with probability 0.81, the highest of any setting; with 4 adjacent neighbours it is only 0.54. |
| In a clutter of 12 neighbours, does neighbour 11 standing along the fingers' closing axis cause it to be disturbed by the pick? | answered: with neighbour 11 standing along the closing axis, neighbour 11 is disturbed with probability 0.18, the highest of any setting; with neighbour 11 standing across the closing axis it is only 0.03. | refused: the fitted table has no column for the queried variables. | refused: the fitted table has no column for the queried variables. | refused: the fitted table has no column for the queried variables. | refused: the fitted table has no column for the queried variables. | answered: with neighbour 11 standing along the closing axis, neighbour 11 is disturbed with probability 0.12, the highest of any setting; with neighbour 11 standing across the closing axis it is only 0.08. |

A question about the crowding count needs the count: the scalars-only tree refuses it. A question whose cause and effect live on one neighbour needs the neighbours: the propositional tree refuses it, the unrolled tree answers it about whatever neighbour the attempts list at that position, and the relational circuit answers it about an exchangeable neighbour. The questions about clutters of other sizes are answered by the flat trees with the same numbers as for the recorded size, since nothing in a flat table tells the sizes apart; only a model that grounds itself for the queried objects gives a size its own answer, and only it can be asked about a neighbour beyond the last column of the unrolled table.

## Trend and contrast

The most effective setting is an argmax over up to twenty sparse regions and moves with the split. Two summaries that do not: *trend* is Spearman's rank correlation between the cause's value and the adjusted probability over the supported regions, for a numeric cause; *contrast* is the adjusted probability at the highest supported region minus at the lowest (for a symbolic cause, at the most effective minus at the least), with Newcombe's interval from the Wilson intervals of the two regions' support.

| question | relational circuit, trend | relational circuit, contrast | propositional tree, trend | propositional tree, contrast | unrolled tree, trend | unrolled tree, contrast | scalars-only tree, trend | scalars-only tree, contrast | regression adjustment, trend | regression adjustment, contrast | neural adjustment, trend | neural adjustment, contrast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| friction_causes_lift_9_neighbours | 0.84 | 1.00 [0.81, 1.00] (0.125 → 0.75) | 0.84 | 1.00 [0.81, 1.00] (0.125 → 0.75) | 0.84 | 1.00 [0.81, 1.00] (0.125 → 0.75) | 0.84 | 1.00 [0.81, 1.00] (0.125 → 0.75) | 1.00 | 0.31 [0.06, 0.46] (0.125 → 0.75) | 1.00 | 0.72 [0.51, 0.82] (0.125 → 0.75) |
| crowding_causes_lift_9_neighbours_adjusting_environment | -1.00 | -0.65 [-0.81, -0.37] (0 → 4) | -1.00 | -0.65 [-0.81, -0.37] (0 → 4) | -1.00 | -0.65 [-0.81, -0.37] (0 → 4) | - | - | -1.00 | -0.37 [-0.60, -0.10] (0 → 4) | -1.00 | -0.27 [-0.53, -0.03] (0 → 4) |
| crowding_causes_lift_9_neighbours_unadjusted | -1.00 | -0.65 [-0.81, -0.37] (0 → 4) | -1.00 | -0.65 [-0.81, -0.37] (0 → 4) | -1.00 | -0.65 [-0.81, -0.37] (0 → 4) | - | - | -1.00 | -0.64 [-0.80, -0.36] (0 → 4) | -1.00 | -0.58 [-0.77, -0.30] (0 → 4) |
| closing_axis_side_causes_disturbance_of_neighbour_0_of_9 | - | 0.15 [0.12, 0.17] (across → along) | - | - | - | 0.19 [0.11, 0.28] (across → along) | - | - | - | - | - | 0.04 [0.01, 0.06] (across → along) |
| friction_causes_lift_4_neighbours | 0.84 | 1.00 [0.81, 1.00] (0.125 → 0.75) | 0.84 | 1.00 [0.81, 1.00] (0.125 → 0.75) | 0.84 | 1.00 [0.81, 1.00] (0.125 → 0.75) | 0.84 | 1.00 [0.81, 1.00] (0.125 → 0.75) | 1.00 | 0.31 [0.06, 0.46] (0.125 → 0.75) | 1.00 | 0.72 [0.51, 0.82] (0.125 → 0.75) |
| crowding_causes_lift_12_neighbours_adjusting_environment | -1.00 | -0.65 [-0.81, -0.37] (0 → 4) | -1.00 | -0.65 [-0.81, -0.37] (0 → 4) | -1.00 | -0.65 [-0.81, -0.37] (0 → 4) | - | - | -1.00 | -0.37 [-0.60, -0.10] (0 → 4) | -1.00 | -0.27 [-0.53, -0.03] (0 → 4) |
| closing_axis_side_causes_disturbance_of_neighbour_11_of_12 | - | 0.15 [0.12, 0.17] (across → along) | - | - | - | - | - | - | - | - | - | 0.04 [0.01, 0.06] (across → along) |

## What adjusting for changes

The same count question under each set of confounders it was asked with, read off the relational circuit. *n* is how many training attempts hold that value of the cause; † marks a region below the support threshold.

### crowding_count

| cause region | n | naive | adjusted for the environment | unadjusted |
|---|---|---|---|---|
| 0 | 101 | 0.881 | 0.881 | 0.881 |
| 1 | 24 | 0.875 | 0.868 | 0.875 |
| 2 | 56 | 0.446 | 0.446 | 0.446 |
| 3 | 41 | 0.415 | 0.415 | 0.415 |
| 4 | 13 | 0.231 | 0.231 | 0.231 |
| 5 † | 5 | 0.400 | 0.400 | 0.400 |


## Fit and likelihood

What each pipeline cost. *Models fitted* counts the plain model plus one support-deterministic model per distinct cause the questions asked about and the pipeline could fit; *training seconds* and the *nodes*/*edges* of every fitted circuit are summed over them, which for the relational circuit includes the part templates.

| pipeline | models fitted | training seconds | nodes | edges |
|---|---|---|---|---|
| relational circuit | 5 | 8.09 | 7266 | 7256 |
| propositional tree | 3 | 0.91 | 2003 | 2000 |
| unrolled tree | 4 | 5.12 | 24312 | 24308 |
| scalars-only tree | 2 | 0.08 | 1022 | 1020 |
| regression adjustment | 6 | 0.26 | 0 | 0 |
| neural adjustment | 8 | 0.26 | 0 | 0 |

How well each explains attempts it never saw, on three views of one attempt: its own scalars, which every pipeline models; its scalars and counts; and the whole attempt, parts included, which only the pipelines that model the parts can score. The relational circuit scores a whole attempt as its class circuit over the scalars and counts times each part template over one part given the counts; the unrolled tree scores it as one row. *Held-out coverage* is the share of held-out attempts that lie inside the plain model's support at all, since a tree's leaves span only the value ranges they were fitted on, and a whole attempt is covered only if every one of its parts is. The *mean log-likelihood* is over the covered attempts only; the last column restricts it to the attempts every pipeline in the table covers, so the numbers are over the same rows.

### scalars

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|
| relational circuit | 88.3% | 15.25 | 15.37 |
| propositional tree | 88.3% | 15.25 | 15.37 |
| unrolled tree | 95.0% | 4.89 | 4.84 |
| scalars-only tree | 90.0% | 9.33 | 9.27 |

### scalars and counts

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|
| relational circuit | 85.0% | 12.67 | 12.86 |
| propositional tree | 85.0% | 12.67 | 12.86 |
| unrolled tree | 95.0% | 3.67 | 3.63 |

### whole attempt

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|
| relational circuit | 78.3% | 141.63 | 154.33 |
| unrolled tree | 43.3% | 91.00 | 88.87 |

## Seconds per question

Wall-clock time from asking to the answer or the refusal. The *first ask* of a cause includes fitting that cause's own support-deterministic model; *asked again* repeats the question with every model fitted, so only grounding (for the relational circuit), verification and backdoor adjustment remain. A refusal is fast when it is a schema check; a relational answer draws Monte-Carlo samples for every count the query leaves open and grounds one part template per sampled value, which is where its time goes.

| question | relational circuit, first ask | relational circuit, asked again | propositional tree, first ask | propositional tree, asked again | unrolled tree, first ask | unrolled tree, asked again | scalars-only tree, first ask | scalars-only tree, asked again | regression adjustment, first ask | regression adjustment, asked again | neural adjustment, first ask | neural adjustment, asked again |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| friction_causes_lift_9_neighbours | 4.90 | 4.38 | 0.89 | 0.60 | 3.35 | 0.96 | 0.66 | 1.02 | 0.05 | - | 0.40 | - |
| crowding_causes_lift_9_neighbours_adjusting_environment | 4.70 | 3.70 | 0.93 | 1.00 | 3.05 | 1.32 | 0.03 | 0.03 | 0.07 | - | 0.08 | - |
| crowding_causes_lift_9_neighbours_unadjusted | 3.77 | 3.67 | 0.33 | 0.33 | 0.60 | 0.56 | 0.03 | 0.03 | 0.06 | - | 0.12 | - |
| closing_axis_side_causes_disturbance_of_neighbour_0_of_9 | 5.91 | 4.15 | 0.03 | 0.03 | 1.16 | 0.24 | 0.03 | 0.03 | 0.03 | - | 1.73 | - |
| friction_causes_lift_4_neighbours | 1.76 | 2.27 | 0.94 | 0.56 | 1.02 | 0.97 | 0.59 | 0.58 | 0.03 | - | 0.40 | - |
| crowding_causes_lift_12_neighbours_adjusting_environment | 4.82 | 4.73 | 0.61 | 0.59 | 1.33 | 1.28 | 0.04 | 0.04 | 0.09 | - | 0.09 | - |
| closing_axis_side_causes_disturbance_of_neighbour_11_of_12 | 7.18 | 6.31 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.05 | 0.07 | - | 1.73 | - |

## Does the order of the parts matter?

Every attempt's parts were put in a random order, 20 times over, and each time the pipelines that model the parts were refitted on the same split and asked the questions about parts again; the parts in the order the dataset lists them is the baseline every reordering is measured against. A relational circuit treats the neighbours as exchangeable, so nothing about it can depend on the order; an unrolled table's column `neighbours[0]` holds a different neighbour of every attempt after each reordering. Per question and pipeline: how many reorderings were answered; over the cause regions every answered ordering distinguishes, the mean standard deviation and the widest range of the adjusted probability; the share of reorderings whose most effective region is not the dataset-order one; and the share whose trend changed sign.

| question | pipeline | reorderings answered | mean sd of adjusted P(effect) | widest range | argmax moved | trend sign flipped |
|---|---|---|---|---|---|---|
| closing_axis_side_causes_disturbance_of_neighbour_0_of_9 | relational circuit | 20 of 20 | 0.000 | 0.00 | 0.0% | - |
| closing_axis_side_causes_disturbance_of_neighbour_0_of_9 | unrolled tree | 20 of 20 | 0.024 | 0.20 | 0.0% | - |

The whole-attempt likelihood of the same held-out attempts with the parts in the order the dataset lists them, and over the reorderings. The recording's order is the order the layout sampler drew the neighbours in, which carries nothing about where they stand. *Largest drop* is how far below the dataset-order likelihood the worst reordering took each pipeline.

| pipeline | dataset order, coverage / mean log-likelihood | reorderings, coverage / mean log-likelihood (mean ± sd) | largest drop |
|---|---|---|---|
| relational circuit | 78.3% / 141.63 | 0.783 ± 0.000 / 141.63 ± 0.00 | 0.00 |
| unrolled tree | 43.3% / 91.00 | 0.248 ± 0.066 / 61.80 ± 2.72 | 33.34 |

## Over several splits

The comparison repeated over 5 random splits (seeds 0, 1, 2, 3, 4), mean ± standard deviation. The likelihoods are over the attempts every pipeline modelling the view covers.

### scalars

| pipeline | held-out coverage | mean log-likelihood (covered by all) |
|---|---|---|
| relational circuit | 0.873 ± 0.020 | 12.83 ± 3.66 |
| propositional tree | 0.873 ± 0.020 | 12.83 ± 3.66 |
| unrolled tree | 0.900 ± 0.037 | 4.80 ± 0.09 |
| scalars-only tree | 0.873 ± 0.040 | 9.06 ± 0.53 |

### scalars and counts

| pipeline | held-out coverage | mean log-likelihood (covered by all) |
|---|---|---|
| relational circuit | 0.857 ± 0.023 | 11.11 ± 2.99 |
| propositional tree | 0.857 ± 0.023 | 11.11 ± 2.99 |
| unrolled tree | 0.893 ± 0.042 | 3.43 ± 0.22 |

### whole attempt

| pipeline | held-out coverage | mean log-likelihood (covered by all) |
|---|---|---|
| relational circuit | 0.780 ± 0.055 | 139.91 ± 8.48 |
| unrolled tree | 0.443 ± 0.044 | 87.08 ± 3.10 |

Per question, how many splits each pipeline answered, and the mean ± standard deviation over the splits of its trend and of its contrast:

| question | relational circuit, answered | relational circuit, trend | relational circuit, contrast | propositional tree, answered | propositional tree, trend | propositional tree, contrast | unrolled tree, answered | unrolled tree, trend | unrolled tree, contrast | scalars-only tree, answered | scalars-only tree, trend | scalars-only tree, contrast | regression adjustment, answered | regression adjustment, trend | regression adjustment, contrast | neural adjustment, answered | neural adjustment, trend | neural adjustment, contrast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| friction_causes_lift_9_neighbours | 5 of 5 | 0.84 ± 0.04 | 1.00 ± 0.00 | 5 of 5 | 0.84 ± 0.04 | 1.00 ± 0.00 | 5 of 5 | 0.84 ± 0.04 | 1.00 ± 0.00 | 5 of 5 | 0.84 ± 0.04 | 1.00 ± 0.00 | 5 of 5 | 1.00 ± 0.00 | 0.32 ± 0.01 | 5 of 5 | 1.00 ± 0.00 | 0.65 ± 0.05 |
| crowding_causes_lift_9_neighbours_adjusting_environment | 5 of 5 | -1.00 ± 0.00 | -0.71 ± 0.08 | 5 of 5 | -1.00 ± 0.00 | -0.71 ± 0.08 | 5 of 5 | -0.92 ± 0.04 | -0.59 ± 0.07 | 0 of 5 | - | - | 5 of 5 | -1.00 ± 0.00 | -0.43 ± 0.05 | 5 of 5 | -1.00 ± 0.00 | -0.33 ± 0.04 |
| crowding_causes_lift_9_neighbours_unadjusted | 5 of 5 | -0.98 ± 0.04 | -0.66 ± 0.08 | 5 of 5 | -0.98 ± 0.04 | -0.66 ± 0.08 | 5 of 5 | -0.98 ± 0.04 | -0.66 ± 0.08 | 0 of 5 | - | - | 5 of 5 | -1.00 ± 0.00 | -0.65 ± 0.02 | 5 of 5 | -1.00 ± 0.00 | -0.60 ± 0.03 |
| closing_axis_side_causes_disturbance_of_neighbour_0_of_9 | 5 of 5 | - | 0.16 ± 0.02 | 0 of 5 | - | - | 5 of 5 | - | 0.20 ± 0.01 | 0 of 5 | - | - | 0 of 5 | - | - | 5 of 5 | - | 0.06 ± 0.03 |
| friction_causes_lift_4_neighbours | 5 of 5 | 0.84 ± 0.04 | 1.00 ± 0.00 | 5 of 5 | 0.84 ± 0.04 | 1.00 ± 0.00 | 5 of 5 | 0.84 ± 0.04 | 1.00 ± 0.00 | 5 of 5 | 0.84 ± 0.04 | 1.00 ± 0.00 | 5 of 5 | 1.00 ± 0.00 | 0.32 ± 0.01 | 5 of 5 | 1.00 ± 0.00 | 0.65 ± 0.05 |
| crowding_causes_lift_12_neighbours_adjusting_environment | 5 of 5 | -1.00 ± 0.00 | -0.71 ± 0.08 | 5 of 5 | -1.00 ± 0.00 | -0.71 ± 0.08 | 5 of 5 | -0.92 ± 0.04 | -0.59 ± 0.07 | 0 of 5 | - | - | 5 of 5 | -1.00 ± 0.00 | -0.43 ± 0.05 | 5 of 5 | -1.00 ± 0.00 | -0.33 ± 0.04 |
| closing_axis_side_causes_disturbance_of_neighbour_11_of_12 | 5 of 5 | - | 0.16 ± 0.02 | 0 of 5 | - | - | 0 of 5 | - | - | 0 of 5 | - | - | 0 of 5 | - | - | 5 of 5 | - | 0.06 ± 0.03 |

## How much training data it takes

Every pipeline's plain model fitted on a growing share of the attempts and scored on the same held-out fifth, over 3 splits, mean ± standard deviation of the held-out coverage and of the mean log-likelihood over the covered attempts. The relational circuit's templates pool every part of every training attempt, where the unrolled tree sees one row per attempt. Every neighbour of every training attempt goes into the template.

### scalars and counts

| training share | relational circuit, coverage | relational circuit, mean log-likelihood | propositional tree, coverage | propositional tree, mean log-likelihood | unrolled tree, coverage | unrolled tree, mean log-likelihood |
|---|---|---|---|---|---|---|
| 20.0% | 0.611 ± 0.083 | 8.03 ± 1.01 | 0.611 ± 0.083 | 8.03 ± 1.01 | 0.483 ± 0.083 | 2.63 ± 0.59 |
| 40.0% | 0.739 ± 0.067 | 9.54 ± 1.17 | 0.739 ± 0.067 | 9.54 ± 1.17 | 0.778 ± 0.075 | 2.87 ± 0.43 |
| 60.0% | 0.817 ± 0.014 | 10.00 ± 1.98 | 0.817 ± 0.014 | 10.00 ± 1.98 | 0.806 ± 0.070 | 3.38 ± 0.26 |
| 80.0% | 0.850 ± 0.000 | 9.91 ± 2.00 | 0.850 ± 0.000 | 9.91 ± 2.00 | 0.883 ± 0.047 | 3.51 ± 0.12 |

### whole attempt

| training share | relational circuit, coverage | relational circuit, mean log-likelihood | unrolled tree, coverage | unrolled tree, mean log-likelihood |
|---|---|---|---|---|
| 20.0% | 0.300 ± 0.076 | 102.05 ± 4.47 | 0.022 ± 0.031 | 98.09 ± 0.00 |
| 40.0% | 0.544 ± 0.087 | 119.82 ± 4.00 | 0.206 ± 0.075 | 92.56 ± 1.90 |
| 60.0% | 0.667 ± 0.036 | 125.32 ± 7.05 | 0.283 ± 0.085 | 87.69 ± 4.08 |
| 80.0% | 0.761 ± 0.021 | 134.67 ± 6.98 | 0.433 ± 0.054 | 89.09 ± 2.82 |

## Error against known truth

Attempts sampled from a structural causal model over the same domain, whose interventional probabilities are known by construction. The mechanism the synthetic attempts are drawn from gives the hold's probability in closed form, so forcing the friction or the crowding leaves an expectation over layouts and the truth is exact in the outcome. The environment is the confounder, since a bin packs the cartons more tightly and holds only the slippery ones, and one setting removes that by letting both environments draw from the whole friction ladder. Every pipeline was fitted on 400 attempts per setting and asked the questions; *mean* and *max absolute error* are over every supported cause region of every answered question, the *support-weighted* error weighs each region by the training rows it holds, *worst ordering* is the mean absolute error under the reordering of the parts the pipeline did worst on, and *rank correlation* is Spearman's between the answered and the true probabilities over a question's regions.

| pipeline | questions answered | mean abs. error | support-weighted abs. error | max abs. error | mean abs. error, worst ordering | rank correlation with truth |
|---|---|---|---|---|---|---|
| relational circuit | 100.0% | 0.087 | 0.074 | 0.410 | 0.087 | 0.97 |
| propositional tree | 100.0% | 0.087 | 0.074 | 0.410 | 0.087 | 0.97 |
| unrolled tree | 100.0% | 0.084 | 0.066 | 0.410 | 0.092 | 0.98 |
| scalars-only tree | 50.0% | 0.119 | 0.075 | 0.410 | 0.119 | 1.00 |
| regression adjustment | 100.0% | 0.050 | 0.048 | 0.247 | 0.050 | 1.00 |
| neural adjustment | 100.0% | 0.035 | 0.040 | 0.108 | 0.035 | 1.00 |

Mean absolute error per setting of the model:

| hold lost per neighbour | confounded | relational circuit | propositional tree | unrolled tree | scalars-only tree | regression adjustment | neural adjustment |
|---|---|---|---|---|---|---|---|
| 0.10 | yes | 0.091 | 0.091 | 0.088 | 0.092 | 0.082 | 0.034 |
| 0.25 | yes | 0.116 | 0.116 | 0.110 | 0.167 | 0.040 | 0.040 |
| 0.40 | yes | 0.114 | 0.114 | 0.112 | 0.181 | 0.030 | 0.035 |
| 0.25 | no | 0.028 | 0.028 | 0.027 | 0.038 | 0.047 | 0.032 |

Mean absolute error per question, over every setting:

| question | relational circuit | propositional tree | unrolled tree | scalars-only tree | regression adjustment | neural adjustment |
|---|---|---|---|---|---|---|
| friction_causes_lift_9_neighbours | 0.119 | 0.119 | 0.119 | 0.119 | 0.065 | 0.032 |
| crowding_causes_lift_9_neighbours_adjusting_environment | 0.049 | 0.049 | 0.043 | - | 0.031 | 0.039 |

## How many grounding samples it takes

Inference on a grounded circuit is exact; grounding itself draws Monte-Carlo samples for every count the query leaves open and mixes one copy of the part templates per sampled value, so marginalising the open counts is a consistent estimate, not an exact sum. The relational circuit was fitted once and asked the same two questions with grounding drawing more and more samples; *deviation* is the largest difference, over the cause regions, from the answer at 32,000 samples, and *settled from* is the smallest number of samples from which every larger one stays within 0.01 of it.

### crowding_causes_lift_9_neighbours_adjusting_environment

Settled from 50 samples.

| samples | answered | deviation from reference | seconds |
|---|---|---|---|
| 50 | answered | 0.000 | 4.3 |
| 200 | answered | 0.000 | 3.9 |
| 1,000 | answered | 0.000 | 3.9 |
| 2,000 | answered | 0.000 | 3.8 |
| 8,000 | answered | 0.000 | 4.4 |
| 32,000 | answered | 0.000 | 3.8 |

### closing_axis_side_causes_disturbance_of_neighbour_0_of_9

Settled from 50 samples.

| samples | answered | deviation from reference | seconds |
|---|---|---|---|
| 50 | answered | 0.000 | 5.5 |
| 200 | answered | 0.000 | 4.6 |
| 1,000 | answered | 0.000 | 4.8 |
| 2,000 | answered | 0.000 | 4.6 |
| 8,000 | answered | 0.000 | 4.7 |
| 32,000 | answered | 0.000 | 4.6 |


## Cost against the number of objects

The pipelines that model the parts, fitted on synthetic attempts of growing size (400 attempts each) and asked one question about a part. The relational circuit's part templates pool every part of every attempt into one circuit, so their size follows the number of distinct part attributes, not the number of parts; the unrolled table carries one block of columns per position, so its tree grows with the widest attempt. *First ask* includes fitting the cause-specific model, *asked again* is grounding and adjustment alone. The synthetic attempts come from the same layout sampler as the recorded ones, with a random outcome in place of the simulator.

| parts per attempt | relational circuit, fit seconds | relational circuit, nodes | relational circuit, first ask | relational circuit, asked again | unrolled tree, fit seconds | unrolled tree, nodes | unrolled tree, first ask | unrolled tree, asked again |
|---|---|---|---|---|---|---|---|---|
| 4 | 2.4 | 1808 | 1.9 | 0.4 | 1.2 | 2769 | 0.8 | 0.1 |
| 9 | 3.2 | 3103 | 2.8 | 0.9 | 1.9 | 5779 | 1.3 | 0.2 |
| 16 | 3.3 | 4870 | 3.1 | 1.0 | 3.8 | 9815 | 3.2 | 0.3 |
| 25 | 5.0 | 7148 | 5.0 | 2.3 | 9.2 | 14709 | 6.8 | 0.5 |

## What the results show

- The relational circuit answered 7 of 7 questions.
- The propositional tree answered 5 of 7 questions, refusing `closing_axis_side_causes_disturbance_of_neighbour_0_of_9` because the fitted table has no column for the queried variables; `closing_axis_side_causes_disturbance_of_neighbour_11_of_12` because the fitted table has no column for the queried variables.
- The unrolled tree answered 6 of 7 questions, refusing `closing_axis_side_causes_disturbance_of_neighbour_11_of_12` because the fitted table has no column for the queried variables.
- The scalars-only tree answered 2 of 7 questions, refusing `crowding_causes_lift_9_neighbours_adjusting_environment` because the fitted table has no column for the queried variables; `crowding_causes_lift_9_neighbours_unadjusted` because the fitted table has no column for the queried variables; `closing_axis_side_causes_disturbance_of_neighbour_0_of_9` because the fitted table has no column for the queried variables; `crowding_causes_lift_12_neighbours_adjusting_environment` because the fitted table has no column for the queried variables; `closing_axis_side_causes_disturbance_of_neighbour_11_of_12` because the fitted table has no column for the queried variables.
- The regression adjustment answered 5 of 7 questions, refusing `closing_axis_side_causes_disturbance_of_neighbour_0_of_9` because the fitted table has no column for the queried variables; `closing_axis_side_causes_disturbance_of_neighbour_11_of_12` because the fitted table has no column for the queried variables.
- The neural adjustment answered 7 of 7 questions.
- On `friction_causes_lift_9_neighbours`, the pipelines disagree on the most effective setting: the relational circuit, the propositional tree, the unrolled tree and the scalars-only tree say a grasp friction coefficient of 0.375 (1.00, 1.00, 1.00, 1.00); the regression adjustment and the neural adjustment say a grasp friction coefficient of 0.75 (0.90, 1.00).
- On `crowding_causes_lift_9_neighbours_adjusting_environment`, every pipeline that answered finds 0 adjacent neighbours the most effective setting (adjusted probabilities: relational circuit 0.88, propositional tree 0.88, unrolled tree 0.88, regression adjustment 0.80, neural adjustment 0.81).
- On `crowding_causes_lift_9_neighbours_unadjusted`, every pipeline that answered finds 0 adjacent neighbours the most effective setting (adjusted probabilities: relational circuit 0.88, propositional tree 0.88, unrolled tree 0.88, regression adjustment 0.87, neural adjustment 0.89).
- On `closing_axis_side_causes_disturbance_of_neighbour_0_of_9`, every pipeline that answered finds neighbour 0 standing along the closing axis the most effective setting (adjusted probabilities: relational circuit 0.18, unrolled tree 0.22, neural adjustment 0.12).
- On `friction_causes_lift_4_neighbours`, the pipelines disagree on the most effective setting: the relational circuit, the propositional tree, the unrolled tree and the scalars-only tree say a grasp friction coefficient of 0.375 (1.00, 1.00, 1.00, 1.00); the regression adjustment and the neural adjustment say a grasp friction coefficient of 0.75 (0.90, 1.00).
- On `crowding_causes_lift_12_neighbours_adjusting_environment`, every pipeline that answered finds 0 adjacent neighbours the most effective setting (adjusted probabilities: relational circuit 0.88, propositional tree 0.88, unrolled tree 0.88, regression adjustment 0.80, neural adjustment 0.81).
- On `closing_axis_side_causes_disturbance_of_neighbour_11_of_12`, every pipeline that answered finds neighbour 11 standing along the closing axis the most effective setting (adjusted probabilities: relational circuit 0.18, neural adjustment 0.12).
- On the scalars, the relational circuit assigns the highest mean log-likelihood (15.37, against propositional tree 15.37, unrolled tree 4.84, scalars-only tree 9.27) to the held-out attempts every pipeline covers; coverage: relational circuit 88.3%, propositional tree 88.3%, unrolled tree 95.0%, scalars-only tree 90.0%.
- On the scalars and counts, the propositional tree assigns the highest mean log-likelihood (12.86, against relational circuit 12.86, unrolled tree 3.63) to the held-out attempts every pipeline covers; coverage: relational circuit 85.0%, propositional tree 85.0%, unrolled tree 95.0%.
- On the whole attempt, the relational circuit assigns the highest mean log-likelihood (154.33, against unrolled tree 88.87) to the held-out attempts every pipeline covers; coverage: relational circuit 78.3%, unrolled tree 43.3%.
- Over 20 reorderings, the relational circuit's adjusted effect probabilities ranged by up to 0.00 and its most effective region moved in 0.0% of the reorderings; its whole-attempt mean log-likelihood fell by up to 0.00 from the dataset's own order.
- Over 20 reorderings, the unrolled tree's adjusted effect probabilities ranged by up to 0.20 and its most effective region moved in 0.0% of the reorderings; its whole-attempt mean log-likelihood fell by up to 33.34 from the dataset's own order.
- The relational circuit takes 4.17 seconds per answered question on average once its models are fitted.
- The propositional tree takes 0.61 seconds per answered question on average once its models are fitted.
- The unrolled tree takes 0.89 seconds per answered question on average once its models are fitted.
- The scalars-only tree takes 0.80 seconds per answered question on average once its models are fitted.

## In a clutter of 9 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0.125 | 58 | 0.242 | 0.000 | 0.000 | [0.00, 0.06] |
| 0.1875 | 65 | 0.271 | 0.862 | 0.885 | [0.78, 0.94] |
| 0.25 | 66 | 0.275 | 0.758 | 0.790 | [0.68, 0.87] |
| 0.375 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |
| 0.5 | 15 | 0.062 | 1.000 | 1.000 | [0.80, 1.00] |
| 0.75 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |

EQL's own `cause` search settles on 0.1875: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0.125 | 58 | 0.242 | 0.000 | 0.000 | [0.00, 0.06] |
| 0.1875 | 65 | 0.271 | 0.862 | 0.885 | [0.78, 0.94] |
| 0.25 | 66 | 0.275 | 0.758 | 0.790 | [0.68, 0.87] |
| 0.375 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |
| 0.5 | 15 | 0.062 | 1.000 | 1.000 | [0.80, 1.00] |
| 0.75 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |

EQL's own `cause` search settles on 0.1875: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0.125 | 58 | 0.242 | 0.000 | 0.000 | [0.00, 0.06] |
| 0.1875 | 65 | 0.271 | 0.862 | 0.883 | [0.78, 0.94] |
| 0.25 | 66 | 0.275 | 0.758 | 0.782 | [0.67, 0.86] |
| 0.375 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |
| 0.5 | 15 | 0.062 | 1.000 | 1.000 | [0.80, 1.00] |
| 0.75 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |

EQL's own `cause` search settles on 0.1875: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### scalars-only tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0.125 | 58 | 0.242 | 0.000 | 0.000 | [0.00, 0.06] |
| 0.1875 | 65 | 0.271 | 0.862 | 0.885 | [0.78, 0.94] |
| 0.25 | 66 | 0.275 | 0.758 | 0.790 | [0.68, 0.87] |
| 0.375 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |
| 0.5 | 15 | 0.062 | 1.000 | 1.000 | [0.80, 1.00] |
| 0.75 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |

EQL's own `cause` search settles on 0.1875: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0.125 | 58 | 0.242 | 0.000 | 0.591 | [0.46, 0.71] |
| 0.1875 | 65 | 0.271 | 0.862 | 0.632 | [0.51, 0.74] |
| 0.25 | 66 | 0.275 | 0.758 | 0.671 | [0.55, 0.77] |
| 0.375 | 18 | 0.075 | 1.000 | 0.744 | [0.51, 0.89] |
| 0.5 | 15 | 0.062 | 1.000 | 0.807 | [0.56, 0.93] |
| 0.75 | 18 | 0.075 | 1.000 | 0.899 | [0.68, 0.97] |

EQL's own `cause` search settles on 0.75: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.90).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0.125 | 58 | 0.242 | 0.000 | 0.276 | [0.18, 0.40] |
| 0.1875 | 65 | 0.271 | 0.862 | 0.620 | [0.50, 0.73] |
| 0.25 | 66 | 0.275 | 0.758 | 0.856 | [0.75, 0.92] |
| 0.375 | 18 | 0.075 | 1.000 | 0.993 | [0.81, 1.00] |
| 0.5 | 15 | 0.062 | 1.000 | 1.000 | [0.80, 1.00] |
| 0.75 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |

EQL's own `cause` search settles on 0.75: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).


## In a clutter of 9 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 101 | 0.421 | 0.881 | 0.881 | [0.80, 0.93] |
| 1 | 24 | 0.100 | 0.875 | 0.868 | [0.68, 0.95] |
| 2 | 56 | 0.233 | 0.446 | 0.446 | [0.32, 0.58] |
| 3 | 41 | 0.171 | 0.415 | 0.415 | [0.28, 0.57] |
| 4 | 13 | 0.054 | 0.231 | 0.231 | [0.08, 0.50] |
| 5 † | 5 | 0.021 | 0.400 | 0.400 | [0.12, 0.77] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 101 | 0.421 | 0.881 | 0.881 | [0.80, 0.93] |
| 1 | 24 | 0.100 | 0.875 | 0.868 | [0.68, 0.95] |
| 2 | 56 | 0.233 | 0.446 | 0.446 | [0.32, 0.58] |
| 3 | 41 | 0.171 | 0.415 | 0.415 | [0.28, 0.57] |
| 4 | 13 | 0.054 | 0.231 | 0.231 | [0.08, 0.50] |
| 5 † | 5 | 0.021 | 0.400 | 0.400 | [0.12, 0.77] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 101 | 0.421 | 0.881 | 0.881 | [0.80, 0.93] |
| 1 | 24 | 0.100 | 0.875 | 0.870 | [0.68, 0.95] |
| 2 | 56 | 0.233 | 0.446 | 0.446 | [0.32, 0.58] |
| 3 | 41 | 0.171 | 0.415 | 0.415 | [0.28, 0.57] |
| 4 | 13 | 0.054 | 0.231 | 0.231 | [0.08, 0.50] |
| 5 † | 5 | 0.021 | 0.400 | 0.400 | [0.12, 0.77] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 101 | 0.421 | 0.881 | 0.798 | [0.71, 0.86] |
| 1 | 24 | 0.100 | 0.875 | 0.721 | [0.52, 0.86] |
| 2 | 56 | 0.233 | 0.446 | 0.629 | [0.50, 0.74] |
| 3 | 41 | 0.171 | 0.415 | 0.529 | [0.38, 0.67] |
| 4 | 13 | 0.054 | 0.231 | 0.426 | [0.21, 0.68] |
| 5 † | 5 | 0.021 | 0.400 | 0.329 | [0.08, 0.72] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.80).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 101 | 0.421 | 0.881 | 0.814 | [0.73, 0.88] |
| 1 | 24 | 0.100 | 0.875 | 0.775 | [0.58, 0.90] |
| 2 | 56 | 0.233 | 0.446 | 0.682 | [0.55, 0.79] |
| 3 | 41 | 0.171 | 0.415 | 0.608 | [0.46, 0.74] |
| 4 | 13 | 0.054 | 0.231 | 0.541 | [0.29, 0.77] |
| 5 † | 5 | 0.021 | 0.400 | 0.505 | [0.17, 0.83] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.81).


## In a clutter of 9 neighbours, how many of them standing adjacent to the target causes it to be lifted, with nothing adjusted for?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 101 | 0.421 | 0.881 | 0.881 | [0.80, 0.93] |
| 1 | 24 | 0.100 | 0.875 | 0.875 | [0.69, 0.96] |
| 2 | 56 | 0.233 | 0.446 | 0.446 | [0.32, 0.58] |
| 3 | 41 | 0.171 | 0.415 | 0.415 | [0.28, 0.57] |
| 4 | 13 | 0.054 | 0.231 | 0.231 | [0.08, 0.50] |
| 5 † | 5 | 0.021 | 0.400 | 0.400 | [0.12, 0.77] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 101 | 0.421 | 0.881 | 0.881 | [0.80, 0.93] |
| 1 | 24 | 0.100 | 0.875 | 0.875 | [0.69, 0.96] |
| 2 | 56 | 0.233 | 0.446 | 0.446 | [0.32, 0.58] |
| 3 | 41 | 0.171 | 0.415 | 0.415 | [0.28, 0.57] |
| 4 | 13 | 0.054 | 0.231 | 0.231 | [0.08, 0.50] |
| 5 † | 5 | 0.021 | 0.400 | 0.400 | [0.12, 0.77] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 101 | 0.421 | 0.881 | 0.881 | [0.80, 0.93] |
| 1 | 24 | 0.100 | 0.875 | 0.875 | [0.69, 0.96] |
| 2 | 56 | 0.233 | 0.446 | 0.446 | [0.32, 0.58] |
| 3 | 41 | 0.171 | 0.415 | 0.415 | [0.28, 0.57] |
| 4 | 13 | 0.054 | 0.231 | 0.231 | [0.08, 0.50] |
| 5 † | 5 | 0.021 | 0.400 | 0.400 | [0.12, 0.77] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 101 | 0.421 | 0.881 | 0.865 | [0.79, 0.92] |
| 1 | 24 | 0.100 | 0.875 | 0.748 | [0.55, 0.88] |
| 2 | 56 | 0.233 | 0.446 | 0.577 | [0.45, 0.70] |
| 3 | 41 | 0.171 | 0.415 | 0.386 | [0.25, 0.54] |
| 4 | 13 | 0.054 | 0.231 | 0.225 | [0.08, 0.50] |
| 5 † | 5 | 0.021 | 0.400 | 0.118 | [0.01, 0.55] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.87).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 101 | 0.421 | 0.881 | 0.889 | [0.81, 0.94] |
| 1 | 24 | 0.100 | 0.875 | 0.780 | [0.58, 0.90] |
| 2 | 56 | 0.233 | 0.446 | 0.491 | [0.37, 0.62] |
| 3 | 41 | 0.171 | 0.415 | 0.388 | [0.25, 0.54] |
| 4 | 13 | 0.054 | 0.231 | 0.306 | [0.13, 0.57] |
| 5 † | 5 | 0.021 | 0.400 | 0.239 | [0.05, 0.66] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.89).


## In a clutter of 9 neighbours, does neighbour 0 standing along the fingers' closing axis cause it to be disturbed by the pick?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| across | 1089 | 0.496 | 0.028 | 0.028 | [0.02, 0.04] |
| along | 1071 | 0.504 | 0.178 | 0.178 | [0.16, 0.20] |

EQL's own `cause` search settles on along: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.18).

### propositional tree

Refused: the fitted table has no column for the queried variables.

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| across | 113 | 0.471 | 0.027 | 0.027 | [0.01, 0.08] |
| along | 127 | 0.529 | 0.220 | 0.220 | [0.16, 0.30] |

EQL's own `cause` search settles on along: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.22).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

Refused: the fitted table has no column for the queried variables.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| across | 1071 | 0.496 | 0.022 | 0.082 | [0.07, 0.10] |
| along | 1089 | 0.504 | 0.213 | 0.117 | [0.10, 0.14] |

EQL's own `cause` search settles on along: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.12).


## In a clutter of 4 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0.125 | 58 | 0.242 | 0.000 | 0.000 | [0.00, 0.06] |
| 0.1875 | 65 | 0.271 | 0.862 | 0.885 | [0.78, 0.94] |
| 0.25 | 66 | 0.275 | 0.758 | 0.790 | [0.68, 0.87] |
| 0.375 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |
| 0.5 | 15 | 0.062 | 1.000 | 1.000 | [0.80, 1.00] |
| 0.75 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |

EQL's own `cause` search settles on 0.1875: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0.125 | 58 | 0.242 | 0.000 | 0.000 | [0.00, 0.06] |
| 0.1875 | 65 | 0.271 | 0.862 | 0.885 | [0.78, 0.94] |
| 0.25 | 66 | 0.275 | 0.758 | 0.790 | [0.68, 0.87] |
| 0.375 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |
| 0.5 | 15 | 0.062 | 1.000 | 1.000 | [0.80, 1.00] |
| 0.75 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |

EQL's own `cause` search settles on 0.1875: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0.125 | 58 | 0.242 | 0.000 | 0.000 | [0.00, 0.06] |
| 0.1875 | 65 | 0.271 | 0.862 | 0.883 | [0.78, 0.94] |
| 0.25 | 66 | 0.275 | 0.758 | 0.782 | [0.67, 0.86] |
| 0.375 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |
| 0.5 | 15 | 0.062 | 1.000 | 1.000 | [0.80, 1.00] |
| 0.75 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |

EQL's own `cause` search settles on 0.1875: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### scalars-only tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0.125 | 58 | 0.242 | 0.000 | 0.000 | [0.00, 0.06] |
| 0.1875 | 65 | 0.271 | 0.862 | 0.885 | [0.78, 0.94] |
| 0.25 | 66 | 0.275 | 0.758 | 0.790 | [0.68, 0.87] |
| 0.375 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |
| 0.5 | 15 | 0.062 | 1.000 | 1.000 | [0.80, 1.00] |
| 0.75 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |

EQL's own `cause` search settles on 0.1875: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0.125 | 58 | 0.242 | 0.000 | 0.591 | [0.46, 0.71] |
| 0.1875 | 65 | 0.271 | 0.862 | 0.632 | [0.51, 0.74] |
| 0.25 | 66 | 0.275 | 0.758 | 0.671 | [0.55, 0.77] |
| 0.375 | 18 | 0.075 | 1.000 | 0.744 | [0.51, 0.89] |
| 0.5 | 15 | 0.062 | 1.000 | 0.807 | [0.56, 0.93] |
| 0.75 | 18 | 0.075 | 1.000 | 0.899 | [0.68, 0.97] |

EQL's own `cause` search settles on 0.75: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.90).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0.125 | 58 | 0.242 | 0.000 | 0.276 | [0.18, 0.40] |
| 0.1875 | 65 | 0.271 | 0.862 | 0.620 | [0.50, 0.73] |
| 0.25 | 66 | 0.275 | 0.758 | 0.856 | [0.75, 0.92] |
| 0.375 | 18 | 0.075 | 1.000 | 0.993 | [0.81, 1.00] |
| 0.5 | 15 | 0.062 | 1.000 | 1.000 | [0.80, 1.00] |
| 0.75 | 18 | 0.075 | 1.000 | 1.000 | [0.82, 1.00] |

EQL's own `cause` search settles on 0.75: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).


## In a clutter of 12 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 101 | 0.421 | 0.881 | 0.881 | [0.80, 0.93] |
| 1 | 24 | 0.100 | 0.875 | 0.868 | [0.68, 0.95] |
| 2 | 56 | 0.233 | 0.446 | 0.446 | [0.32, 0.58] |
| 3 | 41 | 0.171 | 0.415 | 0.415 | [0.28, 0.57] |
| 4 | 13 | 0.054 | 0.231 | 0.231 | [0.08, 0.50] |
| 5 † | 5 | 0.021 | 0.400 | 0.400 | [0.12, 0.77] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 101 | 0.421 | 0.881 | 0.881 | [0.80, 0.93] |
| 1 | 24 | 0.100 | 0.875 | 0.868 | [0.68, 0.95] |
| 2 | 56 | 0.233 | 0.446 | 0.446 | [0.32, 0.58] |
| 3 | 41 | 0.171 | 0.415 | 0.415 | [0.28, 0.57] |
| 4 | 13 | 0.054 | 0.231 | 0.231 | [0.08, 0.50] |
| 5 † | 5 | 0.021 | 0.400 | 0.400 | [0.12, 0.77] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 101 | 0.421 | 0.881 | 0.881 | [0.80, 0.93] |
| 1 | 24 | 0.100 | 0.875 | 0.870 | [0.68, 0.95] |
| 2 | 56 | 0.233 | 0.446 | 0.446 | [0.32, 0.58] |
| 3 | 41 | 0.171 | 0.415 | 0.415 | [0.28, 0.57] |
| 4 | 13 | 0.054 | 0.231 | 0.231 | [0.08, 0.50] |
| 5 † | 5 | 0.021 | 0.400 | 0.400 | [0.12, 0.77] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 101 | 0.421 | 0.881 | 0.798 | [0.71, 0.86] |
| 1 | 24 | 0.100 | 0.875 | 0.721 | [0.52, 0.86] |
| 2 | 56 | 0.233 | 0.446 | 0.629 | [0.50, 0.74] |
| 3 | 41 | 0.171 | 0.415 | 0.529 | [0.38, 0.67] |
| 4 | 13 | 0.054 | 0.231 | 0.426 | [0.21, 0.68] |
| 5 † | 5 | 0.021 | 0.400 | 0.329 | [0.08, 0.72] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.80).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 101 | 0.421 | 0.881 | 0.814 | [0.73, 0.88] |
| 1 | 24 | 0.100 | 0.875 | 0.775 | [0.58, 0.90] |
| 2 | 56 | 0.233 | 0.446 | 0.682 | [0.55, 0.79] |
| 3 | 41 | 0.171 | 0.415 | 0.608 | [0.46, 0.74] |
| 4 | 13 | 0.054 | 0.231 | 0.541 | [0.29, 0.77] |
| 5 † | 5 | 0.021 | 0.400 | 0.505 | [0.17, 0.83] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.81).


## In a clutter of 12 neighbours, does neighbour 11 standing along the fingers' closing axis cause it to be disturbed by the pick?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| across | 1089 | 0.496 | 0.028 | 0.028 | [0.02, 0.04] |
| along | 1071 | 0.504 | 0.178 | 0.178 | [0.16, 0.20] |

EQL's own `cause` search settles on along: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.18).

### propositional tree

Refused: the fitted table has no column for the queried variables.

### unrolled tree

Refused: the fitted table has no column for the queried variables.

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

Refused: the fitted table has no column for the queried variables.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| across | 1071 | 0.496 | 0.022 | 0.082 | [0.07, 0.10] |
| along | 1089 | 0.504 | 0.213 | 0.117 | [0.10, 0.14] |

EQL's own `cause` search settles on along: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.12).

