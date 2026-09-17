# Tracy clutter picking: relational circuit against flat-table tree

Tracy's left arm picks one milk carton out of a clutter of ten in MuJoCo. The carton is
held by contact friction alone, so a weak grasp visibly fails. Every attempt is recorded
as a relational scene. The scene carries the attempt's own attributes (the environment,
the grasp friction, the grasp yaw, whether the target came up and how far it rose) and
one exchangeable part per neighbouring carton (where it stands relative to the target,
its distance band, which side of the fingers' closing axis it is on, and how far the
pick shoved it).

Two pipelines were fitted on the same recorded attempts and asked the same
`cause`/`causes_effect` EQL queries.

- The **relational circuit** is a relational probabilistic circuit fitted on the
  scenes' relational structure: one circuit over the attempt's own attributes and its
  aggregation statistics, and one template over a neighbour's attributes. For every
  query it grounds itself into a circuit over exactly the objects the query names and
  registers that circuit as a causal circuit.
- The **flat-table tree** is a joint probability tree fitted on the same attempts
  flattened into one fixed-width table, with one block of columns per neighbour
  position. It is registered as a causal circuit in the same way.

Both answer a query by backdoor adjustment. The model is stratified so that it is
support-deterministic over the cause, the probability of the effect is read off every
region of the cause, and any variable the query marks as a confounder is summed out of
that reading.

## Setup

- Recorded attempts: 300, of which 240 were used to fit and 60 were held out.
- Neighbours per attempt: 9.
- Attempts whose target was lifted: 65.3%.
- Fewest training rows per leaf: 15 in a cause-specific model and 50 in the plain
  model that scores the held-out attempts.

## How often the pick came up

These are the recorded attempts themselves, before any model is involved. Each table
gives the share of attempts whose target was still held at the end, grouped by the
environment the clutter stood in, by the friction coefficient of the grasp, and by how
many neighbours stood adjacent to the target, that is, closer than the sweep of the
fingers. This is the picking efficiency in clutter that the models are asked to
explain.

| environment | attempts | lifted |
|---|---|---|
| bin | 158 | 45.6% |
| table | 142 | 87.3% |

| friction coefficient | attempts | lifted |
|---|---|---|
| 0.125 | 76 | 0.0% |
| 0.1875 | 78 | 87.2% |
| 0.25 | 79 | 77.2% |
| 0.375 | 24 | 100.0% |
| 0.5 | 17 | 100.0% |
| 0.75 | 26 | 100.0% |

| adjacent neighbours | attempts | lifted |
|---|---|---|
| 0 | 130 | 86.9% |
| 1 | 28 | 85.7% |
| 2 | 73 | 45.2% |
| 3 | 48 | 41.7% |
| 4 | 16 | 25.0% |
| 5 | 5 | 40.0% |

Three things stand out. Picks in a bin succeed half as often as picks on a table. The
lowest friction level never holds the carton, while everything from 0.375 upwards
always does, so the physics has a sharp threshold between 0.125 and 0.1875. And every
adjacent neighbour costs success: a free-standing target comes up 87% of the time, a
target with four adjacent neighbours only 25% of the time. The bin is where both the
slippery cartons and the tight packing live, which is exactly why the questions below
adjust for it.

## Which questions each pipeline can answer

One row per question and one column per pipeline. An answered cell says in words which
setting of the cause makes the effect most likely after adjustment, how likely the
effect then is, and how that compares with the least favourable setting. A refused cell
says why the pipeline could not answer at all. Neighbours are numbered from 1.

| question | relational circuit | flat-table tree |
|---|---|---|
| In a clutter of 9 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment? | Answered. With a grasp friction coefficient of 0.375 the target is lifted with probability 1.00, the highest of any setting; with a coefficient of 0.125 it is only 0.00. | Answered. With a grasp friction coefficient of 0.375 the target is lifted with probability 1.00, the highest of any setting; with a coefficient of 0.125 it is only 0.00. |
| In a clutter of 9 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment? | Answered. With no adjacent neighbour the target is lifted with probability 0.88, the highest of any setting; with four adjacent neighbours it is only 0.23. | Answered. With no adjacent neighbour the target is lifted with probability 0.88, the highest of any setting; with four adjacent neighbours it is only 0.23. |
| In a clutter of 9 neighbours, does neighbour 1 standing along the fingers' closing axis cause it to be disturbed by the pick? | Answered. With neighbour 1 standing along the closing axis it is disturbed with probability 0.17, the highest of any setting; standing across the axis it is only 0.01. | Answered. With neighbour 1 standing along the closing axis it is disturbed with probability 0.22, the highest of any setting; standing across the axis it is only 0.03. |
| In a clutter of 4 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment? | Answered. With a grasp friction coefficient of 0.375 the target is lifted with probability 1.00, the highest of any setting; with a coefficient of 0.125 it is only 0.00. | Answered. With a grasp friction coefficient of 0.375 the target is lifted with probability 1.00, the highest of any setting; with a coefficient of 0.125 it is only 0.00. |
| In a clutter of 12 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment? | Answered. With no adjacent neighbour the target is lifted with probability 0.88, the highest of any setting; with four adjacent neighbours it is only 0.23. | Refused: the fitted table has no column for the queried variables. |
| In a clutter of 12 neighbours, does neighbour 12 standing along the fingers' closing axis cause it to be disturbed by the pick? | Answered. With neighbour 12 standing along the closing axis it is disturbed with probability 0.17, the highest of any setting; standing across the axis it is only 0.01. | Refused: the fitted table has no column for the queried variables. |

The three questions about a clutter of the recorded size can be put to either
pipeline. The two about a clutter of 12 have no columns in the flat table, so only a
model that grounds itself for the queried objects can answer them. The question about a
clutter of 4 asks about a subset of the fitted columns, which the flat table can still
serve.

## Fit and likelihood

This table shows what each pipeline cost and how well it explains attempts it never
saw.

- *Models fitted* counts the plain model plus one support-deterministic model per
  distinct cause the questions asked about.
- *Training seconds* and the *nodes* and *edges* of every fitted circuit are summed
  over all of those models.
- *Held-out coverage* is the share of held-out attempts that lie inside the plain
  model's support at all. A tree's leaves span only the value ranges they were fitted
  on, so an attempt with any attribute outside every leaf's range has zero
  likelihood.
- The *mean log-likelihood* is taken over the covered attempts only, on an attempt's
  observed attributes (its own scalars and every neighbour's). The last column restricts
  it to the attempts both pipelines cover, so the two numbers are over the same rows.

| pipeline | models fitted | training seconds | nodes | edges | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by both) |
|---|---|---|---|---|---|---|---|
| relational circuit | 5 | 8.01 | 9942 | 9932 | 30.0% | 117.90 | 130.07 |
| flat-table tree | 4 | 3.48 | 6838 | 6834 | 56.7% | 81.03 | 79.96 |

The relational circuit fits one model more than the flat tree, because a cause on a
neighbour attribute needs its own stratified template, and it is larger and slower to
fit. Its template is fitted on every neighbour of every attempt, nine times as many
rows as the flat table has, so its leaves are narrower and cover fewer held-out
attempts. Where both models do cover an attempt, the relational circuit explains it
markedly better: 130 against 80 in mean log-likelihood.

## Seconds per question

Wall-clock time from asking a question to its answer or its refusal. The *first ask*
of a cause includes fitting that cause's own support-deterministic model. *Asked again*
repeats the question with every model already fitted, so only grounding (for the
relational circuit), verification and backdoor adjustment remain. A refusal is fast when
it is a schema check; a relational answer about a larger clutter grounds a larger
circuit and takes longer.

| question | relational circuit, first ask | relational circuit, asked again | flat-table tree, first ask | flat-table tree, asked again |
|---|---|---|---|---|
| friction causes lift, 9 neighbours | 5.37 | 4.82 | 1.59 | 0.16 |
| crowding causes lift, 9 neighbours | 6.18 | 5.43 | 1.22 | 0.19 |
| closing-axis side of neighbour 1 causes disturbance, 9 neighbours | 7.92 | 6.90 | 0.70 | 0.10 |
| friction causes lift, 4 neighbours | 2.57 | 2.60 | 0.15 | 1.11 |
| crowding causes lift, 12 neighbours | 8.21 | 8.11 | 0.04 | 0.05 |
| closing-axis side of neighbour 12 causes disturbance, 12 neighbours | 11.71 | 8.82 | 0.26 | 0.26 |

## What the results show

- The relational circuit answered all six questions.
- The flat-table tree answered four of the six. It refused the two questions about a
  clutter of 12 neighbours, because the fitted table has no column for the queried
  variables.
- On the friction question about 9 neighbours, both pipelines find a grasp friction
  coefficient of 0.375 the most effective setting, with an adjusted probability of
  1.00 each.
- On the crowding question about 9 neighbours, both pipelines find no adjacent
  neighbour the most effective setting, with an adjusted probability of 0.88 each.
- On the question about neighbour 1 of 9, both pipelines find standing along the
  closing axis the most effective setting, with adjusted probabilities of 0.17 and
  0.22.
- On the friction question about 4 neighbours, both pipelines again find a coefficient
  of 0.375 the most effective setting, with an adjusted probability of 1.00 each.
- The flat-table tree covers the most held-out attempts (56.7%). On the attempts both
  pipelines cover, the relational circuit assigns the higher mean log-likelihood
  (130.07).
- Once its models are fitted, the relational circuit takes 6.11 seconds per answered
  question on average and the flat-table tree 0.39 seconds.

Read together: wherever both pipelines can answer, they agree on the effective setting
and, for scene-level causes, on the numbers to the third decimal, because they are
fitted on the same rows and stratified the same way. The relational circuit's advantage
is not a different answer but a wider reach. It answers about clutters of a different
size than it was fitted on, and about any neighbour by position, because it derives the
crowding count and the neighbour attributes from the relational structure instead of
reading them off fixed columns. It pays for that with roughly fifteen times the query
latency of the flat tree.

## In a clutter of 9 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment?

Each interventional table below has one row per region of the cause that the model
distinguishes. *P(region)* is how much of the recorded population that region holds.
*Naive P(effect)* is the probability of the effect simply conditioned on the region.
*Adjusted* is the interventional probability after summing out the question's
confounders, which is what the question asks for. Where the two columns agree, the
confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect given do(cause)) |
|---|---|---|---|
| 0.125 | 0.242 | 0.000 | 0.000 |
| 0.1875 | 0.271 | 0.862 | 0.885 |
| 0.25 | 0.275 | 0.758 | 0.790 |
| 0.375 | 0.075 | 1.000 | 1.000 |
| 0.5 | 0.062 | 1.000 | 1.000 |
| 0.75 | 0.075 | 1.000 | 1.000 |

EQL's own `cause` search settles on 0.1875, the region that is most probable once the
effect is required to hold and from which the query's samples are drawn; its adjusted
probability of the effect is 0.88.

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect given do(cause)) |
|---|---|---|---|
| 0.125 | 0.242 | 0.000 | 0.000 |
| 0.1875 | 0.271 | 0.862 | 0.862 |
| 0.25 | 0.275 | 0.758 | 0.757 |
| 0.375 | 0.075 | 1.000 | 1.000 |
| 0.5 | 0.062 | 1.000 | 1.000 |
| 0.75 | 0.075 | 1.000 | 1.000 |

EQL's own `cause` search settles on 0.1875, with an adjusted probability of the effect
of 0.86.

The effect of friction is monotone and sharp: no lift at 0.125, a lift in most attempts
from 0.1875 upwards, and a certain lift from 0.375. The adjustment for the environment
nudges the two middle levels upwards in the relational circuit (0.862 to 0.885 and
0.758 to 0.790), because at those levels the bin, with its crowded cartons, drags the
naive rate down; the flat tree's leaves are wider and absorb that difference.

## In a clutter of 9 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment?

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect given do(cause)) |
|---|---|---|---|
| 0 | 0.421 | 0.881 | 0.881 |
| 1 | 0.100 | 0.875 | 0.875 |
| 2 | 0.233 | 0.446 | 0.446 |
| 3 | 0.171 | 0.415 | 0.415 |
| 4 | 0.054 | 0.231 | 0.231 |
| 5 | 0.021 | 0.400 | 0.400 |

EQL's own `cause` search settles on 0 adjacent neighbours, with an adjusted probability
of the effect of 0.88.

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect given do(cause)) |
|---|---|---|---|
| 0 | 0.421 | 0.881 | 0.881 |
| 1 | 0.100 | 0.875 | 0.875 |
| 2 | 0.233 | 0.446 | 0.446 |
| 3 | 0.171 | 0.415 | 0.415 |
| 4 | 0.054 | 0.231 | 0.231 |
| 5 | 0.021 | 0.400 | 0.400 |

EQL's own `cause` search settles on 0 adjacent neighbours, with an adjusted probability
of the effect of 0.88.

One adjacent neighbour costs almost nothing; the second halves the chance of a lift and
the fourth quarters it. The value at five neighbours rests on five attempts and should
not be read as a recovery.

## In a clutter of 9 neighbours, does neighbour 1 standing along the fingers' closing axis cause it to be disturbed by the pick?

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect given do(cause)) |
|---|---|---|---|
| across | 0.496 | 0.012 | 0.012 |
| along | 0.504 | 0.170 | 0.170 |

EQL's own `cause` search settles on *along*, with an adjusted probability of the effect
of 0.17.

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect given do(cause)) |
|---|---|---|---|
| across | 0.471 | 0.027 | 0.027 |
| along | 0.529 | 0.220 | 0.220 |

EQL's own `cause` search settles on *along*, with an adjusted probability of the effect
of 0.22.

A neighbour standing where the fingers close is disturbed roughly fourteen times as
often as one standing to the side. The two pipelines differ here because the relational
template pools every neighbour of every attempt, while the flat tree sees only the
columns of neighbour 1, so the relational estimate rests on nine times as many rows.

## In a clutter of 4 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment?

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect given do(cause)) |
|---|---|---|---|
| 0.125 | 0.242 | 0.000 | 0.000 |
| 0.1875 | 0.271 | 0.862 | 0.885 |
| 0.25 | 0.275 | 0.758 | 0.790 |
| 0.375 | 0.075 | 1.000 | 1.000 |
| 0.5 | 0.062 | 1.000 | 1.000 |
| 0.75 | 0.075 | 1.000 | 1.000 |

EQL's own `cause` search settles on 0.1875, with an adjusted probability of the effect
of 0.88.

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect given do(cause)) |
|---|---|---|---|
| 0.125 | 0.242 | 0.000 | 0.000 |
| 0.1875 | 0.271 | 0.862 | 0.862 |
| 0.25 | 0.275 | 0.758 | 0.757 |
| 0.375 | 0.075 | 1.000 | 1.000 |
| 0.5 | 0.062 | 1.000 | 1.000 |
| 0.75 | 0.075 | 1.000 | 1.000 |

EQL's own `cause` search settles on 0.1875, with an adjusted probability of the effect
of 0.86.

The flat tree can answer this one despite the different clutter size because the
question names four neighbours and the table has columns for nine: the query's
variables are a subset of the fitted ones. The answer is the scene-level friction
effect, which does not depend on how many neighbours the query names.

## In a clutter of 12 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment?

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect given do(cause)) |
|---|---|---|---|
| 0 | 0.421 | 0.881 | 0.881 |
| 1 | 0.100 | 0.875 | 0.875 |
| 2 | 0.233 | 0.446 | 0.446 |
| 3 | 0.171 | 0.415 | 0.415 |
| 4 | 0.054 | 0.231 | 0.231 |
| 5 | 0.021 | 0.400 | 0.400 |

EQL's own `cause` search settles on 0 adjacent neighbours, with an adjusted probability
of the effect of 0.88.

### flat-table tree

Refused: the fitted table has no column for the queried variables.

The relational circuit grounds twelve neighbour templates under the same class circuit
and reads the crowding count off them, so its answer is the one it gave for nine
neighbours. The flat tree has no columns for a tenth, eleventh or twelfth neighbour
and cannot be asked.

## In a clutter of 12 neighbours, does neighbour 12 standing along the fingers' closing axis cause it to be disturbed by the pick?

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect given do(cause)) |
|---|---|---|---|
| across | 0.496 | 0.012 | 0.012 |
| along | 0.504 | 0.170 | 0.170 |

EQL's own `cause` search settles on *along*, with an adjusted probability of the effect
of 0.17.

### flat-table tree

Refused: the fitted table has no column for the queried variables.

Because the neighbours are exchangeable parts, the relational circuit's answer for the
twelfth neighbour of twelve is the same as for the first of nine. The flat tree would
need a column block for a twelfth neighbour and a separate stratified model for it.
