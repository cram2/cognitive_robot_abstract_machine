---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(chapter:causal-structure)=
# Causal Structure

{ref}`chapter:causal-reasoning` answers an interventional query on a circuit once the
query says which variables to adjust for. This chapter covers the step before it:
learning the causal structure from the training data, and letting that structure say
which variables to adjust for. It is implemented in `probabilistic_model.causal_structure`.

## Causal circuits and the causal structure

The two packages answer different questions and work together.

|  | `probabilistic_circuit.causal` | `causal_structure` |
|---|---|---|
| answers | *how strong* is the effect of a cause, given the variables to adjust for | *which* variables to adjust for, and whether the effect can be identified at all |
| central class | `CausalCircuit`, a fitted circuit with its cause and effect variables | `CausalGraph` and `AdjustmentSelector`, learned from the circuit's training rows |
| input | a fitted circuit, a cause, an effect, an adjustment set | the training rows, and domain causal knowledge such as temporal order |
| output | the interventional circuit $P(Y \mid do(X))$ by backdoor adjustment | a valid adjustment set, a few candidate sets, or a refusal |
| checks | support determinism: is the circuit able to answer the query exactly | the adjustment criterion: does adjusting for the set answer the causal question |
| what it trusts | the adjustment set it is handed | the independences in the data and the domain causal knowledge |

A `CausalCircuit` computes the backdoor formula exactly for whatever adjustment set it
is given, and has no way to tell whether that set is the right one: a missing confounder
or an adjusted-for mediator gives a wrong number with no warning. The causal structure
supplies or checks that set. Estimation stays in the circuit; nothing in
`causal_structure` changes how an answer is computed.

## The pieces

- `graph.CausalGraph` holds a graph whose edges may be directed or undirected. An
  undirected edge is one the data cannot orient, so the graph stands for every
  directed acyclic graph that orients those edges without adding a collider:
  `member_graphs()` lists them, up to `maximum_member_graphs` of them.
- `identification.AdjustmentSelector` checks the adjustment criterion (Shpitser,
  VanderWeele and Robins, 2010; Perković et al., 2015) in every member graph. Its
  `identify` returns one of:
  - `IdentifiedEffect`: the best set that is valid in every member graph, ranked by
    how many of its combinations with the cause the training rows support only thinly,
    then by size;
  - `PartiallyIdentifiedEffect`: one candidate set per parent set of the cause, when
    the member graphs disagree on the cause's parents (Maathuis, Kalisch and Bühlmann,
    2009);
  - `UnidentifiedEffect`: no set of the eligible variables works.

  `check_declared` raises `InvalidAdjustmentSet` for a declared set that fails in some
  member graph. What a query does with such a set is a `DeclaredAdjustmentPolicy`:
  `InvalidDeclaredAdjustmentRefusal` raises, and `InvalidDeclaredAdjustmentRepair`
  keeps a valid declared set and otherwise prefers the valid sets that drop the fewest
  of its nodes.
- `learning.PeterClarkStructureLearner` runs the Peter-Clark algorithm through
  causal-learn on the rows, discretized into a few levels per column.
  `learning.GreedyEquivalenceSearchStructureLearner` runs greedy equivalence search
  with the Bayesian Dirichlet equivalent uniform score instead; it needs at least three
  columns that vary, and the domain causal knowledge only orients the edges it leaves
  undirected.
  `learning.FastCausalInferenceStructureLearner` allows hidden confounders, see
  {ref}`section:hidden-confounders`.
  `learning.RelationalCausalStructureLearner` does the same with any learner on a
  relational circuit's class table and part tables. A part attribute becomes one node for every
  part of its collection. The learned edges are kept as causal relations of the class
  in `krrood`'s class diagram.
- `domain_causal_knowledge.DomainCausalKnowledge` carries what the data cannot settle:
  - tiers, where a variable is never caused by one in a later tier;
  - required and forbidden edges;
  - deterministic copies, which the tests skip and the graph restores as children of
    their source.
- `relational_checks.ExampleToPartCheck.dependences(tables)` tests the independence a relational circuit
  assumes: an instance's attributes reach its parts' attributes only through the
  aggregation statistics. Given to `RelationalCausalStructureLearner` as its
  `example_to_part_check`, every pair that stays dependent given the statistics and any
  one other attribute the dependence could run through becomes an edge of the learned
  graph, from the instance attribute to the part attribute unless the tiers place the
  part attribute first.

## Using it from a query

`krrood`'s `ProbabilisticBackend` takes the learned structure of each class in
`causal_structures`. For such a class, a query that marks `cause` and `causes_effect`
but no `confounder` has its adjustment set chosen
from the learned graph, a marked `confounder` set is settled by the backend's
`declared_adjustment_policy`, and an effect the graph does not identify raises
`EffectNotIdentified`.
`ProbabilisticBackend.interventional_range` answers a partially identified effect with
the lowest and highest probability across the candidate sets. A query with
`causes_effect` and no `cause` lets `rank_causes` rank every variable that may be an
ancestor of the effect. See the EQL causality guide for examples.

## What the data has to supply

Learning the structure from observational data has limits:

- **Direction.** Without domain causal knowledge only the edges into colliders are
  oriented. Temporal order, such as "the environment is chosen before the friction",
  orients most of the rest.
- **Deterministic copies.** A variable computed from another breaks the independence
  tests. Testing two variables given the copy finds them independent and deletes a real
  edge, so copies have to be declared.
- **Sample size.** A discrete independence test over many cells has little power, so
  small samples miss real edges and the chosen set can be invalid.
- **Hidden confounders.** The Peter-Clark algorithm and greedy equivalence search
  assume every common cause is recorded. Fast causal inference does not; see
  {ref}`section:hidden-confounders`.
- **The relational assumption.** A relational circuit lets an instance's attributes
  affect its parts only through the aggregation statistics, and the relational learner
  follows it unless it is given an `ExampleToPartCheck`, which adds the edges where the
  data disagrees.
- **Interventions on counts.** Setting an aggregation statistic is ambiguous once it
  has parents among the part attributes, since many configurations of the parts give
  the same count (Spirtes and Scheines, 2004).

(section:hidden-confounders)=
## Hidden confounders

With a hidden common cause of the cause and the effect, a graph learned by the
Peter-Clark algorithm can still name an adjustment set, and the circuit then computes
the wrong number exactly, without a warning.
`learning.FastCausalInferenceStructureLearner` learns an `ancestral_graph.AncestralGraph`
instead {cite:p}`spirtes2000causation`: a partial ancestral graph (PAG), which stands for
every maximal ancestral graph (MAG) the data cannot tell apart {cite:p}`zhang2008completeness`.
An edge end is a tail where that node causes the other end, an arrow where it does not,
and a circle where the data leaves it open.

`AdjustmentSelector` checks such a graph with the generalized adjustment criterion
{cite:p}`perkovic2018complete`, through the same `identify` and `check_declared`. A set
is valid if

1. the graph is *amenable*: every possibly directed path from the cause to the effect
   starts with a visible edge out of the cause, one for which no hidden variable can
   cause both ends;
2. it holds neither the cause nor the effect;
3. it holds no node of the *forbidden set*: the nodes on those paths and their possible
   descendants;
4. it m-separates the cause from the effect in a representative MAG of the graph, once
   the edges from the cause into its causal paths are removed.

A set found this way is valid in every MAG the graph stands for, whatever hidden
variables they leave out. The selector refuses instead of guessing:

| reason | when |
|---|---|
| `NOT_AMENABLE` | an edge out of the cause may hide a confounder; `EffectNotIdentified` names it |
| `NO_VALID_SET_AMONG_ELIGIBLE` | the graph is amenable, but no eligible set is valid |
| `SELECTION_BIAS_EDGE` | an edge says the rows were selected, which is not supported |
| `NOT_ANCESTRAL` | the graph stands for no MAG |

There are no candidate-set ranges for this kind of graph: once a hidden variable may
cause the cause and its parents, a parent set is no adjustment set, so
`interventional_range` refuses an effect the graph does not identify.

**When to choose it.** Choose fast causal inference when a common cause of the cause and
the effect may be unrecorded. Its tests are the same G² tests on discretized columns, so
it needs about as many rows as the Peter-Clark algorithm, more for long paths.

**It refuses more often.** That is the price of not assuming every confounder is
recorded. With a recorded confounder $C$ and a hidden variable causing $C$ and $V$
($V \to X \to Y$, $C \to X$, $C \to Y$), $\{C\}$ is valid, but the PAG of that truth
has only circles, so the selector refuses. On 20,000 rows of a hidden confounder of $X$ and
$Y$ ($W \to X \to Y$, $X \leftarrow L \to Y$), the Peter-Clark algorithm returns the
undirected triangle over $W$, $X$ and $Y$, and on rows where $L$ alone links them
($W \to X \leftarrow L \to Y$) it returns $W \to X \leftarrow Y$, claiming $Y$ causes $X$.
Fast causal inference returns $W \circ\!\!\to X \leftarrow\!\!\circ Y$, which claims no
cause of $X$ at all.

**A MAG is not a directed acyclic graph with bidirected edges.** In the MAG
$C \to X$, $C \to Y$, $X \to Y$, the edge $X \to Y$ may still hide a confounder: the same
MAG stands for the graph with a hidden $L \to X$, $L \to Y$ added. Declared by hand, it is
refused. Adding $V \to X$ and $V \leftrightarrow C$ makes $X \to Y$ visible through
$V$, and the effect is identified by adjusting for $C$.

**Knowledge it rejects.** Tiers and forbidden edges raise
`UnsupportedKnowledgeUnderLatentConfounding`: causal-learn turns each into a fully
directed edge, which claims a cause a hidden variable may explain instead. Required
edges are kept.

**Relational circuits.** A union of PAGs learned from overlapping tables is no PAG, and a
part table's rows are not independent. The relational learner therefore learns one
graph per *scope* with fast causal inference and never merges them:

- the class scope, from the class table;
- one scope per collection, from a table of every instance with at least one part
  there, joined to one of its parts drawn at random (`part_draw_seed`).

Each learned relation records its scope. A query uses the graph of the scope its cause,
effect and declared confounders fall in: the class scope if none is a part attribute,
the collection's if they reach one part of it. A query that reaches two collections,
or two parts of one, raises `QuerySpansSeveralParts`. A collection's answer holds for
instances with at least one part there, and drawing one part per instance uses fewer
rows than the part table has. An `ExampleToPartCheck` is rejected, since the collection
tables learn the edges between an instance and its parts directly.

## Evaluation

The learned adjustment set was compared with the hand-declared one on two synthetic
generators whose true causal graph and true interventional probabilities are known:
a clutter-picking model (4 settings, 2 questions) and a cluttered-scene model
(5 settings, 10 questions). Each question was answered three ways on the same fitted
relational circuit, so only the adjustment set differs: with the declared set, with
the set the learned graph chooses, and with none. Each answer was scored by its mean
absolute error (MAE) against the true $P(Y \mid do(X))$, over 3 samples each of 400
and 1,000 examples.

| domain | examples | MAE declared | MAE learned | MAE none | graph SHD |
|---|---|---|---|---|---|
| clutter picking | 400 | 0.090 | 0.087 | 0.085 | 0.3–1.7 |
| clutter picking | 1,000 | 0.070 | 0.071 | 0.080 | 0.0–0.3 |
| cluttered scenes | 400 | 0.078 | 0.079 | 0.081 | 5.3–8.3 |
| cluttered scenes | 1,000 | 0.068 | 0.073 | 0.077 | 4.3–8.0 |

- Where the graph is learned well, the learned set matches the declared one: in the
  clutter model it recovers the declared confounder in 8 of 9 confounded fits at
  1,000 examples.
- Where a declared set is unnecessary or only a proxy, the learned set does better:
  0.028 against 0.035 without confounding in the clutter model, and 0.096 against
  0.105 for a scene question that declares a proxy of the confounder.
- Where the graph misses the confounder, it does worse: the Peter-Clark algorithm
  recovers 30–67% of the scene model's true adjacencies, mostly losing the object
  count's edges into each object, and the learned set then drops the object count the declared queries
  rightly adjust for.
- Learning the graph takes 1–27 s per fit, and choosing a set from it about 0.15 s
  per question.

So the learned structure is a check on a declared adjustment set and a default where
none is declared, not yet a replacement for a correct declaration: how good its answer
is depends on how well the graph is learned.
