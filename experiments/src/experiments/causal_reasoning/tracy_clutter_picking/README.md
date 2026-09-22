# Tracy clutter picking: relational circuit against flat-table trees

## What this experiment is for

Relational sum-product networks (RSPNs) are less expressive than most machine-learning
models. The argument this experiment makes is that they are expressive *enough*. A
relational circuit fitted on a robot's own recorded attempts can answer causal questions
about a cluttered pick, including questions a flat model cannot even pose, and the
answers hold up on a task a robot actually executes.

Tracy's left arm picks one milk carton out of a clutter of ten in MuJoCo. The carton is
held by contact friction between the fingertip pads alone. There is no kinematic
attachment, so a poor grasp visibly fails. Every attempt is recorded as a relational
scene and fed to four pipelines that are asked the very same `cause`/`causes_effect`
EQL queries.

- The **relational circuit** is an RSPN fitted on the scenes' relational structure:
  one circuit over the attempt's own attributes and its aggregation count, one
  template over a neighbour. For every query it grounds itself into a circuit over
  exactly the objects the query names and registers that circuit as a
  `CausalCircuit`.
- The **propositional tree** is a joint probability tree (JPT) on the attempt's own
  attributes and the same count, the classic propositional summary of a relational
  example.
- The **unrolled tree** is a JPT on a table that also carries every neighbour's
  attributes under the neighbour's position, one block of columns per position.
- The **scalars-only tree** is a JPT on the attempt's own attributes alone, what a
  flat learner sees without the relational feature extraction.

All three trees are registered as a `CausalCircuit` the same way the relational circuit
is, and a **regression adjustment** on the propositional table, the textbook backdoor
estimator with a logistic regression in place of a circuit, stands beside them as a
reference that is not a circuit at all.

`results.md` holds the comparison: which questions each pipeline answers and what it
answers, how many models each fitted, how long that took, how big the models are, how
well each explains held-out attempts, how fast each answers, how often the pick came up
in the first place, and the studies around the comparison.

## The domain

`domain.py` holds the classes every other module works on.

| class | what it is |
|---|---|
| `ClutterPickScene` | one recorded attempt: the environment it stood in, where the target stood, the grasp's friction coefficient and yaw, whether the target came up and how far it rose, and its `neighbours` |
| `ClutteredObject` | one neighbour as an exchangeable part of the scene: category, position relative to the target, yaw, distance to the target and its `DistanceBand`, which `ClosingAxisSide` of the fingers it stands on, how far the pick shoved it and whether that counts as `disturbed` |
| `ClutterPickSceneAggregations` | the aggregation statistics over the neighbours the relational model derives: `crowding_count`, how many neighbours stand adjacent |
| `ClutterSceneLayout` / `PlacedObject` | the input side of an attempt: every carton's absolute pose, which one is the target, the grasp's friction and yaw |
| `ClutterPickOutcome` | what an attempt did: lift height, lifted or not, every neighbour's displacement; `to_scene` turns a layout and its outcome into a `ClutterPickScene` |
| `NeighbourThresholds` | the distances and angles that turn measured geometry into bands and sides |
| `FrictionLadder` | the friction levels an attempt can be given |

**Why friction.** A friction-held grasp succeeds or fails by the smallest friction
coefficient it still closes under, so friction is the natural causal knob of a pick.
An attempt's friction coefficient is set on the cartons' geoms *and* on
the picking gripper's fingertip pads. MuJoCo gives a contact the larger of its two
geoms' friction, so a slippery carton only slips if the pads closing on it are no
grippier. The ladder sits around the coefficient below which a carton slips out of the
pads. Every level is exactly representable in single precision, because a circuit's
support is read back in single precision, and a level that rounds there would no longer
match the point its own leaves sit on.

**Why the environment is a confounder.** A clutter stands on a *table* or in a *bin*
(`layout_sampler.py`). A bin packs the cartons more tightly *and* holds only the
slippery ones, so in the recorded attempts friction and crowding are correlated without
either causing the other. A question that marks the environment as a `confounder` has
it summed out by backdoor adjustment.

## The demo and the data

| file | what it holds |
|---|---|
| `scene.py` | `MilkClutterWorld`: Tracy, its table, the cartons, a fixed camera and a light, built from a layout |
| `episode.py` | `PickEpisode`: park, pick the target with `PickUpActionMujoco`, measure how far it rose and how far each neighbour moved |
| `demo.py` | watch one attempt in the viewer, or render before/after screenshots headless |
| `layout_sampler.py` | `ClutterLayoutSampler`: a jittered grid of ten cartons in a table or bin environment, a random target, a friction level and a grasp yaw |
| `collect_data.py` | record many attempts headless into a JSON file, rewritten after every attempt |
| `synthetic.py` | a closed-form stand-in for the attempt with the same causal structure, so the pipelines are tested without a simulator |
| `dataset.py` | the attempts on disk, the hosted set of them fetched into the user's cache, the attempts handed to the shared comparison as an `ExampleDataset`, and the lift rates the report opens with |

Tracy is mounted, servoed and simulated with what `semantic_digital_twin` provides for
that: `RobotSpecification.spawn` bolts it into the world (a stationary robot gets a
fixed odom), its table is a robot part of its own (`TracyTable`, a `Table`), the UR10e
arms and Robotiq 2F-85 grippers declare a `PositionServo` actuator on each of their
degrees of freedom when the robot is set up (`robots/ur10e_arm.py`,
`robots/robotiq_85_gripper.py`, the latter also holding the grasp geometry: the
fingertip pads and the driving knuckle), the contact parameters in `world_description/contact.py` are simulator
properties of the shapes, and `MujocoSim`'s stepped mode
(`start_stepped_simulation`/`step_simulation`) realises every position servo as a
MuJoCo actuator. The `tracy_mujoco_addons/` subpackage holds only what is specific to
picking: `live_motion.py`, where a `MotionRunner` runs Giskard's own control loop
against the physically simulated world and closes the gripper on an object with a
Cartesian goal on the distance between its two fingertip frames, and
`pick_and_place_action.py`, the pick and place actions built on it. Giskard drives the simulated robot live: every control
cycle's command becomes the servos' set point through the world state, and the physics
steps in between, so the robot reaches every pose in the physics rather than in the
world's belief only.

## The pipelines

The pipelines, the studies and the report are not this experiment's own: they live in
the shared `experiments.causal_reasoning.comparison` package and work on any relational
example a `RelationalDomain` describes, so that every dataset compared this way is
compared the same way. This package supplies the domain, the data, the questions and
the words.

| file | what it holds |
|---|---|
| `domain.py` | the attempt, its neighbours and their aggregation count, and `attempt_domain()`, the attempt as the shared comparison sees it |
| `queries.py` | the question catalogue, each question one EQL query that reads the same for every pipeline |
| `run_pipeline.py` | the whole comparison end to end: the `Experiment` handed to the shared runner, and the report's prose |

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
Stratifying on two causes at once cannot serve both, because two partitions that share
a value of one cause overlap on it. Each pipeline therefore keeps one plain model for
everything that is not a causal query, such as scoring held-out attempts, and fits one
further model per cause variable the first time it is asked about that cause. A cause
on a neighbour attribute stratifies the neighbour template in the relational pipeline,
or that position's column in the unrolled tree; the other tables have no column for it.

**The questions.** There are three kinds of cause. Each is asked about a clutter of the
recorded size and again about a clutter of another size.

1. *Friction causes lift*, adjusting for the environment: which grasp friction makes
   the target come up. The cause is an attribute of the attempt itself.
2. *Crowding count causes lift*, adjusting for the environment and once more with no
   adjustment at all: how many adjacent neighbours the target can have and still come
   up. The cause is an aggregation over the exchangeable parts, and what adjusting
   changes is a result in itself.
3. *Closing-axis side causes disturbance*, for one neighbour: whether standing where
   the fingers close causes that neighbour to be shoved aside. Cause and effect both
   live on one part.

Neighbours are numbered from 0. The relational circuit grounds itself for whatever
objects a query names, so it can be asked about 4 or 12 neighbours from attempts
recorded with 9, and about any neighbour by index. A flat table ignores the parts a
query merely lists, so the trees answer the questions about 4 or 12 neighbours with the
very numbers they give for 9; and the unrolled tree has columns for 9 neighbours and
nothing else, so it refuses the question about the twelfth.

Every answer is read per region of the cause with the number of training attempts the
region holds, a Wilson interval on the adjusted probability, and two summaries that do
not depend on an argmax over sparse regions: the *trend*, Spearman's rank correlation
between a numeric cause and the adjusted probability over the supported regions, and
the *contrast*, the adjusted probability at the highest supported region minus at the
lowest, with Newcombe's interval. A region holding fewer attempts than the support
threshold is marked and left out of every summary.

**The studies.** Reordering every attempt's neighbours at random, twenty times over,
and asking the neighbour question again shows what an answer about "neighbour 0" is
worth: nothing about a relational circuit can depend on the order, while an unrolled
table's column holds a different neighbour of every attempt afterwards. Following the
relational circuit's answers as grounding draws more and more Monte-Carlo samples shows
how many it takes for them to settle. Fitting on a growing share of the attempts shows
how much data each pipeline needs to explain a whole attempt, neighbours included.
Fitting on synthetic attempts of growing clutter size, drawn from the same layout
sampler with a random outcome in place of the simulator, shows how each pipeline's fit
time, size and query time grow with the number of neighbours. Repeating the comparison
over several random splits is optional (`--splits N`).

## Reading `results.md`

- **How often the pick came up.** The recorded attempts before any model, grouped by
  environment, friction level and crowding. This is the picking efficiency in clutter.
- **Which questions each pipeline can answer.** One row per question. An answer is put
  into words: the most effective setting of the cause, how likely the effect then is,
  and how that compares with the least effective setting. A refusal says why.
- **Trend and contrast.** The two argmax-free summaries of every answer.
- **What adjusting for changes.** The crowding question with and without the
  environment adjusted for, side by side.
- **Fit and likelihood.** Models fitted, training seconds, circuit size, held-out
  coverage (a tree's leaves span only the ranges they saw), and mean log-likelihood on
  the covered attempts and on the attempts every pipeline covers, on three views of an
  attempt.
- **Seconds per question.** The first ask, which includes fitting the cause's own
  model, and the same question asked again with every model fitted.
- **The studies.** The reorderings, the grounding samples, the learning curve and the
  scaling, each with its own table.
- **What the results show.** The findings read off the numbers above.
- **One section per question** with the full interventional table: for every region of
  the cause, its population share, the naive conditional probability of the effect,
  and the backdoor-adjusted interventional probability.

## What the results show

Numbers from `results.md`: 300 recorded attempts with nine neighbours each, one
240/60 split with seed 0, twenty random orderings of the neighbours, a support
threshold of ten training attempts per cause region.

- **The pick comes up 65% of the time, and the environment decides most of it.** On
  the table 87% of the targets are lifted, in the bin 46%; the two lowest friction
  levels hold the carton in 0% and 87% of attempts, everything from 0.375 up in 100%;
  a target with no adjacent neighbour comes up 87% of the time, with four 25%.
- **Every circuit gives the same answer to every question it can pose.** The
  relational circuit, the propositional tree, the unrolled tree and, where it has the
  column, the scalars-only tree agree to the third decimal: a grasp friction of 0.125
  never lifts the target and 0.375 always does (contrast 1.00, [0.81, 1.00]); zero
  adjacent neighbours lift it with probability 0.88 and four with 0.23 (trend -1.00,
  contrast -0.65, [-0.81, -0.37]). Adjusting the crowding question for the environment
  changes no region by more than 0.01, so the crowding count is not a stand-in for
  the environment here. Regression adjustment finds the same best settings but a
  friction contrast of 0.31 in place of 1.00, and an environment-adjusted crowding
  contrast of -0.37 against -0.64 unadjusted: a logistic model flattens a step and
  hands part of the crowding effect to the environment.
- **Only the models that hold the neighbours can be asked about one.** A neighbour
  standing along the fingers' closing axis is disturbed with probability 0.18, across
  it 0.03, read off the relational circuit over the 2,160 neighbours of the training
  attempts (contrast 0.15, [0.12, 0.17]); the unrolled tree reads 0.22 against 0.03
  off the 240 first-listed neighbours (0.19, [0.11, 0.28]). The propositional and
  scalars-only trees and the regression baseline have no column for it. Over twenty
  reorderings the unrolled tree's answer ranges by 0.20 and the relational circuit's
  by nothing.
- **A clutter of another size is a question only the relational circuit can be asked
  in its own right.** The trees answer the questions about 4 and 12 neighbours with
  the numbers they have for 9, since nothing in a flat table tells the sizes apart;
  the relational circuit grounds itself for the queried clutter and, since the
  recorded mechanism does not depend on the number of neighbours, gives the same
  numbers too. The question about the twelfth neighbour of a clutter of twelve has no
  column anywhere: the relational circuit answers it (0.18 against 0.03), every flat
  estimator refuses.
- **Only the relational circuit explains whole attempts.** On the held-out attempts
  both cover, its mean whole-attempt log-likelihood is 154.3 against the unrolled
  tree's 88.9, and it covers 78% of them against 43%; reordering the neighbours drops
  the unrolled tree's coverage to 25% and its likelihood by 33 nats, and leaves the
  relational circuit's unchanged. On the learning curve the relational circuit climbs
  from 102 nats and 30% coverage at a fifth of the data to 135 nats and 76% at four
  fifths, while the unrolled tree stays near 90 nats and reaches 43% coverage.
- **Grounding needs no more than fifty samples**, since the crowding count a query
  leaves open takes six distinct values and grounding integrates over the distinct
  values it has drawn: every answer is the same from fifty samples to 32,000.
- **Cost against the size of the clutter.** From 4 to 25 neighbours the relational
  circuit grows from 1,808 to 7,148 nodes and 2.1 to 5.2 seconds of fitting; the
  unrolled tree from 2,769 to 14,709 nodes and 1.2 to 9.0 seconds, one block of
  columns per position. On the recorded attempts the relational circuit answers in
  4.3 seconds once fitted, the trees in 0.6 to 0.8, the difference being the
  grounding.

## Running it

The `iai_tracy_description` ROS package must be built and sourced for anything that
builds the MuJoCo scene. Headless runs need MuJoCo's EGL backend
(`export MUJOCO_GL=egl`).

```bash
# watch one attempt in the viewer (add --headless --screenshots DIR for images only)
python -m experiments.causal_reasoning.tracy_clutter_picking.demo --seed 3

# record attempts (about a minute each headless; the file is rewritten after every attempt)
python -m experiments.causal_reasoning.tracy_clutter_picking.collect_data \
    data/milk_clutter_attempts.json --attempts 300

# fit, score and question every pipeline and run the studies; needs the experiments
# ORM interface; the report is written out again after every study
python scripts/regenerate_all_orm.py
python -m experiments.causal_reasoning.tracy_clutter_picking.run_pipeline

# the same on attempts you recorded yourself
python -m experiments.causal_reasoning.tracy_clutter_picking.run_pipeline \
    --dataset data/milk_clutter_attempts.json
```

The recorded attempts `results.md` is built from are not in this repository. They are
hosted in [tracy_clutter_picking_data](https://github.com/Narenvasant/tracy_clutter_picking_data)
and fetched into the user's cache the first time `run_pipeline` needs them;
`ExperimentFiles.recorded_attempts` pins the version.

The tests under `test/experiments_test/causal_reasoning_test/test_tracy_clutter_picking` run the pipelines on the
synthetic attempts, so they need no simulator. The ones that build the MuJoCo scene are
skipped where Tracy's description is not installed.
