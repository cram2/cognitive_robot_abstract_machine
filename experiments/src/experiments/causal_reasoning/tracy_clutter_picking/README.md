# Tracy clutter picking: relational circuit against flat-table tree

## What this experiment is for

Relational sum-product networks (RSPNs) are less expressive than most machine-learning
models. The argument this experiment makes is that they are expressive *enough*. A
relational circuit fitted on a robot's own recorded attempts can answer causal questions
about a cluttered pick, including questions a flat model cannot even pose, and the
answers hold up on a task a robot actually executes.

Tracy's left arm picks one milk carton out of a clutter of ten in MuJoCo. The carton is
held by contact friction between the fingertip pads alone. There is no kinematic
attachment, so a poor grasp visibly fails. Every attempt is recorded as a relational
scene and fed to two pipelines that are asked the very same `cause`/`causes_effect`
EQL queries.

- The **relational circuit** is an RSPN fitted on the scenes' relational structure. For
  every query it grounds itself into a circuit over exactly the objects the query
  names and registers that circuit as a `CausalCircuit`.
- The **flat-table tree** is a joint probability tree (JPT) fitted on the same attempts
  flattened into one fixed-width table. It is registered as a `CausalCircuit` in the
  same way.

`results.md` holds the comparison: which questions each pipeline answers and what it
answers, how many models each fitted, how long that took, how big the models are, how
well each explains held-out attempts, how fast each answers, and how often the pick
came up in the first place.

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
| `scene.py` | `MilkClutterWorld`: Tracy, its table, the cartons, a fixed camera and a light, built from a layout and equipped with position servos |
| `episode.py` | `PickEpisode`: park, pick the target with `PickUpActionMujoco`, measure how far it rose and how far each neighbour moved |
| `demo.py` | watch one attempt in the viewer, or render before/after screenshots headless |
| `layout_sampler.py` | `ClutterLayoutSampler`: a jittered grid of ten cartons in a table or bin environment, a random target, a friction level and a grasp yaw |
| `collect_data.py` | record many attempts headless into a JSON file, rewritten after every attempt |
| `synthetic.py` | a closed-form stand-in for the attempt with the same causal structure, so the pipelines are tested without a simulator |
| `dataset.py` | the attempts on disk, the hosted set of them fetched into the user's cache, their train/test split, and success rates grouped by any key |

The MuJoCo stack the demo drives lives in the `tracy_mujoco_addons/` subpackage:
parsing and mounting Tracy, servos, self-collision exclusion, the real-time simulation,
trajectory planning against a scratch copy of the world, the pick and place actions,
and contact tuning. Its own README explains why motions are planned by Giskard but
executed through MuJoCo actuators.

## The pipelines

| file | what it holds |
|---|---|
| `flat_table.py` | `SceneSchema`, how EQL names every attribute, and `FlatTable`, the attempts flattened into one row each with one block of columns per neighbour position |
| `pipelines.py` | `CausalQueryPipeline` and its two implementations, `RelationalPipeline` and `FlatTablePipeline` |
| `queries.py` | the question catalogue, each question one EQL query that reads the same for either pipeline |
| `evaluation.py` | asking every question to every pipeline and recording what came of it |
| `report.py` | rendering the comparison as Markdown |
| `run_pipeline.py` | the whole comparison end to end |

**One model per cause.** Backdoor adjustment needs the circuit to be
support-deterministic over the cause: no sum unit may mix branches that overlap on it.
A fit guarantees that by stratifying its training rows on the cause's exact value.
Stratifying on two causes at once cannot serve both, because two partitions that share
a value of one cause overlap on it. Each pipeline therefore keeps one plain model for
everything that is not a causal query, such as scoring held-out attempts, and fits one
further model per cause variable the first time it is asked about that cause. A cause
on a neighbour attribute stratifies the neighbour template in the relational pipeline,
or the columns of that one neighbour in the flat pipeline.

**The questions.** There are three kinds of cause. Each is asked about a clutter of the
recorded size and again about a clutter of another size.

1. *Friction causes lift*, adjusting for the environment: which grasp friction makes
   the target come up. The cause is an attribute of the attempt itself.
2. *Crowding count causes lift*, adjusting for the environment: how many adjacent
   neighbours the target can have and still come up. The cause is an aggregation over
   the exchangeable parts.
3. *Closing-axis side causes disturbance*, for one neighbour: whether standing where
   the fingers close causes that neighbour to be shoved aside. Cause and effect both
   live on one part.

Neighbours are numbered from 1. The relational circuit grounds itself for whatever
objects a query names, so it answers about 4 or 12 neighbours from attempts recorded
with 9. The flat table has columns for 9 neighbours and nothing else, so it refuses the
questions about 12.

## Reading `results.md`

- **How often the pick came up.** The recorded attempts before any model, grouped by
  environment, friction level and crowding. This is the picking efficiency in clutter.
- **Which questions each pipeline can answer.** One row per question. An answer is put
  into words: the most effective setting of the cause, how likely the effect then is,
  and how that compares with the least effective setting. A refusal says why.
- **Fit and likelihood.** Models fitted, training seconds, circuit size, held-out
  coverage (a tree's leaves span only the ranges they saw), and mean log-likelihood on
  the covered attempts and on the attempts both pipelines cover.
- **Seconds per question.** The first ask, which includes fitting the cause's own
  model, and the same question asked again with every model fitted.
- **What the results show.** The findings read off the numbers above.
- **One section per question** with the full interventional table: for every region of
  the cause, its population share, the naive conditional probability of the effect,
  and the backdoor-adjusted interventional probability.

## Running it

The `iai_tracy_description` ROS package must be built and sourced for anything that
builds the MuJoCo scene. Headless runs need MuJoCo's EGL backend
(`export MUJOCO_GL=egl`).

```bash
# watch one attempt in the viewer (add --headless --screenshots DIR for images only)
python -m experiments.causal_reasoning.tracy_clutter_picking.demo --seed 3

# record attempts (about 15 s each headless; the file is rewritten after every attempt)
python -m experiments.causal_reasoning.tracy_clutter_picking.collect_data \
    data/milk_clutter_attempts.json --attempts 300

# fit, score and question both pipelines; needs the experiments ORM interface
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

The tests under `test/causal_reasoning_test/test_tracy_clutter_picking` run the pipelines on the
synthetic attempts, so they need no simulator. The ones that build the MuJoCo scene are
skipped where Tracy's description is not installed.
