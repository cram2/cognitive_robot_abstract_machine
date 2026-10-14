# Tracy clutter picking

## What this experiment is for

Relational sum-product networks are less expressive than most machine-learning models.
The argument this experiment makes is that they are expressive *enough*: a relational
circuit fitted on a robot's own recorded attempts can answer causal questions about a
cluttered pick, including questions whose cause or effect is one object of the clutter
rather than the attempt as a whole.

Tracy's left arm picks one milk carton out of a clutter of ten in MuJoCo. The carton is
held by contact friction between the fingertip pads alone, with no kinematic attachment,
so a poor grasp visibly fails. Every attempt was recorded as a relational scene, and the
recorded attempts are what this experiment asks its questions of.

There is one model here, the relational circuit. Nothing is compared against anything
else.

## The domain

One attempt is a `ClutterPickScene`: the environment it stood in, where the target
stood, the friction of the grasp, how the gripper was turned, whether the target came up
and how far it rose. Every other object of the clutter is a `ClutteredObject` held in
the scene's `neighbours` field, an exchangeable part: what kind of object it is, where it
stands relative to the target, which band that distance falls in, which side of the
fingers' closing axis it stands on, how far it moved, and whether the attempt shoved it
aside.

`ClutterPickSceneAggregations.crowding_count` counts the neighbours standing adjacent to
the target. It is a statistic over the parts rather than a column of the attempt, which
is what lets a question be asked about the crowding of a clutter whose size was never
fixed in advance.

## The questions

Each question in `do_query.py` marks one cause and one effect in the query itself. The
circuit grounds against a clutter of the size the question asks about, and the effect is
read off every region of the cause twice: once by conditioning alone, once with backdoor
adjustment. The gap between those two is the point.

- **Friction causes the lift**, adjusting for the environment. The environment decides
  both how slippery and how crowded an attempt is, which is what makes it a confounder
  rather than a nuisance.
- **Crowding causes the lift**, adjusting for the environment, and again adjusting for
  nothing, so the difference adjusting makes is visible. The cause is a count over the
  parts.
- **A neighbour's side of the closing axis causes that neighbour to be disturbed.**
  Cause and effect both live on one part, so this is a question about the clutter's
  structure rather than about the attempt as a whole.

## Grouping the fit

A cause has to come out support-deterministic or the registration rejects the model: two
branches of the circuit must never claim the same value of the cause. The fit is
therefore grouped by the cause, and `CauseStratification` records where the cause sits so
that the right circuit is grouped. An attribute of the attempt or a statistic over its
parts groups the class circuit; one neighbour's own attribute groups the template holding
the neighbours instead, since the class circuit has no column for it.

## Running the pick in MuJoCo

`scene.py` builds the world: Tracy bolted to its table, the cartons standing on it, a
camera and a light. `episode.py` runs one attempt against that world simulated
physically, and `collect_data.py` repeats it to record a dataset. `demo.py` runs a single
attempt with the viewer open.

Every attempt is an ordinary coraplex plan performed inside
`PhysicallySimulatedRobot`, which is what ticks Giskard's control loop against the
running simulation: each control cycle's command reaches the joints' servos as their set
point and the physics steps one control period before the next cycle. A motion is
therefore reached in the physics, as hard as the servos allow, rather than planned
kinematically and played back. Nothing about that lives here any more.

`pick_up_action.py` states the attempt as coraplex motions, and `grasping.py`
holds the little that is particular to this grasp. A grasp frame points its x-axis along
the approach and its y-axis along the axis the fingers close along, and the gripper
states where its own approach and closing axes point, so nothing here needs to know
which way Tracy's tool frame faces. What is still corrected here is the gap between the
tool frame and where the fingers actually meet: on Tracy those are about 4.5cm apart,
and a goal placed on the tool frame alone closes the fingers beside the carton rather
than on it. The pads are then closed to the carton's own width rather than all the way,
since closing fully on a carton that is not centred between them wedges it sideways.

## Watching it happen

`demo.py` runs one attempt with MuJoCo's viewer open, so the pick is watched rather than
only measured:

```bash
python -m experiments.causal_reasoning.tracy_clutter_picking.demo
python -m experiments.causal_reasoning.tracy_clutter_picking.demo --seed 3
```

`--seed N` draws a different clutter, `--headless` runs it with no window, and
`--screenshots DIRECTORY` saves what the scene camera sees before and after the pick.
The `iai_tracy_description` ROS package has to be built and sourced, and the viewer
needs a display; offscreen screenshots additionally need an OpenGL context, which a
headless machine may not have.

## Where the attempts come from

`dataset.recorded_attempts()` names the attempts recorded in MuJoCo, hosted in their own
repository and fetched on first use. `ClutterPickDataset` saves and loads them.

`synthetic.py` is a closed-form stand-in with the same shape and the same causal
structure: friction lets the fingers hold the target, every adjacent neighbour takes a
share of that hold away, and the environment drives both. It needs no simulator, so the
tests fit on it and run anywhere.

## Results

Read off the 300 recorded attempts, each a clutter of nine neighbours around the target.
196 of them lifted the target, 65.3%. Every number below is what the fitted circuit
answers for the questions in `do_query.py`: `P` is how much of the population the region
accounts for, *naive* is plain conditioning, *adjusted* is backdoor adjustment.

**Friction causes the lift**, adjusting for the environment. The fingers either hold or
they do not, and the circuit finds the threshold rather than being told it:

| Grasp friction | P | Naive P(lifted) | Adjusted P(lifted) |
|---:|---:|---:|---:|
| 0.125 | 0.2533 | 0.0000 | 0.0000 |
| 0.1875 | 0.2600 | 0.8718 | 0.8947 |
| 0.25 | 0.2633 | 0.7722 | 0.8104 |
| 0.375 | 0.0800 | 1.0000 | 1.0000 |
| 0.5 | 0.0567 | 1.0000 | 1.0000 |
| 0.75 | 0.0867 | 1.0000 | 1.0000 |

**Crowding causes the lift**, adjusting for the environment. The cause here is a count
over the exchangeable parts, not a column of the attempt, which is the point: the
question can be asked of a clutter whose size was never fixed in advance.

| Adjacent neighbours | P | Naive P(lifted) | Adjusted P(lifted) |
|---:|---:|---:|---:|
| 0 | 0.4333 | 0.8692 | 0.9371 |
| 1 | 0.0933 | 0.8571 | 0.8533 |
| 2 | 0.2433 | 0.4521 | 0.4521 |
| 3 | 0.1600 | 0.4167 | 0.4167 |
| 4 | 0.0533 | 0.2500 | 0.2500 |
| 5 | 0.0167 | 0.4000 | 0.4000 |

A target with no adjacent neighbour comes up 87% of the time; with four, 25%. Adjusting
for the environment moves the uncrowded region from 0.8692 to 0.9371, the largest shift
of any region at 0.0679: part of what looked like the crowding's doing was the
environment choosing both how crowded and how slippery an attempt is. Asked again
adjusting for *nothing*, the same question gives naive and adjusted answers that agree
to the last digit, a shift of exactly 0.0000, which is the sanity check that adjustment
is doing the work and not the grounding.

**A neighbour's side of the closing axis causes that neighbour to be disturbed.** Cause
and effect both live on one exchangeable part, so this is a question about the clutter's
structure rather than about the attempt as a whole:

| Where the neighbour stands | P | Naive P(disturbed) | Adjusted P(disturbed) |
|---:|---:|---:|---:|
| across the closing axis | 0.4959 | 0.0547 | 0.0547 |
| along the closing axis | 0.5041 | 0.3278 | 0.3278 |

A neighbour standing along the axis the fingers close along is disturbed six times as
often as one standing across it, which is what the geometry predicts: the fingers sweep
through it on their way around the target.

**What this shows.** A relational circuit fitted on a robot's own recorded attempts
answers causal questions about a cluttered pick, including questions whose cause is a
statistic over the clutter and questions whose cause and effect both sit on one object
of it. **What it does not show** is that these are the true causal effects: backdoor
adjustment is only as sound as the confounder set it is given, and here that set is the
environment the attempt stood in.

To reproduce, fit the questions on the recorded attempts:

```python
from experiments.causal_reasoning.tracy_clutter_picking.dataset import recorded_attempts
from experiments.causal_reasoning.tracy_clutter_picking.do_query import (
    ClutterDoQuery, question_catalogue,
)

attempts = recorded_attempts().load().scenes
for question in question_catalogue(neighbour_count=9):
    answer = ClutterDoQuery(question=question).run(attempts)
```
