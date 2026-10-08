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

Every motion goes through `tracy_mujoco_addons/live_motion.py`, which ticks Giskard's own
control loop against the running simulation: each control cycle's command reaches the
joints' servos as their set point and the physics steps one control period before the
next cycle. A motion is therefore reached in the physics, as hard as the servos allow,
rather than planned kinematically and played back.

`tracy_mujoco_addons/pick_and_place_action.py` turns a point to grasp into the goals the
arm reaches for. It builds a grasp frame, x-axis along the approach and y-axis along the
axis the fingers close along, and lets the gripper state where its own approach and
closing axes point, so nothing here needs to know which way Tracy's tool frame faces.
What it does still correct is the gap between the tool frame and where the fingers
actually meet: on Tracy those are about 4.5cm apart, and a goal placed on the tool frame
alone closes the fingers beside the carton rather than on it.

## Where the attempts come from

`dataset.recorded_attempts()` names the attempts recorded in MuJoCo, hosted in their own
repository and fetched on first use. `ClutterPickDataset` saves and loads them.

`synthetic.py` is a closed-form stand-in with the same shape and the same causal
structure: friction lets the fingers hold the target, every adjacent neighbour takes a
share of that hold away, and the environment drives both. It needs no simulator, so the
tests fit on it and run anywhere.
