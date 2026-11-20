# Parameter Mixins

Every parameter of an action and of a Giskard motion statechart node is declared once, in a mixin of its own. A
class takes a parameter by inheriting its mixin, so the parameter has the same name, type, default and
documentation in every class that takes it.

```{contents}
:local:
:depth: 1
```

## Where the mixins live

Action parameter mixins
: {mod}`coraplex.robot_plans.mixins` declares them, sorted by what they describe: what an action acts on, where it
  goes, which robot state it sets, and the values that tune how it carries the action out.

Node parameter mixins
: {mod}`giskardpy.motion_statechart.mixins` declares the parameters that unrelated tasks, goals and monitors share,
  such as the kinematic chain they control, their goal, their threshold and their reference velocity.
  Each kind of goal has a mixin of its own, such as a goal pose, a goal point or a goal joint state.

One parameter per mixin
: Every parameter mixin derives from {class}`~krrood.patterns.mixin.ParameterMixin`. A mixin named
  `Has{Parameter}Parameter` declares exactly one parameter, named by the mixin in snake case, and inherits no other
  parameter mixin: {class}`~coraplex.robot_plans.mixins.HasPlanarGoalPoseParameter` declares `planar_goal_pose`.
  So the name of a parameter tells which mixin to inherit, and the name of a mixin tells which parameter it adds.

One meaning per mixin
: A class may give a parameter another default, but never another meaning or unit. A parameter in other units has a
  mixin of its own: a task with an error in radians takes its cruise velocity from
  {class}`~giskardpy.motion_statechart.mixins.HasCruiseAngularVelocityParameter`, not from
  {class}`~giskardpy.motion_statechart.mixins.HasCruiseVelocityParameter` with a default in radians per second.

Values that come together
: Parameters a class always takes together are one parameter whose type holds them. A node moving the tip of a chain
  relative to its root takes a
  {class}`~semantic_digital_twin.world_description.kinematic_chain.KinematicChain` through
  {class}`~giskardpy.motion_statechart.mixins.HasKinematicChainParameter`, rather than a root link and a tip link.

Mixins shared by both
: An action that passes a parameter on unchanged to the node it builds takes that node's mixin, rather than
  declaring a parameter of its own. {class}`~coraplex.robot_plans.actions.core.pick_up.PickUpAction`, for example,
  passes the finger stall tolerance on to the node that closes the gripper.

{mod}`coraplex.robot_plans.mixins` also defines behaviours that several actions share, such as
{class}`~coraplex.robot_plans.mixins.MovesToolCenterPoint`. A behaviour can take a parameter mixin and declare its
parameters again with defaults that suit the actions inheriting it.

## Who takes what

The map below is read from the code each time this documentation is built. Coraplex actions are on the left,
parameter mixins in the middle and Giskard nodes on the right. Hover over or tap a name to trace its connections and
to see the parameters it declares or takes.

```{parameter-mixin-map}
```

## Adding a parameter

When an action or a node needs a parameter, look for a mixin above that already declares it, and inherit that mixin.
Declare a new mixin only for a parameter no mixin covers yet, in the section of
{mod}`coraplex.robot_plans.mixins` or {mod}`giskardpy.motion_statechart.mixins` it belongs to; the map picks it up
the next time the documentation is built.
