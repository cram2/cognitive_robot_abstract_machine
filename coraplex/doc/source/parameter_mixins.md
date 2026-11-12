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
: {mod}`coraplex.robot_plans.mixins` declares them as subclasses of
  {class}`~coraplex.robot_plans.mixins.DesignatorParameterMixin`, sorted by what they describe: what an action acts on,
  where it goes, which robot state it sets, and the values that tune how it carries the action out.

Node parameter mixins
: {mod}`giskardpy.motion_statechart.mixins` declares the parameters that unrelated tasks, goals and monitors share,
  such as the links of the kinematic chain they control, their goal, their threshold and their reference velocity.
  Each kind of goal has a mixin of its own, such as a goal pose, a goal point or a goal joint state.

One parameter or a bundle
: A mixin named `Has…Parameter` declares exactly one parameter and inherits no other parameter mixin. A mixin named
  `Has…Parameters` bundles parameters that always come together: it either declares them itself, as
  {class}`~giskardpy.motion_statechart.mixins.HasPoseThresholdParameters` does, or combines single parameter mixins
  without adding any, as {class}`~giskardpy.motion_statechart.mixins.HasKinematicChainParameters` does.

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
