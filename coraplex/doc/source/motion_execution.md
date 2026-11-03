# Motion Execution in CoraPlex

Motion execution turns the actions of a plan into the motions a robot performs. The actions of a plan expand into
giskard goals and tasks, all of them nodes of one statechart, and an executor runs that statechart, either in
simulation or on a real robot. How the robot is actuated is up to the executor, so the same plan runs on every robot
and with every executor.

## Motions

A motion is a giskard goal or task. When an action expands, it creates the motions it needs from
{mod}`giskardpy.motion_statechart` and puts them below itself in the statechart, for example a
{class}`~giskardpy.motion_statechart.tasks.joint_tasks.JointPositionList` to move a joint, or a
{class}`~giskardpy.motion_statechart.goals.gripper.MoveGripper` to open a gripper to a width.

Before an action creates a goal, it looks up the values that come from coraplex rather than from its own parameters:
an arm's tool frame comes from its
{class}`~semantic_digital_twin.robots.robot_parts.EndEffector`, and the link a Cartesian
goal is expressed relative to comes from
{attr}`~coraplex.plans.context_extensions.RobotAccess.controlled_root`. The mixins in
{mod}`coraplex.robot_plans.mixins` do this for the goals several actions share.

## The Plan Executors

A {class}`~coraplex.plans.executors.PlanExecutor` runs a plan, the top-level nodes of a
{class}`~cramph.statechart.Statechart` built in its context. It builds that context over its world out of the context
extensions it is given, such as the {class}`~coraplex.plans.context_extensions.RobotAccess` of the robot performing
the plan. {meth}`~coraplex.plans.executors.PlanExecutor.compile` adds the collision avoidance goals when collision
avoidance is enabled and an end motion once every plan node succeeded, and
{meth}`~coraplex.plans.executors.PlanExecutor.execute` runs the plan. An executor runs one statechart, so every plan
gets an executor of its own.

## Choosing Between Simulated and Real Execution

Which executor runs a plan decides how it is executed:

```python
from coraplex.plans.context_extensions import RobotAccess
from cramph.context import StatechartContext
from coraplex.plans.executors import (
    RealRobotPlanExecutor,
    KinematicallySimulatedPlanExecutor,
)
from cramph.statechart import Statechart

executor = KinematicallySimulatedPlanExecutor(
    StatechartContext.with_extensions(world, [RobotAccess(robot)])
)
statechart = Statechart(context=executor.context)
statechart.add_node(plan)
executor.compile(statechart)
executor.execute()
```

A plan for the real robot is built and run the same way in a
`RealRobotPlanExecutor(StatechartContext.with_extensions(world, [RobotAccess(robot)]), ros_node=node)`.
Code that is told how to execute at run time takes the executor type, as
{attr}`~coraplex.demonstrations.RobotDemonstration.executor_type` does. A plan does not ask what executes it: each
executor adapts it with its own {attr}`~coraplex.plans.executors.PlanExecutor.plan_adaptations`, see below.

A plan can also be performed against a physically simulated world, which has to be started with
{meth}`~semantic_digital_twin.adapters.multi_sim.MujocoSim.start_stepped_simulation` already:

```python
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor

executor = PhysicallySimulatedPlanExecutor(
    StatechartContext.with_extensions(world, [RobotAccess(robot)]),
    simulation=simulation,
)
```

Kinematically, a joint arrives wherever it was commanded and a picked object is attached to the gripper. Physically, the
commands are set points for the simulation's servos, an object is held by friction alone, and a base cannot be put at a
pose by writing it into the world.

Both executors build on cramph's {class}`~cramph.executor.Executor`. The simulated one is a
{class}`~cramph.executor.StatechartExecutor` that ticks the chart itself, and its
{class}`~giskardpy.motion_control.MotionControl` extension holds the controller configuration:

```python
from giskardpy.motion_control import MotionControl
from giskardpy.qp.qp_controller_config import QPControllerConfig

executor = KinematicallySimulatedPlanExecutor(
    StatechartContext.with_extensions(world, [RobotAccess(robot)]),
    extensions=[MotionControl(QPControllerConfig(target_frequency=50))],
)
```

Collision avoidance is a setting of the executor:

```python
executor = KinematicallySimulatedPlanExecutor(
    StatechartContext.with_extensions(world, [RobotAccess(robot)]),
    collision_avoidance=True,
)
```

## What happens for each executor

- {class}`~coraplex.plans.executors.KinematicallySimulatedPlanExecutor`: the chart is compiled and ticked against the world until
  every plan node succeeded. A plan that stops approaching its goal gives up and raises
  {class}`~coraplex.plans.failures.MotionMadeNoProgress`, naming the tasks that stalled; one that keeps converging is
  never cut off for taking many ticks, only once it exceeds the simulated time limit.
- {class}`~coraplex.plans.executors.PhysicallySimulatedPlanExecutor`: as the kinematically simulated one, but the physics advances one
  control period after every tick, and the robot is given time to come to a standstill once the plan ended.
- {class}`~coraplex.plans.executors.RealRobotPlanExecutor`: the chart is sent to giskard via the `GiskardClientExecutor`, which
  grounds underspecified actions while giskard runs it.
- {meth}`~coraplex.plans.executors.PlanExecutor.prepare` builds the chart without compiling or running it, for
  inspecting a plan as it would run.

## Adapting a plan to its executor

Every executor offers each node of a plan to its
{attr}`~coraplex.plans.executors.PlanExecutor.robot_adaptations` (see below), then to its
{attr}`~coraplex.plans.executors.PlanExecutor.plan_adaptations`, and then to the transformations of the plan's own
{class}`~coraplex.plans.plan_transformation.PlanRewriting`:

- {class}`~coraplex.plans.executors.KinematicallySimulatedPlanExecutor` puts the base at its goal by writing the odometry
  ({class}`~coraplex.robot_plans.plan_transformations.WriteTheOdometryInsteadOfCommandingAPose`), unless the robot
  drives differentially, and keeps a Stretch in front of the container it closes. The trial of a candidate for an
  underspecified action writes the odometry for a differentially driven robot too
  ({class}`~coraplex.robot_plans.plan_transformations.WriteTheOdometryInsteadOfDriving`), see
  {attr}`~coraplex.plans.underspecified.ActionTrial.plan_adaptations`.
- {class}`~coraplex.plans.executors.PhysicallySimulatedPlanExecutor` closes a grasp to the object's width
  ({class}`~coraplex.robot_plans.plan_transformations.CloseFingersToObjectWidth`) and lets contact carry a picked
  object ({class}`~coraplex.robot_plans.plan_transformations.LetContactCarryTheBodies`).
- {class}`~coraplex.plans.executors.RealRobotPlanExecutor` closes a grasp to the object's width and detects with
  RoboKudo ({class}`~coraplex.robot_plans.plan_transformations.DetectWithRoboKudo`).

An executor is given other adaptations as `plan_adaptations=[...]`.

## Robot-Specific Motions

Some robots need a different implementation of a motion. Such a robot gets a
{class}`~coraplex.plans.plan_transformation.PlanTransformation` that replaces the goal an action builds once the
action is expanded, and that applies only to its robot type. Every executor applies these through its
{attr}`~coraplex.plans.executors.PlanExecutor.robot_adaptations`, whichever executor it is:

- {class}`~coraplex.robot_plans.robot_specific_plan_transformations.DriveWithTheDifferentialDrive` drives the base of
  every robot on a differential drive with a
  {class}`~giskardpy.motion_statechart.goals.base_pose_goals.DifferentialDriveBaseGoal`, which arrives within the
  {attr}`~coraplex.robot_plans.actions.core.navigation.DrivesBase.base_pose_threshold` of the action.
- {class}`~coraplex.robot_plans.robot_specific_plan_transformations.MoveStretchToolCenterPointInAStraightLine` moves
  the tool center point of a Stretch to each of its targets along a straight line.
- {class}`~coraplex.robot_plans.robot_specific_plan_transformations.KeepStretchFromTurningWhileMovingItsTool` holds the
  heading of a Stretch's base while its tool moves, so the base only drives along it. A target the arm does not reach
  from where the base faces fails rather than being reached by turning, so the plan has to stand the Stretch facing
  it. Opening and closing a container are left as they are.
- {class}`~coraplex.robot_plans.robot_specific_plan_transformations.KeepTheTorsoUprightWhileUsingATool` keeps Justin's
  torso upright while it moves a tool.

A robot-specific transformation that only one way of executing needs belongs to that executor's
{attr}`~coraplex.plans.executors.PlanExecutor.plan_adaptations` instead, such as
{class}`~coraplex.robot_plans.robot_specific_plan_transformations.KeepStretchInFrontOfTheContainerItCloses` for a
kinematically simulated Stretch. An executor is given other robot adaptations as `robot_adaptations=[...]`; an empty
list turns them off.

A transformation replaces a step of a control structure node with
{meth}`~cramph.composites.ControlStructureNode.replace`; an action running several steps is the control structure node of those
steps, so its steps are replaced, inserted before and after the same way. A
{class}`~coraplex.plans.plan_transformation.ConfigurationTransformation` instead changes an action's settings right
before it expands, so nothing it expanded into has to be replaced.

## Key takeaways

- Actions build giskard goals directly; plans never execute them one at a time.
- A {class}`~coraplex.plans.executors.PlanExecutor` compiles the statechart holding a plan and executes it.
- The executor's type chooses kinematically simulated, physically simulated or real execution; collision avoidance is
  one of its settings.
