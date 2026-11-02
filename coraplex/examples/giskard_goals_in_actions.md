---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Giskard Goals in Actions

coraplex has no motions of its own. What the robot actually does is a Giskard goal, a node of
{mod}`giskardpy.motion_statechart`, and an action is where coraplex puts those goals:

- A {class}`~coraplex.robot_plans.actions.base.GoalAction` creates one goal in
  {meth}`~coraplex.robot_plans.actions.base.GoalAction.create_goal`, and succeeds once the goal is reached.
- An {class}`~coraplex.robot_plans.actions.base.ActionOfSteps` creates its steps in
  {meth}`~coraplex.robot_plans.actions.base.ActionOfSteps.create_steps`; a step is either another action or a goal.

When the plan joins a statechart, every action expands, and the goals it creates become its children in the plan's
statechart, which Giskard runs as a whole. Before creating them, the action fills in what the plan already knows: the
robot from the {class}`~coraplex.plans.context_extensions.RobotAccess` of the context, the link a Cartesian goal is
expressed in as {attr}`~coraplex.plans.context_extensions.RobotAccess.controlled_root` and the tool frame of the arm it
uses. The executor may
still change a goal afterwards with a plan transformation, for example to drive a robot's base with its differential
drive.

An action reaching a goal of your own looks like this:

```python
from dataclasses import dataclass

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.robot_plans.actions.base import GoalAction
from coraplex.testing import setup_world
from cramph.context import StatechartContext
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass(eq=False, repr=False, kw_only=True)
class MoveToolCenterPointAction(GoalAction):
    """
    Moves the tool center point of an arm to a pose.
    """

    arm: Arm
    """
    The arm whose tool center point moves.
    """

    target: Pose
    """
    Where the tool center point should end up.
    """

    def create_goal(self) -> StatechartNode:
        return CartesianPose(
            root_link=self.controlled_root,
            tip_link=self.arm.end_effector.tool_frame,
            goal_pose=self.target,
        )


world = setup_world()
pr2 = PR2.from_world(world)
executor = KinematicallySimulatedPlanExecutor(
    StatechartContext.with_extensions(world, [RobotAccess(pr2)])
)
statechart = Statechart(context=executor.context)
statechart.add_node(
    MoveToolCenterPointAction(
        arm=pr2.left_arm,
        target=Pose.from_xyz_rpy(0.6, 0.3, 0.9, reference_frame=pr2.root),
    )
)
executor.compile(statechart)
executor.execute()
```

The goals themselves are documented with Giskard:

- [Basic motion](https://cram2.github.io/cognitive_robot_abstract_machine/giskardpy/examples/basic_motion.html) runs a
  first goal and shows how goals are combined.
- [Cartesian goals](https://cram2.github.io/cognitive_robot_abstract_machine/giskardpy/examples/cartesian_goals.html)
  moves links to poses, positions and orientations.
- [Motion statecharts](https://cram2.github.io/cognitive_robot_abstract_machine/giskardpy/motion_statecharts.html)
  explains how goals, tasks and monitors are put together into a statechart.
