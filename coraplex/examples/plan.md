---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---
# Introduction to Plans
A plan in CoraPlex is what the robot does: the nodes a statechart holds at its top level, usually cramph composites
such as `Sequence`, `Parallel` or `TryInOrder` holding actions. Plans are built from the constructs introduced in the
[Plan Language](https://cram2.github.io/cognitive_robot_abstract_machine/cramph/examples/plan_language.html) of cramph. An executor then compiles the statechart and executes it, in a simulated environment
or on a real robot.

We will now go through a simple example of how to create and execute a plan.

# Setup a World

```python
from coraplex.plans.context_extensions import RobotAccess
from cramph.context import StatechartContext
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.testing import setup_world
from semantic_digital_twin.robots.pr2 import PR2

world = setup_world()

pr2 = PR2.from_world(world)
```

## The Executor
An executor runs plans in a world. What a plan's nodes read from their context, such as the robot performing them, is
given to it as context extensions; the executor builds the statechart context out of them. A `KinematicallySimulatedPlanExecutor`
ticks the plan in the world itself, a `RealRobotPlanExecutor` sends it to Giskard driving the real robot.

```python
executor = KinematicallySimulatedPlanExecutor(
    StatechartContext.with_extensions(world, [RobotAccess(pr2)])
)
```

## Example Plan
The plan is built in a statechart of the executor's context.

```python
from coraplex.robot_plans import *
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from cramph.composites import Sequence
from cramph.statechart import Statechart
from semantic_digital_twin.spatial_types.spatial_types import Pose

navigate = NavigateAction(target_location=Pose.from_xyz_quaternion(1, 1, 0, reference_frame=world.root))
park = ParkArmsAction(arms=pr2.all_arms)

plan = Sequence(nodes=[navigate, park])
statechart = Statechart(context=executor.context)
statechart.add_node(plan)
```

This creates a plan with a `Sequence` at the top level of the statechart and the two actions as its children.

You can open an interactive, real-time visualization of the statechart the plan runs in using its `visualize` method.

```python
statechart.visualize()
```

## Plan Execution
`compile` prepares the statechart, adding what the plan runs with, and compiles it; `execute` runs it until every
top-level node of the plan succeeded. Underspecified actions are grounded while the plan runs.

```python
executor.compile(statechart)
executor.execute()
```

This will execute the plan in a simulated environment.

### Collision Avoidance
An executor accepts a `collision_avoidance` flag. When set to `True`, collision avoidance goals are added to the
statechart, keeping the robot from colliding with the rest of the world while the motions run.

```python
executor = KinematicallySimulatedPlanExecutor(
    StatechartContext.with_extensions(world, [RobotAccess(pr2)]),
    collision_avoidance=True,
)
```

An executor runs one statechart, so every plan gets an executor of its own.

## Inspecting a Plan

Every node of an executed plan reports how its run went:

* life_cycle_state: Whether the node has not started, is running or paused, or succeeded, failed or was interrupted
* start_time/end_time: When the node started and ended

```python
print(plan.life_cycle_state)
print(plan.children[0].life_cycle_state)
print(plan.start_time)
print(plan.end_time)
```
