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

# Plan Language

The CoraPlex plan language is a way to structure the execution of your plan. In generally the plan language allows to
execute designators either sequential or in parallel. Furthermore, exceptions that occur during execution of a plan with
the plan language do not interrupt the execution instead they are caught and handed to the failure handling module.
The language create a tree structure of the plan where the language expressions one kine of nodes among designators
these nodes store additional information about the execution of the plan including the exceptions that occurred and the 
status of execution.

There are 4 language expressions:

| Name             | Description                                                                                                                                                                                                                                                                                | 
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Sequential**   | Executes the designators one after another, if one of the designators raises an exception the execution is aborted and the state FAILED will be returned.                                                                                                                                  |
| **Try In Order** | Executes the designators one after another, if one designator raises an exception the exception is caught and saved but the execution is not interrupted and the other designators are executed. Returns the state SUCCEDED if at least one designator can be executed without exception. |
| **Repeat**       | Repeat the previous language expression a number of time. Has to be used with a language expression and an integer.                                                                                                                                                                        | 
| **Parallel**     | Executes all designators in parallel. For each designator there will be a new thread created and the designator is executed in this thread. If one of the designators raises an exception the returned state will be FAILED.                                                               |
| **Try All**      | Executes all designators in parallel with a designated thread for each designator. Returns the state SUCCEDED if at least one designator can be executed without an exception                                                                                                              |
| **Monitor**      | Monitors the execution of the attached langauge expression, will interrupt or pause it once a given condition is fulfilled.                                                                                                                                                                 | 

The Sequential plan is the only one which aborts the execution once an error is raised.

When using the plan language a tree structure of the plan is created where the language expressions are nodes and
designators are leafs. 

# Setup the World

If you are performing a plan with a simulated robot, you need a BulletWorld.

```python
from coraplex.execution_environment import simulated_robot
from coraplex.testing import setup_world
from coraplex.datastructures.dataclasses import Context
from semantic_digital_twin.robots.pr2 import PR2

world = setup_world()
pr2 = PR2.from_world(world)

context = Context(world, pr2)
```


## Sequential

This language expression allows to execute designators one after another, if one of the designators raises an exception
the execution will be aborted and the state FAILED will be returned.

We will start with a simple example that uses an action designator for moving the robot and parking its arms.

```python
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.spatial_types import Pose

navigate = NavigateAction(Pose.from_xyz_rpy(1, 1, 0, reference_frame=world.root))
park = ParkArmsAction(Arms.BOTH)

plan = sequential([navigate, park], context=context).plan
```

With this simple plan created we can inspect it and open an interactive visualization of the tree structure.

```python
plan.visualize()
```

As you can see there is the root node which is the language expression and then there are the leafs which are the
designators. When executing this plan the Sequential node will try to execute the NavigateAction and if that is finished
without any error the ParkArmsAction will be executed.

The plan can be executed by wrapping it inside a ```with simulated_robot``` environment and calling perform on the
plan.

```python
with simulated_robot:
    plan.perform()
```

## Try In Order

Try in order is similar to Sequential, it also executes all designators one after another but the key difference is that
an exception in one of the designators does not terminate the whole execution. Furthermore, the state FAILED will only
be returned if all designator executions raise an error.

Besides the described difference in behaviour this language expression can be used in the same way as Sequential.

```python
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import try_in_order
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.spatial_types import Pose

navigate = NavigateAction(Pose.from_xyz_rpy(1, 1, 0, reference_frame=world.root))
park = ParkArmsAction(Arms.BOTH)

plan = try_in_order([navigate, park], context=context).plan

with simulated_robot:
    plan.perform()
```

## Parallel

Parallel executes all designator at once in dedicated threads. The execution of other designators is not aborted when a
exception is raised, this is the case since threads can not be killed from the outside and this would also cause
unforeseen problems. The state returned will be SUCCEDED if all designators could be executed without an exception raised
in any other case FAILED will be returned.

Using the parallel expressions works like Sequential and TryInOrder.

```python
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import parallel
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.spatial_types import Pose

navigate = NavigateAction(Pose.from_xyz_rpy(1, 1, 0, reference_frame=world.root))
park = ParkArmsAction(Arms.BOTH)

plan = parallel([navigate, park], context=context).plan

with simulated_robot:
    plan.perform()
```

## Try All

TryAll is to Parallel what TryInOrder is to Sequential, meaning TryAll will also execute all designators in parallel but
will return SUCCEEDED if at least one designator is executed without raising an exception.

TryAll can be used like any other language expression.

```python
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import try_all
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.spatial_types import Pose

navigate = NavigateAction(Pose.from_xyz_rpy(1, 1, 0, reference_frame=world.root))
park = ParkArmsAction(Arms.BOTH)

plan = try_all([navigate, park], context=context).plan

with simulated_robot:
    plan.perform()
```

## Combination of Expressions

You can also combine different language expressions to further structure your plans. For example, you can nest a
Sequential expression inside a Parallel one by passing the result of one factory as a child of another.

```python
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import parallel, sequential
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.spatial_types import Pose

navigate = NavigateAction(Pose.from_xyz_rpy(1, 1, 0, reference_frame=world.root))
park = ParkArmsAction(Arms.BOTH)
move_torso = MoveTorsoAction(TorsoState.HIGH)

plan = parallel([navigate, sequential([park, move_torso])], context=context).plan

with simulated_robot:
    plan.perform()
```

In this case 'park' and 'move_torso' form a Sequential expression, and that Sequential expression forms a Parallel
expression together with 'navigate'.

## Code Objects

You can not only use designators in the plan language but also python code. For this there is the {func}`~coraplex.plans.factories.code`
factory which takes a callable and wraps it in a {class}`~coraplex.language.CodeNode`. This allows you to execute
arbitrary code in a plan.

The callable can either be a lambda expression or, for more complex code, a function.

Although this expression is more intended for debugging and testing purposes since the code can not really interact with 
other parts of the plan.

```python
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import code, parallel
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction


def code_test():
    print("-" * 20)
    print("Code function")


park = ParkArmsAction(Arms.BOTH)
code_lambda = code(lambda: print("This is from the code object"), context=context)
code_func = code(code_test, context=context)

plan = parallel([park, code_lambda, code_func], context=context).plan

with simulated_robot:
    plan.perform()
```

## Exception Handling

If an exception is raised during the execution of a designator when it is used in a language expression the exception
is caught and saved on that designator's node. Sequential is the only expression that stops the rest of the plan once
this happens; TryInOrder and TryAll continue with their remaining children and only fail the whole expression if all
of them failed. Parallel continues running its remaining children as well, but re-raises the failure once every child
has finished.

The language will only catch exceptions that are of type {class}`~coraplex.plans.failures.PlanFailure` meaning errors that are defined in
plan_failures.py in CoraPlex. This also means normal Python errors, such as KeyError, will interrupt the execution of your
designators.

We will see how exceptions are handled at a simple example using TryAll, so that a failing designator does not fail
the whole plan.

```python
from coraplex.plans.factories import code, try_all
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.plans.failures import PlanFailure
from semantic_digital_twin.spatial_types import Pose


def code_test():
    raise PlanFailure


navigate = NavigateAction(Pose.from_xyz_rpy(1, 1, 0, reference_frame=world.root))
code_func = code(code_test, context=context)

plan = try_all([navigate, code_func], context=context).plan

with simulated_robot:
    plan.perform()

print(plan.root.status)
print(code_func.reason)
```

## Repeat

Repeat attempts a language expression again whenever an attempt fails, until it either succeeds or runs out of
attempts. Running out of attempts raises {class}`~coraplex.plans.failures.RepetitionsExhausted`.

You can see an example of how to use Repeat below.

```python
from coraplex.plans.factories import repeat
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from semantic_digital_twin.datastructures.definitions import TorsoState

move_torso_up = MoveTorsoAction(TorsoState.HIGH)
move_torso_down = MoveTorsoAction(TorsoState.LOW)

plan = repeat([move_torso_up, move_torso_down], maximum_repetitions=3, context=context).plan

with simulated_robot:
    plan.perform()
```

## Monitors

A monitor lets you attach a condition to a language expression that is evaluated by the control loop alongside it,
so it can act on the expression's children while they are running. The condition is a
{class}`~giskardpy.motion_statechart.graph_node.MotionStatechartNode`, for instance a monitor that turns True after a
fixed amount of simulation time.

There are three factories for this:

* {func}`~coraplex.plans.factories.cancel_when` stops the children once the monitor observes True and gives up on
  the plan with a {class}`~coraplex.plans.failures.PlanCancelled`, instead of leaving the rest of the plan waiting for
  a subtree that will not finish.
* {func}`~coraplex.plans.factories.pause_while` holds the children for as long as the monitor observes True.
* {func}`~coraplex.plans.factories.pause_until` holds the children until the monitor observes True, then lets them run.

For the example we will use the previous example with the robot moving up and down, and stop it after 2 seconds of
simulation time. Since cancelling gives up on the plan, performing it raises {class}`~coraplex.plans.failures.PlanCancelled`.

```python
from coraplex.plans.factories import cancel_when, repeat
from coraplex.plans.failures import PlanCancelled
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from giskardpy.motion_statechart.monitors.payload_monitors import CountSimulationTimeSeconds
from semantic_digital_twin.datastructures.definitions import TorsoState

move_torso_up = MoveTorsoAction(TorsoState.HIGH)
move_torso_down = MoveTorsoAction(TorsoState.LOW)

plan = cancel_when(
    [repeat([move_torso_up, move_torso_down], maximum_repetitions=3)],
    monitor=CountSimulationTimeSeconds(seconds=2),
    context=context,
).plan

try:
    with simulated_robot:
        plan.perform()
except PlanCancelled as cancelled:
    print(cancelled)
```

{func}`~coraplex.plans.factories.pause_until` can be used the same way to launch a subtree in a paused state that is
only released once the monitor's condition is fulfilled.

```python
from coraplex.plans.factories import pause_until, repeat
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from giskardpy.motion_statechart.monitors.payload_monitors import CountSimulationTimeSeconds
from semantic_digital_twin.datastructures.definitions import TorsoState

move_torso_up = MoveTorsoAction(TorsoState.HIGH)
move_torso_down = MoveTorsoAction(TorsoState.LOW)

plan = pause_until(
    [repeat([move_torso_up, move_torso_down], maximum_repetitions=3)],
    monitor=CountSimulationTimeSeconds(seconds=2),
    context=context,
).plan

with simulated_robot:
    plan.perform()
```
This will hold the wrapped plan for the first 2 seconds of simulation time before letting it run.
