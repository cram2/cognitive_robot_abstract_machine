# Motion Statecharts

Motion Statecharts are a core concept in Giskard for composing complex robot motions. A
motion statechart is a plain `Statechart` from
[cramph](https://cram2.github.io/cognitive_robot_abstract_machine/cramph/statecharts.html)
whose nodes can add constraints to the motion problem. Life cycles, transition conditions,
ticking, the composite templates (`Sequence`, `Parallel`, `TryInOrder`, `TryAll`,
`RepeatUntil`, …), generic monitors and plotting are documented in
[cramph](https://cram2.github.io/cognitive_robot_abstract_machine/cramph). This page covers
what giskardpy adds on top.

## The Problem

Traditional robot motion planning often involves a sequence of fixed waypoints or a single, monolithic trajectory. This approach faces several challenges:

- **Complex Sequencing**: Coordinating multiple movements (e.g., "move arm to pre-grasp," then "close gripper," then "lift arm") can become hard to manage as the number of steps increases.
- **Error Handling**: What happens if a collision is detected mid-motion? Or if the gripper fails to close? Handling these contingencies in a flat script often leads to "spaghetti code."
- **Reactivity**: Modern robots need to respond to their environment. A simple trajectory doesn't easily allow for behavior like "move until a certain force is felt" or "stop if a human enters the workspace."

A Motion Statechart solves them by composing a motion out of statechart nodes. Giskard ticks
the statechart once per control cycle, before the QP controller computes the next command.

## Node types

- **Task**: A specific, single-purpose segment of the overall motion. Tasks add constraints to
  the motion problem and observe whether those constraints are currently satisfied. For
  example, a Cartesian position task observes whether the distance to its target is below a
  threshold. Only running tasks influence the motion.
- **Monitor**: A node that observes a condition of the world without adding constraints to the
  motion, for example `PoseReached`, `JointPositionReached` or `Stalled`. A monitor is a
  `MotionStatechartNode` that also says how it succeeds, usually by inheriting `EndedByOwner`
  (see [Who ends a task](#who-ends-a-task)); a class that inherits only
  `MotionStatechartNode` cannot be constructed.
- **Goal**: A composite node that combines tasks and monitors into a reusable motion, for
  example a Cartesian goal or opening a door.
- **EndMotion**: An `EndStatechart` that additionally waits for the robot to come to rest (see
  [Ending the motion](#ending-the-motion)).

## Who ends a task

A task that is ended stops being enforced, and the robot can then be pulled out of the pose
that task had just reached, for example by another task that is still running. That is why
every `Task` is `EndedByOwner`: it never ends itself on reaching its goal, and
whoever runs it writes its success and interrupt conditions. Monitors such as `PoseReached` and
`JointPositionReached` do the same. Nodes whose ending undoes nothing, such as `SetOdometry` and
`SetSeedConfiguration`, are `SucceedsOnObservingTrue`. The ordering templates run those as
they are and end every `EndedByOwner` node once it observes True, see
[Who ends a node](https://cram2.github.io/cognitive_robot_abstract_machine/cramph/statecharts.html#who-ends-a-node).

## Ending the motion

The motion ends once an `EndMotion` node is running and observes True. `EndMotion` observes
True once every active degree of freedom with a velocity limit has come to rest and more than
one second of trajectory time has passed since the start of the whole motion, not since the
`EndMotion` started. In a world without active degrees of freedom it observes True as soon as
it runs. `EndMotion` offers the same factory methods as `EndStatechart`, such as
`EndMotion.when_true(node)`, and must be added at the top level of the statechart. A
`CancelStatechart` node ends the motion by raising its exception at the end of the control
cycle it starts in.

## Example

A plan that moves to a joint goal and then ends once the robot has come to rest. A statechart
is built in the context of the executor that runs it, see
[Running a motion statechart](#running-a-motion-statechart):

```python
from cramph.composites import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList

# goal_state is a JointState of the robot's joints
motion_statechart = Statechart(context=executor.context)
plan = Sequence(nodes=[JointPositionList(goal_state=goal_state)])
motion_statechart.add_node(plan)
motion_statechart.add_node(EndMotion.when_true(plan))
```

## Running a motion statechart

giskardpy does not have its own statechart or executor. Motion control is an extension of
cramph's `StatechartExecutor`, so the same statechart can also hold nodes of other modules,
and their executor extensions can run alongside it. The executor is created first, because
the statechart is built in its context:

```python
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from giskardpy.motion_control import MotionControl

executor = StatechartExecutor(
    context=StatechartContext(world=world),
    extensions=[MotionControl()],
)
# ... build motion_statechart = Statechart(context=executor.context) as above
executor.compile(motion_statechart)
executor.tick_until_end()
```

`MotionControl` gives the nodes a `MotionControlContext` with the QP controller configuration
and the collision variable managers, sets the tick duration to the control time step, and
applies the commands of the QP controller after every tick. Further extensions add what a
run needs on top: `WorldStateTrajectoryRecording` and `DebugExpressionRecording` record the
motion, `RosNodeAccess` gives nodes a ROS2 node, and `DebugExpressionPublishing` shows debug
expressions in RViz. Extensions are called in the order they are listed, so list the
recordings after `MotionControl`.

## Benefits

- **Constraint-Based**: The constraints of all currently active tasks influence the motion, ensuring the robot satisfies all requirements simultaneously (e.g., "reach for the cup while keeping the arm away from the table").
- **Robustness**: Collisions, stalls and failed grasps are handled by monitors and templates in the structure of the motion rather than in the script that runs it.

For practical examples of how to use Motion Statecharts, see the [Basic Motion](examples/basic_motion.md) and [Cartesian Goals](examples/cartesian_goals.md) tutorials.
