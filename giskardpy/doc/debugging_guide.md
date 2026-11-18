# Debugging Guide

**Warning:** You should only use these tools actively while debugging and preferably only in simulation, because it slows down the control loop.

When a motion does the wrong thing — or the controller fails to produce a command at all — you
need to see what Giskard is computing internally. This guide covers two tools, each answering
a different question:

- **Debug expressions**, drawn live in RViz, answer *"where is this point/vector/pose right
  now, and where does the goal want it?"*
- The **`QuadraticProgramDebugger`** answers *"why did the QP behave this way?"* — it labels the
  raw solver arrays with constraint and degree-of-freedom names so you can read bounds, weights,
  and constraint violations by name.

## Debug expressions

A `DebugExpression`
(`giskardpy.motion_statechart.graph_node.DebugExpression`) wraps three things:

- a `name`, used as the label in every debugging tool,
- an `expression` — either a `Scalar` or a spatial type (`CartesianPoint3Dimensional`, `Vector3Dimensional`, `Quaternion`,
  `RotationMatrix`, `HomogeneousTransformationMatrix3Dimensional`, or `Pose3Dimensional`), and
- a `color`, used when the expression is rendered.

A node registers a debug expression by appending it to `artifacts.debug_expressions` inside its
`build_artifacts()` method:

```python
from semantic_digital_twin.world_description.geometry import Color
from giskardpy.motion_statechart.graph_node import DebugExpression

artifacts.debug_expressions.append(
    DebugExpression(
        name=f"{self.name}/current_normal",
        expression=root_V_tip_normal,
        color=Color.from_rgba(red=1, green=0, blue=0, opacity=1),
    )
)
```

See `tasks/align_planes.py` for a worked example. For Cartesian tasks there is a convenience
helper, `CartesianTask.add_goal_and_current_debug_expressions(artifacts=..., goal=..., current=...)`
(`tasks/cartesian_tasks.py`), which registers `<task name>/goal` (green) and
`<task name>/current` (red) so the two can be told apart in RViz.

```{note}
Only spatial types are drawn in RViz. A `Scalar` debug expression is not shown anywhere.
```

## Visualizing debug expressions in RViz

The `DebugExpressionPublishing` executor extension
(`giskardpy.motion_statechart.debug_expression_publisher`) publishes the spatial debug
expressions of every *running* node as RViz markers. A node's markers appear when it starts
and disappear when it stops, and all markers are deleted once the run ends.

### With Giskard

Giskard already has the extension, with markers switched off. Switch them on for a single goal
through the `DebugOutput` you pass to `GiskardClientExecutor.execute`:

```python
from giskardpy.middleware.ros2.motion_goal import DebugOutput

client.execute(debug_output=DebugOutput(publishes_expression_markers=True))
```

### With your own executor

Add the extension to the executor; it needs the ROS2 node it publishes on:

```python
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.debug_expression_publisher import (
    DebugExpressionPublishing,
)
from giskardpy.motion_statechart.ros_context import RosNodeAccess

executor = StatechartExecutor(
    context=StatechartContext(world=world),
    extensions=[
        RosNodeAccess(ros_node=node),
        MotionControl(),
        DebugExpressionPublishing(ros_node=node),
    ],
)
executor.compile(statechart=motion_statechart)
```

On `compile()`, the extension compiles the spatial debug expressions of every node together
with the statechart. While the statechart runs, it evaluates them `publish_frequency` times per
second of control time (10 by default); rendering and publishing the markers happens on a
thread of its own. `publishes_markers` switches all of this off for the statecharts compiled
afterwards.

Each expression's `name` becomes the marker label, and its namespace is the name followed by
the index of the node that registered it, so individual expressions can be toggled on and off
in RViz's display tree. `Vector3Dimensional` expressions are anchored to their `visualisation_frame` so
the arrow is drawn at the right body.

This is the right tool for spatial reasoning: comparing a goal pose against the current pose,
checking which way a normal points, or confirming a point is where you expect it in the world.

Each spatial type renders as a different marker, and the `<task name>/goal` (green) /
`<task name>/current` (red) convention makes the comparison easy to read:

```{figure} img/pose_marker.png
---
width: 600px
---
A `Pose3Dimensional` is drawn as a coordinate frame. Here `CartesianPose/goal` and `CartesianPose/current`
show the target and actual end-effector pose side by side.
```

```{figure} img/cartesian_position_marker.png
---
width: 600px
---
A `CartesianPoint3Dimensional` is drawn as a sphere — `CartesianPosition/goal` (green) and
`CartesianPosition/current` (red).
```

```{figure} img/pointing_marker.png
---
width: 500px
---
A `Vector3Dimensional` is drawn as an arrow anchored to its `visualisation_frame`. Here the green
`Pointing/goal` and red `Pointing/current` arrows show where the robot should point versus
where it currently points.
```

## Inspecting the QP with `QuadraticProgramDebugger`

When implementing new motion statechart nodes, the controller may produce an unintended command, runs into limits, or the solve fails.
To debug this, it may help to look directly into the quadratic program (QP) itself. 
The QP is just large, anonymous numeric arrays;
the `QuadraticProgramDebugger` (`giskardpy.qp.qp_debugger`) re-labels those arrays with the corresponding
constraint and degree-of-freedom names and exposes them as `pandas.DataFrame`s, so you can read
the problem by name.

Every `QPController` builds one automatically as `controller.debugger` (with no solution yet).
To inspect it:

 - you should run the program in debug mode 
 - place a break point in the last line of `QPController.compute_command(...)`
 - add a watch saying `self.debugger.update(solution)` (when the solve fails, use `update(None)`)
 - look at `self.debugger` in the debugger

### Reading the output

The debugger produces five labeled frames:

- **`direct_limits`** — one row per free variable, with its lower/upper box bounds, the solution
  value, and its quadratic and linear weight. Use it to spot a variable pinned at a bound, a
  zero or oversized weight, or a solution that landed outside its box. Constraints with a weight of `0` are inactive and will have `nan` values.
- **`equality_constraints`** — columns for the constraint value (matrix × decision variables,
  excluding slack), the `slack` (`bounds − value`, i.e. how much the constraint is violated),
  and the `bounds`. A large `slack` pinpoints the violated equality constraint by name.
- **`inequality_constraints`** — same as the previous, but for inequality constraints.
- **`equality_matrix` / `inequality_matrix`** — the labeled constraint matrices (rows are
  constraint names, columns are degree-of-freedom names). It's essentially a Jacobian matrix, so look at how degrees of freedom influence your constraints. `0` means no influence, a positive value means increasing the value will also increase constraints' expression, a negative value means decreasing the value will increase the constraints' expression.
