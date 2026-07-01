---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(soft-robotics)=
# Soft Robotics

This notebook demonstrates how to build and control continuum soft robot models using Piecewise Constant Curvature (PCC) and Cosserat Rod Theory. We utilize Giskardpy's Motion Statecharts for high-level Cartesian control.

## 0. Setup and ROS 2 Initialization
We initialize the ROS 2 node using Giskardpy's `rospy` middleware wrapper, which automatically manages the node's lifecycle.

```{code-cell} ipython3
import numpy as np
import time
from giskardpy.middleware.ros2 import rospy
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.soft_trunk import SoftTrunk, SoftTrunkSection

# Initialize Giskardpy ROS 2 wrapper
rospy.init_node("soft_robotics_examples")
```

## 1. Piecewise Constant Curvature (PCC)
PCC approximates the soft robot's shape as a series of circular arcs.

### 1.1 Build PCC Robot
We build a 3-section uniform robot using the symbolic factory.

```{code-cell} ipython3
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher

world_pcc = World()
sections = [SoftTrunkSection(0.3, 0.02, 10)] * 3

# Build the robot in the world
trunk_pcc = SoftTrunk.build_piecewise_constant_curvature(world_pcc, sections)

# Set up visualizer with automatic TF publishing
viz_pcc = VizMarkerPublisher(_world=world_pcc, node=rospy.node).with_tf_publisher()
world_pcc.notify_state_change()

print("PCC Robot Ready. Set fixed frame to 'piecewise_constant_curvature/base' in RViz.")
```

### 1.2 Controlling PCC with Giskardpy Motion Statecharts
We use Giskardpy's constraint-based task planner to move the soft tip through a sequence of 3D Cartesian coordinates.

```{code-cell} ipython3
from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from semantic_digital_twin.spatial_types import Point3

msc_pcc = MotionStatechart()
msc_pcc.add_node(
    goal_pcc := Sequence(
        [
            CartesianPosition(
                root_link=world_pcc.root,
                tip_link=trunk_pcc.arms[0].tip,
                goal_point=Point3(x=0.4, y=0.0, z=0.5, reference_frame=world_pcc.root),
            ),
            CartesianPosition(
                root_link=world_pcc.root,
                tip_link=trunk_pcc.arms[0].tip,
                goal_point=Point3(x=0.0, y=0.4, z=0.5, reference_frame=world_pcc.root),
            ),
            CartesianPosition(
                root_link=world_pcc.root,
                tip_link=trunk_pcc.arms[0].tip,
                goal_point=Point3(x=0.0, y=0.0, z=0.9, reference_frame=world_pcc.root),
            )
        ]
    )
)
msc_pcc.add_node(EndMotion.when_true(goal_pcc))

# Execute the motion statechart
executor_pcc = Executor(
    context=MotionStatechartContext(world=world_pcc),
    pacer=SimulationPacer(real_time_factor=1),
)
executor_pcc.compile(msc_pcc)
executor_pcc.tick_until_end()
```

## 2. Cosserat Rod
Cosserat models allow for torsion (twist) and stretching (extension), which PCC cannot represent.

### 2.1 Build Cosserat Robot

```{code-cell} ipython3
world_cosserat = World()
sections = [SoftTrunkSection(length=0.3, radius=0.02, resolution=10)] * 3

# Build the robot
trunk_cos = SoftTrunk.build_cosserat(world_cosserat, sections)

# Set up visualizer
viz_cos = VizMarkerPublisher(_world=world_cosserat, node=rospy.node).with_tf_publisher()
world_cosserat.notify_state_change()

print("Cosserat Robot Ready. Set fixed frame to 'cosserat/base' in RViz.")
```

### 2.2 Controlling Cosserat with Giskardpy Motion Statecharts
We run the same Giskardpy sequence on the Cosserat model. We include a distal target to trigger stretching.

```{code-cell} ipython3
msc_cos = MotionStatechart()
msc_cos.add_node(
    goal_cos := Sequence(
        [
            CartesianPosition(
                root_link=world_cosserat.root,
                tip_link=trunk_cos.arms[0].tip,
                goal_point=Point3(x=0.4, y=0.0, z=0.5, reference_frame=world_cosserat.root),
            ),
            CartesianPosition(
                root_link=world_cosserat.root,
                tip_link=trunk_cos.arms[0].tip,
                goal_point=Point3(x=0.0, y=0.7, z=0.4, reference_frame=world_cosserat.root),
            ),
            # Distal target (stretching)
            CartesianPosition(
                root_link=world_cosserat.root,
                tip_link=trunk_cos.arms[0].tip,
                goal_point=Point3(x=0.4, y=0.5, z=1.2, reference_frame=world_cosserat.root),
            )
        ]
    )
)
msc_cos.add_node(EndMotion.when_true(goal_cos))

executor_cos = Executor(
    context=MotionStatechartContext(world=world_cosserat),
    pacer=SimulationPacer(real_time_factor=1),
)
executor_cos.compile(msc_cos)
executor_cos.tick_until_end()
```

## Cleanup

```{code-cell} ipython3
rospy.shutdown()
print("ROS Node shut down.")
```