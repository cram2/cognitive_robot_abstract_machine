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

(physics-simulators)=
# Physics Simulators

This tutorial explains how to run physics simulations for a given world description.
We use **[MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html)** as an example backend, but the same workflow applies to other physics engines supported by MultiSim.

# 1. Simulating a Predefined World

A world can be loaded from a predefined scene description, tutored in the [Loading Worlds](loading-worlds) tutorial,
in this tutorial, we show how to run a physics simulation for such a predefined world description.

## 1.1 Required Imports

We begin by importing the necessary components:

* `MJCFParser` — parses a world description from an MJCF file.
* `MujocoSim` — runs the simulation.
* `SimulatorConstraints` — defines termination conditions.

```{code-cell} ipython3
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from physics_simulators.base_simulator import SimulatorConstraints
from semantic_digital_twin.spatial_computations.raytracer import RayTracer

import os # For path handling
import time # For measuring simulation time
import threading
import math
```

## 1.2 Parsing a World Description

The world can either:

* Be loaded from a predefined MJCF (recommended), or
* Be constructed manually (shown later in this tutorial).

Using predefined scenes is preferred because they are typically validated against the physics engine.

```{note}
Always validate your MJCF scene directly in MuJoCo before running it in MultiSim:
```

Only a physically stable and functional scene can be expected to behave correctly inside MultiSim.

Below is a minimal example scene defined directly as an XML string.

```{code-cell} ipython3
scene_xml_str = """
<mujoco>
<worldbody>
    <body name="robot">
        <geom type="box" pos="0 0 0.5" size="0.2 0.2 0.5" rgba="0.9 0.9 0.9 1"/>
        <body name="left_shoulder" pos="0 0.3 0.9" quat="0.707 0.707 0 0">
            <joint name="left_shoulder_joint" type="hinge" axis="0 0 1"/>
            <geom type="cylinder" size="0.1 0.1 0.3" rgba="0.9 0.1 0.1 1"/>
            <body name="left_arm" pos="0 -0.4 -0.1" quat="0.707 0.707 0 0">
                <joint name="left_arm_joint" type="hinge" axis="0 0 1"/>
                <geom type="box" size="0.1 0.1 0.3" rgba="0.1 0.9 0.1 1"/>
            </body>
        </body>
        <body name="right_shoulder" pos="0 -0.3 0.9" quat="0.707 0.707 0 0">
            <joint name="right_shoulder_joint" type="hinge" axis="0 0 1"/>
            <geom type="cylinder" size="0.1 0.1 0.3" rgba="0.9 0.1 0.1 1"/>
            <body name="right_arm" pos="0 -0.4 0.1" quat="0.707 0.707 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom type="box" size="0.1 0.1 0.3" rgba="0.1 0.9 0.1 1"/>
            </body>
        </body>
    </body>

    <body name="table" pos="0.5 0 0.25">
        <geom type="box" size="0.2 0.2 0.5" rgba="0.5 0.5 0.5 1"/>
    </body>

    <body name="object" pos="0.5 0.0 1">
        <freejoint/>
        <geom type="box" size="0.05 0.05 0.05" rgba="0.1 0.1 0.9 1"/>
    </body>

    <body name="object2" pos="0. 0.0 1.5">
        <freejoint/>
        <geom type="box" size="0.05 0.05 0.05" rgba="0.1 0.9 0.9 1"/>
    </body>
    
</worldbody>
</mujoco>
"""
world = MJCFParser.from_xml_string(scene_xml_str).parse()


rt = RayTracer(world)
rt.update_scene()
rt.scene.show("jupyter")
```

This scene contains:

* A simple robot with two revolute arms
* A static table
* Two dynamic object with free joint

## 1.3 Running the Simulation

```python
headless = (
    os.environ.get("CI", "false").lower() == "true"
)

multi_sim = MujocoSim(
    world=world,
    headless=headless,
    step_size=0.001,
)
time_start = time.time()

constraints = SimulatorConstraints(max_number_of_steps=10000)
multi_sim.start_simulation(constraints=constraints)

if multi_sim.is_running():
    world.connections[3].position = math.pi / 3

# if multi_sim.is_running():
#     multi_sim.stop_simulation()

print(f"Time elapsed: {time.time() - time_start:.2f}s")

rt.update_scene()
rt.scene.show("jupyter")
```

### Common Mistakes to Avoid

**1. Always define termination conditions**

Never run a simulation without explicit termination conditions.
Failing to do so can result in infinite loops and unresponsive processes.
Always specify appropriate stopping criteria using `SimulatorConstraints` or call `multi_sim.stop_simulation()` manually.

**2. Avoid busy waiting**

Do not implement [busy waiting](https://en.wikipedia.org/wiki/Busy_waiting) inside the simulation loop.
Busy waiting can cause excessive CPU usage and degrade overall system responsiveness.
If CPU usage becomes high, the simulation may run significantly slower than expected.

### Performance Considerations

In this example, the scene completes **10,000 simulation steps in under 1.0 second**.

Simulation performance depends primarily on:

* The number of contact points
* The number of collision geometries
* Mesh complexity (vertex count)

### For optimal performance:

* Prefer primitive geometries (boxes, cylinders, spheres)
* Avoid high-resolution meshes unless strictly necessary

# 2. Persistent World Structure Manipulation

During execution, the world structure can be modified dynamically.
Bodies, connections, and degrees of freedom may be added or removed at runtime.
These changes are immediately reflected in the physics simulation.

In the following example, we illustrate how new bodies and connections can be introduced while the simulation is already running.
We start by importing the necessary components for constructing a world programmatically, along with the optional ROS adapters `TFPublisher` and `VizMarkerPublisher`, which stream transforms and visualization markers to RViz in real time.
We then define three helper functions: `spawn_robot_body` creates a static robot base fixed to the world root, `spawn_arm` attaches a single-link arm via a revolute joint, and `spawn_free_box` spawns a free-floating box through a 6-DoF connection.
Detailed implementation of this body-construction style is provided in the [Creating Custom Bodies](creating-custom-bodies) tutorial.


```python tags=["hide-input"]
import os
import threading
import time
import math
import rclpy

from physics_simulators.base_simulator import SimulatorConstraints
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def spawn_free_box(
    spawn_world: World,
    name: str = "box",
    position: tuple = (0.0, 0.0, 1.5),
    scale: Scale = Scale(0.1, 0.1, 0.1),
    color: Color = Color(1.0, 1.0, 0.0, 1.0),
) -> Body:
    """
    Spawn a free-floating box attached to the world root via a 6-DoF connection.
    """
    spawn_body = Body(name=PrefixedName(name))

    box = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            reference_frame=spawn_body,
        ),
        scale=scale,
        color=color,
    )
    spawn_body.collision = ShapeCollection([box], reference_frame=spawn_body)

    with spawn_world.modify_world():
        connection = Connection6DoF.create_with_dofs(
            parent=spawn_world.root,
            child=spawn_body,
            world=spawn_world,
        )
        spawn_world.add_connection(connection)

        # Set the initial world pose of the box via the 6-DoF DoF state.
        connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=position[0],
            y=position[1],
            z=position[2],
            reference_frame=spawn_body,
        )

    return spawn_body


def spawn_robot_body(spawn_world: World) -> Body:
    """
    Spawn the static robot base as a tall box rigidly fixed to the world root.
    """
    spawn_body = Body(name=PrefixedName("robot"))

    # Shape origin is expressed relative to the body frame.
    box_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        z=0.5,
        reference_frame=spawn_body,
    )
    box = Box(
        origin=box_origin,
        scale=Scale(0.4, 0.4, 1.0),
        color=Color(0.9, 0.9, 0.9, 1.0),
    )
    spawn_body.collision = ShapeCollection([box], reference_frame=spawn_body)

    with spawn_world.modify_world():
        spawn_world.add_connection(
            FixedConnection(parent=spawn_world.root, child=spawn_body)
        )

    return spawn_body


def spawn_arm(spawn_world: World, root_body: Body) -> RevoluteConnection:
    """
    Spawn a single-link arm attached to ``root_body`` via a revolute joint about Z.
    """
    spawn_arm_body = Body(name=PrefixedName("arm"))

    # Offset the box shape so the link extends along +Y from the joint origin.
    box_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        y=0.25,
        reference_frame=spawn_arm_body,
    )
    box = Box(
        origin=box_origin,
        scale=Scale(0.1, 0.5, 0.1),
        color=Color(0.9, 0.1, 0.1, 1.0),
    )
    spawn_arm_body.collision = ShapeCollection(
        [box], reference_frame=spawn_arm_body
    )

    dof = DegreeOfFreedom(name=PrefixedName("arm_joint"))

    # Mount the joint on top of the robot base and rotate into a horizontal pose.
    connection_origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0.0,
        pos_y=0.3,
        pos_z=0.9,
        quat_w=0.707,
        quat_x=0.707,
        quat_y=0.0,
        quat_z=0.0,
    )

    with spawn_world.modify_world():
        spawn_world.add_degree_of_freedom(dof)
        arm_connection = RevoluteConnection(
            name=dof.name,
            parent=root_body,
            child=spawn_arm_body,
            axis=Vector3.Z(reference_frame=spawn_arm_body),
            dof_id=dof.id,
            parent_T_connection_expression=connection_origin,
        )
        spawn_world.add_connection(arm_connection)

    return arm_connection

```

Unlike the previous section, we start from an empty `World` rather than from a predefined scene.

```python
world = World()
```

Before launching the simulation we bring up a ROS node and attach `TFPublisher` and `VizMarkerPublisher`, so that transforms and visualization markers are streamed to RViz as the world changes.

```python
rclpy.init()
node = rclpy.create_node("semantic_digital_twin")
spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
spin_thread.start()

tf_publisher = TFPublisher(_world=world, node=node)
viz_publisher = VizMarkerPublisher(_world=world, node=node)
```

`MujocoSim` is then started on this empty world.

```python
headless = os.environ.get("CI", "false").lower() == "true"

multi_sim = MujocoSim(
    world=world,
    headless=headless,
    step_size=0.001,
)
multi_sim.start_simulation()

```

While the simulation is already running, we dynamically spawn the robot base, the revolute arm, and a free-floating box using the helper functions defined above.
Each `modify_world()` block is propagated to the physics engine immediately, without pausing or restarting it.
Finally, the arm joint position is commanded in a loop to exercise the state-synchronization path between the world description and the simulator.

```python
try:
    # Build the scene: fixed robot base, revolute arm, and a free-falling box.
    robot_body = spawn_robot_body(world)
    arm_connection = spawn_arm(spawn_world=world, root_body=robot_body)
    spawn_free_box(world)

    # Slowly rotate the arm joint to exercise state synchronization.
    for i in range(60):
        arm_connection.position = 2 * math.pi / 60 * i
        time.sleep(0.5)
finally:
    multi_sim.stop_simulation()
    rclpy.shutdown()
```

As reflected in the output, new bodies, connections, and degrees of freedom are inserted into the running physics engine without any visible interruption.
The simulated state remains continuous, and subsequent arm-joint commands are forwarded to the simulator on the fly, while RViz reflects the evolving scene through the TF and marker publishers.

### Common Mistakes to Avoid

**1. Do not rely on catching an exact step number inside the simulation loop**
(e.g., `if multi_sim.simulator.current_number_of_steps == 100`)

The simulation runs asynchronously in a very high-frequency loop. Reliably catching an exact step index would require polling that condition at the same high frequency.
This introduces unnecessary overhead, degrades performance, and can significantly slow down the simulation.
Instead, use event-driven logic, time-based conditions, or external synchronization mechanisms when precise triggering is required

**2. Do not spawn objects in collision states**

Spawning an object at a pose that immediately results in collisions can destabilize the physics engine. In severe cases, this may cause numerical instability or cause the simulation to “explode.”
Always validate the target pose before spawning:
* Check for collisions in advance, or
* Pause the simulation before spawning and ensure the new object is placed in a collision-free state.
