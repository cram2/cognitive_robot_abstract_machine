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
import os # For path handling
import time # For measuring simulation time
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
if __name__ == "__main__":
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

        <body name="object" pos="0.5 0 1.0">
            <freejoint/>
            <geom type="box" size="0.05 0.05 0.1" rgba="0.1 0.1 0.9 1"/>
        </body>
    </worldbody>
</mujoco>
"""
    world = MJCFParser.from_xml_string(scene_xml_str).parse()
```

This scene contains:

* A simple robot with two revolute arms
* A static table
* A dynamic object with a free joint

## 1.3 Running the Simulation

```{code-cell} ipython3
    headless = (
        os.environ.get("CI", "false").lower() == "true"
    )

    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=0.001,
    )

    constraints = SimulatorConstraints(max_number_of_steps=10000)

    multi_sim.start_simulation(constraints=constraints)

    time_start = time.time()

    while multi_sim.is_running():
        time.sleep(0.1) # Sleep to avoid busy waiting
        print(
            f"Current number of steps: "
            f"{multi_sim.simulator.current_number_of_steps}"
        )

    print(f"Time elapsed: {time.time() - time_start:.2f}s")

    multi_sim.stop_simulation()
```

### Common Mistakes to Avoid

**1. Always define termination conditions**

Never run a simulation without explicit termination conditions.
Failing to do so can result in infinite loops and unresponsive processes.
Always specify appropriate stopping criteria using `SimulatorConstraints`.

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
We start by importing the necessary components for constructing a world programmatically and defining two helper functions: one for spawning a robot body and another for creating shoulder bodies.
Detailed implementation of these functions is provided in the [Creating Custom Bodies](creating-custom-bodies) tutorial.

```{code-cell} ipython3
:tags: [hide-input]

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection, RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Box, Scale, Color, Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

def spawn_robot_body(spawn_world: World) -> Body:
    spawn_body = Body(name=PrefixedName("robot"))

    box_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0, y=0, z=0.5,
        roll=0, pitch=0, yaw=0,
        reference_frame=spawn_body
    )

    box = Box(
        origin=box_origin,
        scale=Scale(0.4, 0.4, 1.0),
        color=Color(0.9, 0.9, 0.9, 1.0),
    )

    spawn_body.collision = ShapeCollection(
        [box], reference_frame=spawn_body
    )

    with spawn_world.modify_world():
        spawn_world.add_connection(
            FixedConnection(
                parent=spawn_world.root,
                child=spawn_body
            )
        )

    return spawn_body
   
def spawn_shoulder_bodies(
    spawn_world: World,
    root_body: Body
) -> tuple[Body, Body]:

    # Left shoulder
    spawn_left_shoulder_body = Body(
        name=PrefixedName("left_shoulder")
    )

    cylinder = Cylinder(
        width=0.2,
        height=0.1,
        color=Color(0.9, 0.1, 0.1, 1.0),
    )

    spawn_left_shoulder_body.collision = ShapeCollection(
        [cylinder],
        reference_frame=spawn_left_shoulder_body
    )

    dof = DegreeOfFreedom(
        name=PrefixedName("left_shoulder_joint")
    )

    left_origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0, pos_y=0.3, pos_z=0.9,
        quat_w=0.707, quat_x=0.707,
        quat_y=0, quat_z=0
    )

    with spawn_world.modify_world():
        spawn_world.add_degree_of_freedom(dof)
        spawn_world.add_connection(
            RevoluteConnection(
                name=dof.name,
                parent=root_body,
                child=spawn_left_shoulder_body,
                axis=Vector3.Z(reference_frame=spawn_left_shoulder_body),
                dof_id=dof.id,
                parent_T_connection_expression=left_origin,
            )
        )

    # Right shoulder
    spawn_right_shoulder_body = Body(
        name=PrefixedName("right_shoulder")
    )

    spawn_right_shoulder_body.collision = ShapeCollection(
        [cylinder],
        reference_frame=spawn_right_shoulder_body
    )

    dof = DegreeOfFreedom(
        name=PrefixedName("right_shoulder_joint")
    )

    right_origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0, pos_y=-0.3, pos_z=0.9,
        quat_w=0.707, quat_x=0.707,
        quat_y=0, quat_z=0
    )

    with spawn_world.modify_world():
        spawn_world.add_degree_of_freedom(dof)
        spawn_world.add_connection(
            RevoluteConnection(
                name=dof.name,
                parent=root_body,
                child=spawn_right_shoulder_body,
                axis=Vector3.Z(reference_frame=spawn_right_shoulder_body),
                dof_id=dof.id,
                parent_T_connection_expression=right_origin,
            )
        )

    return spawn_left_shoulder_body, spawn_right_shoulder_body
```

As before, we start by running the simulation using a predefined world description.
However, after 100 simulation steps, we dynamically introduce additional bodies and connections. 
The simulation proceeds without interruption, and the newly added elements are incorporated into the physics engine immediately, becoming fully active in the ongoing simulation.

```{code-cell} ipython3
if __name__ == "__main__":
    scene_xml_str = """
<mujoco>
</mujoco>
"""
    world = MJCFParser.from_xml_string(scene_xml_str).parse()
    headless = (
        os.environ.get("CI", "false").lower() == "true"
    )

    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=0.001,
    )

    constraints = SimulatorConstraints(max_number_of_steps=10000)

    multi_sim.start_simulation(constraints=constraints)

    time_start = time.time()

    spawned = False
    while multi_sim.is_running():
        if multi_sim.simulator.current_number_of_steps >= 100 and not spawned:
            spawned = True
            time_spawn_start = time.time()
            robot_body = spawn_robot_body(world)
            spawn_shoulder_bodies(
                spawn_world=world,
                root_body=robot_body
            )
            print(
                f"Time to spawn bodies: "
                f"{time.time() - time_spawn_start:.2f}s"
            )
        time.sleep(0.1)

    print(f"Time elapsed: {time.time() - time_start:.2f}s")

    multi_sim.stop_simulation()
```

As reflected in the output, the new bodies and connections are created in under **0.5 seconds**, while the simulation continues uninterrupted. The physical state remains continuous, and the world is modified dynamically at runtime without resetting or restarting the engine.

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
