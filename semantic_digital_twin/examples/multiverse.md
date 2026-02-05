---
jupyter:
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

(multiverse)=
# Simulation with Multiverse

Multiverse Framework is a system that connects different robotics simulation and hardware components together. 
It serves as a unifying infrastructure that addresses the fragmentation across simulation and controller software. 
Instead of being a single simulator extended with controllers and reasoning modules, 
it provides an architectural foundation with interoperability mechanisms that allow simulators, controllers, 
and reasoning systems to remain independent while still operating together as a coherent ecosystem.

This package uses Multiverse Simulators (MultiSim) to provide a unified abstraction layer over different simulation engines.
This high-level abstraction enables us to switch between multiple backends with minimal effort, 
while maintaining a consistent logic for world representation and manipulation.
Communication between the simulator and other components running in separate processes is managed by the Multiverse Framework through its plugins.

Below is an example of how to set up an empty simulation using the Mujoco backend and run it for 5 seconds.

```{code-cell} ipython3
from semantic_digital_twin.world import World
from multiverse_simulator import MultiverseSimulatorState, MultiverseViewer
from semantic_digital_twin.adapters.multi_sim import MujocoSim
import os
import time

world = World()
viewer = MultiverseViewer()
headless = os.environ.get("CI", "false").lower() == "true" # headless in CI environments
multi_sim = MujocoSim(world=world, viewer=viewer, headless=headless, step_size=1E-3)
multi_sim.start_simulation()

start_time = time.time()
time.sleep(5.0)
multi_sim.stop_simulation()
```

A MultiSim simulator takes a `World` object as input and synchronizes its state with the simulation engine, 
a `MultiverseViewer` for reading and write data in run-time, and other configuration parameters.

You can spawn objects in the simulation in run-time just by changing the `World` object.
The world changes will be automatically synchronized with the simulation engine.

```{code-cell} ipython3
from semantic_digital_twin.world import World
from multiverse_simulator import MultiverseSimulatorState, MultiverseViewer
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
import os
import time

world = World()
viewer = MultiverseViewer()
headless = os.environ.get("CI", "false").lower() == "true" # headless in CI environments
multi_sim = MujocoSim(world=world, viewer=viewer, headless=headless, step_size=1E-3)
multi_sim.start_simulation()

start_time = time.time()
time.sleep(1.0)
print(f"Time to start creating a new body: {time.time() - start_time}s")
new_body = Body(name=PrefixedName("test_body"))
box_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
    x=0.2, y=0.4, z=-0.3, roll=0, pitch=0.5, yaw=0, reference_frame=new_body
)
box = Box(
    origin=box_origin,
    scale=Scale(1.0, 1.5, 0.5),
    color=Color(
        1.0,
        0.0,
        0.0,
        1.0,
    ),
)
new_body.collision = ShapeCollection([box], reference_frame=new_body)

with world.modify_world():
    world.add_connection(
        Connection6DoF.create_with_dofs(
            world=world,
            parent=world.root,
            child=new_body,
        )
    )
print(f"Time to add new body: {time.time() - start_time}s")

time.sleep(5.0)
multi_sim.stop_simulation()
```
