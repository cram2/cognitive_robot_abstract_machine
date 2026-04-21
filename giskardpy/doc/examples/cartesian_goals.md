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

# Cartesian Goals Example

This example demonstrates how to use `CartesianPose` goals to move a robot to a specific pose. We will use the PR2 robot for this demonstration.

```{code-cell} ipython3
import os
from giskardpy.executor import SimulationPacer, Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World
from semantic_digital_twin.spatial_types.spatial_types import Pose, Vector3, Quaternion
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import Connection6DoF, OmniDrive

# 1. Load the PR2 robot into the world
# We use the same configuration as in pr2_world_setup
urdf_path = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
urdf_parser = URDFParser.from_file(file_path=urdf_path)
world = urdf_parser.parse()

# Initialize PR2 semantic annotation
try:
    pr2 = PR2.from_world(world)
except Exception as e:
    print(f"Warning: Could not fully initialize PR2 semantic annotation: {e}")

# Replicate the world structure from world_with_urdf_factory:
# map -> odom_combined -> base_footprint (robot root)
with world.modify_world():
    map_body = Body(name=PrefixedName("map"))
    localization_body = Body(name=PrefixedName("odom_combined"))

    map_C_localization = Connection6DoF.create_with_dofs(
        world, map_body, localization_body
    )
    world.add_connection(map_C_localization)

    c_root_bf = OmniDrive.create_with_dofs(
        parent=localization_body,
        child=world.root,
        world=world,
    )
    world.add_connection(c_root_bf)
    c_root_bf.has_hardware_interface = True

# 2. Create a Motion Statechart
msc = MotionStatechart()

# 3. Define a Cartesian Pose Goal
# We want to move the left gripper to a specific pose relative to the world root
root_link = world.root
tip_link = world.get_body_by_name("l_gripper_tool_frame")

goal_pose = Pose(
    position=Vector3(0.5, 0.5, 1.2), 
    orientation=Quaternion(0, 0, 0, 1), 
    reference_frame=root_link
)
goal = CartesianPose(
    name="move_to_goal", 
    root_link=root_link, 
    tip_link=tip_link, 
    goal_pose=goal_pose
)

msc.add_node(goal)
msc.add_node(EndMotion.when_true(goal))

# 4. Set up the Executor
kin_sim = Executor(
    context=MotionStatechartContext(
        world=world,
        qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
    ),
    pacer=SimulationPacer(real_time_factor=None), # None for fastest execution in docs
)

kin_sim.compile(msc)
kin_sim.tick_until_end(timeout=100)
print(f"Motion finished after {kin_sim.control_cycles} cycles.")
```
