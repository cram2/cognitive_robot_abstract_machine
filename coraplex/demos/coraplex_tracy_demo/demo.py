import os
from math import pi

from coraplex.alternative_motion_mappings.tracy_motion_mapping import TracyGripperMotion
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.grasp import GraspDescription
from coraplex.motion_executor import real_robot, simulated_robot
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
import rclpy
from semantic_digital_twin.spatial_types.spatial_types import Pose

from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans import MoveGripperMotion
import coraplex.alternative_motion_mappings.tracy_motion_mapping  # type: ignore
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.testing import setup_world
from coraplex.language import SequentialNode
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from define_sim_tracy import setup_sim_tracy

#  Environmental Variables Setup
real = False

# %% Robot and World Setup
if real:
    node, world, robot_view, context = setup_tracy_context()
else:
    node, world, robot_view, context = setup_sim_tracy()

# %% Define Additional Objects
try:
    cup = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "objects", "jeroen_cup.stl"
        )
    ).parse()
    cup_root = cup.root

    bowl = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "objects", "bowl.stl"
        )
    ).parse()
    bowl_root = bowl.root

    with world.modify_world():
        world.merge_world(
            cup,
            FixedConnection(
                world.root,
                cup.root,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_quaternion(
                    0.5, 0.15, 0.88, reference_frame=world.root
                ),
            ),
        )

except Exception as e:
    print(e)
    print(
        "Bowl already exists in the world. Using existing bowl instead of creating a new one."
    )
    bowl = world.get_body_by_name(PrefixedName("bowl"))


# %% Demo
print(world.root.name)
print([body.name for body in world.bodies])

# Print joint states
for dof in context.robot.degrees_of_freedom_with_hardware_interface:
    print(f"{dof.name}: {dof.variables.position.resolve()}")

# Grasp description configured to comfortably reach the front-positioned cup
pick_up_grasp = GraspDescription(
    approach_direction=ApproachDirection.LEFT,
    vertical_alignment=VerticalAlignment.NoAlignment,
    end_effector=context.robot.get_left_arm_if_specified().end_effector,
    manipulation_offset=0.02,

)

plan = sequential(
    [
        ParkArmsAction(arm=Arms.BOTH),
        SetGripperAction(gripper=Arms.BOTH, motion=GripperState.OPEN),
        PickUpAction(world.get_body_by_name("jeroen_cup.stl"), Arms.LEFT, pick_up_grasp),
        PlaceAction(
            world.get_body_by_name("jeroen_cup.stl"),
            HomogeneousTransformationMatrix.from_xyz_rpy(
                0.6, -0.1, 0.9, reference_frame=world.root
            ).to_pose(),
            Arms.LEFT,
        ),
    ],
    context,
)

if real:
    with real_robot:
        plan.perform()
else:
    with simulated_robot:
        plan.perform()

print("Plan finished.")
print("Done.")