import os
from dataclasses import dataclass
from enum import Enum, auto

from typing_extensions import Type

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot, simulated_robot_advanced
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.view_manager import ViewManager

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.armar7 import Armar7
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import OmniDrive

# %% Robot selection


class DemoRobot(Enum):
    """
    The robots that this demo can spawn into the apartment world.
    """

    ARMAR7 = auto()
    UNITREE_G1 = auto()
    GARMI = auto()
    PR2 = auto()
    TIAGO = auto()
    HSR = auto()


@dataclass
class RobotSpecification:
    """
    Everything needed to spawn one of the supported robots into the demo world.
    """

    semantic_annotation: Type[AbstractRobot]
    """
    The semantic annotation class used to recognize the robot in the world.
    """

    urdf_path: str
    """
    The ROS package path to the robot's URDF (or xacro) description.
    """

    starting_pose: HomogeneousTransformationMatrix
    """
    The pose of the robot's root body relative to the apartment world's root.
    """

    can_transport_bowl: bool
    """
    Whether this robot can run the bowl :class:`TransportAction` in this apartment.

    Armar7's mobile base lacks collision geometry that :class:`TransportAction` relies
    on, and Garmi's and UnitreeG1's arm reach for the bowl fails reachability planning
    regardless of grasp pose or approach direction. Until those robot models are fixed,
    these robots only park their arms.
    """


ROBOT_SPECIFICATIONS: dict[DemoRobot, RobotSpecification] = {
    DemoRobot.ARMAR7: RobotSpecification(
        Armar7,
        "package://iai_kit_armar7/urdf/Armar7.urdf",
        HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2.5, 0),
        can_transport_bowl=True,
    ),
    DemoRobot.UNITREE_G1: RobotSpecification(
        UnitreeG1,
        "package://iai_offis_g1_description/urdf/offis_unitree_g1.urdf",
        HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2.5, 0),
        can_transport_bowl=True,
    ),
    DemoRobot.GARMI: RobotSpecification(
        Garmi,
        "package://garmi_description/urdf/garmi.urdf",
        # Garmi's arms sit further forward than the other robots', so it is spawned
        # 0.5m further back to keep them clear of the table on spawn.
        HomogeneousTransformationMatrix.from_xyz_rpy(1.0, 2.5, 0),
        can_transport_bowl=True,
    ),
    DemoRobot.PR2: RobotSpecification(
        PR2,
        "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro",
        HomogeneousTransformationMatrix.from_xyz_rpy(1.2, 2.5, 0),
        can_transport_bowl=True,
    ),
    DemoRobot.TIAGO: RobotSpecification(
        Tiago,
        "package://iai_tiago_description/urdf/tiago_from_our_robot.urdf",
        HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2.5, 0),
        can_transport_bowl=True,
    ),
    DemoRobot.HSR: RobotSpecification(
        HSRB,
        "package://hsr_description/robots/hsrb4s.urdf.xacro",
        HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2.5, 0),
        can_transport_bowl=True,
    ),
}

# Change this to switch which robot is spawned into the apartment.
SELECTED_ROBOT = DemoRobot.GARMI
robot_specification = ROBOT_SPECIFICATIONS[SELECTED_ROBOT]

# %% World setup

apartment_world = URDFParser.from_file(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "worlds", "apartment.urdf"
    )
).parse()
robot_world = URDFParser.from_file(robot_specification.urdf_path).parse()

milk_world = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "milk.stl"
    )
).parse()
cereal_world = STLParser(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "resources",
        "objects",
        "breakfast_cereal.stl",
    )
).parse()
spoon_world = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "spoon.stl"
    )
).parse()

with apartment_world.modify_world():
    root_connection = OmniDrive.create_with_dofs(
        parent=apartment_world.root, child=robot_world.root, world=apartment_world
    )
    apartment_world.merge_world(robot_world, root_connection)
    root_connection.origin = robot_specification.starting_pose
    apartment_world.merge_world_at_pose(
        milk_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 2, 1.05, reference_frame=apartment_world.root
        ),
    )
    apartment_world.merge_world_at_pose(
        cereal_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 1.8, 1.05, reference_frame=apartment_world.root
        ),
    )
    apartment_world.merge_world_at_pose(
        spoon_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 1.6, 1.05, reference_frame=apartment_world.root
        ),
    )

world = apartment_world

try:
    import rclpy
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    rclpy.init()
    node = rclpy.create_node("viz_marker")
    VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
except ImportError:
    node = None

# %% Demo

robot = robot_specification.semantic_annotation.from_world(world)
robot.mobile_base.full_body_controlled = True
context = Context(world=world, robot=robot, _debug=False, ros_node=node)
context.evaluate_conditions = False
context.teleport_to_navigate_in_simulation = True

actions = [ParkArmsAction(Arms.BOTH)]
actions.append(
    TransportAction(
        world.get_body_by_name("milk.stl"),
        Pose.from_xyz_rpy(5, 3.3, 0.75, yaw=1.57, reference_frame=world.root),
        Arms.LEFT,
    ))
actions.append(
    TransportAction(
        world.get_body_by_name("breakfast_cereal.stl"),
        Pose.from_xyz_rpy(5.2, 3.3, 0.75, yaw=1.57, reference_frame=world.root),
        Arms.LEFT,
    ))
# actions.append(
#     TransportAction(
#         world.get_body_by_name("spoon.stl"),
#         Pose.from_xyz_rpy(4.8, 3.3, 0.75, yaw=1.57, reference_frame=world.root),
#         Arms.LEFT,
#         GraspDescription(
#             ApproachDirection.FRONT,
#             VerticalAlignment.TOP,
#             ViewManager.get_end_effector_view(Arms.LEFT, robot),
#         ),
#     ))

plan = sequential(actions, context=context).plan

with simulated_robot:
    plan.perform()
