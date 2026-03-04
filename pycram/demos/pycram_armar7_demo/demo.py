import os
from typing import Type

import numpy as np

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import MoveTorsoActionDescription, TransportActionDescription
from pycram.robot_plans import ParkArmsActionDescription
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.package_resolver import PathResolver
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.armar7 import Armar7
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Spoon
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    OmniDrive,
    DiffDrive,
    Connection6DoF,
)
from semantic_digital_twin.world_description.world_entity import Body


def world_with_urdf_factory(
    urdf_path: str,
    robot_semantic_annotation: Type[AbstractRobot] | None,
    drive_connection_type: Type[OmniDrive | DiffDrive],
    robot_starting_pose: HomogeneousTransformationMatrix | None = None,
    urdf_path_resolver: PathResolver | None = None,
):
    """
    Builds this tree:
    map -> odom_combined -> "urdf tree"
    """
    urdf_parser = URDFParser.from_file(
        file_path=urdf_path, path_resolver=urdf_path_resolver
    )
    world_with_urdf = urdf_parser.parse()
    if robot_semantic_annotation is not None:
        robot_semantic_annotation.from_world(world_with_urdf)

    with world_with_urdf.modify_world():
        map = Body(name=PrefixedName("map"))
        localization_body = Body(name=PrefixedName("odom_combined"))

        map_C_localization = Connection6DoF.create_with_dofs(
            world_with_urdf, map, localization_body
        )
        world_with_urdf.add_connection(map_C_localization)

        c_root_bf = drive_connection_type.create_with_dofs(
            parent=localization_body,
            child=world_with_urdf.root,
            world=world_with_urdf,
        )
        world_with_urdf.add_connection(c_root_bf)
        c_root_bf.has_hardware_interface = True
        if robot_starting_pose is not None:
            c_root_bf.origin = robot_starting_pose

    return world_with_urdf


robot_path = os.path.join("package://iai_kit_armar7/urdf/Armar7.urdf")

robot_starting_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
    1.5, 2.5, 0, yaw=-np.pi / 2
)

robot_world = world_with_urdf_factory(
    robot_path, Armar7, OmniDrive, robot_starting_pose
)

# environment_path = os.path.join("package://iai_kit_mobile_lab/urdf/mobile_kitchen.urdf")
# environment_path = os.path.join("package://iai_kit_mobile_lab/urdf/R007.urdf")
environment_path = os.path.join("package://iai_apartment/urdf/apartment.urdf")
world = URDFParser.from_file(environment_path).parse()
with world.modify_world():
    world.merge_world(robot_world)

spoon = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "spoon.stl"
    )
).parse()
bowl = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "bowl.stl"
    )
).parse()
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
with world.modify_world():
    world.merge_world_at_pose(
        milk_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 2, 1.05, reference_frame=world.root
        ),
    )
    world.merge_world_at_pose(
        cereal_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 1.8, 1.05, reference_frame=world.root
        ),
    )

with world.modify_world():
    world.merge_world_at_pose(
        bowl,
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            2.4, 2.2, 1, reference_frame=world.root
        ),
    )
    connection = FixedConnection(
        parent=world.get_body_by_name("cabinet10_drawer_top"), child=spoon.root
    )
    world.merge_world(spoon, connection)

try:
    import rclpy

    rclpy.init()
    rclpy_node = rclpy.create_node("ros_node")
    tf_wrapper = TFWrapper(node=rclpy_node)
    viz = VizMarkerPublisher(_world=world, node=rclpy_node)
    viz.with_tf_publisher()
except ImportError:
    pass

context = Context.from_world(world)

with world.modify_world():
    world_reasoner = WorldReasoner(world)
    world_reasoner.reason()
    world.add_semantic_annotations(
        [
            Bowl(root=world.get_body_by_name("bowl.stl")),
            Spoon(root=world.get_body_by_name("spoon.stl")),
        ]
    )

plan = SequentialPlan(
    context,
    ParkArmsActionDescription(Arms.BOTH),
    # MoveTorsoActionDescription(TorsoState.HIGH),
    TransportActionDescription(
        world.get_body_by_name("milk.stl"),
        PoseStamped.from_list([4.9, 3.3, 0.8], frame=world.root),
        Arms.LEFT,
    ),
    # TransportActionDescription(
    #     world.get_body_by_name("spoon.stl"),
    #     PoseStamped.from_list([5.1, 3.3, 0.75], [0, 0, 1, 1], frame=world.root),
    #     Arms.LEFT,
    # ),
    TransportActionDescription(
        world.get_body_by_name("breakfast_cereal.stl"),
        PoseStamped.from_list([5, 3.3, 0.75], frame=world.root),
        Arms.LEFT,
    ),
    TransportActionDescription(
        world.get_body_by_name("bowl.stl"),
        PoseStamped.from_list([5, 3.3, 0.75], frame=world.root),
        Arms.LEFT,
    ),
)

with simulated_robot:
    plan.perform()
