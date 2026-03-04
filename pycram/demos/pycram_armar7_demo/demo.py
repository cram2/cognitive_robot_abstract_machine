import os

import numpy as np

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import ParkArmsActionDescription, MoveTorsoActionDescription
from pycram.robot_plans import TransportActionDescription
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.armar7 import Armar7
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Spoon
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.utils import world_with_urdf_factory

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

try:
    import rclpy

    rclpy.init()
    rclpy_node = rclpy.create_node("ros_node")
    viz = VizMarkerPublisher(_world=world, node=rclpy_node)
    viz.with_tf_publisher()
except ImportError:
    pass

context = Context.from_world(world)

with world.modify_world():
    world_reasoner = WorldReasoner(world)
    world_reasoner.reason()

plan = SequentialPlan(
    context,
    ParkArmsActionDescription(Arms.BOTH),
    TransportActionDescription(
        world.get_body_by_name("milk.stl"),
        PoseStamped.from_list([4.9, 3.3, 0.8], [0, 0, 1, 1], frame=world.root),
        Arms.LEFT,
    ),
    TransportActionDescription(
        world.get_body_by_name("breakfast_cereal.stl"),
        PoseStamped.from_list([5.1, 3.3, 0.8], [0, 0, 1, 1], frame=world.root),
        Arms.LEFT,
    ),
)

with simulated_robot:
    plan.perform()
