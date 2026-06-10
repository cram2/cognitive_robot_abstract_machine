import os
from pathlib import Path

from krrood.entity_query_language.factories import entity, an, variable, count
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
)
import pycram
from test.conftest import world_with_urdf_factory

# %% Environment Setup
environment_path = os.path.join("package://iai_apartment/urdf/apartment.urdf")
# environment_path = os.path.join("package://iai_kit_mobile_lab/urdf/mobile_kitchen.urdf")
# environment_path = os.path.join("package://iai_kit_mobile_lab/urdf/R007.urdf")
environment_world = URDFParser.from_file(environment_path).parse()

# %% Robot Setup
robot_path = os.path.join("package://garmi_description/urdf/garmi.urdf")
robot_starting_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
    3.8,
    8.40,
    0,
)
robot_world = world_with_urdf_factory(robot_path, Garmi, OmniDrive, robot_starting_pose)

robot_world.merge_world(environment_world)
world = robot_world


bowl = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "bowl.stl"
    )
).parse()
with world.modify_world():
    world.merge_world_at_pose(
        bowl,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            5.3, 8.40, 1.09, reference_frame=world.root
        ),
    )


# %% Visualization
try:
    import rclpy

    rclpy.init()
    rclpy_node = rclpy.create_node("ros_node")
    viz = VizMarkerPublisher(_world=world, node=rclpy_node)
    viz.with_tf_publisher()
except ImportError:
    pass

# %% Demo
context = Context.from_world(world)
garmi = world.get_semantic_annotations_by_type(Garmi)[0]
milk_place_pose = Pose(Point3(x=2.2, y=7.6, z=0.865), reference_frame=world.root)

# robot = variable(AbstractRobot, [garmi])
# number_of_arms = an(entity(count(robot.end_effectors))).tolist()

# print(number_of_arms)
with simulated_robot:
    sequential(
        [ParkArmsAction(arm=Arms.BOTH)],
        context,
    ).perform()
