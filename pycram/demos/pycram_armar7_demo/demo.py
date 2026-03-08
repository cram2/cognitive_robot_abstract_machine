import os
from pathlib import Path
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped
from pycram.designators.location_designator import CostmapLocation
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import (
    ParkArmsActionDescription,
    NavigateActionDescription,
    PickUpActionDescription,
    PlaceActionDescription,
)
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.armar7 import Armar7
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.utils import get_path_to_project_root
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
)
from semantic_digital_twin.world_description.utils import world_with_urdf_factory
import pycram


# %% Environment Setup
# environment_path = os.path.join("package://iai_apartment/urdf/apartment.urdf")
# environment_path = os.path.join("package://iai_kit_mobile_lab/urdf/mobile_kitchen.urdf")
environment_path = os.path.join("package://iai_kit_mobile_lab/urdf/R007.urdf")
world = URDFParser.from_file(environment_path).parse()

# %% Robot Setup
robot_path = os.path.join("package://iai_kit_armar7/urdf/Armar7.urdf")
robot_starting_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
    3.8,
    8.40,
    0,
)
robot_world = world_with_urdf_factory(
    robot_path, Armar7, OmniDrive, robot_starting_pose
)

with world.modify_world():
    world.merge_world(robot_world)

# %% Spawn Objects
project_root = get_path_to_project_root(Path(pycram.__file__).resolve())
milk_world = STLParser(
    os.path.join(project_root, "resources", "objects", "milk.stl")
).parse()
cereal_world = STLParser(
    os.path.join(
        project_root,
        "resources",
        "objects",
        "breakfast_cereal.stl",
    )
).parse()
with world.modify_world():
    world.merge_world_at_pose(
        milk_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            5.3, 8.40, 1.09, reference_frame=world.root
        ),
    )
    world.merge_world_at_pose(
        cereal_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            5.3, 8.25, 1.09, reference_frame=world.root
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
armar7 = world.get_semantic_annotations_by_type(Armar7)[0]
milk_place_pose = PoseStamped.from_list(
    [2.2, 7.6, 0.865], [0, 0, 1, 0], frame=world.root
)

with simulated_robot:
    SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        NavigateActionDescription(
            pickup_loc := CostmapLocation(
                target=PoseStamped.from_spatial_type(
                    world.get_body_by_name("milk.stl").global_pose
                ),
                reachable_arm=Arms.LEFT,
                reachable_for=armar7,
            ),
            teleport=False,
        ),
        PickUpActionDescription(
            world.get_body_by_name("milk.stl"),
            pickup_loc.last_result_arm,
            pickup_loc.last_result_grasp,
        ),
        ParkArmsActionDescription(Arms.BOTH),
        NavigateActionDescription(
            place_loc := CostmapLocation(
                target=milk_place_pose,
                reachable_arm=pickup_loc.last_result_arm,
                reachable_for=armar7,
                grasp_descriptions=pickup_loc.last_result_grasp,
            ),
            teleport=False,
        ),
        PlaceActionDescription(
            world.get_body_by_name("milk.stl"),
            milk_place_pose,
            place_loc.last_result_arm,
        ),
        ParkArmsActionDescription(Arms.BOTH),
    ).perform()
