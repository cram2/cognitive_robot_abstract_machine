import os
import threading

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from krrood.entity_query_language.factories import underspecified, variable
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
    MovementType,
    ExecutionType,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.locations.locations import CostmapLocation
from pycram.motion_executor import real_robot, simulated_robot, ExecutionEnvironment
from pycram.plans.factories import execute_single, sequential
from pycram.robot_plans import MoveToolCenterPointMotion, MoveGripperMotion
from pycram.robot_plans.actions.core.navigation import NavigateAction, LookAtAction
from pycram.robot_plans.actions.core.pick_up import ReachAction, PickUpAction
from pycram.robot_plans.actions.core.placing import PlaceAction
from pycram.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    SetGripperAction,
    ParkArmsAction,
)
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.package_resolver import CompositePathResolver
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    ModelSynchronizer,
    StateSynchronizer,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState, GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Shelf,
    ShelfLayer,
    Wall,
    Cereal,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
import pycram.alternative_motion_mappings.stretch_motion_mapping  # type: ignore
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    DifferentialDrive,
    OmniDrive,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

rclpy.init()
node = rclpy.create_node("stretch_demo_node")

executor = SingleThreadedExecutor()
executor.add_node(node)

thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
thread.start()

exec_type = ExecutionType.SIMULATED

exec_env = ExecutionEnvironment(exec_type)

if exec_type == ExecutionType.SIMULATED:
    stretch_parse = URDFParser.from_file(
        "package://stretch_description/urdf/stretch_description_RE2V0_tool_stretch_dex_wrist.xacro"
    )
    world = stretch_parse.parse()
    Stretch.from_world(world)
    with world.modify_world():

        world.add_body(map_body := Body(name=PrefixedName("map")))

        world.add_connection(
            drive := DifferentialDrive.create_with_dofs(world, map_body, world.root)
        )
    VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
    drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, 1, reference_frame=world.root
    )
else:

    world = fetch_world_from_service(node)
    ModelSynchronizer(_world=world, node=node)
    StateSynchronizer(_world=world, node=node)

if not world.is_entity_in_world_by_name("cheeze_it.obj"):
    with world.modify_world():
        shelf = Shelf.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("shelf"),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.455 + (0.85 / 2),
                0,
                1.9 / 2,
                yaw=-np.pi / 2,
                reference_frame=world.root,
            ),
            scale=Scale(0.305, 0.85, 1.9),
            wall_thickness=0.035,
        )
        shelf_layer1 = ShelfLayer.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("shelf_layer1"),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.455 + (0.85 / 2), 0, 0.283, yaw=-np.pi / 2, reference_frame=world.root
            ),
            scale=Scale(0.305, 0.85, 0.018),
        )
        shelf.add_shelf_layer(shelf_layer1)
        shelf_layer2 = ShelfLayer.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("shelf_layer2"),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.455 + (0.85 / 2), 0, 0.63, yaw=-np.pi / 2, reference_frame=world.root
            ),
            scale=Scale(0.305, 0.85, 0.018),
        )
        shelf.add_shelf_layer(shelf_layer2)
        shelf_layer3 = ShelfLayer.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("shelf_layer3"),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.455 + (0.85 / 2), 0, 1.265, yaw=-np.pi / 2, reference_frame=world.root
            ),
            scale=Scale(0.305, 0.85, 0.018),
        )
        shelf.add_shelf_layer(shelf_layer3)
        shelf_layer4 = ShelfLayer.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("shelf_layer4"),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.455 + (0.85 / 2), 0, 1.613, yaw=-np.pi / 2, reference_frame=world.root
            ),
            scale=Scale(0.305, 0.85, 0.018),
        )
        shelf.add_shelf_layer(shelf_layer4)

        Wall.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("wall"),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                0, 0.29 + (2.81 / 2), 0, reference_frame=world.root
            ),
            scale=Scale(0.03, 2.81, 0.265),
        )
        sofa_world = STLParser(
            file_path=CompositePathResolver().resolve(
                "package://iai_apartment/meshes/visual/sofa_bed.obj"
            )
        ).parse()
        sofa_height = (
            sofa_world.root.collision.max_point[2]
            - sofa_world.root.collision.min_point[2]
        )
        world_T_sofa = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.0,
            y=3.45,
            z=sofa_height / 2,
            yaw=13 * (-np.pi / 24),
            reference_frame=world.root,
        )
        world.merge_world_at_pose(sofa_world, world_T_sofa)

        bedside_table_world = STLParser(
            file_path=CompositePathResolver().resolve(
                "package://iai_apartment/meshes/visual/bedside_table.dae"
            )
        ).parse()
        world.merge_world_at_pose(
            bedside_table_world,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1.97, y=2.950, yaw=-np.pi / 24, reference_frame=world.root
            ),
        )

        wall_world = STLParser(
            file_path=CompositePathResolver().resolve(
                "package://iai_apartment/meshes/visual/walls.dae"
            )
        ).parse()

        world.merge_world_at_pose(
            wall_world,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-7.34, y=1.58, yaw=0, reference_frame=world.root
            ),
        )

        wardrobe_world = STLParser(
            file_path=CompositePathResolver().resolve(
                "package://iai_apartment/meshes/visual/wardrobe.dae"
            )
        ).parse()
        wardrobe_door_left_world = STLParser(
            file_path=CompositePathResolver().resolve(
                "package://iai_apartment/meshes/visual/wardrobe_door_left.dae"
            )
        ).parse()

        wardrobe_door_left_handle_world = STLParser(
            file_path=CompositePathResolver().resolve(
                "package://iai_apartment/meshes/visual/wardrobe_door_handle.dae"
            )
        ).parse()
        wardrobe_door_left_handle_world.root.name.name = "wardrobe_door_handle_left"
        wardrobe_door_left_world.merge_world_at_pose(
            wardrobe_door_left_handle_world,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.032089,
                -0.460513,
                0.973703,
                reference_frame=wardrobe_door_left_world.root,
            ),
        )

        wardrobe_world.merge_world_at_pose(
            wardrobe_door_left_world,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.3246, 0.5, reference_frame=world.root
            ),
        )

        wardrobe_door_right_world = STLParser(
            file_path=CompositePathResolver().resolve(
                "package://iai_apartment/meshes/visual/wardrobe_door_right.dae"
            )
        ).parse()

        wardrobe_door_right_handle_world = STLParser(
            file_path=CompositePathResolver().resolve(
                "package://iai_apartment/meshes/visual/wardrobe_door_handle.dae"
            )
        ).parse()
        wardrobe_door_right_handle_world.root.name.name = "wardrobe_door_handle_right"
        wardrobe_door_right_world.merge_world_at_pose(
            wardrobe_door_right_handle_world,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.032089,
                0.460513,
                0.973703,
                reference_frame=wardrobe_door_right_world.root,
            ),
        )

        wardrobe_world.merge_world_at_pose(
            wardrobe_door_right_world,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.3246, -0.5, reference_frame=world.root
            ),
        )

        world.merge_world_at_pose(
            wardrobe_world,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1.95, y=0, yaw=-np.pi / 2, reference_frame=world.root
            ),
        )

        cereal = STLParser(
            file_path=CompositePathResolver().resolve(
                "package://iai_apartment/meshes/visual/cheeze_it.obj"
            )
        ).parse()

        world.merge_world(
            cereal,
            FixedConnection(
                shelf_layer2.root,
                cereal.root,
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=0.105, yaw=-np.pi / 2, reference_frame=shelf_layer2.root
                ),
            ),
        )

print(len(world.bodies))

robot_annotation = world.get_semantic_annotations_by_type(Stretch)[0]
context = Context(
    world, robot_annotation, node, _debug=False, evaluate_conditions=False
)

grasp_desc = GraspDescription(
    ApproachDirection.LEFT,
    VerticalAlignment.TOP,
    robot_annotation.arm.manipulator,
    rotate_gripper=True,
)

# input("Ready ...")

plan = sequential(
    [
        # LookAtAction(Pose.from_xyz_rpy(-0.04, 0.69, 0.7, reference_frame=world.root)),
        ParkArmsAction(Arms.BOTH),
        NavigateAction(Pose.from_xyz_rpy(1.0, 0.45, 0, reference_frame=world.root)),
        PickUpAction(world.get_body_by_name("cheeze_it.obj"), Arms.LEFT, grasp_desc),
        ParkArmsAction(Arms.BOTH),
        NavigateAction(
            Pose.from_xyz_rpy(2, 2.4, 0, yaw=np.pi, reference_frame=world.root)
        ),
        PlaceAction(
            object_designator=world.get_body_by_name("cheeze_it.obj"),
            target_location=Pose.from_xyz_rpy(2, 2.9, 0.46, reference_frame=world.root),
            arm=Arms.LEFT,
        ),
        ParkArmsAction(Arms.BOTH),
    ],
    context,
).plan

with exec_env:
    plan.perform()


rclpy.shutdown()
