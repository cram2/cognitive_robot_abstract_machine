import threading
from time import sleep

import threading
from time import sleep

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

import pycram.alternative_motion_mappings.stretch_motion_mapping  # type: ignore
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
    ExecutionType,
    DetectionTechnique,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.motion_executor import ExecutionEnvironment
from pycram.plans.factories import sequential
from pycram.robot_plans import (
    MoveJointsMotion,
    MoveGripperMotion,
)
from pycram.robot_plans.actions.core.misc import DetectAction
from pycram.robot_plans.actions.core.navigation import NavigateAction, LookAtAction
from pycram.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    StretchExtendArm,
    StretchRetractArm,
    StretchTorsoHeightDirectlyBelowShelf,
    StretchTorsoShelfPickPlaceHeight,
    StretchTorsoTablePickPlaceHeight,
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
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Shelf,
    ShelfLayer,
    Wall,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    DifferentialDrive,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

rclpy.init()
node = rclpy.create_node("stretch_demo_node")

executor = SingleThreadedExecutor()
executor.add_node(node)

thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
thread.start()

exec_type = ExecutionType.REAL
demo_start_of_table_instead_of_shelf = False

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
                -0.17,
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
                0.455 + (0.85 / 2),
                -0.17,
                0.283,
                yaw=-np.pi / 2,
                reference_frame=world.root,
            ),
            scale=Scale(0.305, 0.85, 0.018),
        )
        shelf.add_shelf_layer(shelf_layer1)
        shelf_layer2 = ShelfLayer.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("shelf_layer2"),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.455 + (0.85 / 2),
                -0.17,
                0.63,
                yaw=-np.pi / 2,
                reference_frame=world.root,
            ),
            scale=Scale(0.305, 0.85, 0.018),
        )
        shelf.add_shelf_layer(shelf_layer2)
        shelf_layer3 = ShelfLayer.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("shelf_layer3"),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.455 + (0.85 / 2),
                -0.17,
                1.265,
                yaw=-np.pi / 2,
                reference_frame=world.root,
            ),
            scale=Scale(0.305, 0.85, 0.018),
        )
        shelf.add_shelf_layer(shelf_layer3)
        shelf_layer4 = ShelfLayer.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("shelf_layer4"),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.455 + (0.85 / 2),
                -0.17,
                1.613,
                yaw=-np.pi / 2,
                reference_frame=world.root,
            ),
            scale=Scale(0.305, 0.85, 0.018),
        )
        shelf.add_shelf_layer(shelf_layer4)

        Wall.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("wall"),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                0, (2.81 / 2), 0, reference_frame=world.root
            ),
            scale=Scale(0.03, 2.81, 0.265),
        )
    sofa_world = STLParser(
        file_path=CompositePathResolver().resolve(
            "package://iai_apartment/meshes/visual/sofa_bed.obj"
        )
    ).parse()

    bedside_table_world = STLParser(
        file_path=CompositePathResolver().resolve(
            "package://iai_apartment/meshes/visual/bedside_table.dae"
        )
    ).parse()
    world_T_bedside_table = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.92, y=2.68, yaw=17.8 * (-np.pi / 32), reference_frame=world.root
    )
    world.merge_world(
        bedside_table_world,
        FixedConnection(
            parent=world.root,
            child=bedside_table_world.root,
            parent_T_connection_expression=world_T_bedside_table,
        ),
    )

    sofa_height = (
        sofa_world.root.collision.max_point[2] - sofa_world.root.collision.min_point[2]
    )
    world_T_sofa = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0,
        y=3.15,
        z=sofa_height / 2,
        yaw=17.75 * (-np.pi / 32),
        reference_frame=world.root,
    )
    world.merge_world(
        sofa_world, FixedConnection(world.root, sofa_world.root, world_T_sofa)
    )

    wall_world = STLParser(
        file_path=CompositePathResolver().resolve(
            "package://iai_apartment/meshes/visual/walls.dae"
        )
    ).parse()

    world_T_wall = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=-7.34, y=1.43, z=-0.2, yaw=0, reference_frame=world.root
    )
    world.merge_world(
        wall_world,
        FixedConnection(world.root, wall_world.root, world_T_wall),
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
    wardrobe_door_left_world.merge_world(
        wardrobe_door_left_handle_world,
        FixedConnection(
            wardrobe_door_left_world.root,
            wardrobe_door_left_handle_world.root,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.032089,
                -0.460513,
                0.973703,
                reference_frame=wardrobe_door_left_world.root,
            ),
        ),
    )

    wardrobe_world.merge_world(
        wardrobe_door_left_world,
        FixedConnection(
            wardrobe_world.root,
            wardrobe_door_left_world.root,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.3246, 0.5, reference_frame=wardrobe_world.root
            ),
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
    wardrobe_door_right_world.merge_world(
        wardrobe_door_right_handle_world,
        FixedConnection(
            wardrobe_door_right_world.root,
            wardrobe_door_right_handle_world.root,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.032089,
                0.460513,
                0.973703,
                reference_frame=wardrobe_door_right_world.root,
            ),
        ),
    )

    wardrobe_world.merge_world(
        wardrobe_door_right_world,
        FixedConnection(
            wardrobe_world.root,
            wardrobe_door_right_world.root,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.3246, -0.5, reference_frame=wardrobe_world.root
            ),
        ),
    )

    world.merge_world(
        wardrobe_world,
        FixedConnection(
            world.root,
            wardrobe_world.root,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=2, y=-0.17, yaw=-np.pi / 2, reference_frame=world.root
            ),
        ),
    )

    cereal = STLParser(
        file_path=CompositePathResolver().resolve(
            "package://iai_apartment/meshes/visual/cheeze_it.obj"
        )
    ).parse()

    with world.modify_world():
        if demo_start_of_table_instead_of_shelf:
            parent = world.get_body_by_name("bedside_table.dae")
            bedside_height = (
                parent.collision.max_point[2] - parent.collision.min_point[2]
            )
            surface_T_cereal = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.10,
                y=0.0,
                # frame is at the bottom, not the middle
                z=bedside_height + 0.105,
                yaw=np.pi / 2,
                reference_frame=parent,
            )
        else:
            parent = shelf_layer2.root
            surface_T_cereal = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-0.10,
                y=0.0,
                z=0.115,
                yaw=-np.pi / 2,
                reference_frame=parent,
            )

        world.merge_world(
            cereal,
            FixedConnection(
                parent,
                cereal.root,
                surface_T_cereal,
            ),
        )

print(len(world.bodies))

robot_annotation = world.get_semantic_annotations_by_type(Stretch)[0]
context = Context(
    world, robot_annotation, node, _debug=False, evaluate_conditions=False
)

grasp_desc = GraspDescription(
    ApproachDirection.FRONT,
    VerticalAlignment.NoAlignment,
    robot_annotation.arm.manipulator,
    rotate_gripper=True,
)

# input("Ready ...")
cereal_body = world.get_body_by_name("cheeze_it.obj")
shelf_body = shelf_layer2.root
bedside_table_body = world.get_body_by_name("bedside_table.dae")

nav_perceive_pose_relative_to_cereal = Pose.from_xyz_rpy(
    y=-0.55, yaw=np.pi / 2, reference_frame=cereal_body
)
nav_grasp_pose_relative_to_cereal = Pose.from_xyz_rpy(
    y=-0.35, yaw=np.pi, reference_frame=cereal_body
)
nav_place_pose_relative_to_bedside_table = Pose.from_xyz_rpy(
    x=0.45, yaw=-np.pi / 2, reference_frame=bedside_table_body
)
nav_place_pose_relative_to_shelf = Pose.from_xyz_rpy(
    x=-0.45, yaw=np.pi / 2, reference_frame=shelf_body
)


def detect_cereal():
    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            NavigateAction(nav_perceive_pose_relative_to_cereal),
            LookAtAction(cereal_body.global_pose),
            DetectAction(
                technique=DetectionTechnique.TYPES,
            ),
        ],
        context,
    )
    with exec_env:
        plan.perform()


# %% Shelf Pickup
def shelf_pickup():
    if not demo_start_of_table_instead_of_shelf:
        detect_cereal()
        start_action = [NavigateAction(nav_grasp_pose_relative_to_cereal)]
    else:
        start_action = []
    plan = sequential(
        [
            *start_action,
            MoveGripperMotion(motion=GripperState.OPEN, gripper=Arms.LEFT),
            StretchTorsoHeightDirectlyBelowShelf(),
            StretchExtendArm(),
            StretchTorsoShelfPickPlaceHeight(),
            MoveGripperMotion(motion=GripperState.CLOSE, gripper=Arms.LEFT),
        ],
        context,
    ).plan

    with exec_env:
        plan.perform()

    with world.modify_world():
        world.move_branch_with_fixed_connection(
            cereal_body, robot_annotation.arm.manipulator.tool_frame
        )

    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
        ],
        context,
    ).plan

    with exec_env:
        plan.perform()


# %% Table Place
def table_place():
    plan = sequential(
        [
            NavigateAction(nav_place_pose_relative_to_bedside_table),
            StretchExtendArm(),
            StretchTorsoTablePickPlaceHeight(),
            MoveGripperMotion(motion=GripperState.OPEN, gripper=Arms.LEFT),
        ],
        context,
    ).plan

    with exec_env:
        plan.perform()

    with world.modify_world():
        world.move_branch_with_fixed_connection(cereal_body, bedside_table_body)

    plan = sequential(
        [
            StretchTorsoHeightDirectlyBelowShelf(),
            MoveGripperMotion(motion=GripperState.CLOSE, gripper=Arms.LEFT),
            ParkArmsAction(Arms.BOTH),
        ],
        context,
    ).plan

    with exec_env:
        plan.perform()


# %% Table Pickup
def table_pickup():
    if demo_start_of_table_instead_of_shelf:
        detect_cereal()
        start_action = [NavigateAction(nav_grasp_pose_relative_to_cereal)]
    else:
        start_action = []
    plan = sequential(
        [
            *start_action,
            StretchExtendArm(),
            MoveGripperMotion(motion=GripperState.OPEN, gripper=Arms.LEFT),
            StretchTorsoTablePickPlaceHeight(),
            MoveGripperMotion(motion=GripperState.CLOSE, gripper=Arms.LEFT),
        ],
        context,
    ).plan

    with exec_env:
        plan.perform()

    with world.modify_world():
        world.move_branch_with_fixed_connection(
            cereal_body, robot_annotation.arm.manipulator.tool_frame
        )

    plan = sequential(
        [
            StretchTorsoHeightDirectlyBelowShelf(),
            ParkArmsAction(Arms.BOTH),
        ],
        context,
    ).plan
    with exec_env:
        plan.perform()


# %% Shelf Place
def shelf_place():
    plan = sequential(
        [
            NavigateAction(nav_place_pose_relative_to_shelf),
            StretchTorsoHeightDirectlyBelowShelf(),
            StretchExtendArm(),
            StretchTorsoShelfPickPlaceHeight(),
            MoveGripperMotion(motion=GripperState.OPEN, gripper=Arms.LEFT),
        ],
        context,
    ).plan

    with exec_env:
        plan.perform()

    with world.modify_world():
        world.move_branch_with_fixed_connection(cereal_body, shelf_body)

    plan = sequential(
        [
            StretchTorsoHeightDirectlyBelowShelf(),
            StretchRetractArm(),
            ParkArmsAction(Arms.BOTH),
        ],
        context,
    ).plan

    with exec_env:
        plan.perform()


if demo_start_of_table_instead_of_shelf:
    table_pickup()
    shelf_place()
    shelf_pickup()
    table_place()
else:
    shelf_pickup()
    table_place()
    table_pickup()
    shelf_place()

rclpy.shutdown()
