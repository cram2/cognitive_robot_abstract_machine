import io
import logging
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

import pycram.motion_executor as motion_executor_module
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.motion_executor import (
    simulated_robot_without_collision,
    simulated_robot_with_collision,
)
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    SimplePouringAction,
    WipingAction,
)
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import SetGripperAction
from pycram.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
)

from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Cup,
    Knife,
    Whisk,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection

try:
    from thesis_new.src.demo_cut_all_breads_retry import (
        CUTTING_NUM_CUTS_X,
        CUTTING_POINTER_STRIDE,
        CUTTING_SLICE_THICKNESS_M,
        CUTTING_TECHNIQUE,
        _body_name,
        _cut_object_execution_config,
        _parse_stl,
    )
    from thesis_new.src.demo_mix_all_bowls_retry import _try_mix
    from thesis_new.src.demo_wipe_all_spaces_retry import (
        _attach_sponges_for_available_arms,
        _create_target_pose_marker_publisher,
        _publish_target_pose_markers,
        _try_wipe,
    )
    from thesis_new.src.spawn_random_bowls import _parse_stl as _parse_bowl_stl
    from thesis_new.src.spawn_random_breads import (
        _pose_xyz,
        _set_uniform_scale,
        get_cut_object_config,
    )
    from thesis_new.src.tool_mounts import get_tool_mount_pose_kwargs
    from thesis_new.src.world_setup import (
        resolve_environment_name,
        resolve_robot_name,
        setup_thesis_world,
    )
except ImportError:
    from ..thesis_new.src.demo_cut_all_breads_retry import (
        CUTTING_NUM_CUTS_X,
        CUTTING_POINTER_STRIDE,
        CUTTING_SLICE_THICKNESS_M,
        CUTTING_TECHNIQUE,
        _body_name,
        _cut_object_execution_config,
        _parse_stl,
    )
    from ..thesis_new.src.demo_mix_all_bowls_retry import _try_mix
    from ..thesis_new.src.demo_wipe_all_spaces_retry import (
        _attach_sponges_for_available_arms,
        _create_target_pose_marker_publisher,
        _publish_target_pose_markers,
        _try_wipe,
    )
    from ..thesis_new.src.spawn_random_bowls import _parse_stl as _parse_bowl_stl
    from ..thesis_new.src.spawn_random_breads import (
        _pose_xyz,
        _set_uniform_scale,
        get_cut_object_config,
    )
    from ..thesis_new.src.tool_mounts import get_tool_mount_pose_kwargs
    from ..thesis_new.src.world_setup import (
        resolve_environment_name,
        resolve_robot_name,
        setup_thesis_world,
    )
from pycram.robot_plans.actions.composite.thesis_math import body_local_aabb
from demos.thesis_new.src.utils import (
    attach_available_tools,
    collect_named_targets,
    get_available_arm_tool_frames,
    get_park_arms_argument,
    publish_demo_camera_target,
    setup_experiment_runtime,
    shutdown_experiment_runtime,
)
from semantic_digital_twin.world_description.geometry import Color

HANDPICKED_PICKUP_POSES = {
    "g1": {
        "apartment": (1.7, 2.11, 0.99, 0),
        "apartment_without_walls": (1.7, 2.11, 0.99, 0),
        "kitchen": (-0.4, 0.74, 0.99, np.pi),
        "test-kitchen-chat": (-0.4, 0.74, 0.99, np.pi),
        "isr": (1, 0, 0.0, 0.0),
    },
    "stretch": {
        "apartment": (1.7, 2.11, 0.99, -np.pi / 2),
        "apartment_without_walls": (1.7, 2.11, 0.99, -np.pi / 2),
        "kitchen": (-0.4, 0.34, 0.99, np.pi),
        "test-kitchen-chat": (-0.4, 0.34, 0.99, np.pi),
        "isr": (1, 0, 0.0, 0.0),
    },
    "tiago": {
        "apartment": (1.7, 2.11, 0.99, 0),
        "apartment_without_walls": (1.7, 2.11, 0.99, 0),
        "kitchen": (-0.2, 0.34, 0.99, np.pi),
        "test-kitchen-chat": (-0.2, 0.34, 0.99, np.pi),
        "isr": (1, 0, 0.0, 0.0),
    },
    "armar7": {
        "apartment": (1.6, 2.3, 0.99, -np.pi / 2),
        "apartment_without_walls": (1.6, 2.3, 0.99, -np.pi / 2),
        "kitchen": (-0.2, 0.34, 0.99, np.pi / 2),
        "test-kitchen-chat": (-0.2, 0.34, 0.99, np.pi / 2),
        "isr": (0.8, 0, 0.0, -np.pi / 2),
    },
    "hsrb": {
        "apartment": (1.7, 2.11, 0.99, 0),
        "apartment_without_walls": (1.7, 2.11, 0.99, 0),
        "kitchen": (-0.4, 0.74, 0.99, np.pi),
        "test-kitchen-chat": (-0.4, 0.74, 0.99, np.pi),
        "isr": (1, 0, 0.0, 0.0),
    },
    "justin": {
        "apartment": (1.5, 2.11, 0.99, -np.pi / 2),
        "apartment_without_walls": (1.5, 2.11, 0.99, -np.pi / 2),
        "kitchen": (-0.2, 0.34, 0.99, np.pi),
        "test-kitchen-chat": (-0.2, 0.34, 0.99, np.pi),
        "isr": (0.4, 0, 0.0, 0.0),
    },
    "pr2": {
        "apartment": (1.5, 2.11, 0.99, 0),
        "apartment_without_walls": (1.5, 2.11, 0.99, 0),
        "kitchen": (-0.2, 0.34, 0.99, np.pi),
        "test-kitchen-chat": (-0.2, 0.34, 0.99, np.pi),
        "isr": (1, 0, 0.0, 0.0),
    },
}
HANDPICKED_SPAWN_POSES = {
    "apartment": (2.48, 2.11, 0.99, -np.pi / 2),
    "apartment_without_walls": (2.48, 2.11, 0.99, -np.pi / 2),
    "kitchen": (-0.8, 0.74, 0.885, np.pi / 2),
    "test-kitchen-chat": (-0.8, 0.74, 0.885, np.pi / 2),
    "isr": (1.48, -0.12, 0.80, -np.pi / 2),
}
HANDPICKED_CAMERA_POSES = {
    "apartment": (2.83, 2.11, 0.99, 0),
    "apartment_without_walls": (2.83, 2.11, 0.99, 0),
    "kitchen": (-1.6, 0.74, 1, np.pi),
    "test-kitchen-chat": (-0.8, 1.09, 0.885, 0),
    "isr": (1.88, -0.47, 0.99, 0),
}
CUTTING_BOARD_COLOR = Color(R=0.50, G=0.36, B=0.22)
MIXING_BOWL_COLOR = Color(R=0.78, G=0.80, B=0.86)


def _suppress_demo_noise():
    logging.getLogger("pycram.motion_executor").setLevel(logging.ERROR)
    logging.getLogger("pycram.robot_plans.actions.base").setLevel(logging.WARNING)
    motion_executor_module.DEBUG_PROFILE_MOTION_EXECUTOR = False


def _perform_attempt_quietly(fn):
    logger = logging.getLogger("motion_executor.py")
    previous_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        return fn()
    finally:
        logger.setLevel(previous_level)


def _try_cut(
    context,
    target,
    pickup_pose,
    arm,
    tool,
    *,
    cutting_technique,
    num_cuts_x,
):
    with simulated_robot_without_collision:
        sequential(
            [
                ParkArmsAction(
                    get_park_arms_argument(context.world),
                    joint_state=StaticJointState.PARKTOOL,
                ),
            ],
            context,
        ).perform()

    with simulated_robot_without_collision:
        sequential(
            [
                ParkArmsAction(
                    get_park_arms_argument(context.world),
                    joint_state=StaticJointState.PARKTOOL,
                ),
            ],
            context,
        ).perform()
        sequential([MoveTorsoAction(TorsoState.HIGH)], context).perform()
        sequential(
            [NavigateAction(pickup_pose, True, teleport=True)], context
        ).perform()

    with simulated_robot_with_collision:
        sequential(
            [
                CuttingAction(
                    container=target,
                    arm=arm,
                    tool=tool,
                    technique=cutting_technique,
                    clear_viz=True,
                    pointer_stride=CUTTING_POINTER_STRIDE,
                    num_cuts_x=num_cuts_x,
                    slice_thickness=CUTTING_SLICE_THICKNESS_M,
                ),
                ParkArmsAction(get_park_arms_argument(context.world)),
            ],
            context,
        ).perform()


def _try_surface_contact(
    context,
    target_pose,
    pickup_pose,
    arm,
    tool,
    *,
    technique,
    target_container=None,
):
    with simulated_robot_without_collision:
        sequential(
            [
                ParkArmsAction(
                    get_park_arms_argument(context.world),
                    joint_state=StaticJointState.PARKTOOL,
                ),
            ],
            context,
        ).perform()
        sequential([MoveTorsoAction(TorsoState.HIGH)], context).perform()
        sequential(
            [NavigateAction(pickup_pose, True, teleport=True)], context
        ).perform()

    with simulated_robot_with_collision:
        sequential(
            [
                WipingAction(
                    container=target_container,
                    target_pose=target_pose,
                    arm=arm,
                    tool=tool,
                    technique=technique,
                    clear_viz=True,
                ),
                ParkArmsAction(get_park_arms_argument(context.world)),
            ],
            context,
        ).perform()


def _try_pour(
    context,
    source_container,
    target_container,
    pickup_pose,
    arm,
    tool,
):
    with simulated_robot_without_collision:
        sequential(
            [
                ParkArmsAction(
                    get_park_arms_argument(context.world),
                    joint_state=StaticJointState.PARKTOOL,
                ),
            ],
            context,
        ).perform()
        sequential([MoveTorsoAction(TorsoState.HIGH)], context).perform()
        sequential(
            [NavigateAction(pickup_pose, True, teleport=True)], context
        ).perform()

    with simulated_robot_with_collision:
        sequential(
            [
                SimplePouringAction(
                    object_designator=target_container,
                    arm=arm,
                ),
                ParkArmsAction(get_park_arms_argument(context.world)),
            ],
            context,
        ).perform()


def _spawn_single_object(
    *,
    world,
    object_kind,
    spawn_position,
    spawn_yaw,
    spawn_scale,
):
    object_cfg = get_cut_object_config(object_kind)
    object_name = f"{object_cfg['object_name_prefix']}_0001"
    x_world, y_world, z_world = [float(value) for value in spawn_position]
    spawned = _parse_stl(*object_cfg["mesh_parts"])
    spawned.root.name.name = object_name
    _set_uniform_scale(
        spawned,
        (float(spawn_scale), float(spawn_scale), float(spawn_scale)),
        color=object_cfg["object_color"],
    )
    world_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=x_world,
        y=y_world,
        z=z_world,
        yaw=float(spawn_yaw),
        reference_frame=world.root,
    )
    world.merge_world_at_pose(spawned, world_pose)
    world.update_forward_kinematics()
    return object_name, _pose_xyz(world_pose)


def _spawn_single_mixing_container(
    *,
    world,
    object_kind,
    spawn_position,
    spawn_yaw,
    spawn_scale,
):
    container_kind = str(object_kind).strip().lower()
    mesh_parts = (
        ("pycram_object_gap_demo", "pot_1.stl")
        if container_kind == "pot"
        else ("objects", "bowl.stl")
    )
    object_name = "bowl_0001"
    x_world, y_world, z_world = [float(value) for value in spawn_position]
    spawned = _parse_bowl_stl(*mesh_parts)
    spawned.root.name.name = object_name
    _set_uniform_scale(
        spawned,
        (float(spawn_scale), float(spawn_scale), float(spawn_scale)),
        color=MIXING_BOWL_COLOR,
    )
    world_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=x_world,
        y=y_world,
        z=z_world,
        yaw=float(spawn_yaw),
        reference_frame=world.root,
    )
    world.merge_world_at_pose(spawned, world_pose)
    world.update_forward_kinematics()
    return object_name, _pose_xyz(world_pose)


def _attach_single_hand_tool(
    *,
    world,
    parse_stl_fn,
    mesh_parts,
    tool_name,
    arm,
    pose_kwargs,
    tool_cls,
):
    available_frames = dict(get_available_arm_tool_frames(world))
    if arm not in available_frames:
        raise ValueError(f"Requested arm '{arm}' is not available on this robot.")

    tool_world = parse_stl_fn(*mesh_parts)
    tool_world.root.name.name = tool_name
    tip = available_frames[arm]

    with world.modify_world():
        connection = FixedConnection(
            parent=tip,
            child=tool_world.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=tip,
                **pose_kwargs,
            ),
        )
        world.merge_world(tool_world, connection)

    return tool_cls(root=connection.child)


def _single_wipe_target_data(world, spawn_position, spawn_yaw):
    x_world, y_world, z_world = [float(value) for value in spawn_position]
    world_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=x_world,
        y=y_world,
        z=z_world,
        yaw=float(spawn_yaw) + -np.pi / 2,
        reference_frame=world.root,
    )
    return {
        "bowl_name": "wipe_target_0001",
        "surface_name": "single_target",
        "scale": 1.0,
        "pose_xyz": _pose_xyz(world_pose),
        "world_pose": world_pose,
    }


def _supported_handpicked_pose_targets(pose_map):
    targets = []
    for robot_name, environment_poses in sorted(pose_map.items()):
        if isinstance(environment_poses, dict):
            environments = ", ".join(sorted(environment_poses))
            targets.append(f"{robot_name}: {environments}")
    return "; ".join(targets)


def _resolve_pickup_pose(
    world, robot_name, environment_name, pickup_position=None, pickup_yaw=None
):
    if pickup_position is not None:
        x, y, z = [float(value) for value in pickup_position]
        yaw = 0.0 if pickup_yaw is None else float(pickup_yaw)
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            reference_frame=world.root,
        )

    normalized_robot = resolve_robot_name(robot_name)
    normalized_environment = resolve_environment_name(environment_name)
    robot_pose_map = HANDPICKED_PICKUP_POSES.get(normalized_robot)
    if robot_pose_map is None or normalized_environment not in robot_pose_map:
        supported = _supported_handpicked_pose_targets(HANDPICKED_PICKUP_POSES)
        raise ValueError(
            "No handpicked pickup pose configured for "
            f"robot '{normalized_robot}' in environment "
            f"'{normalized_environment}'. Supported: {supported}. "
            "Set PICKUP_POSITION and PICKUP_YAW in main.py for this environment."
        )

    x, y, z, yaw = robot_pose_map[normalized_environment]
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        x=float(x),
        y=float(y),
        z=float(z),
        yaw=float(yaw),
        reference_frame=world.root,
    )


def _resolve_spawn_pose(
    world, robot_name, environment_name, spawn_position=None, spawn_yaw=None
):
    if spawn_position is not None:
        x, y, z = [float(value) for value in spawn_position]
        yaw = 0.0 if spawn_yaw is None else float(spawn_yaw)
        return (x, y, z), yaw

    normalized_environment = resolve_environment_name(environment_name)
    if normalized_environment not in HANDPICKED_SPAWN_POSES:
        supported = ", ".join(sorted(HANDPICKED_SPAWN_POSES))
        raise ValueError(
            "No handpicked spawn pose configured for "
            f"'{normalized_environment}'. Supported: {supported}. "
            "Set SPAWN_POSITION and SPAWN_YAW in main.py for this environment."
        )

    pose = HANDPICKED_SPAWN_POSES[normalized_environment]
    if len(pose) == 4:
        x, y, z, yaw = pose
        if spawn_yaw is not None:
            yaw = float(spawn_yaw)
    else:
        raise ValueError(
            f"Invalid handpicked spawn pose for '{normalized_environment}': {pose}"
        )

    return (float(x), float(y), float(z)), float(yaw)


def _resolve_camera_pose(world, environment_name):
    normalized_environment = resolve_environment_name(environment_name)
    if normalized_environment not in HANDPICKED_CAMERA_POSES:
        supported = ", ".join(sorted(HANDPICKED_CAMERA_POSES))
        raise ValueError(
            "No handpicked camera pose configured for "
            f"'{normalized_environment}'. Supported: {supported}."
        )

    x, y, z, yaw = HANDPICKED_CAMERA_POSES[normalized_environment]
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        x=float(x),
        y=float(y),
        z=float(z),
        yaw=float(yaw),
        reference_frame=world.root,
    )


def _spawn_cutting_board_under_object(
    *,
    world,
    object_kind,
    spawn_position,
    spawn_yaw,
    spawn_scale,
    board_scale=1.0,
):
    object_cfg = get_cut_object_config(object_kind)
    object_preview = _parse_stl(*object_cfg["mesh_parts"])
    _set_uniform_scale(
        object_preview,
        (float(spawn_scale), float(spawn_scale), float(spawn_scale)),
        color=object_cfg["object_color"],
    )
    object_mins, _ = body_local_aabb(
        object_preview.root, use_visual=False, apply_shape_scale=True
    )

    board = _parse_stl("pycram_object_gap_demo", "board.stl")
    board.root.name.name = "cutting_board_0001"
    _set_uniform_scale(
        board,
        (float(board_scale), float(board_scale), float(board_scale)),
        color=CUTTING_BOARD_COLOR,
    )
    _, board_maxs = body_local_aabb(
        board.root, use_visual=False, apply_shape_scale=True
    )

    x_world, y_world, z_world = [float(value) for value in spawn_position]
    board_z = z_world + float(object_mins[2]) - float(board_maxs[2]) - 0.002
    board_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=x_world,
        y=y_world,
        z=board_z,
        yaw=float(spawn_yaw),
        reference_frame=world.root,
    )
    world.merge_world_at_pose(board, board_pose)
    world.update_forward_kinematics()
    return _pose_xyz(board_pose)


def run_single_object_cut_demo(
    *,
    robot_name=None,
    environment_name=None,
    object_kind="bread",
    spawn_position=None,
    spawn_yaw=None,
    spawn_scale=1.0,
    pickup_position=None,
    pickup_yaw=None,
    quiet_logs=True,
):
    if quiet_logs:
        _suppress_demo_noise()
    parse_output = io.StringIO()
    with redirect_stdout(parse_output), redirect_stderr(parse_output):
        world = setup_thesis_world(
            robot_name=robot_name,
            environment_name=environment_name,
        )
    spawn_position, spawn_yaw = _resolve_spawn_pose(
        world,
        robot_name,
        environment_name,
        spawn_position=spawn_position,
        spawn_yaw=spawn_yaw,
    )
    board_xyz = _spawn_cutting_board_under_object(
        world=world,
        object_kind=object_kind,
        spawn_position=spawn_position,
        spawn_yaw=spawn_yaw,
        spawn_scale=spawn_scale,
    )
    object_name, world_xyz = _spawn_single_object(
        world=world,
        object_kind=object_kind,
        spawn_position=spawn_position,
        spawn_yaw=spawn_yaw,
        spawn_scale=spawn_scale,
    )

    node = setup_experiment_runtime(
        world=world,
        node_name="pycram_single_object_cut_demo",
    )

    try:
        resolved_robot_name = resolve_robot_name(robot_name)
        arm_tools = attach_available_tools(
            world,
            _parse_stl,
            mesh_parts=("pycram_object_gap_demo", "big-knife.stl"),
            right_name="knife_right",
            left_name="knife_left",
            right_pose_kwargs=get_tool_mount_pose_kwargs(
                "cut", resolved_robot_name, Arms.RIGHT
            ),
            left_pose_kwargs=get_tool_mount_pose_kwargs(
                "cut", resolved_robot_name, Arms.LEFT
            ),
            tool_cls=Knife,
        )
        targets = collect_named_targets(
            world, f"{get_cut_object_config(object_kind)['object_name_prefix']}_"
        )

        target = targets[0]
        publish_demo_camera_target(
            node,
            world,
            lambda: target.global_pose,
            camera_pose_fn=lambda: _resolve_camera_pose(world, environment_name),
        )
        cut_cfg = _cut_object_execution_config(object_kind)
        context = Context.from_world(world)
        context.ros_node = node

        with simulated_robot_without_collision:
            sequential(
                [SetGripperAction(Arms.BOTH, GripperState.CLOSE)],
                context,
            ).perform()

        pickup_pose = _resolve_pickup_pose(
            world,
            robot_name,
            environment_name,
            pickup_position=pickup_position,
            pickup_yaw=pickup_yaw,
        )
        pickup_xyz = _pose_xyz(pickup_pose)

        last_error = None
        for arm, tool in arm_tools:
            try:
                _perform_attempt_quietly(
                    lambda: _try_cut(
                        context,
                        target,
                        pickup_pose,
                        arm,
                        tool,
                        cutting_technique=cut_cfg.get("technique", CUTTING_TECHNIQUE),
                        num_cuts_x=cut_cfg.get("num_cuts_x", CUTTING_NUM_CUTS_X),
                    )
                )
                return
            except Exception as exc:
                last_error = exc
                print(f"The given robot is not suitable for this environment")

        if last_error is not None:
            raise last_error
    finally:
        shutdown_experiment_runtime(node)


def run_single_object_mix_demo(
    *,
    robot_name=None,
    environment_name=None,
    object_kind="bowl",
    spawn_position=None,
    spawn_yaw=None,
    spawn_scale=1.0,
    pickup_position=None,
    pickup_yaw=None,
    quiet_logs=True,
):
    if quiet_logs:
        _suppress_demo_noise()
    parse_output = io.StringIO()
    with redirect_stdout(parse_output), redirect_stderr(parse_output):
        world = setup_thesis_world(
            robot_name=robot_name,
            environment_name=environment_name,
        )
    spawn_position, spawn_yaw = _resolve_spawn_pose(
        world,
        robot_name,
        environment_name,
        spawn_position=spawn_position,
        spawn_yaw=spawn_yaw,
    )
    object_name, world_xyz = _spawn_single_mixing_container(
        world=world,
        object_kind=object_kind,
        spawn_position=spawn_position,
        spawn_yaw=spawn_yaw,
        spawn_scale=spawn_scale,
    )

    node = setup_experiment_runtime(
        world=world,
        node_name="pycram_single_object_mix_demo",
    )

    try:
        resolved_robot_name = resolve_robot_name(robot_name)
        arm_tools = attach_available_tools(
            world,
            _parse_stl,
            mesh_parts=("pycram_object_gap_demo", "whisk.stl"),
            right_name="whisk_right",
            left_name="whisk_left",
            right_pose_kwargs=get_tool_mount_pose_kwargs(
                "mix", resolved_robot_name, Arms.RIGHT
            ),
            left_pose_kwargs=get_tool_mount_pose_kwargs(
                "mix", resolved_robot_name, Arms.LEFT
            ),
            tool_cls=Whisk,
        )
        targets = collect_named_targets(world, "bowl_")
        target = targets[0]
        publish_demo_camera_target(
            node,
            world,
            lambda: target.global_pose,
            camera_pose_fn=lambda: _resolve_camera_pose(world, environment_name),
        )
        context = Context.from_world(world)
        context.ros_node = node

        with simulated_robot_without_collision:
            sequential(
                [SetGripperAction(Arms.BOTH, GripperState.CLOSE)],
                context,
            ).perform()

        pickup_pose = _resolve_pickup_pose(
            world,
            robot_name,
            environment_name,
            pickup_position=pickup_position,
            pickup_yaw=pickup_yaw,
        )

        last_error = None
        for arm, tool in arm_tools:
            try:
                _perform_attempt_quietly(
                    lambda: _try_mix(
                        context,
                        target,
                        pickup_pose,
                        arm,
                        tool,
                        environment_name=environment_name,
                    )
                )
                return
            except Exception as exc:
                last_error = exc
                print("The given robot is not suitable for this environment")

        if last_error is not None:
            raise last_error
    finally:
        shutdown_experiment_runtime(node)


def run_single_object_pour_demo(
    *,
    robot_name=None,
    environment_name=None,
    object_kind="bowl",
    spawn_position=None,
    spawn_yaw=None,
    spawn_scale=1.0,
    pickup_position=None,
    pickup_yaw=None,
    quiet_logs=True,
):
    if quiet_logs:
        _suppress_demo_noise()
    parse_output = io.StringIO()
    with redirect_stdout(parse_output), redirect_stderr(parse_output):
        world = setup_thesis_world(
            robot_name=robot_name,
            environment_name=environment_name,
        )
    spawn_position, spawn_yaw = _resolve_spawn_pose(
        world,
        robot_name,
        environment_name,
        spawn_position=spawn_position,
        spawn_yaw=spawn_yaw,
    )
    _spawn_single_mixing_container(
        world=world,
        object_kind=object_kind,
        spawn_position=spawn_position,
        spawn_yaw=spawn_yaw,
        spawn_scale=spawn_scale,
    )

    node = setup_experiment_runtime(
        world=world,
        node_name="pycram_single_object_pour_demo",
    )

    try:
        resolved_robot_name = resolve_robot_name(robot_name)
        cup_tool = _attach_single_hand_tool(
            world=world,
            parse_stl_fn=_parse_bowl_stl,
            mesh_parts=("objects", "jeroen_cup.stl"),
            tool_name="jeroen_cup_right",
            arm=Arms.RIGHT,
            pose_kwargs=get_tool_mount_pose_kwargs(
                "pour", resolved_robot_name, Arms.RIGHT
            ),
            tool_cls=Cup,
        )
        bowl_body = world.get_body_by_name("bowl_0001")
        bowl_target = Bowl(root=bowl_body)
        publish_demo_camera_target(
            node,
            world,
            lambda: bowl_body.global_pose,
            camera_pose_fn=lambda: _resolve_camera_pose(world, environment_name),
        )
        context = Context.from_world(world)
        context.ros_node = node

        with simulated_robot_without_collision:
            sequential(
                [SetGripperAction(Arms.BOTH, GripperState.CLOSE)],
                context,
            ).perform()

        pickup_pose = _resolve_pickup_pose(
            world,
            robot_name,
            environment_name,
            pickup_position=pickup_position,
            pickup_yaw=pickup_yaw,
        )

        _perform_attempt_quietly(
            lambda: _try_pour(
                context,
                cup_tool.root,
                bowl_target.root,
                pickup_pose,
                Arms.RIGHT,
                cup_tool,
            )
        )
    finally:
        shutdown_experiment_runtime(node)


def run_single_object_wipe_demo(
    *,
    robot_name=None,
    environment_name=None,
    object_kind="wipe",
    technique="wipe",
    spawn_position=None,
    spawn_yaw=None,
    spawn_scale=1.0,
    pickup_position=None,
    pickup_yaw=None,
    quiet_logs=True,
):
    if quiet_logs:
        _suppress_demo_noise()
    parse_output = io.StringIO()
    with redirect_stdout(parse_output), redirect_stderr(parse_output):
        world = setup_thesis_world(
            robot_name=robot_name,
            environment_name=environment_name,
        )
    spawn_position, spawn_yaw = _resolve_spawn_pose(
        world,
        robot_name,
        environment_name,
        spawn_position=spawn_position,
        spawn_yaw=spawn_yaw,
    )
    target_data = _single_wipe_target_data(world, spawn_position, spawn_yaw)

    node = setup_experiment_runtime(
        world=world,
        node_name="pycram_single_object_wipe_demo",
    )

    try:
        resolved_technique = str(technique).strip().lower()
        if resolved_technique == "spread":
            resolved_robot_name = resolve_robot_name(robot_name)
            slice_preview = _parse_stl("pycram_object_gap_demo", "slice_of_bread.stl")
            _set_uniform_scale(
                slice_preview,
                (float(spawn_scale), float(spawn_scale), float(spawn_scale)),
                color=Color(R=0.76, G=0.60, B=0.42),
            )
            slice_mins, _ = body_local_aabb(
                slice_preview.root, use_visual=False, apply_shape_scale=True
            )

            board = _parse_stl("pycram_object_gap_demo", "board.stl")
            board.root.name.name = "cutting_board_0001"
            _set_uniform_scale(
                board,
                (1.0, 1.0, 1.0),
                color=CUTTING_BOARD_COLOR,
            )
            _, board_maxs = body_local_aabb(
                board.root, use_visual=False, apply_shape_scale=True
            )

            x_world, y_world, z_world = [float(value) for value in spawn_position]
            board_z = z_world + float(slice_mins[2]) - float(board_maxs[2]) - 0.002
            board_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=x_world,
                y=y_world,
                z=board_z,
                yaw=float(spawn_yaw),
                reference_frame=world.root,
            )
            world.merge_world_at_pose(board, board_pose)
            world.update_forward_kinematics()

            slice_world = _parse_stl("pycram_object_gap_demo", "slice_of_bread.stl")
            slice_world.root.name.name = "slice_of_bread_0001"
            _set_uniform_scale(
                slice_world,
                (float(spawn_scale), float(spawn_scale), float(spawn_scale)),
                color=Color(R=0.76, G=0.60, B=0.42),
            )
            slice_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=x_world,
                y=y_world,
                z=z_world,
                yaw=float(spawn_yaw),
                reference_frame=world.root,
            )
            world.merge_world_at_pose(slice_world, slice_pose)
            world.update_forward_kinematics()

            arm_tools = attach_available_tools(
                world,
                _parse_stl,
                mesh_parts=("pycram_object_gap_demo", "butter_knife.stl"),
                right_name="knife_right",
                left_name="knife_left",
                right_pose_kwargs=get_tool_mount_pose_kwargs(
                    "cut", resolved_robot_name, Arms.RIGHT
                ),
                left_pose_kwargs=get_tool_mount_pose_kwargs(
                    "cut", resolved_robot_name, Arms.LEFT
                ),
                tool_cls=Knife,
            )
            target_container = world.get_body_by_name("slice_of_bread_0001")
            right_tool = next(tool for arm, tool in arm_tools if arm == Arms.RIGHT)
            publish_demo_camera_target(
                node,
                world,
                lambda: target_container.global_pose,
                camera_pose_fn=lambda: _resolve_camera_pose(world, environment_name),
            )
        else:
            target_marker_pub = _create_target_pose_marker_publisher(node)
            _publish_target_pose_markers(node, target_marker_pub, world, [target_data])
            publish_demo_camera_target(
                node,
                world,
                lambda: target_data["world_pose"],
                camera_pose_fn=lambda: _resolve_camera_pose(world, environment_name),
            )
            arm_tools = _attach_sponges_for_available_arms(world)
            target_container = None
        context = Context.from_world(world)
        context.ros_node = node

        with simulated_robot_without_collision:
            sequential(
                [SetGripperAction(Arms.BOTH, GripperState.CLOSE)],
                context,
            ).perform()

        pickup_pose = _resolve_pickup_pose(
            world,
            robot_name,
            environment_name,
            pickup_position=pickup_position,
            pickup_yaw=pickup_yaw,
        )

        if resolved_technique == "spread":
            _perform_attempt_quietly(
                lambda: _try_surface_contact(
                    context,
                    None,
                    pickup_pose,
                    Arms.RIGHT,
                    right_tool,
                    technique="spread",
                    target_container=target_container,
                )
            )
            return

        last_error = None
        for arm, tool in arm_tools:
            try:
                _perform_attempt_quietly(
                    lambda: _try_wipe(
                        context,
                        target_data["world_pose"],
                        pickup_pose,
                        arm,
                        tool,
                        environment_name=environment_name,
                    )
                )
                return
            except Exception as exc:
                last_error = exc
                print("The given robot is not suitable for this environment")

        if last_error is not None:
            raise last_error
    finally:
        shutdown_experiment_runtime(node)


def run_single_object_spread_demo(
    **kwargs,
):
    return run_single_object_wipe_demo(
        technique="spread",
        object_kind="bread",
        **kwargs,
    )


def run_single_object_demo(*, action="cut", technique=None, **kwargs):
    normalized_action = str(action).strip().lower()
    if normalized_action == "cut":
        return run_single_object_cut_demo(**kwargs)
    if normalized_action == "mix":
        return run_single_object_mix_demo(**kwargs)
    if normalized_action == "pour":
        return run_single_object_pour_demo(**kwargs)
    if normalized_action == "wipe":
        return run_single_object_wipe_demo(
            technique="wipe" if technique is None else technique,
            **kwargs,
        )
    if normalized_action == "spread":
        return run_single_object_wipe_demo(technique="spread", **kwargs)
    raise ValueError(
        f"Unsupported single-object action '{action}'. Supported: cut, mix, pour, wipe, spread"
    )
