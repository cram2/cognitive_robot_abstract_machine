import os
import time

import numpy as np
from geometry_msgs.msg import Point as RosPoint
from geometry_msgs.msg import Pose as RosPose
from rclpy.duration import Duration as RclpyDuration
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

try:
    from demos.thesis.simulation_setup import BoxSpec, add_box
    from .spawn_random_bowls import sample_random_bowl_poses
    from .tool_mounts import get_tool_mount_pose_kwargs
    from .world_setup import resolve_environment_name
except ModuleNotFoundError:
    from ...thesis.simulation_setup import BoxSpec, add_box
    from .spawn_random_bowls import sample_random_bowl_poses
    from .tool_mounts import get_tool_mount_pose_kwargs
    from .world_setup import resolve_environment_name
from .utils.demo_utils import (
    build_navigation_costmaps,
    get_available_arm_tool_frames,
    get_park_arms_argument,
    get_primary_robot_name,
    resolve_navigation_target_for_environment,
    setup_experiment_runtime,
    shutdown_experiment_runtime,
    update_navigation_costmap_debug_publishers,
    commit_plan_to_db,
)
from .utils.experiment_logging import (
    BASE_RESULT_FIELDNAMES,
    append_csv_row,
    body_name as _body_name,
    build_base_result_row,
    format_attempt_error as _format_attempt_error,
    initialize_csv,
    is_collision_like_failure as _is_collision_like_failure,
    new_run_id,
    robot_name as _robot_name,
    tool_name as _tool_name,
)
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.locations.locations import CostmapLocation
from pycram.motion_executor import (
    simulated_robot_with_collision,
    simulated_robot_without_collision,
)
from pycram.orm.ormatic_interface import Base
from pycram.orm.utils import pycram_sessionmaker
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.composite.tool_based import WipingAction
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from pycram.tf_transformations import euler_from_quaternion
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Sponge
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color

ACTIVE_TARGET_COLOR = Color(R=0.52, G=0.82, B=0.98)
FAILED_TARGET_COLOR = Color(R=0.95, G=0.20, B=0.20)
SUCCESS_TARGET_COLOR = Color(R=0.62, G=0.92, B=0.62)
DEFAULT_TARGET_COLOR = Color(R=0.78, G=0.80, B=0.86)
RECORDS_DIR = os.path.join(os.path.dirname(__file__), "../records")
RESULTS_CSV_PATH = os.environ.get(
    "THESIS_WIPE_RESULTS_CSV_PATH",
    os.path.join(RECORDS_DIR, "wipe_all_spaces_results.csv"),
)
EXPERIMENT_CONDITION = "full_system"
BASELINE_NAME = "task_knowledge+htn+constraint_planning"
TASK_NAME = "space_wiping"
POINTER_STRIDE = 3
MAX_WIPE_TARGET_HEIGHT_M = 1.30
TARGET_MARKER_TOPIC = "/pycram/wipe_targets"
VERTICAL_TARGET_AXIS_LENGTH_M = 0.18
VERTICAL_TARGET_AXIS_WIDTH_M = 0.012
DEBUG_PROFILE_WIPING = True
session = None


def _timed(label, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    if DEBUG_PROFILE_WIPING:
        print(f"[profile] {label}: {elapsed:.3f}s")
    return result, elapsed


def _update_costmap_debug_publishers(node, robot, world, target_pose, publishers):
    occupancy, ring, final_map = build_navigation_costmaps(robot, world, target_pose)
    return update_navigation_costmap_debug_publishers(
        node,
        world,
        publishers,
        occupancy,
        ring,
        final_map,
        namespace_prefix="wiping",
    )


def _record_space_result(
    results,
    target_name,
    robot_name,
    outcome,
    succeeded_arm,
    tool_name,
    phase,
    failures,
    *,
    spawn_pose_xyz,
    environment_name,
    task_instance_id,
    experiment_condition,
    baseline_name,
    task_name,
    seed,
    world_name,
    run_id,
    knowledge_query_success,
    knowledge_query_error,
    knowledge_prior_task,
    knowledge_cutting_tool,
    knowledge_cutting_position,
    knowledge_repetition,
    required_prerequisite,
    prerequisite_source,
    prerequisite_satisfied_initially,
    autonomous_execution_feasible,
    feasibility_reason,
    robot_decision,
    decision_reason,
    assistance_requested,
    assistance_type,
    assistance_completed,
    task_blocked_by_prerequisite,
    task_resumed_after_assistance,
    final_success,
    total_attempts,
    retry_count,
    collision_failure_count,
    recovery_used,
    recovery_success,
    perturbation_applied,
    perturbation_type,
    execution_time_s,
    geometry_binding,
):
    return build_base_result_row(
        results,
        robot_name,
        outcome,
        succeeded_arm,
        tool_name,
        phase,
        failures,
        target_name=target_name,
        spawn_pose_xyz=spawn_pose_xyz,
        environment_name=environment_name,
        task_name=task_name,
        run_id=run_id,
        task_instance_id=task_instance_id,
        seed=seed if seed is not None else "",
        world_name=world_name,
        experiment_condition=experiment_condition,
        baseline_name=baseline_name,
        knowledge_query_success=knowledge_query_success,
        knowledge_query_error=knowledge_query_error,
        knowledge_prior_task=knowledge_prior_task,
        knowledge_cutting_tool=knowledge_cutting_tool,
        knowledge_cutting_position=knowledge_cutting_position,
        knowledge_repetition=knowledge_repetition,
        required_prerequisite=required_prerequisite,
        prerequisite_source=prerequisite_source,
        prerequisite_satisfied_initially=prerequisite_satisfied_initially,
        autonomous_execution_feasible=autonomous_execution_feasible,
        feasibility_reason=feasibility_reason,
        robot_decision=robot_decision,
        decision_reason=decision_reason,
        assistance_requested=assistance_requested,
        assistance_type=assistance_type,
        assistance_completed=assistance_completed,
        task_blocked_by_prerequisite=task_blocked_by_prerequisite,
        task_resumed_after_assistance=task_resumed_after_assistance,
        final_success=final_success,
        total_attempts=total_attempts,
        retry_count=retry_count,
        collision_failure_count=collision_failure_count,
        recovery_used=recovery_used,
        recovery_success=recovery_success,
        perturbation_applied=perturbation_applied,
        perturbation_type=perturbation_type,
        **geometry_binding,
        execution_time_s=round(execution_time_s, 4),
    )


def _results_csv_fieldnames():
    return [
        "target_name",
        "spawn_pose_xyz",
        "environment_name",
        *BASE_RESULT_FIELDNAMES,
    ]


def _create_target_pose_marker_publisher(node):
    qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    return node.create_publisher(MarkerArray, TARGET_MARKER_TOPIC, qos)


def _marker_color_for_target(
    target_name,
    active_target_name,
    failed_target_names,
    successful_target_names,
):
    if target_name == active_target_name:
        return ACTIVE_TARGET_COLOR
    if target_name in failed_target_names:
        return FAILED_TARGET_COLOR
    if target_name in successful_target_names:
        return SUCCESS_TARGET_COLOR
    return DEFAULT_TARGET_COLOR


def _ros_point_from_xyz(xyz):
    point = RosPoint()
    point.x = float(xyz[0])
    point.y = float(xyz[1])
    point.z = float(xyz[2])
    return point


def _rgba(r, g, b, a=1.0):
    color = ColorRGBA()
    color.r = float(r)
    color.g = float(g)
    color.b = float(b)
    color.a = float(a)
    return color


def _make_vertical_pose_axes_marker(node, world_pose, marker_id, frame_id):
    position = np.asarray(world_pose.to_position().to_np(), dtype=float).reshape(-1)[:3]
    rotation = np.asarray(world_pose.to_rotation_matrix().to_np(), dtype=float)[:3, :3]
    axis_colors = (
        _rgba(1.0, 0.1, 0.1, 1.0),
        _rgba(0.1, 0.9, 0.1, 1.0),
        _rgba(0.1, 0.35, 1.0, 1.0),
    )

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.ns = "wipe_target_pose_axes"
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = VERTICAL_TARGET_AXIS_WIDTH_M
    marker.lifetime = RclpyDuration(seconds=0.0).to_msg()
    marker.frame_locked = False

    for axis_index, axis_color in enumerate(axis_colors):
        end = position + rotation[:, axis_index] * VERTICAL_TARGET_AXIS_LENGTH_M
        marker.points.append(_ros_point_from_xyz(position))
        marker.points.append(_ros_point_from_xyz(end))
        marker.colors.append(axis_color)
        marker.colors.append(axis_color)

    return marker


def _publish_target_pose_markers(
    node,
    publisher,
    world,
    sampled_targets,
    *,
    active_target_name=None,
    failed_target_names=None,
    successful_target_names=None,
):
    failed_target_names = failed_target_names or set()
    successful_target_names = successful_target_names or set()
    frame_id = str(world.root.name)
    marker_array = MarkerArray()

    clear = Marker()
    clear.action = Marker.DELETEALL
    marker_array.markers.append(clear)

    vertical_marker_count = 0
    for idx, target_data in enumerate(sampled_targets):
        world_pose = target_data["world_pose"]
        position = world_pose.to_position()
        orientation = world_pose.to_quaternion()
        color = _marker_color_for_target(
            target_data["bowl_name"],
            active_target_name,
            failed_target_names,
            successful_target_names,
        )
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.ns = "wipe_targets"
        marker.id = idx
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = RosPose()
        marker.pose.position.x = float(position.x)
        marker.pose.position.y = float(position.y)
        marker.pose.position.z = float(position.z)
        marker.pose.orientation.x = float(orientation.x)
        marker.pose.orientation.y = float(orientation.y)
        marker.pose.orientation.z = float(orientation.z)
        marker.pose.orientation.w = float(orientation.w)
        marker.scale.x = 0.10
        marker.scale.y = 0.10
        marker.scale.z = 0.005
        marker.color.r = color.R
        marker.color.g = color.G
        marker.color.b = color.B
        marker.color.a = 0.85
        marker.lifetime = RclpyDuration(seconds=0.0).to_msg()
        marker.frame_locked = False
        marker_array.markers.append(marker)

        if _is_vertical_wipe_pose(world_pose):
            marker_array.markers.append(
                _make_vertical_pose_axes_marker(
                    node,
                    world_pose,
                    10_000 + idx,
                    frame_id,
                )
            )
            vertical_marker_count += 1

    publisher.publish(marker_array)
    print(
        f"[viz] published {len(sampled_targets)} wipe target markers "
        f"and {vertical_marker_count} vertical pose axes on {TARGET_MARKER_TOPIC}"
    )


def _attach_sponges_for_available_arms(world):
    arm_frames = get_available_arm_tool_frames(world)
    tools_by_arm = {}
    robot_name = get_primary_robot_name(world)

    with world.modify_world():
        for arm, tool_frame in arm_frames:
            sponge_name = "sponge_right" if arm == Arms.RIGHT else "sponge_left"
            sponge = add_box(
                world,
                BoxSpec(name=sponge_name, scale_xyz=(0.05, 0.05, 0.05)),
                tf_frame="/map",
                color=Color(R=1, G=1, B=0),
            )
            world.add_kinematic_structure_entity(sponge)
            world.add_connection(
                FixedConnection(
                    parent=tool_frame,
                    child=sponge,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        reference_frame=tool_frame,
                        **get_tool_mount_pose_kwargs("wipe", robot_name, arm),
                    ),
                )
            )
            tools_by_arm[arm] = Sponge(root=sponge)

    return [(arm, tools_by_arm[arm]) for arm, _ in arm_frames]


def _is_costmap_merge_error(exc):
    return (
        isinstance(exc, ValueError)
        and "To merge locations" in str(exc)
        and "must be equal" in str(exc)
    )


def _filter_wipe_targets_by_height(sampled_targets):
    kept = []
    discarded = []
    for target_data in sampled_targets:
        try:
            z = float(target_data["world_pose"].to_position().z)
        except Exception:
            kept.append(target_data)
            continue
        if z > MAX_WIPE_TARGET_HEIGHT_M:
            discarded.append((target_data.get("bowl_name", "unknown"), z))
            continue
        kept.append(target_data)
    return kept, discarded


def _try_wipe(context, target_pose, pickup_pose, arm, tool, *, environment_name=None):
    with simulated_robot_without_collision:
        _, _ = _timed(
            "wipe/reset_pose",
            lambda: sequential(
                [
                    ParkArmsAction(get_park_arms_argument(context.world)),
                    NavigateAction(
                        Pose(
                            position=Point3(1, 1, 0),
                            reference_frame=context.world.root,
                        ),
                        teleport=True,
                    ),
                ],
                context,
            ).perform(),
        )

    with simulated_robot_without_collision:
        _, _ = _timed(
            "wipe/park_arms",
            lambda: sequential(
                [ParkArmsAction(get_park_arms_argument(context.world))],
                context,
            ).perform(),
        )
        _, _ = _timed(
            "wipe/move_torso",
            lambda: sequential([MoveTorsoAction(TorsoState.HIGH)], context).perform(),
        )
        _, _ = _timed(
            "wipe/navigate_action",
            lambda: sequential(
                [NavigateAction(pickup_pose, True, teleport=True)],
                context,
            ).perform(),
        )
    print(context.world.name)
    current_plan = None
    try:
        with simulated_robot_with_collision:
            current_plan = sequential(
                [
                    WipingAction(
                        target_pose=target_pose,
                        arm=arm,
                        tool=tool,
                        clear_viz=True,
                        pointer_stride=POINTER_STRIDE,
                    ),
                ],
                context,
            )
            _, _ = _timed("wipe/action_plan_perform", current_plan.perform)
    finally:
        if current_plan is not None:
            _, _ = _timed(
                "wipe/db_commit", lambda: commit_plan_to_db(session, current_plan)
            )


def _is_vertical_wipe_pose(target_pose):
    rotation = np.asarray(target_pose.to_rotation_matrix().to_np(), dtype=float)[:3, :3]
    local_z_world = rotation[:, 2]
    return abs(float(local_z_world[2])) < 0.5


def _build_wipe_geometry_binding(target_data, target_pose, *, robot=None):
    target_pose = target_pose
    pos = np.asarray(target_pose.to_position().to_np(), dtype=float).reshape(-1)[:3]
    quat = np.asarray(target_pose.to_quaternion().to_np(), dtype=float).reshape(-1)[:4]
    roll, pitch, yaw = euler_from_quaternion(quat)
    robot_pos = np.full(3, np.nan, dtype=float)
    robot_quat = np.full(4, np.nan, dtype=float)
    robot_roll = float("nan")
    robot_pitch = float("nan")
    robot_yaw = float("nan")
    if robot is not None:
        try:
            robot_base = getattr(robot, "base", None)
            robot_pose_owner = (
                getattr(robot_base, "root", None)
                or getattr(robot, "root", None)
                or robot_base
            )
            robot_pose = robot_pose_owner.global_pose
            robot_pos = np.asarray(
                robot_pose.to_position().to_np(), dtype=float
            ).reshape(-1)[:3]
            robot_quat = np.asarray(
                robot_pose.to_quaternion().to_np(), dtype=float
            ).reshape(-1)[:4]
            robot_roll, robot_pitch, robot_yaw = euler_from_quaternion(robot_quat)
        except Exception:
            pass
    return {
        "target_world_x": round(float(pos[0]), 6),
        "target_world_y": round(float(pos[1]), 6),
        "target_world_z": round(float(pos[2]), 6),
        "support_surface_name": target_data.get("surface_name", ""),
        "object_world_x": round(float(pos[0]), 6),
        "object_world_y": round(float(pos[1]), 6),
        "object_world_z": round(float(pos[2]), 6),
        "object_quat_x": round(float(quat[0]), 6),
        "object_quat_y": round(float(quat[1]), 6),
        "object_quat_z": round(float(quat[2]), 6),
        "object_quat_w": round(float(quat[3]), 6),
        "object_roll_rad": round(float(roll), 6),
        "object_pitch_rad": round(float(pitch), 6),
        "object_yaw_rad": round(float(yaw), 6),
        "robot_world_x": (
            round(float(robot_pos[0]), 6) if np.isfinite(robot_pos[0]) else ""
        ),
        "robot_world_y": (
            round(float(robot_pos[1]), 6) if np.isfinite(robot_pos[1]) else ""
        ),
        "robot_world_z": (
            round(float(robot_pos[2]), 6) if np.isfinite(robot_pos[2]) else ""
        ),
        "robot_quat_x": (
            round(float(robot_quat[0]), 6) if np.isfinite(robot_quat[0]) else ""
        ),
        "robot_quat_y": (
            round(float(robot_quat[1]), 6) if np.isfinite(robot_quat[1]) else ""
        ),
        "robot_quat_z": (
            round(float(robot_quat[2]), 6) if np.isfinite(robot_quat[2]) else ""
        ),
        "robot_quat_w": (
            round(float(robot_quat[3]), 6) if np.isfinite(robot_quat[3]) else ""
        ),
        "robot_roll_rad": (
            round(float(robot_roll), 6) if np.isfinite(robot_roll) else ""
        ),
        "robot_pitch_rad": (
            round(float(robot_pitch), 6) if np.isfinite(robot_pitch) else ""
        ),
        "robot_yaw_rad": (
            round(float(robot_yaw), 6) if np.isfinite(robot_yaw) else ""
        ),
        "technique_name": "wipe",
        "pointer_stride": int(POINTER_STRIDE),
    }


def main_wiping(seed=None, robot_name=None, environment_name=None):
    global session
    initialize_csv(RESULTS_CSV_PATH, _results_csv_fieldnames())

    if session is None:
        session = pycram_sessionmaker()()
        Base.metadata.create_all(session.bind)
        session.commit()

    effective_seed = (
        int(seed)
        if seed is not None
        else int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    )
    world, sampled_targets, surface_plan = sample_random_bowl_poses(
        seed=effective_seed,
        robot_name=robot_name,
        environment_name=environment_name,
    )
    sampled_targets, discarded_high_targets = _filter_wipe_targets_by_height(
        sampled_targets
    )
    if discarded_high_targets:
        print(
            f"[setup] discarded {len(discarded_high_targets)} wipe target poses above "
            f"{MAX_WIPE_TARGET_HEIGHT_M:.2f}m"
        )
        for target_name, z in discarded_high_targets[:10]:
            print(f"  - {target_name}: z={z:.3f}m")
        if len(discarded_high_targets) > 10:
            print(f"  - ... {len(discarded_high_targets) - 10} more")

    node = setup_experiment_runtime(
        world=world,
        node_name="pycram_wipe_all_spaces_retry",
    )
    target_marker_pub = _create_target_pose_marker_publisher(node)
    arm_tools = _attach_sponges_for_available_arms(world)
    _publish_target_pose_markers(node, target_marker_pub, world, sampled_targets)

    context = Context.from_world(world)
    context.ros_node = node
    robot_name = _robot_name(context.robot)
    logged_environment_name = resolve_environment_name(environment_name)
    world_name = logged_environment_name
    run_id = new_run_id()

    with simulated_robot_without_collision:
        sequential(
            [SetGripperAction(Arms.BOTH, GripperState.CLOSE)],
            context,
        ).perform()

    print("[setup] surface plan:")
    print(f"[setup] seed: {effective_seed}")
    for surface_name, area_m2, target_count, placed_count in surface_plan:
        print(
            f"  - {surface_name}: area={area_m2:.3f}m^2 target={target_count} placed={placed_count}"
        )
    print(f"[setup] sampled target poses to wipe: {len(sampled_targets)}")

    success_primary = 0
    success_fallback = 0
    failed = 0
    failed_target_names = set()
    successful_target_names = set()
    target_results = []
    debug_costmap_publishers = {}

    with simulated_robot_without_collision:
        sequential(
            [
                ParkArmsAction(get_park_arms_argument(world)),
                MoveTorsoAction(TorsoState.HIGH),
            ],
            context,
        ).perform()

    total_targets = len(sampled_targets)
    for target_index, target_data in enumerate(sampled_targets, start=1):
        target_name = target_data["bowl_name"]
        target_pose = target_data["world_pose"]
        wiped_count = success_primary + success_fallback
        print(
            f"[Demo time] {wiped_count}/{total_targets} wiped surfaces "
            f"(next {target_index}/{total_targets}: {target_name})"
        )
        target_start_time = time.perf_counter()
        spawn_xyz = np.round(
            np.asarray(target_data["pose_xyz"], dtype=float), 4
        ).tolist()
        common_result_kwargs = {
            "spawn_pose_xyz": spawn_xyz,
            "environment_name": logged_environment_name,
            "task_instance_id": target_name,
            "experiment_condition": EXPERIMENT_CONDITION,
            "baseline_name": BASELINE_NAME,
            "task_name": TASK_NAME,
            "seed": effective_seed,
            "world_name": world_name,
            "run_id": run_id,
            "knowledge_query_success": True,
            "knowledge_query_error": "",
            "knowledge_prior_task": "",
            "knowledge_cutting_tool": "",
            "knowledge_cutting_position": "",
            "knowledge_repetition": "",
            "required_prerequisite": "",
            "prerequisite_source": "none",
            "prerequisite_satisfied_initially": True,
            "autonomous_execution_feasible": False,
            "assistance_type": "",
        }
        try:
            debug_costmap_publishers, preview_elapsed = _timed(
                f"target/{target_name}/costmap_preview",
                lambda: _update_costmap_debug_publishers(
                    node,
                    context.robot,
                    world,
                    target_pose,
                    debug_costmap_publishers,
                ),
            )
        except Exception as exc:
            if not _is_costmap_merge_error(exc):
                raise
            print(
                f"[skip] {target_name}: costmap preview failed, skipping pose "
                f"({type(exc).__name__}: {exc})"
            )
            failed += 1
            failed_target_names.add(target_name)
            _publish_target_pose_markers(
                node,
                target_marker_pub,
                world,
                sampled_targets,
                failed_target_names=failed_target_names,
                successful_target_names=successful_target_names,
            )
            result_row = _record_space_result(
                target_results,
                target_name,
                robot_name,
                "skipped",
                "",
                _tool_name(arm_tools[-1][1]) if arm_tools else "",
                "costmap_preview",
                [f"costmap_preview -> {type(exc).__name__}: {exc}"],
                **common_result_kwargs,
                feasibility_reason="costmap_merge_error",
                robot_decision="skip_pose",
                decision_reason="navigation_costmap_preview_failed",
                assistance_requested=False,
                assistance_completed=False,
                task_blocked_by_prerequisite=False,
                task_resumed_after_assistance=False,
                final_success=False,
                total_attempts=0,
                retry_count=0,
                collision_failure_count=0,
                recovery_used=False,
                recovery_success=False,
                perturbation_applied=False,
                perturbation_type="",
                execution_time_s=time.perf_counter() - target_start_time,
                geometry_binding=_build_wipe_geometry_binding(
                    target_data, target_pose, robot=context.robot
                ),
            )
            append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
            continue
        attempt_failures = []
        attempt_count = 0
        collision_failure_count = 0
        perturbation_applied = False
        perturbation_type = ""
        _publish_target_pose_markers(
            node,
            target_marker_pub,
            world,
            sampled_targets,
            active_target_name=target_name,
            failed_target_names=failed_target_names,
            successful_target_names=successful_target_names,
        )
        highlight_elapsed = 0.0
        pickup_loc, _ = _timed(
            f"target/{target_name}/pickup_loc_build",
            lambda: CostmapLocation(
                target=target_pose,
                reachable=True,
                reachable_arm=arm_tools[0][0] if arm_tools else None,
                validate_reachability=False,
                samples=1000,
                context=context,
            ),
        )
        try:
            pickup_pose, pickup_resolve_elapsed = _timed(
                f"target/{target_name}/pickup_loc_resolve",
                lambda: resolve_navigation_target_for_environment(
                    pickup_loc,
                    description="wiping target",
                    environment_name=environment_name,
                )[0],
            )
        except RuntimeError as exc:
            if "No collision-free navigation pose found" not in str(exc):
                raise
            print(
                f"[skip] {target_name}: pickup navigation pose failed, skipping pose "
                f"({type(exc).__name__}: {exc})"
            )
            failed += 1
            failed_target_names.add(target_name)
            _publish_target_pose_markers(
                node,
                target_marker_pub,
                world,
                sampled_targets,
                failed_target_names=failed_target_names,
                successful_target_names=successful_target_names,
            )
            result_row = _record_space_result(
                target_results,
                target_name,
                robot_name,
                "skipped",
                "",
                _tool_name(arm_tools[-1][1]) if arm_tools else "",
                "pickup_loc_resolve",
                [f"pickup_loc_resolve -> {type(exc).__name__}: {exc}"],
                **common_result_kwargs,
                feasibility_reason="navigation_pose_unreachable",
                robot_decision="skip_pose",
                decision_reason="navigation_target_resolution_failed",
                assistance_requested=False,
                assistance_completed=False,
                task_blocked_by_prerequisite=False,
                task_resumed_after_assistance=False,
                final_success=False,
                total_attempts=0,
                retry_count=0,
                collision_failure_count=0,
                recovery_used=False,
                recovery_success=False,
                perturbation_applied=False,
                perturbation_type="",
                execution_time_s=time.perf_counter() - target_start_time,
                geometry_binding=_build_wipe_geometry_binding(
                    target_data, target_pose, robot=context.robot
                ),
            )
            append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
            continue
        spawn_xyz = np.round(
            np.asarray(target_data["pose_xyz"], dtype=float), 4
        ).tolist()
        common_result_kwargs = {
            "spawn_pose_xyz": spawn_xyz,
            "environment_name": logged_environment_name,
            "task_instance_id": target_name,
            "experiment_condition": EXPERIMENT_CONDITION,
            "baseline_name": BASELINE_NAME,
            "task_name": TASK_NAME,
            "seed": effective_seed,
            "world_name": world_name,
            "run_id": run_id,
            "knowledge_query_success": True,
            "knowledge_query_error": "",
            "knowledge_prior_task": "",
            "knowledge_cutting_tool": "",
            "knowledge_cutting_position": "",
            "knowledge_repetition": "",
            "required_prerequisite": "",
            "prerequisite_source": "none",
            "prerequisite_satisfied_initially": True,
            "autonomous_execution_feasible": True,
            "assistance_type": "",
        }
        attempt_specs = [
            (arm_index, arm, tool) for arm_index, (arm, tool) in enumerate(arm_tools)
        ]
        attempt_succeeded = False

        for attempt_spec_index, (arm_index, arm, tool) in enumerate(attempt_specs):
            phase_name = "primary" if arm_index == 0 else "fallback"
            decision = "wipe" if arm_index == 0 else "retry_with_left_arm"
            decision_reason = (
                "primary_attempt" if arm_index == 0 else "right_arm_failed"
            )
            result_perturbation_applied = False
            result_perturbation_type = ""
            print(f"[wipe] {target_name}: try {arm.name} arm")

            try:
                attempt_count += 1
                _try_wipe(
                    context,
                    target_pose,
                    pickup_pose,
                    arm,
                    tool,
                    environment_name=environment_name,
                )
                if arm_index == 0:
                    success_primary += 1
                else:
                    success_fallback += 1
                wiped_count = success_primary + success_fallback
                print(f"[Demo time] {wiped_count}/{total_targets} wiped surfaces")
                successful_target_names.add(target_name)
                _publish_target_pose_markers(
                    node,
                    target_marker_pub,
                    world,
                    sampled_targets,
                    failed_target_names=failed_target_names,
                    successful_target_names=successful_target_names,
                )
                result_row = _record_space_result(
                    target_results,
                    target_name,
                    robot_name,
                    "success",
                    arm.name,
                    _tool_name(tool),
                    phase_name,
                    attempt_failures,
                    **common_result_kwargs,
                    feasibility_reason="ok",
                    robot_decision=decision,
                    decision_reason=decision_reason,
                    assistance_requested=False,
                    assistance_completed=False,
                    task_blocked_by_prerequisite=False,
                    task_resumed_after_assistance=False,
                    final_success=True,
                    total_attempts=attempt_count,
                    retry_count=max(0, attempt_count - 1),
                    collision_failure_count=collision_failure_count,
                    recovery_used=attempt_count > 1,
                    recovery_success=attempt_count > 1,
                    perturbation_applied=result_perturbation_applied,
                    perturbation_type=result_perturbation_type,
                    execution_time_s=time.perf_counter() - target_start_time,
                    geometry_binding=_build_wipe_geometry_binding(
                        target_data, target_pose, robot=context.robot
                    ),
                )
                append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
                suffix = " (fallback)" if arm_index > 0 else ""
                print(f"[ok] {target_name}: wiped with {arm.name} arm{suffix}")
                if DEBUG_PROFILE_WIPING:
                    print(
                        f"[profile] target/{target_name}/summary: "
                        f"preview={preview_elapsed:.3f}s "
                        f"pickup_resolve={pickup_resolve_elapsed:.3f}s "
                        f"highlight={highlight_elapsed:.3f}s "
                        f"total={time.perf_counter() - target_start_time:.3f}s"
                    )
                attempt_succeeded = True
                break
            except TimeoutError as exc:
                collision_failure_count += 1
                attempt_failures.append(
                    f"{arm.name} {phase_name} -> {_format_attempt_error(exc)}"
                )
                print(
                    f"[{'retry' if attempt_spec_index < len(attempt_specs) - 1 else 'fail'}] "
                    f"{target_name}: {arm.name}"
                    + f" timed out ({type(exc).__name__}: {exc})"
                )
            except Exception as exc:
                if _is_collision_like_failure(exc):
                    collision_failure_count += 1
                attempt_failures.append(
                    f"{arm.name} {phase_name} -> {_format_attempt_error(exc)}"
                )
                print(
                    f"[{'retry' if attempt_spec_index < len(attempt_specs) - 1 else 'fail'}] "
                    f"{target_name}: {arm.name}"
                    + f" failed ({type(exc).__name__}: {exc})"
                )

        if attempt_succeeded:
            continue

        failed += 1
        failed_target_names.add(target_name)
        _publish_target_pose_markers(
            node,
            target_marker_pub,
            world,
            sampled_targets,
            failed_target_names=failed_target_names,
            successful_target_names=successful_target_names,
        )
        last_tool = arm_tools[-1][1]
        result_row = _record_space_result(
            target_results,
            target_name,
            robot_name,
            "failed",
            "",
            _tool_name(last_tool),
            "failed",
            attempt_failures,
            **common_result_kwargs,
            feasibility_reason="collision_or_motion_failure",
            robot_decision="task_failed",
            decision_reason="all_wipe_attempts_failed",
            assistance_requested=False,
            assistance_completed=False,
            task_blocked_by_prerequisite=False,
            task_resumed_after_assistance=False,
            final_success=False,
            total_attempts=attempt_count,
            retry_count=max(0, attempt_count - 1),
            collision_failure_count=collision_failure_count,
            recovery_used=attempt_count > 1,
            recovery_success=False,
            perturbation_applied=False,
            perturbation_type="",
            execution_time_s=time.perf_counter() - target_start_time,
            geometry_binding=_build_wipe_geometry_binding(
                target_data, target_pose, robot=context.robot
            ),
        )
        append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
        if DEBUG_PROFILE_WIPING:
            print(
                f"[profile] target/{target_name}/failed_total: "
                f"{time.perf_counter() - target_start_time:.3f}s"
            )

    print("[summary]")
    print(f"  total sampled target poses: {len(sampled_targets)}")
    print(f"  success primary (RIGHT): {success_primary}")
    print(f"  success fallback (LEFT): {success_fallback}")
    print(f"  failed both arms: {failed}")
    print(f"  results csv: {RESULTS_CSV_PATH}")

    shutdown_experiment_runtime(node)


def main_mixing(seed=None):
    return main_wiping(seed=seed)


if __name__ == "__main__":
    main_wiping()
