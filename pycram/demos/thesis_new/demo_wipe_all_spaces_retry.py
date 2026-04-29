import os
import time

import numpy as np
from geometry_msgs.msg import Pose as RosPose
from rclpy.duration import Duration as RclpyDuration
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

try:
    from demos.thesis.simulation_setup import BoxSpec, add_box
    from demos.thesis_new.spawn_random_bowls import sample_random_bowl_poses
    from demos.thesis_new.tool_mounts import get_tool_mount_pose_kwargs
except ModuleNotFoundError:
    from thesis.simulation_setup import BoxSpec, add_box
    from thesis_new.spawn_random_bowls import sample_random_bowl_poses
    from thesis_new.tool_mounts import get_tool_mount_pose_kwargs
from pycram.robot_plans.actions.composite.utils.demo_utils import (
    build_navigation_costmaps,
    commit_plan_to_db,
    get_available_arm_tool_frames,
    get_park_arms_argument,
    get_primary_robot_name,
    resolve_navigation_target_for_environment,
    setup_experiment_runtime,
    shutdown_experiment_runtime,
    update_navigation_costmap_debug_publishers,
)
from pycram.robot_plans.actions.composite.utils.experiment_logging import (
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
from pycram.tf_transformations import quaternion_from_euler, quaternion_multiply
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
RECORDS_DIR = os.path.join(os.path.dirname(__file__), "records")
RESULTS_CSV_PATH = os.path.join(RECORDS_DIR, "wipe_all_spaces_results.csv")
EXPERIMENT_CONDITION = "full_system"
BASELINE_NAME = "task_knowledge+htn+constraint_planning"
TASK_NAME = "space_wiping"
POINTER_STRIDE = 3
TARGET_MARKER_TOPIC = "/pycram/wipe_targets"
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
    return ["target_name", "spawn_pose_xyz", *BASE_RESULT_FIELDNAMES]


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

    publisher.publish(marker_array)
    print(
        f"[viz] published {len(sampled_targets)} wipe target markers on {TARGET_MARKER_TOPIC}"
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

    _, _ = _timed("wipe/db_commit", lambda: commit_plan_to_db(session, current_plan))


def _rotate_pose_180deg_z(target_pose):
    target_pose = target_pose
    pos = np.asarray(target_pose.to_position().to_np(), dtype=float).reshape(-1)[:3]
    quat = np.asarray(target_pose.to_quaternion().to_np(), dtype=float).reshape(-1)[:4]
    rot_quat = quaternion_from_euler(0.0, 0.0, np.pi)
    new_quat = quaternion_multiply(rot_quat, quat)
    return HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=float(pos[0]),
        pos_y=float(pos[1]),
        pos_z=float(pos[2]),
        quat_x=float(new_quat[0]),
        quat_y=float(new_quat[1]),
        quat_z=float(new_quat[2]),
        quat_w=float(new_quat[3]),
        reference_frame=target_pose.reference_frame,
    )


def _build_wipe_geometry_binding(target_data, target_pose):
    target_pose = target_pose
    pos = np.asarray(target_pose.to_position().to_np(), dtype=float).reshape(-1)[:3]
    quat = np.asarray(target_pose.to_quaternion().to_np(), dtype=float).reshape(-1)[:4]
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
    world_name = _body_name(world.root)
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
    success_rotated_right = 0
    success_rotated_left = 0
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

    for target_data in sampled_targets:
        target_name = target_data["bowl_name"]
        target_pose = target_data["world_pose"]
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
        attempt_failures = []
        attempt_count = 0
        collision_failure_count = 0
        target_start_time = time.perf_counter()
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
        pickup_pose, pickup_resolve_elapsed = _timed(
            f"target/{target_name}/pickup_loc_resolve",
            lambda: resolve_navigation_target_for_environment(
                pickup_loc,
                description="wiping target",
                environment_name=environment_name,
            )[0],
        )
        spawn_xyz = np.round(
            np.asarray(target_data["pose_xyz"], dtype=float), 4
        ).tolist()
        common_result_kwargs = {
            "spawn_pose_xyz": spawn_xyz,
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
        arm_attempt_groups = [
            ("primary", arm_tools),
            ("after_rotation", arm_tools),
        ]
        attempt_succeeded = False

        for group_index, (phase_name, current_arm_tools) in enumerate(
            arm_attempt_groups
        ):
            phase_target_pose = target_pose
            if group_index == 1:
                print(
                    f"[retry] {target_name}: rotate target pose 180deg around Z and try again"
                )
                perturbation_applied = True
                perturbation_type = "z_rotation_180"
                phase_target_pose = _rotate_pose_180deg_z(target_pose)
                target_data["world_pose"] = phase_target_pose
                debug_costmap_publishers, _ = _timed(
                    f"target/{target_name}/costmap_preview_rotated",
                    lambda: _update_costmap_debug_publishers(
                        node,
                        context.robot,
                        world,
                        phase_target_pose,
                        debug_costmap_publishers,
                    ),
                )
                _publish_target_pose_markers(
                    node,
                    target_marker_pub,
                    world,
                    sampled_targets,
                    active_target_name=target_name,
                    failed_target_names=failed_target_names,
                    successful_target_names=successful_target_names,
                )

            for attempt_index, (arm, tool) in enumerate(current_arm_tools):
                is_primary_phase = group_index == 0 and attempt_index == 0
                is_fallback_phase = group_index == 0 and attempt_index > 0
                if group_index == 0:
                    decision = "wipe" if attempt_index == 0 else "retry_with_left_arm"
                    decision_reason = (
                        "primary_success" if attempt_index == 0 else "right_arm_failed"
                    )
                else:
                    decision = "rotate_object_and_retry"
                    decision_reason = "both_arms_failed_before_rotation"
                print(
                    f"[wipe] {target_name}: try {arm.name} arm"
                    + (" after rotation" if group_index == 1 else "")
                )
                # _try_wipe(context, phase_target_pose, arm, tool)

                try:
                    attempt_count += 1
                    _try_wipe(
                        context,
                        phase_target_pose,
                        pickup_pose,
                        arm,
                        tool,
                        environment_name=environment_name,
                    )
                    if is_primary_phase:
                        success_primary += 1
                    elif is_fallback_phase:
                        success_fallback += 1
                    elif attempt_index == 0:
                        success_rotated_right += 1
                    else:
                        success_rotated_left += 1
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
                        perturbation_applied=perturbation_applied,
                        perturbation_type=perturbation_type,
                        execution_time_s=time.perf_counter() - target_start_time,
                        geometry_binding=_build_wipe_geometry_binding(
                            target_data, phase_target_pose
                        ),
                    )
                    append_csv_row(
                        RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row
                    )
                    suffix = (
                        " after rotation"
                        if group_index == 1
                        else (" (fallback)" if attempt_index > 0 else "")
                    )
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
                        f"[{'retry' if not (group_index == len(arm_attempt_groups) - 1 and attempt_index == len(current_arm_tools) - 1) else 'fail'}] "
                        f"{target_name}: {arm.name}"
                        + (" after rotation" if group_index == 1 else "")
                        + f" timed out ({type(exc).__name__}: {exc})"
                    )
                except Exception as exc:
                    if _is_collision_like_failure(exc):
                        collision_failure_count += 1
                    attempt_failures.append(
                        f"{arm.name} {phase_name} -> {_format_attempt_error(exc)}"
                    )
                    print(
                        f"[{'retry' if not (group_index == len(arm_attempt_groups) - 1 and attempt_index == len(current_arm_tools) - 1) else 'fail'}] "
                        f"{target_name}: {arm.name}"
                        + (" after rotation" if group_index == 1 else "")
                        + f" failed ({type(exc).__name__}: {exc})"
                    )
            if attempt_succeeded:
                break

        if attempt_succeeded:
            continue

        perturbation_applied = True
        perturbation_type = "z_rotation_180"
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
            "after_rotation",
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
            perturbation_applied=perturbation_applied,
            perturbation_type=perturbation_type,
            execution_time_s=time.perf_counter() - target_start_time,
            geometry_binding=_build_wipe_geometry_binding(
                target_data, target_data["world_pose"]
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
    print(f"  success after rotation (RIGHT): {success_rotated_right}")
    print(f"  success after rotation (LEFT): {success_rotated_left}")
    print(f"  failed both arms: {failed}")
    print(f"  results csv: {RESULTS_CSV_PATH}")

    shutdown_experiment_runtime(node)


def main_mixing(seed=None):
    return main_wiping(seed=seed)


if __name__ == "__main__":
    main_wiping()
