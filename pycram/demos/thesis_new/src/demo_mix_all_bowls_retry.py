import os
import time

import numpy as np

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.external_interfaces.sparql_queries.mixing import safe_get_mixing_knowledge
from pycram.locations.locations import CostmapLocation
from pycram.motion_executor import (
    simulated_robot_with_collision,
    simulated_robot_without_collision,
)
from pycram.orm.ormatic_interface import Base
from pycram.orm.utils import pycram_sessionmaker
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.composite.tool_based import MixingAction
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from pycram.tf_transformations import (
    euler_from_quaternion,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Whisk
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import Color

try:
    from .spawn_random_bowls import (
        _parse_stl,
        setup_random_bowl_world,
        setup_random_mixing_container_world,
    )
    from .tool_mounts import get_tool_mount_pose_kwargs
    from .world_setup import resolve_robot_name
except ModuleNotFoundError:
    from .spawn_random_bowls import (
        _parse_stl,
        setup_random_bowl_world,
        setup_random_mixing_container_world,
    )
    from .tool_mounts import get_tool_mount_pose_kwargs
    from .world_setup import resolve_robot_name
from pycram.robot_plans.actions.composite.thesis_math import body_local_aabb
from .utils.demo_utils import (
    attach_available_tools,
    collect_named_targets,
    get_park_arms_argument,
    highlight_current_target,
    resolve_navigation_target_for_environment,
    setup_experiment_runtime,
    shutdown_experiment_runtime,
    update_navigation_costmap_debug_publishers,
    build_navigation_costmaps,
    commit_plan_to_db,
)
from .utils.experiment_logging import (
    BASE_RESULT_FIELDNAMES,
    append_csv_row,
    assistance_type_from_knowledge,
    body_name as _body_name,
    build_base_result_row,
    format_attempt_error as _format_attempt_error,
    initialize_csv,
    is_collision_like_failure as _is_collision_like_failure,
    knowledge_source,
    new_run_id,
    required_prerequisite_text,
    robot_name as _robot_name,
    tool_name as _tool_name,
)

DEFAULT_BOWL_COLOR = Color(R=0.78, G=0.80, B=0.86)
ACTIVE_BOWL_COLOR = Color(R=0.52, G=0.82, B=0.98)
FAILED_BOWL_COLOR = Color(R=0.95, G=0.20, B=0.20)
SUCCESS_BOWL_COLOR = Color(R=0.62, G=0.92, B=0.62)
RECORDS_DIR = os.path.join(os.path.dirname(__file__), "../records")
RESULTS_CSV_PATH = os.environ.get(
    "THESIS_MIX_RESULTS_CSV_PATH",
    os.path.join(RECORDS_DIR, "mix_all_bowls_results.csv"),
)
EXPERIMENT_CONDITION = "full_system"
BASELINE_NAME = "task_knowledge+htn+constraint_planning"
TASK_NAME = "bowl_mixing"
MIX_DURATION_S = 2.0
POINTER_STRIDE = 0
MIXING_QUERY_TASK = "Whisking"
DEBUG_PROFILE_MIXING = True
session = None


def _timed(label, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    if DEBUG_PROFILE_MIXING:
        print(f"[profile] {label}: {elapsed:.3f}s")
    return result, elapsed


def _update_costmap_debug_publishers(node, robot, world, bowl, publishers):
    occupancy, ring, final_map = build_navigation_costmaps(
        robot, world, bowl.global_pose
    )
    return update_navigation_costmap_debug_publishers(
        node,
        world,
        publishers,
        occupancy,
        ring,
        final_map,
        namespace_prefix="mixing",
    )


def _record_bowl_result(
    results,
    bowl_name,
    robot_name,
    outcome,
    succeeded_arm,
    tool_name,
    phase,
    failures,
    *,
    knowledge_motion,
    knowledge_mixing_tool,
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
        bowl_name=bowl_name,
        knowledge_motion=knowledge_motion,
        knowledge_mixing_tool=knowledge_mixing_tool,
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
        "bowl_name",
        "knowledge_motion",
        "knowledge_mixing_tool",
        *BASE_RESULT_FIELDNAMES,
    ]


def _try_mix(context, bowl, pickup_pose, arm, tool, *, environment_name=None):
    with simulated_robot_without_collision:
        _, _ = _timed(
            "mix/reset_pose",
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
            "mix/park_arms",
            lambda: sequential(
                [ParkArmsAction(get_park_arms_argument(context.world))],
                context,
            ).perform(),
        )
        _, _ = _timed(
            "mix/move_torso",
            lambda: sequential([MoveTorsoAction(TorsoState.HIGH)], context).perform(),
        )
        _, _ = _timed(
            "mix/navigate_action",
            lambda: sequential(
                [NavigateAction(pickup_pose, True, teleport=True)],
                context,
            ).perform(),
        )

    current_plan = None
    try:
        with simulated_robot_with_collision:
            current_plan = sequential(
                [
                    MixingAction(
                        container=bowl,
                        arm=arm,
                        tool=tool,
                        clear_viz=True,
                        pointer_stride=POINTER_STRIDE,
                        mix_duration_s=MIX_DURATION_S,
                    ),
                ],
                context,
            )
            _, _ = _timed("mix/action_plan_perform", current_plan.perform)
    finally:
        if current_plan is not None:
            _, _ = _timed(
                "mix/db_commit", lambda: commit_plan_to_db(session, current_plan)
            )


def _build_mixing_geometry_binding(bowl, *, robot=None):
    mins, maxs = body_local_aabb(
        bowl,
        use_visual=True,
        apply_shape_scale=True,
    )
    size = maxs - mins
    pose = bowl.global_pose
    pos = np.asarray(pose.to_position().to_np(), dtype=float).reshape(-1)[:3]
    quat = np.asarray(pose.to_quaternion().to_np(), dtype=float).reshape(-1)[:4]
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

    support = getattr(getattr(bowl, "parent_connection", None), "parent", None)
    support_name = _body_name(support) if support is not None else ""
    support_pos = np.full(3, np.nan, dtype=float)
    support_yaw = float("nan")
    support_size = np.full(3, np.nan, dtype=float)
    if support is not None:
        try:
            support_pose = support.global_pose
            support_pos = np.asarray(
                support_pose.to_position().to_np(), dtype=float
            ).reshape(-1)[:3]
            support_quat = np.asarray(
                support_pose.to_quaternion().to_np(), dtype=float
            ).reshape(-1)[:4]
            _, _, support_yaw = euler_from_quaternion(support_quat)
            support_mins, support_maxs = body_local_aabb(
                support,
                use_visual=False,
                apply_shape_scale=True,
            )
            support_size = support_maxs - support_mins
        except Exception:
            pass

    return {
        "object_aabb_min_x": round(float(mins[0]), 6),
        "object_aabb_min_y": round(float(mins[1]), 6),
        "object_aabb_min_z": round(float(mins[2]), 6),
        "object_aabb_max_x": round(float(maxs[0]), 6),
        "object_aabb_max_y": round(float(maxs[1]), 6),
        "object_aabb_max_z": round(float(maxs[2]), 6),
        "object_size_x": round(float(size[0]), 6),
        "object_size_y": round(float(size[1]), 6),
        "object_size_z": round(float(size[2]), 6),
        "object_volume_aabb": round(float(size[0] * size[1] * size[2]), 8),
        "target_world_x": round(float(pos[0]), 6),
        "target_world_y": round(float(pos[1]), 6),
        "target_world_z": round(float(pos[2]), 6),
        "support_surface_name": support_name or "",
        "support_world_x": (
            round(float(support_pos[0]), 6) if np.isfinite(support_pos[0]) else ""
        ),
        "support_world_y": (
            round(float(support_pos[1]), 6) if np.isfinite(support_pos[1]) else ""
        ),
        "support_world_z": (
            round(float(support_pos[2]), 6) if np.isfinite(support_pos[2]) else ""
        ),
        "support_yaw_rad": (
            round(float(support_yaw), 6) if np.isfinite(support_yaw) else ""
        ),
        "support_size_x": (
            round(float(support_size[0]), 6) if np.isfinite(support_size[0]) else ""
        ),
        "support_size_y": (
            round(float(support_size[1]), 6) if np.isfinite(support_size[1]) else ""
        ),
        "support_size_z": (
            round(float(support_size[2]), 6) if np.isfinite(support_size[2]) else ""
        ),
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
        "technique_name": MIXING_QUERY_TASK,
        "pointer_stride": int(POINTER_STRIDE),
    }


def main_mixing(
    seed=None, robot_name=None, environment_name=None, container_kind="bowl"
):
    global session
    if session is None:
        session = pycram_sessionmaker()()
        Base.metadata.create_all(session.bind)
        # session.commit()

    effective_seed = (
        int(seed)
        if seed is not None
        else int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    )
    if str(container_kind).lower() == "pot":
        world, _, surface_plan = setup_random_mixing_container_world(
            seed=effective_seed,
            robot_name=robot_name,
            environment_name=environment_name,
            container_kind="pot",
        )
    else:
        world, _, surface_plan = setup_random_bowl_world(
            seed=effective_seed,
            robot_name=robot_name,
            environment_name=environment_name,
        )

    node = setup_experiment_runtime(
        world=world,
        node_name="pycram_mix_all_bowls_retry",
    )

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
    bowls = collect_named_targets(world, "bowl_")

    context = Context.from_world(world)
    context.ros_node = node
    robot_name = _robot_name(context.robot)
    world_name = environment_name
    run_id = new_run_id()
    mixing_knowledge = safe_get_mixing_knowledge(MIXING_QUERY_TASK)

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
    print(f"[setup] bowls to mix: {len(bowls)}")

    success_primary = 0
    success_fallback = 0
    failed = 0
    failed_bowls = set()
    successful_bowls = set()
    bowl_results = []
    initialize_csv(RESULTS_CSV_PATH, _results_csv_fieldnames())
    debug_costmap_publishers = {}
    with simulated_robot_without_collision:
        sequential(
            [
                ParkArmsAction(get_park_arms_argument(world)),
                MoveTorsoAction(TorsoState.HIGH),
            ],
            context,
        ).perform()

    total_bowls = len(bowls)
    for bowl_index, bowl in enumerate(bowls, start=1):
        bowl_name = _body_name(bowl)
        mixed_count = success_primary + success_fallback
        print(
            f"[Demo time] {mixed_count}/{total_bowls} mixed bowls "
            f"(next {bowl_index}/{total_bowls}: {bowl_name})"
        )
        debug_costmap_publishers, preview_elapsed = _timed(
            f"bowl/{bowl_name}/costmap_preview",
            lambda: _update_costmap_debug_publishers(
                node, context.robot, world, bowl, debug_costmap_publishers
            ),
        )
        pickup_loc, _ = _timed(
            f"bowl/{bowl_name}/pickup_loc_build",
            lambda: CostmapLocation(
                target=bowl.global_pose,
                reachable=True,
                reachable_arm=arm_tools[0][0] if arm_tools else None,
                validate_reachability=False,
                samples=1000,
                context=context,
            ),
        )
        attempt_failures = []
        attempt_count = 0
        collision_failure_count = 0
        bowl_start_time = time.perf_counter()
        perturbation_applied = False
        perturbation_type = ""
        _, highlight_elapsed = _timed(
            f"bowl/{bowl_name}/highlight",
            lambda: highlight_current_target(
                world,
                bowls,
                bowl,
                default_color=DEFAULT_BOWL_COLOR,
                active_color=ACTIVE_BOWL_COLOR,
                failed_color=FAILED_BOWL_COLOR,
                success_color=SUCCESS_BOWL_COLOR,
                failed_targets=failed_bowls,
                successful_targets=successful_bowls,
            ),
        )
        common_result_kwargs = {
            "knowledge_motion": mixing_knowledge.get("motion") or "",
            "knowledge_mixing_tool": mixing_knowledge.get("mixing_tool") or "",
            "task_instance_id": bowl_name,
            "experiment_condition": EXPERIMENT_CONDITION,
            "baseline_name": BASELINE_NAME,
            "task_name": TASK_NAME,
            "seed": effective_seed,
            "world_name": world_name,
            "run_id": run_id,
            "knowledge_query_success": mixing_knowledge.get("query_success", False),
            "knowledge_query_error": mixing_knowledge.get("query_error", ""),
            "knowledge_prior_task": "",
            "knowledge_cutting_tool": "",
            "knowledge_cutting_position": "",
            "knowledge_repetition": "",
            "required_prerequisite": required_prerequisite_text(mixing_knowledge),
            "prerequisite_source": knowledge_source(mixing_knowledge),
            "prerequisite_satisfied_initially": not bool(
                mixing_knowledge.get("required_prerequisites")
            ),
            "autonomous_execution_feasible": not bool(
                mixing_knowledge.get("required_prerequisites")
            ),
            "assistance_type": assistance_type_from_knowledge(mixing_knowledge),
        }
        try:
            pickup_pose, pickup_resolve_elapsed = _timed(
                f"bowl/{bowl_name}/pickup_loc_resolve",
                lambda: resolve_navigation_target_for_environment(
                    pickup_loc,
                    description=f"mixing {bowl.name}",
                    environment_name=environment_name,
                )[0],
            )
        except Exception as exc:
            failed += 1
            failed_bowls.add(bowl)
            attempt_failures.append(f"navigation setup -> {_format_attempt_error(exc)}")
            fallback_tool = arm_tools[-1][1] if arm_tools else None
            result_row = _record_bowl_result(
                bowl_results,
                bowl_name,
                robot_name,
                "failed",
                "",
                _tool_name(fallback_tool),
                "pickup_setup",
                attempt_failures,
                **common_result_kwargs,
                feasibility_reason="navigation_target_resolution_failed",
                robot_decision="skip_object",
                decision_reason="pickup_pose_unavailable",
                assistance_requested=False,
                assistance_completed=False,
                task_blocked_by_prerequisite=False,
                task_resumed_after_assistance=False,
                final_success=False,
                total_attempts=attempt_count,
                retry_count=0,
                collision_failure_count=collision_failure_count,
                recovery_used=False,
                recovery_success=False,
                perturbation_applied=perturbation_applied,
                perturbation_type=perturbation_type,
                execution_time_s=time.perf_counter() - bowl_start_time,
                geometry_binding=_build_mixing_geometry_binding(
                    bowl, robot=context.robot
                ),
            )
            append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
            print(
                f"[skip] {bowl_name}: pickup pose resolution failed "
                f"({type(exc).__name__}: {exc})"
            )
            if DEBUG_PROFILE_MIXING:
                print(
                    f"[profile] bowl/{bowl_name}/setup_failed_total: "
                    f"{time.perf_counter() - bowl_start_time:.3f}s"
                )
            continue
        arm_attempt_groups = [("primary", arm_tools)]
        attempt_succeeded = False

        for group_index, (phase_name, current_arm_tools) in enumerate(
            arm_attempt_groups
        ):
            for attempt_index, (arm, tool) in enumerate(current_arm_tools):
                is_primary_phase = group_index == 0 and attempt_index == 0
                is_fallback_phase = group_index == 0 and attempt_index > 0
                decision = "mix" if attempt_index == 0 else "retry_with_left_arm"
                decision_reason = (
                    "primary_success" if attempt_index == 0 else "right_arm_failed"
                )
                print(f"[mix] {bowl_name}: try {arm.name} arm")

                try:
                    attempt_count += 1
                    _try_mix(
                        context,
                        bowl,
                        pickup_pose,
                        arm,
                        tool,
                        environment_name=environment_name,
                    )
                    if is_primary_phase:
                        success_primary += 1
                    elif is_fallback_phase:
                        success_fallback += 1
                    mixed_count = success_primary + success_fallback
                    print(f"[Demo time] {mixed_count}/{total_bowls} mixed bowls")
                    successful_bowls.add(bowl)
                    result_row = _record_bowl_result(
                        bowl_results,
                        bowl_name,
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
                        execution_time_s=time.perf_counter() - bowl_start_time,
                        geometry_binding=_build_mixing_geometry_binding(
                            bowl, robot=context.robot
                        ),
                    )
                    append_csv_row(
                        RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row
                    )
                    suffix = " (fallback)" if attempt_index > 0 else ""
                    print(f"[ok] {bowl_name}: mixed with {arm.name} arm{suffix}")
                    if DEBUG_PROFILE_MIXING:
                        print(
                            f"[profile] bowl/{bowl_name}/summary: "
                            f"preview={preview_elapsed:.3f}s "
                            f"pickup_resolve={pickup_resolve_elapsed:.3f}s "
                            f"highlight={highlight_elapsed:.3f}s "
                            f"total={time.perf_counter() - bowl_start_time:.3f}s"
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
                        f"{bowl_name}: {arm.name}"
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
                        f"{bowl_name}: {arm.name}"
                        + f" failed ({type(exc).__name__}: {exc})"
                    )
            if attempt_succeeded:
                break

        if attempt_succeeded:
            continue

        failed += 1
        failed_bowls.add(bowl)
        last_tool = arm_tools[-1][1]
        result_row = _record_bowl_result(
            bowl_results,
            bowl_name,
            robot_name,
            "failed",
            "",
            _tool_name(last_tool),
            "primary",
            attempt_failures,
            **common_result_kwargs,
            feasibility_reason=(
                "prerequisite_requires_human_assistance"
                if common_result_kwargs["required_prerequisite"]
                else "collision_or_motion_failure"
            ),
            robot_decision=(
                "request_human_help"
                if common_result_kwargs["required_prerequisite"]
                else "task_failed"
            ),
            decision_reason=(
                "knowledge_base_prerequisite_detected"
                if common_result_kwargs["required_prerequisite"]
                else "all_mix_attempts_failed"
            ),
            assistance_requested=bool(common_result_kwargs["required_prerequisite"]),
            assistance_completed=False,
            task_blocked_by_prerequisite=bool(
                common_result_kwargs["required_prerequisite"]
            ),
            task_resumed_after_assistance=False,
            final_success=False,
            total_attempts=attempt_count,
            retry_count=max(0, attempt_count - 1),
            collision_failure_count=collision_failure_count,
            recovery_used=attempt_count > 1,
            recovery_success=False,
            perturbation_applied=perturbation_applied,
            perturbation_type=perturbation_type,
            execution_time_s=time.perf_counter() - bowl_start_time,
            geometry_binding=_build_mixing_geometry_binding(
                bowl, robot=context.robot
            ),
        )
        append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
        if DEBUG_PROFILE_MIXING:
            print(
                f"[profile] bowl/{bowl_name}/failed_total: "
                f"{time.perf_counter() - bowl_start_time:.3f}s"
            )

    highlight_current_target(
        world,
        bowls,
        None,
        default_color=DEFAULT_BOWL_COLOR,
        active_color=ACTIVE_BOWL_COLOR,
        failed_color=FAILED_BOWL_COLOR,
        success_color=SUCCESS_BOWL_COLOR,
        failed_targets=failed_bowls,
        successful_targets=successful_bowls,
    )

    print("[summary]")
    print(f"  total bowls: {len(bowls)}")
    print(f"  success primary (RIGHT): {success_primary}")
    print(f"  success fallback (LEFT): {success_fallback}")
    print(f"  failed both arms: {failed}")
    print(f"  results csv: {RESULTS_CSV_PATH}")

    shutdown_experiment_runtime(node)


# if __name__ == "__main__":
#     session = pycram_sessionmaker()()
#     Base.metadata.create_all(session.bind)
#     session.commit()
#
#     main_mixing()
