import os
import inspect
import time

import numpy as np

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
from pycram.robot_plans.actions.composite.tool_based import SimplePouringAction
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionCheck,
    CollisionMatrix,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cup
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body

from .demo_mix_all_bowls_retry import (
    _build_mixing_geometry_binding,
    _empty_mixing_progress,
    _timed,
    _update_costmap_debug_publishers,
)
from .spawn_random_bowls import _parse_stl, setup_random_bowl_world
from .tool_mounts import get_tool_mount_pose_kwargs
from .utils.demo_utils import (
    attach_available_tools,
    collect_named_targets,
    commit_plan_to_db,
    get_park_arms_argument,
    highlight_current_target,
    resolve_navigation_target_for_environment,
    setup_experiment_runtime,
    shutdown_experiment_runtime,
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
from .world_setup import resolve_robot_name

DEFAULT_BOWL_COLOR = Color(R=0.78, G=0.80, B=0.86)
ACTIVE_BOWL_COLOR = Color(R=0.52, G=0.82, B=0.98)
FAILED_BOWL_COLOR = Color(R=0.95, G=0.20, B=0.20)
SUCCESS_BOWL_COLOR = Color(R=0.62, G=0.92, B=0.62)
RECORDS_DIR = os.path.join(os.path.dirname(__file__), "../records")
RESULTS_CSV_PATH = os.environ.get(
    "THESIS_POUR_RESULTS_CSV_PATH",
    os.path.join(RECORDS_DIR, "pour_all_bowls_results.csv"),
)
EXPERIMENT_CONDITION = "full_system"
BASELINE_NAME = "simple_pouring_action"
TASK_NAME = "pour"
DEBUG_PROFILE_POURING = True
session = None
DEFAULT_COSTMAP_CONFIG = {
    "costmap_width": 200,
    "costmap_height": 200,
    "costmap_resolution": 0.02,
    "samples": 1000,
}
LARGE_BASE_COSTMAP_CONFIG = {
    "costmap_width": 400,
    "costmap_height": 400,
    "costmap_resolution": 0.02,
    "obstacle_clearance": 0.20,
    "samples": 2500,
}
LARGE_BASE_ROBOTS = {"garmi", "justin", "rollin_justin"}
RIGHT_SIDE_WITH_LEFT_ARM_ROBOTS = {"stretch"}


def _results_csv_fieldnames():
    return ["bowl_name", *BASE_RESULT_FIELDNAMES]


def _empty_pouring_progress():
    progress = _empty_mixing_progress()
    progress.pop("mixing_phase_reached", None)
    progress.pop("mixing_planned_duration_s", None)
    progress.pop("mixing_estimated_completed_time_s", None)
    return progress


def _pouring_progress_from_action(action):
    if action is None:
        return _empty_pouring_progress()

    def value(name, default=""):
        result = getattr(action, name, default)
        return default if result is None else result

    return {
        "motion_approach_completed": value("logged_motion_progress_note") == "completed",
        "motion_waypoint_count": value("logged_waypoint_count"),
        "motion_stopped_waypoint_index": value("logged_stopped_waypoint_index"),
        "motion_stopped_waypoint_fraction": value(
            "logged_stopped_waypoint_fraction"
        ),
        "motion_stopped_waypoint_x": value("logged_stopped_waypoint_x"),
        "motion_stopped_waypoint_y": value("logged_stopped_waypoint_y"),
        "motion_stopped_waypoint_z": value("logged_stopped_waypoint_z"),
        "motion_stopped_distance_m": value("logged_stopped_distance_m"),
        "motion_progress_note": value("logged_motion_progress_note"),
    }


def _stall_status_from_exception(exc):
    current = exc
    while current is not None:
        message = str(current)
        if "status=wiggle" in message:
            return "wiggle"
        if "status=stagnant" in message:
            return "stagnant"
        current = current.__cause__ or current.__context__
    return ""


def _failure_kind_from_exception(exc):
    message = str(exc)
    stall_status = _stall_status_from_exception(exc)
    if stall_status:
        return f"motion_{stall_status}"
    if "navigation" in message.lower() or "pickup_pose" in message:
        return "navigation_target_resolution_failed"
    if "cup and bowl are in collision" in message:
        return "cup_bowl_collision"
    if _is_collision_like_failure(exc):
        return "collision_or_motion_failure"
    return "execution_failure"


def _pouring_progress_from_exception(exc):
    action = getattr(exc, "pour_action", None)
    progress = _pouring_progress_from_action(action)
    stall_status = _stall_status_from_exception(exc)
    if stall_status:
        progress["motion_progress_note"] = f"failed:{stall_status}"
    elif progress.get("motion_progress_note") == "":
        progress["motion_progress_note"] = "failed"
    return progress


def _record_pour_result(
    results,
    bowl_name,
    robot_name,
    outcome,
    succeeded_arm,
    tool_name,
    phase,
    failures,
    *,
    execution_time_s,
    geometry_binding,
    motion_progress=None,
    **kwargs,
):
    geometry_binding = dict(geometry_binding)
    geometry_binding["technique_name"] = "simple_pour"
    row = build_base_result_row(
        results,
        robot_name,
        outcome,
        succeeded_arm,
        tool_name,
        phase,
        failures,
        bowl_name=bowl_name,
        **kwargs,
        **geometry_binding,
        **(motion_progress or _empty_pouring_progress()),
        execution_time_s=round(execution_time_s, 4),
    )
    return row


def _tool_root(tool):
    if tool is None:
        return None
    return getattr(tool, "root", tool)


def _bodies_are_colliding(world, body_a, body_b):
    if body_a is None or body_b is None:
        return False
    if not body_a.has_collision() or not body_b.has_collision():
        return False

    detector = world.collision_manager.collision_detector
    if hasattr(detector, "sync_world_state"):
        detector.sync_world_state()
    result = detector.check_collisions(
        CollisionMatrix(
            {
                CollisionCheck.create_and_validate(
                    body_a=body_a,
                    body_b=body_b,
                    distance=0.0,
                )
            }
        )
    )
    return any(float(contact.distance) <= 0.0 for contact in result.contacts)


def _raise_if_cup_bowl_collision(bowl: Body, cup_tool: Body, world, pour_action=None):
    """
    Casts a ray from the tool center along its Z-axis.
    Returns True if the ray passes within bowl_radius of the bowl center.
    """
    origin = getattr(pour_action, "pour_ray_origin_xyz", None)
    z_dir = getattr(pour_action, "pour_ray_direction_xyz", None)
    if origin is not None and z_dir is not None:
        origin = np.array(origin, dtype=float)
        z_dir = np.array(z_dir, dtype=float)
    else:
        # Fallback for direct calls without the action object. Avoid this path in
        # causal runs because tool global_pose can trigger expensive symbolic FK.
        tool_pose = cup_tool.root.global_pose
        tool_pose_quat = np.array(
            [float(x) for x in tool_pose.to_quaternion().to_np()],
            dtype=float
        )
        qx, qy, qz, qw = tool_pose_quat
        z_dir = np.array([
            2 * (qx * qz + qw * qy),
            2 * (qy * qz - qw * qx),
            1 - 2 * (qx * qx + qy * qy)
        ], dtype=float)
        origin = np.array(
            [float(tool_pose.x), float(tool_pose.y), float(tool_pose.z)],
            dtype=float,
        )
    z_norm = float(np.linalg.norm(z_dir))
    if z_norm <= 1e-9:
        raise RuntimeError("Pouring failed success check: invalid cup ray direction.")
    z_dir = z_dir / z_norm

    bowl_pose = bowl.global_pose
    bowl_center = np.array([float(bowl_pose.x), float(bowl_pose.y), float(bowl_pose.z)])
    bowl_radius = 0.12

    t = np.dot(bowl_center - origin, z_dir)
    closest = origin + t * z_dir
    dist = np.linalg.norm(closest - bowl_center)

    hit = t > 0 and dist < bowl_radius

    print(
        "[pour raycast] "
        f"z_dir={np.round(z_dir, 3).tolist()} "
        f"t={t:.3f} "
        f"dist_to_bowl={dist:.3f} "
        f"hit={hit}",
        flush=True,
    )
    if not hit:
        raise RuntimeError(
            "Pouring failed success check: cup not tilted towards the bowl."
        )


def _pour_attempts(mounted_cups, robot_name=None):
    cups_by_arm = {arm: tool for arm, tool in mounted_cups}
    if Arms.RIGHT in cups_by_arm and Arms.LEFT in cups_by_arm:
        return [
            (Arms.RIGHT, cups_by_arm[Arms.RIGHT], Arms.RIGHT),
            (Arms.LEFT, cups_by_arm[Arms.LEFT], Arms.LEFT),
        ]

    if Arms.LEFT in cups_by_arm:
        pour_side = (
            Arms.RIGHT
            if robot_name in RIGHT_SIDE_WITH_LEFT_ARM_ROBOTS
            else Arms.LEFT
        )
        return [
            (Arms.LEFT, cups_by_arm[Arms.LEFT], pour_side),
        ]

    if Arms.RIGHT in cups_by_arm:
        return [
            (Arms.RIGHT, cups_by_arm[Arms.RIGHT], Arms.RIGHT),
        ]

    return []


def _try_pour(context, bowl, pickup_pose, arm, cup_tool, pour_side=None):
    if cup_tool is None:
        raise RuntimeError(f"No mounted pouring cup available for {arm.name}.")

    with simulated_robot_without_collision:
        sequential(
            [
                ParkArmsAction(get_park_arms_argument(context.world)),
                MoveTorsoAction(TorsoState.HIGH),
                NavigateAction(pickup_pose, True, teleport=True),
            ],
            context,
        ).perform()

    current_plans = []
    pour_action = None
    try:
        with simulated_robot_with_collision:
            # The cup is attached to the selected hand before Context creation.
            # SimplePouringAction uses that mounted tool frame for the Cartesian tilt.
            print(
                "[pour debug] creating SimplePouringAction "
                f"arm={arm.name} "
                f"pour_side={(pour_side or arm).name} "
                f"action_file={inspect.getsourcefile(SimplePouringAction)}",
                flush=True,
            )
            pour_action = SimplePouringAction(
                object_designator=bowl,
                arm=arm,
                pour_side=pour_side,
                nav=pickup_pose,
                source_object_designator=cup_tool
            )
            pour_plan = sequential([pour_action], context)
            current_plans.append(pour_plan)
            pour_plan.perform()
            _raise_if_cup_bowl_collision(
                bowl=bowl,
                cup_tool=cup_tool,
                world=context.world,
                pour_action=pour_action,
            )
            with simulated_robot_without_collision:
                sequential(
                    [
                        ParkArmsAction(get_park_arms_argument(context.world)),
                        MoveTorsoAction(TorsoState.HIGH),
                    ],
                    context,
                ).perform()
            return pour_action
    except Exception as exc:
        stall_status = _stall_status_from_exception(exc)
        if stall_status and pour_action is not None:
            try:
                _raise_if_cup_bowl_collision(
                    bowl=bowl,
                    cup_tool=cup_tool,
                    world=context.world,
                    pour_action=pour_action,
                )
            except Exception as raycast_exc:
                print(
                    "[pour timeout acceptance] "
                    f"{arm.name}/{(pour_side or arm).name} stalled with "
                    f"{stall_status}, but raycast did not pass "
                    f"({type(raycast_exc).__name__}: {raycast_exc})",
                    flush=True,
                )
            else:
                pour_action.logged_motion_progress_note = (
                    f"accepted_{stall_status}_after_raycast"
                )
                print(
                    "[pour timeout acceptance] "
                    f"{arm.name}/{(pour_side or arm).name} stalled with "
                    f"{stall_status}, accepting as success because raycast passed.",
                    flush=True,
                )
                with simulated_robot_without_collision:
                    sequential(
                        [
                            ParkArmsAction(get_park_arms_argument(context.world)),
                            MoveTorsoAction(TorsoState.HIGH),
                        ],
                        context,
                    ).perform()
                return pour_action

        if pour_action is not None:
            setattr(exc, "pour_action", pour_action)
        raise
    finally:
        for current_plan in current_plans:
            commit_plan_to_db(session, current_plan)


def _park_after_pour_best_effort(context, bowl_name):
    try:
        with simulated_robot_without_collision:
            sequential(
                [ParkArmsAction(get_park_arms_argument(context.world))],
                context,
            ).perform()
    except Exception as exc:
        print(
            f"[cleanup] {bowl_name}: parking after pour failed "
            f"but pour result is already decided "
            f"({type(exc).__name__}: {exc})",
            flush=True,
        )


def _pour_costmap_config(robot_name):
    if robot_name in LARGE_BASE_ROBOTS:
        return LARGE_BASE_COSTMAP_CONFIG
    return DEFAULT_COSTMAP_CONFIG


def _resolve_pour_pickup_pose(context, bowl, arm, environment_name, robot_name):
    costmap_config = _pour_costmap_config(robot_name)
    pickup_loc = CostmapLocation(
        target=bowl.global_pose,
        reachable=True,
        reachable_arm=arm,
        validate_reachability=False,
        context=context,
        **costmap_config,
    )
    print(
        "[pour costmap] "
        f"robot={robot_name} arm={arm.name} "
        f"size={costmap_config['costmap_width']}x{costmap_config['costmap_height']} "
        f"resolution={costmap_config['costmap_resolution']} "
        f"clearance={costmap_config.get('obstacle_clearance', 'auto')} "
        f"samples={costmap_config['samples']}",
        flush=True,
    )
    return resolve_navigation_target_for_environment(
        pickup_loc,
        description=f"pouring into {bowl.name} with {arm.name}",
        environment_name=environment_name,
    )[0]


def main_pouring(seed=None, robot_name=None, environment_name=None):
    global session
    if session is None:
        session = pycram_sessionmaker()()
        Base.metadata.create_all(session.bind)
        session.commit()

    effective_seed = (
        int(seed)
        if seed is not None
        else int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    )
    world, _, surface_plan = setup_random_bowl_world(
        seed=effective_seed,
        robot_name=robot_name,
        environment_name=environment_name,
    )

    print("[setup] starting ROS runtime and RViz publishers", flush=True)
    node = setup_experiment_runtime(
        world=world,
        node_name="pycram_pour_all_bowls_retry",
    )

    try:
        print("[setup] building execution context", flush=True)
        context = Context.from_world(world)
        context.ros_node = node
        robot_label = _robot_name(context.robot)
        world_name = environment_name
        run_id = new_run_id()

        with simulated_robot_without_collision:
            sequential(
                [
                    ParkArmsAction(get_park_arms_argument(context.world)),
                    MoveTorsoAction(TorsoState.HIGH),

                ],
                context,
            ).perform()
        resolved_robot_name = resolve_robot_name(robot_name)
        print("[setup] attaching pouring cups", flush=True)
        mounted_cups = attach_available_tools(
            world,
            _parse_stl,
            mesh_parts=("objects", "jeroen_cup.stl"),
            right_name="jeroen_cup_right",
            left_name="jeroen_cup_left",
            right_pose_kwargs=get_tool_mount_pose_kwargs(
                "pour", resolved_robot_name, Arms.RIGHT
            ),
            left_pose_kwargs=get_tool_mount_pose_kwargs(
                "pour", resolved_robot_name, Arms.LEFT
            ),
            tool_cls=Cup,
        )

        print("[setup] collecting pouring targets", flush=True)
        bowls = collect_named_targets(world, "bowl_")


        print("[setup] closing grippers", flush=True)
        with simulated_robot_without_collision:
            sequential(
                [SetGripperAction(Arms.BOTH, GripperState.CLOSE)],
                context,
            ).perform()

        print("[setup] surface plan:")
        print(f"[setup] seed: {effective_seed}")
        for surface_name, area_m2, target_count, placed_count in surface_plan:
            print(
                f"  - {surface_name}: area={area_m2:.3f}m^2 "
                f"target={target_count} placed={placed_count}"
            )
        print(f"[setup] bowls to pour into: {len(bowls)}")

        initialize_csv(RESULTS_CSV_PATH, _results_csv_fieldnames())
        debug_costmap_publishers = {}
        successful_bowls = set()
        failed_bowls = set()
        pour_results = []
        success_count = 0

        for bowl_index, bowl in enumerate(bowls, start=1):
            bowl_name = _body_name(bowl)
            print(
                f"[Demo time] {success_count}/{len(bowls)} poured bowls "
                f"(next {bowl_index}/{len(bowls)}: {bowl_name})"
            )
            bowl_start_time = time.perf_counter()
            attempt_failures = []
            attempt_count = 0
            collision_failure_count = 0
            last_progress = _empty_pouring_progress()

            debug_costmap_publishers, _ = _timed(
                f"pour/{bowl_name}/costmap_preview",
                lambda: _update_costmap_debug_publishers(
                    node, context.robot, world, bowl, debug_costmap_publishers
                ),
            )
            highlight_current_target(
                world,
                bowls,
                bowl,
                default_color=DEFAULT_BOWL_COLOR,
                active_color=ACTIVE_BOWL_COLOR,
                failed_color=FAILED_BOWL_COLOR,
                success_color=SUCCESS_BOWL_COLOR,
                failed_targets=failed_bowls,
                successful_targets=successful_bowls,
            )
            common_result_kwargs = {
                "task_instance_id": bowl_name,
                "experiment_condition": EXPERIMENT_CONDITION,
                "baseline_name": BASELINE_NAME,
                "task_name": TASK_NAME,
                "seed": effective_seed,
                "world_name": world_name,
                "run_id": run_id,
            }

            attempt_succeeded = False
            attempts = _pour_attempts(mounted_cups, resolved_robot_name)
            for attempt_index, (arm, tool, pour_side) in enumerate(attempts):
                decision = "pour" if attempt_index == 0 else "retry_with_alternate_side"
                decision_reason = (
                    "primary_success" if attempt_index == 0 else "previous_pour_side_failed"
                )
                print(
                    f"[pour] {bowl_name}: try {arm.name} arm "
                    f"from {pour_side.name} side"
                )
                try:
                    attempt_count += 1
                    pickup_pose = _resolve_pour_pickup_pose(
                        context,
                        bowl,
                        arm,
                        environment_name,
                        resolved_robot_name,
                    )
                    pour_action = _try_pour(
                        context,
                        bowl,
                        pickup_pose,
                        arm,
                        tool,
                        pour_side=pour_side,
                    )
                    last_progress = _pouring_progress_from_action(pour_action)
                    success_count += 1
                    successful_bowls.add(bowl)
                    result_row = _record_pour_result(
                        pour_results,
                        bowl_name,
                        robot_label,
                        "success",
                        arm.name,
                        _tool_name(tool),
                        "primary",
                        attempt_failures,
                        **common_result_kwargs,
                        feasibility_reason="ok",
                        robot_decision=decision,
                        decision_reason=decision_reason,
                        final_success=True,
                        total_attempts=attempt_count,
                        retry_count=max(0, attempt_count - 1),
                        collision_failure_count=collision_failure_count,
                        recovery_used=attempt_count > 1,
                        recovery_success=attempt_count > 1,
                        perturbation_applied=False,
                        perturbation_type="",
                        execution_time_s=time.perf_counter() - bowl_start_time,
                        geometry_binding=_build_mixing_geometry_binding(
                            bowl, robot=context.robot
                        ),
                        motion_progress=last_progress,
                    )
                    append_csv_row(
                        RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row
                    )
                    print(
                        f"[ok] {bowl_name}: poured with {arm.name} arm "
                        f"from {pour_side.name} side"
                    )
                    _park_after_pour_best_effort(context, bowl_name)
                    attempt_succeeded = True
                    break
                except Exception as exc:
                    last_progress = _pouring_progress_from_exception(exc)
                    failure_kind = _failure_kind_from_exception(exc)
                    current_failure = (
                        f"{arm.name}/{pour_side.name} primary -> "
                        f"{_format_attempt_error(exc)}"
                    )
                    if (
                        _is_collision_like_failure(exc)
                        or "cup and bowl are in collision" in str(exc)
                    ):
                        collision_failure_count += 1
                    attempt_failures.append(current_failure)
                    result_row = _record_pour_result(
                        pour_results,
                        bowl_name,
                        robot_label,
                        "failed",
                        "",
                        _tool_name(tool),
                        "attempt",
                        [current_failure],
                        **common_result_kwargs,
                        feasibility_reason=failure_kind,
                        robot_decision=decision,
                        decision_reason=(
                            f"{failure_kind}_retry"
                            if attempt_index < len(attempts) - 1
                            else f"{failure_kind}_final"
                        ),
                        final_success=False,
                        total_attempts=attempt_count,
                        retry_count=max(0, attempt_count - 1),
                        collision_failure_count=collision_failure_count,
                        recovery_used=attempt_count > 1,
                        recovery_success=False,
                        perturbation_applied=False,
                        perturbation_type="",
                        execution_time_s=time.perf_counter() - bowl_start_time,
                        geometry_binding=_build_mixing_geometry_binding(
                            bowl, robot=context.robot
                        ),
                        motion_progress=last_progress,
                    )
                    append_csv_row(
                        RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row
                    )
                    print(
                        f"[retry] {bowl_name}: {arm.name} arm from "
                        f"{pour_side.name} side failed "
                        f"({failure_kind}: {type(exc).__name__}: {exc})"
                    )

            if attempt_succeeded:
                continue

            failed_bowls.add(bowl)
            if attempts:
                continue

            last_tool = mounted_cups[-1][1] if mounted_cups else None
            result_row = _record_pour_result(
                pour_results,
                bowl_name,
                robot_label,
                "failed",
                "",
                _tool_name(last_tool),
                "primary",
                attempt_failures,
                **common_result_kwargs,
                feasibility_reason="collision_or_motion_failure",
                robot_decision="task_failed",
                decision_reason="all_pour_attempts_failed",
                final_success=False,
                total_attempts=attempt_count,
                retry_count=max(0, attempt_count - 1),
                collision_failure_count=collision_failure_count,
                recovery_used=attempt_count > 1,
                recovery_success=False,
                perturbation_applied=False,
                perturbation_type="",
                execution_time_s=time.perf_counter() - bowl_start_time,
                geometry_binding=_build_mixing_geometry_binding(
                    bowl, robot=context.robot
                ),
                motion_progress=last_progress,
            )
            append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)

        print(f"[summary] total bowls: {len(bowls)}")
        print(f"[summary] success: {success_count}")
        print(f"[summary] failed: {len(failed_bowls)}")
    finally:
        shutdown_experiment_runtime(node)


if __name__ == "__main__":
    main_pouring()
