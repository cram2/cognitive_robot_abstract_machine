import time
from contextlib import contextmanager, nullcontext

from pycram.locations.locations import CostmapLocation
from pycram.motion_executor import (
    simulated_robot_without_collision,
    simulated_robot_with_collision,
)

from pycram.plans.factories import sequential
from pycram.robot_plans.actions.composite.tool_based import CuttingAction
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    MoveTorsoAction,
    SetGripperAction,
)
from pycram.tf_transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_matrix,
    quaternion_multiply,
)


from .spawn_random_breads import (
    get_cut_object_config,
    setup_random_bread_world,
)
from .spawn_random_breads import build_cutting_reachability_costmaps
from pycram.robot_plans.actions.composite.thesis_math import body_local_aabb
from .tool_mounts import get_tool_mount_pose_kwargs
from .world_setup import resolve_robot_name
from .utils.jpt_sampler import get_pose_scorer, rerank_by_score
from .utils.demo_utils import (
    attach_available_tools,
    update_navigation_costmap_debug_publishers,
    collect_named_targets,

    get_park_arms_argument,
    highlight_current_target,
    resolve_navigation_target_for_environment,
    set_entity_global_pose,
    setup_experiment_runtime,
    shutdown_experiment_runtime,
    body_name as _body_name,
)
from .utils.experiment_logging import (
    BASE_RESULT_FIELDNAMES,
    append_csv_row,

    build_base_result_row,
    format_attempt_error as _format_attempt_error,
    initialize_csv,
    is_collision_like_failure as _is_collision_like_failure,
    new_run_id,
    robot_name as _robot_name,
    tool_name as _tool_name,
)
import os
import numpy as np
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.orm.ormatic_interface import Base
from pycram.orm.utils import pycram_sessionmaker


from semantic_digital_twin.adapters.mesh import STLParser

from semantic_digital_twin.datastructures.definitions import TorsoState, GripperState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Knife
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import Color

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..", "..", "resources")
)
DEFAULT_BREAD_COLOR = Color(R=0.76, G=0.60, B=0.42)
DEFAULT_APPLE_COLOR = Color(R=0.55, G=0.08, B=0.08)
DEFAULT_CUCUMBER_COLOR = Color(R=0.12, G=0.38, B=0.14)
ACTIVE_BREAD_COLOR = Color(R=0.52, G=0.82, B=0.98)
FAILED_BREAD_COLOR = Color(R=0.95, G=0.20, B=0.20)
SUCCESS_BREAD_COLOR = Color(R=0.62, G=0.92, B=0.62)
RECORDS_DIR = os.path.join(os.path.dirname(__file__), "../records")
RESULTS_CSV_PATH = os.environ.get(
    "THESIS_CUT_RESULTS_CSV_PATH",
    os.path.join(RECORDS_DIR, "cut_all_breads_results.csv"),
)
EXPERIMENT_CONDITION = "full_system"
BASELINE_NAME = "base_system"
TASK_NAME = "bread_cutting"
CUTTING_QUERY_VERB = "cut:Slicing"
session = None
CUTTING_RING_DISTANCE = 0.55
CUTTING_RING_STD = 12.0
CUTTING_COSTMAP_WIDTH = 140
CUTTING_COSTMAP_HEIGHT = 140
CUTTING_COSTMAP_RESOLUTION = 0.03
DEBUG_PROFILE_CUTTING = True
ENABLE_CUTTING_HIGHLIGHT = True
PAUSE_ROS_STATE_PUBLISHERS_DURING_CUT = False
CUTTING_TECHNIQUE = "saw"
CUTTING_POINTER_STRIDE = 20
CUTTING_NUM_CUTS_X = 2
CUTTING_SLICE_THICKNESS_M = 0.03


def _cutting_obstacle_clearance(robot):
    base_bb = robot.base.bounding_box
    base_depth = float(getattr(base_bb, "depth", float("nan")))
    base_width = float(getattr(base_bb, "width", float("nan")))
    obstacle_clearance = 0.25 * (base_depth + base_width)
    if (not np.isfinite(obstacle_clearance)) or obstacle_clearance <= 0.0:
        obstacle_clearance = 0.20
    return obstacle_clearance


def _timed(label, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    if DEBUG_PROFILE_CUTTING:
        print(f"[profile] {label}: {elapsed:.3f}s")
    return result, elapsed


def _parse_stl(*relative_path_parts):
    return STLParser(os.path.join(RESOURCES_DIR, *relative_path_parts)).parse()


def _update_costmap_debug_publishers(node, robot, world, bread, publishers):
    target_pose = bread.global_pose
    occupancy, ring, final_map = build_cutting_reachability_costmaps(
        robot, world, target_pose
    )
    return update_navigation_costmap_debug_publishers(
        node,
        world,
        publishers,
        occupancy,
        ring,
        final_map,
        namespace_prefix="cutting",
    )


def _cut_object_default_color(object_kind):
    normalized = str(object_kind).lower()
    if normalized == "apple":
        return DEFAULT_APPLE_COLOR
    if normalized == "cucumber":
        return DEFAULT_CUCUMBER_COLOR
    return DEFAULT_BREAD_COLOR


def _cut_object_execution_config(object_kind):
    normalized = str(object_kind).lower()
    if normalized == "apple":
        return {
            "query_verb": "cut:Halving",
            "technique": "halving",
            "num_cuts_x": 1,
        }
    if normalized == "cucumber":
        return {
            "query_verb": "cut:Slicing",
            "technique": "slice",
            "num_cuts_x": CUTTING_NUM_CUTS_X,
        }
    return {
        "query_verb": CUTTING_QUERY_VERB,
        "technique": CUTTING_TECHNIQUE,
        "num_cuts_x": CUTTING_NUM_CUTS_X,
    }


def _record_bread_result(
    results,
    bread_name,
    robot_name,
    outcome,
    succeeded_arm,
    tool_name,
    phase,
    failures,
    *,
    task_instance_id,
    experiment_condition,
    baseline_name,
    task_name,
    seed,
    world_name,
    run_id,
    feasibility_reason,
    robot_decision,
    decision_reason,
    final_success,
    pose_sampler="costmap_fallback",
    jpt_pose_loglik="",
    total_attempts,
    retry_count,
    collision_failure_count,
    recovery_used,
    recovery_success,
    perturbation_applied,
    perturbation_type,
    execution_time_s,
    geometry_binding,
    motion_progress=None,
):
    return build_base_result_row(
        results,
        robot_name,
        outcome,
        succeeded_arm,
        tool_name,
        phase,
        failures,
        task_name=task_name,
        run_id=run_id,
        task_instance_id=task_instance_id,
        bread_name=bread_name,
        seed=seed if seed is not None else "",
        world_name=world_name,
        experiment_condition=experiment_condition,
        baseline_name=baseline_name,
        feasibility_reason=feasibility_reason,
        robot_decision=robot_decision,
        decision_reason=decision_reason,
        final_success=final_success,
        pose_sampler=pose_sampler,
        jpt_pose_loglik=jpt_pose_loglik,
        total_attempts=total_attempts,
        retry_count=retry_count,
        collision_failure_count=collision_failure_count,
        recovery_used=recovery_used,
        recovery_success=recovery_success,
        perturbation_applied=perturbation_applied,
        perturbation_type=perturbation_type,
        **geometry_binding,
        **(motion_progress or _empty_motion_progress()),
        execution_time_s=round(execution_time_s, 4),
    )


def _results_csv_fieldnames():
    return ["bread_name", "pose_sampler", "jpt_pose_loglik", *BASE_RESULT_FIELDNAMES]


def _empty_motion_progress():
    return {
        "motion_approach_completed": False,
        "motion_waypoint_count": "",
        "motion_stopped_waypoint_index": "",
        "motion_stopped_waypoint_fraction": "",
        "motion_stopped_waypoint_x": "",
        "motion_stopped_waypoint_y": "",
        "motion_stopped_waypoint_z": "",
        "motion_stopped_distance_m": "",
        "motion_progress_note": "",
    }


def _motion_progress_from_action(action):
    if action is None:
        return _empty_motion_progress()

    def value(name, default=""):
        result = getattr(action, name, default)
        return default if result is None else result

    return {
        "motion_approach_completed": True,
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


def _motion_progress_from_exception(exc):
    return getattr(exc, "motion_progress", _empty_motion_progress())


def _rotate_bread_z(world, bread, angle_rad):
    pose = bread.global_pose
    pos = np.asarray(pose.to_position().to_np(), dtype=float).reshape(-1)[:3]
    quat = np.asarray(pose.to_quaternion().to_np(), dtype=float).reshape(-1)[:4]
    rot_quat = quaternion_from_euler(0.0, 0.0, float(angle_rad))
    new_quat = quaternion_multiply(rot_quat, quat)

    rotated_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=float(pos[0]),
        pos_y=float(pos[1]),
        pos_z=float(pos[2]),
        quat_x=float(new_quat[0]),
        quat_y=float(new_quat[1]),
        quat_z=float(new_quat[2]),
        quat_w=float(new_quat[3]),
        reference_frame=world.root,
    )
    set_entity_global_pose(world, bread, rotated_pose)


def _align_bread_to_approach(world, bread, robot_pose):
    """Rotate the bread about Z so its cut-normal (object local +x) is parallel to
    the robot's approach direction (robot base -> bread).

    This is the orientation successful cuts exhibit (cut_normal parallel to the
    approach). Applying it for the chosen base pose makes object orientation correct
    by construction, so it is no longer a source of failure and the demo no longer
    needs rotation-retry attempts.

    :return: the approach yaw the bread was aligned to.
    """
    bread_pos = np.asarray(
        bread.global_pose.to_position().to_np(), dtype=float
    ).reshape(-1)[:3]
    bread_quat = np.asarray(
        bread.global_pose.to_quaternion().to_np(), dtype=float
    ).reshape(-1)[:4]
    _, _, current_yaw = euler_from_quaternion(bread_quat)
    robot_pos = np.asarray(
        robot_pose.to_position().to_np(), dtype=float
    ).reshape(-1)[:3]
    approach_yaw = float(
        np.arctan2(bread_pos[1] - robot_pos[1], bread_pos[0] - robot_pos[0])
    )
    _rotate_bread_z(world, bread, _wrap_angle_rad(approach_yaw - current_yaw))
    return approach_yaw
    world.update_forward_kinematics()


def _wrap_angle_rad(angle):
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _fold_abs_angle_rad(angle):
    return float(abs(np.arctan2(np.sin(angle), np.cos(angle))))


def _fold_parallel_angle_rad(angle):
    folded = _fold_abs_angle_rad(angle)
    return float(min(folded, abs(np.pi - folded)))


def compute_cutting_cause_features(candidate_pose, bread):
    """Cutting JPT cause-feature vector for a hypothetical robot base pose.

    Mirrors the per-robot feature math in :func:`_build_cut_geometry_binding`
    (object yaw / cut-normal derivation + the robot-relative angles), but is
    parameterized by an arbitrary candidate base ``Pose`` so costmap candidates
    can be scored before the robot is moved. Keys match the manifest ``cause_cols``.
    """
    object_pose = bread.global_pose
    object_pos = np.asarray(
        object_pose.to_position().to_np(), dtype=float
    ).reshape(-1)[:3]
    object_quat = np.asarray(
        object_pose.to_quaternion().to_np(), dtype=float
    ).reshape(-1)[:4]
    _, _, object_yaw = euler_from_quaternion(object_quat)
    rotation = quaternion_matrix(object_quat)[:3, :3]
    cut_normal_world = rotation @ np.array([1.0, 0.0, 0.0], dtype=float)
    cut_normal_world_yaw = float(
        np.arctan2(cut_normal_world[1], cut_normal_world[0])
    )

    robot_pos = np.asarray(
        candidate_pose.to_position().to_np(), dtype=float
    ).reshape(-1)[:3]
    robot_quat = np.asarray(
        candidate_pose.to_quaternion().to_np(), dtype=float
    ).reshape(-1)[:4]
    _, _, robot_yaw = euler_from_quaternion(robot_quat)

    robot_to_object_yaw = float(
        np.arctan2(object_pos[1] - robot_pos[1], object_pos[0] - robot_pos[0])
    )
    object_yaw_relative_to_robot = _wrap_angle_rad(object_yaw - robot_yaw)
    cut_normal_relative_to_approach = _wrap_angle_rad(
        cut_normal_world_yaw - robot_to_object_yaw
    )
    return {
        "robot_to_target_dist": float(
            np.hypot(robot_pos[0] - object_pos[0], robot_pos[1] - object_pos[1])
        ),
        "robot_yaw_rad": float(robot_yaw),
        "object_yaw_rad": float(object_yaw),
        "object_yaw_relative_to_robot_rad": float(object_yaw_relative_to_robot),
        "cut_normal_approach_parallel_score": float(
            abs(np.cos(cut_normal_relative_to_approach))
        ),
        "cut_normal_approach_abs_angle_rad": float(
            _fold_parallel_angle_rad(cut_normal_relative_to_approach)
        ),
    }


@contextmanager
def _pause_ros_state_publishers_during_motion(world):
    paused_callbacks = []
    for callback in list(getattr(world.state, "state_change_callbacks", [])):
        callback_type = callback.__class__.__name__
        callback_module = callback.__class__.__module__
        if callback_type == "TFPublisher" or ".adapters.ros." in callback_module:
            was_paused = getattr(callback, "_is_paused", False)
            if not was_paused:
                callback.pause()
                paused_callbacks.append(callback)
    try:
        yield
    finally:
        for callback in paused_callbacks:
            callback.resume()


def _try_cut(
    context,
    bread,
    pickup_pose,
    arm,
    tool,
    *,
    cutting_technique,
    num_cuts_x,
    environment_name=None,
):
    with simulated_robot_without_collision:
        _, _ = _timed(
            "cut/reset_pose",
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
            "cut/park_arms",
            lambda: sequential(
                [ParkArmsAction(get_park_arms_argument(context.world))],
                context,
            ).perform(),
        )
        _, _ = _timed(
            "cut/move_torso",
            lambda: sequential([MoveTorsoAction(TorsoState.HIGH)], context).perform(),
        )
        _, _ = _timed(
            "cut/navigate_action",
            lambda: sequential(
                [NavigateAction(pickup_pose, True, teleport=True)],
                context,
            ).perform(),
        )

    current_plan = None
    cutting_action = None
    try:
        with simulated_robot_with_collision:
            cutting_action = CuttingAction(
                container=bread,
                arm=arm,
                tool=tool,
                technique=cutting_technique,
                pointer_stride=CUTTING_POINTER_STRIDE,
                num_cuts_x=num_cuts_x,
                slice_thickness=CUTTING_SLICE_THICKNESS_M,
            )
            current_plan = sequential(
                [cutting_action],
                context,
            )
            motion_context = (
                _pause_ros_state_publishers_during_motion(context.world)
                if PAUSE_ROS_STATE_PUBLISHERS_DURING_CUT
                else nullcontext()
            )
            with motion_context:
                _, _ = _timed("cut/action_plan_perform", current_plan.perform)
            return _motion_progress_from_action(cutting_action)
    except Exception as exc:
        exc.motion_progress = _motion_progress_from_action(cutting_action)
        raise
    finally:
        pass

def _build_cut_geometry_binding(bread, *, cutting_technique, num_cuts_x, robot=None):
    mins, maxs = body_local_aabb(
        bread,
        use_visual=True,
        apply_shape_scale=True,
    )
    size = maxs - mins
    margin_x = min(0.01, 0.15 * size[0])
    margin_y = min(0.01, 0.10 * size[1])
    requested_thickness = max(float(CUTTING_SLICE_THICKNESS_M), 1e-4)
    usable_x = max(0.0, size[0] - 2.0 * margin_x)
    anchor_local_x = mins[0] + margin_x + min(0.5 * requested_thickness, 0.5 * usable_x)
    anchor_local_y = 0.5 * (mins[1] + maxs[1])
    anchor_local_z = mins[2] + max(0.003, 0.05 * size[2])

    pose = bread.global_pose
    pos = np.asarray(pose.to_position().to_np(), dtype=float).reshape(-1)[:3]
    quat = np.asarray(pose.to_quaternion().to_np(), dtype=float).reshape(-1)[:4]
    roll, pitch, yaw = euler_from_quaternion(quat)
    rotation = quaternion_matrix(quat)[:3, :3]
    cut_normal_local = np.array([1.0, 0.0, 0.0], dtype=float)
    cut_normal_world = rotation @ cut_normal_local
    cut_normal_world_yaw = float(np.arctan2(cut_normal_world[1], cut_normal_world[0]))
    robot_pos = np.full(3, np.nan, dtype=float)
    robot_quat = np.full(4, np.nan, dtype=float)
    robot_roll = float("nan")
    robot_pitch = float("nan")
    robot_yaw = float("nan")
    robot_to_object_yaw = float("nan")
    object_yaw_relative_to_robot = float("nan")
    object_yaw_relative_to_approach = float("nan")
    cut_normal_relative_to_robot = float("nan")
    cut_normal_relative_to_approach = float("nan")
    cut_normal_approach_abs_angle = float("nan")
    cut_normal_approach_parallel_score = float("nan")
    cut_normal_approach_perpendicular_score = float("nan")
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
            robot_to_object_yaw = float(
                np.arctan2(float(pos[1] - robot_pos[1]), float(pos[0] - robot_pos[0]))
            )
            object_yaw_relative_to_robot = _wrap_angle_rad(yaw - robot_yaw)
            object_yaw_relative_to_approach = _wrap_angle_rad(
                yaw - robot_to_object_yaw
            )
            cut_normal_relative_to_robot = _wrap_angle_rad(
                cut_normal_world_yaw - robot_yaw
            )
            cut_normal_relative_to_approach = _wrap_angle_rad(
                cut_normal_world_yaw - robot_to_object_yaw
            )
            cut_normal_approach_abs_angle = _fold_parallel_angle_rad(
                cut_normal_relative_to_approach
            )
            cut_normal_approach_parallel_score = float(
                abs(np.cos(cut_normal_relative_to_approach))
            )
            cut_normal_approach_perpendicular_score = float(
                abs(np.sin(cut_normal_relative_to_approach))
            )
        except Exception:
            pass

    eps = 1e-9
    anchor_norm = np.array(
        [
            (anchor_local_x - mins[0]) / max(size[0], eps),
            (anchor_local_y - mins[1]) / max(size[1], eps),
            (anchor_local_z - mins[2]) / max(size[2], eps),
        ],
        dtype=float,
    )

    support = getattr(getattr(bread, "parent_connection", None), "parent", None)
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
        "anchor_local_x": round(float(anchor_local_x), 6),
        "anchor_local_y": round(float(anchor_local_y), 6),
        "anchor_local_z": round(float(anchor_local_z), 6),
        "anchor_norm_x": round(float(anchor_norm[0]), 6),
        "anchor_norm_y": round(float(anchor_norm[1]), 6),
        "anchor_norm_z": round(float(anchor_norm[2]), 6),
        "cut_normal_local_x": round(float(cut_normal_local[0]), 6),
        "cut_normal_local_y": round(float(cut_normal_local[1]), 6),
        "cut_normal_local_z": round(float(cut_normal_local[2]), 6),
        "cut_normal_world_x": round(float(cut_normal_world[0]), 6),
        "cut_normal_world_y": round(float(cut_normal_world[1]), 6),
        "cut_normal_world_z": round(float(cut_normal_world[2]), 6),
        "cut_normal_world_yaw_rad": round(cut_normal_world_yaw, 6),
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
        "robot_to_object_yaw_rad": (
            round(float(robot_to_object_yaw), 6)
            if np.isfinite(robot_to_object_yaw)
            else ""
        ),
        "object_yaw_relative_to_robot_rad": (
            round(float(object_yaw_relative_to_robot), 6)
            if np.isfinite(object_yaw_relative_to_robot)
            else ""
        ),
        "object_yaw_relative_to_approach_rad": (
            round(float(object_yaw_relative_to_approach), 6)
            if np.isfinite(object_yaw_relative_to_approach)
            else ""
        ),
        "cut_normal_relative_to_robot_rad": (
            round(float(cut_normal_relative_to_robot), 6)
            if np.isfinite(cut_normal_relative_to_robot)
            else ""
        ),
        "cut_normal_relative_to_approach_rad": (
            round(float(cut_normal_relative_to_approach), 6)
            if np.isfinite(cut_normal_relative_to_approach)
            else ""
        ),
        "cut_normal_approach_abs_angle_rad": (
            round(float(cut_normal_approach_abs_angle), 6)
            if np.isfinite(cut_normal_approach_abs_angle)
            else ""
        ),
        "cut_normal_approach_parallel_score": (
            round(float(cut_normal_approach_parallel_score), 6)
            if np.isfinite(cut_normal_approach_parallel_score)
            else ""
        ),
        "cut_normal_approach_perpendicular_score": (
            round(float(cut_normal_approach_perpendicular_score), 6)
            if np.isfinite(cut_normal_approach_perpendicular_score)
            else ""
        ),
        "technique_name": cutting_technique,
        "slice_thickness_m": round(float(CUTTING_SLICE_THICKNESS_M), 6),
        "num_cuts_x": int(num_cuts_x),
        "pointer_stride": int(CUTTING_POINTER_STRIDE),
    }


def main_cutting(
    seed=None,
    robot_name=None,
    environment_name=None,
    object_kind="bread",
    object_name=None,
    container_kind=None,
    container_name=None,
):

    if object_name is not None:
        object_kind = object_name
    if container_name is not None:
        object_kind = container_name
    if container_kind is not None:
        object_kind = container_kind
    object_cfg = get_cut_object_config(object_kind)
    default_object_color = _cut_object_default_color(object_kind)
    cut_cfg = _cut_object_execution_config(object_kind)
    effective_seed = (
        int(seed)
        if seed is not None
        else int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    )
    world, _, surface_plan = setup_random_bread_world(
        seed=effective_seed,
        robot_name=robot_name,
        environment_name=environment_name,
        object_kind=object_kind,
    )

    print("[setup] starting ROS runtime and RViz publishers", flush=True)
    node, _ = _timed(
        "cut/setup_runtime",
        lambda: setup_experiment_runtime(
            world=world,
            node_name="pycram_cut_all_breads_retry",
        ),
    )

    resolved_robot_name = resolve_robot_name(robot_name)
    pose_scorer = get_pose_scorer("cutting", resolved_robot_name)
    if pose_scorer is not None:
        print(
            f"[setup] using JPT base-pose sampler for cutting/{pose_scorer.robot}",
            flush=True,
        )
    else:
        print(
            f"[setup] no cutting JPT for robot '{resolved_robot_name}'; "
            "using costmap base-pose sampling",
            flush=True,
        )
    print("[setup] attaching cutting tools", flush=True)
    arm_tools, _ = _timed(
        "cut/attach_tools",
        lambda: attach_available_tools(
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
        ),
    )
    print("[setup] collecting cutting targets", flush=True)
    breads = collect_named_targets(world, f"{object_cfg['object_name_prefix']}_")

    print("[setup] building execution context", flush=True)
    context = Context.from_world(world)
    context.ros_node = node
    context.robot.full_body_controlled = True
    robot_name = _robot_name(context.robot)
    world_name = environment_name
    run_id = new_run_id()
    print("[setup] closing grippers", flush=True)
    with simulated_robot_without_collision:
        _, _ = _timed(
            "cut/close_grippers",
            lambda: sequential(
                [SetGripperAction(Arms.BOTH, GripperState.CLOSE)],
                context,
            ).perform(),
        )

    print("[setup] surface plan:")
    print(f"[setup] seed: {effective_seed}")
    for surface_name, area_m2, target_count, placed_count in surface_plan:
        print(
            f"  - {surface_name}: area={area_m2:.3f}m^2 target={target_count} placed={placed_count}"
        )
    print(f"[setup] {object_cfg['object_label']}s to cut: {len(breads)}")

    success_primary = 0
    success_fallback = 0
    failed = 0
    failed_breads = set()
    successful_breads = set()
    bread_results = []
    initialize_csv(RESULTS_CSV_PATH, _results_csv_fieldnames())
    debug_costmap_publishers = {}

    total_breads = len(breads)
    for bread_index, bread in enumerate(breads, start=1):
        bread_name = _body_name(bread)
        cut_count = success_primary + success_fallback
        print(
            f"[Demo time] {cut_count}/{total_breads} cut breads "
            f"(next {bread_index}/{total_breads}: {bread_name})"
        )
        bread_start_time = time.perf_counter()
        attempt_failures = []
        attempt_count = 0
        collision_failure_count = 0
        perturbation_applied = False
        perturbation_type = ""
        last_motion_progress = _empty_motion_progress()
        debug_costmap_publishers, preview_elapsed = _timed(
            f"bread/{bread_name}/costmap_preview",
            lambda: _update_costmap_debug_publishers(
                node, context.robot, world, bread, debug_costmap_publishers
            ),
        )
        candidate_scorer = (
            (lambda pose: pose_scorer.score(compute_cutting_cause_features(pose, bread)))
            if pose_scorer is not None
            else None
        )
        pose_candidate_reranker = (
            (
                lambda candidates: rerank_by_score(
                    candidates,
                    lambda pose: compute_cutting_cause_features(pose, bread),
                    pose_scorer,
                )
            )
            if pose_scorer is not None
            else None
        )
        pickup_loc, _ = _timed(
            f"bread/{bread_name}/pickup_loc_build",
            lambda: CostmapLocation(
                target=bread.global_pose,
                reachable=True,
                reachable_arm=arm_tools[0][0] if arm_tools else None,
                validate_reachability=False,
                samples=1000,
                costmap_width=CUTTING_COSTMAP_WIDTH,
                costmap_height=CUTTING_COSTMAP_HEIGHT,
                costmap_resolution=CUTTING_COSTMAP_RESOLUTION,
                ring_std=CUTTING_RING_STD,
                ring_distance=CUTTING_RING_DISTANCE,
                obstacle_clearance=_cutting_obstacle_clearance(context.robot),
                context=context,
                candidate_reranker=pose_candidate_reranker,
            ),
        )
        highlight_elapsed = 0.0
        if ENABLE_CUTTING_HIGHLIGHT:
            _, highlight_elapsed = _timed(
                f"bread/{bread_name}/highlight",
                lambda: highlight_current_target(
                    world,
                    breads,
                    bread,
                    default_color=default_object_color,
                    active_color=ACTIVE_BREAD_COLOR,
                    failed_color=FAILED_BREAD_COLOR,
                    success_color=SUCCESS_BREAD_COLOR,
                    failed_targets=failed_breads,
                    successful_targets=successful_breads,
                ),
            )
        common_result_kwargs = {
            "task_instance_id": bread_name,
            "experiment_condition": EXPERIMENT_CONDITION,
            "baseline_name": BASELINE_NAME,
            "task_name": TASK_NAME,
            "seed": effective_seed,
            "world_name": world_name,
            "run_id": run_id,
            "pose_sampler": "costmap_fallback",
            "jpt_pose_loglik": "",
        }
        try:
            pickup_pose, pickup_resolve_elapsed = _timed(
                f"bread/{bread_name}/pickup_loc_resolve",
                lambda: resolve_navigation_target_for_environment(
                    pickup_loc,
                    description=f"cutting {bread.name}",
                    environment_name=environment_name,
                )[0],
            )
        except Exception as exc:
            failed += 1
            failed_breads.add(bread)
            attempt_failures.append(f"navigation setup -> {_format_attempt_error(exc)}")
            fallback_tool = arm_tools[-1][1] if arm_tools else None
            result_row = _record_bread_result(
                bread_results,
                bread_name,
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
                final_success=False,
                total_attempts=attempt_count,
                retry_count=0,
                collision_failure_count=collision_failure_count,
                recovery_used=False,
                recovery_success=False,
                perturbation_applied=perturbation_applied,
                perturbation_type=perturbation_type,
                execution_time_s=time.perf_counter() - bread_start_time,
                geometry_binding=_build_cut_geometry_binding(
                    bread,
                    cutting_technique=cut_cfg["technique"],
                    num_cuts_x=cut_cfg["num_cuts_x"],
                    robot=context.robot,
                ),
                motion_progress=last_motion_progress,
            )
            append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
            print(
                f"[skip] {bread_name}: pickup pose resolution failed "
                f"({type(exc).__name__}: {exc})"
            )
            if DEBUG_PROFILE_CUTTING:
                print(
                    f"[profile] bread/{bread_name}/setup_failed_total: "
                    f"{time.perf_counter() - bread_start_time:.3f}s"
                )
            continue
        if candidate_scorer is not None:
            chosen_score = candidate_scorer(pickup_pose)
            if chosen_score is not None and np.isfinite(chosen_score):
                common_result_kwargs["pose_sampler"] = "jpt"
                common_result_kwargs["jpt_pose_loglik"] = round(float(chosen_score), 8)
        # Orient the bread correctly for the chosen base pose (cut-normal parallel to
        # the approach). Object orientation is therefore correct by construction and
        # no longer a source of failure, so no rotation-retry attempts are needed.
        _align_bread_to_approach(world, bread, pickup_pose)
        arm_attempt_groups = [("primary", arm_tools, None)]
        attempt_succeeded = False

        for group_index, (phase_name, current_arm_tools, rotation_delta) in enumerate(
            arm_attempt_groups
        ):
            for attempt_index, (arm, tool) in enumerate(current_arm_tools):
                is_primary_phase = group_index == 0 and attempt_index == 0
                is_fallback_phase = group_index == 0 and attempt_index > 0
                is_rotated_phase = group_index > 0
                decision = (
                    "cut"
                    if attempt_index == 0 and not is_rotated_phase
                    else (
                        "retry_after_rotation"
                        if is_rotated_phase
                        else "retry_with_left_arm"
                    )
                )
                decision_reason = (
                    "primary_success"
                    if attempt_index == 0 and not is_rotated_phase
                    else (
                        phase_name
                        if is_rotated_phase
                        else "right_arm_failed"
                    )
                )
                print(
                    f"[cut] {bread_name}: try {arm.name} arm"
                    + (f" after {phase_name}" if is_rotated_phase else "")
                )
                try:
                    attempt_count += 1
                    last_motion_progress = _try_cut(
                        context,
                        bread,
                        pickup_pose,
                        arm,
                        tool,
                        cutting_technique=cut_cfg["technique"],
                        num_cuts_x=cut_cfg["num_cuts_x"],
                        environment_name=environment_name,
                    )
                    if is_primary_phase:
                        success_primary += 1
                    elif is_fallback_phase:
                        success_fallback += 1
                    cut_count = success_primary + success_fallback
                    print(f"[Demo time] {cut_count}/{total_breads} cut breads")
                    successful_breads.add(bread)
                    result_row = _record_bread_result(
                        bread_results,
                        bread_name,
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
                        final_success=True,
                        total_attempts=attempt_count,
                        retry_count=max(0, attempt_count - 1),
                        collision_failure_count=collision_failure_count,
                        recovery_used=attempt_count > 1,
                        recovery_success=attempt_count > 1,
                        perturbation_applied=perturbation_applied,
                        perturbation_type=perturbation_type,
                        execution_time_s=time.perf_counter() - bread_start_time,
                        geometry_binding=_build_cut_geometry_binding(
                            bread,
                            cutting_technique=cut_cfg["technique"],
                            num_cuts_x=cut_cfg["num_cuts_x"],
                            robot=context.robot,
                        ),
                        motion_progress=last_motion_progress,
                    )
                    append_csv_row(
                        RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row
                    )
                    suffix = (
                        f" after {phase_name}"
                        if is_rotated_phase
                        else (" (fallback)" if attempt_index > 0 else "")
                    )
                    print(f"[ok] {bread_name}: cut with {arm.name} arm{suffix}")
                    if DEBUG_PROFILE_CUTTING:
                        print(
                            f"[profile] bread/{bread_name}/summary: "
                            f"preview={preview_elapsed:.3f}s "
                            f"pickup_resolve={pickup_resolve_elapsed:.3f}s "
                            f"highlight={highlight_elapsed:.3f}s "
                            f"total={time.perf_counter() - bread_start_time:.3f}s"
                        )
                    attempt_succeeded = True
                    break
                except TimeoutError as exc:
                    last_motion_progress = _motion_progress_from_exception(exc)
                    collision_failure_count += 1
                    attempt_failures.append(
                        f"{arm.name} {phase_name} -> {_format_attempt_error(exc)}"
                    )
                    print(
                        f"[{'retry' if not (group_index == len(arm_attempt_groups) - 1 and attempt_index == len(current_arm_tools) - 1) else 'fail'}] "
                        f"{bread_name}: {arm.name}"
                        + (f" after {phase_name}" if is_rotated_phase else "")
                        + f" timed out ({type(exc).__name__}: {exc})"
                    )
                except Exception as exc:
                    last_motion_progress = _motion_progress_from_exception(exc)
                    if _is_collision_like_failure(exc):
                        collision_failure_count += 1
                    attempt_failures.append(
                        f"{arm.name} {phase_name} -> {_format_attempt_error(exc)}"
                    )
                    print(
                        f"[{'retry' if not (group_index == len(arm_attempt_groups) - 1 and attempt_index == len(current_arm_tools) - 1) else 'fail'}] "
                        f"{bread_name}: {arm.name}"
                        + (f" after {phase_name}" if is_rotated_phase else "")
                        + f" failed ({type(exc).__name__}: {exc})"
                    )
            if attempt_succeeded:
                break

        if attempt_succeeded:
            continue

        failed += 1
        failed_breads.add(bread)
        last_tool = arm_tools[-1][1]
        result_row = _record_bread_result(
            bread_results,
            bread_name,
            robot_name,
            "failed",
            "",
            _tool_name(last_tool),
            "primary",
            attempt_failures,
            **common_result_kwargs,
            feasibility_reason="collision_or_motion_failure",
            robot_decision="task_failed",
            decision_reason="all_cut_attempts_failed",
            final_success=False,
            total_attempts=attempt_count,
            retry_count=max(0, attempt_count - 1),
            collision_failure_count=collision_failure_count,
            recovery_used=attempt_count > 1,
            recovery_success=False,
            perturbation_applied=perturbation_applied,
            perturbation_type=perturbation_type,
            execution_time_s=time.perf_counter() - bread_start_time,
            geometry_binding=_build_cut_geometry_binding(
                bread,
                cutting_technique=cut_cfg["technique"],
                num_cuts_x=cut_cfg["num_cuts_x"],
                robot=context.robot,
            ),
            motion_progress=last_motion_progress,
        )
        append_csv_row(RESULTS_CSV_PATH, _results_csv_fieldnames(), result_row)
        if DEBUG_PROFILE_CUTTING:
            print(
                f"[profile] bread/{bread_name}/failed_total: "
                f"{time.perf_counter() - bread_start_time:.3f}s"
            )

    if ENABLE_CUTTING_HIGHLIGHT:
        highlight_current_target(
            world,
            breads,
            None,
            default_color=default_object_color,
            active_color=ACTIVE_BREAD_COLOR,
            failed_color=FAILED_BREAD_COLOR,
            success_color=SUCCESS_BREAD_COLOR,
            failed_targets=failed_breads,
            successful_targets=successful_breads,
        )

    print("[summary]")
    print(f"  total breads: {len(breads)}")
    print(f"  success primary (RIGHT): {success_primary}")
    print(f"  success fallback (LEFT): {success_fallback}")
    print(f"  failed all attempts: {failed}")
    print(f"  results csv: {RESULTS_CSV_PATH}")

    shutdown_experiment_runtime(node)
