import csv
import os
import uuid

from pycram.plans.failures import NavigationGoalNotReachedError


BASE_RESULT_FIELDNAMES = [
    "task_name",
    "run_id",
    "task_instance_id",
    "seed",
    "world_name",
    "experiment_condition",
    "baseline_name",
    "feasibility_reason",
    "robot_decision",
    "decision_reason",
    "final_success",
    "robot_name",
    "outcome",
    "succeeded_arm",
    "tool_name",
    "phase",
    "total_attempts",
    "retry_count",
    "collision_failure_count",
    "recovery_used",
    "recovery_success",
    "perturbation_applied",
    "perturbation_type",
    "object_aabb_min_x",
    "object_aabb_min_y",
    "object_aabb_min_z",
    "object_aabb_max_x",
    "object_aabb_max_y",
    "object_aabb_max_z",
    "object_size_x",
    "object_size_y",
    "object_size_z",
    "object_volume_aabb",
    "target_world_x",
    "target_world_y",
    "target_world_z",
    "support_surface_name",
    "support_world_x",
    "support_world_y",
    "support_world_z",
    "support_yaw_rad",
    "support_size_x",
    "support_size_y",
    "support_size_z",
    "object_world_x",
    "object_world_y",
    "object_world_z",
    "object_quat_x",
    "object_quat_y",
    "object_quat_z",
    "object_quat_w",
    "object_roll_rad",
    "object_pitch_rad",
    "object_yaw_rad",
    "anchor_local_x",
    "anchor_local_y",
    "anchor_local_z",
    "anchor_norm_x",
    "anchor_norm_y",
    "anchor_norm_z",
    "cut_normal_local_x",
    "cut_normal_local_y",
    "cut_normal_local_z",
    "cut_normal_world_x",
    "cut_normal_world_y",
    "cut_normal_world_z",
    "cut_normal_world_yaw_rad",
    "robot_world_x",
    "robot_world_y",
    "robot_world_z",
    "robot_quat_x",
    "robot_quat_y",
    "robot_quat_z",
    "robot_quat_w",
    "robot_roll_rad",
    "robot_pitch_rad",
    "robot_yaw_rad",
    "robot_to_object_yaw_rad",
    "object_yaw_relative_to_robot_rad",
    "object_yaw_relative_to_approach_rad",
    "cut_normal_relative_to_robot_rad",
    "cut_normal_relative_to_approach_rad",
    "cut_normal_approach_abs_angle_rad",
    "cut_normal_approach_parallel_score",
    "cut_normal_approach_perpendicular_score",
    "technique_name",
    "slice_thickness_m",
    "num_cuts_x",
    "pointer_stride",
    "motion_approach_completed",
    "motion_waypoint_count",
    "motion_stopped_waypoint_index",
    "motion_stopped_waypoint_fraction",
    "motion_stopped_waypoint_x",
    "motion_stopped_waypoint_y",
    "motion_stopped_waypoint_z",
    "motion_stopped_distance_m",
    "motion_progress_note",
    "execution_time_s",
    "failure_count",
    "failure_reasons",
]


def body_name(body):
    maybe_name = getattr(body, "name", None)
    if hasattr(maybe_name, "name"):
        maybe_name = maybe_name.name
    return maybe_name if isinstance(maybe_name, str) else ""


def robot_name(robot):
    if robot is None:
        return ""
    return body_name(robot) or getattr(robot.__class__, "__name__", "")


def tool_name(tool):
    if tool is None:
        return ""
    root = getattr(tool, "root", None)
    if root is not None:
        return body_name(root)
    return getattr(tool.__class__, "__name__", "")


def format_attempt_error(exc):
    if isinstance(exc, TimeoutError):
        return f"CollisionError (reported as timeout): {exc}"
    return f"{type(exc).__name__}: {exc}"


def is_collision_like_failure(exc):
    return isinstance(exc, (TimeoutError, NavigationGoalNotReachedError))


def new_run_id():
    return str(uuid.uuid4())


def initialize_csv(csv_path, fieldnames):
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        ensure_csv_fieldnames(csv_path, fieldnames)
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def ensure_csv_fieldnames(csv_path, fieldnames):
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        existing_fieldnames = reader.fieldnames or []
        if all(fieldname in existing_fieldnames for fieldname in fieldnames):
            return
        rows = list(reader)

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {fieldname: row.get(fieldname, "") for fieldname in fieldnames}
            )


def append_csv_row(csv_path, fieldnames, row):
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    file_exists = os.path.exists(csv_path)
    needs_header = (not file_exists) or os.path.getsize(csv_path) == 0
    if file_exists and not needs_header:
        ensure_csv_fieldnames(csv_path, fieldnames)
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def build_base_result_row(
    results,
    robot_name,
    outcome,
    succeeded_arm,
    tool_name,
    phase,
    failures,
    **kwargs,
):
    row = {
        "robot_name": robot_name,
        "outcome": outcome,
        "succeeded_arm": succeeded_arm,
        "tool_name": tool_name,
        "phase": phase,
        "failure_count": len(failures),
        "failure_reasons": " | ".join(failures),
        **kwargs,
    }
    results.append(row)
    return row
