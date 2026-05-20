import csv
import os
from collections import defaultdict
from statistics import mean


BASE_SUMMARY_FIELDS = [
    "run_id",
    "task_name",
    "task_instance_id",
    "seed",
    "experiment_condition",
    "baseline_name",
    "robot_name",
    "success",
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
    "execution_time_s",
    "failure_count",
    "failure_reasons",
]


BASE_AGGREGATED_FIELDS = [
    "run_id",
    "task_name",
    "experiment_condition",
    "baseline_name",
    "robot_name",
    "n_trials",
    "success_rate",
    "avg_execution_time_s",
    "avg_retry_count",
    "avg_total_attempts",
    "avg_collision_failure_count",
    "recovery_used_rate",
    "recovery_success_rate",
    "success_under_perturbation_rate",
]


def load_rows(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def append_csv(csv_path, fieldnames, rows):
    if not rows:
        return
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    file_exists = os.path.exists(csv_path)
    needs_header = (not file_exists) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no", ""):
        return False
    return None


def to_int(value):
    if value in (None, ""):
        return 0
    return int(value)


def to_float(value):
    if value in (None, ""):
        return 0.0
    return float(value)


def safe_mean(values):
    return mean(values) if values else 0.0


def build_trial_summary(rows, extra_fields=None, extra_row_fn=None):
    extra_fields = extra_fields or []
    summary = []
    for row in rows:
        summary_row = {
            "run_id": row.get("run_id"),
            "task_name": row.get("task_name"),
            "task_instance_id": row.get("task_instance_id"),
            "seed": row.get("seed"),
            "experiment_condition": row.get("experiment_condition"),
            "baseline_name": row.get("baseline_name"),
            "robot_name": row.get("robot_name"),
            "success": row.get("outcome") == "success",
            "succeeded_arm": row.get("succeeded_arm"),
            "tool_name": row.get("tool_name"),
            "phase": row.get("phase"),
            "total_attempts": to_int(row.get("total_attempts")),
            "retry_count": to_int(row.get("retry_count")),
            "collision_failure_count": to_int(row.get("collision_failure_count")),
            "recovery_used": to_bool(row.get("recovery_used")),
            "recovery_success": to_bool(row.get("recovery_success")),
            "perturbation_applied": to_bool(row.get("perturbation_applied")),
            "perturbation_type": row.get("perturbation_type"),
            "execution_time_s": to_float(row.get("execution_time_s")),
            "failure_count": to_int(row.get("failure_count")),
            "failure_reasons": row.get("failure_reasons"),
        }
        if extra_row_fn is not None:
            summary_row.update(extra_row_fn(row))
        else:
            for field in extra_fields:
                summary_row[field] = row.get(field)
        summary.append(summary_row)
    return summary


def build_aggregated(rows, extra_group_fields=None, extra_aggregate_fn=None):
    extra_group_fields = extra_group_fields or []
    groups = defaultdict(list)
    for row in rows:
        key = tuple(
            [
                row.get("run_id"),
                row.get("task_name"),
                row.get("experiment_condition"),
                row.get("baseline_name"),
                row.get("robot_name"),
                *[row.get(field) for field in extra_group_fields],
            ]
        )
        groups[key].append(row)

    aggregated = []
    for key, items in groups.items():
        (
            run_id,
            task_name,
            experiment_condition,
            baseline_name,
            robot_name,
            *extra_values,
        ) = key
        success_values = [
            1.0 if row.get("outcome") == "success" else 0.0 for row in items
        ]
        execution_times = [to_float(row.get("execution_time_s")) for row in items]
        retry_counts = [to_int(row.get("retry_count")) for row in items]
        total_attempts = [to_int(row.get("total_attempts")) for row in items]
        collision_failures = [
            to_int(row.get("collision_failure_count")) for row in items
        ]
        recovery_used = [
            1.0 if to_bool(row.get("recovery_used")) else 0.0 for row in items
        ]
        recovery_success = [
            1.0 if to_bool(row.get("recovery_success")) else 0.0 for row in items
        ]
        perturbation_success = [
            1.0
            for row in items
            if to_bool(row.get("perturbation_applied"))
            and row.get("outcome") == "success"
        ]
        perturbation_total = [
            1.0 for row in items if to_bool(row.get("perturbation_applied"))
        ]

        aggregated_row = {
            "run_id": run_id,
            "task_name": task_name,
            "experiment_condition": experiment_condition,
            "baseline_name": baseline_name,
            "robot_name": robot_name,
            "n_trials": len(items),
            "success_rate": round(safe_mean(success_values), 4),
            "avg_execution_time_s": round(safe_mean(execution_times), 4),
            "avg_retry_count": round(safe_mean(retry_counts), 4),
            "avg_total_attempts": round(safe_mean(total_attempts), 4),
            "avg_collision_failure_count": round(safe_mean(collision_failures), 4),
            "recovery_used_rate": round(safe_mean(recovery_used), 4),
            "recovery_success_rate": round(safe_mean(recovery_success), 4),
            "success_under_perturbation_rate": round(
                (
                    (sum(perturbation_success) / len(perturbation_total))
                    if perturbation_total
                    else 0.0
                ),
                4,
            ),
        }
        for field, value in zip(extra_group_fields, extra_values):
            aggregated_row[field] = value
        if extra_aggregate_fn is not None:
            aggregated_row.update(extra_aggregate_fn(items))
        aggregated.append(aggregated_row)
    return aggregated
