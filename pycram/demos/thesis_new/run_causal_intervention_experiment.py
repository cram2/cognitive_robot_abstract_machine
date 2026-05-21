"""
Run the controlled causal intervention experiment from PyCharm.

This file intentionally has no command-line arguments. Press Run in PyCharm to
execute the planned do(robot_name=...) runs.

Default plan:
    tasks       = cut, mix, wipe
    environments = kitchen, apartment, isr
    seeds       = 910001..910005
    robots      = tiago, justin, stretch, hsrb, pr2, armar7, g1

This creates true paired intervention data:
    same task + same environment + same seed + same task_instance_id
    only robot_name changes
"""

from src.causal_intervention_experiment import (
    build_manifest,
    execute_manifest,
    normalize_environment_name,
    normalize_robot_name,
    normalize_task_name,
    write_manifest,
)

TASKS = ( "mix", "wipe", "cut")
ROBOTS = (
    "tiago",
    "justin",
    "stretch",
    "hsrb",
    "pr2",
    "armar7",
    "g1",
)
ENVIRONMENTS = (
    "kitchen",
    "apartment",
    "isr",
)
SEEDS = tuple(range(910001, 910006))


def main() -> None:
    manifest = build_manifest(
        tasks=tuple(normalize_task_name(task) for task in TASKS),
        robots=tuple(normalize_robot_name(robot) for robot in ROBOTS),
        environments=tuple(
            normalize_environment_name(environment) for environment in ENVIRONMENTS
        ),
        seeds=SEEDS,
    )
    write_manifest(manifest)
    execute_manifest(manifest)


if __name__ == "__main__":
    main()
