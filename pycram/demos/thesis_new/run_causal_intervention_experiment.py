"""
Run the controlled causal intervention experiment from PyCharm.

This file intentionally has no command-line arguments. Press Run in PyCharm to
execute the planned do(robot_name=...) runs.

Default plan:
    task        = cut
    environment = apartment
    seeds       = 910001..910005
    robots      = pr2, hsrb

This creates true paired intervention data:
    same environment + same seed + same bread_name
    only robot_name changes
"""

from src.causal_intervention_experiment import (
    build_manifest,
    execute_manifest,
    normalize_environment_name,
    normalize_robot_name,
    write_manifest,
)

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
