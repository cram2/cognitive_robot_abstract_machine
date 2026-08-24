"""
Regenerate the ORM interfaces of every package that has one.

Runnable from any working directory: the interfaces are resolved relative to the
installed :mod:`cognitive_robot_abstract_machine` package.
"""

from __future__ import annotations

import argparse

from cognitive_robot_abstract_machine.orm_interfaces import WORKSPACE_ORM_INTERFACES


def main() -> None:
    """
    Build every ORM interface of this repository anew.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Let the generators write their own logging to the terminal, in place of the progress bar, to follow what a build does."
        ),
    )
    arguments = parser.parse_args()

    WORKSPACE_ORM_INTERFACES.regenerate(show_generator_output=arguments.debug)


if __name__ == "__main__":
    main()
