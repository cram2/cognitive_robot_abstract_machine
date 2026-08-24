"""
Running the ORM interface generators of a checkout, one after another in one
interpreter.

Every generator imports the packages it maps, and the packages of this repository build
on one another, so a generator started on its own re-imports what the previous one
already imported. Running them here pays for each import once.

Started as ``python -P -m cognitive_robot_abstract_machine.orm_generation <generator>...``
by :mod:`cognitive_robot_abstract_machine.orm_interfaces`, which reads the reports below
back from this run's output.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import runpy
from contextlib import chdir, contextmanager
from dataclasses import dataclass
from pathlib import Path

from krrood.class_diagrams.progress_report import is_progress_wanted
from typing_extensions import Iterator, Optional

LINE_PREFIX = "cram-orm-generator-started "
"""
What marks a line of output as this run naming a generator rather than as something the
generator itself wrote.
"""

# %% what a run reports about itself


@dataclass
class GeneratorStarted:
    """
    The generator of one package, about to run.
    """

    package_name: str
    """
    Name of the package whose interface is being generated.
    """

    @classmethod
    def from_line(cls, line: str) -> Optional[GeneratorStarted]:
        """
        Read a report back from a line of this run's output.

        :param line: One line of output, of any kind.
        :return: The report the line carries, or nothing when it carries none.
        """
        if not line.startswith(LINE_PREFIX):
            return None
        return cls(line[len(LINE_PREFIX) :].strip())

    def to_line(self) -> str:
        """
        Render this report as one line of output.

        :return: The line, without its terminating line break.
        """
        return LINE_PREFIX + self.package_name


# %% running the generators


@contextmanager
def restored_root_logging() -> Iterator[None]:
    """
    Undo what the enclosed code configures on the root logger.

    A generator raises the verbosity of the code it drives for its own run; sharing an
    interpreter would otherwise apply the first one's configuration to all the following
    ones.
    """
    root_logger = logging.getLogger()
    handlers = list(root_logger.handlers)
    level = root_logger.level
    yield
    root_logger.handlers = handlers
    root_logger.setLevel(level)


def run_generator(generator: Path) -> None:
    """
    Run one package's generator, and let the ones after it read what it wrote.

    :param generator: The package's ``generate_orm.py``.
    """
    if is_progress_wanted():
        print(GeneratorStarted(generator.parents[1].name).to_line(), flush=True)
    with chdir(generator.parent), restored_root_logging():
        runpy.run_path(str(generator), run_name="__main__")
    # The interfaces are deleted before a build, so this one is a file that did not
    # exist when its folder was last scanned for imports.
    importlib.invalidate_caches()


def main() -> None:
    """
    Run every generator named on the command line, in the order given.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "generators",
        nargs="+",
        type=Path,
        help="The generate_orm.py scripts to run, in dependency order.",
    )
    for generator in parser.parse_args().generators:
        run_generator(generator)


if __name__ == "__main__":
    main()
