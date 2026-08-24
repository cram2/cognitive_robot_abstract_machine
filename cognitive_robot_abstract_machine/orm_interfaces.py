"""
The ORM interfaces the packages of this repository generate with ORMatic.

The interfaces are generated rather than written, so the repository ignores them instead
of tracking them: a fresh checkout carries no database mapping at all, and nothing can
be persisted or turned into a data access object until they have been generated once.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from krrood.class_diagrams.progress_report import (
    ClassDiagramProgress,
    ProgressEnvironmentVariable,
)
from tqdm import tqdm
from typing_extensions import List, Optional, Sequence, Tuple

from cognitive_robot_abstract_machine import orm_generation
from cognitive_robot_abstract_machine.exceptions import (
    MissingOrmGeneratorError,
    OrmGenerationFailedError,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
"""
Root of the checkout this package is installed from.
"""

INTERFACE_FILE_NAME = "ormatic_interface.py"
"""
Name every package's generator writes its interface to.
"""

PROGRESS_DESCRIPTION = "Building ORM interfaces"
"""
What the progress bar of a build calls itself.
"""

PROGRESS_REQUESTED = "1"
"""
What a generator is told to report the classes it finishes.
"""

# %% what a build shows while it runs


@dataclass
class BuildProgress:
    """
    A bar counting the classes of the interface being built, and how many of the
    interfaces are done.
    """

    total_interfaces: int
    """
    How many interfaces the build covers.
    """

    show_generator_output: bool
    """
    Whether the generators write to the terminal, which leaves no room for a bar.
    """

    completed_interfaces: int = field(default=0, init=False)
    """
    How many of them are built.
    """

    bar: Optional[tqdm] = field(default=None, init=False)
    """
    The bar, absent while the generators have the terminal to themselves.
    """

    counted_classes: bool = field(default=False, init=False)
    """
    Whether the interface being built has said how many classes it holds.
    """

    def __enter__(self) -> BuildProgress:
        if not self.show_generator_output:
            self.bar = tqdm(unit="class")
            self.show_interfaces_done()
        return self

    def __exit__(self, *exception: object) -> None:
        if self.bar is not None:
            self.bar.close()

    def show_interfaces_done(self) -> None:
        """
        Put how far along the interfaces are beside the bar.
        """
        self.bar.set_description_str(
            f"{PROGRESS_DESCRIPTION} {self.completed_interfaces}/{self.total_interfaces}"
        )

    def start(self, package_name: str) -> None:
        """
        Begin reporting on the interface of a package.

        :param package_name: The package whose interface is being built.
        """
        self.counted_classes = False
        if self.bar is None:
            return
        self.bar.set_postfix_str(package_name)

    def advance(self, report: ClassDiagramProgress) -> None:
        """
        Count one class of the interface being built as done.

        :param report: What the generator said about the class it finished.
        """
        if self.bar is None:
            return
        if not self.counted_classes:
            self.bar.reset(total=report.total_classes)
            self.counted_classes = True
            self.show_interfaces_done()
        self.bar.update(1)

    def finish(self) -> None:
        """
        Count the interface being built as done.
        """
        self.completed_interfaces += 1
        if self.bar is None:
            return
        self.show_interfaces_done()


# %% a single package's interface


@dataclass
class OrmInterface:
    """
    The ORM interface a single package generates.
    """

    package_name: str
    """
    Name of the package, which is also the name of its source folder and module.
    """

    repository_root: Path
    """
    Root of the checkout the package lives in.
    """

    dependencies: Tuple[str, ...] = ()
    """
    Names of the packages whose ORM model this interface builds on.
    """

    @property
    def generator(self) -> Path:
        """
        The script that generates this interface.
        """
        return self.repository_root / self.package_name / "scripts" / "generate_orm.py"

    @property
    def path(self) -> Path:
        """
        The generated interface file.
        """
        return (
            self.repository_root
            / self.package_name
            / "src"
            / self.package_name
            / "orm"
            / INTERFACE_FILE_NAME
        )

    def remove(self) -> None:
        """
        Delete the interface, so that a stale version cannot be imported while the new
        one is generated.
        """
        self.path.unlink(missing_ok=True)

    def require_generator(self) -> None:
        """
        Make sure the package still has the script that generates its interface.

        :raises MissingOrmGeneratorError: If the package has no generator.
        """
        if not self.generator.exists():
            raise MissingOrmGeneratorError(self.package_name, self.generator)


# %% reading a run's output back


@dataclass
class GenerationOutput:
    """
    What a generation run writes, read back one line at a time.
    """

    progress: BuildProgress
    """
    What the classes and interfaces the run finishes are reported to.
    """

    package_name: Optional[str] = field(default=None, init=False)
    """
    The package whose generator is running, absent until the run names one.
    """

    written: List[str] = field(default_factory=list, init=False)
    """
    The lines carrying no report, kept for a failure to quote.
    """

    def read(self, line: str) -> None:
        """
        Take in one line of the run's output.

        :param line: The line, as the run wrote it.
        """
        started = orm_generation.GeneratorStarted.from_line(line)
        if started is not None:
            self.start(started.package_name)
            return
        report = ClassDiagramProgress.from_line(line)
        if report is None:
            self.written.append(line)
            return
        self.progress.advance(report)

    def start(self, package_name: str) -> None:
        """
        Begin reading the output of a package's generator.

        :param package_name: The package the run has moved on to.
        """
        if self.package_name is not None:
            self.progress.finish()
        self.package_name = package_name
        self.progress.start(package_name)

    def finish(self) -> None:
        """
        Count the interface being generated as done.
        """
        if self.package_name is None:
            return
        self.progress.finish()


# %% every interface of the repository


@dataclass
class WorkspaceOrmInterfaces:
    """
    The ORM interfaces of a checkout, as one unit.
    """

    interfaces: Sequence[OrmInterface]
    """
    The interfaces in the order they are built, which follows their dependencies.
    """

    def regenerate(self, show_generator_output: bool = False) -> None:
        """
        Build every interface anew, from an empty state and in dependency order.

        ..note:: This takes about a minute, since every package's generator introspects
            its whole class hierarchy.

        :param show_generator_output: Whether to let the generators write to the
            terminal. Their logging and the progress bar cannot share it, so asking for
            one leaves out the other.
        """
        for interface in self.interfaces:
            interface.remove()
        for interface in self.interfaces:
            interface.require_generator()

        with BuildProgress(len(self.interfaces), show_generator_output) as progress:
            if show_generator_output:
                self.run_writing_to_the_terminal()
                return
            self.run_reporting_to(progress)

    @property
    def command(self) -> List[str]:
        """
        The command that runs every generator in one interpreter.

        ..note:: ``-P`` keeps the working directory off :data:`sys.path`, where the
            source folder of a package would shadow the installed one.
        """
        return [
            sys.executable,
            "-P",
            "-m",
            orm_generation.__name__,
            *(str(interface.generator) for interface in self.interfaces),
        ]

    def run_writing_to_the_terminal(self) -> None:
        """
        Run the generators with the terminal, so their logging can be read as it
        happens.

        :raises OrmGenerationFailedError: If a generator exits without having built its
            interface.
        """
        if subprocess.run(self.command).returncode != 0:
            raise OrmGenerationFailedError(self.unbuilt_package_name, "")

    def run_reporting_to(self, progress: BuildProgress) -> None:
        """
        Run the generators, counting the classes they report and keeping the rest of
        what they write for a failure to report.

        A generator logs its way through a whole class hierarchy, which would bury the
        bar, so its logging is held back rather than shown.

        :param progress: What to report the classes and interfaces they finish to.
        :raises OrmGenerationFailedError: If a generator exits without having built its
            interface.
        """
        generation = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **os.environ,
                ProgressEnvironmentVariable.REPORT_PROGRESS: PROGRESS_REQUESTED,
            },
        )
        output = GenerationOutput(progress)
        for line in generation.stdout:
            output.read(line)
        if generation.wait() != 0:
            raise OrmGenerationFailedError(
                self.unbuilt_package_name, "".join(output.written)
            )
        output.finish()

    @property
    def unbuilt_package_name(self) -> str:
        """
        The package whose interface a failed build left unwritten.

        A generator writes its interface last, so the first one missing is the one that
        gave up.
        """
        for interface in self.interfaces:
            if not interface.path.exists():
                return interface.package_name
        return self.interfaces[-1].package_name


# Every generator runs in the same interpreter, which imports each package once for the
# whole build instead of once per generator. The order therefore has to satisfy two
# constraints: a package follows the packages whose ORM model it builds on, and it
# precedes every package it does not build on that defines alternative mappings, because
# ORMatic collects those from all imported subclasses.
WORKSPACE_ORM_INTERFACES = WorkspaceOrmInterfaces(
    (
        OrmInterface("semantic_digital_twin", REPOSITORY_ROOT),
        OrmInterface("giskardpy", REPOSITORY_ROOT, ("semantic_digital_twin",)),
        OrmInterface("segmind", REPOSITORY_ROOT, ("semantic_digital_twin",)),
        OrmInterface("coraplex", REPOSITORY_ROOT, ("giskardpy",)),
        OrmInterface("experiments", REPOSITORY_ROOT, ("coraplex",)),
    )
)
"""
The ORM interfaces of this repository.
"""
