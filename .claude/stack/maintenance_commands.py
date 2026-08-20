"""
The commands the executor answers, and what one run has resolved for them.

A command owns its own name, its own flags and its own work, so adding one is writing a
subclass: :data:`COMMANDS` finds it, and nothing else has to be told it exists.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass

from class_property import classproperty
from maintenance_board import BoardExport
from maintenance_fast_forward import fast_forward
from maintenance_git_commands import GitCommandRunner
from maintenance_github import GitHubRepository
from maintenance_promotion import clear_spent_promotion_labels, promote
from maintenance_report import (
    MaintenanceExitCode,
    MaintenanceReport,
    build_report,
    exit_code_for,
    print_board_export,
    print_fast_forward,
    print_promotions,
    print_restack,
)
from maintenance_restack_procedure import restack
from stack import BOARD_PATH, Configuration, Stack, load_stack


@dataclass(frozen=True)
class MaintenancePass:
    """
    What one run has resolved so far, built lazily as a command asks for it.

    The board is derived before the credential is resolved, so a caller missing both is
    sent after the board - the thing the previous command produces - rather than after a
    token that would not help them yet.
    """

    configuration: Configuration
    """
    The resolved configuration naming both repositories and every label.
    """

    git: GitCommandRunner
    """
    The runner every git command goes through.
    """

    def fork(self) -> GitHubRepository:
        """:return: The fork, as this run's credential can read and write it."""
        return GitHubRepository.from_environment(self.configuration.fork_repository)

    def stack(self) -> Stack:
        """:return: The derived stack, read from the exported board."""
        return load_stack()


@dataclass(frozen=True)
class MaintenanceCommand(ABC):
    """
    One command this executor answers.

    The name and the description belong to the class rather than to an instance - the
    parser reads them to build the command line before anything is constructed - so they
    are abstract :class:`class_property.classproperty` members. A subclass supplying
    neither stays abstract, and :data:`COMMANDS` builds every subclass, so a command
    that cannot say what it is called is refused as this module is imported.
    """

    @classproperty
    @abstractmethod
    def invoked_as(cls) -> str:
        """
        The name it is invoked by on the command line.
        """

    @classproperty
    @abstractmethod
    def description(cls) -> str:
        """
        What it does, as ``--help`` puts it.
        """

    def declare_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Declare this command's own flags.

        Concrete rather than abstract: most commands take none, and requiring an empty
        override of every one of them would say nothing.

        :param parser: The subparser to declare them on.
        """

    @abstractmethod
    def run(
        self, maintenance: MaintenancePass, arguments: argparse.Namespace
    ) -> MaintenanceExitCode:
        """
        Perform the command.

        :param maintenance: What this run has resolved.
        :param arguments: The parsed command line.
        :return: The process exit code.
        """


@dataclass(frozen=True)
class BoardCommand(MaintenanceCommand):
    """
    Fetches the fork's open pull requests and exports them as the board.
    """

    @classproperty
    def invoked_as(cls) -> str:
        """
        The name it is invoked by on the command line.
        """
        return "board"

    @classproperty
    def description(cls) -> str:
        """
        What it does, as ``--help`` puts it.
        """
        return "export the fork's open pull requests"

    def declare_arguments(self, parser: argparse.ArgumentParser) -> None:
        """:param parser: The subparser to declare ``--write`` on."""
        parser.add_argument(
            "--write",
            action="store_true",
            help="write board.json rather than printing the export",
        )

    def run(
        self, maintenance: MaintenancePass, arguments: argparse.Namespace
    ) -> MaintenanceExitCode:
        """:param maintenance: What this run has resolved.
        :param arguments: The parsed command line.
        :return: The process exit code."""
        export = BoardExport.from_api_records(maintenance.fork().open_pull_requests())
        print_board_export(export, export.write() if arguments.write else None)
        return MaintenanceExitCode.SUCCESS


@dataclass(frozen=True)
class FastForwardCommand(MaintenanceCommand):
    """
    Moves the fork's base branch onto the upstream, refusing to force.
    """

    @classproperty
    def invoked_as(cls) -> str:
        """
        The name it is invoked by on the command line.
        """
        return "fast-forward"

    @classproperty
    def description(cls) -> str:
        """
        What it does, as ``--help`` puts it.
        """
        return "move the fork's base branch onto the upstream"

    def run(
        self, maintenance: MaintenancePass, arguments: argparse.Namespace
    ) -> MaintenanceExitCode:
        """:param maintenance: What this run has resolved.
        :param arguments: The parsed command line.
        :return: The process exit code."""
        report = fast_forward(maintenance.configuration, maintenance.git)
        print_fast_forward(report)
        return exit_code_for(MaintenanceReport(fast_forward=report))


@dataclass(frozen=True)
class RestackCommand(MaintenanceCommand):
    """
    Integrates every moved parent and publishes what merged cleanly.
    """

    @classproperty
    def invoked_as(cls) -> str:
        """
        The name it is invoked by on the command line.
        """
        return "restack"

    @classproperty
    def description(cls) -> str:
        """
        What it does, as ``--help`` puts it.
        """
        return "integrate every moved parent and publish the result"

    def run(
        self, maintenance: MaintenancePass, arguments: argparse.Namespace
    ) -> MaintenanceExitCode:
        """:param maintenance: What this run has resolved.
        :param arguments: The parsed command line.
        :return: The process exit code."""
        stack = maintenance.stack()
        outcomes = restack(stack, maintenance.git, maintenance.fork())
        print_restack(outcomes)
        return exit_code_for(MaintenanceReport(restacked=tuple(outcomes)))


@dataclass(frozen=True)
class PromoteCommand(MaintenanceCommand):
    """
    Records the upstream link on every branch ready to be promoted.
    """

    @classproperty
    def invoked_as(cls) -> str:
        """
        The name it is invoked by on the command line.
        """
        return "promote"

    @classproperty
    def description(cls) -> str:
        """
        What it does, as ``--help`` puts it.
        """
        return "record the upstream link on every promotable branch"

    def run(
        self, maintenance: MaintenancePass, arguments: argparse.Namespace
    ) -> MaintenanceExitCode:
        """:param maintenance: What this run has resolved.
        :param arguments: The parsed command line.
        :return: The process exit code."""
        stack = maintenance.stack()
        fork = maintenance.fork()
        print_promotions(
            promote(stack, fork), clear_spent_promotion_labels(stack, fork)
        )
        return MaintenanceExitCode.SUCCESS


@dataclass(frozen=True)
class RunReportCommand(MaintenanceCommand):
    """
    Performs the whole pass and reports it as one document.
    """

    @classproperty
    def invoked_as(cls) -> str:
        """
        The name it is invoked by on the command line.
        """
        return "run-report"

    @classproperty
    def description(cls) -> str:
        """
        What it does, as ``--help`` puts it.
        """
        return "perform the whole pass and report it"

    def declare_arguments(self, parser: argparse.ArgumentParser) -> None:
        """:param parser: The subparser to declare ``--json`` on."""
        parser.add_argument(
            "--json",
            action="store_true",
            help="emit the machine-readable document rather than a summary",
        )

    def run(
        self, maintenance: MaintenancePass, arguments: argparse.Namespace
    ) -> MaintenanceExitCode:
        """
        Perform every step of the pass, then discard the board it derived from.

        The board is a snapshot of one moment's open pull requests, and a stale one read
        by a later run is worse than none at all - so a whole pass ends without one, and
        the next begins by exporting a fresh one.

        :param maintenance: What this run has resolved.
        :param arguments: The parsed command line.
        :return: The process exit code.
        """
        stack = maintenance.stack()
        fork = maintenance.fork()
        fast_forward_report = fast_forward(stack.configuration, maintenance.git)
        report = build_report(
            stack,
            fast_forward_report,
            restack(stack, maintenance.git, fork),
            promote(stack, fork),
            clear_spent_promotion_labels(stack, fork),
        )
        BOARD_PATH.unlink(missing_ok=True)
        if arguments.json:
            print(report.as_json())
        else:
            print_fast_forward(fast_forward_report)
            print_restack(report.restacked)
            print_promotions(report.promoted, report.promotion_labels_cleared)
        return exit_code_for(report)


COMMANDS: tuple[MaintenanceCommand, ...] = tuple(
    subclass() for subclass in MaintenanceCommand.__subclasses__()
)
"""
Every command this executor answers, found from the subclasses themselves so a command
cannot exist without being reachable, in the order they are defined.
"""
