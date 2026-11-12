"""
The ``krrood`` command.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from functools import cached_property
from pathlib import Path

from typing_extensions import Sequence

from krrood.development_tools.agents.checker import CodeRulesCheck, StagedLines
from krrood.development_tools.agents.instructions import (
    AgentsInstructions,
    GeneratedInstructions,
    RepositorySkills,
)
from krrood.development_tools.formatting import DocstringFormatter
from krrood.development_tools.agents.repository_files import (
    RepositoryInstructionsFile,
)
from krrood.development_tools.subcommand import Subcommand

logger = logging.getLogger(__name__)


# %% command line vocabulary


class Option(StrEnum):
    """
    The command line options of the subcommands.
    """

    CHECK = "--check"
    """
    Verify a generated file instead of writing it.
    """

    ALL_LINES = "--all-lines"
    """
    Check every line of a file, not only those the staged change touches.
    """


class ExitStatus(IntEnum):
    """
    The exit status of a subcommand, following pre-commit's convention for hooks.
    """

    SUCCESS = 0
    """
    Nothing had to change.
    """

    FAILURE = 1
    """
    Files were modified, or a checked file is stale.
    """


# %% command line


@dataclass(kw_only=True)
class CommandLine:
    """
    The ``krrood`` command line: its subcommands, their arguments and running the one a
    command line names.
    """

    program_name: str = Subcommand.program_name
    """
    The name the command is invoked by.
    """

    @classmethod
    def main(cls, *, command_line: Sequence[str] | None = None) -> int:
        """
        Run the subcommand the command line names.

        :param command_line: The arguments after the program name; the process's own when
            omitted.
        :return: The subcommand's exit status.
        """
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        interface = cls()
        arguments = interface.parser.parse_args(command_line)
        return arguments.run(arguments=arguments)

    @cached_property
    def parser(self) -> argparse.ArgumentParser:
        """
        The parser for every subcommand, each bound to the method that runs it.
        """
        parser = argparse.ArgumentParser(prog=self.program_name)
        subparsers = parser.add_subparsers(required=True)

        format_parser = subparsers.add_parser(
            Subcommand.FORMAT,
            help="Format Python files with ruff, then docformatter where they agree.",
        )
        format_parser.add_argument("files", nargs="*", type=Path)
        format_parser.set_defaults(run=self.format_files)

        agents_parser = subparsers.add_parser(
            Subcommand.AGENTS_MD,
            help="Generate AGENTS.md from the rules krrood ships and the repository's own.",
        )
        agents_parser.add_argument(
            Option.CHECK,
            action="store_true",
            help="Only verify that AGENTS.md is up to date; exit non-zero if it is not.",
        )
        agents_parser.set_defaults(run=self.generate_agents_instructions)

        skills_parser = subparsers.add_parser(
            Subcommand.SKILLS,
            help="Generate the SKILL.md of every skill the repository declares.",
        )
        skills_parser.add_argument(
            Option.CHECK,
            action="store_true",
            help="Only verify that every SKILL.md is up to date; exit non-zero if one is not.",
        )
        skills_parser.set_defaults(run=self.generate_skill_instructions)

        check_parser = subparsers.add_parser(
            Subcommand.CHECK,
            help="Report the code that breaks a rule a checker enforces.",
        )
        check_parser.add_argument("files", nargs="*", type=Path)
        check_parser.add_argument(
            Option.ALL_LINES,
            action="store_true",
            help="Check every line, not only those the staged change touches.",
        )
        check_parser.set_defaults(run=self.check_files)

        return parser

    def format_files(self, *, arguments: argparse.Namespace) -> ExitStatus:
        """
        Format the files named on the command line.

        :param arguments: The parsed command line.
        :return: :attr:`ExitStatus.FAILURE` if any file was modified.
        """
        if DocstringFormatter().format_files(paths=arguments.files):
            return ExitStatus.FAILURE
        return ExitStatus.SUCCESS

    def generate_agents_instructions(
        self, *, arguments: argparse.Namespace
    ) -> ExitStatus:
        """
        Generate, or with :attr:`Option.CHECK` verify, the ``AGENTS.md`` of the current
        directory.

        :param arguments: The parsed command line.
        :return: :attr:`ExitStatus.FAILURE` if the file was rewritten or is stale.
        """
        instructions = AgentsInstructions.from_repository(repository_root=Path.cwd())
        return self.generate_instructions(
            instructions=[instructions],
            subcommand=Subcommand.AGENTS_MD,
            only_check=arguments.check,
        )

    def generate_skill_instructions(
        self, *, arguments: argparse.Namespace
    ) -> ExitStatus:
        """
        Generate, or with :attr:`Option.CHECK` verify, the ``SKILL.md`` of every skill
        the current directory declares.

        :param arguments: The parsed command line.
        :return: :attr:`ExitStatus.FAILURE` if a file was rewritten or is stale.
        """
        return self.generate_instructions(
            instructions=RepositorySkills(repository_root=Path.cwd()).instructions,
            subcommand=Subcommand.SKILLS,
            only_check=arguments.check,
        )

    def generate_instructions(
        self,
        *,
        instructions: list[GeneratedInstructions],
        subcommand: Subcommand,
        only_check: bool,
    ) -> ExitStatus:
        """
        Write generated instruction files, or only verify that they are current.

        :param instructions: The files.
        :param subcommand: The subcommand that regenerates them, named when one is stale.
        :param only_check: Whether to verify the files instead of writing them.
        :return: :attr:`ExitStatus.FAILURE` if a file was rewritten or is stale.
        """
        if only_check:
            stale = [file for file in instructions if not file.is_current()]
            for file in stale:
                logger.error(
                    "%s is missing, out of date or edited by hand. Change the module "
                    "the comment in it names, never the file; then run `%s %s`.",
                    file.path,
                    self.program_name,
                    subcommand,
                )
            return ExitStatus.FAILURE if stale else ExitStatus.SUCCESS
        written = [file for file in instructions if file.write()]
        for file in written:
            logger.info("Wrote %s", file.path)
        return ExitStatus.FAILURE if written else ExitStatus.SUCCESS

    def check_files(self, *, arguments: argparse.Namespace) -> ExitStatus:
        """
        Report the code in the files named on the command line that breaks a rule a
        checker enforces.

        :param arguments: The parsed command line.
        :return: :attr:`ExitStatus.FAILURE` if any code breaks a rule.
        """
        instructions = AgentsInstructions.from_repository(repository_root=Path.cwd())
        staged = None if arguments.all_lines else StagedLines(repository=Path.cwd())
        violations = CodeRulesCheck(
            checkers=instructions.checkers, staged=staged
        ).violations(paths=arguments.files)
        for violation in violations:
            logger.error(violation.render())
        if violations:
            return ExitStatus.FAILURE
        return ExitStatus.SUCCESS


if __name__ == "__main__":
    sys.exit(CommandLine.main())
