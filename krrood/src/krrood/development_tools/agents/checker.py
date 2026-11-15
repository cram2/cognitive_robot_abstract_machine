"""
The base every code rule checker extends, and the violations checkers report.
"""

from __future__ import annotations

import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Iterator, Sequence

from krrood.development_tools.exceptions import GitCommandFailed

# %% violations


@dataclass(kw_only=True, frozen=True)
class CodeRuleViolation:
    """
    One place in a source file that breaks a rule.
    """

    path: Path
    """
    The file breaking the rule.
    """

    first_line: int
    """
    The first line of the offending code, counting from one.
    """

    last_line: int
    """
    The last line of the offending code, counting from one.
    """

    description: str
    """
    What is wrong and how to fix it, as one sentence fragment.
    """

    def touches(self, *, lines: set[int]) -> bool:
        """
        :param lines: Line numbers in the file.
        :return: Whether any of them falls within the offending code.
        """
        return any(self.first_line <= line <= self.last_line for line in lines)

    def render(self) -> str:
        """
        :return: The violation as it is printed.
        """
        return f"{self.path}:{self.first_line}: {self.description}"


# %% checkers


@dataclass(kw_only=True, frozen=True)
class CodeRuleChecker(ABC):
    """
    Finds the code that breaks one agent rule, the rule that names this checker.

    Checkers are frozen so that a rule holding one stays hashable.
    """

    @abstractmethod
    def violations(self, *, path: Path) -> Iterator[CodeRuleViolation]:
        """
        :param path: A Python source file.
        :return: Every place in it that breaks the rule.
        """


# %% the staged change


class GitArgument(StrEnum):
    """
    The git subcommand and options listing a staged change without surrounding context.
    """

    GIT = "git"
    DIFFERENCE = "diff"
    CACHED = "--cached"
    NO_CONTEXT = "--unified=0"
    NO_COLOR = "--no-color"
    PATHS_FOLLOW = "--"


class DifferencePattern(StrEnum):
    """
    The patterns read from the output of ``git diff``.
    """

    HUNK_HEADER = r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@"
    """
    The header of a hunk, capturing the first line it covers in the new file and, when
    it is not one, how many lines.
    """


@dataclass(kw_only=True, frozen=True)
class StagedLines:
    """
    The lines a commit being prepared adds or changes.
    """

    repository: Path
    """
    The directory of the git repository the commit is prepared in.
    """

    def find_changed_lines(self, *, path: Path) -> set[int]:
        """
        :param path: A file in the repository.
        :return: The line numbers, in the staged file, that the staged change adds or
            changes.
        :raises GitCommandFailed: If git cannot list the staged change.
        """
        arguments = (
            GitArgument.DIFFERENCE,
            GitArgument.CACHED,
            GitArgument.NO_CONTEXT,
            GitArgument.NO_COLOR,
            GitArgument.PATHS_FOLLOW,
            str(path),
        )
        result = subprocess.run(
            [GitArgument.GIT, *arguments],
            cwd=self.repository,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise GitCommandFailed(arguments=arguments, standard_error=result.stderr)
        lines: set[int] = set()
        for hunk in re.finditer(
            DifferencePattern.HUNK_HEADER, result.stdout, re.MULTILINE
        ):
            start = int(hunk[1])
            count = 1 if hunk[2] is None else int(hunk[2])
            lines.update(range(start, start + count))
        return lines


# %% checking files


@dataclass(kw_only=True, frozen=True)
class CodeRulesCheck:
    """
    Runs a set of checkers over a set of files.
    """

    checkers: tuple[CodeRuleChecker, ...]
    """
    The checkers to run.
    """

    staged: StagedLines | None = None
    """
    The staged change whose code alone is checked; none to check all of it.
    """

    def violations(self, *, paths: Sequence[Path]) -> list[CodeRuleViolation]:
        """
        :param paths: The Python files to check.
        :return: Their violations, file by file, top to bottom.
        """
        violations: list[CodeRuleViolation] = []
        for path in paths:
            changed = (
                None
                if self.staged is None
                else self.staged.find_changed_lines(path=path)
            )
            found = [
                violation
                for checker in self.checkers
                for violation in checker.violations(path=path)
                if changed is None or violation.touches(lines=changed)
            ]
            violations.extend(sorted(found, key=lambda violation: violation.first_line))
        return violations
