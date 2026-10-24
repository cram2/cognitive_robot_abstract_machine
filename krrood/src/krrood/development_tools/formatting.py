"""
Formats Python files with ruff, then applies docformatter only where its output stays
stable under a second ruff pass.

docformatter can miscount the blank lines it leaves after an attribute docstring that
immediately precedes a decorated top-level class or function definition (it forgets to
special-case the decorator), so it sometimes undoes exactly what ruff then re-adds. Re-
running both tools forever would never converge for those files, so the plain ruff-
formatted content is kept for a file whenever docformatter's result does not survive a
subsequent ruff pass unchanged.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path

from tqdm import tqdm
from typing_extensions import Sequence

from krrood.development_tools.exceptions import FormatterNotInstalled

logger = logging.getLogger(__name__)


# %% tool invocation


class Formatter(StrEnum):
    """
    The formatting tools, by the module name they run under.
    """

    RUFF = "ruff"
    DOCFORMATTER = "docformatter"

    def run(self, arguments: Sequence[str], file_path: Path) -> int:
        """
        Run this tool on ``file_path`` with the interpreter running this code.

        :param arguments: The command line arguments to pass before the file.
        :param file_path: The file to format, in place.
        :return: The tool's exit code.
        :raises FormatterNotInstalled: If the tool is not installed for this
            interpreter.
        """
        if importlib.util.find_spec(self) is None:
            raise FormatterNotInstalled(self)
        return subprocess.run(
            [sys.executable, "-m", self, *arguments, str(file_path)],
            capture_output=True,
            text=True,
        ).returncode


class RuffArgument(StrEnum):
    """
    The command line arguments ruff is run with.
    """

    FORMAT = "format"
    QUIET = "--quiet"
    ISOLATED = "--isolated"
    """
    Ignore every configuration file, so each repository formats alike.
    """


class DocformatterOption(StrEnum):
    """
    The command line options docformatter is run with.
    """

    IN_PLACE = "--in-place"
    MAKE_SUMMARY_MULTI_LINE = "--make-summary-multi-line"
    PRE_SUMMARY_NEWLINE = "--pre-summary-newline"
    CLOSE_QUOTES_ON_NEWLINE = "--close-quotes-on-newline"
    BLACK_COMPATIBLE = "--black"


class RuffExitCode(IntEnum):
    """
    The exit code ruff reports when it formatted a file successfully.
    """

    SUCCESS = 0


class DocformatterExitCode(IntEnum):
    """
    The exit codes docformatter reports when it formatted a file successfully.
    """

    UNCHANGED = 0
    CHANGED = 3


# %% formatting


@dataclass
class FileFormattingOutcome:
    """
    The result of formatting a single file.
    """

    path: Path
    """
    The file that was formatted.
    """

    original_content: str
    """
    The file's content before formatting.
    """

    final_content: str
    """
    The file's content to write back.
    """

    docstrings_reformatted: bool
    """
    Whether docformatter's changes were kept.

    False means they were discarded because they conflicted with ruff.
    """

    @property
    def changed(self) -> bool:
        """
        Whether formatting changed the file's content.
        """
        return self.final_content != self.original_content


@dataclass
class DocstringFormatter:
    """
    Formats files with ruff, and their docstrings with docformatter where docformatter
    agrees with ruff.
    """

    def format_file(self, path: Path) -> FileFormattingOutcome:
        """
        Compute the formatted content of one file without writing it back.

        :param path: The file to format.
        :return: The formatting outcome for ``path``.
        """
        original_content = path.read_text()

        with tempfile.TemporaryDirectory() as scratch_directory:
            scratch_path = Path(scratch_directory) / path.name
            scratch_path.write_text(original_content)

            if not self._run_ruff(scratch_path):
                return FileFormattingOutcome(
                    path, original_content, original_content, False
                )
            ruff_only_content = scratch_path.read_text()

            docformatter_succeeded = self._run_docformatter(scratch_path)
            after_docformatter_content = (
                scratch_path.read_text() if docformatter_succeeded else None
            )

            ruff_succeeded_again = docformatter_succeeded and self._run_ruff(
                scratch_path
            )
            reconverged_content = (
                scratch_path.read_text() if ruff_succeeded_again else None
            )

        docstrings_are_stable = (
            ruff_succeeded_again and reconverged_content == after_docformatter_content
        )
        final_content = (
            reconverged_content if docstrings_are_stable else ruff_only_content
        )

        return FileFormattingOutcome(
            path=path,
            original_content=original_content,
            final_content=final_content,
            docstrings_reformatted=docstrings_are_stable,
        )

    def format_files(self, paths: Sequence[Path]) -> bool:
        """
        Format each file, writing back changes and reporting discarded docstring
        changes.

        :param paths: The files to format.
        :return: Whether any file was modified.
        """
        any_file_changed = False

        for path in tqdm(paths, desc="Formatting"):
            outcome = self.format_file(path)
            if not outcome.changed:
                continue

            outcome.path.write_text(outcome.final_content)
            any_file_changed = True

            if not outcome.docstrings_reformatted:
                logger.info(
                    "%s: formatted with ruff; docformatter's changes conflicted with "
                    "ruff and were left as-is",
                    outcome.path,
                )

        return any_file_changed

    @staticmethod
    def _run_ruff(file_path: Path) -> bool:
        """
        Format ``file_path`` with ruff, in place.

        :param file_path: The file to format.
        :return: Whether ruff accepted and formatted the file.
        """
        return Formatter.RUFF.run(list(RuffArgument), file_path) == RuffExitCode.SUCCESS

    @staticmethod
    def _run_docformatter(file_path: Path) -> bool:
        """
        Format ``file_path``'s docstrings with docformatter, in place.

        :param file_path: The file to format.
        :return: Whether docformatter ran successfully.
        """
        exit_code = Formatter.DOCFORMATTER.run(list(DocformatterOption), file_path)
        return exit_code in list(DocformatterExitCode)
