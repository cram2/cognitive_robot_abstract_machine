"""
Pre-commit entry point that formats Python files with black, then applies docformatter
only where its output stays stable under a second black pass.

docformatter can miscount the blank lines it leaves after an attribute docstring that
immediately precedes a decorated top-level class or function definition (it forgets to
special-case the decorator), so it sometimes undoes exactly what black then re-adds. Re-
running both tools forever would never converge for those files, so this script keeps
the plain black-formatted content for a file whenever docformatter's result does not
survive a subsequent black pass unchanged.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
from typing_extensions import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DOCFORMATTER_CONFIG_PATH = REPOSITORY_ROOT / "pyproject.toml"

logger = logging.getLogger(__name__)


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
    The file's content before this hook ran.
    """

    final_content: str
    """
    The file's content to write back.
    """

    docstrings_reformatted: bool
    """
    Whether docformatter's changes were kept.

    False means they were discarded because they conflicted with black.
    """


def _run_black(file_path: Path) -> bool:
    """
    Format ``file_path`` with black, in place.

    :param file_path: The file to format.
    :return: Whether black accepted and formatted the file.
    """
    result = subprocess.run(
        ["black", "--quiet", str(file_path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _run_docformatter(file_path: Path) -> bool:
    """
    Format ``file_path``'s docstrings with docformatter, in place.

    :param file_path: The file to format.
    :return: Whether docformatter ran successfully.
    """
    result = subprocess.run(
        [
            "docformatter",
            "--in-place",
            "--config",
            str(DOCFORMATTER_CONFIG_PATH),
            str(file_path),
        ],
        capture_output=True,
        text=True,
    )
    # docformatter exits 3 when it changed the file and 0 when it left it alone.
    return result.returncode in (0, 3)


def _format_file(path: Path) -> FileFormattingOutcome:
    """
    Format one file, falling back to a black-only result if docformatter conflicts with
    black.

    :param path: The file to format.
    :return: The formatting outcome for ``path``.
    """
    original_content = path.read_text()

    with tempfile.TemporaryDirectory() as scratch_directory:
        scratch_path = Path(scratch_directory) / path.name
        scratch_path.write_text(original_content)

        if not _run_black(scratch_path):
            return FileFormattingOutcome(
                path, original_content, original_content, False
            )
        black_only_content = scratch_path.read_text()

        docformatter_succeeded = _run_docformatter(scratch_path)
        after_docformatter_content = (
            scratch_path.read_text() if docformatter_succeeded else None
        )

        black_succeeded_again = docformatter_succeeded and _run_black(scratch_path)
        reconverged_content = (
            scratch_path.read_text() if black_succeeded_again else None
        )

    docstrings_are_stable = (
        black_succeeded_again and reconverged_content == after_docformatter_content
    )
    final_content = reconverged_content if docstrings_are_stable else black_only_content

    return FileFormattingOutcome(
        path=path,
        original_content=original_content,
        final_content=final_content,
        docstrings_reformatted=docstrings_are_stable,
    )


def main(file_arguments: Sequence[str]) -> int:
    """
    Format each given file, writing back changes and reporting skipped docstrings.

    :param file_arguments: File paths to format, as passed by pre-commit.
    :return: 1 if any file was modified (matching pre-commit's convention for
        autoformatting hooks), 0 otherwise.
    """
    any_file_changed = False

    for file_argument in tqdm(file_arguments, desc="Formatting"):
        outcome = _format_file(Path(file_argument))
        if outcome.final_content == outcome.original_content:
            continue

        outcome.path.write_text(outcome.final_content)
        any_file_changed = True

        if not outcome.docstrings_reformatted:
            logger.info(
                "%s: formatted with black; docformatter's changes conflicted with "
                "black and were left as-is",
                outcome.path,
            )

    return 1 if any_file_changed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main(sys.argv[1:]))
