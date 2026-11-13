"""
Tests of formatting files with ruff and docformatter.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import importlib.util

import pytest

from krrood.development_tools.command_line import CommandLine, ExitStatus, Subcommand
from krrood.development_tools.exceptions import FormatterNotInstalled
from krrood.development_tools.formatting import (
    DocstringFormatter,
    Formatter,
    RuffArgument,
)

from .dataset_files import DatasetFile

# %% single file outcomes


def test_formats_code_and_docstrings_when_the_tools_agree(
    copy_into_temporary_directory,
) -> None:
    """
    Where docformatter's output survives a second ruff pass, both tools' changes are
    kept.
    """
    path = copy_into_temporary_directory(DatasetFile.UNFORMATTED_MODULE, "module.py")

    outcome = DocstringFormatter().format_file(path=path)

    assert outcome.final_content == DatasetFile.FORMATTED_MODULE.read_text()
    assert outcome.docstrings_reformatted


def test_keeps_ruff_output_when_docformatter_conflicts_with_ruff(
    copy_into_temporary_directory, tmp_path: Path
) -> None:
    """
    Where ruff changes docformatter's output again, ruff's output alone is kept.
    """
    path = copy_into_temporary_directory(
        DatasetFile.DOCSTRING_CONFLICTING_MODULE, "module.py"
    )
    ruff_only_path = tmp_path / "ruff_only.py"
    shutil.copyfile(path, ruff_only_path)
    Formatter.RUFF.run(arguments=list(RuffArgument), file_path=ruff_only_path)

    outcome = DocstringFormatter().format_file(path=path)

    assert outcome.final_content == ruff_only_path.read_text()
    assert not outcome.docstrings_reformatted


def test_leaves_a_file_ruff_rejects_unchanged(copy_into_temporary_directory) -> None:
    """
    A file ruff cannot parse is left as it is.
    """
    path = copy_into_temporary_directory(DatasetFile.INVALID_SYNTAX_MODULE, "module.py")

    outcome = DocstringFormatter().format_file(path=path)

    assert outcome.final_content == outcome.original_content
    assert not outcome.docstrings_reformatted


def test_formatting_a_file_does_not_write_it(copy_into_temporary_directory) -> None:
    """
    Computing a file's formatted content leaves the file on disk unchanged.
    """
    path = copy_into_temporary_directory(DatasetFile.UNFORMATTED_MODULE, "module.py")

    DocstringFormatter().format_file(path=path)

    assert path.read_text() == DatasetFile.UNFORMATTED_MODULE.read_text()


def test_a_missing_formatting_tool_is_reported(
    copy_into_temporary_directory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Formatting without the tools installed raises :class:`FormatterNotInstalled`.
    """
    path = copy_into_temporary_directory(DatasetFile.UNFORMATTED_MODULE, "module.py")
    monkeypatch.setattr(importlib.util, "find_spec", lambda module_name: None)

    with pytest.raises(FormatterNotInstalled):
        DocstringFormatter().format_file(path=path)


# %% command line


def test_format_command_writes_changes_and_reports_failure(
    copy_into_temporary_directory,
) -> None:
    """
    ``krrood format`` writes the formatted content back and exits with a failure, as
    pre-commit expects of a hook that changed a file.
    """
    path = copy_into_temporary_directory(DatasetFile.UNFORMATTED_MODULE, "module.py")

    exit_status = CommandLine.main(command_line=[Subcommand.FORMAT, str(path)])

    assert exit_status == ExitStatus.FAILURE
    assert path.read_text() == DatasetFile.FORMATTED_MODULE.read_text()


def test_format_command_succeeds_on_formatted_files(
    copy_into_temporary_directory,
) -> None:
    """
    ``krrood format`` leaves a formatted file alone and exits successfully.
    """
    path = copy_into_temporary_directory(DatasetFile.FORMATTED_MODULE, "module.py")

    exit_status = CommandLine.main(command_line=[Subcommand.FORMAT, str(path)])

    assert exit_status == ExitStatus.SUCCESS
    assert path.read_text() == DatasetFile.FORMATTED_MODULE.read_text()
