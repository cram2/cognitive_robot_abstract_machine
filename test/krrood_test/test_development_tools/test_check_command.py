from __future__ import annotations

from pathlib import Path

from krrood.development_tools.command_line import (
    CommandLine,
    ExitStatus,
    Option,
)
from krrood.development_tools.subcommand import Subcommand

from .dataset_files import DocstringSnippet, ProseSnippet

# %% check command


def test_checking_fails_when_code_breaks_a_rule(repository_root: Path) -> None:
    exit_status = CommandLine.main(
        [Subcommand.CHECK, Option.ALL_LINES, str(ProseSnippet.COMMENT_BLOCKS)]
    )

    assert exit_status == ExitStatus.FAILURE


def test_checking_passes_when_nothing_breaks_a_rule(repository_root: Path) -> None:
    exit_status = CommandLine.main(
        [Subcommand.CHECK, Option.ALL_LINES, str(DocstringSnippet.DOCUMENTED_MODULE)]
    )

    assert exit_status == ExitStatus.SUCCESS


def test_checking_without_files_passes(repository_root: Path) -> None:
    assert CommandLine.main([Subcommand.CHECK]) == ExitStatus.SUCCESS
