from __future__ import annotations

from pathlib import Path

import pytest

import krrood
from krrood.development_tools.agents.optional_imports import OptionalImportChecker
from krrood.development_tools.agents.shipped_rules.imports import ImportsRules
from krrood.development_tools.command_line import CommandLine, ExitStatus, Option
from krrood.development_tools.subcommand import Subcommand

from .dataset_files import DatasetMarker, OptionalLibrarySnippet

# %% module-level imports in core


def test_module_level_imports_of_an_optional_library_in_core_are_reported() -> None:
    path = OptionalLibrarySnippet.UNGUARDED_CORE_MODULE
    marked_lines = [
        line_number
        for line_number, line in enumerate(path.read_text().splitlines(), start=1)
        if line.endswith(DatasetMarker.UNGUARDED_IMPORT)
    ]

    reported_lines = [
        violation.first_line for violation in OptionalImportChecker().violations(path)
    ]

    assert reported_lines == marked_lines


@pytest.mark.parametrize(
    "path",
    [
        OptionalLibrarySnippet.GUARDED_CORE_MODULE,
        OptionalLibrarySnippet.MODULE_OUTSIDE_CORE,
        OptionalLibrarySnippet.MODULE_OF_UNDECLARED_CORE,
    ],
)
def test_imports_outside_module_level_core_code_are_not_reported(path: Path) -> None:
    assert list(OptionalImportChecker().violations(path)) == []


def test_krrood_core_imports_no_optional_library_at_module_level() -> None:
    source = Path(krrood.__file__).parent
    checker = OptionalImportChecker()

    violations = [
        violation
        for path in sorted(source.rglob("*.py"))
        for violation in checker.violations(path)
    ]

    assert violations == []


# %% the rule and the command


def test_the_optional_import_rule_carries_the_checker() -> None:
    assert ImportsRules.IMPORT_OPTIONAL_LIBRARIES_WHERE_USED.checker == (
        OptionalImportChecker()
    )


def test_checking_a_core_module_importing_an_optional_library_fails(
    repository_root: Path,
) -> None:
    exit_status = CommandLine.main(
        [
            Subcommand.CHECK,
            Option.ALL_LINES,
            str(OptionalLibrarySnippet.UNGUARDED_CORE_MODULE),
        ]
    )

    assert exit_status == ExitStatus.FAILURE
