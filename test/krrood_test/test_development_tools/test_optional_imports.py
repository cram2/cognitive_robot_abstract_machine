from __future__ import annotations

from pathlib import Path

import pytest

import krrood

from krrood.development_tools.command_line import (
    CommandLine,
    ExitStatus,
    Option,
    Subcommand,
)
from krrood.development_tools.optional_imports import (
    OptionalImportCheck,
    UnguardedOptionalImport,
)
from krrood.optional_modules import KrroodOptionalModule

from .dataset_files import DatasetFile, DatasetMarker


@pytest.fixture
def check() -> OptionalImportCheck:
    return OptionalImportCheck(list(KrroodOptionalModule))


# %% finding unguarded imports


def test_reports_every_unguarded_import(check: OptionalImportCheck) -> None:
    path = DatasetFile.OPTIONAL_IMPORTS_UNGUARDED
    marked_lines = [
        line_number
        for line_number, line in enumerate(path.read_text().splitlines(), start=1)
        if line.endswith(DatasetMarker.UNGUARDED_IMPORT)
    ]

    assert check.find_unguarded_imports([path]) == [
        UnguardedOptionalImport(path, line_number, KrroodOptionalModule.MATPLOTLIB)
        for line_number in marked_lines
    ]


def test_accepts_decorated_functions_and_type_checking_blocks(
    check: OptionalImportCheck,
) -> None:
    assert check.find_unguarded_imports([DatasetFile.OPTIONAL_IMPORTS_GUARDED]) == []


def test_ignores_modules_that_only_resemble_an_optional_one(
    check: OptionalImportCheck,
) -> None:
    assert check.find_unguarded_imports([DatasetFile.OPTIONAL_IMPORTS_UNRELATED]) == []


def test_krrood_imports_its_optional_modules_only_where_guarded(
    check: OptionalImportCheck,
) -> None:
    source_directory = Path(krrood.__file__).parent

    assert check.find_unguarded_imports(sorted(source_directory.rglob("*.py"))) == []


# %% command line


@pytest.mark.parametrize(
    "dataset_file, exit_status",
    [
        (DatasetFile.OPTIONAL_IMPORTS_GUARDED, ExitStatus.SUCCESS),
        (DatasetFile.OPTIONAL_IMPORTS_UNGUARDED, ExitStatus.FAILURE),
    ],
)
def test_command_line_exit_status(
    dataset_file: DatasetFile, exit_status: ExitStatus
) -> None:
    command_line = [
        Subcommand.OPTIONAL_IMPORTS,
        Option.OPTIONAL_MODULES,
        f"{KrroodOptionalModule.__module__}:{KrroodOptionalModule.__name__}",
        str(dataset_file),
    ]

    assert CommandLine.main(command_line) == exit_status
