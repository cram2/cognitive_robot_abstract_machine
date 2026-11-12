from __future__ import annotations

from pathlib import Path

import pytest

from krrood.development_tools.agents.core_boundary import (
    CoreBoundary,
    CoreBoundaryChecker,
    DeclarationFile,
)
from krrood.development_tools.agents.shipped_rules.imports import ImportsRules
from krrood.development_tools.command_line import CommandLine, ExitStatus, Option
from krrood.development_tools.subcommand import Subcommand

from .dataset_files import (
    CoreBoundarySnippet,
    DatasetMarker,
    MimicDistribution,
    MimicModule,
)

# %% imports a core module runs when it is loaded


def test_every_import_crossing_the_boundary_is_reported() -> None:
    path = CoreBoundarySnippet.CROSSING_THE_BOUNDARY
    marked_lines = [
        line_number
        for line_number, line in enumerate(path.read_text().splitlines(), start=1)
        if line.endswith(DatasetMarker.CROSSING_IMPORT)
    ]

    reported_lines = [
        violation.first_line
        for violation in CoreBoundaryChecker().violations(path=path)
    ]

    assert reported_lines == marked_lines


@pytest.mark.parametrize(
    "path",
    [
        CoreBoundarySnippet.INSIDE_THE_BOUNDARY,
        CoreBoundarySnippet.MODULE_OUTSIDE_CORE,
        CoreBoundarySnippet.MODULE_OF_UNDECLARED_CORE,
    ],
)
def test_imports_inside_the_boundary_are_not_reported(path: Path) -> None:
    assert list(CoreBoundaryChecker().violations(path=path)) == []


def test_a_module_outside_core_reports_what_it_loads_through_its_imports() -> None:
    boundary = CoreBoundary.from_pyproject(
        path=CoreBoundarySnippet.DISTRIBUTION / DeclarationFile.PYPROJECT
    )

    assert boundary.distributions_loaded_beyond_core(module=MimicModule.DRAWING) == {
        MimicDistribution.MATPLOTLIB
    }


# %% the rule and the command


def test_the_core_boundary_rule_carries_the_checker() -> None:
    assert ImportsRules.CORE_LOADS_ONLY_CORE.checker == CoreBoundaryChecker()


def test_checking_a_core_module_crossing_the_boundary_fails(
    repository_root: Path,
) -> None:
    exit_status = CommandLine.main(
        command_line=[
            Subcommand.CHECK,
            Option.ALL_LINES,
            str(CoreBoundarySnippet.CROSSING_THE_BOUNDARY),
        ]
    )

    assert exit_status == ExitStatus.FAILURE
