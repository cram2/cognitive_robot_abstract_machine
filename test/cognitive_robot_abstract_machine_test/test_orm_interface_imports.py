"""
Tests of the check that generated ORM interfaces are imported only inside functions.
"""

from __future__ import annotations

import shutil
from enum import StrEnum
from pathlib import Path

from cognitive_robot_abstract_machine.orm_interface_imports import (
    ModuleLevelOrmInterfaceImportChecker,
)
from cognitive_robot_abstract_machine.orm_interfaces import OrmScriptFileName
from krrood.patterns.path_enumeration import PathEnumeration

# %% the snippet the check reads


class OrmInterfaceImportSnippet(PathEnumeration):
    """
    The Python sources the check reads.
    """

    INTERFACE_IMPORTS = (
        Path(__file__).parent
        / "dataset"
        / "orm_interface_imports"
        / "interface_imports.py"
    )
    """
    Imports of generated interfaces, those run when the module is loaded marked with
    :attr:`SnippetMarker.REPORTED`.
    """


class SnippetMarker(StrEnum):
    """
    The comments marking the lines of a snippet.
    """

    REPORTED = "# reported"
    """
    A line the check reports.
    """


# %% reporting module-level imports


def test_the_imports_run_when_the_module_is_loaded_are_reported() -> None:
    """
    Every form of import naming the interface is reported, unless it sits inside a
    function or under ``TYPE_CHECKING``.
    """
    path = OrmInterfaceImportSnippet.INTERFACE_IMPORTS
    marked_lines = [
        number
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if line.endswith(SnippetMarker.REPORTED)
    ]

    violations = ModuleLevelOrmInterfaceImportChecker().violations(path=path)

    assert [violation.first_line for violation in violations] == marked_lines


def test_a_generator_may_import_interfaces_when_it_is_loaded(tmp_path: Path) -> None:
    """
    A generator's module-level imports declare the interfaces it builds on.
    """
    path = tmp_path / OrmScriptFileName.GENERATOR
    shutil.copy(OrmInterfaceImportSnippet.INTERFACE_IMPORTS, path)

    assert list(ModuleLevelOrmInterfaceImportChecker().violations(path=path)) == []
