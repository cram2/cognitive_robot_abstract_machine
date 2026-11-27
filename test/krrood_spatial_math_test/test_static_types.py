"""
What a static type checker reads for spatial values: the stated types of
:mod:`.dataset.static_types` and the README's example of adding a space, checked by mypy.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

import krrood
import krrood_spatial_math
import pytest
from mypy import api

from .dataset import static_types

# %% running mypy over the code a user of the package writes


class TypeCheckerOption(StrEnum):
    """
    The options mypy runs with: only the files given are reported, and the packages
    they import are read from their sources.
    """

    FOLLOW_IMPORTS_SILENTLY = "--follow-imports=silent"
    """Reads the imported modules without reporting their own errors."""

    IGNORE_MISSING_IMPORTS = "--ignore-missing-imports"
    """Reads a library without type information, such as casadi, as untyped."""

    EXPLICIT_PACKAGE_BASES = "--explicit-package-bases"
    """Names each module by its path below one of the search path's directories."""

    NO_ERROR_SUMMARY = "--no-error-summary"
    """Leaves out the closing count, so a clean run prints nothing."""

    NO_CACHE = f"--cache-dir={os.devnull}"
    """Writes no cache into the checkout."""

    PROGRAM_TEXT = "--command"
    """Checks the text that follows instead of a file."""


@pytest.fixture
def type_checker_search_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Lets mypy read krrood, krrood_spatial_math and the test datasets from their sources,
    which carry no ``py.typed`` marker.
    """
    source_roots = [
        *(Path(package_root).parent for package_root in krrood.__path__),
        Path(krrood_spatial_math.__file__).parents[1],
        Path(__file__).parents[2],
    ]
    monkeypatch.setenv("MYPYPATH", os.pathsep.join(map(str, source_roots)))


def type_check(*, arguments: list[str]) -> list[str]:
    """
    :param arguments: The files or program text mypy checks.
    :return: The lines mypy reports, one per error or note; none for clean code.
    """
    report, _, _ = api.run(
        [
            TypeCheckerOption.FOLLOW_IMPORTS_SILENTLY,
            TypeCheckerOption.IGNORE_MISSING_IMPORTS,
            TypeCheckerOption.EXPLICIT_PACKAGE_BASES,
            TypeCheckerOption.NO_ERROR_SUMMARY,
            TypeCheckerOption.NO_CACHE,
            *arguments,
        ]
    )
    return report.splitlines()


# %% the types a type checker reads


def test_a_type_checker_reads_the_stated_types_of_spatial_values(
    type_checker_search_path,
):
    """
    Every :func:`typing_extensions.assert_type` in the static types dataset holds.
    """
    assert type_check(arguments=[static_types.__file__]) == []


def test_the_readme_example_of_adding_a_space_passes_a_type_checker(
    type_checker_search_path, adding_a_space_example_code
):
    """
    The README's pixel space, its change and its uses type-check as written.
    """
    assert (
        type_check(
            arguments=[TypeCheckerOption.PROGRAM_TEXT, adding_a_space_example_code]
        )
        == []
    )
