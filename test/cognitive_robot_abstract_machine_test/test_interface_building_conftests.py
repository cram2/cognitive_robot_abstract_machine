"""
The conftests that build the ORM interfaces a test package reads.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from typing_extensions import Sequence

from cognitive_robot_abstract_machine.orm_interfaces import REPOSITORY_ROOT

BUILDING_TEST_PACKAGES: Sequence[str] = (
    "semantic_digital_twin_test",
    "coraplex_test",
    "giskardpy_test",
    "experiments_test",
)
"""
The test packages that read a mapped datastructure, and so build the interfaces.
"""

BUILD_CALL = "regenerate_orm_interfaces"
"""
What a conftest calls to have the interfaces built.
"""


def conftest_of(test_package: str) -> Path:
    """
    Locate the conftest of a test package.

    :param test_package: Name of the test package.
    :return: Path of its conftest.
    """
    return REPOSITORY_ROOT / "test" / test_package / "conftest.py"


@pytest.fixture(params=BUILDING_TEST_PACKAGES)
def conftest(request) -> Path:
    """
    Each conftest that builds the interfaces, one per test.

    :return: Path of the conftest under test.
    """
    return conftest_of(request.param)


# %% a conftest python accepts


def test_a_building_conftest_is_valid_python(conftest: Path):
    """
    Prepending the build to a conftest must not displace what has to come first: a
    module docstring, or a ``__future__`` import python only accepts on the first line.
    """
    compile(conftest.read_text(encoding="utf-8"), str(conftest), "exec")


def test_a_building_conftest_keeps_the_docstring_it_had(conftest: Path):
    tree = ast.parse(conftest.read_text(encoding="utf-8"))
    strings_before_the_build = [
        node
        for node in tree.body[: _build_call_index(tree)]
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    ]

    assert len(strings_before_the_build) <= 1


# %% a conftest that actually builds


def test_every_package_reading_a_mapping_builds_the_interfaces(conftest: Path):
    assert BUILD_CALL in conftest.read_text(encoding="utf-8")


def _build_call_index(tree: ast.Module) -> int:
    """
    Find where a conftest calls for the interfaces to be built.

    :param tree: The parsed conftest.
    :return: Index of the call among the module's statements.
    """
    for index, node in enumerate(tree.body):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == BUILD_CALL
        ):
            return index
    raise AssertionError(f"no {BUILD_CALL} call")
