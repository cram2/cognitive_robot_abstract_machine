from __future__ import annotations

import subprocess
import sys
import tomllib
from enum import StrEnum
from pathlib import Path

import pytest
import rustworkx as rx
from packaging.requirements import Requirement

import krrood
import krrood.entity_query_language.factories
import krrood.entity_query_language.query_graph
import krrood.ormatic.ormatic
import krrood.rustworkx_utils.rxnode
import krrood.symbol_graph.symbol_graph
from krrood.ripple_down_rules.rdr import SingleClassRDR
from krrood.ripple_down_rules.utils import draw_tree
from krrood.rustworkx_utils.graph_visualizer import GraphVisualizer
from krrood.rustworkx_utils.rxnode import RWXNode

PACKAGE_DIRECTORY = Path(__file__).parents[2] / "krrood"
"""
The directory holding krrood's ``pyproject.toml``.
"""


class OptionalModule(StrEnum):
    """
    The modules krrood imports only inside the functions that need them.
    """

    MATPLOTLIB = "matplotlib"
    """
    Draws graphs and rule trees.
    """


@pytest.fixture
def matplotlib_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Make matplotlib look uninstalled for the duration of a test.
    """
    monkeypatch.setitem(sys.modules, OptionalModule.MATPLOTLIB, None)


# %% importing loads no optional dependency


@pytest.mark.parametrize(
    "module_name",
    [
        krrood.symbol_graph.symbol_graph.__name__,
        krrood.entity_query_language.factories.__name__,
        krrood.entity_query_language.query_graph.__name__,
        krrood.ormatic.ormatic.__name__,
        krrood.rustworkx_utils.rxnode.__name__,
    ],
)
def test_importing_loads_no_optional_dependency(module_name: str):
    """
    Importing a module that does not draw must not load a package krrood declares only
    as optional.
    """
    declaration = tomllib.loads((PACKAGE_DIRECTORY / "pyproject.toml").read_text())
    project = declaration["project"]
    required = {Requirement(r).name for r in project["dependencies"]}
    optional = {
        Requirement(requirement).name
        for extra in project["optional-dependencies"].values()
        for requirement in extra
    } - required
    # A fresh interpreter, because the test run has already loaded what it would load.
    # ``-X importtime`` reports every module the import loads, one per stderr line.
    import_report = subprocess.run(
        [sys.executable, "-I", "-X", "importtime", "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded_top_level_packages = {
        line.rpartition("|")[2].strip().partition(".")[0]
        for line in import_report.stderr.splitlines()
    }

    assert loaded_top_level_packages & optional == set()


# %% drawing without matplotlib


def test_rendering_a_graph_without_matplotlib_raises(matplotlib_not_installed: None):
    visualizer = GraphVisualizer(RWXNode("Root", rx.PyDAG()))

    with pytest.raises(ModuleNotFoundError) as error:
        visualizer.render()

    assert error.value.name == OptionalModule.MATPLOTLIB


def test_animating_a_rule_tree_without_matplotlib_raises(
    matplotlib_not_installed: None,
):
    with pytest.raises(ModuleNotFoundError) as error:
        SingleClassRDR().fit([], animate_tree=True)

    assert error.value.name == OptionalModule.MATPLOTLIB


def test_updating_rule_tree_figures_without_matplotlib_raises(
    matplotlib_not_installed: None,
):
    with pytest.raises(ModuleNotFoundError) as error:
        SingleClassRDR().update_figures()

    assert error.value.name == OptionalModule.MATPLOTLIB


def test_drawing_a_rule_tree_without_matplotlib_raises(
    matplotlib_not_installed: None,
):
    with pytest.raises(ModuleNotFoundError) as error:
        draw_tree(None, None)

    assert error.value.name == OptionalModule.MATPLOTLIB
