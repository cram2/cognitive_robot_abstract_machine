from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

import krrood.symbol_graph.symbol_graph

PACKAGE_DIRECTORY = Path(__file__).parents[3] / "krrood"
"""
The directory holding krrood's ``pyproject.toml``.
"""

# %% optional dependencies


def test_importing_the_symbol_graph_loads_no_optional_dependency():
    """
    Every :class:`~krrood.symbol_graph.symbol_graph.Symbol` subclass imports the symbol
    graph, so importing it must not load a package krrood declares only as optional.
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
        [
            sys.executable,
            "-I",
            "-X",
            "importtime",
            "-m",
            krrood.symbol_graph.symbol_graph.__name__,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded_top_level_packages = {
        line.rpartition("|")[2].strip().partition(".")[0]
        for line in import_report.stderr.splitlines()
    }

    assert loaded_top_level_packages & optional == set()
