"""
The runtime backstop of krrood's core boundary: loading all of core in a fresh
interpreter loads no distribution the static import graph does not predict.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

from packaging.utils import canonicalize_name

import krrood
from krrood.development_tools.agents.core_boundary import (
    CoreBoundary,
    DeclarationFile,
)

# %% loading core


def top_level_modules_loaded_by(source: str) -> set[str]:
    """
    :param source: The Python source a fresh, isolated interpreter runs.
    :return: The top-level modules the interpreter loads while starting and running it,
        as ``python -X importtime`` reports them.
    """
    import_report = subprocess.run(
        [sys.executable, "-I", "-X", "importtime", "-c", source],
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        line.rpartition("|")[2].strip().partition(".")[0]
        for line in import_report.stderr.splitlines()
        if line.startswith("import time:")
    }


def test_loading_core_loads_only_the_distributions_its_import_graph_predicts() -> None:
    """
    Catches what the static graph cannot see, such as a dynamic import or a dependency
    that loads a distribution it does not declare.
    """
    boundary = CoreBoundary.from_pyproject(
        Path(krrood.__file__).parents[2] / DeclarationFile.PYPROJECT
    )
    source = "\n".join(f"import {module}" for module in boundary.core_modules)
    loaded_packages = top_level_modules_loaded_by(source) - top_level_modules_loaded_by(
        ""
    )
    distributions_by_package = importlib.metadata.packages_distributions()
    loaded_distributions = {
        canonicalize_name(distribution)
        for package in loaded_packages
        for distribution in distributions_by_package.get(package, [])
    }
    predicted = boundary.with_requirements(
        set().union(
            *map(boundary.distributions_loaded_beyond_core, boundary.core_modules)
        )
    )

    assert loaded_distributions - boundary.allowed_distributions - predicted == set()
