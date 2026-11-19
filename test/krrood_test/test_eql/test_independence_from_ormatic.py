"""
Tests that the entity query language loads without ORMatic, so the two ship as separate
distributions.
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys

import krrood.entity_query_language
import krrood.ormatic


# %% importing the query language loads no part of ORMatic


def test_importing_every_query_language_module_loads_no_ormatic_module() -> None:
    """
    Every module of the entity query language imports in a fresh interpreter without
    loading a module of :mod:`krrood.ormatic`.
    """
    module_names = [
        module.name
        for module in pkgutil.walk_packages(
            krrood.entity_query_language.__path__,
            f"{krrood.entity_query_language.__name__}.",
        )
    ]
    program = "\n".join(
        [
            "import importlib, sys",
            f"for name in {module_names!r}:",
            "    importlib.import_module(name)",
            "print('\\n'.join(sys.modules))",
        ]
    )
    loaded = subprocess.run(
        [sys.executable, "-I", "-c", program],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    ormatic_prefix = f"{krrood.ormatic.__name__}."
    assert [
        name
        for name in loaded
        if name == krrood.ormatic.__name__ or name.startswith(ormatic_prefix)
    ] == []
