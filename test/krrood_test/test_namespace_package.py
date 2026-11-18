"""
Tests that ``krrood`` is a namespace package whose distributions read their version from
their metadata.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
from pathlib import Path

from krrood.utils import logger

import krrood


# %% namespace package
def test_krrood_is_a_namespace_package():
    assert importlib.util.find_spec("krrood").origin is None


def test_krrood_ships_no_package_initializer():
    initializers = [Path(entry) / "__init__.py" for entry in krrood.__path__]
    assert not any(initializer.exists() for initializer in initializers)


# %% version
def test_distribution_version_matches_root_version():
    root_version = (Path(__file__).parents[2] / "VERSION").read_text().strip()
    assert importlib.metadata.version("krrood") == root_version


# %% logger
def test_module_logger_is_below_the_krrood_logger():
    assert logger is logging.getLogger("krrood.utils")
