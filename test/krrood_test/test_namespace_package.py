"""
Tests that ``krrood`` is a namespace package whose distributions read their version from
their metadata, and whose modules keep their full import paths.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
from pathlib import Path

import krrood.entity_query_language
from krrood.source_analysis.import_scope import logger
from krrood.utils import get_import_path_from_path

import krrood


# %% namespace package
def test_krrood_is_a_namespace_package() -> None:
    """
    :mod:`krrood` is found without an initializer, so every distribution adds to it.
    """
    assert importlib.util.find_spec("krrood").origin is None


def test_krrood_ships_no_package_initializer() -> None:
    """
    No distribution ships ``krrood/__init__.py``, which would end the namespace package.
    """
    initializers = [Path(entry) / "__init__.py" for entry in krrood.__path__]
    assert not any(initializer.exists() for initializer in initializers)


# %% version
def test_distribution_version_matches_root_version() -> None:
    """
    The installed ``krrood`` distribution has the version of the repository.
    """
    root_version = (Path(__file__).parents[2] / "VERSION").read_text().strip()
    assert importlib.metadata.version("krrood") == root_version


# %% logger
def test_module_logger_is_below_the_krrood_logger() -> None:
    """
    A module logger is a child of the ``krrood`` logger, so configuring that logger
    still configures every module of every krrood distribution.
    """
    assert logger is logging.getLogger("krrood.source_analysis.import_scope")


# %% import path
def test_import_path_of_a_subpackage_keeps_the_namespace() -> None:
    """
    A subpackage of the namespace package is importable only through the namespace, so
    its import path starts with ``krrood``.
    """
    subpackage = Path(krrood.entity_query_language.__file__).parent
    assert (
        get_import_path_from_path(str(subpackage))
        == krrood.entity_query_language.__name__
    )


def test_import_path_of_a_package_outside_the_search_path_starts_at_the_package(
    tmp_path: Path,
) -> None:
    """
    Generated code is written to a regular package outside every search path, which is
    imported by its own name once its parent directory is put on the search path.
    """
    package = tmp_path / "generated_model"
    package.mkdir()
    (package / "__init__.py").touch()
    assert get_import_path_from_path(str(package)) == package.name
