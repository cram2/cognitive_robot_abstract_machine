"""
Building a verbalization clause is declarative: the inflected words it decides are
realized later, so building one loads none of the libraries realization depends on.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path

from krrood.development_tools.agents.core_boundary import CoreBoundary
from ...test_core_boundary.test_core_boundary import (
    krrood_core_boundary,
    packages_imported_while_running,
)

# %% realization


class RealizationModule(StrEnum):
    """
    The modules that realize a clause as text, whose libraries building a clause must
    not load.
    """

    MORPHOLOGY = "krrood.entity_query_language.verbalization.morphology"
    """
    Inflects words.
    """

    PIPELINE = "krrood.entity_query_language.verbalization.pipeline"
    """
    Renders the realized clause.
    """


@dataclass(kw_only=True)
class RealizationLibraries:
    """
    The libraries that the realization modules load and core may not.
    """

    boundary: CoreBoundary
    """
    The boundary whose core the libraries lie beyond.
    """

    @cached_property
    def distributions(self) -> set[str]:
        """
        The normalized names of the distributions the realization modules import and of
        every distribution they require, except those core may load.
        """
        imported_packages = {
            loaded.module
            for module in RealizationModule
            for loaded in self.boundary.graph.imports(module=module)
            if loaded.module not in self.boundary.graph.paths
        }
        imported_distributions = set().union(
            *(
                self.boundary.disallowed_distributions(package=package)
                for package in imported_packages
            )
        )
        return (
            self.boundary.requirement_closure(distributions=imported_distributions)
            - self.boundary.allowed_distributions
        )

    @cached_property
    def packages(self) -> set[str]:
        """
        The top-level packages that the :attr:`distributions` provide.
        """
        return {
            package
            for package in importlib.metadata.packages_distributions()
            if self.boundary.disallowed_distributions(package=package)
            & self.distributions
        }


# %% building a clause


def test_realization_libraries_include_what_the_realization_modules_import(
    krrood_core_boundary: CoreBoundary,
) -> None:
    """
    The libraries derived from the realization modules include the ones those modules
    import by name, so the check below forbids them.
    """
    packages = RealizationLibraries(boundary=krrood_core_boundary).packages

    assert {"inflect", "lemminflect", "jinja2"} <= packages


def test_building_clauses_loads_no_realization_library(
    krrood_core_boundary: CoreBoundary,
) -> None:
    """
    Every vocabulary path that decides an inflected word - a verb read off a class name,
    a set summarised by count, an indefinite article, an indexed collection hop - builds
    its clause without loading a realization library.
    """
    source = (Path(__file__).parent / "dataset" / "clause_building.py").read_text()

    loaded = {imported.package for imported in packages_imported_while_running(source)}

    realization_libraries = RealizationLibraries(boundary=krrood_core_boundary)
    assert loaded & realization_libraries.packages == set()
