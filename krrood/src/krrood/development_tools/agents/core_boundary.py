"""
Finds the imports that cross a distribution's core boundary when a module is loaded.
"""

from __future__ import annotations

import ast
import functools
import importlib.metadata
import sys
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from typing_extensions import Any, Iterator, override

from krrood.development_tools.agents.checker import CodeRuleChecker, CodeRuleViolation
from krrood.source_analysis.import_graph import ModuleGraph

# %% file and key names


class DeclarationFile(StrEnum):
    """
    The files that declare a distribution and its packages.
    """

    PYPROJECT = "pyproject.toml"
    """
    The distribution's project declaration.
    """

    SOURCE_DIRECTORY = "src"
    """
    The directory of a src layout, which holds the distribution's packages.
    """


class PyprojectKey(StrEnum):
    """
    The keys read from a ``pyproject.toml``.
    """

    PROJECT = "project"
    """
    The table declaring the distribution's metadata.
    """

    NAME = "name"
    """
    The key in ``[project]`` holding the distribution's name.
    """

    OPTIONAL_DEPENDENCIES = "optional-dependencies"
    """
    The table in ``[project]`` listing the requirements of each extra.
    """

    TOOL = "tool"
    """
    The table holding the configuration of each tool.
    """

    KRROOD = "krrood"
    """
    The table in ``[tool]`` holding krrood's configuration.
    """

    CORE_BOUNDARY = "core-boundary"
    """
    The table in ``[tool.krrood]`` declaring the core boundary.
    """
    CORE = "core"
    """
    The key in ``[tool.krrood.core-boundary]`` listing the packages and modules of core.
    """

    DEPENDENCIES = "dependencies"
    """
    The key in ``[tool.krrood.core-boundary]`` listing the distributions core may load.
    """


# %% the boundary


@dataclass(kw_only=True, eq=False)
class CoreBoundary:
    """
    A distribution's core and the distributions core may load, as its
    ``pyproject.toml`` declares them in ``[tool.krrood.core-boundary]``.

    Every module of the distribution outside :attr:`core` is not core.
    """

    distribution: str
    """
    The normalized name of the distribution.
    """

    core: list[str]
    """
    The dotted names of the packages and modules core consists of.
    """

    dependencies: list[str]
    """
    The distributions core may load, without what they require in turn.
    """

    extras_by_distribution: dict[str, list[str]]
    """
    The extras requiring each distribution, keyed by its normalized name.
    """

    graph: ModuleGraph
    """
    The distribution's modules.
    """

    @classmethod
    def from_json(
        cls, *, pyproject: dict[str, Any], directory: Path
    ) -> CoreBoundary | None:
        """
        :param pyproject: The parsed content of a ``pyproject.toml``.
        :param directory: The directory holding it.
        :return: The boundary it declares, or none if it declares no core.
        """
        tool = pyproject.get(PyprojectKey.TOOL, {})
        declaration = tool.get(PyprojectKey.KRROOD, {}).get(
            PyprojectKey.CORE_BOUNDARY, {}
        )
        if PyprojectKey.CORE not in declaration:
            return None
        project = pyproject[PyprojectKey.PROJECT]
        extras_by_distribution: dict[str, list[str]] = {}
        for extra, requirements in project.get(
            PyprojectKey.OPTIONAL_DEPENDENCIES, {}
        ).items():
            for requirement in requirements:
                extras_by_distribution.setdefault(
                    canonicalize_name(Requirement(requirement).name), []
                ).append(extra)
        top_level_packages = {
            module.partition(".")[0] for module in declaration[PyprojectKey.CORE]
        }
        source_directory = (
            directory / DeclarationFile.SOURCE_DIRECTORY
            if (directory / DeclarationFile.SOURCE_DIRECTORY).is_dir()
            else directory
        )
        return cls(
            distribution=canonicalize_name(project[PyprojectKey.NAME]),
            core=declaration[PyprojectKey.CORE],
            dependencies=declaration[PyprojectKey.DEPENDENCIES],
            extras_by_distribution=extras_by_distribution,
            graph=ModuleGraph.from_packages(
                package_directories=[
                    source_directory / package for package in sorted(top_level_packages)
                ]
            ),
        )

    @staticmethod
    @functools.cache
    def from_pyproject(*, path: Path) -> CoreBoundary | None:
        """
        :param path: A ``pyproject.toml``.
        :return: The boundary it declares, or none if it declares no core.
        """
        return CoreBoundary.from_json(
            pyproject=tomllib.loads(path.read_text()), directory=path.parent
        )

    @classmethod
    def from_enclosed_python_source_file(cls, *, path: Path) -> CoreBoundary | None:
        """
        :param path: A Python source file.
        :return: The boundary the nearest ``pyproject.toml`` above it declares, or none
            if there is none or it declares no core.
        """
        pyproject = next(
            (
                directory / DeclarationFile.PYPROJECT
                for directory in path.parents
                if (directory / DeclarationFile.PYPROJECT).is_file()
            ),
            None,
        )
        return None if pyproject is None else cls.from_pyproject(path=pyproject)

    @cached_property
    def core_modules(self) -> list[str]:
        """
        The dotted names of every module of core.
        """
        return [
            module for module in self.graph.paths if self.includes_module(module=module)
        ]

    def includes_module(self, *, module: str) -> bool:
        """
        :param module: The dotted name of one of the distribution's modules.
        :return: Whether it is part of core: one of :attr:`core`, a module below one,
            or a package above one, which loading it loads.
        """
        return any(
            module == core_module
            or module.startswith(f"{core_module}.")
            or core_module.startswith(f"{module}.")
            for core_module in self.core
        )

    @cached_property
    def allowed_distributions(self) -> set[str]:
        """
        The normalized names of the distribution itself, of :attr:`dependencies` and of
        every installed distribution they require.
        """
        return {self.distribution} | self.requirement_closure(
            distributions=set(self.dependencies)
        )

    @staticmethod
    def requirement_closure(*, distributions: set[str]) -> set[str]:
        """
        :param distributions: Names of distributions.
        :return: Their normalized names and those of every installed distribution they
            require, directly or in turn, without extras.
        """
        installed = {
            canonicalize_name(distribution.name): distribution
            for distribution in importlib.metadata.distributions()
        }
        closure: set[str] = set()
        pending = [canonicalize_name(name) for name in distributions]
        while pending:
            distribution = pending.pop()
            if distribution in closure:
                continue
            closure.add(distribution)
            if distribution not in installed:
                continue
            pending.extend(
                canonicalize_name(requirement.name)
                for requirement in map(
                    Requirement, installed[distribution].requires or []
                )
                if requirement.marker is None
                or requirement.marker.evaluate({"extra": ""})
            )
        return closure

    @cached_property
    def distributions_by_package(self) -> dict[str, list[str]]:
        """
        The installed distributions providing each top-level package.
        """
        return importlib.metadata.packages_distributions()

    def disallowed_distributions(self, *, package: str) -> set[str]:
        """
        :param package: A top-level package outside the distribution.
        :return: The normalized names of the installed distributions providing it that
            core may not load; empty for the standard library.
        """
        if package in sys.stdlib_module_names:
            return set()
        provided_by = {
            canonicalize_name(distribution)
            for distribution in self.distributions_by_package.get(package, [])
        }
        if provided_by & self.allowed_distributions:
            return set()
        return provided_by or {canonicalize_name(package)}

    @functools.cache
    def distributions_loaded_beyond_core(self, *, module: str) -> frozenset[str]:
        """
        :param module: The dotted name of one of the distribution's modules.
        :return: The disallowed distributions that loading it loads, through any chain
            of module-level imports.
        """
        reached: set[str] = set()
        visited: set[str] = set()
        pending = [module]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            for loaded in self.graph.imports(module=current):
                if loaded.module in self.graph.paths:
                    pending.append(loaded.module)
                else:
                    reached |= self.disallowed_distributions(package=loaded.module)
        return frozenset(reached)

    def describe_violations_to_core_boundary_in_loaded_modules(
        self, *, loaded: list[str]
    ) -> str | None:
        """
        :param loaded: The modules one import statement of a core module loads.
        :return: Why running the statement when the module is loaded breaks the
            boundary, or none if it does not.
        """
        outside_core = [
            module
            for module in loaded
            if module in self.graph.paths and not self.includes_module(module=module)
        ]
        external = {
            module: self.disallowed_distributions(package=module)
            for module in loaded
            if module not in self.graph.paths
            and self.disallowed_distributions(package=module)
        }
        if outside_core:
            distributions = set().union(
                *(
                    self.distributions_loaded_beyond_core(module=module)
                    for module in outside_core
                )
            )
            loading = (
                f" and loads {', '.join(sorted(distributions))}"
                if distributions
                else ""
            )
            return f"imports `{outside_core[-1]}`, which is not core{loading}"
        if not external:
            return None
        distributions = set().union(*external.values())
        extras = sorted(
            {
                extra
                for distribution in distributions
                for extra in self.extras_by_distribution.get(distribution, [])
            }
        )
        installed_by = (
            f" (installed by the {', '.join(extras)} extra)" if extras else ""
        )
        return (
            f"imports `{', '.join(external)}` from {', '.join(sorted(distributions))}, "
            f"which is not a core dependency{installed_by}"
        )


# %% the checker


@dataclass(kw_only=True, frozen=True)
class CoreBoundaryChecker(CodeRuleChecker):
    """
    Finds the imports a core module runs when it is loaded that load a module outside
    core or a distribution core may not load.

    The boundary is read from the nearest ``pyproject.toml`` above the file, as
    :class:`CoreBoundary` describes; a file of no declared core is not checked.
    """

    @override
    def violations(self, *, path: Path) -> Iterator[CodeRuleViolation]:
        path = path.resolve()
        boundary = CoreBoundary.from_enclosed_python_source_file(path=path)
        if boundary is None:
            return
        module = boundary.graph.module_name(path=path)
        if module is None or not boundary.includes_module(module=module):
            return
        loaded_by_statement: dict[ast.stmt, list[str]] = {}
        for loaded in boundary.graph.imports(module=module):
            loaded_by_statement.setdefault(loaded.statement, []).append(loaded.module)
        for statement, loaded in loaded_by_statement.items():
            description = (
                boundary.describe_violations_to_core_boundary_in_loaded_modules(
                    loaded=loaded
                )
            )
            if description is None:
                continue
            yield CodeRuleViolation(
                path=path,
                first_line=statement.lineno,
                last_line=statement.end_lineno,
                description=f"{description}; import it inside the functions that need it",
            )
