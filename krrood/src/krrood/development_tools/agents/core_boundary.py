"""
Finds the imports that cross a distribution's core boundary when a module is loaded.
"""

from __future__ import annotations

import ast
import functools
import importlib.metadata
import sys
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from typing_extensions import Any, Iterator, override

from krrood.development_tools.agents.checker import CodeRuleChecker, CodeRuleViolation

# %% file and key names


class DeclarationFile(StrEnum):
    """
    The files that declare a distribution and its packages.
    """

    PYPROJECT = "pyproject.toml"
    """
    The distribution's project declaration.
    """

    PACKAGE_INITIALIZER = "__init__.py"
    """
    The file that makes a directory a package.
    """

    SOURCE_DIRECTORY = "src"
    """
    The directory of a src layout, which holds the distribution's packages.
    """

    PYTHON_SOURCE_PATTERN = "*.py"
    """
    The pattern of a Python source file.
    """


class PyprojectKey(StrEnum):
    """
    The keys read from a ``pyproject.toml``.
    """

    PROJECT = "project"
    NAME = "name"
    OPTIONAL_DEPENDENCIES = "optional-dependencies"
    TOOL = "tool"
    KRROOD = "krrood"
    CORE_BOUNDARY = "core-boundary"
    CORE = "core"
    """
    The key in ``[tool.krrood.core-boundary]`` listing the packages and modules of core.
    """

    DEPENDENCIES = "dependencies"
    """
    The key in ``[tool.krrood.core-boundary]`` listing the distributions core may load.
    """


class TypeCheckingFlag(StrEnum):
    """
    The name of the constant that is true only while a type checker reads the code.
    """

    NAME = "TYPE_CHECKING"


# %% imports run when a module is loaded


@dataclass
class ModuleLevelImportFinder(ast.NodeVisitor):
    """
    Collects the imports a module runs when it is loaded: those outside every function
    and outside every ``if TYPE_CHECKING:`` block.
    """

    imports: list[ast.Import | ast.ImportFrom] = field(default_factory=list)
    """
    The imports found so far.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """
        Skip the function: its body runs only when it is called.
        """

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        """
        Skip the body of an ``if TYPE_CHECKING:`` block, which never runs.
        """
        if ast.unparse(node.test).rpartition(".")[2] != TypeCheckingFlag.NAME:
            self.generic_visit(node)
            return
        for statement in node.orelse:
            self.visit(statement)

    def visit_Import(self, node: ast.Import) -> None:
        """
        Collect the import.
        """
        self.imports.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        Collect the import.
        """
        self.imports.append(node)


@dataclass(frozen=True)
class ModuleImport:
    """
    One module that an import statement loads, together with the statement.
    """

    statement: ast.Import | ast.ImportFrom
    """
    The import statement.
    """

    module: str
    """
    The absolute dotted name of a module the statement loads; a module outside the
    distribution is named by its top-level package.
    """


# %% the distribution's modules


@dataclass(eq=False)
class ModuleGraph:
    """
    The modules of a distribution and the modules each one loads when it is loaded.
    """

    paths: dict[str, Path]
    """
    The file of each module of the distribution, keyed by its dotted name.
    """

    @classmethod
    def from_packages(cls, package_directories: list[Path]) -> ModuleGraph:
        """
        :param package_directories: The directories of the distribution's top-level
            packages.
        :return: The graph of every module in them.
        """
        paths = {}
        for package_directory in package_directories:
            for path in package_directory.rglob(DeclarationFile.PYTHON_SOURCE_PATTERN):
                parts = path.relative_to(package_directory.parent).with_suffix("").parts
                if path.name == DeclarationFile.PACKAGE_INITIALIZER:
                    parts = parts[:-1]
                paths[".".join(parts)] = path
        return cls(paths)

    def module_name(self, path: Path) -> str | None:
        """
        :param path: A Python source file.
        :return: The dotted name of the module it holds, or none if it is not one of the
            distribution's modules.
        """
        return next(
            (name for name, module_path in self.paths.items() if module_path == path),
            None,
        )

    @functools.cache
    def imports(self, module: str) -> tuple[ModuleImport, ...]:
        """
        :param module: The dotted name of one of the distribution's modules.
        :return: Every module that loading it loads directly, parent packages included.
        """
        path = self.paths[module]
        finder = ModuleLevelImportFinder()
        finder.visit(ast.parse(path.read_text(), filename=str(path)))
        package = (
            module
            if path.name == DeclarationFile.PACKAGE_INITIALIZER
            else module.rpartition(".")[0]
        )
        return tuple(
            ModuleImport(statement, loaded)
            for statement in finder.imports
            for loaded in self.loaded_modules(statement, package)
        )

    def loaded_modules(
        self, statement: ast.Import | ast.ImportFrom, package: str
    ) -> list[str]:
        """
        :param statement: An import statement.
        :param package: The package of the module holding the statement.
        :return: The modules the statement loads: each imported module, the packages
            above it and, for a ``from`` import, every imported name that is a module.
        """
        if isinstance(statement, ast.Import):
            imported = [alias.name for alias in statement.names]
        else:
            base = statement.module or ""
            if statement.level > 0:
                anchor = package.rsplit(".", statement.level - 1)[0]
                base = f"{anchor}.{base}" if base else anchor
            imported = [base] + [
                f"{base}.{alias.name}"
                for alias in statement.names
                if f"{base}.{alias.name}" in self.paths
            ]
        loaded = []
        for name in imported:
            parts = name.split(".")
            if parts[0] not in self.top_level_packages:
                loaded.append(parts[0])
                continue
            loaded.extend(
                ".".join(parts[:length])
                for length in range(1, len(parts) + 1)
                if ".".join(parts[:length]) in self.paths
            )
        return list(dict.fromkeys(loaded))

    @cached_property
    def top_level_packages(self) -> set[str]:
        """
        The top-level packages of the distribution.
        """
        return {name.partition(".")[0] for name in self.paths}


# %% the boundary


@dataclass(eq=False)
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
        cls, pyproject: dict[str, Any], directory: Path
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
            canonicalize_name(project[PyprojectKey.NAME]),
            declaration[PyprojectKey.CORE],
            declaration[PyprojectKey.DEPENDENCIES],
            extras_by_distribution,
            ModuleGraph.from_packages(
                [source_directory / package for package in sorted(top_level_packages)]
            ),
        )

    @staticmethod
    @functools.cache
    def from_pyproject(path: Path) -> CoreBoundary | None:
        """
        :param path: A ``pyproject.toml``.
        :return: The boundary it declares, or none if it declares no core.
        """
        return CoreBoundary.from_json(tomllib.loads(path.read_text()), path.parent)

    @classmethod
    def enclosing(cls, path: Path) -> CoreBoundary | None:
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
        return None if pyproject is None else cls.from_pyproject(pyproject)

    @cached_property
    def core_modules(self) -> list[str]:
        """
        The dotted names of every module of core.
        """
        return [module for module in self.graph.paths if self.includes(module)]

    def includes(self, module: str) -> bool:
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
        return {self.distribution} | self.with_requirements(set(self.dependencies))

    @staticmethod
    def with_requirements(distributions: set[str]) -> set[str]:
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

    def disallowed_distributions(self, package: str) -> set[str]:
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
    def distributions_loaded_beyond_core(self, module: str) -> frozenset[str]:
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
            for loaded in self.graph.imports(current):
                if loaded.module in self.graph.paths:
                    pending.append(loaded.module)
                else:
                    reached |= self.disallowed_distributions(loaded.module)
        return frozenset(reached)

    def crossing_description(self, loaded: list[str]) -> str | None:
        """
        :param loaded: The modules one import statement of a core module loads.
        :return: Why running the statement when the module is loaded breaks the
            boundary, or none if it does not.
        """
        outside_core = [
            module
            for module in loaded
            if module in self.graph.paths and not self.includes(module)
        ]
        external = {
            module: self.disallowed_distributions(module)
            for module in loaded
            if module not in self.graph.paths and self.disallowed_distributions(module)
        }
        if outside_core:
            distributions = set().union(
                *map(self.distributions_loaded_beyond_core, outside_core)
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


@dataclass(frozen=True)
class CoreBoundaryChecker(CodeRuleChecker):
    """
    Finds the imports a core module runs when it is loaded that load a module outside
    core or a distribution core may not load.

    The boundary is read from the nearest ``pyproject.toml`` above the file, as
    :class:`CoreBoundary` describes; a file of no declared core is not checked.
    """

    @override
    def violations(self, path: Path) -> Iterator[CodeRuleViolation]:
        path = path.resolve()
        boundary = CoreBoundary.enclosing(path)
        if boundary is None:
            return
        module = boundary.graph.module_name(path)
        if module is None or not boundary.includes(module):
            return
        loaded_by_statement: dict[ast.stmt, list[str]] = {}
        for loaded in boundary.graph.imports(module):
            loaded_by_statement.setdefault(loaded.statement, []).append(loaded.module)
        for statement, loaded in loaded_by_statement.items():
            description = boundary.crossing_description(loaded)
            if description is None:
                continue
            yield CodeRuleViolation(
                path,
                statement.lineno,
                statement.end_lineno,
                f"{description}; import it inside the functions that need it",
            )
