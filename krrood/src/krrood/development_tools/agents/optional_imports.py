"""
Finds core modules that import an optional library at module level.
"""

from __future__ import annotations

import ast
import functools
import importlib.metadata
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from typing_extensions import Any, Iterator, override

from krrood.development_tools.agents.checker import CodeRuleChecker, CodeRuleViolation

# %% the distribution's declaration


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


class PyprojectKey(StrEnum):
    """
    The keys read from a ``pyproject.toml``.
    """

    PROJECT = "project"
    DEPENDENCIES = "dependencies"
    OPTIONAL_DEPENDENCIES = "optional-dependencies"
    TOOL = "tool"
    IMPORT_LINTER = "importlinter"
    CONTRACTS = "contracts"
    CONTRACT_IDENTIFIER = "id"
    SOURCE_MODULES = "source_modules"
    KRROOD = "krrood"
    CORE_BOUNDARY = "core-boundary"
    CORE_CONTRACT = "core-contract"
    """
    The key in ``[tool.krrood.core-boundary]`` naming the import-linter contract whose
    source modules are the distribution's core.
    """


@dataclass(frozen=True)
class CoreDeclaration:
    """
    A distribution's core modules and the libraries only its extras install, as its
    ``pyproject.toml`` declares them.
    """

    core_modules: tuple[str, ...]
    """
    The dotted names of the modules and packages that make up core.
    """

    extras_by_optional_distribution: dict[str, tuple[str, ...]] = field(hash=False)
    """
    The extras installing each distribution that the required dependencies do not,
    keyed by the distribution's normalized name.
    """

    @classmethod
    def from_json(cls, pyproject: dict[str, Any]) -> CoreDeclaration | None:
        """
        :param pyproject: The parsed content of a ``pyproject.toml``.
        :return: Its declaration, or none if it names no core contract.
        """
        tool = pyproject.get(PyprojectKey.TOOL, {})
        boundary = tool.get(PyprojectKey.KRROOD, {}).get(PyprojectKey.CORE_BOUNDARY, {})
        if PyprojectKey.CORE_CONTRACT not in boundary:
            return None
        contract = next(
            contract
            for contract in tool[PyprojectKey.IMPORT_LINTER][PyprojectKey.CONTRACTS]
            if contract[PyprojectKey.CONTRACT_IDENTIFIER]
            == boundary[PyprojectKey.CORE_CONTRACT]
        )
        project = pyproject[PyprojectKey.PROJECT]
        required = {
            canonicalize_name(Requirement(requirement).name)
            for requirement in project.get(PyprojectKey.DEPENDENCIES, [])
        }
        extras_by_optional_distribution: dict[str, tuple[str, ...]] = {}
        for extra, requirements in project.get(
            PyprojectKey.OPTIONAL_DEPENDENCIES, {}
        ).items():
            for requirement in requirements:
                distribution = canonicalize_name(Requirement(requirement).name)
                if distribution in required:
                    continue
                extras_by_optional_distribution[distribution] = (
                    *extras_by_optional_distribution.get(distribution, ()),
                    extra,
                )
        return cls(
            tuple(contract[PyprojectKey.SOURCE_MODULES]),
            extras_by_optional_distribution,
        )

    @staticmethod
    @functools.cache
    def from_pyproject(path: Path) -> CoreDeclaration | None:
        """
        :param path: A ``pyproject.toml``.
        :return: Its declaration, or none if it names no core contract.
        """
        return CoreDeclaration.from_json(tomllib.loads(path.read_text()))

    def includes(self, module_name: str) -> bool:
        """
        :param module_name: The dotted name of a module.
        :return: Whether the module is part of core.
        """
        return any(
            module_name == core_module or module_name.startswith(f"{core_module}.")
            for core_module in self.core_modules
        )

    def extras_installing(self, top_level_package: str) -> dict[str, tuple[str, ...]]:
        """
        :param top_level_package: The first component of an imported module's name.
        :return: The extras installing each optional distribution that provides the
            package, keyed by the distribution's name; empty for a package of an
            installed required or standard library distribution, or of none installed.
        """
        distributions = importlib.metadata.packages_distributions().get(
            top_level_package, []
        )
        return {
            distribution: self.extras_by_optional_distribution[
                canonicalize_name(distribution)
            ]
            for distribution in distributions
            if canonicalize_name(distribution) in self.extras_by_optional_distribution
        }


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
        if ast.unparse(node.test).rpartition(".")[2] != "TYPE_CHECKING":
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
        Collect the import unless it is relative, and so of the distribution itself.
        """
        if node.level == 0:
            self.imports.append(node)

    @property
    def imported_top_level_packages(self) -> Iterator[tuple[ast.stmt, str]]:
        """
        Each import found with the top-level package of every module it imports.
        """
        for node in self.imports:
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module]
            )
            for name in names:
                yield node, name.partition(".")[0]


# %% the checker


@dataclass(frozen=True)
class OptionalImportChecker(CodeRuleChecker):
    """
    Finds the imports of an optional library that a core module runs when it is loaded.

    Core and the extras are read from the nearest ``pyproject.toml`` above the file, as
    :class:`CoreDeclaration` describes; only installed distributions are recognized.
    """

    @override
    def violations(self, path: Path) -> Iterator[CodeRuleViolation]:
        path = path.resolve()
        pyproject = next(
            (
                directory / DeclarationFile.PYPROJECT
                for directory in path.parents
                if (directory / DeclarationFile.PYPROJECT).is_file()
            ),
            None,
        )
        if pyproject is None:
            return
        declaration = CoreDeclaration.from_pyproject(pyproject)
        if declaration is None or not declaration.includes(self.module_name(path)):
            return
        finder = ModuleLevelImportFinder()
        finder.visit(ast.parse(path.read_text(), filename=str(path)))
        for node, package in finder.imported_top_level_packages:
            for distribution, extras in declaration.extras_installing(package).items():
                yield CodeRuleViolation(
                    path,
                    node.lineno,
                    node.end_lineno,
                    f"imports `{package}` at module level, which only the "
                    f"{', '.join(extras)} extra of the distribution installs (through "
                    f"`{distribution}`); import it inside the functions that need it",
                )

    @staticmethod
    def module_name(path: Path) -> str:
        """
        :param path: A Python source file.
        :return: The dotted name it is imported by, from the outermost package that
            holds it.
        """
        components = (
            [] if path.name == DeclarationFile.PACKAGE_INITIALIZER else [path.stem]
        )
        directory = path.parent
        while (directory / DeclarationFile.PACKAGE_INITIALIZER).is_file():
            components.insert(0, directory.name)
            directory = directory.parent
        return ".".join(components)
