"""
The modules a Python module loads when it is loaded, read from its source.
"""

from __future__ import annotations

import ast
import functools
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from pathlib import Path

from krrood.ormatic.utils import classproperty

# %% file names


class PythonSourceFile(StrEnum):
    """
    The names that mark the files of Python modules and packages.
    """

    PACKAGE_INITIALIZER = "__init__.py"
    """
    The file that makes a directory a package.
    """

    PYTHON_SOURCE_PATTERN = "*.py"
    """
    The pattern of a Python source file.
    """


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

    @classproperty
    def type_checking_flag(cls) -> str:
        """
        The name of the constant that is true only while a type checker reads the code.
        """
        return "TYPE_CHECKING"

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """
        Skip the function: its body runs only when it is called.
        """

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        """
        Skip the body of an ``if TYPE_CHECKING:`` block, which never runs.
        """
        if ast.unparse(node.test).rpartition(".")[2] != self.type_checking_flag:
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
            for path in package_directory.rglob(PythonSourceFile.PYTHON_SOURCE_PATTERN):
                parts = path.relative_to(package_directory.parent).with_suffix("").parts
                if path.name == PythonSourceFile.PACKAGE_INITIALIZER:
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
            if path.name == PythonSourceFile.PACKAGE_INITIALIZER
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
