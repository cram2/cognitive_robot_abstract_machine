"""
Finding imports of optional modules that the module's decorator does not guard.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import Iterable

from krrood.patterns.optional_module import OptionalModule

# %% unguarded imports


@dataclass(frozen=True)
class UnguardedOptionalImport:
    """
    An import of an optional module outside every function decorated with that module's
    :meth:`~krrood.patterns.optional_module.OptionalModule.require` and outside a
    ``TYPE_CHECKING`` block.
    """

    path: Path
    """
    The file holding the import.
    """

    line_number: int
    """
    The line the import statement starts on.
    """

    module: OptionalModule
    """
    The optional module it imports.
    """

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.line_number}: imports {self.module} outside a function "
            f"decorated with @{OptionalImportCheck.guarding_decorator(self.module)} "
            "and outside `if TYPE_CHECKING:`"
        )


# %% finding them in one file


@dataclass
class UnguardedOptionalImportFinder(ast.NodeVisitor):
    """
    Walks the syntax tree of one file, collecting its unguarded optional imports.
    """

    path: Path
    """
    The file being walked.
    """

    optional_modules: list[OptionalModule]
    """
    The modules whose imports must be guarded.
    """

    guarded_modules: frozenset[OptionalModule] = frozenset()
    """
    The modules the decorators of the functions enclosing the current node guard.
    """

    inside_type_checking_block: bool = False
    """
    Whether the current node is inside an ``if TYPE_CHECKING:`` block.
    """

    unguarded_imports: list[UnguardedOptionalImport] = field(default_factory=list)
    """
    The unguarded imports found so far.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        enclosing_guarded_modules = self.guarded_modules
        self.guarded_modules = enclosing_guarded_modules | {
            module
            for module in self.optional_modules
            for decorator in node.decorator_list
            if ast.unparse(decorator) == OptionalImportCheck.guarding_decorator(module)
        }
        self.generic_visit(node)
        self.guarded_modules = enclosing_guarded_modules

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        if ast.unparse(node.test).rpartition(".")[2] != "TYPE_CHECKING":
            self.generic_visit(node)
            return
        enclosing_state = self.inside_type_checking_block
        self.inside_type_checking_block = True
        for statement in node.body:
            self.visit(statement)
        self.inside_type_checking_block = enclosing_state
        for statement in node.orelse:
            self.visit(statement)

    def visit_Import(self, node: ast.Import) -> None:
        self.record_unguarded_import(node, [alias.name for alias in node.names])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level > 0:
            return
        self.record_unguarded_import(node, [node.module])

    def record_unguarded_import(
        self, node: ast.Import | ast.ImportFrom, imported_names: list[str]
    ) -> None:
        """
        Record the optional modules an import statement loads, unless they are guarded.

        :param node: The import statement.
        :param imported_names: The absolute names of the modules it imports.
        """
        if self.inside_type_checking_block:
            return
        self.unguarded_imports.extend(
            UnguardedOptionalImport(self.path, node.lineno, module)
            for module in self.optional_modules
            if module not in self.guarded_modules
            and any(
                name == module.value or name.startswith(f"{module.value}.")
                for name in imported_names
            )
        )


# %% the check


@dataclass
class OptionalImportCheck:
    """
    The check that a package imports its optional modules only where they are guarded.
    """

    optional_modules: list[OptionalModule]
    """
    The modules whose imports must be guarded.
    """

    @staticmethod
    def guarding_decorator(module: OptionalModule) -> str:
        """
        :param module: An optional module.
        :return: The source of the decorator that guards an import of it.
        """
        return (
            f"{type(module).__name__}.{module._name_}.{OptionalModule.require.__name__}"
        )

    def find_unguarded_imports(
        self, paths: Iterable[Path]
    ) -> list[UnguardedOptionalImport]:
        """
        :param paths: The Python files to check.
        :return: Every unguarded optional import in them, in file and line order.
        """
        unguarded_imports = []
        for path in paths:
            finder = UnguardedOptionalImportFinder(path, self.optional_modules)
            finder.visit(ast.parse(path.read_text(), filename=str(path)))
            unguarded_imports.extend(finder.unguarded_imports)
        return unguarded_imports
