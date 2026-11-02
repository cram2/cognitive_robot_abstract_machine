"""Finds modules, classes, functions and fields that have no docstring."""

from __future__ import annotations

import abc
import ast
import importlib
from dataclasses import dataclass
from functools import cached_property
from enum import StrEnum
from pathlib import Path
from types import ModuleType

from typing_extensions import Iterator, override

from krrood.development_tools.agents.checker import CodeRuleChecker, CodeRuleViolation
from krrood.development_tools.agents.prose_length import SourceProse
from krrood.utils import get_import_path_from_path, make_path_importable

# %% what is documented


class DocumentedKind(StrEnum):
    """The kinds of code that carry a docstring."""

    MODULE = "module"
    """A whole source file."""

    CLASS = "class"
    """A class definition."""

    FUNCTION = "function"
    """A function or method definition."""

    FIELD = "field"
    """An annotated assignment in a class body."""

@dataclass(frozen=True)
class ImportedModule:
    """A source file as the module Python imports it, which tells whether a
    method implements an abstract method of a base class."""

    path: Path
    """The source file of the module."""

    @cached_property
    def module(self) -> ModuleType | None:
        """The module imported from :attr:`path`; none when it cannot be
        imported."""
        make_path_importable(self.path.parent)
        package = get_import_path_from_path(str(self.path.parent))
        name = self.path.stem if package is None else f"{package}.{self.path.stem}"
        try:
            module = importlib.import_module(name.removesuffix(".__init__"))
        except Exception:
            # Handle unimportable modules gracefully: whatever the import raises, only
            # the methods of this module go unchecked.
            return None
        if module.__file__ is None or Path(module.__file__).resolve() != (
            self.path.resolve()
        ):
            return None
        return module

    def class_named(self, scope: tuple[str, ...] | None) -> type | None:
        """
        :param scope: The names of a class and the classes enclosing it; none for a
            class defined inside a function.
        :return: The class; none when it cannot be found.
        """
        if scope is None or self.module is None:
            return None
        found: ModuleType | type = self.module
        for name in scope:
            found = vars(found).get(name)
            if not isinstance(found, type):
                return None
        return found

    def implements_abstract_method(
        self, scope: tuple[str, ...] | None, method_name: str
    ) -> bool:
        """
        :param scope: The names of a class and the classes enclosing it.
        :param method_name: The name of a method of the class.
        :return: Whether the first base class defining *method_name* leaves it abstract.
        """
        for base in self.class_named(scope).__mro__[1:]:
            if method_name in vars(base):
                return (
                    isinstance(base, abc.ABCMeta)
                    and method_name in base.__abstractmethods__
                )
        return False


# %% the check


@dataclass(frozen=True)
class MissingDocstringChecker(CodeRuleChecker):
    """Finds undocumented code; a method implementing an abstract method
    inherits its base's docstring and needs none of its own."""

    @override
    def violations(self, path: Path) -> Iterator[CodeRuleViolation]:
        module = ast.parse(path.read_text())
        if module.body and ast.get_docstring(module) is None:
            yield self.violation(
                path, DocumentedKind.MODULE, 1, module.body[-1].end_lineno
            )
        yield from self.undocumented_definitions(path, module, (), ImportedModule(path))

    def undocumented_definitions(
        self,
        path: Path,
        parent: ast.AST,
        scope: tuple[str, ...] | None,
        imported: ImportedModule,
    ) -> Iterator[CodeRuleViolation]:
        """
        :param parent: A node whose descendants are checked.
        :param scope: The names of the classes enclosing *parent*; none inside a
            function, where a class cannot be looked up.
        :param imported: The module of *path*.
        :return: A violation for every undocumented class, function and field below
            *parent*. A method is skipped when its class cannot be looked up, and
            when it implements an abstract method, whose docstring suffices.
        """
        for node in ast.iter_child_nodes(parent):
            if isinstance(node, ast.ClassDef):
                yield from self.undocumented_fields(path, node)
                if ast.get_docstring(node) is None:
                    yield self.violation(
                        path,
                        DocumentedKind.CLASS,
                        self.first_line(node),
                        node.end_lineno,
                    )
                class_scope = None if scope is None else (*scope, node.name)
                yield from self.undocumented_definitions(
                    path, node, class_scope, imported
                )
                continue
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                yield from self.undocumented_definitions(path, node, scope, imported)
                continue
            if ast.get_docstring(node) is None and not self.is_exempt_method(
                parent, scope, node, imported
            ):
                yield self.violation(
                    path,
                    DocumentedKind.FUNCTION,
                    self.first_line(node),
                    node.end_lineno,
                )
            yield from self.undocumented_definitions(path, node, None, imported)

    @staticmethod
    def is_exempt_method(
        parent: ast.AST,
        scope: tuple[str, ...] | None,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        imported: ImportedModule,
    ) -> bool:
        """
        :return: Whether *node* is a method that needs no docstring of its own: its class
            cannot be looked up, or it implements an abstract method.
        """
        if not isinstance(parent, ast.ClassDef):
            return False
        if imported.class_named(scope) is None:
            return True
        return imported.implements_abstract_method(scope, node.name)

    @staticmethod
    def first_line(
        definition: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> int:
        """
        :return: The line of the first decorator of *definition*, else of its keyword.
        """
        decorators = definition.decorator_list
        return decorators[0].lineno if decorators else definition.lineno

    def undocumented_fields(
        self, path: Path, definition: ast.ClassDef
    ) -> Iterator[CodeRuleViolation]:
        """
        :return: A violation for every annotated assignment in *definition*'s body that no
            string literal directly follows.
        """
        for index, statement in enumerate(definition.body):
            if not isinstance(statement, ast.AnnAssign):
                continue
            following = definition.body[index + 1 : index + 2]
            if following and SourceProse.is_string(following[0]):
                continue
            yield self.violation(
                path, DocumentedKind.FIELD, statement.lineno, statement.end_lineno
            )

    @staticmethod
    def violation(
        path: Path, kind: DocumentedKind, first_line: int, last_line: int
    ) -> CodeRuleViolation:
        """
        :return: The violation of the undocumented *kind* of code spanning *first_line* to
            *last_line* of *path*.
        """
        return CodeRuleViolation(
            path, first_line, last_line, f"{kind} has no docstring; add one"
        )
