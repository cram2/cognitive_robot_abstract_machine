"""Finds modules, classes, functions and fields that have no docstring."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Iterator, override

from krrood.development_tools.agents.checker import CodeRuleChecker, CodeRuleViolation
from krrood.development_tools.agents.imported_module import ImportedModule
from krrood.development_tools.agents.prose_length import SourceProse

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


# %% the check


@dataclass(kw_only=True, frozen=True)
class MissingDocstringChecker(CodeRuleChecker):
    """Finds undocumented code; a method implementing an abstract method
    inherits its base's docstring and needs none of its own."""

    @override
    def violations(self, *, path: Path) -> Iterator[CodeRuleViolation]:
        module = ast.parse(path.read_text())
        if module.body and ast.get_docstring(module) is None:
            yield self.violation(
                path=path,
                kind=DocumentedKind.MODULE,
                first_line=1,
                last_line=module.body[-1].end_lineno,
            )
        yield from self.undocumented_definitions(
            path=path, parent=module, scope=(), imported=ImportedModule(path=path)
        )

    def undocumented_definitions(
        self,
        *,
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
                yield from self.undocumented_fields(path=path, definition=node)
                if ast.get_docstring(node) is None:
                    yield self.violation(
                        path=path,
                        kind=DocumentedKind.CLASS,
                        first_line=self.first_line(definition=node),
                        last_line=node.end_lineno,
                    )
                class_scope = None if scope is None else (*scope, node.name)
                yield from self.undocumented_definitions(
                    path=path, parent=node, scope=class_scope, imported=imported
                )
                continue
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                yield from self.undocumented_definitions(
                    path=path, parent=node, scope=scope, imported=imported
                )
                continue
            if ast.get_docstring(node) is None and not self.is_exempt_method(
                parent=parent, scope=scope, node=node, imported=imported
            ):
                yield self.violation(
                    path=path,
                    kind=DocumentedKind.FUNCTION,
                    first_line=self.first_line(definition=node),
                    last_line=node.end_lineno,
                )
            yield from self.undocumented_definitions(
                path=path, parent=node, scope=None, imported=imported
            )

    @staticmethod
    def is_exempt_method(
        *,
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
        if imported.class_named(scope=scope) is None:
            return True
        return imported.implements_abstract_method(scope=scope, method_name=node.name)

    @staticmethod
    def first_line(
        *,
        definition: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> int:
        """
        :return: The line of the first decorator of *definition*, else of its keyword.
        """
        decorators = definition.decorator_list
        return decorators[0].lineno if decorators else definition.lineno

    def undocumented_fields(
        self, *, path: Path, definition: ast.ClassDef
    ) -> Iterator[CodeRuleViolation]:
        """
        :return: A violation for every annotated assignment in *definition*'s body that no
            string literal directly follows.
        """
        for index, statement in enumerate(definition.body):
            if not isinstance(statement, ast.AnnAssign):
                continue
            following = definition.body[index + 1 : index + 2]
            if following and SourceProse.is_string(statement=following[0]):
                continue
            yield self.violation(
                path=path,
                kind=DocumentedKind.FIELD,
                first_line=statement.lineno,
                last_line=statement.end_lineno,
            )

    @staticmethod
    def violation(
        *, path: Path, kind: DocumentedKind, first_line: int, last_line: int
    ) -> CodeRuleViolation:
        """
        :return: The violation of the undocumented *kind* of code spanning *first_line* to
            *last_line* of *path*.
        """
        return CodeRuleViolation(
            path=path,
            first_line=first_line,
            last_line=last_line,
            description=f"{kind} has no docstring; add one",
        )
