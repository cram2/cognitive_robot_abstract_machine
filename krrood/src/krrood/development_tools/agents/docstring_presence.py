"""
Finds modules, classes, functions and fields that have no docstring.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Iterator, override

from krrood.development_tools.agents.checker import CodeRuleChecker, CodeRuleViolation
from krrood.development_tools.agents.prose_length import SourceProse

# %% what is documented


class DocumentedKind(StrEnum):
    """
    The kinds of code that carry a docstring.
    """

    MODULE = "module"
    """
    A whole source file.
    """

    CLASS = "class"
    """
    A class definition.
    """

    FUNCTION = "function"
    """
    A function or method definition.
    """

    FIELD = "field"
    """
    An annotated assignment in a class body.
    """


class DecoratorName(StrEnum):
    """
    The decorators the checks recognise by name.
    """

    OVERRIDE = "override"
    """
    :func:`typing_extensions.override`, which marks a method inheriting its base's
    docstring.
    """


# %% the check


@dataclass(frozen=True)
class MissingDocstringChecker(CodeRuleChecker):
    """
    Finds undocumented code; a method decorated with :func:`typing_extensions.override`
    inherits its base's docstring and needs none of its own.
    """

    @override
    def violations(self, path: Path) -> Iterator[CodeRuleViolation]:
        module = ast.parse(path.read_text())
        if module.body and ast.get_docstring(module) is None:
            yield self.violation(
                path, DocumentedKind.MODULE, 1, module.body[-1].end_lineno
            )
        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef):
                yield from self.undocumented_fields(path, node)
            if not isinstance(
                node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ):
                continue
            if ast.get_docstring(node) is not None or self.is_override(node):
                continue
            kind = (
                DocumentedKind.CLASS
                if isinstance(node, ast.ClassDef)
                else DocumentedKind.FUNCTION
            )
            first = node.decorator_list[0] if node.decorator_list else node
            yield self.violation(path, kind, first.lineno, node.end_lineno)

    @staticmethod
    def is_override(
        definition: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        """
        :return: Whether *definition* is decorated with :func:`typing_extensions.override`.
        """
        return any(
            (isinstance(decorator, ast.Name) and decorator.id == DecoratorName.OVERRIDE)
            or (
                isinstance(decorator, ast.Attribute)
                and decorator.attr == DecoratorName.OVERRIDE
            )
            for decorator in definition.decorator_list
        )

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
