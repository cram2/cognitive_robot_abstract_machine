"""
Finds field accessors written by hand instead of on
:class:`~krrood.patterns.dataclass_field_descriptor.DataclassFieldDescriptor`.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Iterator, override

from krrood.development_tools.agents.checker import CodeRuleChecker, CodeRuleViolation
from krrood.patterns.dataclass_field_descriptor import DataclassFieldDescriptor

# %% what is written by hand


class HandWrittenAccessor(StrEnum):
    """
    The ways of managing a field by hand, each valued with how to fix it.
    """

    DATA_DESCRIPTOR = (
        "class defines `__set__`; subclass `DataclassFieldDescriptor` and implement "
        "`read` and `write` instead"
    )
    """
    A class of its own defining ``__set__``.
    """

    DATACLASS_PROPERTY_SETTER = (
        "dataclass field is managed by a property setter; manage it with a "
        "`DataclassFieldDescriptor` subclass instead"
    )
    """
    A property setter in a dataclass named like one of its fields, or like a field
    with a leading underscore.
    """


# %% the check


@dataclass(kw_only=True, frozen=True)
class HandWrittenFieldAccessorChecker(CodeRuleChecker):
    """
    Finds every class but :attr:`base_class` that defines ``__set__``, and every
    property setter that manages a field of a dataclass.
    """

    base_class: type = DataclassFieldDescriptor
    """
    The one class allowed to define ``__set__``.
    """

    @override
    def violations(self, *, path: Path) -> Iterator[CodeRuleViolation]:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            for statement in node.body:
                accessor = self.hand_written_accessor(
                    path=path, definition=node, statement=statement
                )
                if accessor is None:
                    continue
                decorators = statement.decorator_list
                first_line = decorators[0].lineno if decorators else statement.lineno
                yield CodeRuleViolation(
                    path=path,
                    first_line=first_line,
                    last_line=statement.end_lineno,
                    description=accessor,
                )

    def hand_written_accessor(
        self, *, path: Path, definition: ast.ClassDef, statement: ast.stmt
    ) -> HandWrittenAccessor | None:
        """
        :param path: The file *definition* is in.
        :param definition: A class definition.
        :param statement: A statement of its body.
        :return: How *statement* manages a field by hand; none if it does not.
        """
        if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            return None
        if statement.name == "__set__" and not self.is_base_class(
            path=path, definition=definition
        ):
            return HandWrittenAccessor.DATA_DESCRIPTOR
        if not self.is_dataclass(definition=definition):
            return None
        if any(
            isinstance(decorator, ast.Attribute)
            and decorator.attr == "setter"
            and statement.name in self.field_names(definition=definition)
            for decorator in statement.decorator_list
        ):
            return HandWrittenAccessor.DATACLASS_PROPERTY_SETTER
        return None

    @staticmethod
    def field_names(*, definition: ast.ClassDef) -> set[str]:
        """
        :return: The names a field of *definition* is reached by: each field's own name
            and, for a field with a leading underscore, the name without it.
        """
        names = {
            statement.target.id
            for statement in definition.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        }
        return names | {name.removeprefix("_") for name in names}

    def is_base_class(self, *, path: Path, definition: ast.ClassDef) -> bool:
        """
        :return: Whether *definition* in *path* is the definition of :attr:`base_class`.
        """
        return definition.name == self.base_class.__name__ and path.resolve() == (
            Path(inspect.getsourcefile(self.base_class)).resolve()
        )

    @staticmethod
    def is_dataclass(*, definition: ast.ClassDef) -> bool:
        """
        :return: Whether *definition* is decorated with :func:`dataclasses.dataclass`,
            called or not.
        """
        for decorator in definition.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            name = target.attr if isinstance(target, ast.Attribute) else target
            if isinstance(name, ast.Name):
                name = name.id
            if name == dataclasses.dataclass.__name__:
                return True
        return False
