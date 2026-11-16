"""
A module whose dataclasses and methods take their fields and parameters positionally,
next to the definitions the keyword-only rules accept or exempt.
"""

from __future__ import annotations

import ast
import dataclasses
import logging
from dataclasses import dataclass
from enum import Enum

from typing_extensions import Callable

from krrood.patterns import called_positionally as marker_module
from krrood.patterns.called_positionally import called_positionally

# %% dataclasses


@dataclass
class PositionalFields:
    """
    A dataclass whose fields can be passed positionally.
    """

    value: int
    """
    A field.
    """


@dataclasses.dataclass(frozen=True)
class PositionalFrozenFields:
    """
    A frozen dataclass, declared through the module, whose fields can be passed
    positionally.
    """

    value: int
    """
    A field.
    """


@dataclass(kw_only=True)
class KeywordOnlyFields:
    """
    A dataclass whose fields are keyword-only.
    """

    value: int
    """
    A field.
    """


@dataclass
class EnumMemberValue:
    """
    A dataclass an enum mixes in, which receives the member values positionally.
    """

    label: str
    """
    A field.
    """


class Colour(EnumMemberValue, Enum):
    """
    An enum whose members are :class:`EnumMemberValue` instances.
    """

    RED = ("red",)


# %% methods


class Methods:
    """
    A class with methods taking parameters positionally and keyword-only.
    """

    def positional(self, value: int) -> int:
        """
        A method taking a parameter positionally.
        """
        return value

    @staticmethod
    def positional_static(value: int) -> int:
        """
        A static method taking a parameter positionally.
        """
        return value

    def keyword_only(self, *, value: int) -> int:
        """
        A method taking a keyword-only parameter.
        """
        return value

    @classmethod
    def keyword_only_class_method(cls, *, value: int) -> int:
        """
        A class method taking a keyword-only parameter.
        """
        return value

    def without_parameters(self) -> None:
        """
        A method taking nothing but the instance.
        """

    def __eq__(self, other: object) -> bool:
        return other is self

    @property
    def stored(self) -> int:
        """
        A property.
        """
        return 0

    @stored.setter
    def stored(self, value: int) -> None:
        pass


class DerivedMethods(Methods):
    """
    A class overriding a method of a base class from the same repository.
    """

    def positional(self, value: int) -> int:
        """
        An override of a method the repository defines, taking a parameter positionally.
        """
        return value


class LibraryHandler(logging.Handler):
    """
    A class overriding a method of a standard library base class.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        An override the standard library calls with a positional argument.
        """

    def positional(self, value: int) -> int:
        """
        A method no base class defines, taking a parameter positionally.
        """
        return value


class NodeNames(ast.NodeVisitor):
    """
    A class whose visiting method the standard library base class dispatches to by name.
    """

    def visit_Name(self, node: ast.Name) -> None:
        """
        A method the base class calls with a positional argument.
        """


class PositionalDecorator:
    """
    A class whose methods are applied as decorators and marked as called positionally.
    """

    @called_positionally
    def decorate(self, function: Callable) -> Callable:
        """
        A method Python calls with the decorated function.
        """
        return function

    @marker_module.called_positionally
    def decorate_through_module(self, function: Callable) -> Callable:
        """
        A method marked through the marker's module.
        """
        return function
