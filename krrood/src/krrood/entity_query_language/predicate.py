"""
Predicates and symbolic function utilities for the Entity Query Language.

This module defines predicate classes for boolean checks and a decorator to build symbolic expressions
from regular Python functions when variables are present.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps

from typing_extensions import (
    Callable,
    Optional,
    Any,
    Type,
    Tuple,
    ClassVar,
    Sized,
    Dict,
)

from .utils import T, merge_args_and_kwargs
from .core.variable import Variable, InstantiatedVariable
from .core.base_expressions import Selectable
from ..symbol_graph.symbol_graph import Symbol


def symbolic_function(
    function: Callable[..., T],
) -> Callable[..., Variable[T]]:
    """
    Function decorator that constructs a symbolic expression representing the function call
     when inside a symbolic_rule context.

    When symbolic mode is active, calling the method returns a Call instance which is a SymbolicExpression bound to
    representing the method call that is not evaluated until the evaluate() method is called on the query/rule.

    :param function: The function to decorate.
    :return: The decorated function.
    """

    @wraps(function)
    def wrapper(*args, **kwargs) -> Optional[Any]:
        all_kwargs = merge_args_and_kwargs(function, args, kwargs)
        if _any_of_the_kwargs_is_a_variable(all_kwargs):
            return InstantiatedVariable(
                _type_=function,
                _kwargs_=all_kwargs,
            )
        return function(*args, **kwargs)

    return wrapper


@dataclass(eq=False)
class Predicate(Symbol, ABC):
    """
    The super predicate class that represents a filtration operation or asserts a relation.
    """

    _cache_instances_: ClassVar[bool] = False
    """
    Predicates should not be cached for now as they are not persisting.
    """

    def __new__(cls, *args, **kwargs):
        all_kwargs = merge_args_and_kwargs(
            cls.__init__, args, kwargs, ignore_first=True
        )
        if _any_of_the_kwargs_is_a_variable(all_kwargs):
            return InstantiatedVariable(
                _type_=cls,
                _kwargs_=all_kwargs,
            )
        return super().__new__(cls)

    @abstractmethod
    def __call__(self) -> bool:
        """
        Evaluate the predicate for the supplied values.
        """

    def __bool__(self):
        """
        Bool casting a predicate evaluates it.
        """
        return bool(self.__call__())


@dataclass(eq=False)
class HasType(Predicate):
    """
    Represents a predicate to check if a given variable is an instance of a specified type.

    This class is used to evaluate whether the domain value belongs to a given type by leveraging
    Python's built-in `isinstance` functionality. It provides methods to retrieve the domain and
    range values and perform direct checks.
    """

    variable: Any
    """
    The variable whose type is being checked.
    """
    types_: Type
    """
    The type or tuple of types against which the `variable` is validated.
    """

    def __call__(self) -> bool:
        return isinstance(self.variable, self.types_)


@dataclass(eq=False)
class HasTypes(HasType):
    """
    Represents a specialized data structure holding multiple types.

    This class is a data container designed to store and manage a tuple of
    types. It inherits from the `HasType` class and extends its functionality
    to handle multiple types efficiently. The primary goal of this class is to
    allow structured representation and access to a collection of type
    information with equality comparison explicitly disabled.
    """

    types_: Tuple[Type, ...]
    """
    A tuple containing Type objects that are associated with this instance.
    """


@symbolic_function
def length(iterable: Sized) -> int:
    """
    :param iterable: The iterable.
    :return: The length of the iterable.
    """
    return len(iterable)


def _any_of_the_kwargs_is_a_variable(bindings: Dict[str, Any]) -> bool:
    """
    :param bindings: A kwarg like dict mapping strings to objects
    :return: Rather any of the objects is a variable or not.
    """
    return any(isinstance(binding, Selectable) for binding in bindings.values())
