from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps, lru_cache

from typing_extensions import (
    Callable,
    Optional,
    Any,
    Type,
    Tuple,
    ClassVar,
    Dict,
    List,
)

from .enums import PredicateType
from .symbol_graph import (
    WrappedInstance,
    SymbolGraph,
)
from .symbolic import (
    SymbolicExpression,
    Variable,
    _any_of_the_kwargs_is_a_variable,
)
from .utils import T


def symbolic_function(
    function: Callable[..., T],
) -> Callable[..., SymbolicExpression[T]]:
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
            return Variable(
                _name__=function.__name__,
                _type_=function,
                _kwargs_=all_kwargs,
                _predicate_type_=PredicateType.DecoratedMethod,
            )
        return function(*args, **kwargs)

    return wrapper


@dataclass(eq=False)
class Symbol:
    """Base class for things that can be cached in the symbol graph."""

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        update_cache(instance)
        return instance


@dataclass(eq=False)
class Predicate(Symbol, ABC):
    """
    The super predicate class that represents a filtration operation or asserts a relation.
    """

    is_expensive: ClassVar[bool] = False

    def __new__(cls, *args, **kwargs):
        all_kwargs = merge_args_and_kwargs(
            cls.__init__, args, kwargs, ignore_first=True
        )
        if _any_of_the_kwargs_is_a_variable(all_kwargs):
            return Variable(
                _type_=cls,
                _name__=cls.__name__,
                _kwargs_=all_kwargs,
                _predicate_type_=PredicateType.SubClassOfPredicate,
            )
        return super().__new__(cls)

    @abstractmethod
    def __call__(self) -> bool:
        """
        Evaluate the predicate for the supplied values.
        """

    def __bool__(self):
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


def update_cache(instance: Symbol):
    """
    Updates the cache with the given instance of a symbolic type.

    :param instance: The symbolic instance to be cached.
    """
    if not isinstance(instance, Predicate):
        SymbolGraph().add_node(WrappedInstance(instance))


@lru_cache
def get_function_argument_names(function: Callable) -> List[str]:
    """
    :param function: A function to inspect
    :return: The argument names of the function
    """
    return list(inspect.signature(function).parameters.keys())


def merge_args_and_kwargs(
    function: Callable, args, kwargs, ignore_first: bool = False
) -> Dict[str, Any]:
    """
    Merge the arguments and keyword-arguments of a function into a dict of keyword-arguments.

    :param function: The function to get the argument names from
    :param args: The arguments passed to the function
    :param kwargs: The keyword arguments passed to the function
    :param ignore_first: Rather to ignore the first argument or not.
    Use this when `function` contains something like `self`
    :return: The dict of assigned keyword-arguments.
    """
    starting_index = 1 if ignore_first else 0
    all_kwargs = {
        name: arg
        for name, arg in zip(
            get_function_argument_names(function)[starting_index:], args
        )
    }
    all_kwargs.update(kwargs)
    return all_kwargs
