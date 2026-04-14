from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, MISSING
from functools import lru_cache, wraps
from inspect import isclass
from typing import Union, Type, Any

from typing_extensions import TypeVar, Type, List, Optional, Callable

T = TypeVar("T")


def recursive_subclasses(cls: Type[T]) -> List[Type[T]]:
    """
    :param cls: The class.
    :return: A list of the classes subclasses without the class itself.
    """
    return cls.__subclasses__() + [
        g for s in cls.__subclasses__() for g in recursive_subclasses(s)
    ]


@dataclass
class DataclassException(Exception):
    """
    A base exception class for dataclass-based exceptions.
    The way this is used is by inheriting from it and setting the `message` field in the __post_init__ method,
    then calling the super().__post_init__() method.
    """

    message: str = field(kw_only=True, default=None)

    def __post_init__(self):
        super().__init__(self.message)


def get_full_class_name(cls):
    """
    Returns the full name of a class, including the module name.

    :param cls: The class.
    :return: The full name of the class
    """
    return cls.__module__ + "." + cls.__name__


@lru_cache
def inheritance_path_length(child_class: Type, parent_class: Type) -> Optional[int]:
    """
    Calculate the inheritance path length between two classes.
    Every inheritance level that lies between `child_class` and `parent_class` increases the length by one.
    In case of multiple inheritance, the path length is calculated for each branch and the minimum is returned.

    :param child_class: The child class.
    :param parent_class: The parent class.
    :return: The minimum path length between `child_class` and `parent_class` or None if no path exists.
    """
    if not (
        isclass(child_class)
        and isclass(parent_class)
        and issubclass(child_class, parent_class)
    ):
        return None

    return _inheritance_path_length(child_class, parent_class, 0)


def _inheritance_path_length(
    child_class: Type, parent_class: Type, current_length: int = 0
) -> int:
    """
    Helper function for :func:`inheritance_path_length`.

    :param child_class: The child class.
    :param parent_class: The parent class.
    :param current_length: The current length of the inheritance path.
    :return: The minimum path length between `child_class` and `parent_class`.
    """

    if child_class == parent_class:
        return current_length
    else:
        return min(
            _inheritance_path_length(base, parent_class, current_length + 1)
            for base in child_class.__bases__
            if issubclass(base, parent_class)
        )


def module_and_class_name(t: Union[Type, _SpecialForm]) -> str:
    return f"{t.__module__}.{t.__name__}"


def get_default_value(dataclass_type, field_name):
    """
    Return the default value for a given field in a dataclass.

    :param dataclass_type: The dataclass type to get the default value for.
    :param field_name: The name of the field to get the default value for.

    :return: The default value for the field.
    """
    for f in fields(dataclass_type):
        if f.name != field_name:
            continue
        if f.default is not MISSING:
            return f.default
        elif f.default_factory is not MISSING:  # handles mutable defaults
            return f.default_factory()
        else:
            raise KeyError(f"No default value for field '{field_name}'")
    return None


def get_default_values_for_dataclass(dataclass_type):
    """
    Return a dict mapping field names to their default values.
    Only includes fields that actually define a default.

    :param dataclass_type: The dataclass type to get the default values for.

    :return: A dict mapping field names to their default values.
    """
    defaults = {}

    for f in fields(dataclass_type):
        if f.default is not MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not MISSING:
            defaults[f.name] = f.default_factory()

    return defaults


T = TypeVar("T", bound=Callable[..., Any])


def memoize(function: T) -> T:
    """
    Caches the return value of a function call at the instance level.
    """

    @wraps(function)
    def wrapper(self, *args: Any, **kwargs: Any) -> T:
        if not hasattr(self, "__memo__"):
            self.__memo__ = {}
        memo = self.__memo__

        key = (function, self, args, frozenset(kwargs.items()))
        try:
            return memo[key]
        except KeyError:
            rv = function(self, *args, **kwargs)
            memo[key] = rv
            return rv

    return wrapper  # type: ignore


def copy_memoize(function: T) -> T:
    """
    Caches the return value of a function call at the instance level but returns a deepcopy of the value.
    """

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "__memo__"):
            self.__memo__ = {}
        memo = self.__memo__

        key = (function, self, args, frozenset(kwargs.items()))
        try:
            return deepcopy(memo[key])
        except KeyError:
            rv = function(self, *args, **kwargs)
            memo[key] = rv
            return deepcopy(rv)

    return wrapper


def clear_memoization_cache(instance):
    """
    Clears the memoization cache of an instance.
    """
    if hasattr(instance, "__memo__"):
        instance.__memo__.clear()
