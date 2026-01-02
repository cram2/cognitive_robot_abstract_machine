import inspect
import sys
from abc import abstractmethod, ABC
from dataclasses import dataclass, Field
from enum import Enum
from functools import lru_cache, cached_property
from uuid import UUID

from typing_extensions import List, Type, Generic, TYPE_CHECKING, Optional
from typing_extensions import TypeVar, get_origin, get_args


def classes_of_module(module) -> List[Type]:
    """
    Get all classes of a given module.

    :param module: The module to inspect.
    :return: All classes of the given module.
    """

    result = []
    for name, obj in inspect.getmembers(sys.modules[module.__name__]):
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            result.append(obj)
    return result


def behaves_like_a_built_in_class(
    clazz: Type,
) -> bool:
    return (
        is_builtin_class(clazz)
        or clazz == UUID
        or (inspect.isclass(clazz) and issubclass(clazz, Enum))
    )


def is_builtin_class(clazz: Type) -> bool:
    return clazz.__module__ == "builtins"


T = TypeVar("T")


@dataclass
class Role(Generic[T], ABC):
    """
    Represents a role with generic typing. This is used in Role Design Pattern in OOP.

    This class serves as a container for defining roles with associated generic
    types, enabling flexibility and type safety when modeling role-specific
    behavior and data.
    """

    @classmethod
    @lru_cache(maxsize=None)
    def get_role_taker_type(cls) -> Type[T]:
        """
        :return: The type of the role taker.
        """
        return get_generic_type_param(cls, Role)[0]

    @classmethod
    @abstractmethod
    def role_taker_field(cls) -> Field:
        """
        :return: the field that holds the role taker instance.
        """
        ...

    @cached_property
    def role_taker(self) -> T:
        """
        :return: The role taker instance.
        """
        return getattr(self, self.role_taker_field().name)

    def __getattr__(self, item):
        """
        Get an attribute from the role taker when not found on the class.

        :param item: The attribute name to retrieve.
        :return: The attribute value if found in the role taker, otherwise raises AttributeError.
        """
        if hasattr(self.role_taker, item):
            return getattr(self.role_taker, item)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )

    def __setattr__(self, key, value):
        """
        Set an attribute on the role taker instance if the role taker has this attribute,
         otherwise set on this instance directly.
        """
        if key != self.role_taker_field().name and hasattr(self.role_taker, key):
            setattr(self.role_taker, key, value)
        else:
            super().__setattr__(key, value)

    def __hash__(self):
        return hash((self.__class__, hash(self.role_taker)))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return hash(self) == hash(other)


def get_generic_type_param(cls, generic_base: Type[T]) -> Optional[List[Type[T]]]:
    """
    Given a subclass and its generic base, return the concrete type parameter(s).

    Example:
        get_generic_type_param(Employee, Role) -> (<class '__main__.Person'>,)
    """
    for base in getattr(cls, "__orig_bases__", []):
        base_origin = get_origin(base)
        if base_origin is None:
            continue
        if issubclass(get_origin(base), generic_base):
            args = get_args(base)
            return list(args) if args else None
    return None
