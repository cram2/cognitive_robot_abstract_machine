from abc import ABC, abstractmethod
from dataclasses import dataclass, Field
from functools import lru_cache, cached_property
from typing_extensions import Generic, Type

from krrood.class_diagrams.utils import T, get_generic_type_param


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

    @cached_property
    def root_persistent_entity(self):
        root = self
        while isinstance(root, Role):
            root = root.role_taker
        return root

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
        super().__setattr__(key, value)
        if key != self.role_taker_field().name:
            setattr(self.role_taker, key, value)

    def __hash__(self):
        return hash(self.root_persistent_entity)

    def __eq__(self, other):
        return hash(self) == hash(other)
