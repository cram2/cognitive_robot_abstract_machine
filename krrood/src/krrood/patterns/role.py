from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, Field, fields, MISSING, field
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

    Roles are extensions of the role taker's behavior and data in different contexts. Roles live side-by-side with the
     role taker, which means that roles never overwrite the role taker's behavior or data but instead only extend it.
    ..warning:: Never overwrite the role taker's behavior or data in a role.

    Setting an attribute on a role will set it on the role taker instance recursively, but the opposite is not true.
    All attributes of all roles (except for the attributes that point to the role_taker inside the role) and the role
    taker are accessible from any role or role taker instance.

    Roles and role takers are considered the same entity, having the same hash value and are equal.
    >>> student = Student(person=person)
    >>> person == student
    True
    >>> hash(person) == hash(student)
    True

    The Role Pattern is meant for easy semantic access to attributes without having to abide to memory layout. For example
    no need to do `PrivateSchoolStudent.Student.Person.age`, instead you can do `PrivateSchoolStudent.age`. Or even the
    other way around when wanting to access student context attributes from person instance like:
    >>> courses = private_school_student = next(psc for psc in PrivateSchoolStudent if psc.name == person.name).courses
    >>> # Instead Do:
    >>> courses = person.courses if hasattr(person, "courses") else None
    Thus not only allowing easy semantic access but also reducing the number of joins (searches), which further improves
     the performance.
    ..warning:: Always check if the attribute exists before accessing it if it is an attribute that is introduced by a
    Role, because if no Role instance exists, the attribute will not be accessible.
    """

    _role_taker_field_set: bool = field(default=False, init=False)

    @classmethod
    @lru_cache(maxsize=None)
    def get_root_role_taker_type(cls) -> Type[T]:
        """
        :return: The type of the role taker.
        """
        current_cls = cls
        while issubclass(current_cls, Role):
            current_cls = current_cls.get_role_taker_type()
        return current_cls

    @classmethod
    @lru_cache
    def get_role_generic_type(cls) -> Type[T]:
        """
        :return: The type of the role taker.
        """
        return get_generic_type_param(cls, Role)[0]

    @classmethod
    @lru_cache
    def get_role_taker_type(cls) -> Type[T]:
        """
        :return: The type of the role taker.
        """
        from ..symbol_graph.helpers import get_field_type_endpoint

        return get_field_type_endpoint(cls, cls.role_taker_field().name)

    @classmethod
    @lru_cache
    def updates_role_taker_type(cls) -> bool:
        return (cls.get_role_taker_type() is not cls.get_role_generic_type()) and any(
            parent.get_root_role_taker_type() is not cls.get_role_taker_type()
            for parent in cls.__bases__
            if issubclass(parent, Role)
        )

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
        """
        :return: The root persistent entity in the role hierarchy.
        """
        curr = self
        while isinstance(curr, Role):
            rt = getattr(curr, "_direct_role_taker", None)
            if rt is not None:
                curr = rt
            else:
                curr = curr.role_taker
        return curr

    def __getattr__(self, item):
        """
        Get an attribute from the role taker when not found on the class.

        :param item: The attribute name to retrieve.
        :return: The attribute value if found in the role taker, otherwise raises AttributeError.
        """
        if self._role_taker_field_set and hasattr(self.role_taker, item):
            return getattr(self.role_taker, item)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )

    def __setattr__(self, key, value):
        """
        Set an attribute on the role taker instance if the role taker has this attribute,
         otherwise set on this instance directly.
        """
        if key == self.role_taker_field().name:
            self._role_taker_field_set = True
            object.__setattr__(self, "_direct_role_taker", value)
        if key != self.role_taker_field().name and self._role_taker_field_set:
            setattr(self.role_taker, key, value)
        if key == self.role_taker_field().name or hasattr(self, key):
            super().__setattr__(key, value)

    def __hash__(self):
        """
        A persistent entity and its roles should be considered the same entity, so we hash based on the root persistent
         entity.
        """
        return hash(self.root_persistent_entity)

    def __eq__(self, other):
        return hash(self) == hash(other)


def role_enabled_dataclass(cls):
    cls = dataclass(cls)

    print("Fields:", [f.name for f in fields(cls)])

    return cls
