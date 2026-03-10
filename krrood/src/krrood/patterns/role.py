from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, Field, fields, MISSING
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
    >>> private_school_student = next(psc for psc in PrivateSchoolStudent if psc.name == person.name).courses
    >>> # Instead Do:
    >>> person.courses
    Thus not only allowing easy semantic access but also reducing the number of joins (searches), which further improves
     the performance.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        role_specific_attribute_names = [
            attr_name
            for attr_name in dir(cls)
            if all(
                attr_name not in dir(base)
                for base in cls.__bases__ + (cls.get_role_taker_type(),)
            )
        ]
        for attribute_name in role_specific_attribute_names:
            attribute_value = getattr(cls, attribute_name)
            if isinstance(attribute_value, Field):
                if attribute_value.default is not MISSING:
                    attribute_value = attribute_value.default
                else:
                    continue
            setattr(cls.get_root_role_taker_type(), attribute_name, attribute_value)

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
        """
        :return: The root persistent entity in the role hierarchy.
        """
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
