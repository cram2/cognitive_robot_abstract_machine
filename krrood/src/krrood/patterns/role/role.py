import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from functools import lru_cache, cached_property
from typing import List, TypeVar, ClassVar

from typing_extensions import Type, get_origin, Any, Dict, Iterable

from krrood.class_diagrams.utils import T, get_type_hints_of_object
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import get_generic_type_param


@dataclass
class Role(SubClassSafeGeneric[T], ABC):
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
    _to_set_in_role_taker: Dict[str, Any] = field(default_factory=dict, init=False)

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
        if cls is Role:
            return T
        res = get_generic_type_param(cls, Role)
        return res[0] if res else T

    @classmethod
    @lru_cache
    def get_role_taker_type(cls) -> Type[T]:
        """
        :return: The type of the role taker.
        """
        type_ = next(
            f.type for f in fields(cls) if f.name == cls.role_taker_attribute_name()
        )
        if isinstance(type_, str):
            type_ = sys.modules[cls.__module__].__dict__[type_]
        if isinstance(type_, TypeVar):
            if type_.__bound__ is not None:
                type_ = type_.__bound__
            else:
                raise ValueError(f"TypeVar {type_} has no bound")
        return type_

    @classmethod
    @lru_cache
    def updates_role_taker_type(cls) -> bool:
        if Role in cls.__bases__:
            return False
        my_rt = cls.get_role_taker_type()
        my_rt_name = getattr(my_rt, "__name__", str(my_rt))
        for parent in cls.__bases__:
            if issubclass(parent, Role) and parent is not Role:
                p_origin = get_origin(parent) or parent
                p_rt = p_origin.get_role_taker_type()
                p_rt_name = getattr(p_rt, "__name__", str(p_rt))
                if p_rt_name != my_rt_name:
                    return True
        return False

    @classmethod
    @abstractmethod
    def role_taker_attribute(cls) -> Attribute:
        """
        :return: The symbolic representation of the attribute that holds the role taker instance.
        """
        ...

    @classmethod
    def role_taker_attribute_name(cls) -> str:
        """
        :return: The name of the attribute that holds the role taker instance.
        """
        return cls.role_taker_attribute()._attribute_name_

    @cached_property
    def role_taker(self) -> T:
        """
        Retrieves the role taker instance.

        Uses object.__getattribute__ to avoid triggering __getattr__ recursion.
        """
        attr_name = self.role_taker_attribute_name()
        try:
            return object.__getattribute__(self, attr_name)
        except AttributeError:
            raise AttributeError(f"Role taker attribute '{attr_name}' not found.")

    @cached_property
    def root_persistent_entity(self):
        """
        :return: The root persistent entity in the role hierarchy.
        """
        curr = self
        while isinstance(curr, Role):
            rt = getattr(curr, self.role_taker_attribute_name())
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
        # Avoid recursion when looking up the role taker attribute itself
        if item == self.role_taker_attribute_name():
            raise AttributeError(item)

        if self._role_taker_field_set:
            rt = self.role_taker
            return getattr(rt, item)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )

    def __setattr__(self, key, value):
        """
        Set an attribute on the role taker instance if the role taker has this attribute,
         otherwise set on this instance directly.
        """
        role_taker_attr = self.role_taker_attribute_name()

        if key == role_taker_attr:
            object.__setattr__(self, "_role_taker_field_set", True)
            # Also set the actual attribute defined in the dataclass
            super().__setattr__(key, value)

            for attribute_name, attribute_value in self._to_set_in_role_taker.items():
                setattr(value, attribute_name, attribute_value)
            self._to_set_in_role_taker.clear()
        elif self._role_taker_field_set:
            setattr(self.role_taker, key, value)
            # Ensure the attribute is also set on this instance if it's a field
            # of the Role itself (and not just intended for delegation).
            # This is important for dataclasses to work correctly.
            if key in [f.name for f in fields(self)]:
                super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)
            if key not in ["_to_set_in_role_taker"]:
                self._to_set_in_role_taker[key] = value

    def __hash__(self):
        """
        A persistent entity and its roles should be considered the same entity, so we hash based on the root persistent
         entity.
        """
        return hash(self.root_persistent_entity)

    def __eq__(self, other):
        return hash(self) == hash(other)


@dataclass(eq=False)
class RoleTaker(ABC):
    roles: ClassVar[Dict[Type[Role], List[Role]]] = {}

    def as_role(self, role_type: Type[T]) -> List[Role[T]]:
        return self.roles.get(role_type, [])
