from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, fields
from functools import lru_cache, cached_property
from typing import List, TypeVar, ClassVar

from typing_extensions import Type, get_origin, Any, Dict, Set

from krrood.class_diagrams.utils import T
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import get_generic_type_param


@dataclass
class EntityAndType(SubClassSafeGeneric[T]):
    entity: T
    type: Type[T] = field(init=False)

    def __post_init__(self):
        self.type = type(self.entity)

    def __hash__(self):
        return hash((self.entity, self.type))

    def __eq__(self, other):
        return hash(self) == hash(other)


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
    _role_taker_roles: ClassVar[Dict[Any, List[Role]]] = defaultdict(list)
    _role_role_takers: ClassVar[Dict[EntityAndType, Set[EntityAndType]]] = defaultdict(set)

    @property
    def role_taker_roles(self) -> List[Role]:
        """
        :return: All roles of the role taker instance.
        """
        return self._role_taker_roles[self.role_taker]

    @property
    def all_role_takers(self) -> Set[Any]:
        """
        :return: All role takers of the role instance.
        """
        return self._role_role_takers[EntityAndType(self)]

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
    def get_role_generic_type(cls) -> Type[T] | TypeVar:
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
        """
        :return: True if this role inherits from another role and updates its role-taker type, False otherwise.
        """
        if Role in cls.__bases__:
            return False
        role_taker_type = cls.get_role_taker_type()
        for parent in cls.__bases__:
            if not issubclass(parent, Role):
                continue
            parent_origin_type = get_origin(parent) or parent
            parent_role_taker_type = parent_origin_type.get_role_taker_type()
            if parent_role_taker_type is not role_taker_type:
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

    @property
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
        Get an attribute from the role taker when not found on the role itself, otherwise raise AttributeError.

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
        self._bootstrap_inner_attributes()

        if key == self.role_taker_attribute_name():
            self._set_role_taker(value)
        elif self._role_taker_field_set:
            setattr(self.role_taker, key, value)
            # Ensure the attribute is also set on this instance if it's a field
            # of the Role itself (and not just intended for delegation).
            # This is important for dataclasses to work correctly.
            if key in [f.name for f in fields(self)]:
                super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)
            if key not in Role.__dict__:
                self._to_set_in_role_taker[key] = value

    def _set_role_taker(self, value: T):
        """
        Handle setting attributes when the role taker is set.
        Ensure that attributes intended for delegation are correctly set on the role taker.
        """
        object.__setattr__(self, "_role_taker_field_set", True)
        # Also set the actual attribute defined in the dataclass
        super().__setattr__(self.role_taker_attribute_name(), value)

        # Set the attributes that were set before the role taker was set
        for attribute_name, attribute_value in self._to_set_in_role_taker.items():
            setattr(value, attribute_name, attribute_value)
        self._to_set_in_role_taker.clear()
        self._update_mapping_between_roles_and_role_takers(value)

    def _update_mapping_between_roles_and_role_takers(self, role_taker: T):
        """
        Update the mapping between roles and role takers.
        Ensures that the role taker and its role are correctly linked in the mapping.

        :param role_taker: The role taker instance to update the mapping for.
        """
        Role._role_taker_roles[role_taker].append(self)
        Role._role_role_takers[EntityAndType(self)].add(EntityAndType(role_taker))
        if isinstance(role_taker, Role):
            Role._role_role_takers[EntityAndType(self)].update(Role._role_role_takers[EntityAndType(role_taker)])

    def _bootstrap_inner_attributes(self):
        """
        Initialize internal attributes with default values if they don't exist.
        """
        for bootstrap_attr, default in [
            ("_to_set_in_role_taker", {}),
            ("_role_taker_field_set", False),
        ]:
            try:
                object.__getattribute__(self, bootstrap_attr)
            except AttributeError:
                object.__setattr__(self, bootstrap_attr, default)

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
