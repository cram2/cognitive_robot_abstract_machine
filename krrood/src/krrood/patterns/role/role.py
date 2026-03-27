from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field, fields, Field, is_dataclass
from functools import lru_cache, cached_property

from typing_extensions import Type, get_origin, Any, Dict, List, TypeVar, Iterator

from krrood.class_diagrams.utils import T
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.symbol_graph.symbol_graph import Symbol, PredicateClassRelation, SymbolGraph
from krrood.utils import get_generic_type_param


@dataclass
class Role(SubClassSafeGeneric[T], Symbol, ABC):
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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # redefine fields of role taker to be init=False, these are fields that are inherited
        # from bases that the role taker also inherits from
        for base in cls.__bases__:
            if not issubclass(cls.get_role_taker_type(), base):
                continue
            if not is_dataclass(base):
                continue
            for field_ in fields(base):
                if issubclass(base, Role) and field_.name in ["_role_taker_field_set", "_to_set_in_role_taker", "_conflicting_fields_with_role_taker"]:
                    continue
                cls._update_field_kwargs(field_.name, {"init": False}, type_=field_.type)
                setattr(cls, field_.name, delegate_property(field_.name, cls.role_taker_attribute_name()))

    @classmethod
    def has_role(cls, role_taker: T, role_type: Type[Role]) -> bool:
        """
        :return: Whether the role taker has the given role type.
        """
        return any(cls.yield_taker_roles_of_type(role_taker, role_type))

    @property
    def role_taker_roles(self) -> List[Role]:
        """
        :return: All roles of the role taker instance.
        """
        return self.get_taker_roles_of_type(self.role_taker, Role)

    @classmethod
    def get_taker_roles_of_type(cls, role_taker: T, role_type: Type[Role[T]]) -> List[Role[T]]:
        """
        :return: All roles of the given type for the role taker instance.
        """
        return list(cls.yield_taker_roles_of_type(role_taker, role_type))

    @classmethod
    def yield_taker_roles_of_type(cls, role_taker: T, role_type: Type[Role[T]]) -> Iterator[Role[T]]:
        """
        :return: All roles of the given type for the role taker instance.
        """
        wrapped_taker = SymbolGraph().get_wrapped_instance(role_taker)
        yield from (relation.source.instance for relation in
                    SymbolGraph().get_incoming_relations_with_type(wrapped_taker, HasRoleTaker) if
                    isinstance(relation.source.instance, role_type))

    @property
    def all_role_takers(self) -> List[Any]:
        """
        :return: All role takers of the role instance.
        """
        return list(self.yield_takers_of_role(self))

    @classmethod
    def yield_takers_of_role(cls, role: Role) -> Iterator[Any]:
        """
        :return: All role takers of the given role.
        """
        wrapped_role = SymbolGraph().get_wrapped_instance(role)
        yield from (relation.target.instance for relation in
                    SymbolGraph().get_outgoing_relations_with_type(wrapped_role, HasRoleTaker))

    @classmethod
    @lru_cache
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
    def get_role_taker_type(cls) -> Type[T]:
        """
        :return: The type of the role taker.
        """
        try:
            type_ = next(
                f.type for f in fields(cls) if f.name == cls.role_taker_attribute_name()
            )
        except StopIteration:
            # get it by extracting the generic parameter
            type_ = get_generic_type_param(cls, Role)[0]
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
        wrapped_self = SymbolGraph().get_wrapped_instance(self)
        wrapped_role_taker = SymbolGraph().ensure_wrapped_instance(role_taker)
        SymbolGraph().add_relation(
            HasRoleTaker(wrapped_self, wrapped_role_taker,
                         self.role_taker_wrapped_field))
        if isinstance(role_taker, Role):
            for relation in SymbolGraph().get_outgoing_relations_with_type(wrapped_role_taker, HasRoleTaker):
                SymbolGraph().add_relation(HasRoleTaker(wrapped_self, relation.target, relation.wrapped_field))

    @cached_property
    def role_taker_wrapped_field(self) -> WrappedField:
        """
        :return: The wrapped field of this class that is pointing to the role taker.
        """
        return next(wf for wf in SymbolGraph().class_diagram.get_wrapped_class(self.__class__).fields if
                    wf.name == self.role_taker_attribute_name())

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


class HasRoleTaker(PredicateClassRelation[Role]):
    ...


def delegate_property(name, role_taker):
    """
    Creates a property that delegates to another attribute's attribute.
    """
    def getter(self):
        target = getattr(self, role_taker)
        return getattr(target, name)

    def setter(self, value):
        try:
            target = getattr(self, role_taker)
        except AttributeError as e:
            # the role taker is not set yet
            return
        setattr(target, name, value)

    return property(getter, setter)