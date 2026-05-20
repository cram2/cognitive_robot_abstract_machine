from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass, MISSING, Field
from functools import lru_cache, cached_property

from typing_extensions import (
    Type,
    Any,
    Dict,
    List,
    TypeVar,
    Iterator,
    Tuple,
)

from krrood.class_diagrams.exceptions import ClassIsUnMappedInClassDiagram
from krrood.class_diagrams.utils import (
    T,
    all_nearest_common_ancestors,
)
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.symbol_graph.symbol_graph import Symbol, PredicateClassRelation, SymbolGraph
from krrood.utils import get_generic_type_param


@dataclass
class Role(Symbol, SubClassSafeGeneric[T], ABC):
    """
    Represents a role with generic typing. This is used in Role Design Pattern in OOP.

    Roles are extensions of the role taker's behaviour and data in different contexts.
    Roles live side-by-side with the role taker: they never overwrite the role taker's
    data or behaviour, only extend it.

    Role-native attributes are accessed directly from the role instance.  Attributes that
    belong to the role taker are exposed on the role through dynamic delegation by overriding
    the ``__getattr__`` and ``__setattr__`` methods.

    Roles and role takers are considered the same entity (same hash, equal):
    >>> student = Student(person=person)
    >>> person == student
    True
    >>> hash(person) == hash(student)
    True
    """

    _role_taker_field_set: bool = field(default=False, init=False)
    _to_set_in_role_taker: Dict[str, Any] = field(default_factory=dict, init=False)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def new_init(self, *args, **init_kwargs):
            """
            Initializes the role instance with provided keyword arguments. It first sets the role taker attribute using
            the value from kwargs, then sets any additional attributes on the role instance. If any fields defined in
             the role class are not provided in kwargs and are not present on the role taker, it attempts to set them
              to their default values if available, otherwise raises a ValueError.
            """
            role_taker_name = self.role_taker_attribute_name()

            if role_taker_name not in init_kwargs:
                self._initialize_role_taker_and_self(*args, **init_kwargs)
                return

            setattr(self, role_taker_name, init_kwargs.pop(role_taker_name))
            for k, v in init_kwargs.items():
                setattr(self, k, v)
            self._set_default_fields_not_in_kwargs(**init_kwargs)

        cls._handle_conflicting_fields_of_role_with_role_taker()
        cls.__init__ = new_init

    def _initialize_role_taker_and_self(self, *args, **init_kwargs):
        """
        Initializes the role taker instance and sets attributes for the role instance.
        Assumes that you want to create a new role taker instance as well, if the role taker is not provided
        in the kwargs.

        :param args: Positional arguments passed to the role instance.
        :param init_kwargs: Keyword arguments passed to the role instance.
        """
        role_taker_init_kwargs = {
            k: v for k, v in init_kwargs.items() if k in self.role_taker_field_names
        }
        setattr(
            self,
            self.role_taker_attribute_name(),
            self.get_role_taker_type()(*args, **role_taker_init_kwargs),
        )
        for k, v in init_kwargs.items():
            if k not in self.role_taker_field_names:
                setattr(self, k, v)

    @classmethod
    def _handle_conflicting_fields_of_role_with_role_taker(cls):
        """Make fields from common bases (shared between the role class and its role-taker
        type) init=False so the dataclass constructor does not require them.
        """
        for common_base in all_nearest_common_ancestors(
            (cls.get_role_taker_type(), cls)
        ):
            if common_base in [ABC, object, Role] or not is_dataclass(common_base):
                continue
            for field_ in fields(common_base):
                if cls._is_not_conflicting_field_or_already_resolved(
                    field_, common_base
                ):
                    continue
                cls._make_field_init_false_and_delegate_to_role_taker(
                    field_, common_base
                )

    @classmethod
    def _is_not_conflicting_field_or_already_resolved(
        cls, field_: Field, common_base: Type
    ) -> bool:
        """
        :param field_: The field to check.
        :param common_base: The common base class.

        :return: True if the field is conflicting or already resolved, False otherwise.
        """
        if not field_.init:
            return True
        if field_.name == cls.role_taker_attribute_name():
            return True
        if issubclass(common_base, Role) and field_.name in Role.__annotations__:
            return True
        if hasattr(common_base, field_.name) and isinstance(
            getattr(common_base, field_.name), property
        ):
            # That means this field was already seen before and was assigned a property.
            return True
        return False

    @classmethod
    def _make_field_init_false_and_delegate_to_role_taker(
        cls, field_: Field, field_owner: Type
    ):
        """Make a field init=False and delegate to the role taker.

        :param field_: The field to make init=False and delegate to the role taker.
        :param field_owner: The type that owns the field.
        """
        type_ = field_.type
        if isinstance(field_.type, str):
            try:
                type_ = eval(
                    field_.type,
                    sys.modules[field_owner.__module__].__dict__,
                )
            except NameError:
                pass
        cls._update_field_kwargs(field_.name, {"init": False}, type_=type_)
        setattr(
            cls,
            field_.name,
            delegate_property(field_.name, cls.role_taker_attribute_name()),
        )

    def _set_default_fields_not_in_kwargs(self, **kwargs):
        """
        Set default values for fields that are not provided in kwargs and are not present on the role taker.

        :param kwargs: The keyword arguments passed to the role instance.
        """
        for field_name in self.__annotations__:
            if (
                field_name not in kwargs
                and field_name not in self.all_role_taker_names
                and not hasattr(self.role_taker, field_name)
            ):
                field_ = next(f for f in fields(self) if f.name == field_name)
                if field_.default != MISSING:
                    setattr(self, field_name, field_.default)
                elif field_.default_factory != MISSING:
                    setattr(self, field_name, field_.default_factory())

    @cached_property
    def all_role_taker_names(self) -> List[str]:
        """
        :return: a list of all role taker attribute names in the role hierarchy.
        """
        curr_taker = self.role_taker
        all_takers = [self.role_taker_attribute_name()]
        while isinstance(curr_taker, Role):
            all_takers.append(curr_taker.role_taker_attribute_name())
            curr_taker = getattr(curr_taker, curr_taker.role_taker_attribute_name())
        return all_takers

    @classmethod
    def from_role_taker(cls, role_taker: T) -> Role[T]:
        """
        Factory method to create a role instance for a given role taker.

        :param role_taker: The role taker instance to create the role for.
        :return: An instance of the role for the given role taker.
        """
        return cls(**{cls.role_taker_attribute_name(): role_taker})

    @classmethod
    def has_role(
        cls, role_taker: T, role_types: Type[Role] | Tuple[Type[Role], ...]
    ) -> bool:
        """
        :param role_taker: The role taker instance to query.
        :param role_types: The type or tuple of types of roles to check for.
        :return: Whether the role taker has any of the given role type(s).
        """
        return any(cls.yield_taker_roles_of_type(role_taker, role_types))

    @classmethod
    def roles_for(cls, role_taker: T, role_type: Type[Role] = None) -> List[Role]:
        """
        :param role_taker: The role taker instance to query.
        :param role_type: The type of roles to check for.
        :return: All roles of the given type for the role taker instance.
        """
        role_type = role_type or Role
        return list(cls.yield_taker_roles_of_type(role_taker, role_type))

    @property
    def role_taker_roles(self) -> List[Role]:
        """
        :return: All roles of the role taker instance.
        """
        return self.get_taker_roles_of_type(self.role_taker, Role)

    @classmethod
    def get_taker_roles_of_type(
        cls, role_taker: T, role_type: Type[Role[T]]
    ) -> List[Role[T]]:
        """
        :return: All roles of the given type for the role taker instance.
        """
        return list(cls.yield_taker_roles_of_type(role_taker, role_type))

    @classmethod
    def yield_taker_roles_of_type(
        cls, role_taker: T, role_types: Type[Role[T]] | Tuple[Type[Role[T]], ...]
    ) -> Iterator[Role[T]]:
        """
        :param role_taker: The role taker instance to query.
        :param role_types: The type or tuple of types of roles to yield.
        :return: All roles of the given type(s) for the role taker instance.
        """
        wrapped_taker = SymbolGraph().get_wrapped_instance(role_taker)
        if wrapped_taker is None:
            return
        yield from (
            relation.source.instance
            for relation in SymbolGraph().get_incoming_relations_with_type(
                wrapped_taker, HasRoleTaker
            )
            if isinstance(relation.source.instance, role_types)
        )

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
        if wrapped_role is None:
            return
        yield from (
            relation.target.instance
            for relation in SymbolGraph().get_outgoing_relations_with_type(
                wrapped_role, HasRoleTaker
            )
        )

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

    @classmethod
    def get_role_taker_type(cls) -> Type[T]:
        """Return the type of the role taker field."""
        try:
            type_ = next(
                f.type for f in fields(cls) if f.name == cls.role_taker_attribute_name()
            )
        except StopIteration:
            type_ = get_generic_type_param(cls, Role)[0]
        if isinstance(type_, str):
            try:
                type_ = sys.modules[cls.__module__].__dict__[type_]
            except KeyError:
                type_ = eval(type_, sys.modules[cls.__module__].__dict__)
        if isinstance(type_, TypeVar):
            if type_.__bound__ is not None:
                type_ = type_.__bound__
            else:
                raise ValueError(f"TypeVar {type_} has no bound")
        return type_

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
    def root_persistent_entity(self):
        """
        :return: The root persistent entity in the role hierarchy.
        """
        curr = self
        while isinstance(curr, Role):
            rt = getattr(curr, curr.role_taker_attribute_name())
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
            if key in self.role_taker_field_names:
                setattr(self.role_taker, key, value)
                return
            super().__setattr__(key, value)
        else:
            if key in self.__annotations__ and key not in self.role_taker_field_names:
                super().__setattr__(key, value)
            if key not in Role.__annotations__ and not isinstance(value, property):
                self._to_set_in_role_taker[key] = value

    @cached_property
    def role_taker_field_names(self) -> List[str]:
        """
        Returns a list of field names that are defined on the role taker class.
        """
        return [f.name for f in fields(self.get_role_taker_type())]

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
        Update the SymbolGraph mapping between this role and its role taker.

        Silently skips if this class is not registered in the SymbolGraph class
        diagram (e.g. test-only or dynamically created classes).

        :param role_taker: The role taker instance to link.
        """
        try:
            wrapped_self = SymbolGraph().get_wrapped_instance(self)
            wrapped_role_taker = SymbolGraph().ensure_wrapped_instance(role_taker)
            SymbolGraph().add_relation(
                HasRoleTaker(
                    wrapped_self, wrapped_role_taker, self.role_taker_wrapped_field
                )
            )
            if isinstance(role_taker, Role):
                for relation in SymbolGraph().get_outgoing_relations_with_type(
                    wrapped_role_taker, HasRoleTaker
                ):
                    SymbolGraph().add_relation(
                        HasRoleTaker(
                            wrapped_self, relation.target, relation.wrapped_field
                        )
                    )
        except ClassIsUnMappedInClassDiagram:
            pass

    @cached_property
    def role_taker_wrapped_field(self) -> WrappedField:
        """
        :return: The wrapped field of this class that is pointing to the role taker.
        """
        return next(
            wf
            for wf in SymbolGraph()
            .class_diagram.get_wrapped_class(self.__class__)
            .fields
            if wf.name == self.role_taker_attribute_name()
        )

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


class HasRoleTaker(PredicateClassRelation[Role]): ...


def delegate_property(name, role_taker):
    """
    Creates a property that delegates to another attribute's attribute.
    """

    def getter(self):
        target = getattr(self, role_taker)
        return getattr(target, name)

    def setter(self, value):
        if not self._role_taker_field_set:
            return
        target = getattr(self, role_taker)
        setattr(target, name, value)

    return property(getter, setter)
