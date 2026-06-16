from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from functools import lru_cache

from typing_extensions import (
    Any,
    Iterator,
    List,
    Tuple,
    Type,
    TypeVar,
)

from krrood.class_diagrams.exceptions import ClassIsUnMappedInClassDiagram
from krrood.class_diagrams.utils import ROLE_TAKER_METADATA_KEY, T
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.exceptions import DataclassException
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.symbol_graph.symbol_graph import (
    PredicateClassRelation,
    Symbol,
    SymbolGraph,
)
from krrood.utils import get_generic_type_params


@dataclass
class RoleTakerFieldNotFound(DataclassException):
    """
    Raised when a role class has no field declared as its role taker.
    """

    role_type: Type
    """
    The role class that is missing a role-taker field.
    """

    def __post_init__(self):
        self.message = (
            f"{self.role_type.__name__} declares no role-taker field. A role must mark "
            f"exactly one field with role_taker_field()."
        )
        super().__post_init__()


def role_taker_field(*, kw_only: bool = True, **kwargs: Any) -> Any:
    """
    Declare the dataclass field that stores a role's role taker.

    A thin wrapper around :func:`dataclasses.field` that tags the field in its metadata so
    both the role pattern and the class diagram recognise it as the role taker.

    :param kw_only: Whether the role-taker field is keyword-only in the generated constructor.
    :param kwargs: Any other keyword arguments accepted by :func:`dataclasses.field`.
    :return: A dataclass field marked as the role taker.
    """
    metadata = {**kwargs.pop("metadata", {}), ROLE_TAKER_METADATA_KEY: True}
    return field(kw_only=kw_only, metadata=metadata, **kwargs)


@dataclass(eq=False)
class Role(Symbol, SubClassSafeGeneric[T]):
    """
    Base class for the Role design pattern.

    A role adds context-specific attributes and behaviour to an existing object (the *role
    taker*) without altering the role taker's identity. A role and its role taker are
    considered the same entity — they compare equal and share the same hash (Not same memory address nor python id).

    **Pure composition.** A role class must not inherit from its role taker type. Role
    membership is expressed through :meth:`has_role` and :meth:`roles_for`, not through
    ``isinstance`` checks. Constructing a role always requires the role taker to be passed
    explicitly.

    **Defining a role.** Inherit from ``Role[T]`` (where ``T`` is the role taker type),
    decorate with ``@dataclass(eq=False)``, and mark exactly one field as the role taker with
    :func:`role_taker_field`. Add any role-specific fields alongside it.

    **Attribute delegation.** Attributes not declared on the role are forwarded to the role
    taker via ``__getattr__`` and ``__setattr__``. Role-native attributes are always
    resolved first.

    **Identity sharing.** The hash and equality of a role are derived from the root
    non-role entity at the bottom of the taker chain, so any collection or dictionary keyed
    by the taker automatically reflects the role:
    >>> from dataclasses import dataclass
    >>> @dataclass(unsafe_hash=True)
    ... class Person:
    ...     name: str
    >>> @dataclass(eq=False)
    ... class Student(Role[Person]):
    ...     person: Person = role_taker_field()
    ...     major: str = ""
    >>> person = Person(name="Alice")
    >>> student = Student(person=person)
    >>> student.name is person.name
    True
    >>> person == student
    True
    >>> hash(person) == hash(student)
    True
    >>> person is student
    False
    >>> id(person) == id(student)
    False
    """

    @classmethod
    @lru_cache
    def role_taker_field_name(cls) -> str:
        """
        :return: The name of the field that holds the role taker instance.
        """
        for declared_field in fields(cls):
            if declared_field.metadata.get(ROLE_TAKER_METADATA_KEY, False):
                return declared_field.name
        raise RoleTakerFieldNotFound(role_type=cls)

    @property
    def role_taker(self) -> T:
        """
        :return: The role taker instance wrapped by this role.
        """
        return object.__getattribute__(self, type(self).role_taker_field_name())

    @classmethod
    def get_role_taker_type(cls) -> Type[T]:
        """
        :return: The concrete type of this role's role taker.
        """
        try:
            type_ = next(
                declared_field.type
                for declared_field in fields(cls)
                if declared_field.name == cls.role_taker_field_name()
            )
        except (StopIteration, RoleTakerFieldNotFound):
            type_ = get_generic_type_params(cls, Role)[0]
        if isinstance(type_, str):
            module_namespace = sys.modules[cls.__module__].__dict__
            try:
                type_ = module_namespace[type_]
            except KeyError:
                type_ = eval(type_, module_namespace)
        if isinstance(type_, TypeVar):
            if type_.__bound__ is None:
                raise ValueError(f"TypeVar {type_} has no bound")
            type_ = type_.__bound__
        return type_

    @classmethod
    @lru_cache
    def get_root_role_taker_type(cls) -> Type[T]:
        """
        :return: The type of the non-role entity at the bottom of this role's taker chain.
        """
        current_cls = cls
        while issubclass(current_cls, Role):
            current_cls = current_cls.get_role_taker_type()
        return current_cls

    @property
    def root_persistent_entity(self) -> Any:
        """
        :return: The non-role entity at the bottom of this role's taker chain.
        """
        current = self
        while isinstance(current, Role):
            current = current.role_taker
        return current

    def __post_init__(self):
        super_post_init = getattr(super(), "__post_init__", None)
        if super_post_init is not None:
            super_post_init()
        self._update_mapping_between_roles_and_role_takers(self.role_taker)

    def __getattr__(self, item: str) -> Any:
        """
        Delegate attribute reads that are not role-native to the role taker.

        ``__getattr__`` only runs when normal attribute lookup fails, so role-native
        attributes never reach here.

        :param item: The attribute name being read.
        :return: The attribute value taken from the role taker.
        """
        taker_field_name = type(self).role_taker_field_name()
        if item == taker_field_name:
            raise AttributeError(item)
        try:
            role_taker = object.__getattribute__(self, taker_field_name)
        except AttributeError:
            raise AttributeError(item)
        return getattr(role_taker, item)

    def __setattr__(self, key: str, value: Any):
        """
        Set role-native attributes on the role and delegate the rest to the role taker.

        :param key: The attribute name being set.
        :param value: The value to set.
        """
        if key in type(self).role_native_field_names() or not self._role_taker_is_set():
            super().__setattr__(key, value)
            return
        role_taker = self.role_taker
        if hasattr(role_taker, key):
            setattr(role_taker, key, value)
        else:
            super().__setattr__(key, value)

    @classmethod
    @lru_cache
    def role_native_field_names(cls) -> Tuple[str, ...]:
        """
        :return: The names of the dataclass fields declared on the role itself.
        """
        return tuple(declared_field.name for declared_field in fields(cls))

    def _role_taker_is_set(self) -> bool:
        """
        :return: Whether the role taker field has been assigned yet.
        """
        try:
            object.__getattribute__(self, type(self).role_taker_field_name())
        except (AttributeError, RoleTakerFieldNotFound):
            return False
        return True

    @classmethod
    def from_role_taker(cls, role_taker: T) -> Role[T]:
        """
        Factory method to create a role instance for a given role taker.

        :param role_taker: The role taker instance to create the role for.
        :return: An instance of the role for the given role taker.
        """
        return cls(**{cls.role_taker_field_name(): role_taker})

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
        :return: All roles of this role's role taker instance.
        """
        return self.get_taker_roles_of_type(self.role_taker, Role)

    @classmethod
    def get_taker_roles_of_type(
        cls, role_taker: T, role_type: Type[Role[T]]
    ) -> List[Role[T]]:
        """
        :param role_taker: The role taker instance to query.
        :param role_type: The type of roles to return.
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
        :param role: The role whose role takers are requested.
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

    def _update_mapping_between_roles_and_role_takers(self, role_taker: T):
        """
        Update the SymbolGraph mapping between this role and its role taker.

        Silently skips if this class is not registered in the SymbolGraph class
        diagram (for example, test-only or dynamically created classes).

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

    @property
    def role_taker_wrapped_field(self) -> WrappedField:
        """
        :return: The wrapped field of this class that points to the role taker.
        """
        return next(
            wrapped_field
            for wrapped_field in SymbolGraph()
            .class_diagram.get_wrapped_class(self.__class__)
            .fields
            if wrapped_field.name == self.role_taker_field_name()
        )

    def __hash__(self):
        """
        A persistent entity and its roles are considered the same entity, so the hash is
        based on the root persistent entity.
        """
        return hash(self.root_persistent_entity)

    def __eq__(self, other):
        return hash(self) == hash(other)


class HasRoleTaker(PredicateClassRelation[Role]): ...
