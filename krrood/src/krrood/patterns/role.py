from __future__ import annotations

import sys
from dataclasses import Field, dataclass, field, fields
from functools import lru_cache

from typing_extensions import (
    Any,
    ClassVar,
    Iterator,
    List,
    Tuple,
    Type,
    TypeVar,
)

from krrood.class_diagrams.method_classifier import factory_method, is_factory_method
from krrood.class_diagrams.utils import T
from krrood.patterns.exceptions import (
    DelegatedFactoryMethodError,
    RoleAttributeNotDeclaredError,
    RoleTakerFieldNotFound,
)
from krrood.patterns.role_registry import RoleRegistry
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.symbol_graph.symbol_graph import Symbol
from krrood.utils import get_generic_type_params


class RoleTakerField(Field):
    """
    Dataclass field that marks the role taker of a role.

    Subclasses :class:`dataclasses.Field` as a type-level marker: the role pattern and the
    class-diagram detection identify the role taker by ``isinstance`` against this type rather
    than by inspecting field metadata. It adds no state, so it declares empty ``__slots__`` to
    keep its memory layout identical to :class:`dataclasses.Field`; this lets
    :func:`role_taker_field` produce one by reassigning the ``__class__`` of a field built by
    :func:`dataclasses.field`.
    """

    __slots__ = ()


def role_taker_field(*, kw_only: bool = True, **kwargs: Any) -> Any:
    """
    Declare the dataclass field that stores a role's role taker.

    A thin wrapper around :func:`dataclasses.field` that produces a :class:`RoleTakerField` so
    both the role pattern and the class diagram recognise it as the role taker.

    :param kw_only: Whether the role-taker field is keyword-only in the generated constructor.
    :param kwargs: Any other keyword arguments accepted by :func:`dataclasses.field`.
    :return: A dataclass field marked as the role taker.
    """
    declared_field = field(kw_only=kw_only, **kwargs)
    # Reassign __class__ instead of constructing RoleTakerField directly: this delegates all
    # argument handling and validation to dataclasses.field(). RoleTakerField declares
    # __slots__ = (), so its memory layout matches Field and the reassignment is safe.
    declared_field.__class__ = RoleTakerField
    return declared_field


@dataclass(eq=False)
class Role(Symbol, SubClassSafeGeneric[T]):
    """
    Base class for the Role design pattern.

    A role adds context-specific attributes and behaviour to an existing object (the *role
    taker*) without altering the role taker's identity. A role is an ordinary object with
    its own identity: it is equal only to itself and is **not** equal to its role taker. To
    ask whether two objects refer to the same underlying entity, use the :class:`IsSameEntity
    <krrood.patterns.role_predicates.IsSameEntity>` predicate.

    **Pure composition.** A role class must not inherit from its role taker type. Role
    membership is expressed through :meth:`has_role` and :meth:`roles_for`, not through
    ``isinstance`` checks. Constructing a role always requires the role taker to be passed
    explicitly.

    **Defining a role.** Inherit from ``Role[T]`` (where ``T`` is the role taker type),
    decorate with ``@dataclass(eq=False)``, and mark exactly one field as the role taker with
    :func:`role_taker_field`. Add any role-specific fields alongside it.

    **Attribute access.** Reading an attribute that is not declared on the role is delegated
    to the role taker via ``__getattr__`` (role-native attributes are resolved first).
    Assignments always set the attribute on the role itself and never modify the role taker:
    only the role's own declared fields may be assigned, and any other name raises
    :class:`RoleAttributeNotDeclaredError
    <krrood.patterns.exceptions.RoleAttributeNotDeclaredError>`. To change the role taker,
    assign through ``role.role_taker``.

    **Distinct identity.** A role is a distinct object from its role taker: they do not
    compare equal and do not share a hash, so multiple roles (even of the same type) on one
    taker stay distinct in sets and dicts. Semantic equivalence ("same underlying entity")
    is expressed through the :class:`IsSameEntity
    <krrood.patterns.role_predicates.IsSameEntity>` predicate, which unwraps roles to their
    :attr:`root_persistent_entity`:
    >>> from dataclasses import dataclass
    >>> from krrood.patterns.role_predicates import IsSameEntity
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
    False
    >>> person is student
    False
    >>> len({person, student})
    2
    >>> bool(IsSameEntity(person, student))
    True
    """

    _cache_instances_: ClassVar[bool] = False
    """Roles are not cached as instances in the :class:`SymbolGraph
    <krrood.symbol_graph.symbol_graph.SymbolGraph>`. Membership queries use :attr:`_role_registry`
    and persistence uses the role's own ORM mapping, so a role need not be a graph node. Role
    classes still take part in the class diagram and ORM through ``Symbol`` subclassing.

    A subclass that needs its instances enumerable through the SymbolGraph (for example to match an
    entity-query-language query) may override this to ``True`` for itself; the override applies to
    that subclass and its descendants and does not affect membership queries."""

    _role_registry: ClassVar[RoleRegistry] = RoleRegistry()
    """Inverse index from takers to roles, shared by all role types, that backs the membership
    queries (:meth:`has_role`, :meth:`roles_for`). Replace it with a fresh
    :class:`RoleRegistry <krrood.patterns.role_registry.RoleRegistry>` to isolate state."""

    @classmethod
    @lru_cache
    def role_taker_field_name(cls) -> str:
        """
        :return: The name of the field that holds the role taker instance.
        """
        for declared_field in fields(cls):
            if isinstance(declared_field, RoleTakerField):
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
        type(self)._role_registry.register(self)

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
        if not isinstance(role_taker, Role) and is_factory_method(
            type(role_taker), item
        ):
            return self._delegated_factory_guard(role_taker, item)
        return getattr(role_taker, item)

    def _delegated_factory_guard(self, role_taker: T, item: str):
        """
        Build a callable that refuses to run a role-taker factory method delegated through a role.

        :param role_taker: The role taker that declares the factory method.
        :param item: The name of the factory method.
        :return: A callable that raises :class:`DelegatedFactoryMethodError` when invoked.
        """

        def guard(*args: Any, **kwargs: Any):
            raise DelegatedFactoryMethodError(
                role_type=type(self),
                taker_type=type(role_taker),
                method_name=item,
            )

        guard.__name__ = item
        return guard

    @classmethod
    @lru_cache
    def _declared_field_names(cls) -> Tuple[str, ...]:
        """
        :return: The names of the dataclass fields declared on the role class.
        """
        return tuple(declared_field.name for declared_field in fields(cls))

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Set an attribute on the role itself; assignment never modifies the role taker.

        Only the role's own declared fields and private (underscore-prefixed) attributes may be
        assigned. Any other name raises :class:`RoleAttributeNotDeclaredError`, so a write cannot
        silently shadow a role-taker attribute. To change the role taker, assign through
        :attr:`role_taker`.

        :param key: The attribute name being set.
        :param value: The value to set.
        :raises RoleAttributeNotDeclaredError: If *key* is neither a declared field nor private.
        """
        if key.startswith("_") or key in type(self)._declared_field_names():
            super().__setattr__(key, value)
            return
        raise RoleAttributeNotDeclaredError(role_type=type(self), attribute_name=key)

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
        yield from (
            role
            for role in cls._role_registry.roles_of(role_taker)
            if isinstance(role, role_types)
        )

    @property
    def all_role_takers(self) -> List[Any]:
        """
        :return: This role's taker and every transitive taker beneath it, ending at the root entity.
        """
        takers: List[Any] = []
        current = self.role_taker
        while True:
            takers.append(current)
            if not isinstance(current, Role):
                return takers
            current = current.role_taker

    # A role is equal only to itself (identity), never to its taker; "same underlying entity?"
    # is answered by the ``IsSameEntity`` predicate instead. ``__eq__`` returns ``False`` rather
    # than ``NotImplemented`` so Python cannot fall back to the taker's (possibly lenient) equality
    # via the reflected operand. ``__hash__`` is set explicitly because defining ``__eq__`` here,
    # like the plain-dataclass base, would otherwise reset it to ``None`` (unhashable).
    def __eq__(self, other: Any) -> bool:
        return self is other

    __hash__ = object.__hash__
