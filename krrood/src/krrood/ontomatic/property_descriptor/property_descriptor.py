from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, fields
from functools import cached_property, lru_cache

from typing_extensions import (
    ClassVar,
    Set,
    Type,
    Optional,
    Any,
    Iterable,
    Dict,
    Union,
    Tuple,
    List,
    DefaultDict,
)

from .monitored_container import (
    MonitoredContainer,
    monitored_type_map,
)
from .property_descriptor_relation import PropertyDescriptorRelation
from ..failures import UnMonitoredContainerTypeForDescriptor
from ...class_diagrams.class_diagram import WrappedClass, Association
from ...class_diagrams.wrapped_field import WrappedField
from ...entity_query_language.predicate import Symbol
from ...entity_query_language.symbol_graph import (
    SymbolGraph,
)
from ...entity_query_language.utils import make_set

SymbolType = Type[Symbol]
"""
Type alias for symbol types.
"""
DomainRangeMap = Dict[SymbolType, SymbolType]
"""
Type alias for the domain-range map.
"""


@dataclass
class PropertyDescriptor(Symbol):
    """Descriptor managing a data class field while giving it metadata like superproperties,
    sub-properties, inverse, transitivity, ...etc.

    The descriptor injects a hidden dataclass-managed attribute (backing storage) into the owner class
    and collects domain and range types for introspection.

    The way this should be used is after defining your dataclasses you declare either in the same file or in a separate
    file the descriptors for each field that is considered a relation between two symbol types.

    Example:
        >>> from krrood.ontomatic.property_descriptor.mixins import HasInverseProperty
        >>> from dataclasses import dataclass
        >>> from krrood.ontomatic.property_descriptor.property_descriptor import PropertyDescriptor
        >>> @dataclass
        ... class Company(Symbol):
        ...     name: str
        ...     members: Set[Person] = field(default_factory=set)
        ...
        >>> @dataclass
        ... class Person(Symbol):
        ...     name: str
        ...     works_for: Set[Company] = field(default_factory=set)
        ...
        >>> @dataclass
        >>> class Member(PropertyDescriptor):
        ...     pass
        ...
        >>> @dataclass
        ... class MemberOf(PropertyDescriptor, HasInverseProperty):
        ...     @classmethod
        ...     def get_inverse(cls) -> Type[PropertyDescriptor]:
        ...         return Member
        ...
        >>> @dataclass
        >>> class WorksFor(MemberOf):
        ...     pass
        ...
        >>> Person.works_for = WorksFor(Person, "works_for")
        >>> Company.members = Member(Company, "members")
    """

    domain: SymbolType
    """
    The domain type for this descriptor instance.
    """
    field_name: str
    """
    The name of the field on the domain type that this descriptor instance manages.
    """
    wrapped_field: WrappedField = field(init=False)
    """
    The wrapped field instance that this descriptor instance manages.
    """
    domain_range_map: ClassVar[
        DefaultDict[Type[PropertyDescriptor], DomainRangeMap]
    ] = defaultdict(dict)
    """
    A mapping from descriptor class to the mapping from domain types to range types for that descriptor class.
    """
    all_domains: ClassVar[Dict[Type[PropertyDescriptor], Set[SymbolType]]] = (
        defaultdict(set)
    )
    """
    A set of all domain types for this descriptor class.
    """
    all_ranges: ClassVar[Dict[Type[PropertyDescriptor], Set[SymbolType]]] = defaultdict(
        set
    )
    """
    A set of all range types for this descriptor class.
    """

    def __post_init__(self):
        self._validate_non_redundant_domain()
        self._update_wrapped_field()
        self._update_domain_and_range()

    @cached_property
    def private_attr_name(self) -> str:
        """
        The name of the private attribute that stores the values on the owner instance.
        """
        return f"_{self.wrapped_field.name}"

    def _validate_non_redundant_domain(self):
        """
        Validate that this exact descriptor type has not already been defined for this domain type.
        """
        if self.domain in self.domain_range_map[self.__class__]:
            raise ValueError(
                f"Domain {self.domain} already exists, cannot define same descriptor more than once in "
                f"the same class"
            )

    def _update_wrapped_field(self):
        """
        Set the wrapped field attribute using the domain type and field name.
        """
        field_ = [f for f in fields(self.domain) if f.name == self.field_name][0]
        self.wrapped_field = WrappedField(
            WrappedClass(self.domain), field_, property_descriptor=self
        )

    @cached_property
    def is_iterable(self):
        """Whether the field is iterable or not"""
        return self.wrapped_field.is_iterable

    def _update_domain_and_range(self):
        """
        Update the domain and range sets and the domain-range map for this descriptor type.
        """
        range_type = self.wrapped_field.type_endpoint
        assert issubclass(range_type, Symbol)
        self.domain_range_map[self.__class__][self.domain] = range_type
        for super_class in self.__class__.__mro__:
            if (super_class is PropertyDescriptor) or not issubclass(
                super_class, PropertyDescriptor
            ):
                continue
            self.all_domains[super_class].add(self.domain)
            self.all_ranges[super_class].add(range_type)

    @cached_property
    def range(self) -> SymbolType:
        """
        The range type for this descriptor instance.
        """
        return self.domain_range_map[self.__class__][self.domain]

    def add_relation_to_the_graph(
        self, domain_value: Symbol, range_value: Symbol, inferred: bool = False
    ) -> None:
        """
        Add the relation between the domain_value and the range_value to the symbol graph.

        :param domain_value: The domain value (i.e., the instance that this descriptor is attached to).
        :param range_value: The range value (i.e., the value to set on the managed attribute, and is the target of the
         relation).
        :param inferred: Whether the relation is inferred or not.
        """
        if domain_value and range_value:
            for v in make_set(range_value):
                PropertyDescriptorRelation(
                    domain_value, v, self.wrapped_field, inferred=inferred
                ).add_to_graph()

    def __get__(self, obj, objtype=None):
        """
        Get the value of the managed attribute. In addition, ensure that the value is a monitored container type if
        it is an iterable and that the owner instance is bound to the monitored container.

        :param obj: The owner instance (i.e., the instance that this descriptor is attached to).
        :param objtype: The owner type.
        """
        if obj is None:
            return self
        value = getattr(obj, self.private_attr_name)
        self._bind_owner_if_container_type(value, owner=obj)
        return value

    @staticmethod
    def _bind_owner_if_container_type(
        value: Union[Iterable[Symbol], Symbol], owner: Optional[Any] = None
    ):
        """
        Bind the owner instance to the monitored container if the value is a MonitoredContainer type.

        :param value: The value to check and bind the owner to if it is a MonitoredContainer type.
        :param owner: The owner instance.
        """
        if (
            isinstance(value, MonitoredContainer)
            and getattr(value, "owner", None) is not owner
        ):
            value._bind_owner(owner)

    def _ensure_monitored_type(
        self, value: Union[Iterable[Symbol], Symbol], obj: Optional[Any] = None
    ) -> Union[MonitoredContainer[Symbol], Symbol]:
        """
        Ensure that the value is a monitored container type or is not iterable.

        :param value: The value to ensure its type.
        :param obj: The owner instance.
        :return: The value with a monitored container-type if it is iterable, otherwise the value itself.
        """
        if self.is_iterable and not isinstance(value, MonitoredContainer):
            try:
                monitored_type = monitored_type_map[type(value)]
            except KeyError:
                raise UnMonitoredContainerTypeForDescriptor(
                    self.domain, self.wrapped_field.name, type(value)
                )
            monitored_value = monitored_type(descriptor=self)
            for v in make_set(value):
                monitored_value._add_item(v, inferred=False)
            value = monitored_value
        return value

    def __set__(self, obj, value):
        """
        Set the value of the managed attribute and add it to the symbol graph.

        :param obj: The owner instance.
        :param value: The value to set.
        """
        if isinstance(value, PropertyDescriptor):
            return
        attr = getattr(obj, self.private_attr_name, None)
        if self.is_iterable and not isinstance(attr, MonitoredContainer):
            attr = self._ensure_monitored_type(value, obj)
            self._bind_owner_if_container_type(attr, owner=obj)
            setattr(obj, self.private_attr_name, attr)
        if isinstance(attr, MonitoredContainer):
            attr._clear()
            for v in make_set(value):
                attr._add_item(v, inferred=False)
        else:
            setattr(obj, self.private_attr_name, value)
            self.add_relation_to_the_graph(obj, value)

    def update_value(
        self,
        domain_value: Symbol,
        range_value: Symbol,
    ) -> bool:
        """Update the value of the managed attribute

        :param domain_value: The domain value to update (i.e., the instance that this descriptor is attached to).
        :param range_value: The range value to update (i.e., the value to set on the managed attribute).
        """
        v = getattr(domain_value, self.private_attr_name)
        updated = False
        if isinstance(v, MonitoredContainer):
            updated = v._update(range_value, add_relation_to_the_graph=False)
        elif v != range_value:
            setattr(domain_value, self.private_attr_name, range_value)
            updated = True
        return updated

    @classmethod
    @lru_cache(maxsize=None)
    def get_associated_field_of_domain_type(
        cls,
        domain_type: Union[Type[Symbol], WrappedClass],
    ) -> Optional[WrappedField]:
        """
        Get the field of the domain type that is associated with this descriptor class.

        :param domain_type: The domain type that has an associated field with this descriptor class.
        """
        class_diagram = SymbolGraph().class_diagram
        association_condition = (
            lambda association: type(association.field.property_descriptor) is cls
        )
        result = next(
            iter(
                class_diagram.get_associations_with_condition(
                    domain_type, association_condition
                )
            ),
            None,
        )
        return result.field if result else None

    @classmethod
    @lru_cache(maxsize=None)
    def get_fields_of_superproperties_in_role_taker_of_class(
        cls,
        domain_type: Union[SymbolType, WrappedClass],
    ) -> Tuple[Optional[WrappedField], List[WrappedField]]:
        """
        Return the role-taker field and all associated fields that are superproperties of this descriptor class.

        :param domain_type: The domain type that has a role-taker, where the role-taker has associated fields with the
         super properties of this descriptor class.
        """
        class_diagram = SymbolGraph().class_diagram
        role_taker_assoc = class_diagram.get_role_taker_associations_of_cls(domain_type)
        if role_taker_assoc:
            role_taker_fields = cls.get_fields_of_superproperties(
                role_taker_assoc.target
            )
            return role_taker_assoc.field, list(role_taker_fields)
        return None, []

    @classmethod
    @lru_cache(maxsize=None)
    def get_fields_of_superproperties(
        cls,
        domain_type: Union[SymbolType, WrappedClass],
    ) -> Tuple[WrappedField, ...]:
        """
        Get the fields of the domain type that are associated with the super classes of this descriptor class.

        :param domain_type: The domain type that has an associated field with the super classes of this descriptor class.
        """

        def association_condition(association: Association) -> bool:
            return (
                issubclass(cls, type(association.field.property_descriptor))
                and type(association.field.property_descriptor) is not cls
            )

        class_diagram = SymbolGraph().class_diagram

        association_fields = [
            assoc.field
            for assoc in class_diagram.get_associations_with_condition(
                domain_type, association_condition
            )
        ]
        return tuple(association_fields)
