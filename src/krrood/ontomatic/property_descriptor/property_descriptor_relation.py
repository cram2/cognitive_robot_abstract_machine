from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing_extensions import Optional, Type, Iterable, Tuple, List, TYPE_CHECKING

from ...class_diagrams.class_diagram import Association
from ...class_diagrams.wrapped_field import WrappedField
from .mixins import TransitiveProperty, HasInverseProperty
from ...entity_query_language.symbol_graph import (
    PredicateClassRelation,
    SymbolGraph,
    WrappedInstance,
)

if TYPE_CHECKING:
    from .property_descriptor import PropertyDescriptor


@dataclass(unsafe_hash=True)
class PropertyDescriptorRelation(PredicateClassRelation):
    """
    Edge data representing a relation between two wrapped instances that is represented structurally by a property
    descriptor attached to the source instance.
    """

    @cached_property
    def transitive(self) -> bool:
        """
        If the relation is transitive or not.
        """
        if self.property_descriptor_cls:
            return issubclass(self.property_descriptor_cls, TransitiveProperty)
        else:
            return False

    @cached_property
    def inverse_of(self) -> Optional[Type[PropertyDescriptor]]:
        """
        The inverse of the relation if it exists.
        """
        if self.property_descriptor_cls and issubclass(
            self.property_descriptor_cls, HasInverseProperty
        ):
            return self.property_descriptor_cls.get_inverse()
        else:
            return None

    def add_to_graph(self):
        """
        Add the relation to the graph and infer additional relations if possible. In addition, update the value of
         the wrapped field in the source instance if this relation is an inferred relation.
        """
        if super().add_to_graph():
            if self.inferred:
                self.update_source_wrapped_field_value()
            self.infer_super_relations()
            self.infer_inverse_relation()
            self.infer_transitive_relations()

    def update_source_wrapped_field_value(self):
        """
        Update the wrapped field value for the source instance.
        """
        self.wrapped_field.property_descriptor.update_value(
            self.source.instance, self.target.instance
        )

    def infer_super_relations(self):
        """
        Infer all super relations of this relation.
        """
        for super_domain, super_field in self.super_relations:
            self.__class__(
                super_domain, self.target, super_field, inferred=True
            ).add_to_graph()

    def infer_inverse_relation(self):
        """
        Infer the inverse relation if it exists.
        """
        if self.inverse_of:
            inverse_domain, inverse_field = self.inverse_domain_and_field
            self.__class__(
                inverse_domain, self.source, inverse_field, inferred=True
            ).add_to_graph()

    @cached_property
    def super_relations(self) -> Iterable[Tuple[WrappedInstance, WrappedField]]:
        """
        Find neighboring symbols connected by super edges.

        This method identifies neighboring symbols that are connected
        through edge with relation types that are superclasses of the current relation type.

        Also, it looks for role taker super relations of the source if it exists.

        :return: An iterator over neighboring symbols and relations that are super relations.
        """
        yield from self.direct_super_relations
        yield from self.role_taker_super_relations

    @cached_property
    def direct_super_relations(self):
        """
        Return the direct super relations of the source.
        """
        source_type = self.source.instance_type
        property_descriptor_cls: PropertyDescriptor = (
            self.wrapped_field.property_descriptor.__class__
        )
        yield from (
            (self.source, f)
            for f in property_descriptor_cls.get_fields_of_superproperties(source_type)
        )

    @cached_property
    def role_taker_super_relations(self):
        """
        Return the source role taker super relations.
        """
        if not self.role_taker_fields:
            return
        role_taker = getattr(
            self.source.instance, self.source_role_taker_association.field.public_name
        )
        role_taker = SymbolGraph().ensure_wrapped_instance(role_taker)
        yield from ((role_taker, f) for f in self.role_taker_fields)

    @cached_property
    def role_taker_fields(self) -> List[WrappedField]:
        """
        Return the role taker fields of the source role taker association.
        """
        if not self.source_role_taker_association:
            return []
        return list(
            self.property_descriptor_cls.get_fields_of_superproperties(
                self.source_role_taker_association.target
            )
        )

    @cached_property
    def source_role_taker_association(self) -> Optional[Association]:
        """
        Return the source role taker association of the relation.
        """
        class_diagram = SymbolGraph().class_diagram
        return class_diagram.get_role_taker_associations_of_cls(
            self.source.instance_type
        )

    @property
    def inverse_domain_and_field(self) -> Tuple[WrappedInstance, WrappedField]:
        """
        Get the inverse of the property descriptor.

        :return: The inverse domain instance and property descriptor field.
        """
        if self.inverse_field:
            return self.target, self.inverse_field
        elif self.inverse_field_from_target_role_taker:
            return self.target_role_taker, self.inverse_field_from_target_role_taker
        else:
            raise ValueError(
                f"cannot find a field for the inverse {self.inverse_of} defined for the relation {self}"
            )

    @cached_property
    def target_role_taker(self) -> Optional[WrappedInstance]:
        """
        Return the role taker of the target if it exists.
        """
        if not self.target_role_taker_association:
            return None
        role_taker = getattr(
            self.target.instance,
            self.target_role_taker_association.field.public_name,
            None,
        )
        return SymbolGraph().ensure_wrapped_instance(role_taker)

    @cached_property
    def inverse_field_from_target_role_taker(self) -> Optional[WrappedField]:
        """
        Return the inverse field of this relation field that is stored in the role taker of the target.
        """
        if not self.target_role_taker_association:
            return None
        return self.inverse_of.get_associated_field_of_domain_type(
            self.target_role_taker_association.target
        )

    @cached_property
    def target_role_taker_association(self) -> Optional[Association]:
        """
        Return role taker association of the target if it exists.
        """
        class_diagram = SymbolGraph().class_diagram
        return class_diagram.get_role_taker_associations_of_cls(
            self.target.instance_type
        )

    @cached_property
    def inverse_field(self) -> Optional[WrappedField]:
        """
        Return the inverse field (if it exists) stored in the target of this relation.
        """
        return self.inverse_of.get_associated_field_of_domain_type(
            self.target.instance_type
        )

    def infer_transitive_relations(self):
        """
        Add all transitive relations of this relation type that results from adding this relation to the graph.
        """
        if self.transitive:
            self.infer_transitive_relations_outgoing_from_source()
            self.infer_transitive_relations_incoming_to_target()

    def infer_transitive_relations_outgoing_from_source(self):
        """
        Infer transitive relations outgoing from the source.
        """
        for nxt_relation in self.target_outgoing_relations_with_same_descriptor_type:
            self.__class__(
                self.source,
                nxt_relation.target,
                nxt_relation.wrapped_field,
                inferred=True,
            ).add_to_graph()

    def infer_transitive_relations_incoming_to_target(self):
        """
        Infer transitive relations incoming to the target.
        """
        for nxt_relation in self.source_incoming_relations_with_same_descriptor_type:
            self.__class__(
                nxt_relation.source,
                self.target,
                nxt_relation.wrapped_field,
                inferred=True,
            ).add_to_graph()

    @property
    def target_outgoing_relations_with_same_descriptor_type(
        self,
    ) -> Iterable[PredicateClassRelation]:
        """
        Get the outgoing relations from the target that have the same property descriptor type as this relation.
        """
        relation_condition = (
            lambda relation: relation.property_descriptor_cls
            is self.property_descriptor_cls
        )
        yield from SymbolGraph().get_outgoing_relations_with_condition(
            self.target, relation_condition
        )

    @property
    def source_incoming_relations_with_same_descriptor_type(
        self,
    ) -> Iterable[PredicateClassRelation]:
        """
        Get the incoming relations from the source that have the same property descriptor type as this relation.
        """
        relation_condition = (
            lambda relation: relation.property_descriptor_cls
            is self.property_descriptor_cls
        )
        yield from SymbolGraph().get_incoming_relations_with_condition(
            self.source, relation_condition
        )

    @cached_property
    def property_descriptor_cls(self) -> Type[PropertyDescriptor]:
        """
        Return the property descriptor class of the relation.
        """
        return self.wrapped_field.property_descriptor.__class__
