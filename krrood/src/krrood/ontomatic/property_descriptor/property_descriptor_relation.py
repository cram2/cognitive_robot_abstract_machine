from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from typing_extensions import (
    Optional,
    Type,
    Iterable,
    Tuple,
    List,
    TYPE_CHECKING,
    Union,
    Iterator,
)

from .mixins import TransitiveProperty, HasInverseProperty
from ...class_diagrams.class_diagram import Association, AssociationThroughRoleTaker
from ...class_diagrams.wrapped_field import WrappedField
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

    def update_source_and_add_to_graph_and_apply_implications(self):
        """
        Update the source wrapped-field value, add this relation to the graph, and apply all implications of adding this
         relation.
        """
        source_updated = not self.inferred and self.update_source_wrapped_field_value()
        if not source_updated:
            # Means that the value was already set, so we don't need to infer anything.
            return
        self.add_to_graph_and_apply_implications()

    def add_to_graph_and_apply_implications(self):
        """
        Add this relation to the graph and apply all implications of this relation.
        """
        if self.add_to_graph():
            self.infer_and_apply_implications()

    def infer_and_apply_implications(self):
        """
        Infer all implications of adding this relation and apply them to the corresponding objects.
        """
        self.infer_super_relations()
        self.infer_inverse_relation()
        self.infer_transitive_relations()

    def update_source_wrapped_field_value(self) -> bool:
        """
        Update the wrapped field value for the source instance.

        :return: True if the value of the wrapped field was updated, False otherwise (i.e., if the value was already
        set).
        """
        return self.wrapped_field.property_descriptor.update_value(
            self.source.instance, self.target.instance
        )

    def infer_super_relations(self):
        """
        Infer all super relations of this relation.
        """
        for super_domain, super_field in self.super_relations:
            self.__class__(
                super_domain, self.target, super_field, inferred=True
            ).update_source_and_add_to_graph_and_apply_implications()

    def infer_inverse_relation(self):
        """
        Infer the inverse relation if it exists.
        """
        if self.inverse_of:
            inverse_domain, inverse_field = self.inverse_domain_and_field
            self.__class__(
                inverse_domain, self.source, inverse_field, inferred=True
            ).update_source_and_add_to_graph_and_apply_implications()

    @cached_property
    def super_relations(self) -> Iterable[Tuple[WrappedInstance, WrappedField]]:
        """
        Find neighboring symbols connected by super edges.

        This method identifies neighboring symbols that are connected
        through edge with relation types that are superclasses of the current relation type.

        :return: An iterator over neighboring symbols and relations that are super relations.
        """
        source_type = self.source.instance_type
        property_descriptor_cls: Type[PropertyDescriptor] = (
            self.wrapped_field.property_descriptor.__class__
        )
        for association in property_descriptor_cls.get_superproperties_associations(
            source_type
        ):
            original_source_instance = association.get_original_source_instance_given_this_relation_source_instance(
                self.source.instance
            )
            source = SymbolGraph().get_wrapped_instance(original_source_instance)
            yield source, association.field

    @property
    def inverse_domain_and_field(self) -> Tuple[WrappedInstance, WrappedField]:
        """
        Get the inverse of the property descriptor.

        :return: The inverse domain instance and property descriptor field.
        """
        if not self.inverse_association:
            raise ValueError(
                f"cannot find a field for the inverse {self.inverse_of} defined for the relation {self}"
            )
        original_source_instance = self.inverse_association.get_original_source_instance_given_this_relation_source_instance(
            self.target
        )
        source = SymbolGraph().get_wrapped_instance(original_source_instance)
        return source, self.inverse_association.field

    @cached_property
    def inverse_association(
        self,
    ) -> Optional[Union[Association, AssociationThroughRoleTaker]]:
        """
        Return the inverse field (if it exists) stored in the target of this relation.
        """
        return self.inverse_of.get_association_of_source_type(self.target.instance_type)

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
            ).update_source_and_add_to_graph_and_apply_implications()

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
            ).update_source_and_add_to_graph_and_apply_implications()

    @property
    def target_outgoing_relations_with_same_descriptor_type(
        self,
    ) -> Iterator[PredicateClassRelation]:
        """
        Get the outgoing relations from the target that have the same property descriptor type as this relation.
        """
        relation_condition = lambda relation: issubclass(
            relation.property_descriptor_cls, self.property_descriptor_cls
        )
        yield from SymbolGraph().get_outgoing_relations_with_condition(
            self.target, relation_condition
        )

    @property
    def source_incoming_relations_with_same_descriptor_type(
        self,
    ) -> Iterator[PredicateClassRelation]:
        """
        Get the incoming relations from the source that have the same property descriptor type as this relation.
        """
        relation_condition = lambda relation: issubclass(
            relation.property_descriptor_cls, self.property_descriptor_cls
        )
        yield from SymbolGraph().get_incoming_relations_with_condition(
            self.source, relation_condition
        )

    @cached_property
    def property_descriptor_cls(self) -> Type[PropertyDescriptor]:
        """
        Return the property descriptor class of the relation.
        """
        return type(self.wrapped_field.property_descriptor)
