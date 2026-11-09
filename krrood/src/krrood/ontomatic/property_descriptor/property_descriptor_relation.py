"""
Relations between instances held by property descriptors, what each one implies, and how an implied one was derived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property

from typing_extensions import TYPE_CHECKING, Any, Iterator

from krrood.ontomatic.failures import NoObjectOfIndividualDeclaresProperty
from krrood.ontomatic.property_descriptor.mixins import (
    HasEquivalentProperties,
    HasInverseProperty,
    SymmetricProperty,
    TransitiveProperty,
)
from krrood.patterns.role import Role
from krrood.symbol_graph.symbol_graph import (
    PredicateClassRelation,
    SymbolGraph,
    WrappedInstance,
)

if TYPE_CHECKING:
    from krrood.ontomatic.property_descriptor.property_descriptor import (
        PropertyDescriptor,
    )


# %% provenance


class InferredThrough(Enum):
    """
    The rule a relation was inferred through.
    """

    EQUIVALENT = "equivalent"
    """
    The relation of an equivalent property. Premise: that relation.
    """
    INVERSE = "inverse"
    """
    The relation of the inverse property, reversed. Premise: that relation.
    """
    SUPER = "super"
    """
    The relation of a subproperty. Premise: that relation.
    """
    TRANSITIVE = "transitive"
    """
    Two relations of a transitive property, composed. Premises: the two relations.
    """
    SYMMETRY = "symmetry"
    """
    The relation of a symmetric property, reversed. Premise: that relation.
    """
    CHAIN = "chain"
    """
    The relations of a property chain. Premises: the relations of the chain, in order.
    """


@dataclass(frozen=True)
class RelationExplanation:
    """
    How an inferred relation was derived.
    """

    rule: InferredThrough
    """
    The rule the relation was inferred through.
    """

    premises: tuple[PropertyDescriptorRelation, ...]
    """
    The relations the rule was applied to, see :class:`InferredThrough`.
    """


# %% declarations and paths


@dataclass(frozen=True)
class Declaration:
    """
    An object of an individual whose class declares a property, with the descriptor of the declaration.
    """

    holder: Any
    """
    The object holding the field of the property.
    """

    descriptor: PropertyDescriptor
    """
    The descriptor managing that field.
    """


@dataclass(frozen=True)
class RelationPath:
    """
    A path of relations in the symbol graph, from its first node to its last.
    """

    first: WrappedInstance
    """
    The node the path starts from.
    """

    last: WrappedInstance
    """
    The node the path ends in, the first node for an empty path.
    """

    relations: tuple[PropertyDescriptorRelation, ...] = ()
    """
    The relations of the path, in order.
    """


# %% relations


@dataclass(eq=False, repr=False)
class PropertyDescriptorRelation(PredicateClassRelation):
    """
    Edge data representing a relation between two wrapped instances that is represented structurally by a property
    descriptor attached to the source instance.

    An individual is represented by its root role taker and the roles of it; a fact about the individual is stored on
    the first of these objects whose class declares the property (see :meth:`declaring_object`).
    """

    explanation: RelationExplanation | None = field(
        default=None, compare=False, hash=False
    )
    """
    How the relation was derived, or None if it was asserted.
    """

    # %% the objects of an individual

    @staticmethod
    def root_role_taker(instance: Any) -> Any:
        """
        :return: The object at the end of the role-taker chain of the instance, or the instance if it is no role.
        """
        return (
            instance.root_persistent_entity if isinstance(instance, Role) else instance
        )

    @classmethod
    def objects_of_individual(cls, instance: Any) -> list[Any]:
        """
        :param instance: An object of the individual.
        :return: The objects representing the individual of the instance: its root role taker followed by the roles
            of the root in the order they were created.
        """
        root = cls.root_role_taker(instance)
        return [root, *Role.roles_for(root)]

    @classmethod
    def declaring_object(
        cls,
        instance: Any,
        property_descriptor_class: type[PropertyDescriptor],
        value: Any,
    ) -> Declaration | None:
        """
        Find the object of the individual of the instance on which a fact of the property with the value is stored:
        the first object, in the order of :meth:`objects_of_individual`, whose most specific class declaring the
        property admits the value; else the first object declaring the property.

        :return: The declaration, or None if no object of the individual declares the property.
        """
        value_objects = cls.objects_of_individual(value)
        first_declaration = None
        for individual_object in cls.objects_of_individual(instance):
            descriptor = property_descriptor_class.declared_by(type(individual_object))
            if descriptor is None:
                continue
            declaration = Declaration(individual_object, descriptor)
            if any(
                isinstance(value_object, descriptor.range)
                for value_object in value_objects
            ):
                return declaration
            first_declaration = first_declaration or declaration
        return first_declaration

    # %% adding relations

    def update_source_and_add_to_graph_and_apply_implications(self):
        """
        Update the source wrapped-field value, add this relation to the graph, and apply all implications of adding
        this relation.
        """
        if not self.update_source():
            return
        self.add_to_graph_and_apply_implications()

    def add_to_graph_and_apply_implications(self):
        """
        Add this relation to the graph and apply all implications of this relation.
        """
        if self.add_to_graph():
            self.infer_and_apply_implications()

    def update_source(self) -> bool:
        """
        :return: Whether the relation is asserted, or is inferred and changed the value of the field holding it.
        """
        return not self.inferred or self.update_source_wrapped_field_value()

    def update_source_wrapped_field_value(self) -> bool:
        """
        Store the target in the field of the object of the source's individual that declares the property.

        :return: Whether the value of the field changed.
        :raises NoObjectOfIndividualDeclaresProperty: If no object of the individual declares the property.
        """
        declaration = self.declaring_object(
            self.source.instance,
            self.property_descriptor_cls,
            self.target.instance,
        )
        if declaration is None:
            raise NoObjectOfIndividualDeclaresProperty(
                self.source.instance, self.property_descriptor_cls
            )
        return declaration.descriptor.update_value(
            declaration.holder, self.target.instance
        )

    @classmethod
    def add_inferred_relation(
        cls,
        source_instance: Any,
        property_descriptor_class: type[PropertyDescriptor],
        target: WrappedInstance,
        explanation: RelationExplanation,
    ):
        """
        Store an inferred fact of the property about the individual of the source instance on the object of the
        individual that declares the property, add it to the graph and apply its implications. Nothing is stored if
        no object of the individual declares the property.
        """
        declaration = cls.declaring_object(
            source_instance, property_descriptor_class, target.instance
        )
        if declaration is None:
            return
        cls(
            SymbolGraph().ensure_wrapped_instance(declaration.holder),
            target,
            declaration.descriptor.wrapped_field,
            inferred=True,
            explanation=explanation,
        ).update_source_and_add_to_graph_and_apply_implications()

    # %% implications

    def infer_and_apply_implications(self):
        """
        Infer all implications of adding this relation and apply them to the corresponding objects.
        """
        self.infer_equivalent_relations()
        self.infer_super_relations()
        self.infer_inverse_relation()
        self.infer_transitive_relations()
        self.infer_chained_relations()
        self.infer_symmetric_relation()

    def infer_equivalent_relations(self):
        """
        Infer the relations of every property equivalent to this relation's property.
        """
        if self.rule is InferredThrough.EQUIVALENT or not issubclass(
            self.property_descriptor_cls, HasEquivalentProperties
        ):
            return
        for (
            equivalent_descriptor
        ) in self.property_descriptor_cls.get_equivalent_properties():
            self.add_inferred_relation(
                self.source.instance,
                equivalent_descriptor,
                self.target,
                RelationExplanation(InferredThrough.EQUIVALENT, (self,)),
            )

    def infer_super_relations(self):
        """
        Infer the relations of the superproperties of this relation's property.
        """
        for super_descriptor_type in self.property_descriptor_cls.super_classes():
            self.add_inferred_relation(
                self.source.instance,
                super_descriptor_type,
                self.target,
                RelationExplanation(InferredThrough.SUPER, (self,)),
            )

    def infer_inverse_relation(self):
        """
        Infer the relation of the inverse property from the target back to the source.
        """
        if self.inverse_of and self.rule is not InferredThrough.INVERSE:
            self.add_inferred_relation(
                self.target.instance,
                self.inverse_of,
                self.source,
                RelationExplanation(InferredThrough.INVERSE, (self,)),
            )

    def infer_symmetric_relation(self):
        """
        Infer the relation back from the target to the source if the property is symmetric.
        """
        if self.rule is InferredThrough.SYMMETRY:
            return
        if issubclass(self.property_descriptor_cls, SymmetricProperty):
            self.add_inferred_relation(
                self.target.instance,
                self.property_descriptor_cls,
                self.source,
                RelationExplanation(InferredThrough.SYMMETRY, (self,)),
            )

    def infer_transitive_relations(self):
        """
        Infer the relations composing this relation with the relations of its property before and after it, if the
        property is transitive.
        """
        if not self.transitive:
            return
        for relation in list(self.target_outgoing_relations_with_same_descriptor_type):
            self.add_inferred_relation(
                self.source.instance,
                self.property_descriptor_cls,
                relation.target,
                RelationExplanation(InferredThrough.TRANSITIVE, (self, relation)),
            )
        for relation in list(self.source_incoming_relations_with_same_descriptor_type):
            self.add_inferred_relation(
                relation.source.instance,
                self.property_descriptor_cls,
                self.target,
                RelationExplanation(InferredThrough.TRANSITIVE, (relation, self)),
            )

    def infer_chained_relations(self):
        """
        Infer the relations implied by every property chain this relation completes, together with the relations
        before and after it in the graph.
        """
        for chain_position in self.property_descriptor_cls.chain_positions():
            for path_before in self.paths_to(
                self.source, chain_position.descriptors_before
            ):
                for path_after in self.paths_from(
                    self.target, chain_position.descriptors_after
                ):
                    self.add_inferred_relation(
                        path_before.first.instance,
                        chain_position.implied_descriptor,
                        path_after.last,
                        RelationExplanation(
                            InferredThrough.CHAIN,
                            (*path_before.relations, self, *path_after.relations),
                        ),
                    )

    @classmethod
    def paths_to(
        cls,
        end: WrappedInstance,
        descriptor_classes: tuple[type[PropertyDescriptor], ...],
    ) -> Iterator[RelationPath]:
        """
        :param end: The node the paths end in.
        :param descriptor_classes: The descriptor classes of the relations of a path, in order.
        :return: Every path of relations of the descriptor classes ending in the node.
        """
        if not descriptor_classes:
            yield RelationPath(end, end)
            return
        *before, last = descriptor_classes
        for relation in list(cls.incoming_relations_of_descriptor(end, last)):
            for path in cls.paths_to(relation.source, tuple(before)):
                yield RelationPath(path.first, end, (*path.relations, relation))

    @classmethod
    def paths_from(
        cls,
        start: WrappedInstance,
        descriptor_classes: tuple[type[PropertyDescriptor], ...],
    ) -> Iterator[RelationPath]:
        """
        :param start: The node the paths start from.
        :param descriptor_classes: The descriptor classes of the relations of a path, in order.
        :return: Every path of relations of the descriptor classes starting from the node.
        """
        if not descriptor_classes:
            yield RelationPath(start, start)
            return
        first, *after = descriptor_classes
        for relation in list(cls.outgoing_relations_of_descriptor(start, first)):
            for path in cls.paths_from(relation.target, tuple(after)):
                yield RelationPath(start, path.last, (relation, *path.relations))

    # %% relations in the graph

    @staticmethod
    def incoming_relations_of_descriptor(
        node: WrappedInstance, descriptor_class: type[PropertyDescriptor]
    ) -> Iterator[PropertyDescriptorRelation]:
        """
        :return: The relations of the descriptor class, or of a subclass of it, that end in the node.
        """
        yield from SymbolGraph().get_incoming_relations_with_condition(
            node,
            lambda relation: (
                isinstance(relation, PropertyDescriptorRelation)
                and issubclass(relation.property_descriptor_cls, descriptor_class)
            ),
        )

    @staticmethod
    def outgoing_relations_of_descriptor(
        node: WrappedInstance, descriptor_class: type[PropertyDescriptor]
    ) -> Iterator[PropertyDescriptorRelation]:
        """
        :return: The relations of the descriptor class, or of a subclass of it, that start from the node.
        """
        yield from SymbolGraph().get_outgoing_relations_with_condition(
            node,
            lambda relation: (
                isinstance(relation, PropertyDescriptorRelation)
                and issubclass(relation.property_descriptor_cls, descriptor_class)
            ),
        )

    @property
    def target_outgoing_relations_with_same_descriptor_type(
        self,
    ) -> Iterator[PropertyDescriptorRelation]:
        """
        The relations of this relation's property that start from the target.
        """
        yield from SymbolGraph().get_outgoing_relations_with_condition(
            self.target,
            lambda relation: (
                isinstance(relation, PropertyDescriptorRelation)
                and relation.property_descriptor_cls is self.property_descriptor_cls
            ),
        )

    @property
    def source_incoming_relations_with_same_descriptor_type(
        self,
    ) -> Iterator[PropertyDescriptorRelation]:
        """
        The relations of this relation's property that end in the source.
        """
        yield from SymbolGraph().get_incoming_relations_with_condition(
            self.source,
            lambda relation: (
                isinstance(relation, PropertyDescriptorRelation)
                and relation.property_descriptor_cls is self.property_descriptor_cls
            ),
        )

    # %% the property

    @cached_property
    def property_descriptor_cls(self) -> type[PropertyDescriptor]:
        """
        The property descriptor class of the relation.
        """
        return type(self.wrapped_field.property_descriptor)

    @cached_property
    def transitive(self) -> bool:
        """
        Whether the property of the relation is transitive.
        """
        return issubclass(self.property_descriptor_cls, TransitiveProperty)

    @cached_property
    def inverse_of(self) -> type[PropertyDescriptor] | None:
        """
        The inverse of the property of the relation, if it has one.
        """
        if issubclass(self.property_descriptor_cls, HasInverseProperty):
            return self.property_descriptor_cls.get_inverse()
        return None

    # %% explanations

    @property
    def rule(self) -> InferredThrough | None:
        """
        The rule that inferred this relation, or None if it was asserted.
        """
        return self.explanation.rule if self.explanation else None

    @classmethod
    def find(
        cls, source_instance: Any, field_name: str, target_instance: Any
    ) -> PropertyDescriptorRelation | None:
        """
        :param source_instance: The object that holds the field.
        :param field_name: The name of the field.
        :param target_instance: The value.
        :return: The relation in the symbol graph for the fact; None if there is none.
        """
        source = SymbolGraph().get_wrapped_instance(source_instance)
        if source is None:
            return None
        for relation in SymbolGraph().get_outgoing_relations(source):
            if (
                relation.wrapped_field.name == field_name
                and relation.target.instance is target_instance
            ):
                return relation
        return None
