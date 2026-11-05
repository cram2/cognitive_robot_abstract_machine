"""
Relations between instances held by property descriptors, what each one implies, and how an implied one was derived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property

from typing_extensions import TYPE_CHECKING, Any, ClassVar, Iterator

from krrood.class_diagrams.wrapped_field import WrappedField
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
    SYMMETRIC_TRANSITIVE_COMPONENT = "symmetric_transitive_component"
    """
    A component of the relation graph of a symmetric and transitive property. Premises: the property's name, the
    component's identifier and size.
    """
    HAS_VALUE = "has_value"
    """
    The subject belongs to a class that is a ``hasValue`` restriction (OWL 2 RL rule cls-hv1). Premise: the class.
    """


@dataclass(eq=False)
class SymmetricTransitiveComponent:
    """
    A weakly connected component of the relation graph of a property that is both symmetric and transitive: the
    property relates every two members, and each member to itself unless it is irreflexive.
    """

    property_descriptor_class: type[PropertyDescriptor]
    """
    The symmetric and transitive property.
    """
    identifier: int
    """
    The index of the component among the components of the property.
    """
    members: dict[int, WrappedInstance]
    """
    The members of the component by their index in the symbol graph.
    """
    reflexive: bool
    """
    Whether the property relates each member to itself.
    """

    def relates(self, source: WrappedInstance, target: WrappedInstance) -> bool:
        """
        :return: Whether the component entails that the property relates the source to the target.
        """
        return (
            self.members.get(source.index) is source
            and self.members.get(target.index) is target
            and (self.reflexive or source is not target)
        )

    def relation(
        self, source: WrappedInstance, target: WrappedInstance
    ) -> PropertyDescriptorRelation:
        """
        :return: The relation from the source to the target the component explains, not added to the symbol graph.
        """
        wrapped_field = (
            self.property_descriptor_class.get_descriptor_instance_for_domain_type(
                source.instance_type
            ).wrapped_field
        )
        return PropertyDescriptorRelation(
            source,
            target,
            wrapped_field,
            inferred=True,
            inference_explanation=(
                InferredThrough.SYMMETRIC_TRANSITIVE_COMPONENT,
                self.property_descriptor_class.__name__,
                self.identifier,
                len(self.members),
            ),
        )


# %% relations


@dataclass(eq=False, repr=False)
class PropertyDescriptorRelation(PredicateClassRelation):
    """
    Edge data representing a relation between two wrapped instances that is represented structurally by a property
    descriptor attached to the source instance.

    An individual is represented by its root role taker and the roles of it; a fact about the individual is stored on
    the first of these objects whose class declares the property (see :meth:`declaring_object`).
    """

    inference_explanation: tuple[Any, ...] | None = field(
        default=None, compare=False, hash=False
    )
    """
    How an inferred relation was derived: the :class:`InferredThrough` rule followed by its premises. Asserted
    relations have none.
    """

    eager_symmetric_transitive_closure: ClassVar[bool] = False
    """
    Whether relations of symmetric and transitive properties are closed relation by relation while they are added,
    instead of once after loading from the components of their relation graph.
    """

    symmetric_transitive_components: ClassVar[
        dict[tuple[str, int], SymmetricTransitiveComponent]
    ] = {}
    """
    The components whose facts are stored only in the fields of their members, by the field name and the index of
    each member. :meth:`find` explains these facts with them.
    """

    same_individual_role_takers: ClassVar[dict[Any, list[Any]]] = {}
    """
    For a root role taker, the other root role takers that represent the same individual, for an individual one of
    whose classes its root's class cannot take as a role.
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
            of the root in the order they were created, then the same for every linked root (see
            :attr:`same_individual_role_takers`).
        """
        root = cls.root_role_taker(instance)
        objects = []
        seen_identities = set()
        for linked_root in [root, *cls.same_individual_role_takers.get(root, ())]:
            for individual_object in [linked_root, *Role.roles_for(linked_root)]:
                if id(individual_object) not in seen_identities:
                    seen_identities.add(id(individual_object))
                    objects.append(individual_object)
        return objects

    @classmethod
    def nodes_of_individual(cls, node: WrappedInstance) -> list[WrappedInstance]:
        """
        :return: The node and the nodes of the roots linked to it in :attr:`same_individual_role_takers`.
        """
        linked_nodes = (
            SymbolGraph().get_wrapped_instance(root)
            for root in cls.same_individual_role_takers.get(node.instance, ())
        )
        return [node, *(linked for linked in linked_nodes if linked is not None)]

    @classmethod
    def declaring_object(
        cls,
        instance: Any,
        property_descriptor_class: type[PropertyDescriptor],
        value: Any,
    ) -> tuple[Any, PropertyDescriptor] | None:
        """
        Find the object of the individual of the instance on which a fact of the property with the value is stored:
        the first object, in the order of :meth:`objects_of_individual`, whose most specific class declaring the
        property admits the value; else the first object declaring the property.

        :return: The object and the descriptor of its declaration, or None if no object of the individual declares
            the property.
        """
        descriptors = property_descriptor_class.descriptor_instances_by_domain_type[
            property_descriptor_class
        ]
        value_objects = cls.objects_of_individual(value)
        first_declaration = None
        for individual_object in cls.objects_of_individual(instance):
            descriptor = next(
                (
                    descriptors[object_type]
                    for object_type in type(individual_object).__mro__
                    if object_type in descriptors
                ),
                None,
            )
            if descriptor is None:
                continue
            if any(
                isinstance(value_object, descriptor.range)
                for value_object in value_objects
            ):
                return individual_object, descriptor
            first_declaration = first_declaration or (individual_object, descriptor)
        return first_declaration

    @classmethod
    def holder_of_property(
        cls,
        instance: Any,
        property_descriptor_class: type[PropertyDescriptor],
        value: Any,
    ) -> tuple[WrappedInstance, WrappedField] | None:
        """
        :return: The node of the object a fact of the property about the individual of the instance is stored on (see
            :meth:`declaring_object`) and the wrapped field of its declaration, or None if no object declares the
            property.
        """
        declaration = cls.declaring_object(instance, property_descriptor_class, value)
        if declaration is None:
            return None
        individual_object, descriptor = declaration
        node = SymbolGraph().ensure_wrapped_instance(individual_object)
        return node, descriptor.wrapped_field

    # %% adding relations

    def update_source_and_add_to_graph_and_apply_implications(self):
        """
        Update the source wrapped-field value, add this relation to the graph, and apply all implications of adding
        this relation.
        """
        if not self.update_source():
            return
        self.add_to_graph_and_apply_implications()

    def update_source_and_add_to_graph(self):
        """
        Update the source wrapped-field value and add this relation to the graph.
        """
        if not self.update_source():
            return
        self.add_to_graph()

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
        holder, descriptor = declaration
        return descriptor.update_value(holder, self.target.instance)

    @classmethod
    def add_inferred_relation(
        cls,
        source_instance: Any,
        property_descriptor_class: type[PropertyDescriptor],
        target: WrappedInstance,
        inference_explanation: tuple[Any, ...],
    ):
        """
        Store an inferred fact of the property about the individual of the source instance on the object of the
        individual that declares the property, add it to the graph and apply its implications. Nothing is stored if
        no object of the individual declares the property.
        """
        holder = cls.holder_of_property(
            source_instance, property_descriptor_class, target.instance
        )
        if holder is None:
            return
        source, wrapped_field = holder
        cls(
            source,
            target,
            wrapped_field,
            inferred=True,
            inference_explanation=inference_explanation,
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
                (InferredThrough.EQUIVALENT, self),
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
                (InferredThrough.SUPER, self),
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
                (InferredThrough.INVERSE, self),
            )

    def infer_symmetric_relation(self):
        """
        Infer the relation back from the target to the source if the property is symmetric.
        """
        if self.rule in (
            InferredThrough.SYMMETRY,
            InferredThrough.SYMMETRIC_TRANSITIVE_COMPONENT,
        ):
            return
        if issubclass(self.property_descriptor_cls, SymmetricProperty):
            self.add_inferred_relation(
                self.target.instance,
                self.property_descriptor_cls,
                self.source,
                (InferredThrough.SYMMETRY, self),
            )

    def infer_transitive_relations(self):
        """
        Infer the relations composing this relation with the relations of its property before and after it, if the
        property is transitive. Symmetric and transitive properties are closed after loading instead, unless
        :attr:`eager_symmetric_transitive_closure` is set.
        """
        if self.rule is InferredThrough.EQUIVALENT or not self.transitive:
            return
        if (
            issubclass(self.property_descriptor_cls, SymmetricProperty)
            and not self.eager_symmetric_transitive_closure
        ):
            return
        for relation in list(self.target_outgoing_relations_with_same_descriptor_type):
            self.__class__(
                self.source,
                relation.target,
                self.wrapped_field,
                inferred=True,
                inference_explanation=(InferredThrough.TRANSITIVE, self, relation),
            ).update_source_and_add_to_graph_and_apply_implications()
        for relation in list(self.source_incoming_relations_with_same_descriptor_type):
            self.__class__(
                relation.source,
                self.target,
                self.wrapped_field,
                inferred=True,
                inference_explanation=(InferredThrough.TRANSITIVE, relation, self),
            ).update_source_and_add_to_graph_and_apply_implications()

    def infer_chained_relations(self):
        """
        Infer the relations implied by every property chain this relation completes, together with the relations
        before and after it in the graph.
        """
        chain_positions = self.property_descriptor_cls.chains_with_descriptor[
            self.property_descriptor_cls
        ]
        for chain_position in list(chain_positions):
            for chain_start, relations_before in self.paths_to(
                self.source, chain_position.descriptors_before
            ):
                for chain_end, relations_after in self.paths_from(
                    self.target, chain_position.descriptors_after
                ):
                    self.add_inferred_relation(
                        chain_start.instance,
                        chain_position.implied_descriptor,
                        chain_end,
                        (
                            InferredThrough.CHAIN,
                            *relations_before,
                            self,
                            *relations_after,
                        ),
                    )

    @classmethod
    def paths_to(
        cls,
        end: WrappedInstance,
        descriptor_classes: tuple[type[PropertyDescriptor], ...],
    ) -> Iterator[tuple[WrappedInstance, tuple[PropertyDescriptorRelation, ...]]]:
        """
        :param end: The node the paths end in, or a node of its individual.
        :param descriptor_classes: The descriptor classes of the relations of a path, in order.
        :return: The first node and the relations of every path of relations of the descriptor classes.
        """
        if not descriptor_classes:
            yield end, ()
            return
        *before, last = descriptor_classes
        for node in cls.nodes_of_individual(end):
            for relation in list(cls.incoming_relations_of_descriptor(node, last)):
                for start, path in cls.paths_to(relation.source, tuple(before)):
                    yield start, (*path, relation)

    @classmethod
    def paths_from(
        cls,
        start: WrappedInstance,
        descriptor_classes: tuple[type[PropertyDescriptor], ...],
    ) -> Iterator[tuple[WrappedInstance, tuple[PropertyDescriptorRelation, ...]]]:
        """
        :param start: The node the paths start from, or a node of its individual.
        :param descriptor_classes: The descriptor classes of the relations of a path, in order.
        :return: The last node and the relations of every path of relations of the descriptor classes.
        """
        if not descriptor_classes:
            yield start, ()
            return
        first, *after = descriptor_classes
        for node in cls.nodes_of_individual(start):
            for relation in list(cls.outgoing_relations_of_descriptor(node, first)):
                for end, path in cls.paths_from(relation.target, tuple(after)):
                    yield end, (relation, *path)

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
        The relations of this relation's property that start from the target's individual.
        """
        for node in self.nodes_of_individual(self.target):
            yield from SymbolGraph().get_outgoing_relations_with_condition(
                node,
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
        The relations of this relation's property that end in the source's individual.
        """
        for node in self.nodes_of_individual(self.source):
            yield from SymbolGraph().get_incoming_relations_with_condition(
                node,
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
        return self.inference_explanation[0] if self.inference_explanation else None

    @property
    def premises(self) -> tuple[Any, ...]:
        """
        The premises of the rule that inferred this relation.
        """
        return (
            tuple(self.inference_explanation[1:]) if self.inference_explanation else ()
        )

    def explain(self, depth: int = 3) -> dict[str, Any]:
        """
        :param depth: How many levels of premises to expand.
        :return: The fact (source, field, target), the rule (None for asserted relations) and the explanations of the
            premises.
        """
        explanation: dict[str, Any] = {
            "fact": (
                self.identifier_of(self.source.instance),
                self.wrapped_field.name,
                self.identifier_of(self.target.instance),
            ),
            "rule": self.rule.value if self.rule else None,
        }
        if self.inference_explanation and depth > 0:
            explanation["premises"] = [
                (
                    premise.explain(depth - 1)
                    if isinstance(premise, PropertyDescriptorRelation)
                    else premise
                )
                for premise in self.premises
            ]
        return explanation

    @staticmethod
    def identifier_of(instance: Any) -> Any:
        """
        :return: The IRI of an instance of a generated class, or else the instance.
        """
        return vars(instance).get("uri", instance)

    @classmethod
    def find(
        cls, source_instance: Any, field_name: str, target_instance: Any
    ) -> PropertyDescriptorRelation | None:
        """
        :param source_instance: The object that holds the field.
        :param field_name: The name of the field.
        :param target_instance: The value.
        :return: The relation in the symbol graph for the fact, or the relation a symmetric and transitive component
            entails; None if neither exists.
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
        component = cls.symmetric_transitive_components.get((field_name, source.index))
        target = SymbolGraph().get_wrapped_instance(target_instance)
        if (
            component is not None
            and target is not None
            and component.relates(source, target)
        ):
            return component.relation(source, target)
        return None
