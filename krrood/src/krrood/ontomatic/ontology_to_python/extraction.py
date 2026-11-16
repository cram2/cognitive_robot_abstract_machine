"""
Reading classes and properties, with their labels and comments, out of an RDF graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property


import rdflib
from rdflib.namespace import RDF, RDFS, OWL
from typing_extensions import Any, Dict, List, Optional

from krrood.ontomatic.failures import AmbiguousDescribedClass
from krrood.ontomatic.naming import NamingRegistry
from krrood.ontomatic.ontology_to_python.property_type import PropertyType
from krrood.symbol_graph.symbol_graph import Symbol

from krrood.ontomatic.ontology_to_python.ontology_info import (
    ClassInfo,
    OntologyInfo,
    PropertyInfo,
)


@dataclass(kw_only=True)
class MetadataExtractor:
    """
    Helper for extracting metadata (labels, comments) from an RDF graph.
    """

    graph: rdflib.Graph
    """
    The graph whose labels and comments are read.
    """

    def get_label(self, *, uri: Any) -> Optional[str]:
        """
        Get rdfs:label for a URI.
        """
        for label in self.graph.objects(uri, RDFS.label):
            return str(label)
        return None

    def get_comment(self, *, uri: Any) -> Optional[str]:
        """
        Get rdfs:comment for a URI.
        """
        for comment in self.graph.objects(uri, RDFS.comment):
            return str(comment)
        return None


@dataclass(kw_only=True)
class GraphExtractor:
    """
    Reads the information about the classes or properties of an RDF graph.
    """

    graph: rdflib.Graph
    """
    The graph the information is read from.
    """

    metadata_extractor: MetadataExtractor
    """
    Reads the labels and comments.
    """

    def related_names(
        self, *, node: rdflib.URIRef | rdflib.BNode, relation: rdflib.URIRef
    ) -> list[str]:
        """
        :param node: The class or property the relation is stated on.
        :param relation: The relation, such as ``owl:disjointWith``.
        :return: The names of the named classes or properties the node is related to.
        """
        return [
            NamingRegistry.uri_to_python_name(uri=related)
            for related in self.graph.objects(node, relation)
            if isinstance(related, rdflib.URIRef)
        ]

    def mutually_related_names(
        self, *, node: rdflib.URIRef | rdflib.BNode, relation: rdflib.URIRef
    ) -> list[str]:
        """
        :param node: A class or property.
        :param relation: A symmetric relation, such as ``owl:equivalentProperty``.
        :return: The names of the named classes or properties the relation is stated
            between the node and, in either direction.
        """
        return self.related_names(node=node, relation=relation) + [
            NamingRegistry.uri_to_python_name(uri=related)
            for related in self.graph.subjects(relation, node)
            if isinstance(related, rdflib.URIRef)
        ]


@dataclass(kw_only=True)
class ClassExtractor(GraphExtractor):
    """
    Extractor for OWL Class information from an RDF graph.
    """

    def extract(self, *, class_uri: rdflib.URIRef | rdflib.BNode) -> ClassInfo:
        """
        :param class_uri: A named class or an anonymous class expression.
        :return: The information about the class. An anonymous class expression
            describing no named class is a subclass of the named classes it intersects.
        """
        superclasses = self.named_superclasses(class_uri=class_uri)
        described_class = None
        if isinstance(class_uri, rdflib.BNode):
            name = NamingRegistry.bnode_to_name(graph=self.graph, bnode=class_uri)
            described_class = self.described_class(class_expression=class_uri)
            if described_class is None:
                superclasses = self.named_conjuncts_of(class_expression=class_uri)
        else:
            name = NamingRegistry.uri_to_python_name(uri=class_uri)
        return ClassInfo(
            name=name,
            uri=str(class_uri),
            superclasses=superclasses or [Symbol.__name__],
            disjoint_with=self.related_names(node=class_uri, relation=OWL.disjointWith),
            equivalent_classes=self.related_names(
                node=class_uri, relation=OWL.equivalentClass
            ),
            is_description_for=described_class,
            has_descriptions=self.description_names(class_uri=class_uri),
            label=self.metadata_extractor.get_label(uri=class_uri),
            comment=self.metadata_extractor.get_comment(uri=class_uri),
            add_role_taker=True,
        )

    def named_superclasses(
        self, *, class_uri: rdflib.URIRef | rdflib.BNode
    ) -> list[str]:
        """
        :param class_uri: A named class or an anonymous class expression.
        :return: The names of the named classes the class is declared under, each once,
            and for a named class those its definitions intersect.
        """
        superclasses = self.related_names(node=class_uri, relation=RDFS.subClassOf)
        if isinstance(class_uri, rdflib.URIRef):
            superclasses.extend(
                self.named_conjuncts_of_definitions(class_uri=class_uri)
            )
        return list(dict.fromkeys(superclasses))

    def description_names(
        self, *, class_uri: rdflib.URIRef | rdflib.BNode
    ) -> list[str]:
        """
        :param class_uri: A named class or an anonymous class expression.
        :return: The names of the anonymous intersections and unions the class is
            declared under.
        """
        return [
            NamingRegistry.uri_to_python_name(uri=superclass, graph=self.graph)
            for superclass in self.graph.objects(class_uri, RDFS.subClassOf)
            if isinstance(superclass, rdflib.BNode)
            and (
                (superclass, OWL.intersectionOf, None) in self.graph
                or (superclass, OWL.unionOf, None) in self.graph
            )
        ]

    def described_class(self, *, class_expression: rdflib.BNode) -> str | None:
        """
        :param class_expression: An anonymous class expression.
        :return: The name of the named class the expression describes: the one it is
            declared under, else the one declared under it, else the one it defines.
        :raises AmbiguousDescribedClass: When it is declared under, or over, several.
        """
        candidates = self.named_superclasses(class_uri=class_expression) or [
            NamingRegistry.uri_to_python_name(uri=subclass)
            for subclass in self.graph.subjects(RDFS.subClassOf, class_expression)
        ]
        if len(candidates) > 1:
            raise AmbiguousDescribedClass(
                class_expression_name=NamingRegistry.bnode_to_name(
                    graph=self.graph, bnode=class_expression
                ),
                candidates=candidates,
            )
        if candidates:
            return candidates[0]
        return self.defined_class(class_expression=class_expression)

    def defined_class(self, *, class_expression: rdflib.BNode) -> Optional[str]:
        """
        :param class_expression: An anonymous class expression.
        :return: The name of the named class the expression is the ``owl:equivalentClass`` of, or None when it is
            the definition of no named class.
        """
        for defined in [
            *self.graph.subjects(OWL.equivalentClass, class_expression),
            *self.graph.objects(class_expression, OWL.equivalentClass),
        ]:
            if isinstance(defined, rdflib.URIRef):
                return NamingRegistry.uri_to_python_name(uri=defined)
        return None

    def is_attached_to_a_named_class(self, *, class_expression: rdflib.BNode) -> bool:
        """
        :param class_expression: An anonymous class expression.
        :return: Whether the expression describes a named class: as its superclass, its subclass or its definition.
            An expression that only appears inside another one, such as the filler of a restriction, describes none.
        """
        return (
            any(self.graph.objects(class_expression, RDFS.subClassOf))
            or any(self.graph.subjects(RDFS.subClassOf, class_expression))
            or self.defined_class(class_expression=class_expression) is not None
        )

    def named_class_defined_alike(
        self, *, class_expression: rdflib.BNode
    ) -> Optional[str]:
        """
        :param class_expression: An anonymous intersection.
        :return: The name of the named class defined by another intersection of the same members, which the expression
            therefore stands for, or None.
        """
        name = NamingRegistry.uri_to_python_name(uri=class_expression, graph=self.graph)
        for definition in self.graph.subjects(OWL.intersectionOf, None):
            if (
                definition != class_expression
                and NamingRegistry.uri_to_python_name(uri=definition, graph=self.graph)
                == name
            ):
                defined_class = self.defined_class(class_expression=definition)
                if defined_class is not None:
                    return defined_class
        return None

    def named_conjuncts_of(self, *, class_expression: rdflib.BNode) -> List[str]:
        """
        :param class_expression: An anonymous intersection.
        :return: The names of the named classes among its members.
        """
        return [
            NamingRegistry.uri_to_python_name(uri=member)
            for intersection in self.graph.objects(class_expression, OWL.intersectionOf)
            for member in NamingRegistry._get_rdf_list(
                graph=self.graph, head=intersection
            )
            if isinstance(member, rdflib.URIRef)
        ]

    def named_conjuncts_of_definitions(self, *, class_uri: rdflib.URIRef) -> List[str]:
        """
        :param class_uri: A named class.
        :return: The named classes of every intersection the class is defined by (``owl:equivalentClass``), each of
            which is a superclass of the class.
        """
        return [
            conjunct
            for definition in self.graph.objects(class_uri, OWL.equivalentClass)
            for conjunct in self.named_conjuncts_of(class_expression=definition)
        ]


@dataclass(kw_only=True)
class PropertyExtractor(GraphExtractor):
    """
    Extractor for OWL Property information from an RDF graph.
    """

    def extract(self, *, property_uri: rdflib.URIRef) -> PropertyInfo:
        """
        :param property_uri: A named property.
        :return: The information about the property. Of several inverses, the first by
            name is its inverse.
        """
        name = NamingRegistry.uri_to_python_name(uri=property_uri)
        range_nodes = list(self.graph.objects(property_uri, RDFS.range))
        inverses = sorted(
            set(self.mutually_related_names(node=property_uri, relation=OWL.inverseOf))
        )
        types = set(self.graph.objects(property_uri, RDF.type))
        return PropertyInfo(
            name=name,
            uri=str(property_uri),
            type=(
                PropertyType.DATA_PROPERTY
                if OWL.DatatypeProperty in types
                else PropertyType.OBJECT_PROPERTY
            ),
            domains=[
                NamingRegistry.uri_to_python_name(uri=domain)
                for domain in self.graph.objects(property_uri, RDFS.domain)
            ],
            ranges=[
                NamingRegistry.uri_to_python_name(uri=range_node)
                for range_node in range_nodes
            ],
            range_uris=range_nodes,
            equivalent_properties=self.mutually_related_names(
                node=property_uri, relation=OWL.equivalentProperty
            ),
            disjoint_properties=self.mutually_related_names(
                node=property_uri, relation=OWL.propertyDisjointWith
            ),
            label=self.metadata_extractor.get_label(uri=property_uri),
            comment=self.metadata_extractor.get_comment(uri=property_uri),
            field_name=NamingRegistry.to_snake_case(name=name),
            descriptor_name=NamingRegistry.to_pascal_case(name=name),
            superproperties=self.related_names(
                node=property_uri, relation=RDFS.subPropertyOf
            ),
            inverses=inverses,
            inverse_of=inverses[0] if inverses else None,
            is_transitive=OWL.TransitiveProperty in types,
            is_symmetric=OWL.SymmetricProperty in types,
            is_asymmetric=OWL.AsymmetricProperty in types,
            is_reflexive=OWL.ReflexiveProperty in types,
            is_irreflexive=OWL.IrreflexiveProperty in types,
            is_functional=OWL.FunctionalProperty in types,
            is_specialized=False,
        )


@dataclass(kw_only=True)
class OntologyExtractor:
    """
    Reads the information about a whole ontology out of its RDF graph.
    """

    graph: rdflib.Graph
    """
    The graph of the ontology.
    """

    predefined_data_types: Dict[str, Dict[str, str]] = field(default_factory=dict)
    """
    Manual type overrides for properties.
    """

    classes: Dict[str, ClassInfo] = field(default_factory=dict, init=False)
    """
    The named classes read so far, by name.
    """

    class_descriptions: Dict[str, ClassInfo] = field(default_factory=dict, init=False)
    """
    The anonymous classes that describe a named class, by the name of that class.
    """

    properties: Dict[str, PropertyInfo] = field(default_factory=dict, init=False)
    """
    The properties read so far, by name.
    """

    ontology_label: Optional[str] = field(default=None, init=False)
    """
    The label of the ontology.
    """

    @cached_property
    def metadata_extractor(self) -> MetadataExtractor:
        """
        Reads the labels and comments of the graph.
        """
        return MetadataExtractor(graph=self.graph)

    @cached_property
    def class_extractor(self) -> ClassExtractor:
        """
        Reads the classes of the graph.
        """
        return ClassExtractor(
            graph=self.graph, metadata_extractor=self.metadata_extractor
        )

    @cached_property
    def property_extractor(self) -> PropertyExtractor:
        """
        Reads the properties of the graph.
        """
        return PropertyExtractor(
            graph=self.graph, metadata_extractor=self.metadata_extractor
        )

    def extract(self) -> OntologyInfo:
        """
        :return: The information about the ontology: its classes, its class descriptions and its properties.
        """
        for ontology_node in self.graph.subjects(RDF.type, OWL.Ontology):
            self.ontology_label = self.metadata_extractor.get_label(uri=ontology_node)
            if not self.ontology_label:
                self.ontology_label = NamingRegistry.uri_to_python_name(
                    uri=ontology_node
                )
            break

        if not self.ontology_label:
            self.ontology_label = "Thing"

        all_class_subjects = set(self.graph.subjects(RDF.type, OWL.Class))

        for class_node in all_class_subjects:
            # Anonymous unions inside other expressions are checked against their members, and anonymous intersections
            # with the members of a named class's definition stand for that class; neither is a class of its own.
            if isinstance(class_node, rdflib.BNode):
                if not (
                    list(self.graph.objects(class_node, OWL.intersectionOf))
                    or list(self.graph.objects(class_node, OWL.unionOf))
                ):
                    continue
                if not self.class_extractor.is_attached_to_a_named_class(
                    class_expression=class_node
                ) and (
                    list(self.graph.objects(class_node, OWL.unionOf))
                    or self.class_extractor.named_class_defined_alike(
                        class_expression=class_node
                    )
                    is not None
                ):
                    continue

            unions = list(self.graph.objects(class_node, OWL.unionOf))
            if unions:
                # 2. Register the parent class
                class_information = self.class_extractor.extract(class_uri=class_node)
                super_classes = class_information.superclasses[:]

                # 3. Process members and make them inherit from the parent union class
                members = NamingRegistry._get_rdf_list(graph=self.graph, head=unions[0])
                for member in members:
                    member_information = self.class_extractor.extract(class_uri=member)
                    # Ensure member inherits from the union parent
                    for superclass_name in super_classes:
                        if superclass_name not in member_information.superclasses:
                            member_information.superclasses.append(superclass_name)

                    # Update/Register member
                    self._register_class(class_information=member_information)
                continue

            # Handle regular classes and other BNodes (Intersections, Restrictions)
            class_information = self.class_extractor.extract(class_uri=class_node)
            if not class_information:
                continue
            self._register_class(class_information=class_information)

        for property_type in [
            OWL.ObjectProperty,
            OWL.DatatypeProperty,
            OWL.TransitiveProperty,
        ]:
            for property_uri in self.graph.subjects(RDF.type, property_type):
                if isinstance(property_uri, rdflib.term.BNode):
                    continue
                property_information = self.property_extractor.extract(
                    property_uri=property_uri
                )
                if property_information.name in self.properties:
                    existing = self.properties[property_information.name]
                    if property_type == OWL.TransitiveProperty:
                        existing.is_transitive = True
                    existing.domains = sorted(
                        set(existing.domains) | set(property_information.domains)
                    )
                    existing.ranges = sorted(
                        set(existing.ranges) | set(property_information.ranges)
                    )
                    if not existing.inverse_of:
                        existing.inverse_of = property_information.inverse_of
                else:
                    self.properties[property_information.name] = property_information

        return OntologyInfo(
            graph=self.graph,
            classes=self.classes,
            class_descriptions=self.class_descriptions,
            original_properties=self.properties,
            predefined_data_types=self.predefined_data_types,
            ontology_label=self.ontology_label,
        )

    def _register_class(self, *, class_information: ClassInfo):
        """
        Register or update class information.
        """
        if class_information.name in self.classes:
            existing = self.classes[class_information.name]
            # Merge superclasses
            for superclass_name in class_information.superclasses:
                if superclass_name not in existing.superclasses:
                    existing.superclasses.append(superclass_name)

            # Special handling for "Symbol"
            if (
                Symbol.__name__ in existing.superclasses
                and len(existing.superclasses) > 1
            ):
                existing.superclasses.remove(Symbol.__name__)

            # Merge declared properties
            for declared_property in class_information.declared_properties:
                if declared_property not in existing.declared_properties:
                    existing.declared_properties.append(declared_property)

            # Merge other metadata
            if not existing.label:
                existing.label = class_information.label
            if not existing.comment:
                existing.comment = class_information.comment
        elif class_information.is_description_for:
            self.class_descriptions[class_information.is_description_for] = (
                class_information
            )
        else:
            self.classes[class_information.name] = class_information
