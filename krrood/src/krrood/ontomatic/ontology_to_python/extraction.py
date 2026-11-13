"""
Reading classes and properties, with their labels and comments, out of an RDF graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property


import rdflib
from rdflib.namespace import RDF, RDFS, OWL
from typing_extensions import Any, Dict, List, Optional

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
class ClassExtractor:
    """
    Extractor for OWL Class information from an RDF graph.
    """

    graph: rdflib.Graph
    """
    The graph the classes are read from.
    """

    metadata_extractor: MetadataExtractor
    """
    Reads the labels and comments of the classes.
    """

    def extract(self, *, class_uri: Any) -> ClassInfo:
        """
        Extract information about a class.
        """
        is_blank_node = False
        if isinstance(class_uri, rdflib.BNode):
            is_blank_node = True
            class_name = NamingRegistry.bnode_to_name(graph=self.graph, bnode=class_uri)
        else:
            class_name = NamingRegistry.uri_to_python_name(uri=class_uri)

        # Get superclasses from explicit rdfs:subClassOf
        superclasses: List[str] = []
        has_descriptions = []
        for superclass in self.graph.objects(class_uri, RDFS.subClassOf):
            if isinstance(superclass, rdflib.URIRef):
                superclasses.append(NamingRegistry.uri_to_python_name(uri=superclass))
            elif isinstance(superclass, rdflib.BNode):
                # Only include BNode superclasses if they are intersections or unions
                if list(self.graph.objects(superclass, OWL.intersectionOf)) or list(
                    self.graph.objects(superclass, OWL.unionOf)
                ):
                    has_descriptions.append(
                        NamingRegistry.uri_to_python_name(
                            uri=superclass, graph=self.graph
                        )
                    )
        if not is_blank_node:
            superclasses.extend(
                self.named_conjuncts_of_definitions(class_uri=class_uri)
            )

        # De-duplicate while preserving order
        seen = set()
        unique_superclasses: List[str] = []
        for superclass_name in superclasses:
            if superclass_name not in seen:
                unique_superclasses.append(superclass_name)
                seen.add(superclass_name)

        # disjoint with
        disjoint_with: List[str] = []
        for disjoint_class in self.graph.objects(class_uri, OWL.disjointWith):
            if isinstance(disjoint_class, rdflib.URIRef):
                disjoint_with.append(
                    NamingRegistry.uri_to_python_name(uri=disjoint_class)
                )

        # equivalent classes
        equivalent_classes: List[str] = []
        for equivalent_class in self.graph.objects(class_uri, OWL.equivalentClass):
            if isinstance(equivalent_class, rdflib.URIRef):
                equivalent_classes.append(
                    NamingRegistry.uri_to_python_name(uri=equivalent_class)
                )

        # Get label
        label = self.metadata_extractor.get_label(uri=class_uri)
        is_description_for = None
        if is_blank_node and unique_superclasses:
            if len(unique_superclasses) > 1:
                raise NotImplemented(
                    f"BNode class {class_name} has multiple superclasses: {unique_superclasses}"
                )
            is_description_for = unique_superclasses[0]
        elif is_blank_node and not unique_superclasses:
            subclasses = list(self.graph.subjects(RDFS.subClassOf, class_uri))
            if len(subclasses) > 1:
                raise NotImplemented(
                    f"BNode class {class_name} has multiple subclasses: {subclasses}"
                )
            if subclasses:
                is_description_for = NamingRegistry.uri_to_python_name(
                    uri=subclasses[0]
                )
            else:
                is_description_for = self.defined_class(class_expression=class_uri)
            if is_description_for is None:
                unique_superclasses = self.named_conjuncts_of(
                    class_expression=class_uri
                )

        return ClassInfo(
            name=class_name,
            uri=str(class_uri),
            superclasses=unique_superclasses or [Symbol.__name__],
            disjoint_with=disjoint_with,
            equivalent_classes=equivalent_classes,
            is_description_for=is_description_for,
            has_descriptions=has_descriptions,
            label=label,
            comment=self.metadata_extractor.get_comment(uri=class_uri),
            add_role_taker=True,
        )

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
class PropertyExtractor:
    """
    Extractor for OWL Property information from an RDF graph.
    """

    graph: rdflib.Graph
    """
    The graph the properties are read from.
    """

    metadata_extractor: MetadataExtractor
    """
    Reads the labels and comments of the properties.
    """

    def extract(self, *, property_uri: Any) -> PropertyInfo:
        """
        Extract information about a property.
        """
        property_name = NamingRegistry.uri_to_python_name(uri=property_uri)

        # Get domain and range
        domains: List[str] = []
        ranges: List[str] = []
        superproperties: List[str] = []
        inverses: List[str] = []
        is_transitive = False
        equivalent_properties: List[str] = []
        disjoint_properties: List[str] = []
        is_symmetric = False
        is_asymmetric = False
        is_reflexive = False
        is_irreflexive = False
        is_functional = False

        for domain in self.graph.objects(property_uri, RDFS.domain):
            domains.append(NamingRegistry.uri_to_python_name(uri=domain))

        range_uris: List[rdflib.term.Identifier] = []
        for range_class in self.graph.objects(property_uri, RDFS.range):
            ranges.append(NamingRegistry.uri_to_python_name(uri=range_class))
            range_uris.append(range_class)

        # Inheritance between properties
        for superproperty in self.graph.objects(property_uri, RDFS.subPropertyOf):
            if isinstance(superproperty, rdflib.URIRef):
                superproperties.append(
                    NamingRegistry.uri_to_python_name(uri=superproperty)
                )

        # Inverses
        for inverse_property in self.graph.objects(property_uri, OWL.inverseOf):
            if isinstance(inverse_property, rdflib.URIRef):
                inverses.append(NamingRegistry.uri_to_python_name(uri=inverse_property))
        # Also collect when current property is the object of inverseOf
        for inverse_property in self.graph.subjects(OWL.inverseOf, property_uri):
            if isinstance(inverse_property, rdflib.URIRef):
                inverses.append(NamingRegistry.uri_to_python_name(uri=inverse_property))

        # Equivalent properties
        for equivalent_property in self.graph.objects(
            property_uri, OWL.equivalentProperty
        ):
            if isinstance(equivalent_property, rdflib.URIRef):
                equivalent_properties.append(
                    NamingRegistry.uri_to_python_name(uri=equivalent_property)
                )
        for equivalent_property in self.graph.subjects(
            OWL.equivalentProperty, property_uri
        ):
            if isinstance(equivalent_property, rdflib.URIRef):
                equivalent_properties.append(
                    NamingRegistry.uri_to_python_name(uri=equivalent_property)
                )

        # Disjoint properties
        for disjoint_property in self.graph.objects(
            property_uri, OWL.propertyDisjointWith
        ):
            if isinstance(disjoint_property, rdflib.URIRef):
                disjoint_properties.append(
                    NamingRegistry.uri_to_python_name(uri=disjoint_property)
                )
        for disjoint_property in self.graph.subjects(
            OWL.propertyDisjointWith, property_uri
        ):
            if isinstance(disjoint_property, rdflib.URIRef):
                disjoint_properties.append(
                    NamingRegistry.uri_to_python_name(uri=disjoint_property)
                )

        # Determine property type
        property_type = PropertyType.OBJECT_PROPERTY
        for rdf_type in self.graph.objects(property_uri, RDF.type):
            if rdf_type == OWL.DatatypeProperty:
                property_type = PropertyType.DATA_PROPERTY
            if rdf_type == OWL.TransitiveProperty:
                is_transitive = True
            if rdf_type == OWL.SymmetricProperty:
                is_symmetric = True
            if rdf_type == OWL.AsymmetricProperty:
                is_asymmetric = True
            if rdf_type == OWL.ReflexiveProperty:
                is_reflexive = True
            if rdf_type == OWL.IrreflexiveProperty:
                is_irreflexive = True
            if rdf_type == OWL.FunctionalProperty:
                is_functional = True

        # Choose a single inverse if any (stable order)
        inverse_of = None
        if inverses:
            inverse_of = sorted(set(inverses))[0]

        return PropertyInfo(
            name=property_name,
            uri=str(property_uri),
            type=property_type,
            domains=domains,
            ranges=ranges,
            range_uris=range_uris,
            equivalent_properties=equivalent_properties,
            disjoint_properties=disjoint_properties,
            label=self.metadata_extractor.get_label(uri=property_uri),
            comment=self.metadata_extractor.get_comment(uri=property_uri),
            field_name=NamingRegistry.to_snake_case(name=property_name),
            descriptor_name=NamingRegistry.to_pascal_case(name=property_name),
            superproperties=superproperties,
            inverses=sorted(set(inverses)),
            inverse_of=inverse_of,
            is_transitive=is_transitive,
            is_symmetric=is_symmetric,
            is_asymmetric=is_asymmetric,
            is_reflexive=is_reflexive,
            is_irreflexive=is_irreflexive,
            is_functional=is_functional,
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
