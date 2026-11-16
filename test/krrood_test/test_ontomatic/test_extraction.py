"""
Classes and properties read out of an RDF graph (``dataset/ontologies/extraction.ttl``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import rdflib
from rdflib import Namespace
from rdflib.namespace import OWL, RDFS

from krrood.ontomatic.failures import AmbiguousDescribedClass
from krrood.ontomatic.naming import NamingRegistry
from krrood.ontomatic.ontology_to_python.extraction import (
    ClassExtractor,
    MetadataExtractor,
    OntologyExtractor,
    PropertyExtractor,
)
from krrood.ontomatic.ontology_to_python.property_type import PropertyType
from krrood.symbol_graph.symbol_graph import Symbol


@pytest.fixture(scope="module")
def extraction() -> Namespace:
    """
    The namespace of the ontology the tests read.
    """
    return Namespace("http://example.org/extraction#")


@pytest.fixture(scope="module")
def graph() -> rdflib.Graph:
    """
    The graph of the ontology.
    """
    graph = rdflib.Graph()
    graph.parse(
        Path(__file__).parent.parent / "dataset" / "ontologies" / "extraction.ttl"
    )
    return graph


@pytest.fixture(scope="module")
def class_extractor(graph: rdflib.Graph) -> ClassExtractor:
    """
    Reads the classes of the ontology.
    """
    return ClassExtractor(
        graph=graph, metadata_extractor=MetadataExtractor(graph=graph)
    )


@pytest.fixture(scope="module")
def property_extractor(graph: rdflib.Graph) -> PropertyExtractor:
    """
    Reads the properties of the ontology.
    """
    return PropertyExtractor(
        graph=graph, metadata_extractor=MetadataExtractor(graph=graph)
    )


# %% classes


def test_a_class_records_its_named_superclass(class_extractor, extraction):
    """
    A subclass records the class it is declared a subclass of.
    """
    assert class_extractor.extract(class_uri=extraction.Student).superclasses == [
        "Person"
    ]


def test_a_class_without_a_superclass_derives_from_symbol(class_extractor, extraction):
    """
    A class declared under no other class gets the symbol class as its superclass.
    """
    assert class_extractor.extract(class_uri=extraction.Person).superclasses == [
        Symbol.__name__
    ]


def test_a_class_keeps_its_label_and_comment(class_extractor, extraction):
    """
    The label and the comment of a class are kept.
    """
    person = class_extractor.extract(class_uri=extraction.Person)
    assert (person.label, person.comment) == ("person", "A human being.")


# %% class expressions


@pytest.fixture(scope="module")
def class_expressions() -> Namespace:
    """
    The namespace of the ontology of class expressions the tests read.
    """
    return Namespace("http://example.org/class_expressions_extraction#")


@pytest.fixture(scope="module")
def class_expressions_graph() -> rdflib.Graph:
    """
    The graph of the ontology of class expressions.
    """
    graph = rdflib.Graph()
    graph.parse(
        Path(__file__).parent.parent
        / "dataset"
        / "ontologies"
        / "class_expressions_extraction.ttl"
    )
    return graph


@pytest.fixture(scope="module")
def class_expressions_extractor(
    class_expressions_graph: rdflib.Graph,
) -> ClassExtractor:
    """
    Reads the classes of the ontology of class expressions.
    """
    return ClassExtractor(
        graph=class_expressions_graph,
        metadata_extractor=MetadataExtractor(graph=class_expressions_graph),
    )


def test_a_class_records_the_named_classes_it_is_disjoint_with(
    class_expressions_extractor, class_expressions
):
    """
    A class records the named classes it is declared disjoint with.
    """
    assert class_expressions_extractor.extract(
        class_uri=class_expressions.Plant
    ).disjoint_with == ["Animal"]


def test_a_class_records_the_named_classes_it_is_equivalent_to(
    class_expressions_extractor, class_expressions
):
    """
    A class records the named classes it is declared equivalent to.
    """
    assert class_expressions_extractor.extract(
        class_uri=class_expressions.Creature
    ).equivalent_classes == ["Animal"]


def test_the_named_classes_a_definition_intersects_are_superclasses(
    class_expressions_extractor, class_expressions
):
    """
    A class defined by an intersection is a subclass of the named classes in it.
    """
    assert class_expressions_extractor.extract(
        class_uri=class_expressions.Cat
    ).superclasses == ["Animal", "Pet"]


def test_a_superclass_both_declared_and_intersected_is_recorded_once(
    class_expressions_extractor, class_expressions
):
    """
    A class declared under a class its definition also intersects records that
    superclass once.
    """
    assert class_expressions_extractor.extract(
        class_uri=class_expressions.Kitten
    ).superclasses == ["Animal", "Pet"]


def test_an_anonymous_union_superclass_is_a_description_of_the_class(
    class_expressions_extractor, class_expressions, class_expressions_graph
):
    """
    A class declared under an anonymous union records the union as a description.
    """
    union = class_expressions_graph.value(class_expressions.Tiger, RDFS.subClassOf)
    assert class_expressions_extractor.extract(
        class_uri=class_expressions.Tiger
    ).has_descriptions == [
        NamingRegistry.uri_to_python_name(uri=union, graph=class_expressions_graph)
    ]


def test_an_expression_under_a_named_class_describes_it(
    class_expressions_extractor, class_expressions, class_expressions_graph
):
    """
    An anonymous expression declared under one named class describes that class and
    keeps it as its superclass.
    """
    expression = class_expressions_extractor.extract(
        class_uri=class_expressions_graph.value(
            predicate=RDFS.subClassOf, object=class_expressions.Hunter
        )
    )
    assert (expression.is_description_for, expression.superclasses) == (
        "Hunter",
        ["Hunter"],
    )


def test_an_expression_over_a_named_class_describes_it(
    class_expressions_extractor, class_expressions, class_expressions_graph
):
    """
    An anonymous expression one named class is declared under describes that class.
    """
    expression = class_expressions_extractor.extract(
        class_uri=class_expressions_graph.value(
            class_expressions.Tiger, RDFS.subClassOf
        )
    )
    assert (expression.is_description_for, expression.superclasses) == (
        "Tiger",
        [Symbol.__name__],
    )


def test_a_definition_describes_the_class_it_defines(
    class_expressions_extractor, class_expressions, class_expressions_graph
):
    """
    An anonymous expression a named class is equivalent to describes that class.
    """
    expression = class_expressions_extractor.extract(
        class_uri=class_expressions_graph.value(
            class_expressions.Cat, OWL.equivalentClass
        )
    )
    assert (expression.is_description_for, expression.superclasses) == (
        "Cat",
        [Symbol.__name__],
    )


def test_an_intersection_describing_no_class_is_under_its_named_members(
    class_expressions_extractor, class_expressions, class_expressions_graph
):
    """
    An anonymous intersection that describes no named class is a subclass of the named
    classes it intersects.
    """
    restriction = class_expressions_graph.value(
        class_expressions.Owner, RDFS.subClassOf
    )
    expression = class_expressions_extractor.extract(
        class_uri=class_expressions_graph.value(restriction, OWL.someValuesFrom)
    )
    assert (expression.is_description_for, expression.superclasses) == (
        None,
        ["Animal", "Pet"],
    )


def test_an_expression_under_two_named_classes_is_refused(
    class_expressions_extractor, class_expressions, class_expressions_graph
):
    """
    An anonymous expression declared under two named classes describes no single class.
    """
    with pytest.raises(AmbiguousDescribedClass):
        class_expressions_extractor.extract(
            class_uri=class_expressions_graph.value(
                predicate=RDFS.subClassOf, object=class_expressions.Mammal
            )
        )


def test_an_expression_over_two_named_classes_is_refused(
    class_expressions_extractor, class_expressions, class_expressions_graph
):
    """
    An anonymous expression two named classes are declared under describes no single
    class.
    """
    with pytest.raises(AmbiguousDescribedClass):
        class_expressions_extractor.extract(
            class_uri=class_expressions_graph.value(
                class_expressions.Lion, RDFS.subClassOf
            )
        )


# %% properties


def test_an_object_property_records_its_domain_and_range(
    property_extractor, extraction
):
    """
    An object property records the classes of its declared domain and range.
    """
    has_advisor = property_extractor.extract(property_uri=extraction.hasAdvisor)
    assert (has_advisor.type, has_advisor.domains, has_advisor.ranges) == (
        PropertyType.OBJECT_PROPERTY,
        ["Student"],
        ["Professor"],
    )


def test_an_inverse_is_recorded_on_both_properties(property_extractor, extraction):
    """
    ``owl:inverseOf`` stated on one property makes each property the inverse of the
    other.
    """
    assert (
        property_extractor.extract(property_uri=extraction.hasAdvisor).inverse_of,
        property_extractor.extract(property_uri=extraction.advises).inverse_of,
    ) == ("advises", "hasAdvisor")


def test_a_property_records_its_characteristics(property_extractor, extraction):
    """
    A property declared symmetric and transitive is recorded as both.
    """
    knows = property_extractor.extract(property_uri=extraction.knows)
    assert (knows.is_symmetric, knows.is_transitive) == (True, True)


def test_a_functional_property_is_recorded_as_functional(
    property_extractor, extraction
):
    """
    A property declared ``owl:FunctionalProperty`` is recorded as functional; one that
    is not stays not functional.
    """
    assert (
        property_extractor.extract(property_uri=extraction.hasMentor).is_functional,
        property_extractor.extract(property_uri=extraction.hasAdvisor).is_functional,
    ) == (True, False)


def test_a_subproperty_records_its_superproperty(property_extractor, extraction):
    """
    A subproperty records the property it is declared a subproperty of.
    """
    assert property_extractor.extract(
        property_uri=extraction.hasSupervisor
    ).superproperties == ["knows"]


def test_a_datatype_property_is_a_data_property(property_extractor, extraction):
    """
    A datatype property is a data property whose range is its datatype.
    """
    age = property_extractor.extract(property_uri=extraction.age)
    assert (age.type, age.ranges) == (PropertyType.DATA_PROPERTY, ["integer"])


def test_a_property_is_named_for_its_field_and_its_descriptor(
    property_extractor, extraction
):
    """
    A property's field is named in snake case and its descriptor class in Pascal case.
    """
    has_advisor = property_extractor.extract(property_uri=extraction.hasAdvisor)
    assert (has_advisor.field_name, has_advisor.descriptor_name) == (
        "has_advisor",
        "HasAdvisor",
    )


# %% relations between properties


@pytest.fixture(scope="module")
def property_relations() -> Namespace:
    """
    The namespace of the ontology of related properties the tests read.
    """
    return Namespace("http://example.org/property_relations_extraction#")


@pytest.fixture(scope="module")
def property_relations_extractor() -> PropertyExtractor:
    """
    Reads the properties of the ontology of related properties.
    """
    graph = rdflib.Graph()
    graph.parse(
        Path(__file__).parent.parent
        / "dataset"
        / "ontologies"
        / "property_relations_extraction.ttl"
    )
    return PropertyExtractor(
        graph=graph, metadata_extractor=MetadataExtractor(graph=graph)
    )


def test_an_equivalence_is_recorded_on_both_properties(
    property_relations_extractor, property_relations
):
    """
    ``owl:equivalentProperty`` stated on one property makes each property equivalent to
    the other.
    """
    assert (
        property_relations_extractor.extract(
            property_uri=property_relations.livesIn
        ).equivalent_properties,
        property_relations_extractor.extract(
            property_uri=property_relations.resides
        ).equivalent_properties,
    ) == (["resides"], ["livesIn"])


def test_a_disjointness_is_recorded_on_both_properties(
    property_relations_extractor, property_relations
):
    """
    ``owl:propertyDisjointWith`` stated on one property makes each property disjoint
    with the other.
    """
    assert (
        property_relations_extractor.extract(
            property_uri=property_relations.livesIn
        ).disjoint_properties,
        property_relations_extractor.extract(
            property_uri=property_relations.visits
        ).disjoint_properties,
    ) == (["visits"], ["livesIn"])


def test_a_property_with_several_inverses_records_them_all_and_the_first_by_name(
    property_relations_extractor, property_relations
):
    """
    A property stated as the inverse of several records all of them, in order of name,
    and takes the first as its inverse.
    """
    parent_of = property_relations_extractor.extract(
        property_uri=property_relations.parentOf
    )
    assert (parent_of.inverses, parent_of.inverse_of) == (
        ["childOf", "hasParent"],
        "childOf",
    )


def test_a_property_records_asymmetry_and_irreflexivity(
    property_relations_extractor, property_relations
):
    """
    A property declared asymmetric and irreflexive is recorded as both, and as neither
    symmetric nor reflexive.
    """
    dominates = property_relations_extractor.extract(
        property_uri=property_relations.dominates
    )
    assert (
        dominates.is_asymmetric,
        dominates.is_irreflexive,
        dominates.is_symmetric,
        dominates.is_reflexive,
    ) == (True, True, False, False)


def test_a_property_records_reflexivity(
    property_relations_extractor, property_relations
):
    """
    A property declared reflexive is recorded as reflexive.
    """
    assert property_relations_extractor.extract(
        property_uri=property_relations.sameAgeAs
    ).is_reflexive


def test_a_property_keeps_the_nodes_of_its_ranges(
    property_relations_extractor, property_relations
):
    """
    A property keeps the nodes of its ranges beside their names.
    """
    assert sorted(
        property_relations_extractor.extract(
            property_uri=property_relations.livesIn
        ).range_uris
    ) == sorted([property_relations.Place, property_relations.Person])


# %% the whole ontology


def test_the_ontology_holds_its_named_classes_and_its_properties(graph):
    """
    Extracting the ontology collects its named classes and its properties by name.
    """
    ontology = OntologyExtractor(graph=graph).extract()
    assert (sorted(ontology.classes), sorted(ontology.properties)) == (
        ["Person", "Professor", "Student"],
        ["advises", "age", "hasAdvisor", "hasMentor", "hasSupervisor", "knows"],
    )
