"""
Classes and properties read out of an RDF graph (``dataset/ontologies/extraction.ttl``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import rdflib
from rdflib import Namespace

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
    assert class_extractor.extract_info(class_uri=extraction.Student).superclasses == [
        "Person"
    ]


def test_a_class_without_a_superclass_derives_from_symbol(class_extractor, extraction):
    """
    A class declared under no other class gets the symbol class as its superclass.
    """
    assert class_extractor.extract_info(class_uri=extraction.Person).superclasses == [
        Symbol.__name__
    ]


def test_a_class_keeps_its_label_and_comment(class_extractor, extraction):
    """
    The label and the comment of a class are kept.
    """
    person = class_extractor.extract_info(class_uri=extraction.Person)
    assert (person.label, person.comment) == ("person", "A human being.")


# %% properties


def test_an_object_property_records_its_domain_and_range(
    property_extractor, extraction
):
    """
    An object property records the classes of its declared domain and range.
    """
    has_advisor = property_extractor.extract_info(property_uri=extraction.hasAdvisor)
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
        property_extractor.extract_info(property_uri=extraction.hasAdvisor).inverse_of,
        property_extractor.extract_info(property_uri=extraction.advises).inverse_of,
    ) == ("advises", "hasAdvisor")


def test_a_property_records_its_characteristics(property_extractor, extraction):
    """
    A property declared symmetric and transitive is recorded as both.
    """
    knows = property_extractor.extract_info(property_uri=extraction.knows)
    assert (knows.is_symmetric, knows.is_transitive) == (True, True)


def test_a_functional_property_is_recorded_as_functional(
    property_extractor, extraction
):
    """
    A property declared ``owl:FunctionalProperty`` is recorded as functional; one that
    is not stays not functional.
    """
    assert (
        property_extractor.extract_info(
            property_uri=extraction.hasMentor
        ).is_functional,
        property_extractor.extract_info(
            property_uri=extraction.hasAdvisor
        ).is_functional,
    ) == (True, False)


def test_a_subproperty_records_its_superproperty(property_extractor, extraction):
    """
    A subproperty records the property it is declared a subproperty of.
    """
    assert property_extractor.extract_info(
        property_uri=extraction.hasSupervisor
    ).superproperties == ["knows"]


def test_a_datatype_property_is_a_data_property(property_extractor, extraction):
    """
    A datatype property is a data property whose range is its datatype.
    """
    age = property_extractor.extract_info(property_uri=extraction.age)
    assert (age.type, age.ranges) == (PropertyType.DATA_PROPERTY, ["integer"])


def test_a_property_is_named_for_its_field_and_its_descriptor(
    property_extractor, extraction
):
    """
    A property's field is named in snake case and its descriptor class in Pascal case.
    """
    has_advisor = property_extractor.extract_info(property_uri=extraction.hasAdvisor)
    assert (has_advisor.field_name, has_advisor.descriptor_name) == (
        "has_advisor",
        "HasAdvisor",
    )


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
