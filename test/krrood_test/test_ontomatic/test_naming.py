"""
The names the converter gives to the classes, properties and fields of an ontology.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import rdflib
from rdflib import Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from krrood.ontomatic.exceptions import UnsupportedClassExpression
from krrood.ontomatic.naming import NamingRegistry

# %% local names


def test_the_local_name_follows_a_hash():
    """
    The local name of an IRI with a fragment is the fragment.
    """
    assert (
        NamingRegistry.local_name(uri=URIRef("http://example.org/ontology#Person"))
        == "Person"
    )


def test_the_local_name_follows_the_last_slash():
    """
    The local name of an IRI without a fragment is its last path segment.
    """
    assert (
        NamingRegistry.local_name(uri=URIRef("http://example.org/ontology/Person"))
        == "Person"
    )


# %% anonymous class expressions


@pytest.fixture(scope="module")
def class_expression_names() -> Namespace:
    """
    The namespace of the ontology of anonymous class expressions the tests name.
    """
    return Namespace("http://example.org/class_expression_names#")


@pytest.fixture(scope="module")
def class_expressions_graph() -> rdflib.Graph:
    """
    The graph of the ontology of anonymous class expressions.
    """
    graph = rdflib.Graph()
    graph.parse(
        Path(__file__).parent.parent
        / "dataset"
        / "ontologies"
        / "class_expression_names.ttl"
    )
    return graph


@pytest.mark.parametrize(
    "anchor, name",
    [
        ("Intersection", "PizzaANDCheese"),
        ("Union", "MeatORFish"),
        ("Enumeration", "OneOfMargheritaCapricciosa"),
        ("ComplementWithDisjointClasses", "Fish"),
        ("Complement", "NotPizza"),
        ("Nested", "PizzaANDhasToppingSOMECheese"),
        ("SomeValues", "hasToppingSOMECheese"),
        ("AllValues", "hasToppingALLCheese"),
        ("Value", "hasToppingHASMargherita"),
        ("FalseValue", "isSpicyHASfalse"),
        ("Self", "likesSELF"),
        ("MinCardinality", "hasToppingMIN1"),
        ("MaxCardinality", "hasToppingMAX3"),
        ("ExactCardinality", "hasToppingEXACTLY2"),
        ("MinQualifiedCardinality", "hasToppingMIN1Cheese"),
        ("MaxQualifiedCardinality", "hasToppingMAX2Cheese"),
        ("ZeroQualifiedCardinality", "hasToppingMAX0Cheese"),
        ("ExactQualifiedCardinality", "hasToppingEXACTLY2Cheese"),
        ("DataQualifiedCardinality", "hasCaloriesMIN1integer"),
    ],
)
def test_an_anonymous_class_expression_is_named_for_what_it_describes(
    class_expressions_graph, class_expression_names, anchor, name
):
    """
    An anonymous class expression is named from its kind and the names of what it refers
    to.
    """
    expression = class_expressions_graph.value(
        class_expression_names[anchor], RDFS.subClassOf
    )
    assert (
        NamingRegistry.bnode_to_name(graph=class_expressions_graph, bnode=expression)
        == name
    )


def test_a_node_that_is_no_class_expression_is_refused(class_expressions_graph):
    """
    A blank node that is no anonymous class expression, such as a disjointness axiom,
    gets no name.
    """
    axiom = class_expressions_graph.value(
        predicate=RDF.type, object=OWL.AllDisjointClasses
    )
    with pytest.raises(UnsupportedClassExpression):
        NamingRegistry.bnode_to_name(graph=class_expressions_graph, bnode=axiom)
