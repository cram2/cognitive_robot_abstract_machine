"""
The names the converter gives to the classes, properties and fields of an ontology.
"""

from __future__ import annotations

from rdflib import URIRef

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


# %% case conversion


def test_snake_case_splits_an_acronym_as_one_word():
    """
    An acronym inside a name becomes one word of the snake case name.
    """
    assert NamingRegistry.to_snake_case(name="hasURIValue") == "has_uri_value"


def test_snake_case_lowercases_a_pascal_case_name():
    """
    A Pascal case name becomes the same words in snake case.
    """
    assert NamingRegistry.to_snake_case(name="HasTopping") == "has_topping"


def test_pascal_case_capitalises_the_first_letter_and_keeps_the_rest():
    """
    A camel case name becomes Pascal case, keeping the case of the letters after the
    first.
    """
    assert NamingRegistry.to_pascal_case(name="hasURIValue") == "HasURIValue"
