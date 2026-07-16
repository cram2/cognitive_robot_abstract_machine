"""
Unit tests for the curated mass-noun lexicon used to drop the article before uncountable
nouns.
"""

from __future__ import annotations

from krrood.entity_query_language.verbalization.vocabulary.countability import (
    Countability,
    NounCountability,
)


def test_known_mass_noun_is_uncountable():
    assert NounCountability().classify("money") is Countability.UNCOUNTABLE
    assert NounCountability().is_uncountable("water")


def test_ordinary_noun_is_countable():
    assert NounCountability().classify("battery") is Countability.COUNTABLE
    assert not NounCountability().is_uncountable("department")


def test_classification_is_case_insensitive():
    assert NounCountability().is_uncountable("Money")


def test_lexicon_is_overridable_for_a_domain():
    custom = NounCountability(uncountable_nouns=frozenset({"plasma"}))
    assert custom.is_uncountable("plasma")
    # the default mass nouns are not implied when a domain supplies its own set
    assert not custom.is_uncountable("money")


def test_expanded_lexicon_covers_common_mass_nouns():
    """
    The curated lexicon spans the everyday mass nouns: materials, substances,
    abstractions, and collective/activity nouns that take no plural and no indefinite
    article.
    """
    countability = NounCountability()
    for mass_noun in (
        "wood",
        "metal",
        "plastic",
        "leather",
        "wool",
        "cotton",
        "concrete",
        "coffee",
        "tea",
        "wine",
        "bread",
        "butter",
        "cheese",
        "meat",
        "flour",
        "love",
        "anger",
        "courage",
        "honesty",
        "freedom",
        "happiness",
        "wisdom",
        "education",
        "transport",
        "accommodation",
        "luggage",
        "garbage",
        "jewellery",
    ):
        assert countability.is_uncountable(mass_noun), mass_noun


def test_dual_sense_nouns_stay_countable():
    """
    A noun with a dominant countable sense is left countable so it is not wrongly
    stripped of its article.
    """
    countability = NounCountability()
    for countable_noun in ("battery", "department", "robot", "table", "report"):
        assert not countability.is_uncountable(countable_noun), countable_noun
