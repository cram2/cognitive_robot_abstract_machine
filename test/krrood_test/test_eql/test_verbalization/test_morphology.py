"""
Standalone unit tests for the morphology facade
(:mod:`krrood.entity_query_language.verbalization.morphology`).

The facade is the single point of access to ``inflect`` / ``lemminflect``; these tests pin the
behaviour each call site relies on.
"""

from __future__ import annotations

from krrood.entity_query_language.verbalization import morphology


def test_plural_is_unconditional():
    assert morphology.plural("Robot") == "Robots"
    assert morphology.plural("battery") == "batteries"


def test_ensure_plural_does_not_double_pluralise():
    assert morphology.ensure_plural("Robot") == "Robots"
    assert morphology.ensure_plural("Robots") == "Robots"
    assert morphology.ensure_plural("batteries") == "batteries"


def test_is_plural():
    assert morphology.is_plural("Robots") is True
    assert morphology.is_plural("Robot") is False


def test_indefinite_article_is_phonological():
    assert morphology.indefinite_article("Robot") == "a"
    assert morphology.indefinite_article("apple") == "an"
    assert morphology.indefinite_article("hour") == "an"
    assert morphology.indefinite_article("university") == "a"


def test_ordinal_is_zero_based_words():
    assert morphology.ordinal(0) == "first"
    assert morphology.ordinal(1) == "second"
    assert morphology.ordinal(2) == "third"


def test_index_ordinal_counts_from_start_for_non_negative():
    assert morphology.index_ordinal(0) == "first"
    assert morphology.index_ordinal(2) == "third"


def test_index_ordinal_counts_from_end_for_negative():
    assert morphology.index_ordinal(-1) == "last"
    assert morphology.index_ordinal(-2) == "second to last"
    assert morphology.index_ordinal(-3) == "third to last"


def test_cardinal_is_number_words():
    assert morphology.cardinal(2) == "two"
    assert morphology.cardinal(3) == "three"


def test_is_past_participle_distinguishes_participles_from_nouns():
    assert morphology.is_past_participle("assigned") is True
    assert morphology.is_past_participle("written") is True  # irregular
    assert morphology.is_past_participle("battery") is False
    assert morphology.is_past_participle("assign") is False  # base form
