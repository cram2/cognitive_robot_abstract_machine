"""
Standalone unit tests for the morphology facade
(:mod:`krrood.entity_query_language.verbalization.morphology`).

The facade is the single point of access to ``inflect``; these tests pin the
behaviour each call site relies on.
"""

from __future__ import annotations

import pytest

from krrood.entity_query_language.verbalization import morphology


@pytest.fixture(autouse=True)
def _clear_overrides():
    """Each test starts and ends with no registered overrides."""
    morphology.clear_overrides()
    yield
    morphology.clear_overrides()


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


def test_plural_override_beats_inflect():
    morphology.register_plural("Octopus", "Octopodes")
    assert morphology.plural("Octopus") == "Octopodes"
    assert morphology.ensure_plural("Octopus") == "Octopodes"
    assert (
        morphology.ensure_plural("Octopodes") == "Octopodes"
    )  # already the override plural
    assert morphology.is_plural("Octopodes") is True
    assert morphology.is_plural("Octopus") is False


def test_invariant_plural_override():
    morphology.register_plural("sheep", "sheep")
    assert morphology.ensure_plural("sheep") == "sheep"
    assert morphology.is_plural("sheep") is True


def test_indefinite_article_override():
    morphology.register_indefinite_article("FBI", "an")
    assert morphology.indefinite_article("FBI") == "an"


def test_no_override_is_pure_inflect():
    # With nothing registered, behaviour is unchanged.
    assert morphology.plural("Robot") == "Robots"
    assert morphology.indefinite_article("apple") == "an"
