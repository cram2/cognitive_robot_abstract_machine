"""The single English-preposition inventory and the relation recognizer that derives from it.

The verbalizer keeps one preposition enum (:class:`Prepositions`): the words it emits, the
prepositions a clause links its constituents with, and the lexicon a relational-field recognizer
consults all read from it.
"""

from __future__ import annotations

from krrood.entity_query_language.verbalization.vocabulary.english import Prepositions
from krrood.entity_query_language.verbalization.relational_attributes import (
    relational_verb_phrase,
)

#: A representative spread of the simple (one-word) English prepositions the inventory must contain —
#: the everyday core, the spatial set, and the marginal/deverbal prepositions.
_EXPECTED_PREPOSITIONS = {
    "aboard",
    "about",
    "above",
    "across",
    "after",
    "against",
    "along",
    "alongside",
    "amid",
    "amidst",
    "among",
    "amongst",
    "around",
    "as",
    "astride",
    "at",
    "atop",
    "barring",
    "before",
    "behind",
    "below",
    "beneath",
    "beside",
    "besides",
    "between",
    "beyond",
    "but",
    "by",
    "concerning",
    "considering",
    "despite",
    "down",
    "during",
    "except",
    "excluding",
    "following",
    "for",
    "from",
    "given",
    "in",
    "including",
    "inside",
    "into",
    "like",
    "minus",
    "near",
    "notwithstanding",
    "of",
    "off",
    "on",
    "onto",
    "opposite",
    "outside",
    "over",
    "past",
    "pending",
    "per",
    "plus",
    "regarding",
    "round",
    "save",
    "since",
    "than",
    "through",
    "throughout",
    "till",
    "to",
    "toward",
    "towards",
    "under",
    "underneath",
    "unlike",
    "until",
    "unto",
    "up",
    "upon",
    "versus",
    "via",
    "with",
    "within",
    "without",
    "worth",
}


def test_inventory_covers_the_simple_prepositions():
    """Every preposition in the representative set is a member of the one inventory."""
    inventory = {preposition.text for preposition in Prepositions}
    missing = _EXPECTED_PREPOSITIONS - inventory
    assert not missing, f"missing prepositions: {sorted(missing)}"


def test_inventory_holds_only_one_word_prepositions():
    """The inventory is the simple (one-word) closed class — no multi-word complex prepositions."""
    assert all(" " not in preposition.text for preposition in Prepositions)


def test_emitted_prepositions_are_present():
    """The prepositions the verbalizer emits (genitive, aggregate scope, comitative) are members."""
    texts = {preposition.text for preposition in Prepositions}
    assert {"of", "among", "with", "in"} <= texts


def test_relation_recognizer_reads_a_newly_covered_preposition():
    """The relation recognizer derives from the inventory, so a preposition only now covered is read
    off a participle + preposition field name."""
    assert relational_verb_phrase("scheduled_during") == "scheduled during"
    assert relational_verb_phrase("hidden_beneath") == "hidden beneath"


def test_relation_recognizer_still_rejects_non_relations():
    """Adding prepositions does not loosen the participle guard: a noun ending in a preposition, and
    the genitive ``_of``, are still not relations."""
    assert relational_verb_phrase("color_in") is None
    assert relational_verb_phrase("number_of") is None
