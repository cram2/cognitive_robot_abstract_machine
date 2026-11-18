"""
The morphology pass realizes each leaf through the most specific realization that
applies to it, and leaves a leaf no realization applies to as it is.
"""

from __future__ import annotations

import pytest

from krrood.entity_query_language.verbalization.fragments.base import (
    CardinalNumber,
    CollectionElementNoun,
    IndefiniteArticle,
    IndexOrdinal,
    NounPhrase,
    RoleFragment,
    VerbalizationFragment,
    VerbFormFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.rendering.leaf_realization import (
    CardinalNumberRealization,
    CollectionElementNounRealization,
    CopulaRealization,
    IndexOrdinalRealization,
    LeafRealization,
    NounRealization,
    VerbFormRealization,
    VerbRealization,
)
from krrood.entity_query_language.verbalization.rendering.morphology_processor import (
    MorphologyProcessor,
)

# %% choosing the realization


@pytest.mark.parametrize(
    ("leaf", "expected_realization"),
    [
        (CardinalNumber(count=3), CardinalNumberRealization),
        (IndexOrdinal(index=1), IndexOrdinalRealization),
        (WordFragment(text="Robot"), NounRealization),
        (IndefiniteArticle(text="a"), NounRealization),
        (RoleFragment(text="Robot", role=SemanticRole.VARIABLE), NounRealization),
        (
            CollectionElementNoun(text="tasks", role=SemanticRole.ATTRIBUTE),
            CollectionElementNounRealization,
        ),
        (RoleFragment(text="work", role=SemanticRole.VERB), VerbRealization),
        (VerbFormFragment(text="connects"), VerbFormRealization),
        (RoleFragment(text="is", role=SemanticRole.OPERATOR), CopulaRealization),
    ],
)
def test_each_kind_of_leaf_selects_its_realization(
    leaf: VerbalizationFragment, expected_realization: type[LeafRealization]
) -> None:
    """
    The realization chosen for a leaf is the most specific one that applies to it.
    """
    assert LeafRealization.most_applicable(leaf=leaf) is expected_realization


def test_a_leaf_no_realization_applies_to_stays_as_it_is() -> None:
    """
    A noun phrase is lowered by an earlier pass, so the morphology pass has no
    realization for it and returns it unchanged.
    """
    noun_phrase = NounPhrase(head=WordFragment(text="Robot"))

    assert LeafRealization.most_applicable(leaf=noun_phrase) is None
    assert MorphologyProcessor().rewrite(leaf=noun_phrase) is noun_phrase
