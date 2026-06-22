"""
Unit tests for grammatical-number agreement.

Number is now realised in one place — the
:class:`~krrood.entity_query_language.verbalization.rendering.morphology_processor.MorphologyProcessor`
pass.  The lexicon (e.g. ``Copulas.for_number``) only *tags* fragments with
:class:`Number`; these tests therefore run the pass to observe the realised surface.
"""

from __future__ import annotations

from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
)
from krrood.entity_query_language.verbalization.rendering.morphology_processor import (
    MorphologyProcessor,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Copulas,
    ExistentialPhrase,
)
from krrood.entity_query_language.verbalization.vocabulary.words import Number


def _realised(fragment) -> str:
    """Apply the morphology pass (as ``build`` does), then flatten to plain text."""
    return flatten_fragment_to_plain_text(MorphologyProcessor().process(fragment))


def test_number_of_bridges_boolean_plan_features():
    assert Number.of(True) is Number.PLURAL
    assert Number.of(False) is Number.SINGULAR


def test_copula_agreement_is_applied_by_the_pass():
    assert _realised(Copulas.for_number(Number.SINGULAR)) == "is"
    assert _realised(Copulas.for_number(Number.PLURAL)) == "are"


def test_existential_noun_pluralised_by_the_pass():
    assert (
        _realised(ExistentialPhrase.for_number(Number.SINGULAR).build_phrase("Robot"))
        == "there's a Robot"
    )
    assert (
        _realised(ExistentialPhrase.for_number(Number.PLURAL).build_phrase("Robot"))
        == "there are Robots"
    )
