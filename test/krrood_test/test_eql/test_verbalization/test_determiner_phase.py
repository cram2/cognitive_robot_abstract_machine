"""
Unit tests for the determiner realisation pass — the single place the determiner of a
:class:`NounPhrase` is chosen, from its ``definiteness`` × ``number`` (the concord
table).

The pass lowers every ``NounPhrase`` to a determiner-bearing ``PhraseFragment`` and tags
the head's number; it runs *before* the morphology pass, so plural heads are still
tagged (not yet inflected) immediately after it.
"""

from __future__ import annotations

from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    NounPhrase,
    PhraseFragment,
    RoleFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.rendering.determiner_processor import (
    DeterminerProcessor,
)
from krrood.entity_query_language.verbalization.rendering.morphology_processor import (
    MorphologyProcessor,
)


def _noun(text: str) -> RoleFragment:
    return RoleFragment(text=text, role=SemanticRole.VARIABLE)


def _realised(np: NounPhrase) -> str:
    """
    Lower the DP then run morphology — the verbalizer's determiner→morphology order.
    """
    lowered = DeterminerProcessor().process(np)
    return flatten_fragment_to_plain_text(MorphologyProcessor().process(lowered))


# ── the concord table, cell by cell ──────────────────────────────────────────────


def test_indefinite_singular_picks_article_from_head():
    assert _realised(NounPhrase(head=_noun("Robot"))) == "a Robot"
    assert (
        _realised(
            NounPhrase(head=_noun("Apartment"), definiteness=Definiteness.INDEFINITE)
        )
        == "an Apartment"
    )


def test_indefinite_plural_drops_the_determiner():
    # the "a Robot" -> "Robots" determiner-drop, expressed in one cell
    np = NounPhrase(head=_noun("Robot"), number=GrammaticalNumber.PLURAL)
    assert _realised(np) == "Robots"


def test_definite_keeps_the_for_both_numbers():
    assert (
        _realised(NounPhrase(head=_noun("Robot"), definiteness=Definiteness.DEFINITE))
        == "the Robot"
    )
    assert (
        _realised(
            NounPhrase(
                head=_noun("Robot"),
                number=GrammaticalNumber.PLURAL,
                definiteness=Definiteness.DEFINITE,
            )
        )
        == "the Robots"
    )


def test_bare_emits_no_determiner():
    np = NounPhrase(head=_noun("Robot 2"), definiteness=Definiteness.BARE)
    assert _realised(np) == "Robot 2"


# ── structure / modifiers / nesting ──────────────────────────────────────────────


def test_modifiers_follow_the_head():
    np = NounPhrase(
        head=RoleFragment(text="drawer", role=SemanticRole.ATTRIBUTE),
        number=GrammaticalNumber.PLURAL,
        definiteness=Definiteness.INDEFINITE,
        modifiers=[
            WordFragment(text="of"),
            NounPhrase(head=_noun("Cabinet"), number=GrammaticalNumber.PLURAL),
        ],
    )
    assert _realised(np) == "drawers of Cabinets"


def test_modifier_separator_empty_attaches_appositive_clause():
    # a clause-bearing referring NP: "a Robot" + ", where ..." with no spurious space
    np = NounPhrase(
        head=_noun("Robot"),
        modifiers=[WordFragment(text=", where it works")],
        modifier_separator="",
    )
    assert _realised(np) == "a Robot, where it works"


def test_pass_lowers_nested_noun_phrases_inside_a_phrase():
    tree = PhraseFragment(
        parts=[
            WordFragment(text="the sum of"),
            NounPhrase(head=_noun("Robot"), number=GrammaticalNumber.PLURAL),
        ]
    )
    assert _realised(tree) == "the sum of Robots"


def test_pass_is_a_noop_on_a_noun_phrase_free_tree():
    leaf = RoleFragment(text="Robot", role=SemanticRole.VARIABLE)
    assert DeterminerProcessor().process(leaf) is leaf


def test_pass_is_idempotent():
    np = NounPhrase(head=_noun("Robot"), definiteness=Definiteness.DEFINITE)
    once = DeterminerProcessor().process(np)
    twice = DeterminerProcessor().process(once)
    assert flatten_fragment_to_plain_text(twice) == "the Robot"
