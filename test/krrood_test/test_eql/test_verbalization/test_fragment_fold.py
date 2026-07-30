"""
Standalone unit tests for the fragment-IR catamorphism (``fold_fragment``) and the
plain-text flatten built on top of it.

These exercise the fold directly — no verbalizer, no EQL — so the recursion scheme is
validated in isolation from the rules that produce fragments.
"""

from __future__ import annotations

from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    flatten_fragment_to_plain_text,
    fold_fragment,
    PhraseFragment,
    RoleFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole


def _to_text(fragment) -> str:
    """
    Fold a fragment to plain text using a trivial algebra.
    """
    return fold_fragment(
        fragment,
        word=lambda t: t,
        role=lambda t, _role, _ref: t,
        phrase=lambda parts, sep: sep.join(parts),
        block=lambda b: "",
    )


def test_fold_word_and_role_are_leaves():
    assert _to_text(WordFragment("the")) == "the"
    assert _to_text(RoleFragment("Robot", SemanticRole.VARIABLE)) == "Robot"


def test_fold_phrase_recurses_and_joins_with_separator():
    phrase = PhraseFragment(
        parts=[WordFragment("the"), RoleFragment("Robot", SemanticRole.VARIABLE)],
        separator=" ",
    )
    assert _to_text(phrase) == "the Robot"

    nested = PhraseFragment(parts=[WordFragment("a"), phrase], separator="-")
    assert _to_text(nested) == "a-the Robot"


def test_fold_passes_already_folded_children_to_phrase_handler():
    # The phrase handler receives folded child values, not raw fragments.
    seen = []

    def phrase_handler(parts, sep):
        seen.append(parts)
        return sep.join(parts)

    fold_fragment(
        PhraseFragment(parts=[WordFragment("x"), WordFragment("y")]),
        word=lambda t: t.upper(),
        role=lambda t, r, ref: t,
        phrase=phrase_handler,
        block=lambda b: "",
    )
    assert seen == [["X", "Y"]]


def test_fold_gives_block_handler_the_raw_block():
    block = BlockFragment(header=WordFragment("If"), items=[WordFragment("a")])
    captured = {}

    def block_handler(b):
        captured["header"] = b.header
        captured["items"] = b.items
        return "BLOCK"

    result = fold_fragment(
        block,
        word=lambda t: t,
        role=lambda t, r, ref: t,
        phrase=lambda parts, sep: sep.join(parts),
        block=block_handler,
    )
    assert result == "BLOCK"
    assert captured["header"] is block.header
    assert captured["items"] is block.items


def test_fold_counts_leaves_via_a_numeric_algebra():
    tree = PhraseFragment(
        parts=[
            WordFragment("a"),
            PhraseFragment(parts=[WordFragment("b"), WordFragment("c")]),
        ]
    )
    count = fold_fragment(
        tree,
        word=lambda t: 1,
        role=lambda t, r, ref: 1,
        phrase=lambda parts, sep: sum(parts),
        block=lambda b: 0,
    )
    assert count == 3


def test_flatten_matches_manual_fold():
    block = BlockFragment(
        header=WordFragment("Find a Robot"),
        items=[
            PhraseFragment(parts=[WordFragment("whose"), WordFragment("battery")]),
            WordFragment("is high"),
        ],
    )
    # Block items are comma-joined, header prefixed with a space.
    assert (
        flatten_fragment_to_plain_text(block) == "Find a Robot whose battery, is high"
    )


def test_flatten_headerless_block_is_comma_joined_items():
    block = BlockFragment(header=None, items=[WordFragment("a"), WordFragment("b")])
    assert flatten_fragment_to_plain_text(block) == "a, b"
