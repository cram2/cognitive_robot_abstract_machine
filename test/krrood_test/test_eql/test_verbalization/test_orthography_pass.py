"""
Unit tests for the OrthographyProcessor — the realisation pass that drops the space
adjacent to glued punctuation, so rules emit punctuation as ordinary, normally-separated
tokens and need not manage spacing with ``separator=""``.
"""

from __future__ import annotations

from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    PhraseFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.rendering.orthography_processor import (
    OrthographyProcessor,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Punctuation


def _text(fragment) -> str:
    return flatten_fragment_to_plain_text(OrthographyProcessor().process(fragment))


def test_comma_hugs_the_preceding_token():
    tree = PhraseFragment(
        parts=[
            WordFragment(text="a"),
            Punctuation.COMMA.as_fragment(),
            WordFragment(text="b"),
        ]
    )
    assert _text(tree) == "a, b"


def test_parens_hug_their_content():
    tree = PhraseFragment(
        parts=[
            Punctuation.OPEN_PAREN.as_fragment(),
            WordFragment(text="x"),
            Punctuation.CLOSE_PAREN.as_fragment(),
        ]
    )
    assert _text(tree) == "(x)"


def test_parens_inside_a_spaced_phrase():
    tree = PhraseFragment(
        parts=[
            WordFragment(text="not"),
            Punctuation.OPEN_PAREN.as_fragment(),
            WordFragment(text="x"),
            Punctuation.CLOSE_PAREN.as_fragment(),
        ]
    )
    assert _text(tree) == "not (x)"


def test_leading_comma_has_no_preceding_token():
    tree = PhraseFragment(
        parts=[Punctuation.COMMA.as_fragment(), WordFragment(text="where")]
    )
    assert _text(tree) == ", where"


def test_no_glue_is_unchanged():
    tree = PhraseFragment(parts=[WordFragment(text="a"), WordFragment(text="b")])
    assert _text(tree) == "a b"
