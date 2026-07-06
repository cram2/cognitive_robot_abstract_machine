"""
The opening verb (performative) of a verbalized query depends on the *backend* it would be
evaluated with, not only on the query type: a generative backend reads *"Generate"*, a selective
backend reads *"Find"*. With no backend the verb falls back to the query-type default (a match
generates, a plain query finds), so all existing output is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.backends import (
    EntityQueryLanguageBackend,
    ProbabilisticBackend,
)
from krrood.entity_query_language.factories import entity, an, variable
from krrood.entity_query_language.verbalization.pipeline import (
    directive_for_backend,
    verbalize_expression,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Directive


@dataclass
class Position:
    """A minimal structural type to build a match over."""

    x: float
    y: float
    z: float


def test_match_without_backend_defaults_to_generate():
    """With no backend the query-type default holds: a match reads *"Generate"*."""
    assert verbalize_expression(an(Position)(x=1)).startswith("Generate")


def test_match_with_selective_backend_reads_find():
    """A selective backend turns the match's verb into *"Find"* (it searches existing data)."""
    text = verbalize_expression(an(Position)(x=1), backend=EntityQueryLanguageBackend())
    assert text.startswith("Find")


def test_match_with_generative_backend_reads_generate():
    """A generative backend keeps the match's verb as *"Generate"*."""
    text = verbalize_expression(an(Position)(x=1), backend=ProbabilisticBackend())
    assert text.startswith("Generate")


def test_query_without_backend_defaults_to_find():
    """With no backend a plain query reads *"Find"* as before."""
    robot = variable(Position, [])
    assert verbalize_expression(entity(robot)).startswith("Find")


def test_query_with_generative_backend_reads_generate():
    """A generative backend turns a plain query's verb into *"Generate"*."""
    robot = variable(Position, [])
    text = verbalize_expression(entity(robot), backend=ProbabilisticBackend())
    assert text.startswith("Generate")


def test_query_with_selective_backend_reads_find():
    """A selective backend keeps a plain query's verb as *"Find"*."""
    robot = variable(Position, [])
    text = verbalize_expression(entity(robot), backend=EntityQueryLanguageBackend())
    assert text.startswith("Find")


def test_directive_for_backend_maps_backend_kind_to_verb():
    """The resolver maps backend kind to a directive, and ``None`` to no override."""
    assert directive_for_backend(None) is None
    assert directive_for_backend(ProbabilisticBackend()) is Directive.GENERATE
    assert directive_for_backend(EntityQueryLanguageBackend()) is Directive.FIND
