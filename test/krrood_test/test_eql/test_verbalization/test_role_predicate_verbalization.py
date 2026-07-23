"""
Wording tests for the role predicates' verbalization surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.patterns.role_predicates import IsSameSemanticEntity


@dataclass
class Document:
    """
    A minimal entity the identity predicate is verbalized over.
    """

    name: str


def test_is_same_semantic_entity_reads_with_a_single_determiner():
    """
    The identity clause reads *"… is the same entity as …"* — the complement carries
    exactly one determiner, not a doubled *"a another Document"*.

    The two operands share a type and are both fresh (first mention), so they are told
    apart by the indefinite alternative determiner (*"a Document … another Document"*)
    rather than a number.
    """
    first, second = variable(Document, []), variable(Document, [])
    assert (
        verbalize_expression(IsSameSemanticEntity(first, second))
        == "a Document is the same entity as another Document"
    )
