"""
Wording tests for the role predicates' verbalization results.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.entity_query_language.role_predicates import (
    IsSameSemanticEntity,
    IsSubClassOrRole,
)


@dataclass(kw_only=True)
class Document:
    """
    A minimal entity the identity predicate is verbalized over.
    """

    name: str


def test_is_same_semantic_entity_reads_with_a_single_determiner():
    """
    The identity clause reads *"… is the same semantic entity as …"* — the complement
    carries exactly one determiner, not a doubled *"a another Document"*.

    The two operands share a type and are both fresh (first mention), so they are told
    apart by the indefinite alternative determiner (*"a Document … another Document"*)
    rather than a number.
    """
    first, second = variable(Document, []), variable(Document, [])
    assert (
        verbalize_expression(
            IsSameSemanticEntity(first_entity=first, second_entity=second)
        )
        == "a Document is the same semantic entity as another Document"
    )


def test_is_subclass_or_role_gives_each_noun_its_own_preposition():
    """
    The subclass check reads *"… is a subclass of or a role for …"*: a class is a
    subclass *of* another, but a role *for* it.
    """
    assert (
        verbalize_expression(
            IsSubClassOrRole(sub_class=variable(type, []), super_class=Document)
        )
        == "a type is a subclass of or a role for Document"
    )
