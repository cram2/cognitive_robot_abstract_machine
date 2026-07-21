"""Tests for the comparator's operand-ordering optimization.

The comparator evaluates a definite (``the``) query operand first, since it yields a single result.
After quantification became a pipeline stage rather than a tree node, a definite operand is recognised
through its quantifier specification rather than a quantifier node.
"""

from __future__ import annotations

from krrood.entity_query_language.factories import entity, variable, an, the
from krrood.entity_query_language.operators.comparator import Comparator


def test_definite_query_operand_is_recognised():
    """A definite (``the``) query operand is detected; an indefinite query or plain variable is not."""
    definite_operand = the(entity(variable(int, [1])))
    indefinite_operand = an(entity(variable(int, [1, 2, 3])))
    plain_operand = variable(int, [1, 2, 3])

    assert Comparator._operand_contains_definite_query_(definite_operand)
    assert not Comparator._operand_contains_definite_query_(indefinite_operand)
    assert not Comparator._operand_contains_definite_query_(plain_operand)
