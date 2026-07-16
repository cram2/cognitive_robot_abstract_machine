"""
Pure structural and semantic queries over EQL expression trees.

These helpers answer questions about an expression's *shape* — its navigation chain, its
chain root, whether it ends in a boolean attribute, whether it denotes a temporal value
— without building anything or touching any rendering concern. They live in the core
(next to the expression classes) because the facts they expose are domain knowledge of
the query algebra, usable by any consumer (evaluation, optimization, verbalization, …),
and they delegate to the existing :class:`MappedVariable` access-path properties rather
than re-walking the tree.
"""

from __future__ import annotations

import datetime
import uuid

from typing_extensions import Iterable, List, Set, Tuple

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute, MappedVariable
from krrood.entity_query_language.core.variable import Literal, Variable


def walk_chain(
    expression: SymbolicExpression,
) -> Tuple[List[MappedVariable], SymbolicExpression]:
    """
    Walk a ``MappedVariable`` chain outward-first.

    Example: for ``robot.arm.joint`` the chain is
    ``[Attribute('joint'), Attribute('arm')]`` and root is the ``robot`` variable.

    :param expression: Any expression; non-``MappedVariable`` expressions return an empty
        chain with *expression* as the root.
    :return: Tuple ``(chain, root)`` — the access path (root-adjacent first, terminal last)
        and the chain base.
    """
    if isinstance(expression, MappedVariable):
        return list(expression._access_path_), expression._chain_root_
    return [], expression


def chain_root(expression: SymbolicExpression) -> SymbolicExpression:
    """
    :param expression: Any expression.
    :return: The non-``MappedVariable`` root of *expression* (the deepest non-``MappedVariable``
        node, or *expression* itself when it is not a ``MappedVariable``), found without building
        the full chain list.
    """
    return (
        expression._chain_root_
        if isinstance(expression, MappedVariable)
        else expression
    )


def root_variable_ids(expressions: Iterable[SymbolicExpression]) -> Set[uuid.UUID]:
    """
    :param expressions: Any expressions.
    :return: The ids of the distinct ``Variable`` chain-roots among *expressions* (e.g. for
        ``employee.department`` the root is the ``employee`` variable). Expressions whose root is
        not a ``Variable`` contribute nothing.
    """
    return {
        root._id_
        for root in (chain_root(expression) for expression in expressions)
        if isinstance(root, Variable)
    }


def chain_ends_in_boolean_attribute(chain: List[MappedVariable]) -> bool:
    """
    :param chain: A walked chain (root-adjacent first).
    :return: ``True`` when the walked *chain* ends in a ``bool``-typed attribute (the
        predicative *"<navigation> is <attribute>"* form).
    """
    return bool(chain) and isinstance(chain[-1], Attribute) and chain[-1]._type_ is bool


def is_date_type(type_: object) -> bool:
    """
    :param type_: Any value, typically an expression's ``_type_`` (which may be ``None`` or a
        non-type callable).
    :return: ``True`` when *type_* is ``datetime.date`` or a subclass (``datetime.datetime``
        included, since it derives from ``date``).
    """
    return isinstance(type_, type) and issubclass(type_, datetime.date)


def is_temporal(expression: SymbolicExpression) -> bool:
    """
    :param expression: Any EQL expression.
    :return: ``True`` when *expression* denotes a date or datetime value or variable. Date-only
        fields count: ``datetime`` derives from ``date``, and the temporal comparators
        (*"before"*, *"no later than"*) read naturally for both.
    """
    if isinstance(expression, Literal):
        return isinstance(expression._value_, datetime.date)
    if isinstance(expression, Variable):
        return is_date_type(expression._type_)
    if isinstance(expression, MappedVariable):
        chain, _ = walk_chain(expression)
        return bool(chain) and is_date_type(chain[-1]._type_)
    return False
