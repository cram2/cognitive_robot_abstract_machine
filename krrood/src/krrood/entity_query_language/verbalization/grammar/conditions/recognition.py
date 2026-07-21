from __future__ import annotations

import dataclasses
import operator
from dataclasses import dataclass

from typing_extensions import List, Optional

from krrood.symbol_graph.symbol_graph import Symbol
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute, MappedVariable
from krrood.entity_query_language.core.variable import Literal, Variable
from krrood.entity_query_language.operators.aggregators import Aggregator, Extreme
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.core.expression_structure import (
    chain_ends_in_boolean_attribute,
    chain_root,
    walk_chain,
)
from krrood.entity_query_language.query.aggregation_structure import (
    aggregation_source_root,
    is_collapsible_aggregation_subquery,
    selected_aggregator,
    unwrap_result_quantifiers,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    chain_key,
    COINDEXED_OPERATORS,
    fold_range_pairs,
    SharedSubjectComparisons,
    SharedSubjectConjunction,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Operators


def attribute_names(left: SymbolicExpression) -> List[str]:
    """
    :param left: A ``MappedVariable`` chain (or any expression).
    :return: The attribute names along the chain, outermost (last-accessed) first (``[]`` if none).

    >>> attribute_names(variable(BankTransaction, []).amount_details.amount)
    ['amount', 'amount_details']
    >>> attribute_names(variable(Robot, []))
    []
    """
    names: List[str] = []
    current = left
    while isinstance(current, MappedVariable):
        if isinstance(current, Attribute):
            names.append(current._attribute_name_)
        current = current._child_
    return names


def single_hop_attribute(
    expression: SymbolicExpression, subject_variable: Optional[Variable]
) -> Optional[Attribute]:
    """
    :param expression: Candidate expression.
    :param subject_variable: The variable the attribute must be on.
    :return: The attribute node when *expression* is exactly ``subject_variable.<attribute>``, else
        ``None``.

    >>> robot = variable(Robot, [])
    >>> single_hop_attribute(robot.battery, robot)._attribute_name_
    'battery'
    >>> transaction = variable(BankTransaction, [])
    >>> single_hop_attribute(transaction.amount_details.amount, transaction) is None
    True
    """
    if subject_variable is None or not isinstance(expression, MappedVariable):
        return None
    chain, root = walk_chain(expression)
    if not (isinstance(root, Variable) and root._id_ == subject_variable._id_):
        return None
    if len(chain) != 1 or not isinstance(chain[0], Attribute):
        return None
    return chain[0]


def is_none_literal(expression: SymbolicExpression) -> bool:
    """
    :param expression: Candidate expression (typically a comparator's right side).
    :return: ``True`` when *expression* is a literal whose value is ``None`` — the marker for an
        absence comparison (``<chain> == None``) rendered as *"has no <attribute>"* / *"does not
        exist"* rather than a value predicate.

    >>> is_none_literal((variable(Robot, []).name == None).right)
    True
    >>> is_none_literal((variable(Robot, []).battery > 50).right)
    False
    """
    return isinstance(expression, Literal) and expression._value_ is None


def is_concrete_object_literal(expression: SymbolicExpression) -> bool:
    """
    :param expression: Candidate expression.
    :return: ``True`` when *expression* is a literal holding a concrete domain-object instance (a
        dataclass or :class:`Symbol` instance) — rendered by identity (*"a specific Body"*) rather
        than its (possibly huge) ``repr``. A class object, primitive, ``enum`` member, ``datetime``,
        or ``None`` is not.

    >>> robot = variable(Robot, [])
    >>> is_concrete_object_literal((robot == Robot("R2D2", 80, True)).right)
    True
    >>> is_concrete_object_literal((robot.battery > 50).right)
    False
    """
    if not isinstance(expression, Literal):
        return False
    value = expression._value_
    if isinstance(value, type):
        return False
    return dataclasses.is_dataclass(value) or isinstance(value, Symbol)


def is_atomic_value(expression: SymbolicExpression) -> bool:
    """
    :param expression: Candidate value expression (a comparator right side / assignment value).
    :return: ``True`` when the value renders as a single short token — a plain scalar / ``enum`` /
        type-object literal — so it may sit in a *"… are X, Y, and Z respectively"* coordination.
        A concrete-object literal (*"a specific Body"*), a domain-listing variable (*"one of …"*), a
        sub-query, or a range each renders as a phrase and is *not* atomic, so it is said on its own.

    >>> is_atomic_value((variable(Robot, []).battery > 50).right)
    True
    >>> is_atomic_value((variable(Robot, []).name == None).right)
    False
    """
    if not isinstance(expression, Literal):
        return False
    if expression._value_ is None:
        return False
    return not is_concrete_object_literal(expression)


def references(expression: SymbolicExpression, subject_variable: Variable) -> bool:
    """
    :param expression: Candidate expression.
    :param subject_variable: The variable to look for.
    :return: ``True`` when *expression* mentions *subject_variable* (so it is not a clean right-hand side
        value).

    >>> robot = variable(Robot, [])
    >>> references(robot.battery, robot)
    True
    >>> references(robot.battery, variable(Robot, []))
    False
    """
    try:
        return any(
            variable._id_ == subject_variable._id_
            for variable in expression._unique_variables_
        )
    except AttributeError:
        return chain_root(expression) is subject_variable


def is_boolean_attribute_chain(expression: SymbolicExpression) -> bool:
    """
    :param expression: Candidate expression.
    :return: ``True`` when *expression* is a ``MappedVariable`` chain ending in a ``bool``-typed
        attribute.

    >>> is_boolean_attribute_chain(variable(Task, []).completed)
    True
    >>> is_boolean_attribute_chain(variable(Robot, []).battery)
    False
    """
    if not isinstance(expression, MappedVariable):
        return False
    chain, _ = walk_chain(expression)
    return chain_ends_in_boolean_attribute(chain)


def fold_shared_subject_comparisons(
    operands: List[SymbolicExpression],
) -> Optional[SharedSubjectComparisons]:
    """
    :param operands: The flattened operands of a coordination (e.g. an ``OR``'s disjuncts).
    :return: a :class:`SharedSubjectComparisons` when *every* operand is a plain value comparison on
        one shared subject chain (at least two of them), else ``None`` so the caller renders the
        operands unfactored.

    A *plain value comparison* uses an order/equality operator (``>``, ``<``, ``>=``, ``<=``, ``==``)
    to compare the subject chain to a value: its left is a pure attribute chain that is not a boolean
    attribute, and its right is neither ``None`` (an absence) nor a reference back to the subject (a
    co-indexed comparison). Those shapes — and negated/membership operators, whose factored tail
    would drop the negation — keep their own dedicated surface, so they are never folded here.

    >>> robot = variable(Robot, [])
    >>> fold = fold_shared_subject_comparisons([robot.battery > 50, robot.battery < 10])
    >>> [comparator.operation.__name__ for comparator in fold.comparators]
    ['gt', 'lt']
    >>> fold_shared_subject_comparisons([robot.battery > 50, robot.name == 'x']) is None
    True
    """
    if len(operands) < 2:
        return None
    subject_key = None
    for operand in operands:
        if not isinstance(operand, Comparator):
            return None
        if operand.operation not in COINDEXED_OPERATORS:
            return None
        key = chain_key(operand.left)
        if key is None or is_boolean_attribute_chain(operand.left):
            return None
        if subject_key is None:
            subject_key = key
        elif key != subject_key:
            return None
        if is_none_literal(operand.right) or references(
            operand.right, chain_root(operand.left)
        ):
            return None
    return SharedSubjectComparisons(
        subject_expression=operands[0].left, comparators=list(operands)
    )


def fold_shared_subject_conjunction(
    operands: List[SymbolicExpression],
) -> Optional[SharedSubjectConjunction]:
    """
    :param operands: The flattened operands of an ``AND``.
    :return: a :class:`SharedSubjectConjunction` when *every* operand is a plain value comparison on
        one shared *bare variable* (at least two of them), else ``None`` so the caller renders the
        operands as separate clauses.

    Bare-variable only — the relative pronoun of the *"<subject> that is …"* surface attaches
    unambiguously to the subject noun. Unlike the disjunctive fold this admits ``!=`` (the
    relative-clause tail keeps its polarity but shares the lead copula, *"not 5"*), and it range-folds
    a complementary bound pair so the clause reads *"between 1 and 10 and not 5"*.

    >>> x = variable(int, [])
    >>> fold_shared_subject_conjunction([x > 1, x < 10, x != 5]) is not None
    True
    >>> robot = variable(Robot, [])
    >>> fold_shared_subject_conjunction([robot.battery > 1, robot.battery < 10]) is None
    True
    """
    if len(operands) < 2:
        return None
    subject = None
    for operand in operands:
        if not isinstance(operand, Comparator):
            return None
        if not Operators.is_value_comparison(operand.operation):
            return None
        left = operand.left
        if not isinstance(left, Variable) or isinstance(left, Literal):
            return None
        if subject is None:
            subject = left
        elif left._id_ != subject._id_:
            return None
        if is_none_literal(operand.right) or references(operand.right, left):
            return None
    return SharedSubjectConjunction(
        subject_expression=operands[0].left, tails=fold_range_pairs(list(operands))
    )


@dataclass(frozen=True)
class SuperlativeFold:
    """
    A subject restriction of the form ``subject.<chain> == max/min(<same-type>.<same
    chain>)`` folded to a superlative selection modifier — *"with the maximum <leaf>"*.

    English's superlative is the meaning of "equal to the extreme value over the whole
    population", so this fold is meaning-preserving, not an optimisation.
    """

    aggregator: Aggregator
    """
    The ``Max`` / ``Min`` aggregator — supplies the superlative word and the leaf
    attribute for the *"… <leaf>"* tail.
    """


def superlative_aggregation(
    comparator: SymbolicExpression, subject: Optional[Variable]
) -> Optional[SuperlativeFold]:
    """Recognise a superlative restriction on *subject*.  The guard is strict so it never fires
    on a self-join, a constrained sub-query, a different chain, or a non-extreme aggregation.

    :param comparator: Candidate ``==`` comparator.
    :param subject: The variable the restriction is on.
    :return: The fold when *comparator* is ``subject.<chain> == <unconstrained Max/Min over a
        different same-type variable's identical chain>``, else ``None``.

    >>> employee, peers = variable(Employee, []), variable(Employee, [])
    >>> superlative_aggregation(employee.salary == the(entity(max(peers.salary))), employee) is not None
    True
    >>> superlative_aggregation(employee.salary > 50, employee) is None
    True
    """
    if subject is None or not isinstance(comparator, Comparator):
        return None
    if comparator.operation is not operator.eq:
        return None
    left_root = chain_root(comparator.left)
    if left_root._id_ != subject._id_:
        return None

    inner = unwrap_result_quantifiers(comparator.right)
    if not (isinstance(inner, Entity) and is_collapsible_aggregation_subquery(inner)):
        return None
    aggregator = selected_aggregator(inner)
    if not isinstance(aggregator, Extreme) or aggregator._leaf_attribute_ is None:
        return None

    source = aggregation_source_root(inner)
    if source is None or source is subject:
        return None  # the population must be a distinct variable of the same type
    if source._type_ is not subject._type_:
        return None
    if attribute_names(comparator.left) != attribute_names(aggregator._child_):
        return None
    return SuperlativeFold(aggregator=aggregator)
