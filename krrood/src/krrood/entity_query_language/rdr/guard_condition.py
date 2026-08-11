"""
The leaf-level guard value object shared by backward inference and its branch semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from typing_extensions import TYPE_CHECKING, Any

from krrood.entity_query_language.factories import not_

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression
    from krrood.entity_query_language.core.variable import Variable


@dataclass(frozen=True)
class GuardCondition:
    """
    A condition that must be satisfied for a rule to be applied.

    Each guard is one leaf-level predicate extracted from the rule tree's conclusion
    selectors. ``negated=True`` means the rule applies only when the condition is False.

    :attr:`original_expression` is always a leaf-level EQL node (e.g. a ``Comparator``),
    never a ``ConclusionSelector``.
    """

    original_expression: SymbolicExpression
    """
    The leaf-level EQL predicate to evaluate (e.g. a ``Comparator``).
    """

    negated: bool = False
    """
    When ``True`` the guard is satisfied only if :attr:`original_expression` is False.

    Polarity is carried here rather than applied to :attr:`original_expression`, because
    negating an expression reparents it and :attr:`original_expression` belongs to the
    live rule tree.
    """

    def holds_for(
        self,
        shared_variable: Variable,
        case: Any,
    ) -> bool:
        """
        Evaluate this guard against *case* bound to *shared_variable*.

        Respects :attr:`negated`: a negated guard must evaluate to ``False`` for the
        result to be ``True``.

        :param shared_variable: The EQL variable the conditions range over.
        :param case: The concrete case object to evaluate against.
        :return: ``True`` if the guard is satisfied.
        """
        shared_variable._update_domain_([case])
        # A leaf predicate yields its own bound boolean per case; a Not() has no
        # id-keyed payload of its own, so it yields the full binding row when it holds
        # and nothing when it does not. Both read correctly through bool().
        truth = any(bool(result) for result in self.original_expression.evaluate())
        return not truth if self.negated else truth

    @cached_property
    def as_expression(self) -> SymbolicExpression:
        """
        Produce the EQL expression with the negation applied for a negated guard.

        A negated guard is wrapped with ``not_`` so the produced condition expression is
        satisfied when the guard's expression is False.
        """
        return (
            not_(self.original_expression) if self.negated else self.original_expression
        )
