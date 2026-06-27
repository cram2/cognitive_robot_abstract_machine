from __future__ import annotations

import operator
from dataclasses import dataclass

from typing_extensions import List, Optional, Tuple

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.expression_structure import walk_chain
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.query.match import Match, is_underspecified
from krrood.entity_query_language.verbalization.grammar.framework.planner import Planner
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    group_by_owner,
)


@dataclass(frozen=True)
class AttributeAssignment:
    """One ``object.attribute == value`` equality from a match's construction pattern."""

    attribute: Attribute
    """The matched attribute (``position.x``)."""

    value: SymbolicExpression
    """The value the attribute is equated to."""

    is_predicted: bool
    """``True`` when the value is ``...`` (Ellipsis) — the attribute is to be generated, verbalised
    as *"predict …"* rather than an equality."""

    comparator: Comparator
    """The source ``attribute == value`` equality, so an ungrouped concrete assignment can be said
    through the shared comparator-predicate path (and pronominalised by coreference) rather than a
    hand-built genitive."""


@dataclass(frozen=True)
class AttributeGroup:
    """The equality assignments that share one object — e.g. the *x*, *y*, *z* of one position —
    so they can be aggregated into *"x, y, and z of the position are 1, 2, and 3 respectively"*.
    """

    object: SymbolicExpression
    """The object whose attributes these are — the selection itself for a direct attribute, or a
    sub-object chain (``pose.position``) for a nested match's attributes."""

    assignments: List[AttributeAssignment]
    """The attribute assignments on *object*, in construction order."""

    @property
    def concrete(self) -> List[AttributeAssignment]:
        """:return: The assignments with a concrete value (the *"given that …"* part).

        >>> from krrood.entity_query_language.verbalization.grammar.match.planner import MatchPlanner
        >>> plan = MatchPlanner(underspecified(Robot)(name="R2", battery=...)).plan()
        >>> len(plan.groups[0].concrete)
        1
        """
        return [a for a in self.assignments if not a.is_predicted]

    @property
    def predicted(self) -> List[AttributeAssignment]:
        """:return: The Ellipsis assignments (the *"predict …"* part).

        >>> from krrood.entity_query_language.verbalization.grammar.match.planner import MatchPlanner
        >>> plan = MatchPlanner(underspecified(Robot)(name="R2", battery=...)).plan()
        >>> len(plan.groups[0].predicted)
        1
        """
        return [a for a in self.assignments if a.is_predicted]


@dataclass(frozen=True)
class MatchPlan:
    """The *what to say* decomposition of a match: whether it is generative, what it selects, the
    grouped construction-pattern equalities, and the free ``where`` conditions."""

    underspecified: bool
    """``True`` ⇒ a generative request (*"Generate"*); ``False`` ⇒ a domain search (*"Find"*)."""

    selection: SymbolicExpression
    """The variable the match constructs/selects."""

    groups: List[AttributeGroup]
    """Single-hop construction equalities, grouped by their object."""

    other_conditions: List[SymbolicExpression]
    """Construction conditions that don't group (multi-hop chains, type filters); rendered as
    individual *"given that"* points."""

    where_conditions: List[SymbolicExpression]
    """The conditions added via ``.where(...)``. The plan only classifies them as the ``where``
    part; deciding how to say a list of conditions (including folding bound pairs into a *between*)
    belongs to the condition verbalizer, not here."""


@dataclass
class MatchPlanner(Planner[Match, MatchPlan]):
    """
    Decompose a ``Match`` into a ``MatchPlan``: split the construction-pattern equalities (which
    become *"given that"*) from the ``where`` conditions, and aggregate the single-hop equalities by
    their object so related attributes (a position's x/y/z) verbalise together.

    Reference: Reiter & Dale (2000) — content determination + aggregation (microplanning).

    >>> MatchPlanner(underspecified(Robot)(name="R2", battery=80)).plan().underspecified
    True
    """

    def plan(self) -> MatchPlan:
        """:return: The match plan.

        >>> plan = MatchPlanner(underspecified(Robot)(name="R2", battery=80)).plan()
        >>> (type(plan.selection).__name__, len(plan.groups))
        ('Variable', 1)
        """
        match = self.node
        match.resolve()
        # Aggregate single-hop equalities by their object via the shared owner-grouping primitive;
        # conditions that are not such an equality (``_as_assignment`` → ``None``) are the remainder.
        owner_groups, other = group_by_owner(
            list(match.conditions), self._as_assignment
        )
        groups = [
            AttributeGroup(object=group.owner, assignments=group.items)
            for group in owner_groups
        ]
        return MatchPlan(
            underspecified=is_underspecified(match),
            selection=match.variable,
            groups=groups,
            other_conditions=other,
            where_conditions=list(match._where_conditions_),
        )

    @staticmethod
    def _as_assignment(
        condition: SymbolicExpression,
    ) -> Optional[Tuple[SymbolicExpression, AttributeAssignment]]:
        """
        :param condition: A construction-pattern condition.
        :return: ``(object, assignment)`` when *condition* is an attribute equality
            (``object.attr == value``) — grouping by the attribute's immediate owner so a nested
            match's attributes aggregate per sub-object (``pose.position.x/y/z`` group under
            ``pose.position``) — else ``None`` (a non-equality or non-attribute condition).

        >>> MatchPlanner._as_assignment(variable(Robot, []).battery > 50) is None
        True
        >>> owner, assignment = MatchPlanner._as_assignment(variable(Robot, []).name == "R2")
        >>> assignment.is_predicted
        False
        """
        if not (
            isinstance(condition, Comparator) and condition.operation is operator.eq
        ):
            return None
        chain, root = walk_chain(condition.left)
        if not chain or not isinstance(chain[-1], Attribute):
            return None
        # The attribute's owner is the previous hop (``pose.position`` for ``pose.position.x``), or
        # the chain root for a direct attribute (``pose.frame``); group the assignments by it.
        owner = chain[-2] if len(chain) >= 2 else root
        value = condition.right
        is_predicted = isinstance(value, Literal) and value._value_ is Ellipsis
        return owner, AttributeAssignment(
            attribute=chain[-1],
            value=value,
            is_predicted=is_predicted,
            comparator=condition,
        )
