from __future__ import annotations

import uuid
from abc import abstractmethod
from dataclasses import dataclass

from typing_extensions import (
    TYPE_CHECKING,
    ClassVar,
    Dict,
    FrozenSet,
    Optional,
    Protocol,
    runtime_checkable,
)

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.operators.logical_quantifiers import (
    QuantifiedConditional,
)
from krrood.entity_query_language.query.query import Entity, Query
from krrood.entity_query_language.verbalization.grammar.framework.specificity import (
    SpecificityRule,
)
from krrood.entity_query_language.verbalization.grammar.query.planner import (
    QueryPlan,
    QueryPlanner,
)
from krrood.entity_query_language.verbalization.microplanning.microplan import Microplan
from krrood.entity_query_language.query.aggregation_structure import (
    aggregation_leaf_attribute,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.microplanning.coordination import (
        FoldNode,
    )


@runtime_checkable
class DiscourseView(Protocol):
    """The narrow slice of discourse information the coreference pass consumes.

    The pass depends on this, not on ``QueryPlan`` or the planners (ISP/DIP): it only needs to know
    whether a fragment's source node opens a discourse scope, and who its focus referent is.
    """

    def is_scope(self, source: Optional[FoldNode]) -> bool: ...

    def focus_of(self, source: Optional[FoldNode]) -> Optional[uuid.UUID]: ...

    def is_selected_quantity(self, node_id: Optional[uuid.UUID]) -> bool: ...


class DiscourseScopeRule(SpecificityRule):
    """A construct that opens a discourse scope, and who its focus referent is.

    Both a query (its WHERE subject is in focus, so *"its salary"*) and a quantified conditional
    (its bound variable is in focus, so *"its battery"* / *"their batteries"*) introduce a referent
    that a pronoun inside their body resolves to. Each is one alternative here, selected by
    construct, so a new scope-opening construct is a new subclass — the projection loop never grows a
    branch (open/closed).
    """

    construct: ClassVar[type]
    """The EQL node class this scope rule opens a scope for (the ``isinstance`` gate)."""

    @classmethod
    def applies(cls, node: SymbolicExpression, microplan: Microplan) -> bool:
        """:return: ``True`` when *node* is an instance of this rule's :attr:`construct`.

        >>> robot = variable(Robot, [])
        >>> QuantifierScope.applies(exists(robot, robot.battery > 0), None)
        True
        >>> QuantifierScope.applies(robot.battery > 0, None)
        False
        """
        return isinstance(node, cls.construct)

    @classmethod
    @abstractmethod
    def focus(
        cls, node: SymbolicExpression, microplan: Microplan
    ) -> Optional[uuid.UUID]:
        """:return: The id of *node*'s focus referent — the one a possessive pronoun in its body
        resolves to (``None`` when the scope has no single subject)."""


class QueryScope(DiscourseScopeRule):
    """A query opens a scope focused on its WHERE subject (an aggregation's source for a value
    sub-query, else the common chain root) — so a chain rooted there reads *"its …"*.

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(an(entity(employee).where(employee.salary > employee.starting_salary)))
    'Find an Employee such that its salary is greater than its starting_salary'
    """

    construct = Query

    @classmethod
    def focus(
        cls, node: SymbolicExpression, microplan: Microplan
    ) -> Optional[uuid.UUID]:
        """:return: The query's focus referent, read from its plan.

        >>> robot = variable(Robot, [])
        >>> from krrood.entity_query_language.verbalization.context import MicroplanningServices
        >>> query = entity(robot).where(robot.battery > 50)
        >>> services = MicroplanningServices.from_expression(query)
        >>> QueryScope.focus(query, services.microplan) == robot._id_
        True
        """
        return cls._focus_from_plan(microplan.plan_for(node, QueryPlanner))

    @staticmethod
    def _focus_from_plan(plan: QueryPlan) -> Optional[uuid.UUID]:
        """:return: The focus referent id for a query plan — the aggregation source for an
        aggregation value-subquery, else the WHERE subject, else the single common chain root (so a
        subject-less query like a ``set_of`` still pronominalises *"its …"*), else ``None``.

        >>> robot = variable(Robot, [])
        >>> scope = entity(robot).where(robot.battery > 50)
        >>> QueryScope._focus_from_plan(QueryPlanner(scope).plan()) == robot._id_
        True
        """
        if plan.is_aggregation_subquery and plan.aggregation_data is not None:
            source = plan.aggregation_data.source
            return source._id_ if source is not None else None
        if plan.subject is not None:
            return plan.subject._id_
        return plan.discourse_root


class QuantifierScope(DiscourseScopeRule):
    """A quantified conditional (``ForAll`` / ``Exists``) opens a scope focused on its bound
    variable — so the condition over that variable reads *"its battery"* (existential) or
    *"their batteries"* (universal, the population read as plural), not a repeated *"the battery of
    the Robot"*.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(exists(robot, robot.battery > 50))
    'there exists a Robot such that its battery is greater than 50'
    """

    construct = QuantifiedConditional

    @classmethod
    def focus(
        cls, node: SymbolicExpression, microplan: Microplan
    ) -> Optional[uuid.UUID]:
        """:return: The bound variable's referent id — the condition's chains rooted there
        pronominalise.

        >>> robot = variable(Robot, [])
        >>> QuantifierScope.focus(exists(robot, robot.battery > 0), None) == robot._id_
        True
        """
        return node.variable._id_


@dataclass(frozen=True)
class DiscourseModel:
    """
    The focus referent of every discourse scope, projected once via the :class:`DiscourseScopeRule`
    family.

    A scope's focus — the referent a possessive pronoun resolves to (*"its"*/*"their"*) — is a
    *what to say* fact each scope-opening construct fixes: the aggregation source population or WHERE
    subject for a query (and nothing for a set-of), the bound variable for a quantifier. Every such
    decision lives here, derived from the read model, so neither the rules nor the coreference pass
    re-derive it.
    """

    _focus_by_scope: Dict[uuid.UUID, Optional[uuid.UUID]]
    """The focus referent id of each scope, keyed by the scope node's id (``None`` when the scope
    has no single subject, e.g. a set-of)."""

    _selected_quantities: FrozenSet[uuid.UUID] = frozenset()
    """Node ids of the quantities a query *selects* — an aggregation's measured attribute (the
    ``battery`` behind ``average(m.assigned_to.battery)``). A later mention of one (a WHERE on that
    same attribute) reduces to a bare *"the battery"*."""

    @classmethod
    def from_expression(
        cls, expression: SymbolicExpression, microplan: Microplan
    ) -> DiscourseModel:
        """
        :param expression: The root EQL expression.
        :param microplan: The shared plan read model (query plans are taken from it).
        :return: A discourse model mapping each query scope to its focus referent (``None`` for a
            scope with no single subject, e.g. a set-of).

        >>> from krrood.entity_query_language.verbalization.context import MicroplanningServices
        >>> robot = variable(Robot, [])
        >>> query = a(entity(robot).where(robot.battery > 50))
        >>> services = MicroplanningServices.from_expression(query)
        >>> model = DiscourseModel.from_expression(query, services.microplan)
        >>> scope = next(node for node in query._all_expressions_ if isinstance(node, Entity))
        >>> model.is_scope(scope)
        True
        >>> model.focus_of(scope) == robot._id_
        True

        A quantifier opens a scope too, focused on its bound variable:

        >>> robot = variable(Robot, [])
        >>> existential = exists(robot, robot.battery > 50)
        >>> services = MicroplanningServices.from_expression(existential)
        >>> model = DiscourseModel.from_expression(existential, services.microplan)
        >>> model.focus_of(existential) == robot._id_
        True
        """
        focus: Dict[uuid.UUID, Optional[uuid.UUID]] = {}
        selected_quantities: set[uuid.UUID] = set()
        for node in expression._all_expressions_:
            scope_rule = DiscourseScopeRule.most_applicable(node, microplan)
            if scope_rule is not None:
                focus[node._id_] = scope_rule.focus(node, microplan)
            leaf = (
                aggregation_leaf_attribute(node) if isinstance(node, Entity) else None
            )
            if leaf is not None:
                selected_quantities.add(leaf._id_)
        return cls(focus, frozenset(selected_quantities))

    def is_scope(self, source: Optional[FoldNode]) -> bool:
        """:return: ``True`` when *source* is a query node that opens a discourse scope.

        Only a real expression carries an ``_id_``; a synthetic coordination artifact (or ``None``)
        is never a scope.

        >>> EMPTY_DISCOURSE.is_scope(None)
        False
        """
        if not isinstance(source, SymbolicExpression):
            return False
        return source._id_ in self._focus_by_scope

    def focus_of(self, source: Optional[FoldNode]) -> Optional[uuid.UUID]:
        """:return: The focus referent of *source*'s scope, or ``None``.

        >>> EMPTY_DISCOURSE.focus_of(None) is None
        True
        """
        if not isinstance(source, SymbolicExpression):
            return None
        return self._focus_by_scope.get(source._id_)

    def is_selected_quantity(self, node_id: Optional[uuid.UUID]) -> bool:
        """:return: ``True`` when *node_id* is a quantity the query selects (e.g. an aggregation's
        measured attribute), so its repeat mention reduces to a bare *"the <attribute>"*.

        >>> EMPTY_DISCOURSE.is_selected_quantity(None)
        False
        """
        return node_id is not None and node_id in self._selected_quantities


EMPTY_DISCOURSE = DiscourseModel({})
"""A discourse model with no scopes — for local sub-tree realisation (opaque templates)."""
