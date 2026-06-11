"""
Clause **assemblers** — one realisation component per query clause (GROUP BY, HAVING,
ORDER BY).  Each is a self-contained, independently-testable unit that the
:class:`~krrood.entity_query_language.verbalization.grammar.assembly.query.QueryAssembler`
orchestrates as an ordered pipeline, and that the standalone GROUP BY / ORDER BY phrase
rules dispatch to directly — so a clause is rendered in exactly one place.

* :meth:`assemble` (inherited) — render the clause unconditionally (the standalone-node path).
* :meth:`clause` — render the clause *within a query* iff it is present (returns ``None``
  otherwise), the form the query-body orchestrator calls.

Reference: Reiter & Dale (2000) — aggregation / clause structuring; Gatt & Reiter (2009),
SimpleNLG — surface realisation.
"""

from __future__ import annotations

from typing_extensions import Any, Optional

from krrood.entity_query_language.query.query import SetOf
from krrood.entity_query_language.verbalization.fragments.base import (
    oxford_and,
    PhraseFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.grammar.assembly.base import Assembler
from krrood.entity_query_language.verbalization.grammar.planning.clauses import (
    GroupedByPlanner,
    GroupPlan,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Conjunctions,
    Copulas,
    Keywords,
    Punctuation,
    SortDirections,
)


class GroupedByAssembler(Assembler[Any, GroupPlan]):
    """*"grouped by <keys>"* — or *"and the <aggregated> are grouped by <keys>"* in a query."""

    planner = GroupedByPlanner

    def realize(self, node, plan: GroupPlan) -> VerbFragment:
        if not plan.has_keys:
            return Keywords.GROUPED.as_fragment()
        groups_phrase = self._keys_phrase(plan.keys)
        aggregated_frags = [
            self.ctx.child(expr, number=Number.PLURAL) for expr in plan.aggregated
        ]
        if aggregated_frags and not isinstance(node, SetOf):
            aggregated_phrase = oxford_and(
                aggregated_frags, Conjunctions.AND.as_fragment()
            )
            return PhraseFragment(
                parts=[
                    Conjunctions.AND.as_fragment(),
                    Articles.THE.as_fragment(),
                    aggregated_phrase,
                    Copulas.ARE.as_fragment(),
                    Keywords.GROUPED_BY.as_fragment(),
                    groups_phrase,
                ]
            )
        return PhraseFragment(parts=[Keywords.GROUPED_BY.as_fragment(), groups_phrase])

    def clause(self, node) -> Optional[VerbFragment]:
        """The in-query GROUP BY clause, or ``None`` when there are no group keys."""
        plan = self.plan(node)
        return self.realize(node, plan) if plan.has_keys else None

    def _keys_phrase(self, variables) -> VerbFragment:
        return PhraseFragment(
            parts=[self.ctx.child(variable) for variable in variables],
            separator=Punctuation.COMMA.text + " ",
        )


class OrderedByAssembler(Assembler[Any, None]):
    """*"ordered by <variable> (ascending|descending)"*. Realisation-only (no plan)."""

    def realize(self, node, plan: None = None) -> VerbFragment:
        # *node* is "ordered-like": an OrderedBy expression or an OrderedByBuilder, both
        # exposing ``.variable`` and ``.descending``.
        direction = (
            SortDirections.DESCENDING if node.descending else SortDirections.ASCENDING
        )
        paren = PhraseFragment(
            parts=[
                Punctuation.OPEN_PAREN.as_fragment(),
                direction.as_fragment(),
                Punctuation.CLOSE_PAREN.as_fragment(),
            ],
            separator="",
        )
        return PhraseFragment(
            parts=[
                Keywords.ORDERED_BY.as_fragment(),
                self.ctx.child(node.variable),
                paren,
            ]
        )

    def clause(self, query) -> Optional[VerbFragment]:
        """The in-query ORDER BY clause, or ``None`` when the query is unordered."""
        builder = query._ordered_by_builder_
        return self.realize(builder) if builder is not None else None


class HavingAssembler(Assembler[Any, None]):
    """*"having <condition>"* (compact comparators). Realisation-only (no plan)."""

    def realize(self, node, plan: None = None) -> VerbFragment:
        with self.ctx.config.compact_predicates_scope():
            having_frag = self.ctx.child(node._having_expression_.condition)
        return PhraseFragment(parts=[Keywords.HAVING.as_fragment(), having_frag])

    def clause(self, query) -> Optional[VerbFragment]:
        """The in-query HAVING clause, or ``None`` when there is no HAVING."""
        return self.realize(query) if query._having_expression_ is not None else None
