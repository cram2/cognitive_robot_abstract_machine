from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Dict, Tuple, Type

from krrood.entity_query_language.verbalization.grammar.framework.planner import (
    Planner,
    TPlan,
    TSymbolicExpression,
)


@dataclass
class Microplan:
    """
    A read model over the planners: each node's plan is computed once and cached.

    Plans are pure information — the *what to say* decisions, not an action plan — so they are
    computed once here and shared by every consumer (assemblers building fragments, and the
    rendering passes that read provenance) rather than rebuilt on demand. The store has no
    behaviour of its own: it neither builds fragments nor mutates the EQL; it only memoises.

    This removes the current double-compute (the chain planner runs in both a rule's ``when`` and
    its ``build``) and gives later stages a single source of truth to consult.
    """

    _cache: Dict[Tuple[int, type], object] = field(default_factory=dict)
    """
    Memoised plans keyed by ``(id(node), planner)`` — node *identity* (not equality,
    which EQL overrides to build comparators) paired with the planner that produced the
    plan.
    """

    def plan_for(
        self,
        node: TSymbolicExpression,
        planner: Type[Planner[TSymbolicExpression, TPlan]],
    ) -> TPlan:
        """
        :param node: The EQL node to plan.
        :param planner: The planner class that analyses *node*.
        :return: *node*'s plan, computed once and cached. The key is ``(node, planner)`` because one
            node may be analysed by several planners (a query has both a ``QueryPlan`` and a
            ``GroupPlan``).
        """
        key = (id(node), planner)
        if key not in self._cache:
            self._cache[key] = planner(node).plan()
        return self._cache[key]
