from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from typing_extensions import ClassVar, Generic, Optional, Type

from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    RuleContext,
)
from krrood.entity_query_language.verbalization.grammar.framework.planner import (
    Planner,
    TSymbolicExpression,
    TPlan,
)


@dataclass
class Assembler(Generic[TSymbolicExpression, TPlan], SubClassSafeGeneric):
    """
    Realise an EQL *node* into a fragment, planning it first via the paired planner.

    An assembler owns its paired planner: ``assemble`` plans the node, then realises the plan
    into a fragment. A realisation-only construct (e.g. an ORDER BY / HAVING clause, which has
    nothing to decide) sets ``planner = None`` and receives ``plan=None``.

    Reference: :cite:t:`gatt2009simplenlg` — surface realisation as a dedicated stage.
    """

    context: RuleContext
    """The per-node context (recursion entry and microplanning services)."""

    planner: ClassVar[Optional[Type[Planner]]] = None
    """The paired planner (set per family); ``None`` for realisation-only assemblers."""

    def plan(self, node: TSymbolicExpression) -> Optional[TPlan]:
        """
        :param node: The node to plan.
        :return: The plan from the paired planner via the shared read model (computed once per
            node), or ``None`` when there is nothing to plan.
        """
        if self.planner is None:
            return None
        return self.context.microplan.plan_for(node, self.planner)

    def assemble(self, node: TSymbolicExpression) -> VerbalizationFragment:
        """
        Plan *node*, then realise the plan.

        :param node: The node to assemble.
        :return: The fragment for *node*.
        """
        return self.realize(node, self.plan(node))

    @abstractmethod
    def realize(self, node: TSymbolicExpression, plan: TPlan) -> VerbalizationFragment:
        """
        :param node: The node to realise.
        :param plan: The plan computed for *node* (``None`` for realisation-only assemblers).
        :return: The fragment for *node* built from its *plan*.
        """
