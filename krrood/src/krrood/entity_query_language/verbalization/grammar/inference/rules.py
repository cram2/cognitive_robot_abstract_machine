from __future__ import annotations

from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.verbalization.fragments.base import Fragment
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    RuleContext,
)
from krrood.entity_query_language.verbalization.grammar.inference.assembler import (
    InferenceAssembler,
)
from krrood.entity_query_language.verbalization.grammar.inference.planner import (
    InferencePlanner,
)
from krrood.entity_query_language.verbalization.grammar.query.rules import (
    TopLevelEntityRule,
)


class InferenceRuleRule(TopLevelEntityRule):
    """Top-level inference-rule Entity → ``IF … THEN …`` block.

    A refinement of :class:`TopLevelEntityRule`: it applies exactly when that rule does *and* the
    entity is an inference rule, so ``select`` prefers it (more-derived class) over the plain
    top-level form without any tiebreak. Unlike the plain form it does not enter query scope.
    """

    name = "inference-rule"
    enters_query_scope = False

    def when(self, node: Entity, context: RuleContext) -> bool:
        return super().when(node, context) and InferencePlanner.can_handle(node)

    def build(self, node: Entity, context: RuleContext) -> Fragment:
        return InferenceAssembler(context).assemble(node)
