from __future__ import annotations

from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
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
    """
    Top-level inference-rule Entity → ``IF … THEN …`` block.

    A refinement of :class:`TopLevelEntityRule`: it applies exactly when that rule does
    *and* the entity is an inference rule, so ``select`` prefers it (more-derived class)
    over the plain top-level form without any tiebreak. Unlike the plain form it does
    not enter query scope.
    """

    enters_query_scope = False

    def when(self, node: Entity, context: RuleContext) -> bool:
        """:return: ``True`` for a top-level entity whose selected variable is an instantiated
        variable (an inference rule).

        >>> from krrood.entity_query_language.factories import inference
        >>> connection = variable(FixedConnection, [])
        >>> drawer = inference(Drawer)(container=connection.parent, handle=connection.child)
        >>> verbalize_expression(entity(drawer)).startswith("If")
        True
        """
        return super().when(node, context) and InferencePlanner.can_handle(node)

    def build(self, node: Entity, context: RuleContext) -> VerbalizationFragment:
        """:return: the ``IF … THEN …`` block built by the inference assembler.

        >>> from krrood.entity_query_language.factories import inference
        >>> connection = variable(FixedConnection, [])
        >>> drawer = inference(Drawer)(container=connection.parent, handle=connection.child)
        >>> verbalize_expression(entity(drawer).where(
        ...     connection.parent == variable(Container, []))).startswith(
        ...     "If there's a FixedConnection")
        True
        """
        return InferenceAssembler(context).assemble(node)
