from __future__ import annotations

from krrood.entity_query_language.query.quantifiers import ResultQuantifier
from krrood.entity_query_language.query.query import Entity, SetOf
from krrood.entity_query_language.verbalization.fragments.base import Fragment
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    RuleContext,
)
from krrood.entity_query_language.verbalization.grammar.query.assembler import (
    QueryAssembler,
)


class TopLevelEntityRule(PhraseRule):
    """Top-level Entity → imperative *"Find …"* (only at query_depth 0).

    >>> verbalize_expression(an(entity(variable(Robot, []))))
    'Find a Robot'
    """

    construct = Entity
    name = "top-level-entity"
    enters_query_scope = True

    def when(self, node: Entity, context: RuleContext) -> bool:
        return context.configuration.query_depth == 0 and not context.inline

    def build(self, node: Entity, context: RuleContext) -> Fragment:
        return QueryAssembler(context).assemble(node)


class NestedEntityRule(PhraseRule):
    """Nested Entity → noun phrase (query_depth > 0); never emits *"Find"*.

    >>> worker = variable(Worker, [])
    >>> verbalize_expression(
    ...     an(entity(worker).where(contains(worker.tasks, an(entity(variable(Task, []))))))
    ... )
    'Find a Worker whose tasks contains a Task'
    """

    construct = Entity
    name = "nested-entity"
    enters_query_scope = True

    def when(self, node: Entity, context: RuleContext) -> bool:
        return context.configuration.query_depth > 0 and not context.inline

    def build(self, node: Entity, context: RuleContext) -> Fragment:
        return QueryAssembler(context).assemble_nested(node)


class SetOfRule(PhraseRule):
    """SetOf → *"Find v1 and v2 such that …"* (a search), or *"Report …"* / *"For each … report
    …"* when the selection computes aggregates (a report).

    >>> verbalize_expression(an(set_of(variable(Robot, []), variable(Task, []))))
    'Find a Robot and a Task'
    """

    construct = SetOf
    name = "set-of"
    enters_query_scope = True

    def build(self, node: SetOf, context: RuleContext) -> Fragment:
        return QueryAssembler(context).assemble_set_of(node)


class InlineEntityRule(PhraseRule):
    """Entity in chain-root position → the inline-noun form (*"a Robot"*, no *"Find"*, its WHERE
    deferred to the binding scope). Selected when the fold recurses with ``inline`` — so the chain
    assembler just recurses the root and the dispatch picks the form, rather than type-checking and
    calling the query assembler by hand."""

    construct = Entity
    name = "inline-entity"

    def when(self, node: Entity, context: RuleContext) -> bool:
        return context.inline

    def build(self, node: Entity, context: RuleContext) -> Fragment:
        return QueryAssembler(context).inline_noun(node)


class ResultQuantifierRule(PhraseRule):
    """Transparent wrapper (An / The / …) → delegate to the child, forwarding the render context
    (an ``inline`` chain root stays inline through the wrapper)."""

    construct = ResultQuantifier
    name = "result-quantifier"

    def build(self, node: ResultQuantifier, context: RuleContext) -> Fragment:
        return context.child(node._child_, inline=context.inline)
