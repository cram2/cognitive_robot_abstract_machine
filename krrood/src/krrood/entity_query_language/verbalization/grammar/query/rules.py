from __future__ import annotations

from krrood.entity_query_language.query.quantifiers import ResultQuantifier
from krrood.entity_query_language.query.query import Entity, SetOf
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    RuleContext,
)
from krrood.entity_query_language.verbalization.grammar.query.assembler import (
    QueryAssembler,
)


class TopLevelEntityRule(PhraseRule):
    """
    Top-level Entity → imperative *"Find …"* (only at query_depth 0).

    >>> verbalize_expression(an(entity(variable(Robot, []))))
    'Find a Robot'
    """

    construct = Entity
    enters_query_scope = True

    def when(self, node: Entity, context: RuleContext) -> bool:
        """:return: ``True`` only for a top-level (query depth 0), non-inline entity.

        This is the decision that routes the entity to the imperative *"Find …"* surface rather than
        the nested noun-phrase form — it is the outermost entity, so the result opens with the verb:

        >>> verbalize_expression(an(entity(variable(Robot, []))))
        'Find a Robot'
        """
        return context.configuration.query_depth == 0 and not context.inline

    def build(self, node: Entity, context: RuleContext) -> VerbalizationFragment:
        """:return: the imperative *"Find …"* form built by the query assembler.

        It produces the whole imperative sentence by delegating to the query assembler, which emits
        the leading *"Find"* and weaves in the selection and its restrictions:

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(an(entity(robot).where(robot.battery > 50)))
        'Find a Robot whose battery is greater than 50'
        """
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
    enters_query_scope = True

    def when(self, node: Entity, context: RuleContext) -> bool:
        """:return: ``True`` only for a nested (query depth > 0), non-inline entity.

        This is the decision that routes the inner entity to the bare noun-phrase surface — being
        nested, it contributes only the trailing *"a Task"* and never a second *"Find"*:

        >>> worker = variable(Worker, [])
        >>> verbalize_expression(
        ...     an(entity(worker).where(contains(worker.tasks, an(entity(variable(Task, []))))))
        ... )
        'Find a Worker whose tasks contains a Task'
        """
        return context.configuration.query_depth > 0 and not context.inline

    def build(self, node: Entity, context: RuleContext) -> VerbalizationFragment:
        """:return: the noun-phrase form built by the query assembler (never *"Find …"*).

        For the nested entity it produces only the inner noun phrase *"a Task"*, leaving the outer
        query to supply the surrounding *"Find a Worker whose tasks contains …"*:

        >>> worker = variable(Worker, [])
        >>> verbalize_expression(
        ...     an(entity(worker).where(contains(worker.tasks, an(entity(variable(Task, []))))))
        ... )
        'Find a Worker whose tasks contains a Task'
        """
        return QueryAssembler(context).assemble_nested(node)


class SetOfRule(PhraseRule):
    """
    SetOf → *"Find v1 and v2 such that …"* (a search), or *"Report …"* / *"For each …
    report …"* when the selection computes aggregates (a report).

    >>> verbalize_expression(an(set_of(variable(Robot, []), variable(Task, []))))
    'Find a Robot and a Task'
    """

    construct = SetOf
    enters_query_scope = True

    def build(self, node: SetOf, context: RuleContext) -> VerbalizationFragment:
        """:return: the set-of form built by the query assembler.

        It produces the whole set-of sentence by delegating to the query assembler, which here emits
        *"Find"* followed by the coordinated selection *"a Robot and a Task"*:

        >>> verbalize_expression(an(set_of(variable(Robot, []), variable(Task, []))))
        'Find a Robot and a Task'
        """
        return QueryAssembler(context).assemble_set_of(node)


class InlineEntityRule(PhraseRule):
    """
    Entity in chain-root position → the inline-noun form (*"a Robot"*, no *"Find"*, its
    WHERE deferred to the binding scope).

    Selected when the fold recurses with ``inline`` — so the chain assembler just
    recurses the root and the dispatch picks the form, rather than type-checking and
    calling the query assembler by hand.
    """

    construct = Entity

    def when(self, node: Entity, context: RuleContext) -> bool:
        """:return: ``True`` only when the fold recurses in inline (chain-root) position.

        This is the decision that routes the entity to the inline-noun surface — sitting at the root
        of the ``.name`` chain, it contributes only *"a Robot"*, so the chain reads *"the name of a
        Robot"* rather than an imperative *"Find"* sentence:

        >>> verbalize_expression(an(entity(variable(Robot, []))).name)
        'the name of a Robot'
        """
        return context.inline

    def build(self, node: Entity, context: RuleContext) -> VerbalizationFragment:
        """:return: the inline-noun form *"a Robot"* (no *"Find"*; the entity's WHERE deferred).

        It produces only the chain-root noun *"a Robot"* (the *"the name of"* comes from the
        enclosing attribute chain), with the entity's WHERE deferred to the binding scope:

        >>> verbalize_expression(an(entity(variable(Robot, []))).name)
        'the name of a Robot'
        """
        return QueryAssembler(context).inline_noun(node)


class ResultQuantifierRule(PhraseRule):
    """
    Transparent wrapper (An / The / …) → delegate to the child, forwarding the render
    context (an ``inline`` chain root stays inline through the wrapper).
    """

    construct = ResultQuantifier

    def build(
        self, node: ResultQuantifier, context: RuleContext
    ) -> VerbalizationFragment:
        """:return: the child's fragment, forwarding the render context (the wrapper is transparent).

        It adds no words of its own: the entire *"Find the unique Robot"* is produced by the wrapped
        entity, and this method only passes the render context straight through:

        >>> verbalize_expression(the(entity(variable(Robot, []))))
        'Find the unique Robot'
        """
        return context.child(node._child_, inline=context.inline)
