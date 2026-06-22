from __future__ import annotations

from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.verbalization.fragments.base import Fragment
from krrood.entity_query_language.verbalization.grammar.chain.assembler import (
    ChainAssembler,
)
from krrood.entity_query_language.verbalization.grammar.chain.planner import ChainPlanner
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    RuleContext,
)

# One guarded rule per surface form, dispatched by ``select``. The guards are mutually exclusive,
# so at most one fires and no ordering between them is needed: the precedence "the bare-plural
# noun phrase wins over the predicative" lives in ``ChainPlan.renders_as_plural_attribute``, which
# both the plural and boolean rules consult. The guarded forms outrank the unguarded possessive
# fallback. Adding a chain form is a new guarded rule here — no existing rule changes.


class PluralChainAttributeRule(PhraseRule):
    """Plural single-attribute chain → bare plural *"attributes of Roots"*.

    >>> verbalize_expression(sum(variable(Robot, []).battery))
    'the sum of batteries of Robots'
    """

    construct = MappedVariable
    name = "chain-plural-attribute"

    def when(self, node: MappedVariable, context: RuleContext) -> bool:
        plan = context.microplan.plan_for(node, ChainPlanner)
        return plan.renders_as_plural_attribute(context.number)

    def build(self, node: MappedVariable, context: RuleContext) -> Fragment:
        plan = context.microplan.plan_for(node, ChainPlanner)
        return ChainAssembler(context).plural_attribute(plan)


class BooleanAttributeChainRule(PhraseRule):
    """Boolean-terminal chain → predicative *"<navigation> is <attribute>"* (unless the bare-plural
    attribute form takes precedence).

    >>> verbalize_expression(variable(Task, []).completed)
    'a Task is completed'
    """

    construct = MappedVariable
    name = "chain-boolean-attribute"

    def when(self, node: MappedVariable, context: RuleContext) -> bool:
        plan = context.microplan.plan_for(node, ChainPlanner)
        return plan.is_boolean_terminal and not plan.renders_as_plural_attribute(
            context.number
        )

    def build(self, node: MappedVariable, context: RuleContext) -> Fragment:
        plan = context.microplan.plan_for(node, ChainPlanner)
        return ChainAssembler(context).boolean_predicative(plan)


class PossessiveChainRule(PhraseRule):
    """Any attribute / index / call chain → possessive path *"the attribute of the Root"*
    (the unguarded fallback form).

    >>> verbalize_expression(variable(Task, []).name)
    'the name of a Task'
    """

    construct = MappedVariable
    name = "chain-possessive"

    def build(self, node: MappedVariable, context: RuleContext) -> Fragment:
        plan = context.microplan.plan_for(node, ChainPlanner)
        return ChainAssembler(context).possessive(plan)
