"""
Grammar rules for attribute / index / call chains — one guarded :class:`PhraseRule` per surface
form, dispatched by ``select``. The guards are mutually exclusive, so at most one fires and no
ordering between them is needed: the precedence *"the bare-plural noun phrase wins over the
predicative"* lives in :meth:`~krrood.entity_query_language.verbalization.grammar.chain.planner.ChainPlan.renders_as_plural_attribute`,
which both the plural and boolean rules consult. The guarded forms outrank the unguarded possessive
fallback; adding a chain form is a new guarded rule here, with no change to existing rules.
"""

from __future__ import annotations

from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.grammar.chain.assembler import (
    ChainAssembler,
)
from krrood.entity_query_language.verbalization.grammar.chain.planner import (
    ChainPlanner,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    RuleContext,
)


class PluralChainAttributeRule(PhraseRule):
    """Plural single-attribute chain → bare plural *"attributes of Roots"*.

    >>> verbalize_expression(sum(variable(Robot, []).battery))
    'the sum of batteries of Robots'
    """

    construct = MappedVariable

    def when(self, node: MappedVariable, context: RuleContext) -> bool:
        """:return: ``True`` when the chain is a plural single attribute on a variable.

        Its contribution is the guard that admits this rule: ``salary`` over a plural ``Employees``
        variable is a plural single attribute, so this rule fires and the chain surfaces as the bare
        plural *"salaries of Employees"* instead of the possessive fallback.

        >>> verbalize_expression(count(variable(Employee, []).salary))
        'the number of salaries of Employees'
        """
        plan = context.microplan.plan_for(node, ChainPlanner)
        return plan.renders_as_plural_attribute(context.number)

    def build(
        self, node: MappedVariable, context: RuleContext
    ) -> VerbalizationFragment:
        """:return: The bare plural *"attributes of Roots"*.

        Its contribution is the bare-plural surface itself: it builds the *"batteries of Robots"*
        span seen inside *"the sum of …"*, pluralising both attribute and root rather than emitting a
        single possessive.
        """
        plan = context.microplan.plan_for(node, ChainPlanner)
        return ChainAssembler(context).plural_attribute(plan)


class BooleanAttributeChainRule(PhraseRule):
    """Boolean-terminal chain → predicative *"<navigation> is <attribute>"* (unless the bare-plural
    attribute form takes precedence).

    >>> verbalize_expression(variable(Task, []).completed)
    'a Task is completed'
    """

    construct = MappedVariable

    def when(self, node: MappedVariable, context: RuleContext) -> bool:
        """:return: ``True`` for a boolean-terminal chain that is not a bare-plural attribute.

        Its contribution is the guard that admits this rule: ``operational`` is a boolean terminal and
        not a bare-plural attribute, so this rule fires and the chain surfaces predicatively as *"a
        Robot is operational"* rather than as a possessive.

        >>> verbalize_expression(variable(Robot, []).operational)
        'a Robot is operational'
        """
        plan = context.microplan.plan_for(node, ChainPlanner)
        return plan.is_boolean_terminal and not plan.renders_as_plural_attribute(
            context.number
        )

    def build(
        self, node: MappedVariable, context: RuleContext
    ) -> VerbalizationFragment:
        """:return: The predicative *"<navigation> is <attribute>"*.

        Its contribution is the predicative surface itself: it builds the *"a Task is completed"*
        sentence, turning the boolean terminal into an *"is <attribute>"* predicate rather than a
        possessive noun phrase.
        """
        plan = context.microplan.plan_for(node, ChainPlanner)
        return ChainAssembler(context).boolean_predicative(plan)


class PossessiveChainRule(PhraseRule):
    """Any attribute / index / call chain → possessive path *"the attribute of the Root"*
    (the unguarded fallback form).

    >>> verbalize_expression(variable(Task, []).name)
    'the name of a Task'
    """

    construct = MappedVariable

    def build(
        self, node: MappedVariable, context: RuleContext
    ) -> VerbalizationFragment:
        """:return: The possessive path *"the attribute of the Root"*.

        Its contribution is the possessive fallback surface: with no guarded form matching, it builds
        the *"the battery of a Robot"* possessive path — the default when the chain is neither a
        bare-plural attribute nor a boolean terminal.

        >>> verbalize_expression(variable(Robot, []).battery)
        'the battery of a Robot'
        """
        plan = context.microplan.plan_for(node, ChainPlanner)
        return ChainAssembler(context).possessive(plan)
