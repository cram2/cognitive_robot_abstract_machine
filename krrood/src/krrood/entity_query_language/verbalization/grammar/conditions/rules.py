from __future__ import annotations

from krrood.entity_query_language.core.base_expressions import Filter
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    AND,
    OR,
    Not,
    flatten_operands,
)
from krrood.entity_query_language.operators.logical_quantifiers import Exists, ForAll
from krrood.entity_query_language.verbalization.fragments.base import (
    Fragment,
    oxford_comma,
    PhraseFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Number,
    Separator,
)
from krrood.entity_query_language.verbalization.grammar.chain.assembler import (
    ChainAssembler,
)
from krrood.entity_query_language.verbalization.grammar.chain.planner import (
    ChainPlanner,
)
from krrood.entity_query_language.verbalization.grammar.conditions.assembler import (
    ConditionAssembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    is_boolean_attribute_chain,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    RuleContext,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    build_between,
    RangeFold,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    Keywords,
    Logicals,
    Punctuation,
)


class ComparatorRule(PhraseRule):
    """``<left> <operator> <right>`` — e.g. *"is greater than 50"*.

    >>> verbalize_expression(variable(Robot, []).battery > 50)
    'the battery of a Robot is greater than 50'
    """

    construct = Comparator
    name = "comparator"

    def build(self, node: Comparator, context: RuleContext) -> Fragment:
        return ConditionAssembler(context).predicate(node)


class AndRule(PhraseRule):
    """Conjunction *"a, b, and c"* (Oxford comma); flattens nested ANDs. A complementary low/high
    pair on one chain folds into *"… is between low and high"* — handled by the shared conjunct
    rendering, so there is no separate range rule.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(and_(robot.battery > 50, robot.name == 'x'))
    "the battery of a Robot is greater than 50, and the name of the Robot is 'x'"
    >>> verbalize_expression(and_(robot.battery > 10, robot.battery < 90))
    'the battery of a Robot is between 10, and 90'
    """

    construct = AND
    name = "and"

    def build(self, node: AND, context: RuleContext) -> Fragment:
        parts = ConditionAssembler(context).as_statements(flatten_operands(node, AND))
        if len(parts) == 1:
            return parts[0]
        return oxford_comma(parts, Conjunctions.AND.as_fragment())


class RangeFoldRule(PhraseRule):
    """A folded lower/upper bound pair → *"<chain> is between low and high"*.

    A :class:`RangeFold` is the coordination microplanning artifact produced when two complementary
    bounds on one chain are reduced (by :func:`fold_range_pairs`). Making it a first-class
    verbalizable means any caller that has reduced its conjuncts renders the result through the
    normal recursion (``context.child``) — it never has to know a folding helper exists.
    """

    construct = RangeFold
    name = "range-fold"

    def build(self, node: RangeFold, context: RuleContext) -> Fragment:
        return build_between(
            context.child(node.chain_expression),
            context.child(node.lower_expression),
            context.child(node.upper_expression),
            compact=context.configuration.compact_predicates,
        )


class OrRule(PhraseRule):
    """Disjunction *"either a, b, or c"*; flattens nested ORs.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(or_(robot.battery > 50, robot.battery < 10))
    'either the battery of a Robot is greater than 50, or the battery of the Robot is less than 10'
    """

    construct = OR
    name = "or"

    def build(self, node: OR, context: RuleContext) -> Fragment:
        parts = [context.child(conjunct) for conjunct in flatten_operands(node, OR)]
        if len(parts) == 1:
            return parts[0]
        # "either a, b, or c": the head items are comma-joined, then a trailing comma that the
        # orthography pass hugs to the last head item — no separator="" bookkeeping here.
        head = PhraseFragment(parts=parts[:-1], separator=Separator.COMMA)
        return PhraseFragment(
            parts=[
                Logicals.EITHER.as_fragment(),
                head,
                Punctuation.COMMA.as_fragment(),
                Conjunctions.OR.as_fragment(),
                parts[-1],
            ]
        )


class NotRule(PhraseRule):
    """Generic negation *"not (<child>)"* (specialised by the guarded Not rules below).

    >>> verbalize_expression(Not(IsReachable(variable(Location, []))))
    'not (a Location is reachable)'
    """

    construct = Not
    name = "not"

    def build(self, node: Not, context: RuleContext) -> Fragment:
        child_fragment = context.child(node._child_)
        # The parens glue to the child via the orthography pass → "not (child)".
        return PhraseFragment(
            parts=[
                Logicals.NOT.as_fragment(),
                Punctuation.OPEN_PAREN.as_fragment(),
                child_fragment,
                Punctuation.CLOSE_PAREN.as_fragment(),
            ]
        )


class NotComparatorRule(PhraseRule):
    """Inline negated comparator *"a is not greater than b"* (Not over a Comparator).

    >>> verbalize_expression(Not(variable(Robot, []).battery > 50))
    'the battery of a Robot is not greater than 50'
    """

    construct = Not
    name = "not-comparator"

    def when(self, node: Not, context: RuleContext) -> bool:
        return isinstance(node._child_, Comparator)

    def build(self, node: Not, context: RuleContext) -> Fragment:
        return ConditionAssembler(context).predicate(node._child_, negated=True)


class NotBooleanAttributeRule(PhraseRule):
    """Negated boolean attribute chain *"<nav> is not <attribute>"* (Not over a bool-attribute chain).

    >>> verbalize_expression(Not(variable(Task, []).completed))
    'a Task is not completed'
    """

    construct = Not
    name = "not-bool-attribute"

    def when(self, node: Not, context: RuleContext) -> bool:
        return is_boolean_attribute_chain(node._child_)

    def build(self, node: Not, context: RuleContext) -> Fragment:
        plan = context.microplan.plan_for(node._child_, ChainPlanner)
        return ChainAssembler(context).boolean_predicative(plan, negated=True)


class ForAllRule(PhraseRule):
    """*"for all <plural var>, <condition>"*.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(for_all(robot, robot.battery > 0))
    'for all Robots, the battery of the Robot is greater than 0'
    """

    construct = ForAll
    name = "for-all"

    def build(self, node: ForAll, context: RuleContext) -> Fragment:
        variable_fragment = context.child(node.variable, number=Number.PLURAL)
        condition_fragment = context.child(node.condition)
        return PhraseFragment(
            parts=[
                Logicals.FOR_ALL.as_fragment(),
                variable_fragment,
                Punctuation.COMMA.as_fragment(),
                condition_fragment,
            ]
        )


class ExistsRule(PhraseRule):
    """*"there exists <variable> such that <condition>"*.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(exists(robot, robot.battery > 0))
    'there exists a Robot such that the battery of the Robot is greater than 0'
    """

    construct = Exists
    name = "exists"

    def build(self, node: Exists, context: RuleContext) -> Fragment:
        variable_fragment = context.child(node.variable)
        condition_fragment = context.child(node.condition)
        return PhraseFragment(
            parts=[
                Logicals.THERE_EXISTS.as_fragment(),
                variable_fragment,
                Keywords.SUCH_THAT.as_fragment(),
                condition_fragment,
            ]
        )


class FilterRule(PhraseRule):
    """Transparent wrapper (Where / Having) → delegate to the condition."""

    construct = Filter
    name = "filter"

    def build(self, node: Filter, context: RuleContext) -> Fragment:
        return context.child(node.condition)
