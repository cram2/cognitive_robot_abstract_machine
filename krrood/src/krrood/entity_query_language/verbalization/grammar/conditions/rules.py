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
    RoleFragment,
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
from krrood.entity_query_language.verbalization.grammar.conditions.predication import (
    coindexed_operator,
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
    coindexed_natural_parts,
    CoindexedFold,
    RangeFold,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    CoindexedPhrases,
    Conjunctions,
    Keywords,
    Logicals,
    Prepositions,
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
        """Say the comparator as a standalone predicate.

        Delegating to the predicate assembler is what produces the whole *the battery of a Robot is
        greater than 50* sentence; this rule contributes the dispatch from a :class:`Comparator` node
        to that predicate form.

        >>> verbalize_expression(variable(Robot, []).battery > 50)
        'the battery of a Robot is greater than 50'
        """
        return ConditionAssembler(context).predicate(node)


class AndRule(PhraseRule):
    """Conjunction *"a, b, and c"* (Oxford comma); flattens nested ANDs. A complementary low/high
    pair on one chain folds into *"… is between low and high"* — handled by the shared conjunct
    rendering, so there is no separate range rule.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(and_(robot.battery > 50, robot.name == 'x'))
    "the battery of a Robot is greater than 50, and the name of the Robot is 'x'"
    >>> verbalize_expression(and_(robot.battery > 10, robot.battery < 90))
    'the battery of a Robot is between 10 and 90'
    """

    construct = AND
    name = "and"

    def build(self, node: AND, context: RuleContext) -> Fragment:
        """Say the flattened conjuncts, comma-joined with a trailing *"and"*.

        It owns the *, and* coordination between the two conjuncts of the example; the conjunct
        clauses themselves come from the shared statement assembler.

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(and_(robot.battery > 50, robot.name == 'x'))
        "the battery of a Robot is greater than 50, and the name of the Robot is 'x'"
        """
        parts = ConditionAssembler(context).as_statements(flatten_operands(node, AND))
        if len(parts) == 1:
            return parts[0]
        # Conjuncts are independent clauses, so a two-clause coordination keeps its comma.
        return oxford_comma(parts, Conjunctions.AND.as_fragment(), pair_comma=True)


class RangeFoldRule(PhraseRule):
    """A folded lower/upper bound pair → *"<chain> is between low and high"*.

    A :class:`RangeFold` is the coordination microplanning artifact produced when two complementary
    bounds on one chain are reduced (by the conjunct reducer). Making it a first-class verbalizable
    means any caller that has reduced its conjuncts renders the result through the normal recursion
    (``context.child``) — it never has to know a folding helper exists.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(and_(robot.battery > 10, robot.battery < 90))
    'the battery of a Robot is between 10 and 90'
    """

    construct = RangeFold
    name = "range-fold"

    def build(self, node: RangeFold, context: RuleContext) -> Fragment:
        """Say the folded pair as *"<chain> is between low and high"*.

        It owns the *is between 10 and 90* span, emitting the *between … and …* frame over the fold's
        two bounds — which is why the reduced pair reads as one clause rather than two comparisons.

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(and_(robot.battery > 10, robot.battery < 90))
        'the battery of a Robot is between 10 and 90'
        """
        return build_between(
            context.child(node.chain_expression),
            context.child(node.lower_expression),
            context.child(node.upper_expression),
            compact=context.configuration.compact_predicates,
        )


class CoindexedFoldRule(PhraseRule):
    """A group of co-indexed comparators (``p.begin.X == p.end.X`` for several ``X``) reduced by
    :func:`fold_coindexed_groups` → one factored clause that says the shared structure once.

    Two surface forms, chosen here (recognition stays pure):

    * **natural** — a pure-equality fold over *sibling* prefixes (``p.begin`` / ``p.end``) reads
      *"the begin and end of its period have the same month and year"*;
    * **faithful** — every other foldable case (a non-equality operator, or prefixes that are not
      siblings) reads *"the month and year of the begin of its period are equal to those of the
      end of its period"*.
    """

    construct = CoindexedFold
    name = "coindexed-fold"

    def build(self, node: CoindexedFold, context: RuleContext) -> Fragment:
        """Say the factored clause once — the natural *"have the same"* form over sibling prefixes.

        Detecting sibling prefixes selects the natural branch here (the faithful branch would instead
        read *… are equal to those of …*). The two identical comparisons fold to one clause, and the
        leaf they both compare — *name* — is listed once, not *"name and name"*:

        >>> pair = variable(LoveBirds, [])
        >>> verbalize_expression(
        ...     and_(pair.bird_1.name == pair.bird_2.name, pair.bird_1.name == pair.bird_2.name)
        ... )
        'the bird_1 and bird_2 of a LoveBirds have the same name'
        """
        terminals = oxford_comma(
            [self._attribute(pair) for pair in node.terminals],
            Conjunctions.AND.as_fragment(),
        )
        natural = coindexed_natural_parts(node)
        if natural is not None:
            hops = oxford_comma(
                [self._attribute(natural.left_hop), self._attribute(natural.right_hop)],
                Conjunctions.AND.as_fragment(),
            )
            return PhraseFragment(
                parts=[
                    Articles.THE.as_fragment(),
                    hops,
                    Prepositions.OF.as_fragment(),
                    context.child(natural.shared_prefix_expression),
                    CoindexedPhrases.HAVE_THE_SAME.as_fragment(),
                    terminals,
                ]
            )
        return PhraseFragment(
            parts=[
                Articles.THE.as_fragment(),
                terminals,
                Prepositions.OF.as_fragment(),
                context.child(node.left_prefix_expression),
                coindexed_operator(node.operation),
                CoindexedPhrases.THOSE_OF.as_fragment(),
                context.child(node.right_prefix_expression),
            ]
        )

    @staticmethod
    def _attribute(pair: tuple) -> RoleFragment:
        """:return: The role-tagged attribute fragment for a ``(name, owner)`` hop.

        It supplies just one attribute noun such as *name*; :meth:`build` calls it per hop to build
        the *bird_1 and bird_2* and *name and name* lists of the factored clause.

        >>> from krrood.entity_query_language.verbalization.fragments.base import (
        ...     flatten_fragment_to_plain_text,
        ... )
        >>> flatten_fragment_to_plain_text(CoindexedFoldRule._attribute(('name', Bird)))
        'name'
        """
        name, owner = pair
        return RoleFragment.for_attribute(owner, name)


class OrRule(PhraseRule):
    """Disjunction *"either a, b, or c"*; flattens nested ORs.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(or_(robot.battery > 50, robot.battery < 10))
    'either the battery of a Robot is greater than 50, or the battery of the Robot is less than 10'
    """

    construct = OR
    name = "or"

    def build(self, node: OR, context: RuleContext) -> Fragment:
        """Say the flattened disjuncts as *"either a, b, or c"*.

        It owns the *either …, or …* framing around the disjuncts of the example — the leading
        *either* and the comma-*or* before the last — while the disjunct clauses come from the
        recursion.

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(or_(robot.battery > 50, robot.battery < 10))
        'either the battery of a Robot is greater than 50, or the battery of the Robot is less than 10'
        """
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
        """Wrap the child in *"not (<child>)"* via the orthography pass.

        It owns the *not (…)* wrapper of the example — the leading *not* and the parentheses — around
        the *a Location is reachable* child the recursion renders.

        >>> verbalize_expression(Not(IsReachable(variable(Location, []))))
        'not (a Location is reachable)'
        """
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
        """Fires when the negation wraps a comparator.

        Detecting that the negation wraps a :class:`Comparator` is the gate that selects this rule
        over the generic :class:`NotRule`, which is why the example reads the inline *is not greater
        than 50* instead of the wrapped *not (… is greater than 50)*.

        >>> verbalize_expression(Not(variable(Robot, []).battery > 50))
        'the battery of a Robot is not greater than 50'
        """
        return isinstance(node._child_, Comparator)

    def build(self, node: Not, context: RuleContext) -> Fragment:
        """Say the inner comparator with the negation folded into the operator.

        Pushing the negation into the predicate is what yields the inline *is not greater than 50*
        operator span, leaving no outer *not (…)* wrapper.

        >>> verbalize_expression(Not(variable(Robot, []).battery > 50))
        'the battery of a Robot is not greater than 50'
        """
        return ConditionAssembler(context).predicate(node._child_, negated=True)


class NotBooleanAttributeRule(PhraseRule):
    """Negated boolean attribute chain *"<nav> is not <attribute>"* (Not over a bool-attribute chain).

    >>> verbalize_expression(Not(variable(Task, []).completed))
    'a Task is not completed'
    """

    construct = Not
    name = "not-bool-attribute"

    def when(self, node: Not, context: RuleContext) -> bool:
        """Fires when the negation wraps a boolean-attribute chain.

        Detecting a negated boolean-attribute chain is the gate that selects this rule over the
        generic :class:`NotRule`, which is why the example reads the predicative *is not completed*
        instead of the wrapped *not (a Task is completed)*.

        >>> verbalize_expression(Not(variable(Task, []).completed))
        'a Task is not completed'
        """
        return is_boolean_attribute_chain(node._child_)

    def build(self, node: Not, context: RuleContext) -> Fragment:
        """Say the negated predicative *"<nav> is not <attribute>"*.

        It owns the *is not completed* span, emitting the boolean attribute as a negated predicative
        rather than appending a *not (…)* wrapper.

        >>> verbalize_expression(Not(variable(Task, []).completed))
        'a Task is not completed'
        """
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
        """Say *"for all <plural var>, <condition>"*.

        It owns the *for all Robots,* prefix — rendering the variable as plural and joining it to the
        condition — while the condition clause comes from the recursion.

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(for_all(robot, robot.battery > 0))
        'for all Robots, the battery of the Robot is greater than 0'
        """
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
        """Say *"there exists <variable> such that <condition>"*.

        It owns the *there exists a Robot such that* framing around the condition; the condition
        clause itself comes from the recursion.

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(exists(robot, robot.battery > 0))
        'there exists a Robot such that the battery of the Robot is greater than 0'
        """
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
    """Transparent wrapper (Where / Having) → delegate to the condition.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(an(entity(robot).where(robot.battery > 50)))
    'Find a Robot whose battery is greater than 50'
    """

    construct = Filter
    name = "filter"

    def build(self, node: Filter, context: RuleContext) -> Fragment:
        """Delegate transparently to the wrapped condition.

        It adds no surface text of its own — unwrapping the Where/Having node so the *whose battery
        is greater than 50* of the example comes entirely from the condition it forwards to.

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(an(entity(robot).where(robot.battery > 50)))
        'Find a Robot whose battery is greater than 50'
        """
        return context.child(node.condition)
