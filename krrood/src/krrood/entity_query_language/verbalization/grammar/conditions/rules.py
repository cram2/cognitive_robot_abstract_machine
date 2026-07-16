from __future__ import annotations

import operator

from krrood.entity_query_language.core.base_expressions import Filter
from krrood.entity_query_language.core.variable import InstantiatedVariable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    AND,
    OR,
    Not,
    flatten_operands,
)
from krrood.entity_query_language.operators.logical_quantifiers import Exists, ForAll
from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    VerbalizationFragment,
    oxford_comma,
    PhraseFragment,
    RoleFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    GrammaticalNumber,
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
    comparator_operator,
    negate_clause,
)
from krrood.entity_query_language.verbalization.grammar.instantiated.planner import (
    InstantiatedPlanner,
)
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    fold_shared_subject_comparisons,
    fold_shared_subject_conjunction,
    is_boolean_attribute_chain,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    RuleContext,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    between_phrase,
    build_between,
    coindexed_natural_parts,
    CoindexedFold,
    RangeFold,
    SharedSubjectComparisons,
    SharedSubjectConjunction,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    CoindexedPhrases,
    Conjunctions,
    copula_with,
    Keywords,
    Logicals,
    Prepositions,
    Punctuation,
)


class ComparatorRule(PhraseRule):
    """
    ``<left> <operator> <right>`` — e.g. *"is greater than 50"*.

    >>> verbalize_expression(variable(Robot, []).battery > 50)
    'the battery of a Robot is greater than 50'
    """

    construct = Comparator

    def build(self, node: Comparator, context: RuleContext) -> VerbalizationFragment:
        """
        Say the comparator as a standalone predicate.

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
    >>> x = variable(int, [])
    >>> verbalize_expression(and_(x > 1, x < 10, x != 5))
    'an Integer is between 1 and 10 and not 5'
    """

    construct = AND

    def build(self, node: AND, context: RuleContext) -> VerbalizationFragment:
        """Say the flattened conjuncts, comma-joined with a trailing *"and"*.

        It owns the *, and* coordination between the two conjuncts of the example; the conjunct
        clauses themselves come from the shared statement assembler. When every conjunct is a value
        comparison on one bare variable, it factors to the *"<subject> is …"* shared-subject main
        clause via the :class:`SharedSubjectConjunction` fold.

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(and_(robot.battery > 50, robot.name == 'x'))
        "the battery of a Robot is greater than 50, and the name of the Robot is 'x'"
        """
        operands = flatten_operands(node, AND)
        shared_subject = fold_shared_subject_conjunction(operands)
        if shared_subject is not None:
            return context.child(shared_subject)
        parts = ConditionAssembler(context).as_statements(operands)
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

    def build(self, node: RangeFold, context: RuleContext) -> VerbalizationFragment:
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
            compact=False,
        )


class CoindexedFoldRule(PhraseRule):
    """
    A group of co-indexed comparators (``p.begin.X == p.end.X`` for several ``X``)
    reduced by :func:`fold_coindexed_groups` → one factored clause that says the shared
    structure once.

    Two surface forms, chosen here (recognition stays pure):

    * **natural** — a pure-equality fold over *sibling* prefixes (``p.begin`` / ``p.end``) reads
      *"the begin and end of its period have the same month and year"*;
    * **faithful** — every other foldable case (a non-equality operator, or prefixes that are not
      siblings) reads *"the month and year of the begin of its period are equal to those of the
      end of its period"*.
    """

    construct = CoindexedFold

    def build(self, node: CoindexedFold, context: RuleContext) -> VerbalizationFragment:
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


class SharedSubjectComparisonsRule(PhraseRule):
    """Factored disjunction *"the <subject> is <tail>, …, or <tail>"* — the
    :class:`SharedSubjectComparisons` artifact produced when every disjunct of an ``OR`` is a value
    comparison on one shared subject chain.

    Saying the subject and its copula once and coordinating only the operator-and-value tails is
    coordination reduction over the shared subject (the disjunctive analogue of the *"between"* range
    fold). The disjunction is inclusive, so the tails are coordinated with a plain *"or"* and are
    **not** fronted with *"either"* (which reads as exclusive-or). Making it a first-class
    verbalizable means a caller that has folded its disjuncts renders the result through the normal
    recursion (``context.child``) and never has to know a folding helper exists.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(or_(robot.battery > 50, robot.battery < 10))
    'the battery of a Robot is greater than 50 or less than 10'
    """

    construct = SharedSubjectComparisons

    def build(
        self, node: SharedSubjectComparisons, context: RuleContext
    ) -> VerbalizationFragment:
        """
        Say the factored disjunction — subject and copula once, tails coordinated under
        a plain *or*.

        It owns the *is … or …* framing: the shared subject and its copula are stated
        once and each comparator contributes only its copula-less operator-and-value
        tail (*greater than 50*), so the disjuncts read as one clause rather than
        repeating the subject per disjunct.
        """
        tails = [self._tail(comparator, context) for comparator in node.comparators]
        return PhraseFragment(
            parts=[
                context.child(node.subject_expression),
                copula_with("", GrammaticalNumber.SINGULAR),
                oxford_comma(tails, Conjunctions.OR.as_fragment()),
            ]
        )

    @staticmethod
    def _tail(comparator: Comparator, context: RuleContext) -> VerbalizationFragment:
        """:return: a comparator's copula-less operator-and-value tail (*"greater than 50"*) — the
        differing piece coordinated under the shared subject. A bare equality has an empty operator
        core (*"is 30"* → *"30"*), so only the value is kept."""
        operator = comparator_operator(
            comparator, context.services, compact=False, copula=False
        )
        value = context.child(comparator.right, as_value=True)
        if not flatten_fragment_to_plain_text(operator).strip():
            return value
        return PhraseFragment(parts=[operator, value])


class SharedSubjectConjunctionRule(PhraseRule):
    """Factored conjunction *"<subject> is <tail>, …, and <tail>"* — the
    :class:`SharedSubjectConjunction` artifact produced when every conjunct of an ``AND`` is a value
    comparison on one shared *bare variable*.

    The subject and the leading copula are said once and the predicate tails coordinate as one main
    clause; the conjunctive analogue of :class:`SharedSubjectComparisonsRule` (*"… is … or …"*),
    so a conjunction and a disjunction over the same subject read with the same *"<subject> is …"* lead.

    >>> x = variable(int, [])
    >>> verbalize_expression(and_(x > 1, x < 10, x != 5))
    'an Integer is between 1 and 10 and not 5'
    """

    construct = SharedSubjectConjunction

    def build(
        self, node: SharedSubjectConjunction, context: RuleContext
    ) -> VerbalizationFragment:
        """Say the factored conjunction — subject once, the lead copula carried by the first tail, the
        tails coordinated Oxford-style.

        >>> x = variable(int, [])
        >>> verbalize_expression(and_(x > 5, x != 5))
        'an Integer is greater than 5 and not 5'
        """
        tails = [
            self._tail(tail, context, lead=index == 0)
            for index, tail in enumerate(node.tails)
        ]
        return PhraseFragment(
            parts=[
                context.child(node.subject_expression),
                oxford_comma(tails, Conjunctions.AND.as_fragment()),
            ]
        )

    @classmethod
    def _tail(cls, tail, context: RuleContext, *, lead: bool) -> VerbalizationFragment:
        """:return: one relative-clause tail. A folded :class:`RangeFold` reads *"between low and
        high"* (with the lead copula when it leads); a :class:`Comparator` reads its
        operator-and-value tail. The lead tail carries the clause's copula (*"is greater than 1"*); a
        negation drops the shared copula but keeps its *"not"* (*"not 5"*, giving *"is between 1 and
        10 and not 5"*); a bare equality has an empty core, so it keeps the copula (*"is 5"*) to stay
        unambiguous; every other positive tail shares the lead copula and drops it
        (*"less than 10"*)."""
        if isinstance(tail, RangeFold):
            return between_phrase(
                context.child(tail.lower_expression),
                context.child(tail.upper_expression),
                compact=not lead,
            )
        value = context.child(tail.right, as_value=True)
        core = comparator_operator(tail, context.services, compact=False, copula=False)
        if not lead and tail.operation is operator.ne:
            return cls._negated_tail(core, value)
        keeps_copula = lead or not flatten_fragment_to_plain_text(core).strip()
        if keeps_copula:
            copular = comparator_operator(
                tail, context.services, compact=False, copula=True
            )
            return PhraseFragment(parts=[copular, value])
        return PhraseFragment(parts=[core, value])

    @staticmethod
    def _negated_tail(
        core: VerbalizationFragment, value: VerbalizationFragment
    ) -> VerbalizationFragment:
        """:return: a non-lead ``!=`` tail as *"not <core> <value>"* — the negation word is kept while
        the shared lead copula is dropped, so *"is not 5"* reduces to *"not 5"* and the temporal
        *"is not at <date>"* to *"not at <date>"*. The *core* is the copula-less operator remainder,
        empty for the plain inequality."""
        core_is_empty = not flatten_fragment_to_plain_text(core).strip()
        parts = [Logicals.NOT.as_fragment()]
        if not core_is_empty:
            parts.append(core)
        parts.append(value)
        return PhraseFragment(parts=parts)


class OrRule(PhraseRule):
    """Inclusive disjunction *"a, b, or c"*; flattens nested ORs. When every disjunct is a value
    comparison on one shared subject chain it factors to *"the <subject> is … or …"* via the
    :class:`SharedSubjectComparisons` fold. The disjunction is inclusive, so it is not fronted with
    *"either"* (which reads as exclusive-or).

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(or_(robot.battery > 50, robot.name == 'x'))
    "the battery of a Robot is greater than 50, or the name of the Robot is 'x'"
    >>> verbalize_expression(or_(robot.battery > 50, robot.battery < 10))
    'the battery of a Robot is greater than 50 or less than 10'
    """

    construct = OR

    def build(self, node: OR, context: RuleContext) -> VerbalizationFragment:
        """
        Say the flattened disjuncts as *"a, b, or c"*, or the factored *"… is … or …"*
        when they share a subject.

        It owns the *…, or …* framing around the disjuncts of the class example — the
        comma-*or* before the last — while the disjunct clauses come from the recursion;
        a shared-subject disjunction is delegated to the
        :class:`SharedSubjectComparisons` fold instead.
        """
        operands = flatten_operands(node, OR)
        shared_subject = fold_shared_subject_comparisons(operands)
        if shared_subject is not None:
            return context.child(shared_subject)
        parts = [context.child(conjunct) for conjunct in operands]
        if len(parts) == 1:
            return parts[0]
        # "a, b, or c": the head items are comma-joined, then a trailing comma that the
        # orthography pass hugs to the last head item — no separator="" bookkeeping here.
        head = PhraseFragment(parts=parts[:-1], separator=Separator.COMMA)
        return PhraseFragment(
            parts=[
                head,
                Punctuation.COMMA.as_fragment(),
                Conjunctions.OR.as_fragment(),
                parts[-1],
            ]
        )


class NotRule(PhraseRule):
    """Generic negation *"not (<child>)"* (specialised by the guarded Not rules below).

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(Not(or_(robot.battery > 50, robot.name == 'x')))
    "not (the battery of a Robot is greater than 50, or the name of the Robot is 'x')"
    """

    construct = Not

    def build(self, node: Not, context: RuleContext) -> VerbalizationFragment:
        """
        Wrap the child in *"not (<child>)"* via the orthography pass.

        It owns the *not (…)* wrapper of the class example — the leading *not* and the
        parentheses — around the disjunction child the recursion renders.
        """
        return _negation_wrap(context.child(node._child_))


def _negation_wrap(child_fragment: VerbalizationFragment) -> VerbalizationFragment:
    """:return: *child_fragment* wrapped as *"not (<child>)"* — the fallback negation for a clause
    that cannot be negated in place. The parens glue to the child via the orthography pass.
    """
    return PhraseFragment(
        parts=[
            Logicals.NOT.as_fragment(),
            Punctuation.OPEN_PAREN.as_fragment(),
            child_fragment,
            Punctuation.CLOSE_PAREN.as_fragment(),
        ]
    )


class NotVerbalizablePredicateRule(PhraseRule):
    """Inline-negated verbalizable predicate (Not over a predicate whose type builds its own
    verbalization fragment).

    Because the predicate states its clause with the typed clause vocabulary, the negation is set as
    a feature on the clause's head verb or copula — realised by the morphology pass as do-support
    (*"does not work"*) or copula suppletion (*"is not reachable"*) — rather than wrapping the whole
    clause in *"not (...)"*. A clause with no verb or copula head has nothing to negate, so it falls
    back to the wrapped form.

    >>> verbalize_expression(Not(IsReachable(variable(Location, []))))
    'a Location is not reachable'
    >>> employee, department = variable(StaffMember, []), variable(Department, [])
    >>> verbalize_expression(Not(WorksIn(employee, department)))
    'a StaffMember does not work in a Department'
    """

    construct = Not

    def when(self, node: Not, context: RuleContext) -> bool:
        """
        Fires when the negation wraps a predicate that builds its own verbalization
        fragment.

        Detecting a verbalizable-predicate child is the gate that selects this rule over
        the generic :class:`NotRule`, so the class example negates the predicate's head
        verb / copula in place instead of wrapping it in *not (…)*.
        """
        return isinstance(
            node._child_, InstantiatedVariable
        ) and InstantiatedPlanner.has_fragment(node._child_)

    def build(self, node: Not, context: RuleContext) -> VerbalizationFragment:
        """
        Say the predicate with its head verb / copula negated, or wrap it when it has
        neither.

        Marking the clause's verb or copula negated is what yields the inline *does not
        work in* / *is not reachable*; a clause with no such head has nothing to negate,
        so it falls back to the *not (…)* wrapper.
        """
        clause = context.child(node._child_)
        negated = negate_clause(clause)
        return negated if negated is not None else _negation_wrap(clause)


class NotComparatorRule(PhraseRule):
    """
    Inline negated comparator *"a is not greater than b"* (Not over a Comparator).

    >>> verbalize_expression(Not(variable(Robot, []).battery > 50))
    'the battery of a Robot is not greater than 50'
    """

    construct = Not

    def when(self, node: Not, context: RuleContext) -> bool:
        """
        Fires when the negation wraps a comparator.

        Detecting that the negation wraps a :class:`Comparator` is the gate that selects this rule
        over the generic :class:`NotRule`, which is why the example reads the inline *is not greater
        than 50* instead of the wrapped *not (… is greater than 50)*.

        >>> verbalize_expression(Not(variable(Robot, []).battery > 50))
        'the battery of a Robot is not greater than 50'
        """
        return isinstance(node._child_, Comparator)

    def build(self, node: Not, context: RuleContext) -> VerbalizationFragment:
        """
        Say the inner comparator with the negation folded into the operator.

        Pushing the negation into the predicate is what yields the inline *is not greater than 50*
        operator span, leaving no outer *not (…)* wrapper.

        >>> verbalize_expression(Not(variable(Robot, []).battery > 50))
        'the battery of a Robot is not greater than 50'
        """
        return ConditionAssembler(context).predicate(node._child_, negated=True)


class NotBooleanAttributeRule(PhraseRule):
    """
    Negated boolean attribute chain *"<nav> is not <attribute>"* (Not over a bool-
    attribute chain).

    >>> verbalize_expression(Not(variable(Task, []).completed))
    'a Task is not completed'
    """

    construct = Not

    def when(self, node: Not, context: RuleContext) -> bool:
        """
        Fires when the negation wraps a boolean-attribute chain.

        Detecting a negated boolean-attribute chain is the gate that selects this rule over the
        generic :class:`NotRule`, which is why the example reads the predicative *is not completed*
        instead of the wrapped *not (a Task is completed)*.

        >>> verbalize_expression(Not(variable(Task, []).completed))
        'a Task is not completed'
        """
        return is_boolean_attribute_chain(node._child_)

    def build(self, node: Not, context: RuleContext) -> VerbalizationFragment:
        """
        Say the negated predicative *"<nav> is not <attribute>"*.

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
    'for all Robots, their batteries are greater than 0'
    """

    construct = ForAll

    def build(self, node: ForAll, context: RuleContext) -> VerbalizationFragment:
        """Say *"for all <plural var>, <condition>"*.

        It owns the *for all Robots,* prefix — rendering the variable as plural and joining it to the
        condition — while the condition clause comes from the recursion. Rendering the variable
        *plural* is what makes the body distribute over the population: the quantifier opens a
        discourse scope on the variable, so the condition's chain pronominalises to the agreeing
        *their batteries are* rather than repeating *the battery of the Robot is*.

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(for_all(robot, robot.battery > 0))
        'for all Robots, their batteries are greater than 0'
        """
        variable_fragment = context.child(
            node.variable, number=GrammaticalNumber.PLURAL
        )
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
    'there exists a Robot such that its battery is greater than 0'
    """

    construct = Exists

    def build(self, node: Exists, context: RuleContext) -> VerbalizationFragment:
        """Say *"there exists <variable> such that <condition>"*.

        It owns the *there exists a Robot such that* framing around the condition; the condition
        clause itself comes from the recursion. The quantifier opens a discourse scope on the
        (singular) variable, so the condition pronominalises to *its battery* rather than repeating
        *the battery of the Robot*.

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(exists(robot, robot.battery > 0))
        'there exists a Robot such that its battery is greater than 0'
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

    def build(self, node: Filter, context: RuleContext) -> VerbalizationFragment:
        """Delegate transparently to the wrapped condition.

        It adds no surface text of its own — unwrapping the Where/Having node so the *whose battery
        is greater than 50* of the example comes entirely from the condition it forwards to.

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(an(entity(robot).where(robot.battery > 50)))
        'Find a Robot whose battery is greater than 50'
        """
        return context.child(node.condition)
