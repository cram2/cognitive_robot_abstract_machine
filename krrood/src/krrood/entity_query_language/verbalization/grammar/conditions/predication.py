"""
Predicate realization — how a comparator/condition is said as a *predicate*.

This is the predicate counterpart of the surface-form toolkit in
:class:`~…conditions.assembler.ConditionAssembler`: the assembler's ``predicate`` dispatches over the
:class:`PredicateTransform` registry here (mirroring how the query assembler dispatches over
``query/ranking.py``), so the generic *"<left> <op> <right>"* is one transform alongside the absence
(*"has no …"*) and boolean-polarity (*"is [not] <attr>"*) forms — adding a predicate form is a new
subclass, nothing else changes (open/closed). The operator-word selection
(:func:`comparator_operator`, :func:`coindexed_operator`) and the absence rendering
(:func:`render_absence`) live here too, since they are the lower-level pieces a predicate is built
from.
"""

from __future__ import annotations

import operator
from abc import abstractmethod
from dataclasses import replace
from itertools import islice

from typing_extensions import TYPE_CHECKING, List, Optional, Set

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.expression_structure import is_temporal
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import Literal, Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.query.aggregation_structure import (
    is_calculation_value,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
    PhraseFragment,
    RoleFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.grammar.chain.assembler import (
    ChainAssembler,
)
from krrood.entity_query_language.verbalization.grammar.chain.planner import (
    ChainPlanner,
)
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    is_boolean_attribute_chain,
    is_none_literal,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    RuleContext,
)
from krrood.entity_query_language.verbalization.grammar.framework.specificity import (
    SpecificityRule,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    CoindexedFold,
    coindexed_natural_parts,
    coindexed_signature,
)
from krrood.entity_query_language.verbalization.relational_attributes import (
    relational_verb,
    relational_verb_phrase,
)
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    clause,
    Copula,
    Noun,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Absence,
    Copulas,
    Logicals,
    NonExistence,
    Operators,
    PassiveAbsence,
    predicative_core,
    predicative_operator,
    Quantifiers,
)
from krrood.entity_query_language.verbalization.vocabulary.words import OperatorWord

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import MicroplanningServices


# %% predicate-transform registry (the surface a comparator takes as a predicate)


class PredicateTransform(SpecificityRule):
    """
    One way to say a comparator as a standalone predicate — the registry the
    :meth:`~…conditions.assembler.ConditionAssembler.predicate` form dispatches over.

    :class:`GenericComparator` is the unguarded base every special case refines, so
    :meth:`~SpecificityRule.most_applicable` always returns a transform (the more-specific one when it
    applies, else the generic *"<left> <op> <right>"*). Adding a predicate-level transform is a new
    subclass; the ``predicate`` method does not change (open/closed) — mirroring the
    :class:`~…conditions.placement.ConditionForm` registry one level up.

    >>> verbalize_expression(variable(Robot, []).battery > 50)
    'the battery of a Robot is greater than 50'
    """

    @classmethod
    @abstractmethod
    def applies(cls, comparator: Comparator, negated: bool) -> bool:
        """
        :param comparator: The comparator being said as a predicate.
        :param negated: Whether an outer negation applies.
        :return: ``True`` when this transform renders *comparator*.

        It is the gate each concrete transform overrides to claim a comparator; for the plain
        ``battery > 50`` no special case fires, so the unguarded :class:`GenericComparator` is
        selected and the class example renders as the value comparison *is greater than 50*.
        """

    @classmethod
    @abstractmethod
    def render(
        cls, comparator: Comparator, context: RuleContext, negated: bool
    ) -> VerbalizationFragment:
        """
        :param comparator: The comparator to render.
        :param context: The per-node context (recursion and services).
        :param negated: Whether an outer negation applies.
        :return: *comparator* rendered in this transform's form.

        It emits the predicate text; here the selected :class:`GenericComparator` produces the whole
        *the battery of a Robot is greater than 50* sentence in value-comparison form.
        """


class GenericComparator(PredicateTransform):
    """The unguarded default: *"<left> <operator> <right>"* (the right side in value position).

    >>> verbalize_expression(variable(Robot, []).battery > 50)
    'the battery of a Robot is greater than 50'
    """

    @classmethod
    def applies(cls, comparator: Comparator, negated: bool) -> bool:
        """The unguarded base applies to every comparator.

        Returning ``True`` unconditionally, it is the catch-all gate that wins whenever no specific
        transform fires — which is why the class example takes the value-comparison form *is greater
        than 50* rather than an absence or boolean-polarity phrasing.
        """
        return True

    @classmethod
    def render(
        cls, comparator: Comparator, context: RuleContext, negated: bool
    ) -> VerbalizationFragment:
        """Render *"<left> <operator> <right>"* with the right side in value position.

        It owns the whole *the battery of a Robot is greater than 50* span, placing the left chain,
        the selected operator, and the right side in value position.
        """
        return PhraseFragment(
            parts=[
                context.child(comparator.left),
                comparator_operator(comparator, context.services, negated=negated),
                context.child(comparator.right, as_value=True),
            ]
        )


class AbsenceTransform(GenericComparator):
    """A non-negated ``<chain> == None`` comparison → the absence predicate (*"has no …"* / *"does
    not exist"*) instead of a value comparison.

    >>> verbalize_expression(variable(Mission, []).priority == None)
    'a Mission has no priority'
    """

    @classmethod
    def applies(cls, comparator: Comparator, negated: bool) -> bool:
        """Fires on a non-negated ``<chain> == None`` comparison.

        Recognising the ``== None`` shape is the gate that selects this transform over the generic
        comparator, which is why the class example reads *has no priority* instead of a literal *is
        equal to None* value comparison.
        """
        return (
            not negated
            and comparator.operation is operator.eq
            and is_none_literal(comparator.right)
        )

    @classmethod
    def render(
        cls, comparator: Comparator, context: RuleContext, negated: bool
    ) -> VerbalizationFragment:
        """Render the absence predicate instead of a value comparison.

        Delegating to :func:`render_absence` is what supplies the *a Mission has no priority* phrasing,
        flipping owner and attribute into the *has no* frame.
        """
        return render_absence(comparator, context)


def _lone_coindexed_fold(comparator: Comparator) -> Optional[CoindexedFold]:
    """:return: the single-terminal :class:`CoindexedFold` for a co-indexed comparison
    (``p.begin.month == p.end.month``), or ``None`` when *comparator* is not co-indexed — the
    one-terminal case of the conjunct-list fold, so a lone such comparison reuses the same form.
    """
    signature = coindexed_signature(comparator)
    if signature is None:
        return None
    (operation, _, _), leaf = signature
    return CoindexedFold(
        operation=operation,
        terminals=[leaf],
        left_prefix_expression=comparator.left._child_,
        right_prefix_expression=comparator.right._child_,
    )


class CoindexedEqualityTransform(GenericComparator):
    """A lone co-indexed equality over *sibling* prefixes (``p.begin.month == p.end.month``) → the
    natural *"the begin and end of <shared> have the same <leaf>"* form. It is the single-terminal
    case of the co-indexed fold, rendered by the very same rule (``context.child`` on a one-terminal
    :class:`CoindexedFold`) so the phrasing has one home. Guarded to the sibling case the natural
    form covers; a non-sibling lone equality keeps *"the <leaf> of a is the <leaf> of b"*.

    >>> verbalize_expression(
    ...     a(set_of(p.period.begin.month).such_that(p.period.begin.month == p.period.end.month))
    ... )  # doctest: +SKIP
    'the begin and end of the period of a _Statement have the same month'
    """

    @classmethod
    def applies(cls, comparator: Comparator, negated: bool) -> bool:
        """Fires on a non-negated co-indexed equality whose sibling prefixes the natural form covers.

        Recognising the single-terminal co-indexed shape is the gate that selects this transform over
        the generic comparator, which is why the example reads *the begin and end … have the same
        month* instead of the verbose *the month of the begin … is the month of the end …*.
        """
        if negated:
            return False
        fold = _lone_coindexed_fold(comparator)
        return fold is not None and coindexed_natural_parts(fold) is not None

    @classmethod
    def render(
        cls, comparator: Comparator, context: RuleContext, negated: bool
    ) -> VerbalizationFragment:
        """Render the lone equality through the one-terminal co-indexed fold so the *have the same*
        phrasing has a single home, rather than re-emitting it here."""
        return context.child(_lone_coindexed_fold(comparator))


class BooleanPolarityTransform(GenericComparator):
    """A boolean attribute compared to a boolean value → a predicative folding the value into the
    verb's polarity (*"is decaf"* / *"is not decaf"* / *"is either decaf or not"*), never *"is decaf
    is True"*.

    >>> verbalize_expression(variable(Task, []).completed == True)
    'a Task is completed'
    >>> verbalize_expression(variable(Task, []).completed == False)
    'a Task is not completed'
    """

    @classmethod
    def applies(cls, comparator: Comparator, negated: bool) -> bool:
        """Fires on a boolean attribute compared to a boolean value/domain.

        Recognising a boolean attribute against a boolean value is the gate that selects this
        transform over the generic comparator, which is why the class example folds to *is completed*
        instead of the literal *is completed is True*.
        """
        return (
            comparator.operation in (operator.eq, operator.ne)
            and is_boolean_attribute_chain(comparator.left)
            and _boolean_constraint(comparator.right) is not None
        )

    @classmethod
    def render(
        cls, comparator: Comparator, context: RuleContext, negated: bool
    ) -> VerbalizationFragment:
        """Fold the boolean value into the verb's polarity, never *"is completed is True"*.

        It owns the *is not completed* span, reading the ``False`` constraint to negate the
        predicative copula rather than appending the value as a separate term.
        """
        constraint = _boolean_constraint(comparator.right)
        plan = context.microplan.plan_for(comparator.left, ChainPlanner)
        chain = ChainAssembler(context)
        if constraint == {True, False}:
            return chain.boolean_alternative(plan)
        positive = True in constraint
        if comparator.operation is operator.ne:
            positive = not positive
        if negated:
            positive = not positive
        return chain.boolean_predicative(plan, negated=not positive)


class RelationalIdentityTransform(GenericComparator):
    """An equality identifying a variable with a *relational* hop (``m.assigned_to == r``) → the
    active relational predicate *"<subject> is <participle> <preposition> <owner>"* (*"it is assigned
    to a Mission"*), instead of the literal *"the Robot to which a Mission is assigned is the Robot"*.

    Said as a :func:`~…vocabulary.parts_of_speech.clause` so the subject pronominalises (*"it"*) and
    the copula agrees, reusing the relation's participle + preposition
    (:func:`~…relational_attributes.relational_verb`). The owner phrase is the recursion on the
    relation's prefix, so a deeper chain reads *"… of the team of a Mission"* unchanged.

    >>> robot, mission = variable(Robot, []), variable(Mission, [])
    >>> verbalize_expression(an(entity(robot).where(mission.assigned_to == robot)))
    'Find a Robot such that it is assigned to a Mission'
    """

    @classmethod
    def applies(cls, comparator: Comparator, negated: bool) -> bool:
        """Fires on a non-negated equality whose one side is a relational hop and the other a
        variable — the identity shape the active predicate collapses."""
        return (
            not negated
            and comparator.operation is operator.eq
            and cls._relational_identity(comparator) is not None
        )

    @classmethod
    def render(
        cls, comparator: Comparator, context: RuleContext, negated: bool
    ) -> VerbalizationFragment:
        """Render *"<subject> is <participle> <preposition> <owner>"*, the relation's verb supplying
        the participle and preposition and the recursion supplying subject and owner phrases.
        """
        relation_hop, subject = cls._relational_identity(comparator)
        verb = relational_verb(relation_hop._attribute_name_)
        return clause(
            Noun(context.child(subject)),
            Copula(),
            RoleFragment.for_attribute(
                relation_hop._owner_class_,
                relation_hop._attribute_name_,
                text=verb.phrase,
            ),
            Noun(context.child(relation_hop._child_)),
        )

    @staticmethod
    def _relational_identity(
        comparator: Comparator,
    ) -> Optional[tuple]:
        """:return: ``(relational_hop, subject_variable)`` when one operand is a relational attribute
        chain and the other a (non-literal) variable, in either order; else ``None``."""
        for hop, other in (
            (comparator.left, comparator.right),
            (comparator.right, comparator.left),
        ):
            if (
                isinstance(hop, Attribute)
                and relational_verb(hop._attribute_name_) is not None
                and isinstance(other, Variable)
                and not isinstance(other, Literal)
            ):
                return hop, other
        return None


# %% operator-word selection


def comparator_operator(
    comparator: Comparator,
    services: MicroplanningServices,
    *,
    negated: bool = False,
    compact: Optional[bool] = None,
    number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
    copula: bool = True,
) -> VerbalizationFragment:
    """
    Select the operator fragment for *comparator* (e.g. *"is greater than"*,
    *"is not equal to"*, *"is before"*).

    Handles these orthogonal concerns declaratively:

    * **Calculation equality** — ``==`` / ``!=`` against an aggregation reads
      *"is equal to"* / *"is not equal to"* rather than the bare *"is"*.
    * **Temporality** — datetime operands select the temporal phrase variant.
    * **Negation / compactness** — flags forwarded to the vocabulary.
    * **Number agreement** — the predicative (non-compact) surface factors its copula out as an
      agreeing leaf (:func:`~…vocabulary.english.predicative_operator`), so a plural subject reads
      *"are greater than"* without a duplicated plural phrase.

    :param comparator: The comparator expression.
    :param services: Shared verbalization state.
    :param negated: Outer negation (from a wrapping ``Not``).
    :param compact: Copula-less variant — the bare operator core without *"is"*. Defaults to the
        full copular surface (``False``) when ``None``; coordinated tails pass it explicitly.
    :param number: The grammatical number the predicative copula agrees with.
    :param copula: Keep the leading copula on the predicative (non-compact) surface. Pass ``False``
        for the bare core (*"greater than"*) when a shared copula is factored out across coordinated
        tails (*"is greater than 50 or less than 10"*); ignored when *compact*.
    :return: The operator fragment.

    Of the example sentence it supplies only the operator span *is greater than*; the surrounding
    chain and value come from the caller (:meth:`GenericComparator.render`).

    >>> verbalize_expression(variable(Robot, []).battery > 50)
    'the battery of a Robot is greater than 50'
    """
    if compact is None:
        compact = False
    operation = comparator.operation

    is_calculation = operation in (operator.eq, operator.ne) and (
        is_calculation_value(comparator.left) or is_calculation_value(comparator.right)
    )
    if is_calculation:
        calc_negated = (operation is operator.ne) ^ negated
        word = Operators.CALC_EQ.select(negated=calc_negated, compact=compact)
        return _operator_fragment(word, number, compact=compact, copula=copula)

    temporal = is_temporal(comparator.left) or is_temporal(comparator.right)
    operator_member = Operators.for_callable(operation)
    if operator_member is None:
        name = comparator._name_
        return RoleFragment.for_operator(
            f"{Logicals.NOT.text} {name}" if negated else name
        )
    word = operator_member.select(negated=negated, compact=compact, temporal=temporal)
    return _operator_fragment(word, number, compact=compact, copula=copula)


def _operator_fragment(
    word: OperatorWord, number: GrammaticalNumber, *, compact: bool, copula: bool
) -> VerbalizationFragment:
    """:return: the operator surface for a selected *word* — the bare compact verb when *compact*,
    the bare predicative core (*"greater than"*) when ``not copula``, else the agreeing predicative
    *"is greater than"*."""
    if compact:
        return word.as_fragment()
    if not copula:
        return RoleFragment.for_operator(predicative_core(word.text))
    return predicative_operator(word.text, number)


def _is_negatable_head(fragment: VerbalizationFragment) -> bool:
    """:return: whether *fragment* is a clause's negatable head — a ``VERB`` lemma (do-support) or an
    affirmative copula (*"is"* / *"are"*) the ``negated`` feature attaches to."""
    return isinstance(fragment, RoleFragment) and (
        fragment.role is SemanticRole.VERB
        or (
            fragment.role is SemanticRole.OPERATOR
            and fragment.text in (Copulas.IS.text, Copulas.ARE.text)
        )
    )


def negate_clause(fragment: VerbalizationFragment) -> Optional[VerbalizationFragment]:
    """
    :param fragment: A predicate clause built from the typed clause vocabulary.
    :return: *fragment* with its head verb or copula marked ``negated`` (so the morphology pass
        realises *"does not work"* / *"is not reachable"*), or ``None`` when the clause has no verb
        or copula head to negate — the caller then wraps it in *"not (…)"*.

    Negation is set as a feature, not rewritten as text: marking the typed head leaf lets the
    morphology pass realise do-support and copula suppletion uniformly with number agreement, so a
    wrapping ``Not`` reads *"an Employee does not work in a Department"* in place rather than
    *"not (an Employee works in a Department)"*.

    >>> from krrood.entity_query_language.verbalization.fragments.base import (
    ...     PhraseFragment, WordFragment,
    ... )
    >>> from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import Verb
    >>> clause = PhraseFragment(parts=[WordFragment(text="an Employee"), Verb("work").as_fragment()])
    >>> negate_clause(clause).parts[1].negated
    True
    >>> negate_clause(WordFragment(text="a Robot")) is None
    True
    """
    if _is_negatable_head(fragment):
        return replace(fragment, negated=True)
    if isinstance(fragment, PhraseFragment):
        for index, part in enumerate(fragment.parts):
            negated_part = negate_clause(part)
            if negated_part is not None:
                parts = list(fragment.parts)
                parts[index] = negated_part
                return replace(fragment, parts=parts)
    return None


def coindexed_operator(operation) -> VerbalizationFragment:
    """
    :param operation: A foldable comparison operator (``eq``/``gt``/``lt``/``ge``/``le``).
    :return: The plural copular operator fragment for the faithful co-indexed form — *"are equal
        to"* for ``eq`` (the calculation-equality reading, since two coordinated lists cannot read a
        bare *"are"*), *"are greater than"* for ``gt``, etc. The plural copula agrees because the
        coordinated terminals are the grammatical subject.

    The returned fragment carries plural number; the copula only agrees once the morphology pass
    runs, so realising it through :class:`MorphologyProcessor` yields the plural copula.

    >>> from krrood.entity_query_language.verbalization.fragments.base import (
    ...     flatten_fragment_to_plain_text,
    ... )
    >>> from krrood.entity_query_language.verbalization.rendering.morphology_processor import (
    ...     MorphologyProcessor,
    ... )
    >>> realise = lambda fragment: flatten_fragment_to_plain_text(
    ...     MorphologyProcessor().process(fragment)
    ... )
    >>> realise(coindexed_operator(operator.eq))
    'are equal to'
    >>> realise(coindexed_operator(operator.gt))
    'are greater than'
    """
    phrase = (
        Operators.CALC_EQ
        if operation is operator.eq
        else Operators.from_callable(operation)
    )
    return predicative_operator(phrase.value.standard, GrammaticalNumber.PLURAL)


# %% absence rendering (== None)


def render_absence(
    comparator: Comparator,
    context: RuleContext,
    number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
) -> VerbalizationFragment:
    """
    Render an ``<chain> == None`` comparison as an absence predicate rather than a value: an owned
    attribute reads *"<owner> has no <attribute>"* (the owner is the chain minus its terminal), and a
    bare variable reads *"<subject> does not exist"* (no attribute to name).

    Both flip the subject and object relative to the *"<attribute> of <owner> is <value>"* frame, so
    they are standalone predicates — never folded into a *"whose"* / *"respectively"* coordination
    (see :class:`~…conditions.placement.WhosePredicateForm`'s guard and the match assembler's None
    split). The single definition shared by the standalone predicate path (:class:`AbsenceTransform`)
    and the subject path (:class:`~…conditions.placement.AbsenceForm`).

    :param comparator: The ``<chain> == None`` comparator.
    :param context: The per-node context (recursion and services).
    :param number: The number the verb agrees with (plural owner → *"have no"* / *"do not exist"*).
    :return: The absence predicate fragment.

    >>> verbalize_expression(variable(Mission, []).priority == None)
    'a Mission has no priority'
    >>> verbalize_expression(variable(Robot, []) == None)
    'a Robot does not exist'
    """
    left = comparator.left
    if not isinstance(left, Attribute):
        return PhraseFragment(
            parts=[context.child(left), NonExistence.for_number(number).as_fragment()]
        )
    verb_phrase = relational_verb_phrase(left._attribute_name_)
    if verb_phrase is not None:
        return PhraseFragment(
            parts=[
                context.child(left._child_),
                PassiveAbsence.for_number(number).as_fragment(),
                RoleFragment.for_attribute(
                    left._owner_class_, left._attribute_name_, text=verb_phrase
                ),
                *_relation_target(left),
            ]
        )
    return PhraseFragment(
        parts=[
            context.child(left._child_),
            Absence.for_number(number).as_fragment(),
            RoleFragment.for_attribute(left._owner_class_, left._attribute_name_),
        ]
    )


def _relation_target(attribute: Attribute) -> List[VerbalizationFragment]:
    """:return: The object of a passive absence — *"any <Type>"* using the attribute's declared
    related type (*"any Robot"*), or the bare *"anything"* when that type is not a nameable class
    (a primitive, a typing generic, or unknown).

    >>> from krrood.entity_query_language.verbalization.fragments.base import (
    ...     flatten_fragment_to_plain_text,
    ... )
    >>> attribute = (variable(Mission, []).assigned_to == None).left
    >>> [flatten_fragment_to_plain_text(part) for part in _relation_target(attribute)]
    ['any', 'Robot']
    """
    related_type = attribute._type_
    if isinstance(related_type, type) and related_type.__module__ != "builtins":
        return [Quantifiers.ANY.as_fragment(), RoleFragment.for_type(related_type)]
    return [Quantifiers.ANYTHING.as_fragment()]


def _boolean_constraint(right: SymbolicExpression) -> Optional[Set[bool]]:
    """:return: The set of boolean values *right* constrains a boolean attribute to — ``{True}`` /
    ``{False}`` for a boolean literal or singleton domain, ``{True, False}`` for an open domain — or
    ``None`` when *right* is not a boolean literal / bounded boolean-domain variable.

    >>> _boolean_constraint((variable(Task, []).completed == True).right)
    {True}
    >>> _boolean_constraint((variable(Robot, []).battery > 50).right) is None
    True
    """
    if isinstance(right, Literal) and isinstance(right._value_, bool):
        return {right._value_}
    if isinstance(right, Variable) and right._type_ is bool:
        values = list(islice(right._re_enterable_domain_generator_, 3))
        if values and len(values) <= 2 and all(isinstance(v, bool) for v in values):
            return set(values)
    return None
