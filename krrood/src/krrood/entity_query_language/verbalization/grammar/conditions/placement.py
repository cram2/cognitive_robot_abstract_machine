from __future__ import annotations

import inspect
import operator
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

from typing_extensions import ClassVar, List, Optional, Union

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.verbalization.exceptions import (
    UndeclaredFormPositionError,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    PhraseFragment,
    VerbalizationFragment,
    oxford_comma,
)
from krrood.entity_query_language.verbalization.fragments.features import Definiteness
from krrood.entity_query_language.verbalization.grammar.conditions.assembler import (
    ConditionAssembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    is_boolean_attribute_chain,
    is_none_literal,
    references,
    single_hop_attribute,
    superlative_aggregation,
)
from krrood.entity_query_language.verbalization.grammar.conditions.predication import (
    render_absence,
)
from krrood.entity_query_language.verbalization.grammar.conditions.scoping import (
    bind_relational_entities,
    RelationalBindingFold,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    RuleContext,
)
from krrood.entity_query_language.verbalization.grammar.framework.specificity import (
    SpecificityRule,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    reduce_conjuncts,
    RangeFold,
)
from krrood.entity_query_language.verbalization.microplanning.possessive import (
    subject_relative_relation,
)
from krrood.entity_query_language.verbalization.microplanning.referring import (
    referring_noun_with_restrictions,
)
from krrood.entity_query_language.verbalization.relational_attributes import (
    relational_verb,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    Keywords,
)
from krrood.entity_query_language.verbalization.vocabulary.words import (
    GrammaticalNumber,
)


class SurfacePosition(Enum):
    """Where a condition's rendered form attaches in a sentence — the surface-position taxonomy a
    caller maps to an actual position (a noun modifier vs. a standalone clause)."""

    SELECTION_MODIFIER = auto()
    """A post-nominal prepositional phrase on the selection — *"<noun> with the maximum amount"*,
    *"<noun> with priority greater than 2"*."""
    RELATIVE_CLAUSE = auto()
    """A post-nominal subject-relative clause on the selection — *"<noun> that is assigned to a
    Mission …"* (a relational binding to a nested entity)."""
    WHOSE = auto()
    """A bare predicate gathered under a shared *"whose …, and …"* envelope on the subject noun."""
    STANDALONE = auto()
    """Not attachable to the subject noun — said as its own clause (*"such that …"* / *"where …"*)."""


@dataclass(frozen=True)
class Placement:
    """The request a :class:`ConditionForm` reads: one condition to say relative to a subject."""

    item: Union[SymbolicExpression, RangeFold, RelationalBindingFold]
    """The grouped ``WHERE`` conjunct — a raw expression, a range fold, or a relational binding."""

    subject: Variable
    """The variable the condition may attach to."""

    number: GrammaticalNumber = GrammaticalNumber.SINGULAR
    """The number the subject (and so the predicate) agrees with — singular for a query subject,
    plural for an aggregated inference antecedent (*"whose children are …"*)."""

    compact: bool = False
    """``True`` when the condition restricts an *introduced* entity noun (a nested binding), which
    admits the compact *"with <attribute> <comparison>"* form; ``False`` for the query subject, which
    keeps the full *"whose … is …"* clause."""


@dataclass(frozen=True)
class Placed:
    """A rendered condition together with the position it occupies, for the caller to position."""

    position: SurfacePosition
    """The surface position the fragment occupies."""

    fragment: VerbalizationFragment
    """The rendered condition."""


class ConditionForm(SpecificityRule):
    """
    One surface form a condition can take next to a subject noun — recognise it, render it, and
    declare its :class:`SurfacePosition`. The single, *total* registry of these forms is the one authority on
    *how a condition is said relative to a subject*: every position-aware caller (a query's subject
    restriction) asks it via :func:`place` and never picks a form itself.

    The registry is total because :class:`StandaloneForm` is the unguarded base every specific form
    refines, so :meth:`~SpecificityRule.most_applicable` always returns a form (the more-derived
    specific form when it applies, else the standalone fallback) — no ``None``, no residual special
    case. Adding a form is a new subclass; nothing else changes (open/closed).

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(an(entity(robot).where(robot.battery > 50)))
    'Find a Robot whose battery is greater than 50'

    Reference: :cite:t:`dale1995gricean` — referring expressions / post-nominal modification;
    production-rule selection (the most-specific guarded alternative wins).
    """

    position: ClassVar[SurfacePosition]
    """The surface position this form's output occupies."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Fail fast at class-definition time when a concrete form forgets to declare its
        :attr:`position` (otherwise the omission is a silent ``AttributeError`` deep in :func:`place`).
        """
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls) and not hasattr(cls, "position"):
            raise UndeclaredFormPositionError(form=cls)

    @classmethod
    @abstractmethod
    def applies(cls, request: Placement) -> bool:
        """
        :param request: The condition and the subject it may attach to.
        :return: ``True`` when this form renders *request*.

        It is the gate each concrete form overrides to claim a situation; here the winning override
        (:class:`WhosePredicateForm`) is what routes ``robot.battery > 50`` into the *whose* position, so
        the class example renders *whose battery is greater than 50* rather than a standalone clause.
        """

    @classmethod
    @abstractmethod
    def render(cls, request: Placement, context: RuleContext) -> VerbalizationFragment:
        """
        :param request: The condition and its subject.
        :param context: The per-node context (recursion and services).
        :return: *request* rendered in this form.

        It emits the condition's surface text; here the selected form's override produces *battery is
        greater than 50*, the span the *whose* envelope then wraps to give the class example.
        """


class StandaloneForm(ConditionForm):
    """The unguarded fallback: a condition that does not fold onto the subject noun is said as its
    own clause, via the normal recursion. Being the base every specific form refines, it makes the
    registry total — it applies to anything, and a more-specific form outranks it when one fits.

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(an(entity(employee).where(employee.salary > employee.starting_salary)))
    'Find an Employee such that its salary is greater than its starting_salary'
    """

    position = SurfacePosition.STANDALONE

    @classmethod
    def applies(cls, request: Placement) -> bool:
        """A self-referential comparator folds onto no subject noun, so it falls here.

        Returning ``True`` unconditionally, it is the catch-all gate that claims the comparator no
        specific form accepts — which is why the class example lands in a *such that …* clause instead
        of the *whose* position a one-sided attribute comparison would take.
        """
        return True

    @classmethod
    def render(cls, request: Placement, context: RuleContext) -> VerbalizationFragment:
        """Render the residual condition as its own clause via the normal recursion.

        Deferring to the recursion is what supplies the whole *its salary is greater than its
        starting_salary* clause; this form contributes only the decision to render it standalone (its
        :attr:`position` is :attr:`SurfacePosition.STANDALONE`), so the caller frames it with *such that*.
        """
        return context.child(request.item)


class SuperlativeForm(StandaloneForm):
    """``subject.<chain> == max/min(over all <same-type>.<same chain>)`` → the superlative selection
    modifier *"with the maximum/minimum <leaf>"*.

    >>> t1 = variable(BankTransaction, domain=None)
    >>> t2 = variable(BankTransaction, domain=None)
    >>> verbalize_expression(
    ...     the(entity(t1).where(
    ...         t1.amount_details.amount == an(entity(max(t2.amount_details.amount)))
    ...     ))
    ... )
    'Find the unique BankTransaction with the maximum amount'
    """

    position = SurfacePosition.SELECTION_MODIFIER

    @classmethod
    def applies(cls, request: Placement) -> bool:
        """Fires on ``subject.<chain> == max/min(over the same-type population)``.

        Recognising the equality-against-an-aggregate shape is the gate that selects this form over
        the plain :class:`WhosePredicateForm`, which is why the class example collapses to the
        selection modifier *with the maximum amount* instead of *whose amount is equal to …*.
        """
        return superlative_aggregation(request.item, request.subject) is not None

    @classmethod
    def render(cls, request: Placement, context: RuleContext) -> VerbalizationFragment:
        """Render the post-nominal superlative modifier *"with the maximum/minimum <leaf>"*.

        It owns the entire *with the maximum amount* span of the class example, reading the aggregator
        to choose *minimum* or *maximum* and naming the leaf attribute.
        """
        return ConditionAssembler(context).superlative_modifier(
            request.item, request.subject
        )


class WhoseRangeForm(StandaloneForm):
    """A range fold on a single-hop subject attribute → *"<attribute> is between low and high"*.

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(
    ...     an(entity(employee).where(and_(employee.salary > 100, employee.salary < 200)))
    ... )
    'Find an Employee whose salary is between 100 and 200'
    """

    position = SurfacePosition.WHOSE

    @classmethod
    def applies(cls, request: Placement) -> bool:
        """Fires when a lower/upper bound pair on a single-hop subject attribute has folded.

        Detecting the folded :class:`RangeFold` is the gate that selects this form over two separate
        :class:`WhosePredicateForm` comparisons, which is why the bound pair reads as the single
        *whose salary is between 100 and 200* rather than *whose salary is greater than 100, and …*.
        """
        return (
            isinstance(request.item, RangeFold)
            and single_hop_attribute(request.item.chain_expression, request.subject)
            is not None
        )

    @classmethod
    def render(cls, request: Placement, context: RuleContext) -> VerbalizationFragment:
        """Render the *"<attribute> is between low and high"* whose-modifier.

        It owns the *salary is between 100 and 200* span — naming the attribute and emitting the
        *between … and …* frame over the two bounds — which the *whose* envelope then wraps.
        """
        return ConditionAssembler(context).range_modifier(
            request.item, request.subject, request.number
        )


class WhosePredicateForm(StandaloneForm):
    """A single-hop, non-boolean subject-attribute comparator whose right side does not reference the
    subject → *"<attribute> is greater than 100"* / *"<attribute> is equal to <calc>"*.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(an(entity(robot).where(robot.battery > 50)))
    'Find a Robot whose battery is greater than 50'
    """

    position = SurfacePosition.WHOSE

    @classmethod
    def applies(cls, request: Placement) -> bool:
        """Fires on a single-hop, non-boolean subject attribute compared to a subject-free value.

        Passing this guard is what selects the *whose* form for ``robot.battery > 50``, so the
        class example reads *whose battery is greater than 50* rather than the standalone *such that …*
        clause the fallback would produce; the rejected branches route absence, boolean, and
        superlative comparisons to their own forms instead.
        """
        item, subject = request.item, request.subject
        if not isinstance(item, Comparator):
            return False
        attribute = single_hop_attribute(item.left, subject)
        if attribute is None or is_boolean_attribute_chain(item.left):
            return False  # a boolean attribute uses the predicative form, not "whose"
        if is_none_literal(item.right):
            return False  # an absence comparison is AbsenceForm's (standalone, not "whose")
        if superlative_aggregation(item, subject) is not None:
            return False  # a superlative comparator is SuperlativeForm's
        return not references(item.right, subject)

    @classmethod
    def render(cls, request: Placement, context: RuleContext) -> VerbalizationFragment:
        """Render the bare *"<attribute> <operator> <value>"* the *"whose"* envelope wraps.

        It owns the *battery is greater than 50* span of the class example — the attribute noun,
        operator, and value with the subject dropped — leaving only the *whose* header for the
        envelope to add.
        """
        return ConditionAssembler(context).attribute_modifier(
            request.item, request.subject, request.number
        )


class AbsenceForm(StandaloneForm):
    """A non-negated ``<chain> == None`` comparison → the absence predicate *"<owner> has no
    <attribute>"* (an owned attribute) or *"<subject> does not exist"* (a bare variable), said as
    its own clause rather than folded onto the subject noun (the subject/object flip cannot sit in a
    *"whose …"* coordination).

    >>> mission = variable(Mission, [])
    >>> verbalize_expression(an(entity(mission).where(mission.priority == None)))
    'Find a Mission such that the Mission has no priority'
    """

    position = SurfacePosition.STANDALONE

    @classmethod
    def applies(cls, request: Placement) -> bool:
        """Fires on a non-negated ``<chain> == None`` comparison.

        Recognising the ``== None`` shape is the gate that selects this form over
        :class:`WhosePredicateForm`, which is why the class example becomes the standalone *such that
        the Mission has no priority* rather than being folded into a *whose* group.
        """
        item = request.item
        return (
            isinstance(item, Comparator)
            and item.operation is operator.eq
            and is_none_literal(item.right)
        )

    @classmethod
    def render(cls, request: Placement, context: RuleContext) -> VerbalizationFragment:
        """Render the standalone absence predicate *"<owner> has no <attribute>"*.

        It owns the *the Mission has no priority* span, flipping owner and attribute into the *has
        no* frame; tagged :attr:`SurfacePosition.STANDALONE`, it is why the caller frames the result with *such
        that* rather than *whose*.
        """
        return render_absence(request.item, context, number=request.number)


_ORDER_OPERATORS = (operator.lt, operator.le, operator.gt, operator.ge)
"""The order comparisons that read cleanly as a compact *"with <attribute> <comparison>"* modifier;
equality/inequality keep the fuller *"whose … is …"* clause."""


class WithAttributeForm(WhosePredicateForm):
    """A single-hop order comparison on an *introduced* entity → the compact post-nominal *"with
    <attribute> <comparison>"* (*"with battery greater than 50"*). Refines :class:`WhosePredicateForm`
    so it wins only when the placement is compact and the operator is an order comparison; otherwise
    the base *"whose"* form applies (top-level subjects, equality, non-order shapes).

    >>> mission, robot = variable(Mission, []), variable(Robot, [])
    >>> verbalize_expression(an(entity(mission).where(mission.assigned_to == robot, robot.battery > 50)))
    'Find a Mission that is assigned to a Robot with battery greater than 50'
    """

    position = SurfacePosition.SELECTION_MODIFIER

    @classmethod
    def applies(cls, request: Placement) -> bool:
        """Fires on a compact placement of a single-hop order comparison — the refinement over
        :class:`WhosePredicateForm` that routes it to the compact *"with …"* modifier.
        """
        return (
            request.compact
            and super().applies(request)
            and isinstance(request.item, Comparator)
            and request.item.operation in _ORDER_OPERATORS
        )

    @classmethod
    def render(cls, request: Placement, context: RuleContext) -> VerbalizationFragment:
        """Render the compact *"with <attribute> <comparison>"* modifier."""
        return ConditionAssembler(context).with_attribute_modifier(
            request.item, request.subject
        )


class RelationalBindingForm(StandaloneForm):
    """A subject→entity relational binding grouped with the entity's restrictions → the post-nominal
    subject-relative clause *"that is <participle> <preposition> <entity>"*, the entity carrying its
    restrictions nested onto it (*"that is assigned to a Robot with battery greater than 50"*).

    The entity's conjuncts are rendered by recursing through :func:`as_subject_restrictions` in
    *compact* mode, so nesting generalises to arbitrary depth.

    >>> mission, robot = variable(Mission, []), variable(Robot, [])
    >>> verbalize_expression(an(entity(mission).where(mission.assigned_to == robot)))
    'Find a Mission that is assigned to a Robot'
    """

    position = SurfacePosition.RELATIVE_CLAUSE

    @classmethod
    def applies(cls, request: Placement) -> bool:
        """Fires on a :class:`RelationalBindingFold` — the artifact grouping produces."""
        return isinstance(request.item, RelationalBindingFold)

    @classmethod
    def render(cls, request: Placement, context: RuleContext) -> VerbalizationFragment:
        """Render *"that is <participle> <preposition> <entity-with-restrictions>"*."""
        fold = request.item
        nested = as_subject_restrictions(
            fold.nested, fold.entity, context, compact=True
        )
        entity_noun = referring_noun_with_restrictions(
            fold.entity,
            fold.entity._type_.__name__,
            Definiteness.INDEFINITE,
            nested,
        )
        return subject_relative_relation(
            fold.relation_hop._owner_class_,
            fold.relation_hop._attribute_name_,
            relational_verb(fold.relation_hop._attribute_name_),
            entity_noun,
            request.number,
        )


def place(request: Placement, context: RuleContext) -> Placed:
    """
    Render a condition relative to its subject and report the position it occupies — the single entry a
    position-aware caller uses. The form is chosen by the registry (most-specific applicable; the
    standalone fallback guarantees a match), so the caller never inspects the condition's shape.

    :param request: The condition and the subject it may attach to.
    :param context: The per-node context (recursion and services).
    :return: The rendered condition tagged with its surface position.

    Pairing the rendered fragment with the winning form's :attr:`position` is what lets the caller route
    ``robot.battery > 50`` to the *whose* position, producing the *whose battery is greater than 50*
    placement in the example.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(an(entity(robot).where(robot.battery > 50)))
    'Find a Robot whose battery is greater than 50'
    """
    form = ConditionForm.most_applicable(request)
    return Placed(position=form.position, fragment=form.render(request, context))


# %% placing a whole subject restriction


@dataclass(frozen=True)
class RestrictionFragments:
    """The placed pieces of a subject's WHERE, for the caller to position — each goes to a different
    sentence position, so they are kept apart rather than pre-joined."""

    inline_modifiers: List[VerbalizationFragment] = field(default_factory=list)
    """Post-nominal prepositional phrases that attach inline, right after the selection noun — a
    superlative (*"with the maximum amount"*) or a compact attribute restriction (*"with battery
    greater than 50"*)."""

    relative_clauses: List[VerbalizationFragment] = field(default_factory=list)
    """Subject-relative clauses on the selection noun — one per relational binding (*"that is
    assigned to a Robot …"*). Empty when the subject binds no related entity."""

    whose: Optional[VerbalizationFragment] = None
    """The *"whose"* group as a coordinated block (header *"whose"*, one bare predicate per item,
    Oxford-joined) — a sub-list of points in hierarchical rendering, *"whose a, and b"* inline / in
    paragraph. ``None`` when nothing folds into a *"whose"*."""

    residual: Optional[VerbalizationFragment] = None
    """The residual condition for a separate *"such that …"* / *"where …"* clause; the caller picks
    the keyword and position. ``None`` when the WHERE folds entirely into the other pieces."""


def as_subject_restrictions(
    conditions: List[SymbolicExpression],
    subject: Variable,
    context: RuleContext,
    number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
    *,
    compact: bool = False,
) -> RestrictionFragments:
    """
    Say a subject's WHERE conjuncts as restrictions on its noun — the counterpart to
    :meth:`ConditionAssembler.as_statements` for when conditions attach to a subject rather than
    standing alone. Each conjunct is placed and the results bucketed by position into the pieces a
    caller positions: superlative / compact noun modifiers, relative-clause bindings, the shared
    *"whose"* group, and the standalone residual. This is the list form of :func:`place`.

    The conjuncts are reduced here first (a complementary lower/upper bound pair on one chain
    becomes a single *"… is between …"*; co-indexed comparisons across two prefixes fold into one
    *"… have the same …"*), then a relational binding from *subject* to a secondary entity is grouped
    with that entity's own conjuncts (:func:`~…scoping.bind_relational_entities`), so it renders as a
    nested relative clause rather than an orphaned residual.

    :param conditions: The subject's WHERE conjuncts (an ``AND`` already flattened to a list).
    :param subject: The variable the restriction is on.
    :param context: The per-node context (recursion and services).
    :param number: The number the subject agrees with — singular for a query selection, plural for
        an aggregated inference antecedent.
    :param compact: ``True`` when *subject* is an introduced entity noun (the recursive call for a
        binding's nested conjuncts), enabling the compact *"with <attribute> <comparison>"* form.
    :return: The placed restriction pieces.

    Bucketing each placed conjunct by position is what splits the two conditions in the example: the
    battery comparison fills the *whose* group while the ``== None`` conjunct becomes the standalone
    residual, so the sentence joins *whose battery is greater than 50* with *such that the Robot has
    no name*.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(an(entity(robot).where(and_(robot.battery > 50, robot.name == None))))
    'Find a Robot whose battery is greater than 50, such that the Robot has no name'
    """
    grouped = bind_relational_entities(reduce_conjuncts(list(conditions)), subject)
    placed = [
        place(
            Placement(item=item, subject=subject, number=number, compact=compact),
            context,
        )
        for item in grouped
    ]
    whose_clauses = [
        PhraseFragment(parts=[Keywords.WHOSE.as_fragment(), item.fragment])
        for item in placed
        if item.position is SurfacePosition.WHOSE
    ]
    whose = (
        BlockFragment(
            header=None,
            items=whose_clauses,
            conjunction=Conjunctions.AND.as_fragment(),
        )
        if whose_clauses
        else None
    )
    return RestrictionFragments(
        inline_modifiers=[
            item.fragment
            for item in placed
            if item.position is SurfacePosition.SELECTION_MODIFIER
        ],
        relative_clauses=[
            item.fragment
            for item in placed
            if item.position is SurfacePosition.RELATIVE_CLAUSE
        ],
        whose=whose,
        residual=_join_residual(
            [
                item.fragment
                for item in placed
                if item.position is SurfacePosition.STANDALONE
            ]
        ),
    )


def _join_residual(
    fragments: List[VerbalizationFragment],
) -> Optional[VerbalizationFragment]:
    """:return: The standalone conjuncts joined into one residual condition, or ``None``.

    It supplies the comma-and join that knits the two absence clauses of the example into the single
    residual *the Mission has no priority, and the Mission has not been assigned to any Robot* the
    caller frames with *such that*.

    >>> mission = variable(Mission, [])
    >>> verbalize_expression(
    ...     an(entity(mission).where(and_(mission.priority == None, mission.assigned_to == None)))
    ... )
    'Find a Mission such that the Mission has no priority, and the Mission has not been assigned to any Robot'
    """
    if not fragments:
        return None
    # Residual conjuncts are independent clauses, so a two-clause pair keeps its comma.
    return oxford_comma(fragments, Conjunctions.AND.as_fragment(), pair_comma=True)
