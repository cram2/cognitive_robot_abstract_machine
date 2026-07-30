from __future__ import annotations

from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.verbalization.fragments.base import (
    PhraseFragment,
    RoleFragment,
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import (
    Assembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.predication import (
    comparator_operator,
    PredicateTransform,
)
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    single_hop_attribute,
    superlative_aggregation,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    reduce_conjuncts,
    RangeFold,
    build_between,
)
from typing_extensions import List
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Aggregations,
    Articles,
    Copulas,
    Prepositions,
)
from krrood.entity_query_language.verbalization.vocabulary.words import (
    GrammaticalNumber,
)


class ConditionAssembler(Assembler[Comparator, None]):
    """
    Render a condition in a requested surface form (predicate / modifier / …) — the
    single owner of every surface form a condition can take.

    A comparator/condition is said differently depending on where it sits: a standalone predicate
    (*"x is greater than 5"*), a post-nominal attribute modifier on a subject (the bare *"<attribute>
    op <value>"* that a *"whose …"* envelope wraps), a range modifier (*"<attribute> is between low and
    high"*), or the inference whose-attribute body (*"<attribute> is <value>"* agreeing in number).

    Reference: :cite:t:`gatt2009simplenlg` — surface realisation.
    """

    def realize(self, node: Comparator, plan: None = None) -> VerbalizationFragment:
        """
        :param node: The condition (comparator) to render.
        :param plan: Unused (this assembler has no plan).
        :return: The default form — a standalone predicate.

        Choosing the predicate form as the default is what makes a bare comparator render as the full
        *the battery of a Robot is greater than 50* sentence; it adds no phrasing of its own beyond
        that choice.

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(robot.battery > 50)
        'the battery of a Robot is greater than 50'
        """
        return self.predicate(node)

    def predicate(
        self, comparator: Comparator, *, negated: bool = False
    ) -> VerbalizationFragment:
        """
        :param comparator: The comparator to render.
        :param negated: Whether an outer negation applies.
        :return: The standalone predicate — dispatched over the :class:`PredicateTransform` registry,
            so the generic *"<left> <operator> <right>"* is one transform alongside the absence
            (*"has no …"* / *"does not exist"*) and boolean-polarity (*"is [not] <attr>"*) forms; the
            most-specific applicable one wins, and adding a new one is a new subclass.

        Dispatching to the winning transform is what produces the example: the ``== None`` shape
        selects the absence transform, so the result is *a Mission has no priority* rather than a
        value comparison.

        >>> mission = variable(Mission, [])
        >>> verbalize_expression(mission.priority == None)
        'a Mission has no priority'
        """
        transform = PredicateTransform.most_applicable(comparator, negated)
        return transform.render(comparator, self.context, negated)

    def as_statements(
        self, conditions: List[SymbolicExpression]
    ) -> List[VerbalizationFragment]:
        """
        Say a list of conditions as standalone statements — the entry a caller uses when the
        conditions stand on their own (an ``AND``'s operands, a ``where`` block), as opposed to
        attaching to a subject noun (:func:`as_subject_restrictions`). The verbalizer decides
        everything inside: it reduces the conjuncts (a complementary lower/upper bound pair on one
        chain becomes one *"… is between …"*; co-indexed comparisons across two prefixes fold into
        one *"… have the same …"*) and says each resulting condition.

        The caller only knows it has conditions and that this says them; it never sees the folding,
        nor chooses among the per-form methods below.

        :param conditions: The conditions to say, in order.
        :return: One standalone-statement fragment per condition (after reduction), in order.

        It supplies the two clause fragments of the example — *the battery of a Robot is greater than
        50* and *the name of the Robot is 'x'* — as a list; the caller (:meth:`AndRule.build`) adds
        the *, and* that joins them.

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(and_(robot.battery > 50, robot.name == 'x'))
        "the battery of a Robot is greater than 50, and the name of the Robot is 'x'"
        """
        return [self.context.child(item) for item in reduce_conjuncts(list(conditions))]

    def attribute_modifier(
        self,
        comparator: Comparator,
        subject: Variable,
        number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
    ) -> VerbalizationFragment:
        """
        :param comparator: The comparator on *subject*'s single-hop attribute.
        :param subject: The subject variable.
        :param number: The number the attribute noun, operator, and value agree with — singular for a
            singular subject; plural for a plural one (a ranking / ordered report, or an aggregated
            inference antecedent — *"whose salaries are greater than 5"*).
        :return: The bare *"<attribute> <operator> <value>"* grouped predicate a *"whose …"* envelope
            wraps, all agreeing with *number* — the predicative operator factors its copula out so a
            plural subject reads *"are greater than"* (see :func:`~…predication.comparator_operator`).

        It owns the *battery is greater than 50* span of the example — the attribute noun, operator,
        and value with the subject dropped — which the caller's *whose* envelope then wraps.

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(an(entity(robot).where(robot.battery > 50)))
        'Find a Robot whose battery is greater than 50'
        """
        attribute = single_hop_attribute(comparator.left, subject)
        return PhraseFragment(
            parts=[
                RoleFragment.for_attribute(
                    attribute._owner_class_, attribute._attribute_name_, number=number
                ),
                comparator_operator(
                    comparator, self.context.services, compact=False, number=number
                ),
                self.context.child(comparator.right, number=number, as_value=True),
            ]
        )

    def with_attribute_modifier(
        self, comparator: Comparator, subject: Variable
    ) -> VerbalizationFragment:
        """
        :param comparator: An order comparison (``<``/``<=``/``>``/``>=``) on *subject*'s single-hop
            attribute.
        :param subject: The subject variable.
        :return: The compact post-nominal modifier *"with <attribute> <comparison>"* — copula- and
            article-less (*"with priority greater than 2"*), for an introduced entity noun where the
            fuller *"whose … is …"* clause would read heavily.

        It reuses the copula-less operator surface (:func:`~…predication.comparator_operator` with
        ``compact``), so the operator reads *"greater than"* rather than *"is greater than"*.

        >>> mission, robot = variable(Mission, []), variable(Robot, [])
        >>> verbalize_expression(an(entity(mission).where(mission.assigned_to == robot, robot.battery > 50)))
        'Find a Mission that is assigned to a Robot with battery greater than 50'
        """
        attribute = single_hop_attribute(comparator.left, subject)
        return PhraseFragment(
            parts=[
                Prepositions.WITH.as_fragment(),
                RoleFragment.for_attribute(
                    attribute._owner_class_, attribute._attribute_name_
                ),
                comparator_operator(comparator, self.context.services, compact=True),
                self.context.child(comparator.right, as_value=True),
            ]
        )

    def superlative_modifier(
        self, comparator: Comparator, subject: Variable
    ) -> VerbalizationFragment:
        """
        :param comparator: The ``subject.<chain> == max/min(over all <Type>.<chain>)`` comparator.
        :param subject: The subject variable.
        :return: The superlative selection modifier *"with the maximum <leaf>"* / *"with the
            minimum <leaf>"*.

        It owns the whole *with the maximum salary* span of the example — reading the aggregator to
        pick *maximum* over *minimum* and naming the leaf attribute *salary*.

        >>> employee, peers = variable(Employee, []), variable(Employee, [])
        >>> verbalize_expression(
        ...     an(entity(employee).where(employee.salary == the(entity(max(peers.salary)))))
        ... )
        'Find an Employee with the maximum salary'
        """
        fold = superlative_aggregation(comparator, subject)
        leaf = fold.aggregator._leaf_attribute_
        return PhraseFragment(
            parts=[
                Prepositions.WITH.as_fragment(),
                Articles.THE.as_fragment(),
                Aggregations.for_aggregator(type(fold.aggregator)).as_fragment(),
                RoleFragment.for_attribute(leaf._owner_class_, leaf._attribute_name_),
            ]
        )

    def range_modifier(
        self,
        range_fold: RangeFold,
        subject: Variable,
        number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
    ) -> VerbalizationFragment:
        """
        :param range_fold: The folded lower/upper bound pair on *subject*'s single-hop attribute.
        :param subject: The subject variable.
        :param number: The number the attribute noun and copula agree with — *"salaries are between
            …"* for a plural subject.
        :return: The modifier *"<attribute> is between low and high"*.

        It owns the *salary is between 100 and 200* span of the example — naming the attribute and
        emitting the *between … and …* frame over the fold's bounds — which the *whose* envelope wraps.

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(
        ...     an(entity(employee).where(and_(employee.salary > 100, employee.salary < 200)))
        ... )
        'Find an Employee whose salary is between 100 and 200'
        """
        attribute = single_hop_attribute(range_fold.chain_expression, subject)
        left = RoleFragment.for_attribute(
            attribute._owner_class_, attribute._attribute_name_, number=number
        )
        return build_between(
            left,
            self.context.child(range_fold.lower_expression),
            self.context.child(range_fold.upper_expression),
            compact=False,
            number=number,
        )

    def attribute_predicate(
        self,
        attribute_name: str,
        number: GrammaticalNumber,
        value: VerbalizationFragment,
    ) -> VerbalizationFragment:
        """
        The bare *"<attribute> <copula> <value>"* predicate (the noun and copula agree with
        *number*), with no source link on the noun — for a field binding whose owner is implicit.
        The caller prefixes each with its own *"whose"* and joins them *"whose …, and whose …"*,
        exactly as a query subject restriction does.

        :param attribute_name: The attribute / field's name.
        :param number: The grammatical number the noun and copula agree with.
        :param value: The value fragment (supplied by the caller; it may itself be number-folded).
        :return: The bare predicate *"<attribute> <copula> <value>"*.

        Each *"<field> is <value>"* binding of an inference consequent is one such predicate: this
        method owns the *container is …* and *handle is …* spans of the example (noun plus copula plus
        the caller's value), which the caller then prefixes with *"whose"* and joins.

        >>> connection = variable(FixedConnection, [])
        >>> drawer = inference(Drawer)(container=connection.parent, handle=connection.child)
        >>> verbalize_expression(entity(drawer))
        "If true, then there's a Drawer whose container is the parent of a FixedConnection, and whose handle is its child"
        """
        return PhraseFragment(
            parts=[
                self._attribute_noun(attribute_name, number),
                Copulas.for_number(number),
                value,
            ]
        )

    def _attribute_noun(
        self, name: str, number: GrammaticalNumber
    ) -> VerbalizationFragment:
        """
        :param name: The attribute's name.
        :param number: The grammatical number to tag for inflection.
        :return: A role-tagged attribute noun (no source link — name only) tagged with *number*.

        It supplies just the bare attribute noun: the *container* and *handle* words of the example
        come from here, while :meth:`attribute_predicate` adds the copula and value around each.

        >>> connection = variable(FixedConnection, [])
        >>> drawer = inference(Drawer)(container=connection.parent, handle=connection.child)
        >>> verbalize_expression(entity(drawer))
        "If true, then there's a Drawer whose container is the parent of a FixedConnection, and whose handle is its child"
        """
        return RoleFragment.for_attribute(None, name, number)
