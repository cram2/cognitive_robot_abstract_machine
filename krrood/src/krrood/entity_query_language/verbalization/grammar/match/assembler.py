from __future__ import annotations

from typing_extensions import List, Optional

from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    Fragment,
    oxford_comma,
    PhraseFragment,
    RoleFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.grammar.conditions.assembler import (
    ConditionAssembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    is_atomic_value,
    is_none_literal,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import (
    Assembler,
)
from krrood.entity_query_language.verbalization.grammar.match.planner import (
    AttributeGroup,
    MatchPlan,
    MatchPlanner,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Absence,
    Conjunctions,
    Copulas,
    Directive,
    Keywords,
    Prepositions,
    Pronouns,
)

#: The most attribute values coordinated under one *"… are 1, 2, and 3 respectively"* point before
#: each is said separately — beyond this the reader can no longer reliably zip attributes to values.
_MAX_RESPECTIVELY = 3


class MatchAssembler(Assembler[Match, MatchPlan]):
    """
    Realise a match into *"Find/Generate <selection> [, and predict its … values]"* with a
    *"given that"* block (the construction pattern, attributes aggregated per object) and a
    *"where"* block (the free conditions), each condition its own point.

    The selection and every condition/value are recursed through ``context.child``, so the existing
    chain / comparator / coreference machinery renders them; this assembler only decides the
    match-specific structure.

    Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
    """

    planner = MatchPlanner

    def realize(self, node: Match, plan: MatchPlan) -> Fragment:
        """
        :param node: The match being verbalised.
        :param plan: The match plan.
        :return: The match's block, sourced at the resolved query so the coreference pass scopes the
            selection as the discourse subject (*"its …"*).
        """
        predict_groups = [group for group in plan.groups if group.predicted]
        inline_predict = self._inline_predict(predict_groups, plan)

        header_parts: List[Fragment] = [
            Directive.for_underspecified(plan.underspecified).as_fragment(),
            self.context.child(plan.selection),
        ]
        if inline_predict is not None:
            header_parts.append(inline_predict)

        items: List[Fragment] = []
        if inline_predict is None and predict_groups:
            items.append(self._predict_block(predict_groups, plan))
        given = self._given_that_block(plan)
        if given is not None:
            items.append(given)
        where = self._where_block(plan)
        if where is not None:
            items.append(where)

        return BlockFragment(
            header=PhraseFragment(parts=header_parts),
            items=items,
            source=node.expression,
        )

    # ── predict ────────────────────────────────────────────────────────────────

    def _inline_predict(
        self, predict_groups: List[AttributeGroup], plan: MatchPlan
    ) -> Optional[Fragment]:
        """:return: The header-folded *"and predict its <attrs> value(s)"* clause when the only
        predicted attributes are the selection's own (the simple case), else ``None`` (a *"predict"*
        block is used instead — see :meth:`_predict_block`)."""
        if len(predict_groups) != 1:
            return None
        group = predict_groups[0]
        if group.object._id_ != plan.selection._id_:
            return None
        attributes = [assignment.attribute for assignment in group.predicted]
        noun = "values" if len(attributes) > 1 else "value"
        return PhraseFragment(
            parts=[
                Conjunctions.AND.as_fragment(),
                Keywords.PREDICT.as_fragment(),
                Pronouns.ITS.as_fragment(),
                self._attribute_list(attributes),
                WordFragment(text=noun),
            ]
        )

    def _predict_block(
        self, predict_groups: List[AttributeGroup], plan: MatchPlan
    ) -> Fragment:
        """:return: The *"and predict"* block — one point per object whose attributes are generated
        (*"x, y, and z of its position"*, or *"its <attrs>"* for the selection's own).
        """
        points = [self._predict_point(group, plan) for group in predict_groups]
        return BlockFragment(
            header=PhraseFragment(
                parts=[Conjunctions.AND.as_fragment(), Keywords.PREDICT.as_fragment()]
            ),
            items=points,
        )

    def _predict_point(self, group: AttributeGroup, plan: MatchPlan) -> Fragment:
        """:return: *"its <attrs>"* for the selection's own attributes, else
        *"<attrs> of <object>"* (*"x, y, and z of its position"*)."""
        attribute_list = self._attribute_list(
            [assignment.attribute for assignment in group.predicted]
        )
        if group.object._id_ == plan.selection._id_:
            return PhraseFragment(parts=[Pronouns.ITS.as_fragment(), attribute_list])
        return PhraseFragment(
            parts=[
                attribute_list,
                Prepositions.OF.as_fragment(),
                self.context.child(group.object),
            ]
        )

    # ── given that ───────────────────────────────────────────────────────────

    def _given_that_block(self, plan: MatchPlan) -> Optional[Fragment]:
        """:return: The *"given that"* block — one point per attribute group (concrete assignments)
        and per non-grouping condition — or ``None`` when there is nothing to give."""
        points: List[Fragment] = []
        for group in plan.groups:
            if group.concrete:
                points += self._concrete_points(group)
        points += ConditionAssembler(self.context).as_statements(plan.other_conditions)
        if not points:
            return None
        return BlockFragment(header=Keywords.GIVEN_THAT.as_fragment(), items=points)

    def _concrete_points(self, group: AttributeGroup) -> List[Fragment]:
        """:return: The given-that points for a group's concrete assignments. Atomic scalar values
        coordinate under one *"x, y, and z of the <object> are 1, 2, and 3 respectively"* point — but
        only up to :data:`_MAX_RESPECTIVELY` of them, since the reader must zip attributes to values
        in order. A value that renders as a phrase (*"one of …"*, *"a specific …"*, a sub-query), or
        any assignment once the cap is exceeded, gets its own *"x of the <object> is …"* point; and
        ``None`` assignments their own *"the <object> has no <attrs>"* point (an absence flips
        subject/object and cannot fold into the coordination)."""
        present = [a for a in group.concrete if not is_none_literal(a.value)]
        absent = [a for a in group.concrete if is_none_literal(a.value)]
        grouped, singles = self._split_groupable(present)
        points: List[Fragment] = []
        if grouped:
            points.append(self._group_point(group, grouped))
        points += [self._group_point(group, [a]) for a in singles]
        if absent:
            points.append(self._absence_point(group, absent))
        return points

    @staticmethod
    def _split_groupable(present: List) -> tuple:
        """:return: ``(grouped, singles)`` — the atomic-scalar assignments to coordinate under one
        *"respectively"* point, and the assignments to say each on their own. Grouping applies only
        when 2..:data:`_MAX_RESPECTIVELY` atomic values are present; otherwise every assignment stands
        alone (a lone value needs no *"respectively"*, and too many would be unreadable to zip).
        """
        atomic = [a for a in present if is_atomic_value(a.value)]
        if 2 <= len(atomic) <= _MAX_RESPECTIVELY:
            return atomic, [a for a in present if not is_atomic_value(a.value)]
        return [], present

    def _absence_point(self, group: AttributeGroup, absent: List) -> Fragment:
        """:return: *"the <object> has no <attrs>"* for attributes assigned ``None``."""
        return PhraseFragment(
            parts=[
                self.context.child(group.object),
                Absence.HAS_NO.as_fragment(),
                self._attribute_list([a.attribute for a in absent]),
            ]
        )

    def _group_point(self, group: AttributeGroup, concrete: List) -> Fragment:
        """:return: *"x, y, and z of the <object> are 1, 2, and 3 respectively"* for several
        attributes, or *"x of the <object> is 1"* for one."""
        attribute_list = self._attribute_list([a.attribute for a in concrete])
        object_phrase = self.context.child(group.object)
        parts: List[Fragment] = [
            attribute_list,
            Prepositions.OF.as_fragment(),
            object_phrase,
        ]
        if len(concrete) == 1:
            parts += [
                Copulas.IS.as_fragment(),
                self.context.child(concrete[0].value, as_value=True),
            ]
        else:
            value_list = oxford_comma(
                [self.context.child(a.value, as_value=True) for a in concrete],
                Conjunctions.AND.as_fragment(),
            )
            parts += [
                Copulas.ARE.as_fragment(),
                value_list,
                Keywords.RESPECTIVELY.as_fragment(),
            ]
        return PhraseFragment(parts=parts)

    # ── where ────────────────────────────────────────────────────────────────

    def _where_block(self, plan: MatchPlan) -> Optional[Fragment]:
        """:return: The *"where"* block — one point per free condition — or ``None`` when absent.

        The points are whatever the condition verbalizer makes of the ``where`` conditions — the
        assembler only knows it has a list of conditions to say, and hands them over; folding a
        bound pair into a *between* is the verbalizer's concern, not this one's.
        """
        if not plan.where_conditions:
            return None
        points = ConditionAssembler(self.context).as_statements(plan.where_conditions)
        return BlockFragment(header=Keywords.WHERE.as_fragment(), items=points)

    # ── shared ───────────────────────────────────────────────────────────────

    def _attribute_list(self, attributes: List[Attribute]) -> Fragment:
        """:return: The attribute names as a single fragment — *"x, y, and z"* or *"x"*."""
        fragments = [
            RoleFragment.for_attribute(
                attribute._owner_class_, attribute._attribute_name_
            )
            for attribute in attributes
        ]
        if len(fragments) == 1:
            return fragments[0]
        return oxford_comma(fragments, Conjunctions.AND.as_fragment())
