from __future__ import annotations

import uuid

from typing_extensions import List, Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.core.expression_structure import walk_chain
from krrood.entity_query_language.verbalization.navigation_path import (
    build_path_parts,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    PhraseFragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import (
    Assembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.assembler import (
    ConditionAssembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.forms import (
    as_subject_restrictions,
)
from krrood.entity_query_language.verbalization.grammar.inference.planner import (
    AggregationStatus,
    AntecedentInformation,
    ConsequentBinding,
    InferencePlanner,
    RuleStructure,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Conjunctions,
    ExistentialPhrase,
    FallbackNouns,
    GroupKeyPhrases,
    Keywords,
)
from krrood.entity_query_language.verbalization.vocabulary.words import Number


class InferenceAssembler(Assembler[Entity, RuleStructure]):
    """
    Realise the IF/THEN block from a rule structure.

    Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
    """

    planner = InferencePlanner

    def realize(self, node: Entity, plan: RuleStructure) -> Fragment:
        """
        :param node: The inference-rule query.
        :param plan: The IF/THEN rule structure.
        :return: *"If <antecedents…>, then <consequent…>"* — the two-block IF/THEN form.
        """
        return BlockFragment(
            header=None,
            items=[
                BlockFragment(
                    header=Keywords.IF.as_fragment(), items=self._if_items(plan)
                ),
                BlockFragment(
                    header=Keywords.THEN.as_fragment(), items=self._then_items(plan)
                ),
            ],
        )

    @staticmethod
    def _number(antecedent: AntecedentInformation) -> Number:
        """:return: The grammatical number of an antecedent — plural if and only if aggregated."""
        return Number.of(antecedent.aggregation_status == AggregationStatus.AGGREGATED)

    # ── IF clause ───────────────────────────────────────────────────────────────

    def _if_items(self, structure: RuleStructure) -> List[Fragment]:
        """
        :return: One item per antecedent — *"there's a <Type> whose …, and …"* — plus any unmatched
            conditions; *"true"* when there are none.
        """
        items: List[Fragment] = [
            self._antecedent(antecedent) for antecedent in structure.primary_antecedents
        ]
        items += [
            self.context.child(condition)
            for condition in structure.unmatched_conditions
        ]
        return items or [Keywords.TRUE.as_fragment()]

    def _antecedent(self, antecedent: AntecedentInformation) -> Fragment:
        """:return: The antecedent as a bulleted list entry whose conditions hang beneath it — the
        existential intro woven with its conditions by the shared restriction machinery (the same
        *"whose"* group / *"such that …"* form a query selection uses). Inline / in paragraph this
        reads *"there's a <Type> whose a, and b"*; in hierarchical the conditions are sub-points.
        """
        intro = self._antecedent_intro(antecedent)
        if not antecedent.conditions or antecedent.variable is None:
            return intro
        restriction = as_subject_restrictions(
            antecedent.conditions,
            antecedent.variable,
            self.context,
            self._number(antecedent),
        )
        header = (
            PhraseFragment(parts=[intro, *restriction.inline_modifiers])
            if restriction.inline_modifiers
            else intro
        )
        items: List[Fragment] = []
        if restriction.whose is not None:
            items.append(restriction.whose)
        if restriction.residual is not None:
            items.append(
                PhraseFragment(
                    parts=[Keywords.SUCH_THAT.as_fragment(), restriction.residual]
                )
            )
        if not items:
            return header
        return BlockFragment(header=header, items=items, bulleted_header=True)

    def _antecedent_intro(self, antecedent: AntecedentInformation) -> Fragment:
        """:return: *"there's a <Type>"* / *"there are <Types>"* — the antecedent's existential intro."""
        return ExistentialPhrase.for_number(self._number(antecedent)).build_phrase(
            antecedent.type_name, referent_id=self._antecedent_referent_id(antecedent)
        )

    @staticmethod
    def _antecedent_referent_id(
        antecedent: AntecedentInformation,
    ) -> Optional[uuid.UUID]:
        """
        :return: The antecedent's canonical referent id — the selected variable for an entity
            root, else the root's own id (matching the variable the THEN-clause chains reference).
        """
        root = antecedent.root
        if isinstance(root, Entity):
            root.build()
            return getattr(root.selected_variable, "_id_", None)
        return getattr(root, "_id_", None)

    # ── THEN clause ───────────────────────────────────────────────────────────

    def _then_items(self, structure: RuleStructure) -> List[Fragment]:
        """:return: The consequent as a single bulleted entry — *"there's a <Consequent>"* with its
        field bindings under one *"whose"* group (the same form a query subject restriction uses):
        *"whose <field> is <value>, and …"* inline / in paragraph, sub-points in hierarchical.
        """
        intro: Fragment = ExistentialPhrase.for_number(Number.SINGULAR).build_phrase(
            structure.consequent_type
        )
        bindings = [
            self._binding_predicate(binding)
            for binding in structure.consequent_bindings
        ]
        if not bindings:
            return [intro]
        whose = BlockFragment(
            header=Keywords.WHOSE.as_fragment(),
            items=bindings,
            conjunction=Conjunctions.AND.as_fragment(),
        )
        return [BlockFragment(header=intro, items=[whose], bulleted_header=True)]

    def _binding_predicate(self, binding: ConsequentBinding) -> Fragment:
        """:return: The bare *"<field> is/are <value>"* predicate for one consequent binding (the
        shared *"whose"* envelope is added once by :meth:`_then_items`)."""
        number = Number.of(binding.is_plural_field)
        return ConditionAssembler(self.context).attribute_predicate(
            binding.field_name, number, self._binding_value(binding)
        )

    def _binding_value(self, binding: ConsequentBinding) -> Fragment:
        """
        :return: The binding's value: *"the <plural chain>"* (aggregated), bare plural, the
            group-key *"common …"* phrase, or the plain rendering.
        """
        if (
            binding.is_plural_field
            and binding.aggregation_status == AggregationStatus.AGGREGATED
        ):
            return PhraseFragment(
                parts=[
                    Articles.THE.as_fragment(),
                    self.context.child(binding.value_expression, number=Number.PLURAL),
                ]
            )
        if binding.is_plural_field:
            return self.context.child(binding.value_expression, number=Number.PLURAL)
        if binding.aggregation_status == AggregationStatus.GROUP_KEY:
            return self._group_key_value(binding.value_expression)
        return self.context.child(binding.value_expression)

    def _group_key_value(self, expression: SymbolicExpression) -> Fragment:
        """:return: *"the common <field> of the <Roots>"* — a binding that refers to a GROUP BY key."""
        chain, current = walk_chain(expression)
        if not chain or not isinstance(current, Variable):
            return self.context.child(expression)
        root_type = (
            current._type_.__name__
            if getattr(current, "_type_", None)
            else FallbackNouns.ENTITY.text
        )
        parts = build_path_parts(chain)
        field = list(reversed(parts))[0].name if parts else root_type
        return GroupKeyPhrases.COMMON_OF.build_phrase(field, root_type)
