"""
Inference-rule **assembler** — realise a :class:`RuleStructure` (from
:class:`~krrood.entity_query_language.verbalization.grammar.planning.inference.InferencePlanner`)
into an ``IF … THEN …`` :class:`~krrood.entity_query_language.verbalization.fragments.base.BlockFragment`.

Realisation sub-steps are methods sharing ``self.ctx`` (recursion via ``self.ctx.child``,
coreference via ``self.ctx.refer``).  This assembler only decides + *tags* grammatical
:class:`~krrood.entity_query_language.verbalization.fragments.features.Number` (via
``Copulas.for_number`` / the lexicon frames); the actual copula agreement and noun pluralisation are
applied once, later, by the
:class:`~krrood.entity_query_language.verbalization.rendering.morphology_processor.MorphologyProcessor`
pass.  This is the realisation half of the planner/assembler split (see
:class:`~krrood.entity_query_language.verbalization.grammar.assembly.base.Assembler`).

Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation + the MorphologyProcessor.
"""

from __future__ import annotations

from typing_extensions import List

from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.verbalization.chain_utils import (
    build_path_parts,
    walk_chain,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.fragments.factory import phrase
from krrood.entity_query_language.verbalization.grammar.assembly.base import Assembler
from krrood.entity_query_language.verbalization.grammar.conditions.verbalizer import (
    ConditionVerbalizer,
)
from krrood.entity_query_language.verbalization.grammar.planning.inference import (
    AggregationStatus,
    AntecedentInfo,
    ConsequentBinding,
    InferencePlanner,
    ConditionPlan,
    RuleStructure,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    ExistentialPhrase,
    FallbackNouns,
    GroupKeyPhrases,
    Keywords,
)
from krrood.entity_query_language.verbalization.vocabulary.words import Number


class InferenceAssembler(Assembler[Entity, RuleStructure]):
    """Realise the IF/THEN block from a :class:`RuleStructure`."""

    planner = InferencePlanner

    def realize(self, node, plan: RuleStructure) -> VerbFragment:
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
    def _number(antecedent: AntecedentInfo) -> Number:
        """The grammatical number of an antecedent — plural iff aggregated."""
        return Number.of(antecedent.aggregation_status == AggregationStatus.AGGREGATED)

    # ── IF clause ───────────────────────────────────────────────────────────────

    def _if_items(self, s: RuleStructure) -> List[VerbFragment]:
        for antecedent in s.secondary_antecedents:
            self._register_antecedent(antecedent)

        items: List[VerbFragment] = []
        for antecedent in s.primary_antecedents:
            intro = self._antecedent_intro(antecedent)
            self._register_antecedent(antecedent)
            cond_frags = self._condition_frags(antecedent.conditions, antecedent)
            items.append(
                BlockFragment(header=intro, items=cond_frags) if cond_frags else intro
            )

        for condition in s.unmatched_conditions:
            items.append(self.ctx.child(condition))

        return items or [Keywords.TRUE.as_fragment()]

    def _antecedent_intro(self, antecedent: AntecedentInfo) -> VerbFragment:
        # Pass the antecedent's referent so the coreference pass marks it introduced — a later
        # mention (e.g. "the parent of the FixedConnection" in the THEN clause) then reads "the".
        return ExistentialPhrase.for_number(self._number(antecedent)).build_phrase(
            antecedent.type_name, referent_id=self._antecedent_referent_id(antecedent)
        )

    @staticmethod
    def _antecedent_referent_id(antecedent: AntecedentInfo):
        """The antecedent's canonical referent id — the selected variable for an Entity root,
        else the root's own id (matching the variable the THEN-clause chains reference).
        """
        root = antecedent.root
        if isinstance(root, Entity):
            root.build()
            return getattr(root.selected_variable, "_id_", None)
        return getattr(root, "_id_", None)

    def _register_antecedent(self, antecedent: AntecedentInfo) -> None:
        root = antecedent.root
        self.ctx.refer.mark_introduced(root)
        if isinstance(root, Entity):
            root.build()
            sel = root.selected_variable
            if sel is not None and hasattr(sel, "_id_"):
                self.ctx.refer.mark_introduced(sel)

    def _condition_frags(
        self, conditions: List[ConditionPlan], antecedent: AntecedentInfo
    ) -> List[VerbFragment]:
        return [self._condition_frag(pc, antecedent) for pc in conditions]

    def _condition_frag(
        self, pc: ConditionPlan, antecedent: AntecedentInfo
    ) -> VerbFragment:
        """Render one condition: a *"whose <attr> is …"* modifier when foldable, else recurse."""
        if pc.whose_attr is None:
            return self.ctx.child(pc.expression)
        number = self._number(antecedent)
        value = self._value(pc.expression.right, number)
        return ConditionVerbalizer(self.ctx).whose_attribute(
            pc.whose_attr, number, value
        )

    def _value(self, expression, number: Number) -> VerbFragment:
        """Render a value expression agreeing with *number* (plural folds the chain)."""
        return self.ctx.child(expression, number=number)

    # ── THEN clause ───────────────────────────────────────────────────────────

    def _then_items(self, s: RuleStructure) -> List[VerbFragment]:
        intro: VerbFragment = ExistentialPhrase.for_number(
            Number.SINGULAR
        ).build_phrase(s.consequent_type)
        binding_frags = [self._binding_frag(b) for b in s.consequent_bindings]
        if not binding_frags:
            return [intro]
        return [BlockFragment(header=intro, items=binding_frags)]

    def _binding_frag(self, binding: ConsequentBinding) -> VerbFragment:
        number = Number.of(binding.is_plural_field)
        return ConditionVerbalizer(self.ctx).whose_attribute(
            binding.field_name, number, self._binding_value(binding)
        )

    def _binding_value(self, binding: ConsequentBinding) -> VerbFragment:
        if (
            binding.is_plural_field
            and binding.aggregation_status == AggregationStatus.AGGREGATED
        ):
            return phrase(
                Articles.THE.as_fragment(),
                self.ctx.child(binding.value_expression, number=Number.PLURAL),
            )
        if binding.is_plural_field:
            return self.ctx.child(binding.value_expression, number=Number.PLURAL)
        if binding.aggregation_status == AggregationStatus.GROUP_KEY:
            return self._group_key_value(binding.value_expression)
        return self.ctx.child(binding.value_expression)

    def _group_key_value(self, expression) -> VerbFragment:
        chain, current = walk_chain(expression)
        if not chain or not isinstance(current, Variable):
            return self.ctx.child(expression)
        root_type = (
            current._type_.__name__
            if getattr(current, "_type_", None)
            else FallbackNouns.ENTITY.text
        )
        self.ctx.refer.mark_introduced(current)
        parts = build_path_parts(chain)
        field = list(reversed(parts))[0][0] if parts else root_type
        return GroupKeyPhrases.COMMON_OF.build_phrase(field, root_type)
