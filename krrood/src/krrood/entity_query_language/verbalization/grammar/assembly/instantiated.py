"""
InstantiatedVariable **assembler** — realise an :class:`InstantiatedPlan` into
*"a TypeName where the field of the TypeName is … such that …"*.

It owns the order-dependent **constraint-deferral dance** (push a frame → build every
binding value → register the field-reference overrides → pop the frame → render the
deferred constraints): the order matters because no binding's value may be rendered
under a sibling binding's override, and the deferred constraints reference the overrides
(verified order-dependent — it cannot be pre-planned).  Number is only *tagged* here
(``Copulas.for_number`` for the copula, ``ctx.child(value, number=…)`` for the value); the
copula agreement and noun pluralisation are applied later by the
:class:`~krrood.entity_query_language.verbalization.rendering.morphology_processor.MorphologyProcessor`
pass.

Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
"""

from __future__ import annotations

from typing_extensions import Dict, List, Tuple

from krrood.entity_query_language.core.variable import InstantiatedVariable
from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    oxford_and,
    PhraseFragment,
    RoleFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.fragments.factory import phrase, word
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_ref import SourceRef
from krrood.entity_query_language.verbalization.grammar.assembly.base import Assembler
from krrood.entity_query_language.verbalization.grammar.planning.instantiated import (
    BindingPlan,
    InstantiatedPlan,
    InstantiatedPlanner,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Conjunctions,
    Copulas,
    Keywords,
    Prepositions,
)
from krrood.entity_query_language.verbalization.vocabulary.words import Number


class InstantiatedAssembler(Assembler[InstantiatedVariable, InstantiatedPlan]):
    """Realise an InstantiatedVariable from its :class:`InstantiatedPlan`."""

    planner = InstantiatedPlanner

    def realize(self, node, plan: InstantiatedPlan) -> VerbFragment:
        # A referring NP (referent_id below) — the CoreferenceProcessor reduces a repeat
        # mention to "the <type>" in document order, so no build-time seen check here.
        self.ctx.scope.push_constraint_frame()
        binding_frags, overrides = self._bindings(plan, node._type_)
        self.ctx.scope.binding_overrides.update(overrides)
        deferred = self.ctx.scope.pop_constraint_frame()
        constraint_frags = [self.ctx.child(expression) for expression in deferred]

        return self._phrase(node, plan.type_name, binding_frags, constraint_frags)

    # ── bindings ───────────────────────────────────────────────────────────────

    def _bindings(
        self, plan: InstantiatedPlan, type_cls
    ) -> Tuple[List[VerbFragment], Dict]:
        """Build every binding fragment and collect overrides (registered together after)."""
        binding_frags: List[VerbFragment] = []
        overrides: Dict = {}
        for binding in plan.bindings:
            field_ref = self._field_ref(binding.field_name, plan.type_name, type_cls)
            binding_frags.append(
                phrase(field_ref, self._copula(binding), self._value(binding))
            )
            overrides[binding.value._id_] = field_ref
        return binding_frags, overrides

    def _field_ref(self, field_name: str, type_name: str, type_cls) -> VerbFragment:
        """*"the <field> of the <Type>"* with proper semantic roles + source link."""
        return PhraseFragment(
            parts=[
                Articles.THE.as_fragment(),
                RoleFragment(text=field_name, role=SemanticRole.ATTRIBUTE),
                Prepositions.OF.as_fragment(),
                Articles.THE.as_fragment(),
                RoleFragment(
                    text=type_name,
                    role=SemanticRole.VARIABLE,
                    source_ref=(
                        SourceRef.for_type(type_cls)
                        if isinstance(type_cls, type)
                        else None
                    ),
                ),
            ]
        )

    def _copula(self, binding: BindingPlan) -> VerbFragment:
        return Copulas.for_number(Number.of(binding.is_plural))

    def _value(self, binding: BindingPlan) -> VerbFragment:
        return self.ctx.child(binding.value, number=Number.of(binding.is_plural))

    # ── phrase assembly ──────────────────────────────────────────────────────────

    def _phrase(
        self,
        node,
        type_name: str,
        binding_frags: List[VerbFragment],
        constraint_frags: List[VerbFragment],
    ) -> VerbFragment:
        modifiers: List[VerbFragment] = []
        if binding_frags:
            joined = oxford_and(binding_frags, Conjunctions.AND.as_fragment())
            modifiers.append(
                PhraseFragment(parts=[word(","), Keywords.WHERE.as_fragment(), joined])
            )
        if constraint_frags:
            joined_c = oxford_and(constraint_frags, Conjunctions.AND.as_fragment())
            modifiers.append(
                PhraseFragment(
                    parts=[word(","), Keywords.SUCH_THAT.as_fragment(), joined_c]
                )
            )
        # A referring NP: "a <type>" first mention (+ appositive clauses), reduced to
        # "the <type>" on a repeat by the CoreferenceProcessor (which drops the modifiers).
        return NounPhrase(
            head=RoleFragment.for_variable(type_name, node),
            modifiers=modifiers,
            modifier_separator="",
            referent_id=node._id_,
        )
