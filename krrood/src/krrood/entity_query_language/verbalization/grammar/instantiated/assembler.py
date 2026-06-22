from __future__ import annotations

import uuid

from typing_extensions import Dict, List, Tuple

from krrood.entity_query_language.core.variable import InstantiatedVariable
from krrood.entity_query_language.verbalization.navigation_path import PathStep
from krrood.entity_query_language.verbalization.fragments.features import Separator
from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    oxford_comma,
    PhraseFragment,
    RoleFragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import (
    Assembler,
)
from krrood.entity_query_language.verbalization.grammar.instantiated.planner import (
    BindingPlan,
    InstantiatedPlan,
    InstantiatedPlanner,
)
from krrood.entity_query_language.verbalization.microplanning.possessive import (
    possessive_path,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Conjunctions,
    Copulas,
    Keywords,
    Punctuation,
)
from krrood.entity_query_language.verbalization.vocabulary.words import Number


class InstantiatedAssembler(Assembler[InstantiatedVariable, InstantiatedPlan]):
    """
    Realise an instantiated variable from its plan into *"a TypeName where the field of the
    TypeName is … such that …"*.

    It owns the order-dependent constraint-deferral handling: a binding's value must not be
    rendered under a sibling binding's override, and the deferred constraints reference those
    overrides.

    Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
    """

    planner = InstantiatedPlanner

    def realize(self, node: InstantiatedVariable, plan: InstantiatedPlan) -> Fragment:
        """
        :param node: The instantiated variable.
        :param plan: The instantiated plan.
        :return: *"a TypeName, where the <field> of the TypeName is <value> …, such that
            <deferred>"*.
        """
        self.context.scope.push_constraint_frame()
        binding_fragments, overrides = self._bindings(plan, node._type_)
        self.context.scope.binding_overrides.update(overrides)
        deferred = self.context.scope.pop_constraint_frame()
        constraint_fragments = [
            self.context.child(expression) for expression in deferred
        ]

        return self._phrase(
            node, plan.type_name, binding_fragments, constraint_fragments
        )

    # ── bindings ───────────────────────────────────────────────────────────────

    def _bindings(
        self, plan: InstantiatedPlan, instantiated_type: type
    ) -> Tuple[List[Fragment], Dict[uuid.UUID, Fragment]]:
        """:return: Every binding fragment and the field-reference overrides."""
        binding_fragments: List[Fragment] = []
        overrides: Dict[uuid.UUID, Fragment] = {}
        for binding in plan.bindings:
            field_reference = self._field_reference(
                binding.field_name, plan.type_name, instantiated_type
            )
            binding_fragments.append(
                PhraseFragment(
                    parts=[field_reference, self._copula(binding), self._value(binding)]
                )
            )
            overrides[binding.value._id_] = field_reference
        return binding_fragments, overrides

    def _field_reference(
        self, field_name: str, type_name: str, instantiated_type: type
    ) -> Fragment:
        """:return: *"the <field> of the <Type>"* — a single-hop possessive."""
        type_root = PhraseFragment(
            parts=[
                Articles.THE.as_fragment(),
                RoleFragment.for_type(instantiated_type, text=type_name),
            ]
        )
        return possessive_path([PathStep(field_name, None)], type_root)

    def _copula(self, binding: BindingPlan) -> Fragment:
        """:return: *"is"* / *"are"* agreeing with the binding's plurality."""
        return Copulas.for_number(Number.of(binding.is_plural))

    def _value(self, binding: BindingPlan) -> Fragment:
        """:return: The binding's value expression, rendered in the binding's number."""
        return self.context.child(binding.value, number=Number.of(binding.is_plural))

    # ── phrase assembly ──────────────────────────────────────────────────────────

    def _phrase(
        self,
        node: InstantiatedVariable,
        type_name: str,
        binding_fragments: List[Fragment],
        constraint_fragments: List[Fragment],
    ) -> Fragment:
        """:return: *"a <type>, where <bindings>, such that <constraints>"* — the referring noun
        phrase with its appositive clauses as droppable modifiers."""
        modifiers: List[Fragment] = []
        if binding_fragments:
            # Bindings and constraints are independent clauses → a two-clause pair keeps its comma.
            joined = oxford_comma(
                binding_fragments, Conjunctions.AND.as_fragment(), pair_comma=True
            )
            modifiers.append(
                PhraseFragment(
                    parts=[
                        Punctuation.COMMA.as_fragment(),
                        Keywords.WHERE.as_fragment(),
                        joined,
                    ]
                )
            )
        if constraint_fragments:
            joined_constraints = oxford_comma(
                constraint_fragments, Conjunctions.AND.as_fragment(), pair_comma=True
            )
            modifiers.append(
                PhraseFragment(
                    parts=[
                        Punctuation.COMMA.as_fragment(),
                        Keywords.SUCH_THAT.as_fragment(),
                        joined_constraints,
                    ]
                )
            )
        # A referring noun phrase: "a <type>" first mention (+ appositive clauses), reduced to
        # "the <type>" on a repeat by the CoreferenceProcessor (which drops the modifiers).
        return NounPhrase(
            head=RoleFragment.for_variable(type_name, node),
            modifiers=modifiers,
            modifier_separator=Separator.NONE,
            referent_id=node._id_,
        )
