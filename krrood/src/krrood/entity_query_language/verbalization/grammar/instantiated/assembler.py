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
    VerbalizationFragment,
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
from krrood.entity_query_language.verbalization.vocabulary.words import (
    GrammaticalNumber,
)


class InstantiatedAssembler(Assembler[InstantiatedVariable, InstantiatedPlan]):
    """
    Realise an instantiated variable from its plan into *"a TypeName where the field of the
    TypeName is … such that …"*.

    It owns the order-dependent constraint-deferral handling: a binding's value must not be
    rendered under a sibling binding's override, and the deferred constraints reference those
    overrides.

    Reference: :cite:t:`gatt2009simplenlg` — surface realisation.

    >>> connection = variable(FixedConnection, [])
    >>> verbalize_expression(inference(Drawer)(container=connection.parent, handle=connection.child))
    'a Drawer, where the container of the Drawer is the parent of a FixedConnection, and the handle of the Drawer is the child of the FixedConnection'
    """

    planner = InstantiatedPlanner

    def realize(
        self, node: InstantiatedVariable, plan: InstantiatedPlan
    ) -> VerbalizationFragment:
        """
        :param node: The instantiated variable.
        :param plan: The instantiated plan.
        :return: *"a TypeName, where the <field> of the TypeName is <value> …, such that
            <deferred>"*.

        Its contribution is the orchestration that yields the whole shown phrase: it opens a
        constraint frame so sibling-override constraints can be deferred into the *"such that"* tail,
        collects the *"where …"* bindings, then hands both to :meth:`_phrase`. With no deferred
        constraints here the result is just *"a Drawer, where …"* with no *"such that"* tail.

        >>> connection = variable(FixedConnection, [])
        >>> verbalize_expression(inference(Drawer)(container=connection.parent, handle=connection.child))
        'a Drawer, where the container of the Drawer is the parent of a FixedConnection, and the handle of the Drawer is the child of the FixedConnection'
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

    # %% bindings

    def _bindings(
        self, plan: InstantiatedPlan, instantiated_type: type
    ) -> Tuple[List[VerbalizationFragment], Dict[uuid.UUID, VerbalizationFragment]]:
        """:return: Every binding fragment and the field-reference overrides.

        Its contribution is the *"where"* clause body: it builds each *"the container of the Drawer is
        the parent of a FixedConnection"* triple by joining a :meth:`_field_reference`, a
        :meth:`_copula`, and a :meth:`_value`. It also records each field reference as an override so a
        later binding can refer back to it, which is why both bindings appear side by side here.

        >>> connection = variable(FixedConnection, [])
        >>> verbalize_expression(inference(Drawer)(container=connection.parent, handle=connection.child))
        'a Drawer, where the container of the Drawer is the parent of a FixedConnection, and the handle of the Drawer is the child of the FixedConnection'
        """
        binding_fragments: List[VerbalizationFragment] = []
        overrides: Dict[uuid.UUID, VerbalizationFragment] = {}
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
    ) -> VerbalizationFragment:
        """:return: *"the <field> of the <Type>"* — a single-hop possessive (*"the container of the
        Drawer"*).

        Its contribution is only the left side of each binding — the *"the container of the Drawer"*
        and *"the handle of the Drawer"* possessives in the shown phrase; the *"is"* and the value
        after them are supplied by :meth:`_copula` and :meth:`_value`.

        >>> connection = variable(FixedConnection, [])
        >>> verbalize_expression(inference(Drawer)(container=connection.parent, handle=connection.child))
        'a Drawer, where the container of the Drawer is the parent of a FixedConnection, and the handle of the Drawer is the child of the FixedConnection'
        """
        type_root = PhraseFragment(
            parts=[
                Articles.THE.as_fragment(),
                RoleFragment.for_type(instantiated_type, text=type_name),
            ]
        )
        return possessive_path([PathStep(field_name, None)], type_root)

    def _copula(self, binding: BindingPlan) -> VerbalizationFragment:
        """:return: *"is"* / *"are"* agreeing with the binding's plurality (the plural ``drawers``
        field takes *"are"*).

        >>> verbalize_expression(inference(Cabinet)(container=variable(Container, []), drawers=variable(Drawer, [])))
        'a Cabinet, where the container of the Cabinet is a Container, and the drawers of the Cabinet are Drawers'
        """
        return Copulas.for_number(GrammaticalNumber.of(binding.is_plural))

    def _value(self, binding: BindingPlan) -> VerbalizationFragment:
        """:return: The binding's value expression, rendered in the binding's number (*"the parent of
        a FixedConnection"*).

        Its contribution is only the right side of each binding — the *"the parent of a
        FixedConnection"* / *"the child of the FixedConnection"* values after *"is"*; recursing
        through ``context.child`` is what renders them as full chains rather than bare names.

        >>> connection = variable(FixedConnection, [])
        >>> verbalize_expression(inference(Drawer)(container=connection.parent, handle=connection.child))
        'a Drawer, where the container of the Drawer is the parent of a FixedConnection, and the handle of the Drawer is the child of the FixedConnection'
        """
        return self.context.child(
            binding.value, number=GrammaticalNumber.of(binding.is_plural)
        )

    # %% phrase assembly

    def _phrase(
        self,
        node: InstantiatedVariable,
        type_name: str,
        binding_fragments: List[VerbalizationFragment],
        constraint_fragments: List[VerbalizationFragment],
    ) -> VerbalizationFragment:
        """:return: *"a <type>, where <bindings>, such that <constraints>"* — the referring noun
        phrase with its appositive clauses as droppable modifiers.

        Its contribution is the envelope around the bindings: the *"a Drawer"* head plus the *",
        where …"* (and, when present, *", such that …"*) appositive modifiers, joined with the
        oxford comma seen between the two bindings. The modifiers are droppable, so a repeat mention
        reduces to just *"the Drawer"*.

        >>> connection = variable(FixedConnection, [])
        >>> verbalize_expression(inference(Drawer)(container=connection.parent, handle=connection.child))
        'a Drawer, where the container of the Drawer is the parent of a FixedConnection, and the handle of the Drawer is the child of the FixedConnection'
        """
        modifiers: List[VerbalizationFragment] = []
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
