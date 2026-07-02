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
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import (
    Assembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.assembler import (
    ConditionAssembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.placement import (
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
from krrood.entity_query_language.verbalization.vocabulary.words import (
    GrammaticalNumber,
)


class InferenceAssembler(Assembler[Entity, RuleStructure]):
    """
    Realise the IF/THEN block from a rule structure.

    >>> from krrood.entity_query_language.factories import inference
    >>> connection = variable(FixedConnection, [])
    >>> drawer = inference(Drawer)(container=connection.parent, handle=connection.child)
    >>> verbalize_expression(entity(drawer).where(connection.parent == variable(Container, [])))
    "If there's a FixedConnection whose parent is a Container, then there's a Drawer whose container is the parent of the FixedConnection, and whose handle is the child of the FixedConnection"

    Reference: :cite:t:`gatt2009simplenlg` — surface realisation.
    """

    planner = InferencePlanner

    def realize(self, node: Entity, plan: RuleStructure) -> VerbalizationFragment:
        """
        :param node: The inference-rule query.
        :param plan: The IF/THEN rule structure.
        :return: *"If <antecedents…>, then <consequent…>"* — the two-block IF/THEN form.

        Its contribution is the two-block skeleton: it pairs the *"If"* header over
        :meth:`_if_items` with the *"then"* header over :meth:`_then_items`, so the whole class
        example is exactly one *"If …, then …"* construction.
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
    def _number(antecedent: AntecedentInformation) -> GrammaticalNumber:
        """:return: The grammatical number of an antecedent — plural if and only if aggregated.

        >>> aggregated = AntecedentInformation(root=None, variable=None, type_name='Drawer',
        ...     aggregation_status=AggregationStatus.AGGREGATED)
        >>> InferenceAssembler._number(aggregated).name
        'PLURAL'
        """
        return GrammaticalNumber.of(
            antecedent.aggregation_status == AggregationStatus.AGGREGATED
        )

    # %% IF clause

    def _if_items(self, structure: RuleStructure) -> List[VerbalizationFragment]:
        """
        :return: One item per antecedent — *"there's a <Type> whose …, and …"* — plus any unmatched
            conditions; *"true"* when there are none.

        Its contribution is the body of the IF clause: it produces every *"there's a FixedConnection
        whose parent is a Container"* item that the *"If"* header then sits above, which is why the
        shown sentence opens with exactly those antecedents.

        >>> from krrood.entity_query_language.factories import inference
        >>> connection = variable(FixedConnection, [])
        >>> drawer = inference(Drawer)(container=connection.parent, handle=connection.child)
        >>> verbalize_expression(entity(drawer).where(
        ...     connection.parent == variable(Container, []))).startswith(
        ...     "If there's a FixedConnection whose parent is a Container")
        True
        """
        items: List[VerbalizationFragment] = [
            self._antecedent(antecedent) for antecedent in structure.primary_antecedents
        ]
        items += [
            self.context.child(condition)
            for condition in structure.unmatched_conditions
        ]
        return items or [Keywords.TRUE.as_fragment()]

    def _antecedent(self, antecedent: AntecedentInformation) -> VerbalizationFragment:
        """:return: The antecedent's existential intro as a head line with its conditions beneath it —
        the intro woven with its conditions by the shared restriction machinery (the same per-clause
        *"whose"* / *"such that …"* form a query selection uses). Inline / in paragraph this reads
        *"there's a <Type> whose a, and whose b"*; in hierarchical each condition is its own sub-point.

        >>> from krrood.entity_query_language.factories import inference
        >>> connection = variable(FixedConnection, [])
        >>> drawer = inference(Drawer)(container=connection.parent, handle=connection.child)
        >>> "there's a FixedConnection whose parent is a Container" in verbalize_expression(
        ...     entity(drawer).where(connection.parent == variable(Container, [])))
        True
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
        items: List[VerbalizationFragment] = []
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
        return BlockFragment(header=header, items=items, bulleted_header=False)

    def _antecedent_intro(
        self, antecedent: AntecedentInformation
    ) -> VerbalizationFragment:
        """:return: *"there's a <Type>"* / *"there are <Types>"* — the antecedent's existential intro.

        Its contribution is only the leading *"there's a FixedConnection"* noun phrase of the shown
        antecedent; the *"whose parent is a Container"* restriction that follows is woven on by
        :meth:`_antecedent`, not here.

        >>> from krrood.entity_query_language.factories import inference
        >>> connection = variable(FixedConnection, [])
        >>> drawer = inference(Drawer)(container=connection.parent, handle=connection.child)
        >>> verbalize_expression(entity(drawer).where(
        ...     connection.parent == variable(Container, []))).startswith("If there's a FixedConnection")
        True
        """
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

        The shared referent lets the THEN clause refer back to the antecedent (*"the
        FixedConnection"*):

        >>> from krrood.entity_query_language.factories import inference
        >>> connection = variable(FixedConnection, [])
        >>> drawer = inference(Drawer)(container=connection.parent, handle=connection.child)
        >>> verbalize_expression(entity(drawer).where(
        ...     connection.parent == variable(Container, []))).count("FixedConnection")
        3
        """
        root = antecedent.root
        if isinstance(root, Entity):
            root.build()
            selected_variable = root.selected_variable
            return selected_variable._id_ if selected_variable is not None else None
        return root._id_

    # %% THEN clause

    def _then_items(self, structure: RuleStructure) -> List[VerbalizationFragment]:
        """:return: The consequent as a single entry — *"there's a <Consequent>"* head line with each
        field binding prefixed by its own *"whose"* (the same form a query subject restriction uses):
        *"whose <field> is <value>, and whose …"* inline / in paragraph, sub-points in hierarchical.

        Its contribution is the entire span beneath the *"then"* header: the *"there's a Drawer"*
        intro plus the per-field *"whose container is …, and whose handle is …"* bindings, which is
        why every consequent field reads with its own repeated *"whose"* in the result.

        >>> from krrood.entity_query_language.factories import inference
        >>> connection = variable(FixedConnection, [])
        >>> drawer = inference(Drawer)(container=connection.parent, handle=connection.child)
        >>> "then there's a Drawer whose container is the parent of the FixedConnection" in (
        ...     verbalize_expression(entity(drawer).where(connection.parent == variable(Container, []))))
        True
        """
        intro: VerbalizationFragment = ExistentialPhrase.for_number(
            GrammaticalNumber.SINGULAR
        ).build_phrase(structure.consequent_type)
        whose_clauses = [
            PhraseFragment(
                parts=[Keywords.WHOSE.as_fragment(), self._binding_predicate(binding)]
            )
            for binding in structure.consequent_bindings
        ]
        if not whose_clauses:
            return [intro]
        return [
            BlockFragment(
                header=intro,
                items=whose_clauses,
                conjunction=Conjunctions.AND.as_fragment(),
                bulleted_header=False,
            )
        ]

    def _binding_predicate(self, binding: ConsequentBinding) -> VerbalizationFragment:
        """:return: The bare *"<field> is/are <value>"* predicate for one consequent binding (the
        shared *"whose"* envelope is added once by :meth:`_then_items`).

        >>> from krrood.entity_query_language.factories import inference
        >>> connection = variable(FixedConnection, [])
        >>> drawer = inference(Drawer)(container=connection.parent, handle=connection.child)
        >>> "handle is the child of the FixedConnection" in verbalize_expression(
        ...     entity(drawer).where(connection.parent == variable(Container, [])))
        True
        """
        number = GrammaticalNumber.of(binding.is_plural_field)
        return ConditionAssembler(self.context).attribute_predicate(
            binding.field_name, number, self._binding_value(binding)
        )

    def _binding_value(self, binding: ConsequentBinding) -> VerbalizationFragment:
        """
        :return: The binding's value: *"the <plural chain>"* (aggregated), bare plural, the
            group-key *"common …"* phrase, or the plain rendering.

        Its contribution is choosing which of those forms the value takes: here the grouped query
        makes the plural ``drawers`` binding aggregated, so it picks the *"the Drawers"* form seen
        after *"drawers are"* rather than a bare plural or the plain rendering.

        >>> from krrood.entity_query_language.factories import inference
        >>> container = variable(Container, [])
        >>> cabinet = inference(Cabinet)(container=container, drawers=variable(Drawer, []))
        >>> verbalize_expression(entity(cabinet).grouped_by(container))
        "If true, then there's a Cabinet whose container is a Container, and whose drawers are the Drawers"
        """
        if (
            binding.is_plural_field
            and binding.aggregation_status == AggregationStatus.AGGREGATED
        ):
            return PhraseFragment(
                parts=[
                    Articles.THE.as_fragment(),
                    self.context.child(
                        binding.value_expression, number=GrammaticalNumber.PLURAL
                    ),
                ]
            )
        if binding.is_plural_field:
            return self.context.child(
                binding.value_expression, number=GrammaticalNumber.PLURAL
            )
        if binding.aggregation_status == AggregationStatus.GROUP_KEY:
            return self._group_key_value(binding.value_expression)
        return self.context.child(binding.value_expression)

    def _group_key_value(self, expression: SymbolicExpression) -> VerbalizationFragment:
        """:return: *"the common <field> of the <Roots>"* — a binding that refers to a GROUP BY key.

        Its contribution here is the guard, not the *"common …"* phrase: the group key is a bare
        ``Container`` variable rather than an attribute chain, so the method falls back to the plain
        rendering, which is why the binding reads *"container is a Container"* and not *"the common
        … of …"*.

        >>> from krrood.entity_query_language.factories import inference
        >>> container = variable(Container, [])
        >>> cabinet = inference(Cabinet)(container=container, drawers=variable(Drawer, []))
        >>> "container is a Container" in verbalize_expression(entity(cabinet).grouped_by(container))
        True
        """
        chain, current = walk_chain(expression)
        if not chain or not isinstance(current, Variable):
            return self.context.child(expression)
        root_type = FallbackNouns.ENTITY.name_of(current)
        parts = build_path_parts(chain)
        field = list(reversed(parts))[0].name if parts else root_type
        return GroupKeyPhrases.COMMON_OF.build_phrase(field, root_type)
