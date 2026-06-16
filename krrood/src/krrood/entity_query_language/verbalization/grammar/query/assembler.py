from __future__ import annotations

import uuid

from typing_extensions import List, Optional, Tuple

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.query.query import Query, SetOf
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    NounPhrase,
    PhraseFragment,
    RoleFragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Separator,
)
from krrood.entity_query_language.verbalization.grammar.aggregation.assembler import (
    AggregationValueAssembler,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import (
    Assembler,
)
from krrood.entity_query_language.verbalization.grammar.clauses.composer import (
    ClauseComposer,
)
from krrood.entity_query_language.verbalization.grammar.query.planner import (
    QueryPlan,
    QueryPlanner,
    SelectionKind,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    FallbackNouns,
    Keywords,
    Punctuation,
)


def _subject_id(variable: SymbolicExpression) -> Optional[uuid.UUID]:
    """:return: The referent id for a subject variable (``None`` when *variable* is not a single
    variable, which suppresses pronominalisation)."""
    return variable._id_ if isinstance(variable, Variable) else None


class QueryAssembler(Assembler[Query, QueryPlan]):
    """
    Realise a query / nested-entity / set-of from its query plan into the *"Find … such that …
    grouped by … having … ordered by …"* block (and the nested noun-phrase form).

    It dispatches on the selection shape, builds the selection and its restrictions, and combines
    the trailing clauses. Coreference is resolved later: the assembler emits referring noun
    phrases and subject-scope markers, and a document-order pass decides first/subsequent/pronoun
    afterwards (Reiter & Dale 2000).

    Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(an(entity(robot).where(robot.battery > 50)))
    'Find a Robot whose battery is greater than 50'
    """

    planner = QueryPlanner

    # ── entry points ─────────────────────────────────────────────────────────

    def realize(self, node: Query, plan: QueryPlan) -> Fragment:
        """
        :param node: The query being rendered.
        :param plan: The query plan.
        :return: The top-level imperative form *"Find X such that …"*, dispatched on the
            selection shape.
        """
        handlers = {
            SelectionKind.ENTITY_SELECTOR: self._realize_entity_selector,
            SelectionKind.EMPTY: self._realize_empty,
            SelectionKind.SUBJECT: self._assemble_subject,
        }
        return handlers[plan.kind](node, plan)

    def _realize_entity_selector(self, node: Query, plan: QueryPlan) -> Fragment:
        """:return: *"Find <a Robot where …> such that …"* — the selected variable is itself an entity."""
        selection = self._as_noun(node.selected_variable)
        return self._query_body(
            node, plan, selection, where_items=[self._where_clause(plan)]
        )

    def _realize_empty(self, node: Query, plan: QueryPlan) -> Fragment:
        """:return: *"Find entities such that …"* — no selected variable (the fallback form)."""
        return self._query_body(
            node,
            plan,
            FallbackNouns.ENTITY.plural_fragment(),
            where_items=[self._where_clause(plan)],
        )

    def assemble_nested(self, node: Query) -> Fragment:
        """
        :param node: The nested entity.
        :return: The noun-phrase form for a nested entity (never emits *"Find …"*).
        """
        plan = self.plan(node)
        if plan.is_aggregation_subquery:
            return AggregationValueAssembler(self.context).realize(node, plan)
        return self._as_noun(node)

    def assemble_set_of(self, node: SetOf) -> Fragment:
        """
        :param node: The set-of query.
        :return: *"Find sets of (v1, v2, …) such that …"* for a set-of query.
        """
        plan = self.plan(node)
        variable_fragments = [
            self.context.child(variable) for variable in node._selected_variables_
        ]
        variables_phrase = PhraseFragment(
            parts=variable_fragments, separator=Separator.COMMA
        )
        # The parens glue to their content via the orthography pass — no separator="".
        selection = PhraseFragment(
            parts=[
                Punctuation.OPEN_PAREN.as_fragment(),
                variables_phrase,
                Punctuation.CLOSE_PAREN.as_fragment(),
            ]
        )
        return self._query_body(
            node,
            plan,
            selection,
            where_items=[self._where_clause(plan)],
            find_header=Keywords.FIND_SETS_OF.as_fragment(),
        )

    # ── subject selection ──────────────────────────────────────────────────────

    def _assemble_subject(self, node: Query, plan: QueryPlan) -> Fragment:
        """:return: *"Find a Robot whose battery is high, such that … [clauses]"* — the
        plain-variable selection with its WHERE woven in."""
        variable = node.selected_variable
        selected = self._build_selection(node, variable, plan)
        selected, where_items = self._apply_subject_restrictions(plan, selected)
        # No scope marker: the engine stamps this body with its query node, and the coreference
        # pass reads the focus for that query from the discourse view.
        return self._query_body(node, plan, selected, where_items=where_items)

    def _build_selection(
        self, node: Query, variable: SymbolicExpression, plan: QueryPlan
    ) -> Fragment:
        """:return: *"the unique Robot"* (``eql.the``) or *"a Robot"* — the selection's referring
        noun phrase."""
        if plan.is_the:
            # "the unique <type>" first mention; the coreference pass reduces a repeat to
            # "the <type>" (UNIQUE downgrades to DEFINITE) — so it is a referring noun phrase.
            return NounPhrase(
                head=RoleFragment.for_variable(plan.selected_type, variable),
                definiteness=Definiteness.UNIQUE,
                referent_id=_subject_id(variable),
            )
        # context.child(variable) → VariableRule referring noun phrase; the entity shares its referent.
        return self.context.child(variable)

    def _apply_subject_restrictions(
        self, plan: QueryPlan, selected: Fragment
    ) -> Tuple[Fragment, List[Optional[Fragment]]]:
        """:return: The selection with its inline superlative modifiers attached, and the WHERE's
        clause items — the *"whose"* group (a sub-list of points in hierarchical) then a separate
        *"such that <residual>"* clause (each ``None`` when absent)."""
        rendered = ClauseComposer(self.context).restriction(plan)
        if rendered is None:
            return selected, []
        if rendered.inline_modifiers:
            selected = PhraseFragment(parts=[selected, *rendered.inline_modifiers])
        residual = (
            PhraseFragment(parts=[Keywords.SUCH_THAT.as_fragment(), rendered.residual])
            if rendered.residual is not None
            else None
        )
        return selected, [rendered.whose, residual]

    # ── noun forms ───────────────────────────────────────────────────────────

    def _as_noun(self, entity: Query) -> Fragment:
        """
        A referring noun phrase — *"a/the unique <type>"* with the restrictions as appositive
        modifiers.

        :param entity: The nested entity selector.
        :return: The standalone-noun form *"a Robot where …"*.
        """
        plan = self.plan(entity)
        variable = entity.selected_variable
        definiteness = Definiteness.UNIQUE if plan.is_the else Definiteness.INDEFINITE

        modifiers: List[Fragment] = []
        rendered = ClauseComposer(self.context).restriction(plan)
        if rendered is not None:
            modifiers.extend(rendered.inline_modifiers)
            # A nested noun is an inline phrase, so the "whose" block flattens to "whose a, and b"
            # (the renderer expands a block into points only at the item level, not inside a phrase).
            if rendered.whose is not None:
                modifiers.append(rendered.whose)
            if rendered.residual is not None:
                modifiers.append(
                    PhraseFragment(
                        parts=[Keywords.WHERE.as_fragment(), rendered.residual]
                    )
                )

        # A referring noun phrase is the subject of its own modifiers (the coreference pass infers
        # this from the modifiers slot), so no scope marker is emitted here.
        return NounPhrase(
            head=RoleFragment.for_variable(plan.selected_type, variable),
            definiteness=definiteness,
            referent_id=_subject_id(variable),
            modifiers=modifiers,
        )

    # ── query-body clauses ─────────────────────────────────────────────────────

    def _query_body(
        self,
        node: Query,
        plan: QueryPlan,
        selection: Fragment,
        where_items: List[Optional[Fragment]],
        find_header: Optional[Fragment] = None,
    ) -> Fragment:
        """:return: *"Find <selection>"* + the present clauses (the subject restriction's *"whose"*
        / *"such that"*, then *grouped by … having … ordered by …*) as block items — absent
        clauses (``None``) are simply skipped.
        """
        if find_header is None:
            find_header = Keywords.FIND.as_fragment()
        header = PhraseFragment(parts=[find_header, selection])
        clauses = [
            clause
            for clause in [*where_items, *self._trailing_clauses(node)]
            if clause is not None
        ]
        return BlockFragment(header=header, items=clauses)

    def _trailing_clauses(self, node: Query) -> List[Optional[Fragment]]:
        """:return: The post-selection clauses, in canonical reading order (``None`` when absent)."""
        composer = ClauseComposer(self.context)
        return [
            composer.grouped_by(node),
            composer.having(node),
            composer.ordered_by(node),
        ]

    def _where_clause(self, plan: QueryPlan) -> Optional[Fragment]:
        """:return: *"such that <condition>"*, or ``None`` when the query has no WHERE."""
        if plan.where_condition is None:
            return None
        return PhraseFragment(
            parts=[
                Keywords.SUCH_THAT.as_fragment(),
                self.context.child(plan.where_condition),
            ]
        )

    def inline_noun(self, entity: Query) -> Fragment:
        """
        The entity's WHERE condition is deferred to the binding scope so it can be emitted as a
        *"such that …"* clause after all binding overrides are registered.

        :param entity: The entity used as a chain root inside an instantiated variable.
        :return: The inline-noun form for *entity*.
        """
        entity.build()
        variable = entity.selected_variable
        variable_type = getattr(variable, "_type_", None)
        type_name = (
            variable_type.__name__ if variable_type else FallbackNouns.ENTITY.text
        )

        where_expression = entity._where_expression_
        if where_expression is not None:
            self.context.scope.defer_constraint(where_expression.condition)

        # A referring noun phrase (referent_id below) — a repeat reduces to "the <type>" in the pass.
        return NounPhrase(
            head=RoleFragment.for_variable(type_name, variable),
            referent_id=_subject_id(variable),
        )
