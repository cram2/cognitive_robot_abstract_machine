"""
Query **assembler** — realise a :class:`~krrood.entity_query_language.verbalization.grammar.planning.query.QueryPlan`
into the *"Find … such that … grouped by … having … ordered by …"* block (and the nested
noun-phrase / aggregation-value forms).

This is the realisation half of the planner/assembler split: it owns recursion
(``self.ctx.child``) and the render-scope mutations (query-depth, compact predicates).
Coreference is no longer resolved here — the assembler emits referring ``NounPhrase`` s and
``SubjectScope`` markers, and the document-order ``CoreferenceProcessor`` pass decides
first/subsequent/pronoun afterwards (Reiter & Dale 2000).  All *what to say* decisions already
live in the plan; the assembler only combines.

Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
"""

from __future__ import annotations

from typing_extensions import List, Optional, Tuple

from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.query.query import Query
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    NounPhrase,
    oxford_and,
    PhraseFragment,
    RoleFragment,
    SubjectScope,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.fragments.factory import (
    phrase,
    word,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.grammar.aggregation_kinds import (
    AGGREGATION_KIND,
)
from krrood.entity_query_language.verbalization.grammar.assembly.base import Assembler
from krrood.entity_query_language.verbalization.grammar.assembly.clauses import (
    GroupedByAssembler,
    HavingAssembler,
    OrderedByAssembler,
)
from krrood.entity_query_language.verbalization.grammar.planning.query import (
    QueryPlan,
    QueryPlanner,
    RestrictionPlan,
    SelectionKind,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    build_between,
    RangeFold,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    FallbackNouns,
    Keywords,
    Prepositions,
)
from krrood.entity_query_language.verbalization.vocabulary.words import ChildForm


def _subject_id(var):
    """The referent id for a subject variable (``None`` when *var* is not a single Variable,
    e.g. a ``SetOf`` — which suppresses pronominalisation)."""
    return var._id_ if isinstance(var, Variable) else None


class QueryAssembler(Assembler[Query, QueryPlan]):
    """Realise a query / nested-entity / set-of from its :class:`QueryPlan`."""

    planner = QueryPlanner

    # ── entry points ─────────────────────────────────────────────────────────

    def realize(self, node, plan: QueryPlan) -> VerbFragment:
        """Top-level imperative form: *"Find X such that …"*."""
        with self.ctx.context.query_depth_scope():
            if plan.kind is SelectionKind.ENTITY_SELECTOR:
                selection = self._as_noun(node.selected_variable)
                return self._query_body(
                    node, plan, selection, where_item=self._where_clause(plan)
                )
            if plan.kind is SelectionKind.EMPTY:
                return self._query_body(
                    node,
                    plan,
                    FallbackNouns.ENTITY.plural_fragment(),
                    where_item=self._where_clause(plan),
                )
            return self._assemble_subject(node, plan)

    def assemble_nested(self, node) -> VerbFragment:
        """Noun-phrase form for a nested Entity (never emits *"Find …"*)."""
        plan = self.plan(node)
        if plan.is_aggregation_subquery:
            return self._aggregation_value(node, plan)
        return self._as_noun(node)

    def assemble_set_of(self, node) -> VerbFragment:
        """*"Find (v1, v2, …) such that …"* for a SetOf query."""
        plan = self.plan(node)
        with self.ctx.context.query_depth_scope():
            variable_fragments = [
                self.ctx.child(variable) for variable in node._selected_variables_
            ]
            vars_phrase = PhraseFragment(parts=variable_fragments, separator=", ")
            selection = PhraseFragment(
                parts=[word("("), vars_phrase, word(")")], separator=""
            )
            return self._query_body(
                node,
                plan,
                selection,
                where_item=self._where_clause(plan),
                find_header=Keywords.FIND_SETS_OF.as_fragment(),
            )

    # ── subject selection ──────────────────────────────────────────────────────

    def _assemble_subject(self, node, plan: QueryPlan) -> VerbFragment:
        var = node.selected_variable
        selected = self._build_selection(node, var, plan)
        selected, where_item = self._apply_subject_restrictions(plan, selected)
        body = self._query_body(node, plan, selected, where_item=where_item)
        # Mark the subject region so the coreference pass pronominalises chains rooted at it.
        return SubjectScope(subject_id=_subject_id(var), child=body)

    def _build_selection(self, node, var, plan: QueryPlan) -> VerbFragment:
        if plan.is_the:
            # "the unique <type>" first mention; the coreference pass reduces a repeat to
            # "the <type>" (UNIQUE downgrades to DEFINITE) — so it is a referring NP.
            return NounPhrase(
                head=RoleFragment.for_variable(plan.selected_type, var),
                definiteness=Definiteness.UNIQUE,
                referent_id=_subject_id(var),
            )
        # ctx.child(var) → VariableRule referring NP (referent_id = var); the entity shares it.
        return self.ctx.child(var)

    def _apply_subject_restrictions(
        self, plan: QueryPlan, selected: VerbFragment
    ) -> Tuple[VerbFragment, Optional[VerbFragment]]:
        restriction = plan.subject_restriction
        if restriction is None:
            return selected, None
        whose, residual = self._render_restrictions(restriction, plan.subject)
        if whose is not None:
            selected = phrase(selected, whose)
        where_item = (
            phrase(Keywords.SUCH_THAT.as_fragment(), residual)
            if residual is not None
            else None
        )
        return selected, where_item

    # ── noun forms ───────────────────────────────────────────────────────────

    def _as_noun(self, entity) -> VerbFragment:
        """Standalone-noun form: *"a Robot where …"* (for nested Entity selectors).

        A referring NP — *"a/the unique <type>"* first mention with the restrictions as
        appositive modifiers; a repeat is reduced to *"the <type>"* by the coreference pass.
        Wrapped in a ``SubjectScope`` so chains in the restrictions pronominalise to *"its …"*.
        """
        plan = self.plan(entity)
        var = entity.selected_variable
        definiteness = Definiteness.UNIQUE if plan.is_the else Definiteness.INDEFINITE

        modifiers: List[VerbFragment] = []
        if plan.subject_restriction is not None:
            with self.ctx.context.query_depth_scope():
                whose, residual = self._render_restrictions(
                    plan.subject_restriction, plan.subject
                )
            if whose is not None:
                modifiers.append(whose)
            if residual is not None:
                modifiers.append(phrase(Keywords.WHERE.as_fragment(), residual))

        noun = NounPhrase(
            head=RoleFragment.for_variable(plan.selected_type, var),
            definiteness=definiteness,
            referent_id=_subject_id(var),
            modifiers=modifiers,
        )
        return (
            SubjectScope(subject_id=_subject_id(var), child=noun) if modifiers else noun
        )

    # ── aggregation value-subquery ──────────────────────────────────────────────

    def _aggregation_value(self, node, plan: QueryPlan) -> VerbFragment:
        av = plan.aggregation_value
        if av.leaf is None:
            with self.ctx.context.query_depth_scope():
                return self.ctx.child(av.aggregator)

        aggregation_kind = AGGREGATION_KIND[type(av.aggregator)]
        plural_leaf = aggregation_kind.value.child_form == ChildForm.PLURAL
        leaf_frag = RoleFragment.for_attribute(
            av.leaf._owner_class_,
            av.leaf._attribute_name_,
            number=Number.of(plural_leaf),
        )
        aggregate = NounPhrase(
            head=aggregation_kind.as_fragment(),
            definiteness=Definiteness.DEFINITE,
            modifiers=[leaf_frag],
        )

        if not av.is_constrained:
            return aggregate
        return self._aggregation_scope(node, plan, aggregate)

    def _aggregation_scope(self, node, plan: QueryPlan, aggregate) -> VerbFragment:
        """Append *"among <plural source> such that <filter> having <filter>"*."""
        source = plan.aggregation_value.source
        source_frag = (
            self.ctx.child(source, number=Number.PLURAL)
            if source is not None
            else FallbackNouns.ENTITY.plural_fragment()
        )
        parts: List[VerbFragment] = [
            aggregate,
            Prepositions.AMONG.as_fragment(),
            source_frag,
        ]

        if plan.subject_restriction is not None:
            with self.ctx.context.query_depth_scope():
                whose, residual = self._render_restrictions(
                    plan.subject_restriction, plan.subject
                )
            if whose is not None:
                parts.append(whose)
            if residual is not None:
                parts += [Keywords.SUCH_THAT.as_fragment(), residual]

        with self.ctx.context.query_depth_scope():
            having = HavingAssembler(self.ctx).clause(node)
        if having is not None:
            parts.append(having)

        return phrase(*parts)

    # ── restriction rendering ──────────────────────────────────────────────────

    def _render_restrictions(
        self, restriction: RestrictionPlan, subject
    ) -> Tuple[Optional[VerbFragment], Optional[VerbFragment]]:
        """Render the *"whose <grouped>"* modifier and the residual condition."""
        grouped_frags = [
            rule.render(item, subject, self.ctx) for rule, item in restriction.grouped
        ]
        whose = None
        if grouped_frags:
            whose = PhraseFragment(
                parts=[
                    Keywords.WHOSE.as_fragment(),
                    oxford_and(grouped_frags, Conjunctions.AND.as_fragment()),
                ]
            )
        residual = (
            self._render_residual(restriction.residual)
            if restriction.has_residual
            else None
        )
        return whose, residual

    def _render_residual(self, items: List) -> VerbFragment:
        parts: List[VerbFragment] = []
        for item in items:
            if isinstance(item, RangeFold):
                parts.append(
                    build_between(
                        self.ctx.child(item.chain_expression),
                        self.ctx.child(item.lower_expression),
                        self.ctx.child(item.upper_expression),
                        compact=self.ctx.context.compact_predicates,
                    )
                )
            else:
                parts.append(self.ctx.child(item))
        if len(parts) == 1:
            return parts[0]
        return oxford_and(parts, Conjunctions.AND.as_fragment())

    # ── query-body clauses ─────────────────────────────────────────────────────

    def _query_body(
        self,
        node,
        plan: QueryPlan,
        selection: VerbFragment,
        where_item: Optional[VerbFragment],
        find_header: Optional[VerbFragment] = None,
    ) -> VerbFragment:
        if find_header is None:
            find_header = Keywords.FIND.as_fragment()
        header = phrase(find_header, selection)
        clauses = [
            clause
            for clause in [where_item, *self._trailing_clauses(node)]
            if clause is not None
        ]
        return BlockFragment(header=header, items=clauses)

    def _trailing_clauses(self, node) -> List[Optional[VerbFragment]]:
        """The post-selection clauses, in canonical reading order, each rendered by its
        own component (``None`` when absent)."""
        return [
            GroupedByAssembler(self.ctx).clause(node),
            HavingAssembler(self.ctx).clause(node),
            OrderedByAssembler(self.ctx).clause(node),
        ]

    def _where_clause(self, plan: QueryPlan) -> Optional[VerbFragment]:
        if plan.where_condition is None:
            return None
        return phrase(
            Keywords.SUCH_THAT.as_fragment(), self.ctx.child(plan.where_condition)
        )

    def inline_noun(self, entity) -> VerbFragment:
        """
        Inline-noun form used as a chain root inside an InstantiatedVariable.

        Defers the entity's WHERE condition to the binding scope so the enclosing rule
        can emit it as a *"such that …"* clause after all binding overrides are
        registered.  Used by the chain assembler for Entity-rooted chains.
        """
        entity.build()
        var = entity.selected_variable
        variable_type = getattr(var, "_type_", None)
        type_name = (
            variable_type.__name__ if variable_type else FallbackNouns.ENTITY.text
        )

        where_expression = entity._where_expression_
        if where_expression is not None:
            self.ctx.context.defer_constraint(where_expression.condition)

        # A referring NP (referent_id below) — a repeat reduces to "the <type>" in the pass.
        return NounPhrase(
            head=RoleFragment.for_variable(type_name, var),
            referent_id=_subject_id(var),
        )
