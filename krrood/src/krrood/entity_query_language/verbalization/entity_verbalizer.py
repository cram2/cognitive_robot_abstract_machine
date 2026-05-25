"""
EntityVerbalizer — Entity, SetOf, and query-body clause rendering.

Handles the full query form ("Find X such that …"), the inline-noun form
(constraint-deferring), and the standalone-noun form used when an Entity is
the selected variable of an outer query.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from krrood.entity_query_language.core.variable import InstantiatedVariable, Variable
from krrood.entity_query_language.query.quantifiers import An, ResultQuantifier, The
from krrood.entity_query_language.query.query import Entity, Query, SetOf
from krrood.entity_query_language.verbalization.chain_utils import chain_root, verbalize_plural
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    oxford_and,
    PhraseFragment,
    RoleFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.utils import _str
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Conjunctions,
    Copulas,
    FallbackNouns,
    Keywords,
    Prepositions,
    SortDirections,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext


def _word(text: str) -> WordFragment:
    return WordFragment(text=text)


def _role(text, role, ref=None):
    return RoleFragment(text=text, role=role, source_ref=ref)


def _phrase(*parts, sep=" "):
    return PhraseFragment(parts=list(parts), separator=sep)


class EntityVerbalizer:
    """
    Verbalizes :class:`~krrood.entity_query_language.query.query.Entity` and
    :class:`~krrood.entity_query_language.query.query.SetOf` query expressions.

    Provides three entry forms for Entity rendering:

    * :meth:`verbalize_query` — top-level *"Find X such that …"* form.
    * :meth:`as_noun` — standalone noun phrase for nested Entity selectors.
    * :meth:`as_inline_noun` — constraint-deferring noun used as a chain root
      inside an :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`.

    :param delegate: The parent verbalizer used for recursive sub-expression calls.
    :type delegate: ~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer
    """

    def __init__(self, delegate) -> None:
        self._d = delegate

    # ── Public entry points ────────────────────────────────────────────────────

    def verbalize_query(self, expr: "Entity", ctx: "VerbalizationContext") -> VerbFragment:
        """
        Full query form: *"Find X such that …"*.

        Delegates to :class:`~krrood.entity_query_language.verbalization.rule_verbalizer.RuleVerbalizer`
        when the query encodes an inference rule (selected variable is
        :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`).
        Otherwise assembles ``FIND + SUCH THAT + GROUPED BY + HAVING + ORDERED BY`` clauses.

        :param expr: An :class:`~krrood.entity_query_language.query.query.Entity` expression.
        :type expr: ~krrood.entity_query_language.query.query.Entity
        :param ctx: Shared verbalization state.
        :type ctx: ~krrood.entity_query_language.verbalization.context.VerbalizationContext
        :returns: Fragment tree for the full query.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        if expr._id_ in ctx.seen:
            return _phrase(Articles.THE.as_fragment(), _role(ctx.seen[expr._id_], SemanticRole.VARIABLE))

        expr.build()

        if self._d._rule.can_handle(expr):
            return self._d._rule.verbalize(expr, ctx)

        ctx.query_depth += 1
        try:
            is_the = (
                expr._quantifier_builder_ is not None
                and expr._quantifier_builder_.type is The
            )
            var = expr.selected_variable

            if isinstance(var, Entity):
                selected = self.as_noun(var, ctx)
            elif var is None:
                selected_type = FallbackNouns.ENTITY.text
                ctx.seen[expr._id_] = selected_type
                selected = FallbackNouns.ENTITY.plural_fragment()
            elif is_the:
                selected_type = var._type_.__name__ if getattr(var, "_type_", None) else FallbackNouns.ENTITY.text
                ctx.seen[var._id_] = selected_type
                ctx.seen[expr._id_] = selected_type
                selected = _phrase(Articles.THE_UNIQUE.as_fragment(), _role(selected_type, SemanticRole.VARIABLE))
            else:
                selected = self._d.build(var, ctx)
                selected_type = ctx.seen.get(getattr(var, "_id_", None), FallbackNouns.ENTITY.text)
                ctx.seen[expr._id_] = selected_type

            return self._verbalize_query_body_(expr, ctx, selected)
        finally:
            ctx.query_depth -= 1

    def verbalize_nested(self, expr: "Entity", ctx: "VerbalizationContext") -> VerbFragment:
        """
        Noun-phrase form for an :class:`~krrood.entity_query_language.query.query.Entity`
        used as a *value* (e.g. an operand of a comparator) rather than as the
        top-level request.  Never emits the imperative *"Find …"* prefix.

        Dispatch:

        * **Unconstrained aggregation subquery** (``entity(max(t.amount))``) — a
          noun naming a computed value.  Rendered via the aggregator itself
          (Layer B collapses this to *"the maximum amount"*).
        * **Constrained aggregation subquery** (aggregation plus a ``WHERE`` /
          ``HAVING`` / ``GROUP BY``) — the filter is load-bearing, so the full
          query form is retained to preserve it.
        * **Any other nested entity** — delegated to :meth:`as_noun`
          (*"a Robot where …"*).

        :param expr: A nested :class:`~krrood.entity_query_language.query.query.Entity`.
        :param ctx: Shared verbalization state.
        :returns: A noun-phrase fragment for *expr*.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        from krrood.entity_query_language.verbalization.subquery import is_aggregation_subquery

        if expr._id_ in ctx.seen:
            return _phrase(Articles.THE.as_fragment(), _role(ctx.seen[expr._id_], SemanticRole.VARIABLE))

        expr.build()

        if is_aggregation_subquery(expr):
            return self._verbalize_aggregation_value_(expr, ctx)

        return self.as_noun(expr, ctx)

    def _verbalize_aggregation_value_(self, expr: "Entity", ctx: "VerbalizationContext") -> VerbFragment:
        """
        Render an aggregation sub-query as a compact aggregate noun phrase.

        * **Unconstrained** → *"the <aggregation> <leaf>"* (e.g. *"the maximum amount"*).
          The aggregation source variable and intermediate path are dropped — in
          the role of a value, that provenance is noise.
        * **Constrained** → *"the <aggregation> <leaf> among <plural source> such
          that <filter>"* — the filter is load-bearing, so it is preserved (still
          no imperative *"Find"*).
        * **No attribute leaf** (aggregating a bare variable) → falls back to the
          aggregator's own verbose rendering.

        The singular/plural form of the leaf follows the aggregation word's
        :class:`~krrood.entity_query_language.verbalization.vocabulary.words.ChildForm`
        (``maximum`` → *"amount"*; ``sum of`` → *"amounts"*).

        :param expr: An aggregation sub-query Entity.
        :param ctx: Shared verbalization state.
        :returns: Aggregate noun-phrase fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        from krrood.entity_query_language.verbalization.rules.aggregators import _AGGREGATION_KIND
        from krrood.entity_query_language.verbalization.subquery import (
            aggregation_leaf_attribute,
            is_constrained_query,
            selected_aggregator,
        )
        from krrood.entity_query_language.verbalization.vocabulary.words import ChildForm

        aggregator = selected_aggregator(expr)
        leaf = aggregation_leaf_attribute(expr)
        if leaf is None:
            ctx.query_depth += 1
            try:
                return self._d.build(aggregator, ctx)
            finally:
                ctx.query_depth -= 1

        agg_kind = _AGGREGATION_KIND[type(aggregator)]
        plural_leaf = agg_kind.value.child_form == ChildForm.PLURAL
        leaf_frag = RoleFragment.for_attribute(leaf._owner_class_, leaf._attribute_name_, plural=plural_leaf)
        aggregate = _phrase(Articles.THE.as_fragment(), agg_kind.as_fragment(), leaf_frag)

        if aggregator._id_ not in ctx.seen:
            ctx.seen[aggregator._id_] = _str(_phrase(agg_kind.as_fragment(), leaf_frag))

        if not is_constrained_query(expr):
            return aggregate
        return self._aggregation_scope_(expr, aggregate, ctx)

    def _aggregation_scope_(
        self, expr: "Entity", aggregate: VerbFragment, ctx: "VerbalizationContext"
    ) -> VerbFragment:
        """
        Append *"among <plural source> such that <filter>"* to a constrained aggregate.

        :param expr: A constrained aggregation sub-query Entity.
        :param aggregate: The already-built *"the <aggregation> <leaf>"* fragment.
        :param ctx: Shared verbalization state.
        :returns: Aggregate phrase extended with its scope and filter.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        from krrood.entity_query_language.verbalization.subquery import aggregation_source_root

        source = aggregation_source_root(expr)
        source_frag = (
            verbalize_plural(source, ctx, self._d.build)
            if source is not None
            else FallbackNouns.ENTITY.plural_fragment()
        )
        parts = [aggregate, Prepositions.AMONG.as_fragment(), source_frag]

        where_expr = expr._where_expression_
        if where_expr is not None:
            ctx.query_depth += 1
            try:
                condition = self._d.build(where_expr.condition, ctx)
            finally:
                ctx.query_depth -= 1
            parts += [Keywords.SUCH_THAT.as_fragment(), condition]

        having_expr = expr._having_expression_
        if having_expr is not None:
            ctx.compact_predicates = True
            ctx.query_depth += 1
            try:
                having_frag = self._d.build(having_expr.condition, ctx)
            finally:
                ctx.query_depth -= 1
                ctx.compact_predicates = False
            parts += [Keywords.HAVING.as_fragment(), having_frag]

        return _phrase(*parts)

    def as_noun(self, expr: "Entity", ctx: "VerbalizationContext") -> VerbFragment:
        """
        Standalone-noun form: *"a Robot where …"* (for nested Entity selectors).

        Used when an :class:`~krrood.entity_query_language.query.query.Entity` is
        itself the selected variable of an outer query.  All WHERE conditions are
        verbalized inline rather than deferred.

        :param expr: An :class:`~krrood.entity_query_language.query.query.Entity`.
        :type expr: ~krrood.entity_query_language.query.query.Entity
        :param ctx: Shared verbalization state.
        :type ctx: ~krrood.entity_query_language.verbalization.context.VerbalizationContext
        :returns: A noun-phrase fragment for *expr*.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        if expr._id_ in ctx.seen:
            return _phrase(Articles.THE.as_fragment(), _role(ctx.seen[expr._id_], SemanticRole.VARIABLE))

        expr.build()
        is_the = (
            expr._quantifier_builder_ is not None
            and expr._quantifier_builder_.type is The
        )
        var = expr.selected_variable
        selected_type = var._type_.__name__ if var and getattr(var, "_type_", None) else FallbackNouns.ENTITY.text
        ctx.seen[expr._id_] = selected_type
        if var is not None:
            ctx.seen[var._id_] = selected_type

        if is_the:
            article_noun: VerbFragment = _phrase(
                Articles.THE_UNIQUE.as_fragment(), RoleFragment.for_variable(selected_type, var)
            )
        else:
            article_noun = _phrase(
                Articles.indefinite(selected_type),
                RoleFragment.for_variable(selected_type, var),
            )

        where_expr = expr._where_expression_
        if where_expr is not None:
            ctx.query_depth += 1
            try:
                condition = self._d.build(where_expr.condition, ctx)
            finally:
                ctx.query_depth -= 1
            return _phrase(article_noun, Keywords.WHERE.as_fragment(), condition)
        return article_noun

    def as_inline_noun(self, entity: "Entity", ctx: "VerbalizationContext") -> VerbFragment:
        """
        Inline-noun form used as a chain root inside an
        :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`.

        Defers the entity's WHERE condition to
        :attr:`~krrood.entity_query_language.verbalization.context.VerbalizationContext.constraint_exprs`
        so the enclosing :class:`~krrood.entity_query_language.verbalization.rules.variables.InstantiatedVariableRule`
        can emit it as a *"such that …"* clause after all binding overrides are registered.

        :param entity: An :class:`~krrood.entity_query_language.query.query.Entity` used as a chain root.
        :type entity: ~krrood.entity_query_language.query.query.Entity
        :param ctx: Shared verbalization state.
        :type ctx: ~krrood.entity_query_language.verbalization.context.VerbalizationContext
        :returns: A bare noun-phrase fragment (article + type name).
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        if entity._id_ in ctx.seen:
            return _phrase(Articles.THE.as_fragment(), _role(ctx.seen[entity._id_], SemanticRole.VARIABLE))

        entity.build()
        var = entity.selected_variable
        var_type = getattr(var, "_type_", None)
        type_name = var_type.__name__ if var_type else FallbackNouns.ENTITY.text

        ctx.seen[entity._id_] = type_name
        ctx.seen[var._id_] = type_name

        where_expr = entity._where_expression_
        if where_expr is not None:
            ctx.defer_constraint(where_expr.condition)

        return _phrase(Articles.indefinite(type_name), RoleFragment.for_variable(type_name, var))

    def verbalize_set_of(self, expr: "SetOf", ctx: "VerbalizationContext") -> VerbFragment:
        """
        Verbalize a :class:`~krrood.entity_query_language.query.query.SetOf` query.

        Produces *"Find (v1, v2, …) such that …"* using the same body-clause assembly
        as :meth:`verbalize_query`.

        :param expr: A :class:`~krrood.entity_query_language.query.query.SetOf` expression.
        :type expr: ~krrood.entity_query_language.query.query.SetOf
        :param ctx: Shared verbalization state.
        :type ctx: ~krrood.entity_query_language.verbalization.context.VerbalizationContext
        :returns: Fragment tree for the SetOf query.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        expr.build()
        ctx.query_depth += 1
        try:
            var_frags = [self._d.build(v, ctx) for v in expr._selected_variables_]
            vars_phrase = PhraseFragment(parts=var_frags, separator=", ")
            prefix = _phrase(
                Keywords.FIND_SETS_OF.as_fragment(),
                PhraseFragment(parts=[_word("("), vars_phrase, _word(")")], separator=""),
            )
            return self._verbalize_query_body_(expr, ctx, prefix)
        finally:
            ctx.query_depth -= 1

    # ── Query body assembly ────────────────────────────────────────────────────

    def _verbalize_query_body_(
        self, expr, ctx: "VerbalizationContext", selection: VerbFragment
    ) -> VerbFragment:
        """
        Assemble the full *"Find <selection> such that … grouped by … having … ordered by …"* block.

        :param expr: Entity or SetOf expression supplying the clause expressions.
        :param ctx: Shared verbalization state.
        :param selection: Pre-built fragment for the selected variable(s).
        :returns: A :class:`~krrood.entity_query_language.verbalization.fragments.base.BlockFragment`
            with the selection as header and clause items.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        header = _phrase(Keywords.FIND.as_fragment(), selection)
        clauses = [c for c in [
            self._where_clause(expr, ctx),
            self._grouped_by_clause(expr, ctx),
            self._having_clause(expr, ctx),
            self._ordered_by_clause(expr, ctx),
        ] if c is not None]
        return BlockFragment(header=header, items=clauses)

    def _where_clause(self, expr, ctx: "VerbalizationContext") -> Optional[VerbFragment]:
        """
        Build the *"such that <condition>"* fragment, or ``None`` when no WHERE expression exists.

        :param expr: Entity or SetOf expression.
        :param ctx: Shared verbalization state.
        :returns: Phrase fragment or ``None``.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment or None
        """
        where_expr = expr._where_expression_
        if where_expr is None:
            return None
        return _phrase(Keywords.SUCH_THAT.as_fragment(), self._d.build(where_expr.condition, ctx))

    def _grouped_by_clause(self, expr, ctx: "VerbalizationContext") -> Optional[VerbFragment]:
        """
        Build the *"and the <aggregated> are grouped by <keys>"* fragment, or ``None``.

        :param expr: Entity or SetOf expression.
        :param ctx: Shared verbalization state.
        :returns: Phrase fragment or ``None``.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment or None
        """
        grouped_expr = expr._grouped_by_expression_
        if grouped_expr is None or not grouped_expr.variables_to_group_by:
            return None
        group_key_root_ids = self._root_var_ids_(grouped_expr.variables_to_group_by)
        group_frags = [self._d.build(v, ctx) for v in grouped_expr.variables_to_group_by]
        groups_phrase = PhraseFragment(parts=group_frags, separator=", ")
        aggregated_frags = self._aggregated_noun_frags_(expr, group_key_root_ids, ctx)
        if aggregated_frags:
            aggregated_phrase = oxford_and(aggregated_frags, Conjunctions.AND.as_fragment())
            return _phrase(
                Conjunctions.AND.as_fragment(),
                Articles.THE.as_fragment(),
                aggregated_phrase,
                Copulas.ARE.as_fragment(),
                Keywords.GROUPED_BY.as_fragment(),
                groups_phrase,
            )
        return _phrase(Keywords.GROUPED_BY.as_fragment(), groups_phrase)

    def _having_clause(self, expr, ctx: "VerbalizationContext") -> Optional[VerbFragment]:
        """
        Build the *"having <condition>"* fragment with compact comparators, or ``None``.

        Sets :attr:`~krrood.entity_query_language.verbalization.context.VerbalizationContext.compact_predicates`
        to ``True`` while verbalizing the HAVING condition so copulas are omitted.

        :param expr: Entity or SetOf expression.
        :param ctx: Shared verbalization state.
        :returns: Phrase fragment or ``None``.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment or None
        """
        having_expr = expr._having_expression_
        if having_expr is None:
            return None
        ctx.compact_predicates = True
        having_frag = self._d.build(having_expr.condition, ctx)
        ctx.compact_predicates = False
        return _phrase(Keywords.HAVING.as_fragment(), having_frag)

    def _ordered_by_clause(self, expr, ctx: "VerbalizationContext") -> Optional[VerbFragment]:
        """
        Build the *"ordered by <variable> (ascending|descending)"* fragment, or ``None``.

        :param expr: Entity or SetOf expression.
        :param ctx: Shared verbalization state.
        :returns: Phrase fragment or ``None``.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment or None
        """
        ob = expr._ordered_by_builder_
        if ob is None:
            return None
        direction_frag = (
            SortDirections.DESCENDING.as_fragment()
            if ob.descending
            else SortDirections.ASCENDING.as_fragment()
        )
        ordered_frag = self._d.build(ob.variable, ctx)
        paren_frag = PhraseFragment(parts=[_word("("), direction_frag, _word(")")], separator="")
        return _phrase(Keywords.ORDERED_BY.as_fragment(), ordered_frag, paren_frag)

    # ── Grouping helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _root_var_ids_(exprs) -> set:
        ids: set = set()
        for e in exprs:
            root = chain_root(e)
            if isinstance(root, Variable):
                ids.add(root._id_)
        return ids

    @staticmethod
    def _aggregated_expressions_(query_expr, group_key_root_ids: set) -> list:
        selected_var = query_expr.selected_variable if isinstance(query_expr, Entity) else None
        if isinstance(selected_var, InstantiatedVariable):
            result = []
            for child in selected_var._child_vars_.values():
                root = chain_root(child)
                if not (isinstance(root, Variable) and root._id_ in group_key_root_ids):
                    result.append(child)
            return result
        if isinstance(query_expr, Query):
            return [v for v in query_expr._selected_variables_ if v._id_ not in group_key_root_ids]
        return []

    def _aggregated_noun_frags_(
        self, query_expr, group_key_root_ids: set, ctx: VerbalizationContext
    ) -> list[VerbFragment]:
        return [
            verbalize_plural(e, ctx, self._d.build)
            for e in self._aggregated_expressions_(query_expr, group_key_root_ids)
        ]
