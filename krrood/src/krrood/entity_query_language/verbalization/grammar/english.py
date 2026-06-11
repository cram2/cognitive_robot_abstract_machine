"""
The English grammar — one :class:`~krrood.entity_query_language.verbalization.grammar.phrase_rule.PhraseRule`
subclass per EQL construct.

Each rule is a Montague rule-to-rule clause: *for this construct, build this phrase*
(Montague 1970; Bach 1976, the rule-to-rule hypothesis).  A rule only **combines** —
recursion is delegated to ``ctx.child`` (the fold), and cross-cutting decisions to the
microplanning services (``ctx.refer`` / ``ctx.scope`` / ``ctx.config``), morphology, the
coordination module, and the lexicon — so each rule is responsible for a single
construct's surface composition.

:data:`RULES` lists one instance per rule — plain data, so the grammar itself is
EQL-queryable (e.g. ``entity(rule).where(rule.construct == Comparator)`` over
``domain=RULES``).  ``select`` decides specificity, so the list order is irrelevant.
"""

from __future__ import annotations

import sys

from typing_extensions import List

from krrood.entity_query_language.core.mapped_variable import (
    FlatVariable,
    MappedVariable,
)
from krrood.entity_query_language.core.variable import (
    ExternallySetVariable,
    InstantiatedVariable,
    Literal,
    Variable,
)
from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    AND,
    OR,
    Not,
    flatten_operands,
)
from krrood.entity_query_language.operators.logical_quantifiers import Exists, ForAll
from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    oxford_and,
    PhraseFragment,
    RoleFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    is_bool_attr_chain,
)
from krrood.entity_query_language.verbalization.grammar.conditions.verbalizer import (
    ConditionVerbalizer,
)
from krrood.entity_query_language.verbalization.grammar.phrase_rule import (
    Ctx,
    PhraseRule,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    build_between,
    fold_range_pairs,
    has_pair,
    RangeFold,
)
from krrood.entity_query_language.verbalization.rendering.realization import (
    realize_subtree,
)
from krrood.entity_query_language.verbalization.grammar.assembly.chains import (
    ChainAssembler,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    FallbackNouns,
    Keywords,
    Logicals,
    Prepositions,
    Punctuation,
)
from krrood.entity_query_language.verbalization.vocabulary.words import ChildForm
from krrood.ormatic.utils import classes_of_module

# Constructs + relocated helpers for the complex query / inference families
# (Phase A transcription; the helpers move into grammar/ in Phase B).
from krrood.entity_query_language.core.base_expressions import Filter
from krrood.entity_query_language.query.operations import GroupedBy, OrderedBy
from krrood.entity_query_language.query.quantifiers import ResultQuantifier
from krrood.entity_query_language.query.query import Entity, SetOf
from krrood.entity_query_language.verbalization.grammar.aggregation_kinds import (
    AGGREGATION_KIND,
)
from krrood.entity_query_language.verbalization.grammar.assembly.inference import (
    InferenceAssembler,
)
from krrood.entity_query_language.verbalization.grammar.assembly.clauses import (
    GroupedByAssembler,
    OrderedByAssembler,
)
from krrood.entity_query_language.verbalization.grammar.assembly.query import (
    QueryAssembler,
)
from krrood.entity_query_language.verbalization.grammar.planning.inference import (
    InferencePlanner,
)
from krrood.entity_query_language.verbalization.grammar.assembly.instantiated import (
    InstantiatedAssembler,
)
from krrood.entity_query_language.verbalization.grammar.planning.instantiated import (
    InstantiatedPlanner,
)

# ── comparator ───────────────────────────────────────────────────────────────


class ComparatorRule(PhraseRule):
    """``<left> <operator> <right>`` — e.g. *"is greater than 50"*."""

    construct = Comparator
    name = "comparator"

    def build(self, node, ctx: Ctx):
        return ConditionVerbalizer(ctx).predicate(node)


# ── variables ──────────────────────────────────────────────────────────────────


class VariableRule(PhraseRule):
    """*"a/an Robot"* (first mention), *"the Robot"* (subsequent), or *"Robot N"* (numbered)."""

    construct = Variable
    name = "variable"

    def build(self, node, ctx: Ctx):
        if ctx.number is Number.PLURAL:
            return self._plural(node, ctx)
        definiteness, label = ctx.refer.noun_for_parts(node)
        return NounPhrase(
            head=RoleFragment.for_variable(label, node),
            definiteness=definiteness,
            referent_id=node._id_,
        )

    @staticmethod
    def _plural(node, ctx: Ctx):
        """Bare plural variable NP (*"Robots"*); the determiner phase drops the article and
        the morphology pass inflects the head.

        A numbered label (*"Robot 2"*) is surface-final — kept singular and bare; a plain type
        name is a plural indefinite NP (the concord table renders it bare-then-pluralised).
        """
        label, numbered = ctx.refer.numbered_label(node)
        return NounPhrase(
            head=RoleFragment.for_variable(label, node),
            number=Number.SINGULAR if numbered else Number.PLURAL,
            definiteness=Definiteness.BARE if numbered else Definiteness.INDEFINITE,
            referent_id=node._id_,
        )


class LiteralRule(PhraseRule):
    """A literal value (``Literal <: Variable`` — deeper construct wins via ``select``)."""

    construct = Literal
    name = "literal"

    def build(self, node, ctx: Ctx):
        return RoleFragment(
            text=ctx.context.type_name_of_value(node._value_),
            role=SemanticRole.LITERAL,
        )


class ExternalVariableRule(PhraseRule):
    """*"a/an TypeName"* for an opaque externally-set variable (no coreference)."""

    construct = ExternallySetVariable
    name = "external-variable"

    def build(self, node, ctx: Ctx):
        type_name = (
            node._type_.__name__
            if getattr(node, "_type_", None)
            else FallbackNouns.VARIABLE.text
        )
        return NounPhrase(head=RoleFragment(text=type_name, role=SemanticRole.VARIABLE))


# ── logical ──────────────────────────────────────────────────────────────────


class AndRule(PhraseRule):
    """Conjunction *"a, b, and c"* (Oxford comma); flattens nested ANDs."""

    construct = AND
    name = "and"

    def build(self, node, ctx: Ctx):
        parts = [ctx.child(conjunct) for conjunct in flatten_operands(node, AND)]
        if len(parts) == 1:
            return parts[0]
        return oxford_and(parts, Conjunctions.AND.as_fragment())


class RangeConjunctionRule(PhraseRule):
    """A conjunction containing a lo/hi pair on one chain → *"… is between lo and hi"*."""

    construct = AND
    name = "and-range"

    def when(self, node, ctx: Ctx):
        return has_pair(flatten_operands(node, AND))

    def build(self, node, ctx: Ctx):
        parts: List[VerbFragment] = []
        for item in fold_range_pairs(flatten_operands(node, AND)):
            if isinstance(item, RangeFold):
                parts.append(
                    build_between(
                        ctx.child(item.chain_expression),
                        ctx.child(item.lower_expression),
                        ctx.child(item.upper_expression),
                        compact=ctx.config.compact_predicates,
                    )
                )
            else:
                parts.append(ctx.child(item))
        if len(parts) == 1:
            return parts[0]
        return oxford_and(parts, Conjunctions.AND.as_fragment())


class OrRule(PhraseRule):
    """Disjunction *"either a, b, or c"*; flattens nested ORs."""

    construct = OR
    name = "or"

    def build(self, node, ctx: Ctx):
        parts = [ctx.child(conjunct) for conjunct in flatten_operands(node, OR)]
        if len(parts) == 1:
            return parts[0]
        head_with_comma = PhraseFragment(
            parts=[
                PhraseFragment(
                    parts=parts[:-1], separator=Punctuation.COMMA.text + " "
                ),
                Punctuation.COMMA.as_fragment(),
            ],
            separator="",
        )
        return PhraseFragment(
            parts=[
                Logicals.EITHER.as_fragment(),
                head_with_comma,
                Conjunctions.OR.as_fragment(),
                parts[-1],
            ]
        )


class NotRule(PhraseRule):
    """Generic negation *"not (<child>)"* (specialised by the guarded Not rules below)."""

    construct = Not
    name = "not"

    def build(self, node, ctx: Ctx):
        child_fragment = ctx.child(node._child_)
        return PhraseFragment(
            parts=[
                Logicals.NOT.as_fragment(),
                PhraseFragment(
                    parts=[
                        Punctuation.OPEN_PAREN.as_fragment(),
                        child_fragment,
                        Punctuation.CLOSE_PAREN.as_fragment(),
                    ],
                    separator="",
                ),
            ]
        )


class NotComparatorRule(PhraseRule):
    """Inline negated comparator *"a is not greater than b"* (Not over a Comparator)."""

    construct = Not
    name = "not-comparator"

    def when(self, node, ctx: Ctx):
        return isinstance(node._child_, Comparator)

    def build(self, node, ctx: Ctx):
        return ConditionVerbalizer(ctx).predicate(node._child_, negated=True)


class NotBoolAttrRule(PhraseRule):
    """Negated boolean attribute chain *"<nav> is not <attr>"* (Not over a bool-attr chain)."""

    construct = Not
    name = "not-bool-attr"

    def when(self, node, ctx: Ctx):
        return is_bool_attr_chain(node._child_)

    def build(self, node, ctx: Ctx):
        return ChainAssembler(ctx).chain(node._child_, negated=True)


# ── chains (MappedVariable) ──────────────────────────────────────────────────


class MappedVariableRule(PhraseRule):
    """Any attribute / index / call chain → possessive or predicative phrase."""

    construct = MappedVariable
    name = "mapped-variable"

    def build(self, node, ctx: Ctx):
        return ChainAssembler(ctx).chain(node)


class FlatVariableRule(PhraseRule):
    """A transparent SetOf wrapper → unwrap to its child (forwarding the requested number)."""

    construct = FlatVariable
    name = "flat-variable"

    def build(self, node, ctx: Ctx):
        return ctx.child(node._child_, number=ctx.number)


# ── aggregators ──────────────────────────────────────────────────────────────


class AggregatorRule(PhraseRule):
    """*"the <aggregation> <plural child>"* (or *"the <aggregation> of <child>"*)."""

    construct = Aggregator
    name = "aggregator"

    def build(self, node, ctx: Ctx):
        aggregation_kind = AGGREGATION_KIND[type(node)]
        aggregation_word = aggregation_kind.value
        aggregation_fragment = aggregation_kind.as_fragment()

        if aggregation_word.child_form is ChildForm.NONE:
            return aggregation_fragment  # childless aggregate, e.g. "count of all"
        if aggregation_word.child_form == ChildForm.SINGULAR_OF:
            child_fragment = ctx.child(node._child_)
            modifiers = [Prepositions.OF.as_fragment(), child_fragment]
        else:
            child_fragment = ctx.child(node._child_, number=Number.PLURAL)
            modifiers = [child_fragment]
        return NounPhrase(
            head=aggregation_fragment,
            definiteness=Definiteness.DEFINITE,
            modifiers=modifiers,
        )


# ── quantifiers ──────────────────────────────────────────────────────────────


class ForAllRule(PhraseRule):
    """*"for all <plural var>, <condition>"*."""

    construct = ForAll
    name = "for-all"

    def build(self, node, ctx: Ctx):
        variable_fragment = ctx.child(node.variable, number=Number.PLURAL)
        condition_fragment = ctx.child(node.condition)
        return PhraseFragment(
            parts=[
                Logicals.FOR_ALL.as_fragment(),
                variable_fragment,
                Punctuation.COMMA.as_fragment(),
                condition_fragment,
            ]
        )


class ExistsRule(PhraseRule):
    """*"there exists <variable> such that <condition>"*."""

    construct = Exists
    name = "exists"

    def build(self, node, ctx: Ctx):
        variable_fragment = ctx.child(node.variable)
        condition_fragment = ctx.child(node.condition)
        return PhraseFragment(
            parts=[
                Logicals.THERE_EXISTS.as_fragment(),
                variable_fragment,
                Keywords.SUCH_THAT.as_fragment(),
                condition_fragment,
            ]
        )


# ── query / entity family ───────────────────────────────────────────────────
# Phase B: each rule wires the query planner (what to say) + assembler (realise it).


class TopLevelEntityRule(PhraseRule):
    """Top-level Entity → imperative *"Find …"* (only at query_depth 0)."""

    construct = Entity
    name = "top-level-entity"

    def when(self, node, ctx: Ctx):
        return ctx.config.query_depth == 0

    def build(self, node, ctx: Ctx):
        return QueryAssembler(ctx).assemble(node)


class NestedEntityRule(PhraseRule):
    """Nested Entity → noun phrase (query_depth > 0); never emits *"Find"*."""

    construct = Entity
    name = "nested-entity"

    def when(self, node, ctx: Ctx):
        return ctx.config.query_depth > 0

    def build(self, node, ctx: Ctx):
        return QueryAssembler(ctx).assemble_nested(node)


class InferenceRuleRule(PhraseRule):
    """Top-level inference-rule Entity → ``IF … THEN …`` block."""

    construct = Entity
    name = "inference-rule"
    tiebreak = 1  # beats TopLevelEntityRule when both match (same construct + depth 0)

    def when(self, node, ctx: Ctx):
        return ctx.config.query_depth == 0 and InferencePlanner.can_handle(node)

    def build(self, node, ctx: Ctx):
        return InferenceAssembler(ctx).assemble(node)


class SetOfRule(PhraseRule):
    """SetOf → *"Find (v1, v2, …) such that …"*."""

    construct = SetOf
    name = "set-of"

    def build(self, node, ctx: Ctx):
        return QueryAssembler(ctx).assemble_set_of(node)


class ResultQuantifierRule(PhraseRule):
    """Transparent wrapper (An / The / …) → delegate to the child."""

    construct = ResultQuantifier
    name = "result-quantifier"

    def build(self, node, ctx: Ctx):
        return ctx.child(node._child_)


class FilterRule(PhraseRule):
    """Transparent wrapper (Where / Having) → delegate to the condition."""

    construct = Filter
    name = "filter"

    def build(self, node, ctx: Ctx):
        return ctx.child(node.condition)


class GroupedByRule(PhraseRule):
    """GroupedBy → *"grouped by <key1>, <key2>, …"* (or *"grouped"* when keyless)."""

    construct = GroupedBy
    name = "grouped-by"

    def build(self, node, ctx: Ctx):
        return GroupedByAssembler(ctx).assemble(node)


class OrderedByRule(PhraseRule):
    """OrderedBy → *"ordered by <variable> (ascending|descending)"*."""

    construct = OrderedBy
    name = "ordered-by"

    def build(self, node, ctx: Ctx):
        return OrderedByAssembler(ctx).assemble(node)


# ── instantiated variable ────────────────────────────────────────────────────


class InstantiatedVariableRule(PhraseRule):
    """*"a TypeName where the field of the TypeName is … such that …"*."""

    construct = InstantiatedVariable
    name = "instantiated-variable"

    def build(self, node, ctx: Ctx):
        return InstantiatedAssembler(ctx).assemble(node)


class InstantiatedVerbalizableRule(PhraseRule):
    """An InstantiatedVariable whose type supplies a verbalization template string."""

    construct = InstantiatedVariable
    name = "instantiated-verbalizable"

    def when(self, node, ctx: Ctx):
        return InstantiatedPlanner.has_template(node)

    def build(self, node, ctx: Ctx):
        # An opaque format string: it consumes finalized child text, so it realizes its
        # children locally (morphology pass + flatten) rather than deferring to the global pass.
        template = node._type_._verbalization_template_()
        kwargs = {
            name: realize_subtree(ctx.child(child))
            for name, child in node._child_vars_.items()
        }
        return WordFragment(text=template.format(**kwargs))


# ── the full grammar ─────────────────────────────────────────────────────────────
# Auto-discovered: one instance of every concrete PhraseRule subclass *defined in this
# module* (``classes_of_module`` excludes the imported ``PhraseRule`` base).  A new rule is
# registered simply by defining its class here.  Order is irrelevant — ``select`` decides
# specificity — so the discovery order (alphabetical) is fine.
RULES: List[PhraseRule] = [
    rule_cls()
    for rule_cls in classes_of_module(sys.modules[__name__])
    if issubclass(rule_cls, PhraseRule)
]
