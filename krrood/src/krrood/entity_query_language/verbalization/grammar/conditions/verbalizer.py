"""
Condition **verbalizer** — the single owner of every surface form a condition can take.

A Comparator/condition is said differently depending on where it sits: a standalone
*predicate* (*"x is greater than 5"*), a post-nominal *attribute modifier* on a subject
(the bare *"<attr> op <value>"* that a *"whose …"* envelope wraps), a *range* modifier
(*"<attr> is between lo and hi"*), or the inference *whose-attribute* body (*"<attr> is
<value>"* agreeing in number).  Previously each of these lived in a different consumer
(the comparator rule, the restriction rules, the inference assembler); they are
co-located here so one component owns *how a condition is verbalized* and the consumers
merely ask for a form.

Realisation-only (``planner = None``), holding the per-node
:class:`~krrood.entity_query_language.verbalization.grammar.phrase_rule.Ctx`; recursion is
via ``self.ctx.child``.

Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
"""

from __future__ import annotations

from typing_extensions import Any

from krrood.entity_query_language.verbalization.fragments.base import (
    PhraseFragment,
    RoleFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.fragments.factory import phrase, role
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.grammar.assembly.base import Assembler
from krrood.entity_query_language.verbalization.grammar.conditions.operator_phrase import (
    comparator_operator,
)
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    single_hop_attr,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    build_between,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Copulas,
    Keywords,
)
from krrood.entity_query_language.verbalization.vocabulary.words import Number


class ConditionVerbalizer(Assembler[Any, None]):
    """Render a condition in a requested surface form (predicate / modifier / …)."""

    def realize(self, node, plan: None = None) -> VerbFragment:
        """Default form — a standalone predicate (the :class:`Assembler` entry point)."""
        return self.predicate(node)

    def predicate(self, comparator, *, negated: bool = False) -> VerbFragment:
        """*"<left> <operator> <right>"* — the standalone comparator form."""
        return phrase(
            self.ctx.child(comparator.left),
            comparator_operator(comparator, self.ctx.context, negated=negated),
            self.ctx.child(comparator.right),
        )

    def attribute_modifier(self, comparator, subject) -> VerbFragment:
        """Bare *"<attr> <operator> <value>"* on *subject*'s single-hop attribute — the
        grouped predicate a *"whose …"* envelope wraps."""
        attr = single_hop_attr(comparator.left, subject)
        return PhraseFragment(
            parts=[
                RoleFragment.for_attribute(attr._owner_class_, attr._attribute_name_),
                comparator_operator(comparator, self.ctx.context, compact=False),
                self.ctx.child(comparator.right),
            ]
        )

    def range_modifier(self, rangefold, subject) -> VerbFragment:
        """*"<attr> is between lo and hi"* on *subject*'s single-hop attribute."""
        attr = single_hop_attr(rangefold.chain_expression, subject)
        left = RoleFragment.for_attribute(attr._owner_class_, attr._attribute_name_)
        return build_between(
            left,
            self.ctx.child(rangefold.lower_expression),
            self.ctx.child(rangefold.upper_expression),
            compact=False,
        )

    def whose_attribute(
        self, attr_name: str, number: Number, value: VerbFragment
    ) -> VerbFragment:
        """Full *"whose <attr> <copula> <value>"* modifier, agreeing with *number*.

        The *value* fragment is supplied by the caller (it may itself be number-folded);
        the attribute noun and copula agree with *number*.
        """
        return phrase(
            Keywords.WHOSE.as_fragment(),
            self._attr_noun(attr_name, number),
            Copulas.for_number(number),
            value,
        )

    def _attr_noun(self, name: str, number: Number) -> VerbFragment:
        """A role-tagged attribute noun tagged with *number* (the pass inflects it)."""
        return role(name, SemanticRole.ATTRIBUTE, number=number)
