from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Optional

from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    oxford_comma,
    Fragment,
)
from krrood.entity_query_language.verbalization.grammar.conditions.forms import (
    Placed,
    Placement,
    Slot,
    place,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    RuleContext,
)
from krrood.entity_query_language.verbalization.grammar.query.planner import (
    RestrictionPlan,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    Keywords,
)
from krrood.entity_query_language.verbalization.vocabulary.words import Number


@dataclass(frozen=True)
class RestrictionFragments:
    """The rendered pieces of a subject restriction, for the caller to place — each goes to a
    different sentence position, so they are kept apart rather than pre-joined."""

    inline_modifiers: List[Fragment] = field(default_factory=list)
    """Superlative selection phrases (*"with the maximum amount"*) that attach inline, right after
    the selection noun."""

    whose: Optional[Fragment] = None
    """The *"whose"* group as a coordinated block (header *"whose"*, one bare predicate per item,
    Oxford-joined) — a sub-list of points in hierarchical rendering, *"whose a, and b"* inline /
    in paragraph. ``None`` when nothing folds into a *"whose"*."""

    residual: Optional[Fragment] = None
    """The residual condition for a separate *"such that …"* / *"where …"* clause; the caller picks
    the keyword and position. ``None`` when the WHERE folds entirely into the other pieces."""


@dataclass
class RestrictionAssembler:
    """
    Render a subject's WHERE partition into its surface pieces: superlative selection modifiers
    (*"with the maximum <leaf>"*), the appositive *"whose <grouped>"* modifier, and the residual
    *"such that …"* / *"where …"* condition.

    Each folded conjunct is placed by the shared :func:`~...conditions.forms.place` — the one
    authority on a condition's surface form and slot — so this assembler only buckets the results
    by slot into the two output pieces. A restriction yields several pieces, not a single fragment,
    so it is realisation-only but not an ``Assembler`` subclass.

    Reference: Reiter & Dale (2000) — content structuring (the WHERE partition is the plan).
    """

    context: RuleContext
    """The per-node context (recursion entry and microplanning services)."""

    def render(
        self,
        restriction: RestrictionPlan,
        subject: Variable,
        number: Number = Number.SINGULAR,
    ) -> RestrictionFragments:
        """
        Place each folded conjunct via the form registry, then bucket by slot: superlatives and the
        shared *"whose …"* envelope become noun modifiers; standalone conjuncts join into the
        residual clause.

        :param restriction: The subject's folded WHERE conjuncts.
        :param subject: The variable the restriction is on.
        :param number: The number the subject agrees with — singular for a query selection, plural
            for an aggregated inference antecedent.
        :return: The rendered restriction pieces.
        """
        placed = [
            place(Placement(item=item, subject=subject, number=number), self.context)
            for item in restriction.folded
        ]
        superlatives = self._of_slot(placed, Slot.SELECTION_MODIFIER)
        grouped = self._of_slot(placed, Slot.WHOSE)
        standalone = self._of_slot(placed, Slot.STANDALONE)

        whose = (
            BlockFragment(
                header=Keywords.WHOSE.as_fragment(),
                items=grouped,
                conjunction=Conjunctions.AND.as_fragment(),
            )
            if grouped
            else None
        )
        return RestrictionFragments(
            inline_modifiers=superlatives, whose=whose, residual=self._join(standalone)
        )

    @staticmethod
    def _of_slot(placed: List[Placed], slot: Slot) -> List[Fragment]:
        """:return: The rendered fragments occupying *slot*, in order."""
        return [item.fragment for item in placed if item.slot is slot]

    @staticmethod
    def _join(fragments: List[Fragment]) -> Optional[Fragment]:
        """:return: The standalone conjuncts joined into one residual condition, or ``None``."""
        if not fragments:
            return None
        if len(fragments) == 1:
            return fragments[0]
        return oxford_comma(fragments, Conjunctions.AND.as_fragment())
