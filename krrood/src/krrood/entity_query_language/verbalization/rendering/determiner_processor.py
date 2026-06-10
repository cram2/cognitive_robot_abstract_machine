"""
DeterminerProcessor — the realisation pass that **lowers the DP** (determiner phrase).

Rules emit a :class:`~krrood.entity_query_language.verbalization.fragments.base.NounPhrase`
spec carrying grammatical features (number + definiteness) but *no* surface determiner.  This
one pass walks the finished fragment tree and replaces every ``NounPhrase`` with a
:class:`~krrood.entity_query_language.verbalization.fragments.base.PhraseFragment`, choosing the
determiner from the single **concord table**:

==============  ==================  ======================
definiteness    singular            plural
==============  ==================  ======================
INDEFINITE      *"a/an"* + head     Ø (bare) + head
DEFINITE        *"the"* + head      *"the"* + head
BARE            Ø                   Ø
==============  ==================  ======================

The cell ``INDEFINITE × PLURAL → bare`` is the determiner-drop (*"a Robot"* → *"Robots"*): the
indefinite article is inherently singular, so a bare plural is its plural counterpart.  Putting
the determiner decision here means it lives in exactly one place rather than being re-decided at
every noun-phrase site.

It also propagates the phrase's :class:`Number` onto the head leaf, so the *subsequent*
:class:`~krrood.entity_query_language.verbalization.rendering.morphology_processor.MorphologyProcessor`
pass inflects it.  This pass runs **before** morphology and fully eliminates ``NounPhrase`` nodes,
so the morphology pass only ever sees plain Word/Role/Phrase leaves.

Reference: Gatt & Reiter (2009), SimpleNLG — ``NPPhraseSpec`` realisation (the determiner is
chosen by a realisation processor from the spec's features); Reiter & Dale (2000) — microplanning.
"""

from __future__ import annotations

from dataclasses import replace

from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    map_fragment,
    NounPhrase,
    PhraseFragment,
    RoleFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Articles


class DeterminerProcessor:
    """Lower every :class:`NounPhrase` to a determiner-bearing :class:`PhraseFragment`."""

    def process(self, fragment: VerbFragment) -> VerbFragment:
        """Return a new tree with every ``NounPhrase`` lowered (idempotent on NP-free trees)."""
        return map_fragment(fragment, self._lower_if_noun_phrase)

    def _lower_if_noun_phrase(self, leaf: VerbFragment) -> VerbFragment:
        """``map_fragment`` leaf hook — a ``NounPhrase`` is a leaf to be lowered, else identity."""
        return self._lower_noun_phrase(leaf) if isinstance(leaf, NounPhrase) else leaf

    def _lower_noun_phrase(self, np: NounPhrase) -> VerbFragment:
        head = self._tag_number(self.process(np.head), np.number)
        determiner = self._determiner(np.definiteness, np.number, head)
        head_group_parts = [*([determiner] if determiner is not None else []), head]
        head_group = (
            head_group_parts[0]
            if len(head_group_parts) == 1
            else PhraseFragment(parts=head_group_parts)
        )
        if not np.modifiers:
            return head_group
        modifiers = [self.process(m) for m in np.modifiers]
        return PhraseFragment(
            parts=[head_group, *modifiers], separator=np.modifier_separator
        )

    @staticmethod
    def _tag_number(head: VerbFragment, number: Number) -> VerbFragment:
        """Tag the head leaf with the phrase's number (the morphology pass inflects it)."""
        if isinstance(head, (WordFragment, RoleFragment)):
            return replace(head, number=number)
        return head

    @staticmethod
    def _determiner(definiteness: Definiteness, number: Number, head: VerbFragment):
        """The determiner fragment for *(definiteness, number)*, or ``None`` (bare)."""
        if definiteness is Definiteness.UNIQUE:
            return Articles.THE_UNIQUE.as_fragment()
        if definiteness is Definiteness.DEFINITE:
            return Articles.THE.as_fragment()
        if definiteness is Definiteness.INDEFINITE and number is Number.SINGULAR:
            return Articles.indefinite(flatten_fragment_to_plain_text(head))
        return None  # BARE, or INDEFINITE + PLURAL → the determiner-drop
