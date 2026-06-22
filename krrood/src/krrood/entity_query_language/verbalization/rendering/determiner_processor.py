from __future__ import annotations

from dataclasses import replace

from typing_extensions import Optional

from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    map_fragment,
    NounPhrase,
    PhraseFragment,
    RoleFragment,
    Fragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Articles


class DeterminerProcessor:
    """
    Lower every noun phrase to a determiner-bearing phrase.

    Rules emit a noun-phrase specification carrying grammatical features (number + definiteness) but no
    surface determiner. This pass walks the finished fragment tree and replaces every noun phrase
    with a plain phrase, choosing the determiner from the single concord table:

    ==============  ==================  ======================
    definiteness    singular            plural
    ==============  ==================  ======================
    INDEFINITE      *"a/an"* + head     Ø (bare) + head
    DEFINITE        *"the"* + head      *"the"* + head
    UNIQUE          *"the unique"*      *"the unique"*
    BARE            Ø                   Ø
    ==============  ==================  ======================

    The cell ``INDEFINITE × PLURAL → bare`` is the determiner-drop (*"a Robot"* → *"Robots"*):
    the indefinite article is inherently singular, so a bare plural is its plural counterpart.

    Reference: Gatt & Reiter (2009), SimpleNLG — ``NPPhraseSpec`` realisation; Reiter & Dale
    (2000) — microplanning.
    """

    def process(self, fragment: Fragment) -> Fragment:
        """
        :param fragment: Root of the fragment tree.
        :return: A new tree with every noun phrase lowered (idempotent on noun phrase-free trees).
        """
        return map_fragment(fragment, self._lower_if_noun_phrase)

    def _lower_if_noun_phrase(self, leaf: Fragment) -> Fragment:
        """:return: A lowered noun-phrase leaf; any other leaf passes through unchanged."""
        return self._lower_noun_phrase(leaf) if isinstance(leaf, NounPhrase) else leaf

    def _lower_noun_phrase(self, noun_phrase: NounPhrase) -> Fragment:
        head = self._tag_number(self.process(noun_phrase.head), noun_phrase.number)
        # A pre-head qualifier sits between the determiner and the head: "the [first two] Robots" /
        # "a [specific] Body". The indefinite article agrees with the first surface word, so it is
        # chosen from the pre-head when there is one, else the head ("a specific Body", not "an …").
        pre_head_fragment = (
            self.process(noun_phrase.pre_head)
            if noun_phrase.pre_head is not None
            else None
        )
        determiner = self._determiner(
            noun_phrase.definiteness,
            noun_phrase.number,
            pre_head_fragment if pre_head_fragment is not None else head,
        )
        pre_head = [pre_head_fragment] if pre_head_fragment is not None else []
        head_group_parts = [
            *([determiner] if determiner is not None else []),
            *pre_head,
            head,
        ]
        head_group = (
            head_group_parts[0]
            if len(head_group_parts) == 1
            else PhraseFragment(parts=head_group_parts)
        )
        if not noun_phrase.modifiers:
            return head_group
        modifiers = [self.process(modifier) for modifier in noun_phrase.modifiers]
        return PhraseFragment(
            parts=[head_group, *modifiers], separator=noun_phrase.modifier_separator
        )

    @staticmethod
    def _tag_number(head: Fragment, number: Number) -> Fragment:
        """Tag the head leaf with the phrase's number."""
        if isinstance(head, (WordFragment, RoleFragment)):
            return replace(head, number=number)
        return head

    @staticmethod
    def _determiner(
        definiteness: Definiteness, number: Number, article_anchor: Fragment
    ) -> Optional[Fragment]:
        """:return: The determiner fragment for *(definiteness, number)*, or ``None`` (bare). The
        indefinite *a/an* agrees phonologically with *article_anchor* (the first surface word —
        the pre-head when present, else the head)."""
        if definiteness is Definiteness.UNIQUE:
            return Articles.THE_UNIQUE.as_fragment()
        if definiteness is Definiteness.DEFINITE:
            return Articles.THE.as_fragment()
        if definiteness is Definiteness.INDEFINITE and number is Number.SINGULAR:
            return Articles.indefinite(flatten_fragment_to_plain_text(article_anchor))
        return None  # BARE, or INDEFINITE + PLURAL → the determiner-drop
