from __future__ import annotations

from dataclasses import replace

from typing_extensions import Optional

from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    NounPhrase,
    PhraseFragment,
    RoleFragment,
    VerbalizationFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.rendering.passes import RewritePass
from krrood.entity_query_language.verbalization.vocabulary.english import Articles


class DeterminerProcessor(RewritePass):
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

    Reference: :cite:t:`gatt2009simplenlg` — ``NPPhraseSpec`` realisation;
    :cite:t:`reiter2000building` — microplanning.
    """

    def rewrite(self, leaf: VerbalizationFragment) -> VerbalizationFragment:
        """:return: A lowered noun-phrase leaf; any other leaf passes through unchanged.

        >>> verbalize_expression(a(entity(variable(Robot, []))))
        'Find a Robot'
        """
        return self._lower_noun_phrase(leaf) if isinstance(leaf, NounPhrase) else leaf

    def _lower_noun_phrase(self, noun_phrase: NounPhrase) -> VerbalizationFragment:
        """:return: *noun_phrase* lowered to a determiner-bearing phrase — the chosen determiner,
        an optional pre-head qualifier, the number-tagged head, and the recursed modifiers.

        >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
        >>> phrase = NounPhrase(head=RoleFragment.for_type(Robot), definiteness=Definiteness.INDEFINITE)
        >>> flatten_fragment_to_plain_text(DeterminerProcessor()._lower_noun_phrase(phrase))
        'a Robot'
        """
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
    def _tag_number(
        head: VerbalizationFragment, number: GrammaticalNumber
    ) -> VerbalizationFragment:
        """Tag the head leaf with the phrase's number.

        >>> DeterminerProcessor._tag_number(WordFragment(text="Robot"), GrammaticalNumber.PLURAL).number
        <GrammaticalNumber.PLURAL: 'plural'>
        """
        if isinstance(head, (WordFragment, RoleFragment)):
            return replace(head, number=number)
        return head

    @staticmethod
    def _determiner(
        definiteness: Definiteness,
        number: GrammaticalNumber,
        article_anchor: VerbalizationFragment,
    ) -> Optional[VerbalizationFragment]:
        """:return: The determiner fragment for *(definiteness, number)*, or ``None`` (bare). The
        indefinite *a/an* agrees phonologically with *article_anchor* (the first surface word —
        the pre-head when present, else the head).

        >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
        >>> flatten_fragment_to_plain_text(
        ...     DeterminerProcessor._determiner(Definiteness.INDEFINITE, GrammaticalNumber.SINGULAR, WordFragment(text="hour")))
        'an'
        >>> flatten_fragment_to_plain_text(
        ...     DeterminerProcessor._determiner(Definiteness.DEFINITE, GrammaticalNumber.SINGULAR, WordFragment(text="Robot")))
        'the'
        >>> DeterminerProcessor._determiner(Definiteness.INDEFINITE, GrammaticalNumber.PLURAL, WordFragment(text="Robot")) is None
        True
        """
        if definiteness is Definiteness.UNIQUE:
            return Articles.THE_UNIQUE.as_fragment()
        if definiteness is Definiteness.DEFINITE:
            return Articles.THE.as_fragment()
        if (
            definiteness is Definiteness.INDEFINITE
            and number is GrammaticalNumber.SINGULAR
        ):
            return Articles.indefinite(flatten_fragment_to_plain_text(article_anchor))
        return None  # BARE, or INDEFINITE + PLURAL → the determiner-drop
