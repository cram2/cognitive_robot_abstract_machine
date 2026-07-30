from __future__ import annotations

from dataclasses import replace

from typing_extensions import List, Optional

from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    NounPhrase,
    oxford_comma,
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
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Conjunctions,
)


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
        """Lower *noun_phrase* to a determiner-bearing phrase: the chosen determiner, an optional
        ordinal-distinguisher and pre-head qualifier, the number-tagged head, and the recursed
        modifiers. When :attr:`~…fragments.base.NounPhrase.additional_heads` is non-empty, each
        alternative becomes its own determiner-and-head group (the ordinal/pre-head qualifier
        applies only to the first), joined with *"or"*.

        :param noun_phrase: The noun-phrase specification to lower.
        :return: The lowered fragment.

        >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
        >>> phrase = NounPhrase(head=RoleFragment.for_type(Robot), definiteness=Definiteness.INDEFINITE)
        >>> flatten_fragment_to_plain_text(DeterminerProcessor()._lower_noun_phrase(phrase))
        'a Robot'
        """
        # An ordinal distinguisher ("a [second] Robot") sits ahead of any pre-head qualifier ("the
        # [first two] Robots" / "a [specific] Body"), both between the determiner and the head. The
        # indefinite article agrees with the first surface word, so the anchor is the ordinal when
        # present, else the pre-head, else the head ("a specific Body", not "an …"). These qualify
        # only the leading head — each additional disjunctive alternative is a bare head of its own.
        ordinal_fragment = (
            WordFragment(text=morphology.ordinal(noun_phrase.ordinal - 1))
            if noun_phrase.ordinal is not None
            else None
        )
        pre_head_fragment = (
            self.process(noun_phrase.pre_head)
            if noun_phrase.pre_head is not None
            else None
        )
        leading_fragment = (
            ordinal_fragment if ordinal_fragment is not None else pre_head_fragment
        )
        pre_head = [
            fragment
            for fragment in (ordinal_fragment, pre_head_fragment)
            if fragment is not None
        ]
        head_group = self._head_group(
            noun_phrase, noun_phrase.head, pre_head, leading_fragment
        )
        if noun_phrase.additional_heads:
            additional_groups = [
                self._head_group(noun_phrase, additional, [], None)
                for additional in noun_phrase.additional_heads
            ]
            head_group = oxford_comma(
                [head_group, *additional_groups], Conjunctions.OR.as_fragment()
            )
        if not noun_phrase.modifiers:
            return head_group
        modifiers = [self.process(modifier) for modifier in noun_phrase.modifiers]
        return PhraseFragment(
            parts=[head_group, *modifiers], separator=noun_phrase.modifier_separator
        )

    def _head_group(
        self,
        noun_phrase: NounPhrase,
        head: VerbalizationFragment,
        pre_head: List[VerbalizationFragment],
        leading_fragment: Optional[VerbalizationFragment],
    ) -> VerbalizationFragment:
        """
        Build one determiner-and-head group: *head*, number-tagged and processed,
        preceded by its own determiner (chosen from *leading_fragment*, or *head* itself
        when there is none) and any *pre_head* qualifier. *noun_phrase* supplies the
        shared definiteness, number, and alternative/pair-distinguisher features every
        group agrees on; only the article is decided per group.

        :param noun_phrase: The enclosing phrase, for its shared definiteness/number/alternative.
        :param head: The head fragment for this group (the phrase's own head, or one of its
            :attr:`~…fragments.base.NounPhrase.additional_heads`).
        :param pre_head: Any pre-head qualifier fragments (ordinal, ranking phrase) for this
            group; empty for a non-leading additional head.
        :param leading_fragment: The article's phonological anchor when it differs from *head*
            (the ordinal or pre-head qualifier); ``None`` to anchor on *head* itself.
        :return: The determiner-and-head group fragment.
        """
        processed_head = self._tag_number(self.process(head), noun_phrase.number)
        determiner = self._determiner(
            noun_phrase.definiteness,
            noun_phrase.number,
            leading_fragment if leading_fragment is not None else processed_head,
            alternative=noun_phrase.alternative,
        )
        parts = [
            *([determiner] if determiner is not None else []),
            *pre_head,
            processed_head,
        ]
        return parts[0] if len(parts) == 1 else PhraseFragment(parts=parts)

    @staticmethod
    def _tag_number(
        head: VerbalizationFragment, number: GrammaticalNumber
    ) -> VerbalizationFragment:
        """
        Tag the head leaf with the phrase's number.

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
        *,
        alternative: bool = False,
    ) -> Optional[VerbalizationFragment]:
        """:return: The determiner fragment for *(definiteness, number)*, or ``None`` (bare). The
        indefinite *a/an* agrees phonologically with *article_anchor* (the first surface word — the
        ordinal distinguisher or pre-head when present, else the head).

        *alternative* selects the fused indefinite/definite pair-distinguisher determiner in place
        of the ordinary article — *"another"* (indefinite singular) / *"the other"* (definite) —
        for the second of a same-noun pair of distinct referents (:attr:`NounPhrase.alternative`).

        >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
        >>> flatten_fragment_to_plain_text(
        ...     DeterminerProcessor._determiner(Definiteness.INDEFINITE, GrammaticalNumber.SINGULAR, WordFragment(text="hour")))
        'an'
        >>> flatten_fragment_to_plain_text(
        ...     DeterminerProcessor._determiner(Definiteness.DEFINITE, GrammaticalNumber.SINGULAR, WordFragment(text="Robot")))
        'the'
        >>> DeterminerProcessor._determiner(Definiteness.INDEFINITE, GrammaticalNumber.PLURAL, WordFragment(text="Robot")) is None
        True
        >>> flatten_fragment_to_plain_text(
        ...     DeterminerProcessor._determiner(Definiteness.INDEFINITE, GrammaticalNumber.SINGULAR,
        ...                                      WordFragment(text="Robot"), alternative=True))
        'another'
        >>> flatten_fragment_to_plain_text(
        ...     DeterminerProcessor._determiner(Definiteness.DEFINITE, GrammaticalNumber.SINGULAR,
        ...                                      WordFragment(text="Robot"), alternative=True))
        'the other'
        """
        if definiteness is Definiteness.UNIQUE:
            return Articles.THE_UNIQUE.as_fragment()
        if definiteness is Definiteness.DEFINITE:
            return (
                Articles.THE_OTHER.as_fragment()
                if alternative
                else Articles.THE.as_fragment()
            )
        if (
            definiteness is Definiteness.INDEFINITE
            and number is GrammaticalNumber.SINGULAR
        ):
            if alternative:
                return Articles.ANOTHER.as_fragment()
            return Articles.indefinite(flatten_fragment_to_plain_text(article_anchor))
        return None  # BARE, or INDEFINITE + PLURAL → the determiner-drop
