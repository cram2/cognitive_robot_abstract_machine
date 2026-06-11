"""
English vocabulary for EQL verbalization.

Every English word or phrase used in the verbalization output is defined here
as a named constant on one of the namespace Enums.  No natural-language strings
exist outside this module.

Namespace Enums (Keywords, Logicals, …) have frozen-dataclass instances as
member values; the VocabEnum mixin exposes .as_fragment() and .text directly on
the member so callers never write .value explicitly.

Parameterised phrases (ExistentialPhrase, GroupKeyPhrases, FallbackNouns) add
delegation methods on the Enum that forward to build_phrase() / plural_fragment()
on the underlying dataclass value.

Operators uses OperatorPhrase values and exposes .select() + .from_callable().
"""

from __future__ import annotations

import operator as _operator
from dataclasses import dataclass
from enum import Enum

from krrood.entity_query_language.operators.comparator import not_contains as _nc

from krrood.entity_query_language.verbalization import morphology

from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    PhraseFragment,
    RoleFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.vocabulary.words import (
    AggregationWord,
    ChildForm,
    KeyWord,
    LogicalWord,
    Number,
    OperatorPhrase,
    OperatorWord,
    PlainWord,
    PronounWord,
    VocabEnum,
)

# ── English-specific word subtypes ─────────────────────────────────────────────
# These add behaviour for phrases that depend on a runtime type_name argument.


@dataclass(frozen=True)
class SingularExistential(PlainWord):
    """
    Parameterised existential phrase: *"there's a/an TypeName"*.

    The article (*a* / *an*) is computed phonologically at call time using the
    ``inflect`` library.
    """

    def build_phrase(self, type_name: str, referent_id=None) -> VerbFragment:
        """
        Build *"there's a/an <type_name>"* as a
        :class:`~krrood.entity_query_language.verbalization.fragments.base.PhraseFragment`.

        :param type_name: English noun in singular form (e.g. ``"Robot"``, ``"Apple"``).
        :type type_name: str
        :param referent_id: When given, the noun is a **referring** ``NounPhrase`` for that
            entity (so the coreference pass marks it introduced and reduces later mentions to
            *"the <type>"*); the indefinite article is then chosen by the determiner concord.
        :return: Phrase fragment with the correct indefinite article.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        if referent_id is not None:
            return PhraseFragment(
                parts=[
                    WordFragment(text=self.text),
                    NounPhrase(
                        head=RoleFragment(text=type_name, role=SemanticRole.VARIABLE),
                        referent_id=referent_id,
                    ),
                ],
            )
        article = morphology.indefinite_article(type_name)
        return PhraseFragment(
            parts=[
                WordFragment(text=f"{self.text} {article}"),
                RoleFragment(text=type_name, role=SemanticRole.VARIABLE),
            ],
        )


@dataclass(frozen=True)
class PluralExistential(PlainWord):
    """
    Parameterised existential phrase: *"there are TypeNames"*.
    """

    def build_phrase(self, type_name: str) -> VerbFragment:
        """
        Build *"there are <plural_type_name>"* as a
        :class:`~krrood.entity_query_language.verbalization.fragments.base.PhraseFragment`.

        :param type_name: English noun in singular form; pluralised automatically.
        :type type_name: str
        :return: Phrase fragment with pluralised type name.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        return PhraseFragment(
            parts=[
                self.as_fragment(),
                RoleFragment(
                    text=type_name,
                    role=SemanticRole.VARIABLE,
                    number=Number.PLURAL,
                ),
            ],
        )


@dataclass(frozen=True)
class FallbackNounWord(PlainWord):
    """
    A fallback noun used when no type information is available from the expression.

    Provides both :meth:`as_fragment` (singular) and :meth:`plural_fragment`.
    """

    def plural_fragment(self) -> WordFragment:
        """
        Return a :class:`~krrood.entity_query_language.verbalization.fragments.base.WordFragment`
        tagged plural — the morphology pass pluralises it (e.g. ``"entity"`` → ``"entities"``).

        :return: Plural-tagged noun fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.WordFragment
        """
        return WordFragment(text=self.text, number=Number.PLURAL)


@dataclass(frozen=True)
class CommonGroupKeyWord(PlainWord):
    """
    Group-key binding phrase: *"the common <field> of the <plural_root>"*.

    Used in the THEN clause when a consequent binding refers to a GROUP BY key.
    """

    def build_phrase(self, field_name: str, root: str) -> VerbFragment:
        """
        Build *"the common <field_name> of the <plural root>"*.

        The root type name is pluralised here (this fixed lexicon frame owns its own
        inflection, like the existential frames).

        :param field_name: Name of the grouped field (e.g. ``"location"``).
        :type field_name: str
        :param root: Singular root type name (e.g. ``"Robot"``); the morphology pass
            pluralises the tagged root leaf.
        :type root: str
        :return: Phrase fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        return PhraseFragment(
            parts=[
                self.as_fragment(),
                WordFragment(text=field_name),
                Prepositions.OF_THE.as_fragment(),
                WordFragment(text=root, number=Number.PLURAL),
            ],
        )


# ── Namespace Enums ────────────────────────────────────────────────────────────


class Keywords(VocabEnum):
    """EQL structural keywords (IF, THEN, FIND, WHERE, etc.)."""

    IF = KeyWord("If")
    THEN = KeyWord("then")
    FIND = KeyWord("Find")
    FIND_SETS_OF = KeyWord("Find sets of")
    SUCH_THAT = KeyWord("such that")
    WHERE = KeyWord("where")
    WHOSE = KeyWord("whose")
    GROUPED_BY = KeyWord("grouped by")
    GROUPED = KeyWord("grouped")
    HAVING = KeyWord("having")
    ORDERED_BY = KeyWord("ordered by")
    TRUE = KeyWord("true")


class Logicals(VocabEnum):
    """Logical connectives (NOT, FOR ALL, THERE EXISTS, EITHER)."""

    NOT = LogicalWord("not")
    EITHER = LogicalWord("either")
    FOR_ALL = LogicalWord("for all")
    THERE_EXISTS = LogicalWord("there exists")


class Aggregations(VocabEnum):
    """Aggregation function phrases (number of, sum of, average of, etc.)."""

    COUNT = AggregationWord("number of")
    COUNT_ALL = AggregationWord("count of all", child_form=ChildForm.NONE)
    SUM = AggregationWord("sum of")
    AVERAGE = AggregationWord("average of")
    MAX = AggregationWord("maximum", child_form=ChildForm.SINGULAR_OF)
    MIN = AggregationWord("minimum", child_form=ChildForm.SINGULAR_OF)
    MODE = AggregationWord("mode of")
    MULTI_MODE = AggregationWord("all modes of")


class Copulas(VocabEnum):
    """Copula verbs (IS, IS NOT, ARE) — role is OPERATOR so they share colouring with comparators."""

    IS = OperatorWord("is")
    IS_NOT = OperatorWord("is not")
    ARE = OperatorWord("are")
    ARE_NOT = OperatorWord("are not")

    @classmethod
    def for_number(cls, number: Number) -> RoleFragment:
        """The copula tagged with *number* (``OPERATOR`` role) for a directly-built leaf; the
        morphology pass agrees it (*is* / *are*).  This is the one seam for copulas not produced
        via ``ctx.child`` recursion (e.g. an InstantiatedVariable field binding)."""
        return RoleFragment(text=cls.IS.text, role=SemanticRole.OPERATOR, number=number)


class Prepositions(VocabEnum):
    """Prepositions used in possessive path phrases (OF, OF THE) and aggregate scopes (AMONG)."""

    OF = PlainWord("of")
    OF_THE = PlainWord("of the")
    AMONG = PlainWord("among")


class Conjunctions(VocabEnum):
    """Coordinating conjunctions (AND, OR)."""

    AND = PlainWord("and")
    OR = PlainWord("or")


class Punctuation(VocabEnum):
    """Structural punctuation tokens — role-less, like the brackets around a SetOf tuple
    (*"(v1, v2)"*) and the comma in a coordinated list (*"a, b, or c"*).  Kept in the lexicon
    so no punctuation literal is hard-coded in the grammar.  :attr:`COMMA`'s ``text`` also seeds
    the *", "* list separator passed to :class:`PhraseFragment`."""

    COMMA = PlainWord(",")
    OPEN_PAREN = PlainWord("(")
    CLOSE_PAREN = PlainWord(")")


class Pronouns(VocabEnum):
    """Coreference pronouns standing in for a previously introduced variable."""

    ITS = PronounWord("its")


class RangePhrases(VocabEnum):
    """
    Range (``between``) operator phrases produced when a lower-bound and an
    upper-bound comparison on the same attribute are folded together.

    * :attr:`IS_BETWEEN` — standard form keeping the copula (*"is between"*).
    * :attr:`BETWEEN` — copula-less form for HAVING / post-nominal use (*"between"*).
    """

    IS_BETWEEN = OperatorWord("is between")
    BETWEEN = OperatorWord("between")


class SortDirections(VocabEnum):
    """Sort direction words for ORDERED BY clauses."""

    ASCENDING = PlainWord("ascending")
    DESCENDING = PlainWord("descending")


class Articles(VocabEnum):
    """
    Definite articles (THE, THE UNIQUE) and a static helper for indefinite articles.
    """

    THE = PlainWord("the")
    THE_UNIQUE = PlainWord("the unique")

    @staticmethod
    def indefinite(following_word: str) -> WordFragment:
        """
        Return a :class:`~krrood.entity_query_language.verbalization.fragments.base.WordFragment`
        containing *"a"* or *"an"* based on the phonological context of *following_word*.

        :param following_word: The word immediately following the article.
        :type following_word: str
        :return: ``WordFragment("a")`` or ``WordFragment("an")``.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.WordFragment
        """
        text = morphology.indefinite_article(following_word) if following_word else "a"
        return WordFragment(text=text)


class ExistentialPhrase(VocabEnum):
    """
    Parameterised existential phrases (*"there's a/an TypeName"*, *"there are TypeNames"*).

    Call :meth:`build_phrase` on a member to produce the full fragment.
    """

    THERE_IS_A = SingularExistential("there's")
    THERE_ARE = PluralExistential("there are")

    @classmethod
    def for_number(cls, number: Number) -> "ExistentialPhrase":
        """The existential frame agreeing with *number*: ``THERE_ARE`` / ``THERE_IS_A``."""
        return cls.THERE_ARE if number is Number.PLURAL else cls.THERE_IS_A

    def build_phrase(self, type_name: str, referent_id=None) -> VerbFragment:
        """
        Delegate to the underlying :class:`SingularExistential` or :class:`PluralExistential`.

        :param type_name: English noun in singular form.
        :type type_name: str
        :param referent_id: Optional referent for coreference (singular only; see
            :meth:`SingularExistential.build_phrase`).
        :return: Existential phrase fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        if referent_id is not None and self is ExistentialPhrase.THERE_IS_A:
            return self.value.build_phrase(type_name, referent_id=referent_id)
        return self.value.build_phrase(type_name)


class FallbackNouns(VocabEnum):
    """
    Fallback noun used when no type information is available (e.g. *"entity"* / *"entities"*).
    """

    ENTITY = FallbackNounWord("entity")
    VARIABLE = FallbackNounWord("variable")

    def plural_fragment(self) -> WordFragment:
        """
        Return a plural :class:`~krrood.entity_query_language.verbalization.fragments.base.WordFragment`.

        :return: Pluralised noun fragment (e.g. ``"entities"``).
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.WordFragment
        """
        return self.value.plural_fragment()


class GroupKeyPhrases(VocabEnum):
    """
    Group-key binding phrases (*"the common <field> of the <plural_root>"*).
    """

    COMMON_OF = CommonGroupKeyWord("the common")

    def build_phrase(self, field_name: str, plural_root: str) -> VerbFragment:
        """
        Delegate to :meth:`CommonGroupKeyWord.build_phrase`.

        :param field_name: Grouped field name.
        :type field_name: str
        :param plural_root: Plural root type name.
        :type plural_root: str
        :return: Group-key phrase fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        return self.value.build_phrase(field_name, plural_root)


class Operators(Enum):
    """
    Comparison operator phrases.

    Each member's value is an :class:`~krrood.entity_query_language.verbalization.vocabulary.words.OperatorPhrase`
    holding all eight text variants.

    * Use :meth:`select` to obtain the appropriate
      :class:`~krrood.entity_query_language.verbalization.vocabulary.words.OperatorWord`
      for given ``negated``, ``compact``, and ``temporal`` flags.
    * Use :meth:`from_callable` to map a Python ``operator`` module callable
      (e.g. ``operator.gt``) to the correct :class:`Operators` member.
    """

    EQ = OperatorPhrase(
        standard="is",
        compact="equals",
        negated="is not",
        negated_compact="does not equal",
        temporal="is at",
        temporal_compact="at",
        temporal_negated="is not at",
        temporal_negated_compact="not at",
    )
    NE = OperatorPhrase(
        standard="is not",
        compact="does not equal",
        negated="is",
        negated_compact="equals",
        temporal="is not at",
        temporal_compact="not at",
        temporal_negated="is at",
        temporal_negated_compact="at",
    )
    LT = OperatorPhrase(
        standard="is less than",
        compact="less than",
        negated="is not less than",
        negated_compact="not less than",
        temporal="is before",
        temporal_compact="before",
        temporal_negated="is no earlier than",
        temporal_negated_compact="no earlier than",
    )
    LE = OperatorPhrase(
        standard="is at most",
        compact="at most",
        negated="is not at most",
        negated_compact="not at most",
        temporal="is no later than",
        temporal_compact="no later than",
        temporal_negated="is after",
        temporal_negated_compact="after",
    )
    GT = OperatorPhrase(
        standard="is greater than",
        compact="greater than",
        negated="is not greater than",
        negated_compact="not greater than",
        temporal="is after",
        temporal_compact="after",
        temporal_negated="is no later than",
        temporal_negated_compact="no later than",
    )
    GE = OperatorPhrase(
        standard="is at least",
        compact="at least",
        negated="is not at least",
        negated_compact="not at least",
        temporal="is no earlier than",
        temporal_compact="no earlier than",
        temporal_negated="is before",
        temporal_negated_compact="before",
    )
    CALC_EQ = OperatorPhrase(
        standard="is equal to",
        compact="equals",
        negated="is not equal to",
        negated_compact="does not equal",
    )
    CONTAINS = OperatorPhrase(
        standard="contains",
        compact="contains",
        negated="does not contain",
        negated_compact="does not contain",
    )
    NOT_CONTAINS = OperatorPhrase(
        standard="does not contain",
        compact="does not contain",
        negated="contains",
        negated_compact="contains",
    )

    def select(
        self, *, negated: bool = False, compact: bool = False, temporal: bool = False
    ) -> OperatorWord:
        """
        Delegate flag selection to the underlying
        :class:`~krrood.entity_query_language.verbalization.vocabulary.words.OperatorPhrase`.

        :param negated: Use the negated text variant.
        :type negated: bool
        :param compact: Use the copula-less variant for HAVING clauses.
        :type compact: bool
        :param temporal: Use the temporal variant for datetime comparisons.
        :type temporal: bool
        :return: Selected :class:`~krrood.entity_query_language.verbalization.vocabulary.words.OperatorWord`.
        :rtype: ~krrood.entity_query_language.verbalization.vocabulary.words.OperatorWord
        """
        return self.value.select(negated=negated, compact=compact, temporal=temporal)

    @classmethod
    def from_callable(cls, function) -> Operators:
        """
        Map a Python ``operator`` module callable to the matching :class:`Operators` member.

        :param function: A callable from the ``operator`` module (e.g. ``operator.gt``,
            ``operator.eq``) or the custom ``not_contains`` comparator.
        :return: The corresponding :class:`Operators` member.
        :rtype: Operators
        :raises KeyError: If *function* has no registered mapping.
        """
        return _OPERATOR_CALLABLE_MAP[function]


#: Map Python ``operator`` callables to :class:`Operators` members.
#: Built once at module load time — avoids rebuilding on every call.
_OPERATOR_CALLABLE_MAP: dict = {
    _operator.eq: Operators.EQ,
    _operator.ne: Operators.NE,
    _operator.lt: Operators.LT,
    _operator.le: Operators.LE,
    _operator.gt: Operators.GT,
    _operator.ge: Operators.GE,
    _operator.contains: Operators.CONTAINS,
    _nc: Operators.NOT_CONTAINS,
}
