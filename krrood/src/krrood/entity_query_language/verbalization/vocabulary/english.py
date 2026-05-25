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

import inflect

from krrood.entity_query_language.verbalization.fragments.base import (
    PhraseFragment,
    RoleFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.utils import _ensure_plural
from krrood.entity_query_language.verbalization.vocabulary.words import (
    AggregationWord,
    ChildForm,
    KeyWord,
    LogicalWord,
    OperatorPhrase,
    OperatorWord,
    PlainWord,
    VocabEnum,
)

_engine = inflect.engine()


# ── English-specific word subtypes ─────────────────────────────────────────────
# These add behaviour for phrases that depend on a runtime type_name argument.


@dataclass(frozen=True)
class SingularExistential(PlainWord):
    """
    Parameterised existential phrase: *"there's a/an TypeName"*.

    The article (*a* / *an*) is computed phonologically at call time using the
    ``inflect`` library.

    :ivar text: Fixed prefix (e.g. ``"there's"``).
    """

    def build_phrase(self, type_name: str) -> VerbFragment:
        """
        Build *"there's a/an <type_name>"* as a
        :class:`~krrood.entity_query_language.verbalization.fragments.base.PhraseFragment`.

        :param type_name: English noun in singular form (e.g. ``"Robot"``, ``"Apple"``).
        :type type_name: str
        :returns: Phrase fragment with the correct indefinite article.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        article = _engine.a(type_name).split()[0]
        return PhraseFragment(
            parts=[
                WordFragment(text=f"{self.text} {article}"),
                RoleFragment(text=type_name, role=SemanticRole.VARIABLE),
            ],
            separator=" ",
        )


@dataclass(frozen=True)
class PluralExistential(PlainWord):
    """
    Parameterised existential phrase: *"there are TypeNames"*.

    :ivar text: Fixed prefix (e.g. ``"there are"``).
    """

    def build_phrase(self, type_name: str) -> VerbFragment:
        """
        Build *"there are <plural_type_name>"* as a
        :class:`~krrood.entity_query_language.verbalization.fragments.base.PhraseFragment`.

        :param type_name: English noun in singular form; pluralised automatically.
        :type type_name: str
        :returns: Phrase fragment with pluralised type name.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        return PhraseFragment(
            parts=[
                self.as_fragment(),
                RoleFragment(text=_ensure_plural(type_name), role=SemanticRole.VARIABLE),
            ],
            separator=" ",
        )


@dataclass(frozen=True)
class FallbackNounWord(PlainWord):
    """
    A fallback noun used when no type information is available from the expression.

    Provides both :meth:`as_fragment` (singular) and :meth:`plural_fragment`.

    :ivar text: Singular English noun (e.g. ``"entity"``).
    """

    def plural_fragment(self) -> WordFragment:
        """
        Return a :class:`~krrood.entity_query_language.verbalization.fragments.base.WordFragment`
        with the pluralised form.

        :returns: Pluralised noun fragment (e.g. ``"entities"``).
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.WordFragment
        """
        return WordFragment(text=_engine.plural(self.text))


@dataclass(frozen=True)
class CommonGroupKeyWord(PlainWord):
    """
    Group-key binding phrase: *"the common <field> of the <plural_root>"*.

    Used in the THEN clause when a consequent binding refers to a GROUP BY key.

    :ivar text: Fixed prefix (e.g. ``"the common"``).
    """

    def build_phrase(self, field_name: str, plural_root: str) -> VerbFragment:
        """
        Build *"the common <field_name> of the <plural_root>"*.

        :param field_name: Name of the grouped field (e.g. ``"location"``).
        :type field_name: str
        :param plural_root: Plural root type name (e.g. ``"Robots"``).
        :type plural_root: str
        :returns: Phrase fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        return PhraseFragment(
            parts=[
                self.as_fragment(),
                WordFragment(text=field_name),
                Prepositions.OF_THE.as_fragment(),
                WordFragment(text=plural_root),
            ],
            separator=" ",
        )


# ── Namespace Enums ────────────────────────────────────────────────────────────


class Keywords(VocabEnum):
    """EQL structural keywords (IF, THEN, FIND, WHERE, etc.)."""

    IF           = KeyWord("If")
    THEN         = KeyWord("then")
    FIND         = KeyWord("Find")
    FIND_SETS_OF = KeyWord("Find sets of")
    SUCH_THAT    = KeyWord("such that")
    WHERE        = KeyWord("where")
    WHOSE        = KeyWord("whose")
    GROUPED_BY   = KeyWord("grouped by")
    GROUPED      = KeyWord("grouped")
    HAVING       = KeyWord("having")
    ORDERED_BY   = KeyWord("ordered by")
    TRUE         = KeyWord("true")


class Logicals(VocabEnum):
    """Logical connectives (NOT, FOR ALL, THERE EXISTS, EITHER)."""

    NOT          = LogicalWord("not")
    EITHER       = LogicalWord("either")
    FOR_ALL      = LogicalWord("for all")
    THERE_EXISTS = LogicalWord("there exists")


class Aggregations(VocabEnum):
    """Aggregation function phrases (number of, sum of, average of, etc.)."""

    COUNT      = AggregationWord("number of")
    COUNT_ALL  = AggregationWord("count of all")
    SUM        = AggregationWord("sum of")
    AVERAGE    = AggregationWord("average of")
    MAX        = AggregationWord("maximum",   child_form=ChildForm.SINGULAR_OF)
    MIN        = AggregationWord("minimum",   child_form=ChildForm.SINGULAR_OF)
    MODE       = AggregationWord("mode of")
    MULTI_MODE = AggregationWord("all modes of")


class Copulas(VocabEnum):
    """Copula verbs (IS, IS NOT, ARE). Role is OPERATOR — copulas appear alongside comparators visually."""

    # role = OPERATOR — copulas appear alongside comparison operators visually
    IS     = OperatorWord("is")
    IS_NOT = OperatorWord("is not")
    ARE    = OperatorWord("are")


class Prepositions(VocabEnum):
    """Prepositions used in possessive path phrases (OF, OF THE) and aggregate scopes (AMONG)."""

    OF     = PlainWord("of")
    OF_THE = PlainWord("of the")
    AMONG  = PlainWord("among")


class Conjunctions(VocabEnum):
    """Coordinating conjunctions (AND, OR)."""

    AND = PlainWord("and")
    OR  = PlainWord("or")


class SortDirections(VocabEnum):
    """Sort direction words for ORDERED BY clauses."""

    ASCENDING  = PlainWord("ascending")
    DESCENDING = PlainWord("descending")


class Articles(VocabEnum):
    """
    Definite articles (THE, THE UNIQUE) and a static helper for indefinite articles.
    """

    THE        = PlainWord("the")
    THE_UNIQUE = PlainWord("the unique")

    @staticmethod
    def indefinite(following_word: str) -> "WordFragment":
        """
        Return a :class:`~krrood.entity_query_language.verbalization.fragments.base.WordFragment`
        containing *"a"* or *"an"* based on the phonological context of *following_word*.

        :param following_word: The word immediately following the article.
        :type following_word: str
        :returns: ``WordFragment("a")`` or ``WordFragment("an")``.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.WordFragment
        """
        text = _engine.a(following_word).split()[0] if following_word else "a"
        return WordFragment(text=text)


class ExistentialPhrase(VocabEnum):
    """
    Parameterised existential phrases (*"there's a/an TypeName"*, *"there are TypeNames"*).

    Call :meth:`build_phrase` on a member to produce the full fragment.
    """

    THERE_IS_A = SingularExistential("there's")
    THERE_ARE  = PluralExistential("there are")

    def build_phrase(self, type_name: str) -> VerbFragment:
        """
        Delegate to the underlying :class:`SingularExistential` or :class:`PluralExistential`.

        :param type_name: English noun in singular form.
        :type type_name: str
        :returns: Existential phrase fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        return self.value.build_phrase(type_name)


class FallbackNouns(VocabEnum):
    """
    Fallback noun used when no type information is available (e.g. *"entity"* / *"entities"*).
    """

    ENTITY = FallbackNounWord("entity")

    def plural_fragment(self) -> "WordFragment":
        """
        Return a plural :class:`~krrood.entity_query_language.verbalization.fragments.base.WordFragment`.

        :returns: Pluralised noun fragment (e.g. ``"entities"``).
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
        :returns: Group-key phrase fragment.
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
        standard="is",           compact="equals",
        negated="is not",        negated_compact="does not equal",
        temporal="is at",        temporal_compact="at",
        temporal_negated="is not at", temporal_negated_compact="not at",
    )
    NE = OperatorPhrase(
        standard="is not",       compact="does not equal",
        negated="is",            negated_compact="equals",
        temporal="is not at",    temporal_compact="not at",
        temporal_negated="is at", temporal_negated_compact="at",
    )
    LT = OperatorPhrase(
        standard="is less than",     compact="less than",
        negated="is not less than",  negated_compact="not less than",
        temporal="is before",        temporal_compact="before",
        temporal_negated="is no earlier than", temporal_negated_compact="no earlier than",
    )
    LE = OperatorPhrase(
        standard="is at most",   compact="at most",
        negated="is not at most", negated_compact="not at most",
        temporal="is no later than",  temporal_compact="no later than",
        temporal_negated="is after",  temporal_negated_compact="after",
    )
    GT = OperatorPhrase(
        standard="is greater than",     compact="greater than",
        negated="is not greater than",  negated_compact="not greater than",
        temporal="is after",            temporal_compact="after",
        temporal_negated="is no later than", temporal_negated_compact="no later than",
    )
    GE = OperatorPhrase(
        standard="is at least",   compact="at least",
        negated="is not at least", negated_compact="not at least",
        temporal="is no earlier than",  temporal_compact="no earlier than",
        temporal_negated="is before",   temporal_negated_compact="before",
    )
    CONTAINS = OperatorPhrase(
        standard="contains",          compact="contains",
        negated="does not contain",   negated_compact="does not contain",
    )
    NOT_CONTAINS = OperatorPhrase(
        standard="does not contain",  compact="does not contain",
        negated="contains",           negated_compact="contains",
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
        :returns: Selected :class:`~krrood.entity_query_language.verbalization.vocabulary.words.OperatorWord`.
        :rtype: ~krrood.entity_query_language.verbalization.vocabulary.words.OperatorWord
        """
        return self.value.select(negated=negated, compact=compact, temporal=temporal)

    @classmethod
    def from_callable(cls, fn) -> "Operators":
        """
        Map a Python ``operator`` module callable to the matching :class:`Operators` member.

        :param fn: A callable from the ``operator`` module (e.g. ``operator.gt``,
            ``operator.eq``) or the custom ``not_contains`` comparator.
        :returns: The corresponding :class:`Operators` member.
        :rtype: Operators
        :raises KeyError: If *fn* has no registered mapping.
        """
        from krrood.entity_query_language.operators.comparator import not_contains as _nc
        _MAP = {
            _operator.eq:       cls.EQ,
            _operator.ne:       cls.NE,
            _operator.lt:       cls.LT,
            _operator.le:       cls.LE,
            _operator.gt:       cls.GT,
            _operator.ge:       cls.GE,
            _operator.contains: cls.CONTAINS,
            _nc:                cls.NOT_CONTAINS,
        }
        return _MAP[fn]
