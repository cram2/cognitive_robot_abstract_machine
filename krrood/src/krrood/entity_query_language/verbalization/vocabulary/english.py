from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass
from enum import Enum

from typing_extensions import Callable, Dict, List, Optional

from krrood.entity_query_language.operators.comparator import not_contains

from krrood.entity_query_language.verbalization import morphology

from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    PhraseFragment,
    RoleFragment,
    Fragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.features import Spacing
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
    PunctuationWord,
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

    def build_phrase(
        self, type_name: str, referent_id: Optional[uuid.UUID] = None
    ) -> Fragment:
        """
        Build *"there's a/an <type_name>"*.

        :param type_name: English noun in singular form (e.g. ``"Robot"``, ``"Apple"``).
        :param referent_id: When given, the noun is a referring noun phrase for that entity.
        :return: Phrase fragment with the correct indefinite article.
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

    def build_phrase(self, type_name: str) -> Fragment:
        """
        Build *"there are <plural_type_name>"*.

        :param type_name: English noun in singular form; pluralised automatically.
        :return: Phrase fragment with pluralised type name.
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

    Provides both a singular and a plural fragment.
    """

    def plural_fragment(self) -> WordFragment:
        """:return: A noun fragment tagged plural (e.g. ``"entity"`` → ``"entities"``)."""
        return WordFragment(text=self.text, number=Number.PLURAL)


@dataclass(frozen=True)
class CommonGroupKeyWord(PlainWord):
    """
    Group-key binding phrase: *"the common <field> of the <plural_root>"*.
    """

    def build_phrase(self, field_name: str, root: str) -> Fragment:
        """
        Build *"the common <field_name> of the <plural root>"*.

        :param field_name: Name of the grouped field (e.g. ``"location"``).
        :param root: Singular root type name (e.g. ``"Robot"``); pluralised in the phrase.
        :return: Phrase fragment.
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
    SUCH_THAT = KeyWord("such that")
    GIVEN_THAT = KeyWord("given that")
    WHERE = KeyWord("where")
    WHOSE = KeyWord("whose")
    GROUPED_BY = KeyWord("grouped by")
    GROUPED = KeyWord("grouped")
    HAVING = KeyWord("having")
    ORDERED_BY = KeyWord("ordered by")
    PREDICT = KeyWord("predict")
    RESPECTIVELY = KeyWord("respectively")
    TRUE = KeyWord("true")


class Directive(VocabEnum):
    """The imperative verb that opens a request: *"Find"* a match in the domain, or *"Generate"*
    an underspecified one."""

    FIND = KeyWord("Find")
    GENERATE = KeyWord("Generate")

    @classmethod
    def for_underspecified(cls, underspecified: bool) -> "Directive":
        """:return: ``GENERATE`` for an underspecified (generative) request, else ``FIND``."""
        return cls.GENERATE if underspecified else cls.FIND


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

    @property
    def has_child(self) -> bool:
        """:return: ``True`` when the aggregation takes a complement (*"sum of <x>"*); ``False``
        for a childless aggregate (*"count of all"*)."""
        return self.value.child_form is not ChildForm.NONE

    @property
    def child_number(self) -> Number:
        """:return: The grammatical number the complement is rendered in — plural after an *"… of"*
        word (*"sum of amounts"*), singular otherwise (*"maximum … amount"*). This is the lexical
        fact callers used to recompute from ``child_form``."""
        return (
            Number.PLURAL
            if self.value.child_form is ChildForm.PLURAL
            else Number.SINGULAR
        )

    def complement(self, child: Fragment) -> List[Fragment]:
        """
        :param child: The already-rendered complement, built at :attr:`child_number`.
        :return: The complement modifiers — a leading *"of"* when the word needs one
            (*"maximum of …"*), else the child alone (an *"… of"* word carries its own *"of"*);
            empty for a childless aggregate.
        """
        if self.value.child_form is ChildForm.NONE:
            return []
        if self.value.child_form is ChildForm.SINGULAR_OF:
            return [Prepositions.OF.as_fragment(), child]
        return [child]


class Copulas(VocabEnum):
    """Copula verbs (IS, IS NOT, ARE) — role is OPERATOR so they share colouring with comparators."""

    IS = OperatorWord("is")
    IS_NOT = OperatorWord("is not")
    ARE = OperatorWord("are")
    ARE_NOT = OperatorWord("are not")

    @classmethod
    def for_number(cls, number: Number) -> RoleFragment:
        """:return: The copula tagged with *number* for a directly-built leaf; the morphology
        pass agrees it (*is* / *are*)."""
        return RoleFragment(text=cls.IS.text, role=SemanticRole.OPERATOR, number=number)


class Prepositions(VocabEnum):
    """Prepositions used in possessive path phrases (OF, OF THE) and aggregate scopes (AMONG)."""

    OF = PlainWord("of")
    OF_THE = PlainWord("of the")
    AMONG = PlainWord("among")
    WITH = PlainWord("with")


#: The closed-class inventory of common English prepositions, as a lexical datum (distinct from the
#: :class:`Prepositions` enum, which holds the few prepositions the verbalizer *emits*). Prepositions
#: are a fixed, finite class — no inflection library or POS tagger is needed (or appropriate) to
#: enumerate them. Recognizers consult this to read a relation off a field name's final token.
ENGLISH_PREPOSITIONS = frozenset(
    {
        "about",
        "above",
        "across",
        "after",
        "against",
        "along",
        "among",
        "around",
        "at",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "between",
        "beyond",
        "by",
        "for",
        "from",
        "in",
        "inside",
        "into",
        "near",
        "of",
        "off",
        "on",
        "onto",
        "out",
        "outside",
        "over",
        "past",
        "per",
        "through",
        "to",
        "toward",
        "towards",
        "under",
        "underneath",
        "until",
        "up",
        "upon",
        "via",
        "with",
        "within",
        "without",
    }
)


class Conjunctions(VocabEnum):
    """Coordinating conjunctions (AND, OR)."""

    AND = PlainWord("and")
    OR = PlainWord("or")


class Absence(VocabEnum):
    """Absence-of-attribute verb — the *"has no"* / *"have no"* of *"the Pose has no orientation"*,
    produced for an owned attribute compared ``== None``. Number-agreeing, but selected explicitly
    (the morphology pass only agrees the copula), so the right member is chosen here."""

    HAS_NO = OperatorWord("has no")
    HAVE_NO = OperatorWord("have no")

    @classmethod
    def for_number(cls, number: Number) -> "Absence":
        """:return: ``HAVE_NO`` for a plural owner, else ``HAS_NO``."""
        return cls.HAVE_NO if number is Number.PLURAL else cls.HAS_NO


class NonExistence(VocabEnum):
    """Non-existence verb — the *"does not exist"* / *"do not exist"* of *"the Robot does not
    exist"*, produced for a bare variable compared ``== None`` (no attribute to name).
    """

    DOES_NOT_EXIST = OperatorWord("does not exist")
    DO_NOT_EXIST = OperatorWord("do not exist")

    @classmethod
    def for_number(cls, number: Number) -> "NonExistence":
        """:return: ``DO_NOT_EXIST`` for a plural subject, else ``DOES_NOT_EXIST``."""
        return cls.DO_NOT_EXIST if number is Number.PLURAL else cls.DOES_NOT_EXIST


class PassiveAbsence(VocabEnum):
    """Passive absence verb for a *relational* (past-participle) attribute — the *"has not been"* /
    *"have not been"* of *"a Mission has not been assigned to any Robot"*, produced for a
    ``<participle>_<preposition>`` attribute compared ``== None``. Number-agreeing, selected
    explicitly (the morphology pass only agrees the copula)."""

    HAS_NOT_BEEN = OperatorWord("has not been")
    HAVE_NOT_BEEN = OperatorWord("have not been")

    @classmethod
    def for_number(cls, number: Number) -> "PassiveAbsence":
        """:return: ``HAVE_NOT_BEEN`` for a plural owner, else ``HAS_NOT_BEEN``."""
        return cls.HAVE_NOT_BEEN if number is Number.PLURAL else cls.HAS_NOT_BEEN


class Quantifiers(VocabEnum):
    """Indefinite quantifier for the object of a passive absence — the *"any"* of *"… assigned to
    any Robot"*, or the bare *"anything"* when the related type is not a nameable class.
    """

    ANY = PlainWord("any")
    ANYTHING = PlainWord("anything")


class SetMembership(VocabEnum):
    """Membership phrase for a domain-constrained value variable — the *"one of"* of *"one of
    OPTION_A, OPTION_B, or OPTION_C"*."""

    ONE_OF = PlainWord("one of")


class Specificity(VocabEnum):
    """Pre-head marking a concrete object literal as a specific instance (its identity, not its
    repr) — the *"specific"* of *"a specific Body"*."""

    SPECIFIC = PlainWord("specific")


class Punctuation(VocabEnum):
    """Structural punctuation tokens — role-less, like the brackets around a tuple (*"(v1, v2)"*)
    and the comma in a coordinated list (*"a, b, or c"*)."""

    COMMA = PunctuationWord(",", spacing=Spacing.LEFT)
    OPEN_PAREN = PunctuationWord("(", spacing=Spacing.RIGHT)
    CLOSE_PAREN = PunctuationWord(")", spacing=Spacing.LEFT)


class Pronouns(VocabEnum):
    """Coreference pronouns standing in for a previously introduced variable."""

    ITS = PronounWord("its")
    THEIR = PronounWord("their")


class RangePhrases(VocabEnum):
    """
    Range (``between``) operator phrases produced when a lower-bound and an upper-bound
    comparison on the same attribute are folded together.
    """

    IS_BETWEEN = OperatorWord("is between")
    """Standard form keeping the copula (*"is between"*)."""
    BETWEEN = OperatorWord("between")
    """Copula-less form for HAVING / post-nominal use (*"between"*)."""


class CoindexedOperators(VocabEnum):
    """Plural copular operator phrases for a factored co-indexed comparison — the *"are equal to"*
    of *"the month and year of A are equal to those of B"*. Plural because the coordinated
    terminals (*"month and year"*) are the grammatical subject."""

    EQ = OperatorWord("are equal to")
    GT = OperatorWord("are greater than")
    LT = OperatorWord("are less than")
    GE = OperatorWord("are at least")
    LE = OperatorWord("are at most")


class CoindexedPhrases(VocabEnum):
    """Fixed phrases for the factored co-indexed comparison rendering: the *"those of"* anaphor of
    the faithful form, and the *"have the same"* verb of the natural equality form."""

    THOSE_OF = PlainWord("those of")
    HAVE_THE_SAME = PlainWord("have the same")


class SortDirections(VocabEnum):
    """Sort direction words for ORDERED BY clauses."""

    ASCENDING = PlainWord("ascending")
    DESCENDING = PlainWord("descending")


class RankingWords(VocabEnum):
    """The qualifier words a ``limit`` (+ ordering) puts on the selection — *"the first two"*,
    *"the top three"*, *"the lowest"*, *"… by salary"*."""

    FIRST = PlainWord("first")
    """No ordering — the first *n* in natural order."""
    TOP = PlainWord("top")
    """Descending order, *n > 1* (*"the top three …"*)."""
    BOTTOM = PlainWord("bottom")
    """Ascending order, *n > 1* (*"the bottom three …"*)."""
    HIGHEST = PlainWord("highest")
    """Descending order, *n = 1* (*"the highest …"* / *"with the highest …"*)."""
    LOWEST = PlainWord("lowest")
    """Ascending order, *n = 1* (*"the lowest …"* / *"with the lowest …"*)."""
    BY = PlainWord("by")
    """Introduces the order key for a plural attribute ranking (*"… by salary"*)."""


class Articles(VocabEnum):
    """
    Definite articles (THE, THE UNIQUE) and a static helper for indefinite articles.
    """

    THE = PlainWord("the")
    THE_UNIQUE = PlainWord("the unique")

    @staticmethod
    def indefinite(following_word: str) -> WordFragment:
        """
        :param following_word: The word immediately following the article.
        :return: A word fragment containing *"a"* or *"an"* based on the phonological context of
            *following_word* — ``WordFragment("a")`` or ``WordFragment("an")``.
        """
        text = morphology.indefinite_article(following_word) if following_word else "a"
        return WordFragment(text=text)


class ExistentialPhrase(VocabEnum):
    """
    Parameterised existential phrases (*"there's a/an TypeName"*, *"there are TypeNames"*).
    """

    THERE_IS_A = SingularExistential("there's")
    THERE_ARE = PluralExistential("there are")

    @classmethod
    def for_number(cls, number: Number) -> ExistentialPhrase:
        """:return: The existential frame agreeing with *number*: ``THERE_ARE`` / ``THERE_IS_A``."""
        return cls.THERE_ARE if number is Number.PLURAL else cls.THERE_IS_A

    def build_phrase(
        self, type_name: str, referent_id: Optional[uuid.UUID] = None
    ) -> Fragment:
        """
        Build the existential phrase for *type_name*.

        :param type_name: English noun in singular form.
        :param referent_id: Optional referent for coreference (singular only).
        :return: Existential phrase fragment.
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
        """:return: A pluralised noun fragment (e.g. ``"entities"``)."""
        return self.value.plural_fragment()


class GroupKeyPhrases(VocabEnum):
    """
    Group-key binding phrases (*"the common <field> of the <plural_root>"*).
    """

    COMMON_OF = CommonGroupKeyWord("the common")

    def build_phrase(self, field_name: str, plural_root: str) -> Fragment:
        """
        Build the group-key phrase.

        :param field_name: Grouped field name.
        :param plural_root: Plural root type name.
        :return: Group-key phrase fragment.
        """
        return self.value.build_phrase(field_name, plural_root)


class Operators(Enum):
    """
    Comparison operator phrases.

    Each member's value holds all eight text variants for one operator.
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
        Select the operator word for the given flag combination.

        :param negated: Use the negated text variant.
        :param compact: Use the copula-less variant for HAVING clauses.
        :param temporal: Use the temporal variant for datetime comparisons.
        :return: Selected operator word.
        """
        return self.value.select(negated=negated, compact=compact, temporal=temporal)

    @classmethod
    def from_callable(cls, function) -> Operators:
        """
        Map a Python ``operator`` module callable to the matching member.

        :param function: A callable from the ``operator`` module (e.g. ``operator.gt``,
            ``operator.eq``) or the custom ``not_contains`` comparator.
        :return: The corresponding member.
        :raises KeyError: If *function* has no registered mapping.
        """
        return _OPERATOR_CALLABLE_MAP[function]


#: Map Python ``operator`` callables to ``Operators`` members.
_OPERATOR_CALLABLE_MAP: Dict[Callable, Operators] = {
    operator.eq: Operators.EQ,
    operator.ne: Operators.NE,
    operator.lt: Operators.LT,
    operator.le: Operators.LE,
    operator.gt: Operators.GT,
    operator.ge: Operators.GE,
    operator.contains: Operators.CONTAINS,
    not_contains: Operators.NOT_CONTAINS,
}
