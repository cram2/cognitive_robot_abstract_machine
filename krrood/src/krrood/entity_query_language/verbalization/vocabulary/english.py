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


class Conjunctions(VocabEnum):
    """Coordinating conjunctions (AND, OR)."""

    AND = PlainWord("and")
    OR = PlainWord("or")


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
