from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from typing_extensions import (
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    Type,
)

from krrood.entity_query_language.core.base_expressions import Selectable
from krrood.entity_query_language.operators.aggregators import (
    Aggregator,
    Average,
    Count,
    CountAll,
    Max,
    Min,
    Mode,
    MultiMode,
    Sum,
)
from krrood.entity_query_language.operators.comparator import not_contains

from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.exceptions import UnknownAggregatorError

from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    PhraseFragment,
    RoleFragment,
    VerbalizationFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.features import Spacing
from krrood.entity_query_language.verbalization.vocabulary.words import (
    AggregationWord,
    KeyWord,
    LogicalWord,
    GrammaticalNumber,
    OperatorPhrase,
    OperatorWord,
    PlainWord,
    PronounWord,
    PunctuationWord,
    VocabEnum,
)

# %% English-specific word subtypes
# These add behaviour for phrases that depend on a runtime type_name argument.


@dataclass(frozen=True)
class SingularExistential(PlainWord):
    """
    Parameterised existential phrase: *"there's a/an TypeName"*.

    The article (*a* / *an*) is computed phonologically at call time
    using the ``inflect`` library.
    """

    def build_phrase(
        self, type_name: str, referent_id: Optional[uuid.UUID] = None
    ) -> VerbalizationFragment:
        """
        Build *"there's a/an <type_name>"*.

        :param type_name: English noun in singular form (e.g. ``"Robot"``, ``"Apple"``).
        :param referent_id: When given, the noun is a referring noun phrase for that entity.
        :return: Phrase fragment with the correct indefinite article.

        >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
        >>> flatten_fragment_to_plain_text(SingularExistential("there's").build_phrase("Apple"))
        "there's an Apple"
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

    def build_phrase(self, type_name: str) -> VerbalizationFragment:
        """
        Build *"there are <plural_type_name>"*.

        The noun is tagged :attr:`GrammaticalNumber.PLURAL`; its surface inflection is realised by the later
        morphology pass.

        :param type_name: English noun in singular form; pluralised automatically.
        :return: Phrase fragment with pluralised type name.

        >>> PluralExistential("there are").build_phrase("Robot").parts[1].number
        <GrammaticalNumber.PLURAL: 'plural'>
        """
        return PhraseFragment(
            parts=[
                self.as_fragment(),
                RoleFragment(
                    text=type_name,
                    role=SemanticRole.VARIABLE,
                    number=GrammaticalNumber.PLURAL,
                ),
            ],
        )


@dataclass(frozen=True)
class FallbackNounWord(PlainWord):
    """
    A fallback noun used when no type information is available from the
    expression.

    Provides both a singular and a plural fragment.
    """

    def plural_fragment(self) -> WordFragment:
        """:return: A noun fragment tagged plural (e.g. ``"entity"`` → ``"entities"``).

        >>> FallbackNounWord("entity").plural_fragment().number
        <GrammaticalNumber.PLURAL: 'plural'>
        """
        return WordFragment(text=self.text, number=GrammaticalNumber.PLURAL)


@dataclass(frozen=True)
class CommonGroupKeyWord(PlainWord):
    """
    Group-key binding phrase: *"the common <field> of the <plural_root>"*.
    """

    def build_phrase(self, field_name: str, root: str) -> VerbalizationFragment:
        """
        Build *"the common <field_name> of the <plural root>"*.

        :param field_name: Name of the grouped field (e.g. ``"location"``).
        :param root: Singular root type name (e.g. ``"Robot"``); pluralised in the phrase.
        :return: Phrase fragment.

        >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
        >>> flatten_fragment_to_plain_text(CommonGroupKeyWord("the common").build_phrase("location", "Robot"))
        'the common location of the Robot'
        """
        return PhraseFragment(
            parts=[
                self.as_fragment(),
                WordFragment(text=field_name),
                Prepositions.OF.as_fragment(),
                Articles.THE.as_fragment(),
                WordFragment(text=root, number=GrammaticalNumber.PLURAL),
            ],
        )


# %% Namespace Enums


class Keywords(VocabEnum):
    """
    EQL structural keywords (IF, THEN, FIND, WHERE, etc.).
    """

    IF = KeyWord("If")
    THEN = KeyWord("then")
    FIND = KeyWord("Find")
    REPORT = KeyWord("report")
    FOR = KeyWord("for")
    FOR_EACH = KeyWord("For each")
    SUCH_THAT = KeyWord("such that")
    GIVEN_THAT = KeyWord("given that")
    WHERE = KeyWord("where")
    WHOSE = KeyWord("whose")
    WHICH = KeyWord("which")
    WHO = KeyWord("who")
    THAT = KeyWord("that")
    GROUPED_BY = KeyWord("grouped by")
    HAVING = KeyWord("having")
    ORDERED_BY = KeyWord("ordered by")
    PREDICT = KeyWord("predict")
    RESPECTIVELY = KeyWord("respectively")
    TRUE = KeyWord("true")


class Directive(VocabEnum):
    """
    The imperative verb that opens a request: *"Find"* a match in the domain,
    or *"Generate"* an underspecified one.
    """

    FIND = KeyWord("Find")
    GENERATE = KeyWord("Generate")


class Logicals(VocabEnum):
    """
    Logical connectives (NOT, FOR ALL, THERE EXISTS, EITHER).
    """

    NOT = LogicalWord("not")
    EITHER = LogicalWord("either")
    FOR_ALL = LogicalWord("for all")
    THERE_EXISTS = LogicalWord("there exists")


class Aggregations(VocabEnum):
    """
    Aggregation function phrases (number of, sum of, average of, etc.).
    """

    COUNT = AggregationWord("number")
    COUNT_ALL = AggregationWord("count of all", child_form=None)
    SUM = AggregationWord("sum")
    AVERAGE = AggregationWord("average")
    MAX = AggregationWord("maximum", child_form=GrammaticalNumber.SINGULAR)
    MIN = AggregationWord("minimum", child_form=GrammaticalNumber.SINGULAR)
    MODE = AggregationWord("mode")
    MULTI_MODE = AggregationWord("all modes")

    @classmethod
    def for_aggregator(cls, aggregator_type: Type[Aggregator]) -> Aggregations:
        """:return: The aggregation phrase that verbalizes an aggregator type.

        :raises UnknownAggregatorError: When the aggregator type has no phrase.

        >>> from krrood.entity_query_language.operators.aggregators import Sum
        >>> Aggregations.for_aggregator(Sum) is Aggregations.SUM
        True
        """
        phrase = _AGGREGATOR_PHRASES.get(aggregator_type)
        if phrase is None:
            raise UnknownAggregatorError(aggregator_type=aggregator_type)
        return phrase

    @property
    def has_child(self) -> bool:
        """:return: ``True`` when the aggregation takes a complement (*"sum of <x>"*); ``False``
        for a childless aggregate (*"count of all"*).

        >>> Aggregations.SUM.has_child
        True
        >>> Aggregations.COUNT_ALL.has_child
        False
        """
        return self.value.child_form is not None

    @property
    def child_number(self) -> GrammaticalNumber:
        """:return: The grammatical number the complement is rendered in — plural for a
        population (*"sum of amounts"*), singular otherwise (*"maximum of the amount"*).

        >>> Aggregations.SUM.child_number
        <GrammaticalNumber.PLURAL: 'plural'>
        >>> Aggregations.MAX.child_number
        <GrammaticalNumber.SINGULAR: 'singular'>
        """
        return self.value.child_form or GrammaticalNumber.SINGULAR

    def complement(self, child: VerbalizationFragment) -> List[VerbalizationFragment]:
        """
        :param child: The already-rendered complement, built at :attr:`child_number`.
        :return: The complement modifiers — *"of"* then the child — so the bare aggregation noun
            stays the phrase head (a repeat reduces to *"the sum"*); empty for a childless aggregate.

        >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
        >>> [flatten_fragment_to_plain_text(part)
        ...  for part in Aggregations.SUM.complement(WordFragment(text="amounts"))]
        ['of', 'amounts']
        >>> Aggregations.COUNT_ALL.complement(WordFragment(text="amounts"))
        []
        """
        if self.value.child_form is None:
            return []
        return [Prepositions.OF.as_fragment(), child]

    def compact_complement(
        self, leaf: VerbalizationFragment
    ) -> List[VerbalizationFragment]:
        """:return: The complement for the *compact* value form, where the leaf attaches directly to
        the aggregation noun — *"of <plural leaf>"* for a population (*"the sum of amounts"*), the
        bare singular leaf as a noun modifier otherwise (*"the maximum amount"*); empty for a
        childless aggregate.

        >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
        >>> [flatten_fragment_to_plain_text(part)
        ...  for part in Aggregations.SUM.compact_complement(WordFragment(text="amounts"))]
        ['of', 'amounts']
        >>> [flatten_fragment_to_plain_text(part)
        ...  for part in Aggregations.MAX.compact_complement(WordFragment(text="amount"))]
        ['amount']
        """
        if self.value.child_form is GrammaticalNumber.PLURAL:
            return [Prepositions.OF.as_fragment(), leaf]
        return [leaf]


#: Maps each standard aggregator subtype to its lexicon phrase; consulted by
#: :meth:`Aggregations.for_aggregator`.
_AGGREGATOR_PHRASES: Dict[Type[Aggregator], Aggregations] = {
    Count: Aggregations.COUNT,
    CountAll: Aggregations.COUNT_ALL,
    Sum: Aggregations.SUM,
    Average: Aggregations.AVERAGE,
    Max: Aggregations.MAX,
    Min: Aggregations.MIN,
    Mode: Aggregations.MODE,
    MultiMode: Aggregations.MULTI_MODE,
}


class Copulas(VocabEnum):
    """
    Copula verbs (IS, IS NOT, ARE) — role is OPERATOR so they share colouring
    with comparators.
    """

    IS = OperatorWord("is")
    IS_NOT = OperatorWord("is not")
    ARE = OperatorWord("are")
    ARE_NOT = OperatorWord("are not")

    @classmethod
    def for_number(cls, number: GrammaticalNumber) -> RoleFragment:
        """:return: The copula tagged with *number* for a directly-built leaf; the morphology
        pass agrees it (*is* / *are*).

        >>> Copulas.for_number(GrammaticalNumber.PLURAL).number
        <GrammaticalNumber.PLURAL: 'plural'>
        """
        return RoleFragment(text=cls.IS.text, role=SemanticRole.OPERATOR, number=number)


class Prepositions(VocabEnum):
    """
    The closed-class inventory of English prepositions — the single preposition
    vocabulary.

    One enum serves every use of a preposition: the words the verbalizer *emits* (the genitive
    *"of"*, the aggregate scope *"among"*, the comitative *"with"*), the prepositions a clause links
    its constituents with (*"works **in** a department"*), and the lexicon a recognizer consults to
    read a relation off a field name's final token (*assigned_**to***, *owned_**by***). Prepositions
    are a fixed, finite class, so the inventory is enumerated in full rather than inferred — no
    inflection library or part-of-speech tagger is needed (or appropriate).

    ..note:: Only simple (one-word) prepositions are members. Complex prepositions (*"in front of"*,
        *"according to"*) are multi-token, so they could not be matched against a single snake_case
        field token, and the verbalizer emits none of them.
    """

    ABOARD = PlainWord("aboard")
    ABOUT = PlainWord("about")
    ABOVE = PlainWord("above")
    ACROSS = PlainWord("across")
    AFTER = PlainWord("after")
    AGAINST = PlainWord("against")
    ALONG = PlainWord("along")
    ALONGSIDE = PlainWord("alongside")
    AMID = PlainWord("amid")
    AMIDST = PlainWord("amidst")
    AMONG = PlainWord("among")
    AMONGST = PlainWord("amongst")
    AROUND = PlainWord("around")
    AS = PlainWord("as")
    ASTRIDE = PlainWord("astride")
    AT = PlainWord("at")
    ATOP = PlainWord("atop")
    BARRING = PlainWord("barring")
    BEFORE = PlainWord("before")
    BEHIND = PlainWord("behind")
    BELOW = PlainWord("below")
    BENEATH = PlainWord("beneath")
    BESIDE = PlainWord("beside")
    BESIDES = PlainWord("besides")
    BETWEEN = PlainWord("between")
    BEYOND = PlainWord("beyond")
    BUT = PlainWord("but")
    BY = PlainWord("by")
    CONCERNING = PlainWord("concerning")
    CONSIDERING = PlainWord("considering")
    DESPITE = PlainWord("despite")
    DOWN = PlainWord("down")
    DURING = PlainWord("during")
    EXCEPT = PlainWord("except")
    EXCLUDING = PlainWord("excluding")
    FOLLOWING = PlainWord("following")
    FOR = PlainWord("for")
    FROM = PlainWord("from")
    GIVEN = PlainWord("given")
    IN = PlainWord("in")
    INCLUDING = PlainWord("including")
    INSIDE = PlainWord("inside")
    INTO = PlainWord("into")
    LIKE = PlainWord("like")
    MINUS = PlainWord("minus")
    NEAR = PlainWord("near")
    NOTWITHSTANDING = PlainWord("notwithstanding")
    OF = PlainWord("of")
    OFF = PlainWord("off")
    ON = PlainWord("on")
    ONTO = PlainWord("onto")
    OPPOSITE = PlainWord("opposite")
    OUT = PlainWord("out")
    OUTSIDE = PlainWord("outside")
    OVER = PlainWord("over")
    PAST = PlainWord("past")
    PENDING = PlainWord("pending")
    PER = PlainWord("per")
    PLUS = PlainWord("plus")
    REGARDING = PlainWord("regarding")
    RESPECTING = PlainWord("respecting")
    ROUND = PlainWord("round")
    SAVE = PlainWord("save")
    SINCE = PlainWord("since")
    THAN = PlainWord("than")
    THROUGH = PlainWord("through")
    THROUGHOUT = PlainWord("throughout")
    TILL = PlainWord("till")
    TO = PlainWord("to")
    TOWARD = PlainWord("toward")
    TOWARDS = PlainWord("towards")
    UNDER = PlainWord("under")
    UNDERNEATH = PlainWord("underneath")
    UNLIKE = PlainWord("unlike")
    UNTIL = PlainWord("until")
    UNTO = PlainWord("unto")
    UP = PlainWord("up")
    UPON = PlainWord("upon")
    VERSUS = PlainWord("versus")
    VIA = PlainWord("via")
    WITH = PlainWord("with")
    WITHIN = PlainWord("within")
    WITHOUT = PlainWord("without")
    WORTH = PlainWord("worth")

    @classmethod
    @lru_cache(maxsize=None)
    def texts(cls) -> FrozenSet[str]:
        """:return: the surface word of every preposition, for testing whether a token already reads
        as a preposition."""
        return frozenset(preposition.text for preposition in cls)


class Conjunctions(VocabEnum):
    """
    Coordinating conjunctions (AND, OR).
    """

    AND = PlainWord("and")
    OR = PlainWord("or")


class Absence(VocabEnum):
    """
    Absence-of-attribute verb — the *"has no"* / *"have no"* of *"the Pose has
    no orientation"*, produced for an owned attribute compared ``== None``.

    Number-agreeing, but selected explicitly (the morphology pass only
    agrees the copula), so the right member is chosen here.
    """

    HAS_NO = OperatorWord("has no")
    HAVE_NO = OperatorWord("have no")

    @classmethod
    def for_number(cls, number: GrammaticalNumber) -> "Absence":
        """:return: ``HAVE_NO`` for a plural owner, else ``HAS_NO``.

        >>> Absence.for_number(GrammaticalNumber.PLURAL).text
        'have no'
        >>> Absence.for_number(GrammaticalNumber.SINGULAR).text
        'has no'
        """
        return cls.HAVE_NO if number is GrammaticalNumber.PLURAL else cls.HAS_NO


class NonExistence(VocabEnum):
    """
    Non-existence verb — the *"does not exist"* / *"do not exist"* of *"the
    Robot does not exist"*, produced for a bare variable compared ``== None``
    (no attribute to name).
    """

    DOES_NOT_EXIST = OperatorWord("does not exist")
    DO_NOT_EXIST = OperatorWord("do not exist")

    @classmethod
    def for_number(cls, number: GrammaticalNumber) -> "NonExistence":
        """:return: ``DO_NOT_EXIST`` for a plural subject, else ``DOES_NOT_EXIST``.

        >>> NonExistence.for_number(GrammaticalNumber.PLURAL).text
        'do not exist'
        >>> NonExistence.for_number(GrammaticalNumber.SINGULAR).text
        'does not exist'
        """
        return (
            cls.DO_NOT_EXIST
            if number is GrammaticalNumber.PLURAL
            else cls.DOES_NOT_EXIST
        )


class PassiveAbsence(VocabEnum):
    """
    Passive absence verb for a *relational* (past-participle) attribute — the
    *"has not been"* / *"have not been"* of *"a Mission has not been assigned
    to any Robot"*, produced for a ``<participle>_<preposition>`` attribute
    compared ``== None``.

    Number-agreeing, selected explicitly (the morphology pass only
    agrees the copula).
    """

    HAS_NOT_BEEN = OperatorWord("has not been")
    HAVE_NOT_BEEN = OperatorWord("have not been")

    @classmethod
    def for_number(cls, number: GrammaticalNumber) -> "PassiveAbsence":
        """:return: ``HAVE_NOT_BEEN`` for a plural owner, else ``HAS_NOT_BEEN``.

        >>> PassiveAbsence.for_number(GrammaticalNumber.PLURAL).text
        'have not been'
        >>> PassiveAbsence.for_number(GrammaticalNumber.SINGULAR).text
        'has not been'
        """
        return (
            cls.HAVE_NOT_BEEN
            if number is GrammaticalNumber.PLURAL
            else cls.HAS_NOT_BEEN
        )


class Quantifiers(VocabEnum):
    """
    Indefinite quantifier for the object of a passive absence — the *"any"* of
    *"… assigned to any Robot"*, or the bare *"anything"* when the related type
    is not a nameable class.
    """

    ANY = PlainWord("any")
    ANYTHING = PlainWord("anything")


class SetMembership(VocabEnum):
    """
    Membership phrase for a domain-constrained value variable — the *"one of"*
    of *"one of OPTION_A, OPTION_B, or OPTION_C"*.
    """

    ONE_OF = PlainWord("one of")


class GroupingPhrases(VocabEnum):
    """
    The words a grouped query fronts itself with — *"For each <key>, report all
    <selection>"*, or *"Report the distinct <keys>"* when the selection is
    exactly the group key.
    """

    ALL = PlainWord("all")
    """
    Quantifies a per-group selection listing (*"report all Employees"*).
    """

    DISTINCT = PlainWord("distinct")
    """
    Marks a key-only grouped selection as a distinct listing (*"the distinct
    departments"*).
    """


class Specificity(VocabEnum):
    """
    Pre-head marking a concrete object literal as a specific instance (its
    identity, not its repr) — the *"specific"* of *"a specific Body"*.
    """

    SPECIFIC = PlainWord("specific")


class Punctuation(VocabEnum):
    """
    Structural punctuation tokens — role-less, like the brackets around a tuple
    (*"(v1, v2)"*) and the comma in a coordinated list (*"a, b, or c"*).
    """

    COMMA = PunctuationWord(",", spacing=Spacing.LEFT)
    OPEN_PAREN = PunctuationWord("(", spacing=Spacing.RIGHT)
    CLOSE_PAREN = PunctuationWord(")", spacing=Spacing.LEFT)


class Pronouns(VocabEnum):
    """
    Coreference pronouns standing in for a previously introduced variable.
    """

    ITS = PronounWord("its")
    THEIR = PronounWord("their")
    IT = PronounWord("it")
    THEY = PronounWord("they")

    @classmethod
    def possessive(cls, number: GrammaticalNumber) -> "Pronouns":
        """:return: The possessive pronoun — ``THEIR`` for a plural subject, else ``ITS``.

        >>> Pronouns.possessive(GrammaticalNumber.PLURAL).text
        'their'
        >>> Pronouns.possessive(GrammaticalNumber.SINGULAR).text
        'its'
        """
        return cls.THEIR if number is GrammaticalNumber.PLURAL else cls.ITS

    @classmethod
    def nominative(cls, number: GrammaticalNumber) -> "Pronouns":
        """:return: The nominative (subject) pronoun — ``THEY`` for a plural subject, else ``IT``.

        >>> Pronouns.nominative(GrammaticalNumber.PLURAL).text
        'they'
        >>> Pronouns.nominative(GrammaticalNumber.SINGULAR).text
        'it'
        """
        return cls.THEY if number is GrammaticalNumber.PLURAL else cls.IT


class RangePhrases(VocabEnum):
    """
    Range (``between``) operator core, produced when a lower-bound and an
    upper-bound comparison on the same attribute are folded together.

    The single source for *"between"*, shared by its copula-less
    (HAVING) surface and its copula-led predicative surface — agreement
    on the latter is realised by the copula (:func:`copula_with`), not
    by a duplicated plural member.
    """

    BETWEEN = OperatorWord("between")


class CoindexedPhrases(VocabEnum):
    """
    Fixed phrases for the factored co-indexed comparison rendering: the *"those
    of"* anaphor of the faithful form, and the *"have the same"* verb of the
    natural equality form.
    """

    THOSE_OF = PlainWord("those of")
    HAVE_THE_SAME = PlainWord("have the same")


class OrderingRangeWords(VocabEnum):
    """
    The range an ORDER BY clause runs over, read as plain English rather than a
    parenthetical SQL keyword.

    Looked up from a :class:`SortDirection` by the ordered-by assembler.
    """

    LOWEST_TO_HIGHEST = PlainWord("from lowest to highest")
    HIGHEST_TO_LOWEST = PlainWord("from highest to lowest")


class RankingWords(VocabEnum):
    """
    The qualifier words a ``limit`` (+ ordering) puts on the selection — *"the
    first two"*, *"the top three"*, *"the lowest"*, *"… by salary"*.
    """

    FIRST = PlainWord("first")
    """
    No ordering — the first *n* in natural order.
    """

    TOP = PlainWord("top")
    """
    Descending order, *n > 1* (*"the top three …"*).
    """

    BOTTOM = PlainWord("bottom")
    """
    Ascending order, *n > 1* (*"the bottom three …"*).
    """

    HIGHEST = PlainWord("highest")
    """Descending order, *n = 1* (*"the highest …"* / *"with the highest …"*)."""
    LOWEST = PlainWord("lowest")
    """Ascending order, *n = 1* (*"the lowest …"* / *"with the lowest …"*)."""
    BY = PlainWord("by")
    """
    Introduces the order key for a plural attribute ranking (*"… by salary"*).
    """


class Articles(VocabEnum):
    """
    Definite articles (THE, THE UNIQUE) and a static helper for indefinite
    articles.
    """

    THE = PlainWord("the")
    THE_UNIQUE = PlainWord("the unique")

    @staticmethod
    def indefinite(following_word: str) -> WordFragment:
        """
        :param following_word: The word immediately following the article.
        :return: A word fragment containing *"a"* or *"an"* based on the phonological context of
            *following_word* — ``WordFragment("a")`` or ``WordFragment("an")``.

        >>> Articles.indefinite("hour").text
        'an'
        >>> Articles.indefinite("robot").text
        'a'
        """
        text = morphology.indefinite_article(following_word) if following_word else "a"
        return WordFragment(text=text)


class ExistentialPhrase(VocabEnum):
    """
    Parameterised existential phrases (*"there's a/an TypeName"*, *"there are
    TypeNames"*).
    """

    THERE_IS_A = SingularExistential("there's")
    THERE_ARE = PluralExistential("there are")

    @classmethod
    def for_number(cls, number: GrammaticalNumber) -> ExistentialPhrase:
        """:return: The existential frame agreeing with *number*: ``THERE_ARE`` / ``THERE_IS_A``.

        >>> ExistentialPhrase.for_number(GrammaticalNumber.PLURAL).text
        'there are'
        >>> ExistentialPhrase.for_number(GrammaticalNumber.SINGULAR).text
        "there's"
        """
        return cls.THERE_ARE if number is GrammaticalNumber.PLURAL else cls.THERE_IS_A

    def build_phrase(
        self, type_name: str, referent_id: Optional[uuid.UUID] = None
    ) -> VerbalizationFragment:
        """
        Build the existential phrase for *type_name*.

        :param type_name: English noun in singular form.
        :param referent_id: Optional referent for coreference (singular only).
        :return: Existential phrase fragment.

        >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
        >>> flatten_fragment_to_plain_text(ExistentialPhrase.THERE_IS_A.build_phrase("Apple"))
        "there's an Apple"
        """
        if referent_id is not None and self is ExistentialPhrase.THERE_IS_A:
            return self.value.build_phrase(type_name, referent_id=referent_id)
        return self.value.build_phrase(type_name)


class FallbackNouns(VocabEnum):
    """
    Fallback noun used when no type information is available (e.g. *"entity"* /
    *"entities"*).
    """

    ENTITY = FallbackNounWord("entity")
    VARIABLE = FallbackNounWord("variable")

    def plural_fragment(self) -> WordFragment:
        """:return: A pluralised noun fragment (e.g. ``"entities"``).

        >>> FallbackNouns.ENTITY.plural_fragment().number
        <GrammaticalNumber.PLURAL: 'plural'>
        """
        return self.value.plural_fragment()

    def name_of(self, node: object) -> str:
        """
        :param node: A variable/entity-like node, or ``None``.
        :return: *node*'s type name (``_type_.__name__``), or this fallback noun's text when *node*
            is ``None`` or carries no type. Centralises the one optional-``_type_`` read every
            planner/assembler would otherwise repeat.

        >>> FallbackNouns.ENTITY.name_of(variable(Robot, []))
        'Robot'
        >>> FallbackNouns.ENTITY.name_of(None)
        'entity'
        """
        node_type = node._type_ if isinstance(node, Selectable) else None
        return node_type.__name__ if node_type is not None else self.text


class GroupKeyPhrases(VocabEnum):
    """
    Group-key binding phrases (*"the common <field> of the <plural_root>"*).
    """

    COMMON_OF = CommonGroupKeyWord("the common")

    def build_phrase(self, field_name: str, plural_root: str) -> VerbalizationFragment:
        """
        Build the group-key phrase.

        :param field_name: Grouped field name.
        :param plural_root: Plural root type name.
        :return: Group-key phrase fragment.

        >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
        >>> flatten_fragment_to_plain_text(GroupKeyPhrases.COMMON_OF.build_phrase("location", "Robots"))
        'the common location of the Robots'
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

        >>> Operators.GT.select().text
        'is greater than'
        >>> Operators.GT.select(compact=True).text
        'greater than'
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

        >>> Operators.from_callable(operator.gt) is Operators.GT
        True
        """
        return _OPERATOR_CALLABLE_MAP[function]

    @classmethod
    def for_callable(cls, function) -> Optional[Operators]:
        """
        :param function: Any callable.
        :return: The member for *function*, or ``None`` when it has no registered mapping (so a
            caller can fall back without catching an exception).

        >>> Operators.for_callable(operator.lt) is Operators.LT
        True
        >>> Operators.for_callable(len) is None
        True
        """
        return _OPERATOR_CALLABLE_MAP.get(function)

    @classmethod
    def is_value_comparison(cls, function: Callable) -> bool:
        """
        :param function: Any callable.
        :return: whether *function* is a scalar value comparison — an order/equality comparison of a
            subject against a value (``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``), as opposed to
            membership (``contains``) or an unmapped callable. The canonical set the shared-subject
            folds coordinate over, read from this vocabulary rather than re-listed by each caller.

        >>> Operators.is_value_comparison(operator.gt)
        True
        >>> Operators.is_value_comparison(operator.contains)
        False
        """
        member = cls.for_callable(function)
        return member is not None and member not in (cls.CONTAINS, cls.NOT_CONTAINS)


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


def copula_with(
    core: str,
    number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
    *,
    negated: bool = False,
) -> VerbalizationFragment:
    """
    Compose a predicative operator as an agreeing copula followed by an invariant core.

    The copula leaf carries *number*; the morphology pass realises agreement on it alone (*is* →
    *are*), so the core after it is never duplicated per number.

    :param core: The invariant text after the copula (*"greater than"*, *"between"*, *"equal to"*);
        empty for a bare equality (the copula alone).
    :param number: The grammatical number the copula agrees with.
    :param negated: Use the negative copula (*"is not"* / *"are not"*).
    :return: The copula leaf alone when *core* is empty, else a phrase of copula then core.

    Its contribution is making the operator *agree*: the *is* leaf carries *number*, so once the
    morphology pass runs, a plural number realises *are greater than* where a singular one stays
    *is greater than*.

    >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
    >>> from krrood.entity_query_language.verbalization.rendering.morphology_processor import MorphologyProcessor
    >>> realise = lambda fragment: flatten_fragment_to_plain_text(MorphologyProcessor().process(fragment))
    >>> realise(copula_with("greater than", GrammaticalNumber.SINGULAR))
    'is greater than'
    >>> realise(copula_with("greater than", GrammaticalNumber.PLURAL))
    'are greater than'
    >>> realise(copula_with("", GrammaticalNumber.PLURAL))
    'are'
    """
    copula = RoleFragment(
        text=(Copulas.IS_NOT if negated else Copulas.IS).text,
        role=SemanticRole.OPERATOR,
        number=number,
    )
    if not core:
        return copula
    return PhraseFragment(parts=[copula, RoleFragment.for_operator(core)])


def predicative_operator(
    text: str, number: GrammaticalNumber = GrammaticalNumber.SINGULAR
) -> VerbalizationFragment:
    """
    Factor a baked predicative operator surface into an agreeing copula and its invariant core.

    A copular form (*"is greater than"*, *"is not at most"*, *"is no later than"*) splits at its
    leading copula, so number agreement is realised on the copula alone. A verb operator with no
    leading copula (*"contains"*, *"does not contain"*) carries no copula and is returned un-agreed.

    :param text: The selected standard-register operator surface.
    :param number: The grammatical number the copula agrees with.
    :return: The factored operator fragment.

    Its contribution is the split that enables agreement: it separates the leading *is* (which
    agrees) from the invariant *greater than*, so once the morphology pass runs a plural owner reads
    *are greater than*. A verb operator with no leading copula (*contains*) carries none and stays
    un-agreed.

    >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
    >>> from krrood.entity_query_language.verbalization.rendering.morphology_processor import MorphologyProcessor
    >>> realise = lambda fragment: flatten_fragment_to_plain_text(MorphologyProcessor().process(fragment))
    >>> realise(predicative_operator("is greater than", GrammaticalNumber.PLURAL))
    'are greater than'
    >>> realise(predicative_operator("contains", GrammaticalNumber.PLURAL))
    'contains'
    """
    tokens = text.split()
    if not tokens or tokens[0] != Copulas.IS.text:
        return RoleFragment.for_operator(text)
    negated = len(tokens) > 1 and tokens[1] == Logicals.NOT.text
    return copula_with(predicative_core(text), number, negated=negated)


def predicative_core(text: str) -> str:
    """
    :param text: A baked predicative operator surface (*"is greater than"*, *"is"*, *"contains"*).
    :return: the invariant core after a leading copula (and any following *"not"*) — *"is greater
        than"* → *"greater than"*, *"is"* → *""* — so a shared copula can be factored out across
        coordinated tails. A copula-less verb operator (*"contains"*) is returned unchanged.

    >>> predicative_core("is greater than")
    'greater than'
    >>> predicative_core("is")
    ''
    >>> predicative_core("contains")
    'contains'
    """
    tokens = text.split()
    if not tokens or tokens[0] != Copulas.IS.text:
        return text
    negated = len(tokens) > 1 and tokens[1] == Logicals.NOT.text
    return " ".join(tokens[2:] if negated else tokens[1:])
