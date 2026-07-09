from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Iterable, Protocol, Union, runtime_checkable

from krrood.entity_query_language.predicate import VerbalizationField
from krrood.entity_query_language.utils import camel_case_to_words
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.fragments.base import (
    Clause,
    VerbalizationFragment,
    NounPhrase,
    PhraseFragment,
    RoleFragment,
    WordFragment,
    oxford_comma,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    MAX_SET_MEMBERS,
    one_of,
)
from krrood.entity_query_language.verbalization.microplanning.possessive import (
    possessive_path,
)
from krrood.entity_query_language.verbalization.navigation_path import PathStep
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    Copulas,
    Prepositions,
    SetMembership,
)


class ClauseElement(ABC):
    """One typed part-of-speech constituent of a predicate clause.

    A predicate's ``_verbalization_fragment_`` builds its clause from these elements rather than raw
    fragments, so the author writes the affirmative, present-tense form once and the realisation
    passes inflect it (verb agreement, copula suppletion) and negate it (do-support). The element
    only declares *what part of speech* a word is; how it is realised is the morphology pass's job.
    """

    @abstractmethod
    def as_fragment(self) -> VerbalizationFragment:
        """:return: the fragment this element contributes to the clause."""


@dataclass(frozen=True)
class Noun(ClauseElement):
    """A noun constituent — a predicate :class:`~krrood.entity_query_language.predicate.VerbalizationField`, an
    already-rendered fragment, or a literal noun given by its *head word only*."""

    content: Union[str, "ClauseConstituent"]
    """A literal noun *head* (the article is a feature, not part of the text — write ``"instance"``,
    not ``"an instance"``), or any constituent (a field, a rendered fragment) rendered as-is."""

    definiteness: Definiteness = Definiteness.INDEFINITE
    """For a literal-head noun, the article to realise — *"an instance"* (indefinite, the default) vs
    *"the instance"*. Ignored when *content* is already a rendered constituent."""

    def as_fragment(self) -> VerbalizationFragment:
        """:return: an indefinite (or definite) noun phrase for a literal head — so the article is
        chosen by the determiner pass (*"a"* / *"an"*) and the head pluralises and reduces with
        coreference — else the constituent's own fragment.

        >>> Noun("instance").as_fragment().definiteness
        <Definiteness.INDEFINITE: 'indefinite'>
        """
        if isinstance(self.content, str):
            return NounPhrase(
                head=WordFragment(text=self.content), definiteness=self.definiteness
            )
        return self.content.as_fragment()

    @classmethod
    def the(cls, head: str) -> "Noun":
        """:return: a definite literal noun (*"the name"*).

        >>> Noun.the("name").as_fragment().definiteness
        <Definiteness.DEFINITE: 'definite'>
        """
        return cls(head, definiteness=Definiteness.DEFINITE)


@dataclass(frozen=True)
class Verb(ClauseElement):
    """A lexical verb given as its lemma. The morphology pass realises it present-tense
    (*"work"* → *"works"*) and negates it with do-support (*"does not work"*)."""

    lemma: str
    """The verb's base form (*"work"*, *"contain"*, *"love"*)."""

    number: GrammaticalNumber = GrammaticalNumber.SINGULAR
    """The subject number the verb agrees with — ``PLURAL`` reads the bare *"work"* / *"have"* for a
    coordinated or plural subject."""

    def as_fragment(self) -> RoleFragment:
        """:return: a ``VERB``-role leaf carrying the lemma for the morphology pass to inflect.

        >>> Verb("work").as_fragment().role
        <SemanticRole.VERB: 'verb'>
        """
        return RoleFragment(text=self.lemma, role=SemanticRole.VERB, number=self.number)


@dataclass(frozen=True)
class Adjective(ClauseElement):
    """A predicative adjective complement after a copula (*"is **reachable**"*)."""

    word: str
    """The adjective's surface word."""

    def as_fragment(self) -> WordFragment:
        """:return: a plain word leaf for the adjective.

        >>> Adjective("reachable").as_fragment().text
        'reachable'
        """
        return WordFragment(text=self.word)


@dataclass(frozen=True)
class Copula(ClauseElement):
    """The copula *"is"* of a predicative clause — realised for number (*"is"* / *"are"*) and
    negation (*"is not"*) by the morphology pass."""

    def as_fragment(self) -> RoleFragment:
        """:return: the affirmative singular copula leaf the morphology pass inflects.

        >>> Copula().as_fragment().text
        'is'
        """
        return Copulas.IS.as_fragment()


@dataclass(frozen=True)
class OneOf(ClauseElement):
    """A bounded membership set — *"one of A, B, or C"* — over a collection of admissible values.

    This is the high-level element for a "the subject is one of these" clause (a tuple of admissible
    types, a small value domain), so an author never re-implements the listing: each member renders
    as a linked type reference when it is a class, else as a literal value, and a set larger than the
    cap is summarised by count (*"one of seven types"*) rather than spelled out — the same bounded
    surface a domain-constrained variable uses.
    """

    members: Union[Iterable, VerbalizationField]
    """The admissible values — a predicate :class:`~krrood.entity_query_language.predicate.VerbalizationField`
    bound to a collection, or a collection directly. Classes render as linked type references, other
    values as literals."""

    def as_fragment(self) -> VerbalizationFragment:
        """:return: the membership phrase, or a count summary past the cap.

        >>> from krrood.entity_query_language.verbalization.fragments.base import (
        ...     flatten_fragment_to_plain_text,
        ... )
        >>> flatten_fragment_to_plain_text(OneOf((int, str)).as_fragment())
        'one of Integer or Text'
        """
        members = list(
            self.members.value
            if isinstance(self.members, VerbalizationField)
            else self.members
        )
        are_types = bool(members) and all(
            isinstance(member, type) for member in members
        )
        render = RoleFragment.for_type if are_types else RoleFragment.for_literal
        listed = one_of([render(member) for member in members[: MAX_SET_MEMBERS + 1]])
        if listed is not None:
            return listed
        return PhraseFragment(
            parts=[
                SetMembership.ONE_OF.as_fragment(),
                WordFragment(text=morphology.cardinal(len(members))),
                WordFragment(text="types" if are_types else "values"),
            ]
        )


@runtime_checkable
class ClauseConstituent(Protocol):
    """The one contract every clause constituent satisfies: it renders itself to a :class:`VerbalizationFragment`.

    A typed part-of-speech element (:class:`ClauseElement` — :class:`Noun` / :class:`Verb` / …), a
    :class:`~krrood.entity_query_language.verbalization.vocabulary.english.Prepositions` member, a
    predicate :class:`~krrood.entity_query_language.predicate.VerbalizationField`, and a
    raw :class:`VerbalizationFragment` all satisfy it structurally (each defines ``as_fragment``), so
    :func:`clause` depends on this single abstraction rather than enumerating concrete types — a new
    kind of constituent only has to implement the method (open/closed).

    This is a :class:`~typing.Protocol`, not a base class, because the constituents are
    deliberately heterogeneous: ``Prepositions`` is an ``Enum`` (it cannot also inherit an ABC) and
    ``VerbalizationField`` is a core type (it must not depend on this verbalization layer). Structural typing
    unifies them without forcing inheritance.
    """

    def as_fragment(self) -> VerbalizationFragment:
        """:return: the fragment this constituent contributes to a clause."""


def clause(*constituents: ClauseConstituent) -> Clause:
    """
    Build a predicate clause from typed part-of-speech constituents.

    A predicate states its affirmative form once — *"<subject> works in <object>"* —
    ``clause(Noun(subject), Verb("work"), Prepositions.IN, Noun(object))`` — and the realisation
    passes handle agreement and negation. A raw :class:`VerbalizationFragment` is accepted too, so a rendered
    field fragment can be dropped in directly. The result is a :class:`Clause`, so coreference
    treats the first constituent as the clause's subject (pronominalisation, verb agreement).

    :param constituents: The clause's elements in surface order.
    :return: The clause fragment.

    >>> from krrood.entity_query_language.verbalization.fragments.base import (
    ...     flatten_fragment_to_plain_text, WordFragment,
    ... )
    >>> flatten_fragment_to_plain_text(
    ...     clause(Noun(WordFragment(text="an Employee")), Verb("work"), Prepositions.IN,
    ...            Noun(WordFragment(text="a Department")))
    ... )
    'an Employee work in a Department'
    """
    return Clause(parts=[constituent.as_fragment() for constituent in constituents])


_VALUE_GETTER_PREFIX = "get"
"""Leading imperative dropped when reading a getter's name as a noun (``get_quarter`` -> *"quarter"*)."""


def value_function_noun(name: str) -> str:
    """:return: a value (non-boolean) symbolic function class's name as noun words, dropping a leading
    imperative ``get`` so ``GetQuarter`` reads as *"quarter"*; a plain-noun name is unchanged.

    >>> value_function_noun("GetQuarter")
    'quarter'
    >>> value_function_noun("RemainingLoad")
    'remaining load'
    """
    words = camel_case_to_words(name).split()
    if len(words) > 1 and words[0].lower() == _VALUE_GETTER_PREFIX:
        words = words[1:]
    return " ".join(words)


def value_function_phrase(
    name: str, *operands: ClauseConstituent
) -> VerbalizationFragment:
    """Build *"the <noun> of <operands>"* for a value function — the value counterpart of a predicate
    clause, for an operation that computes a value rather than a truth.

    The *name* is read as the value's noun (a leading ``get`` dropped) and the operands are read out
    as a genitive over it. A nullary function is just the noun. This is the default, name-based surface
    a value :class:`~krrood.entity_query_language.predicate.SymbolicFunction` reuses, so the reading
    lives in one place rather than being duplicated per class.

    :param name: The function's identifier (or class name).
    :param operands: The function's already-rendered arguments.
    :return: The value noun phrase.

    >>> from krrood.entity_query_language.verbalization.fragments.base import (
    ...     flatten_fragment_to_plain_text, WordFragment,
    ... )
    >>> flatten_fragment_to_plain_text(value_function_phrase("GetQuarter"))
    'quarter'
    >>> flatten_fragment_to_plain_text(
    ...     value_function_phrase("RemainingLoad", Noun(WordFragment(text="the capacity")))
    ... )
    'the remaining load of the capacity'
    """
    noun = value_function_noun(name)
    if not operands:
        return Noun(WordFragment(text=noun)).as_fragment()
    owner = oxford_comma(
        [operand.as_fragment() for operand in operands], Conjunctions.AND.as_fragment()
    )
    return possessive_path([PathStep(noun)], owner)


_COPULA_LEMMA = "be"
"""The lemma a copular predicate name's leading word reduces to (``is`` / ``are`` -> ``be``)."""

_PREPOSITION_WORDS = frozenset(preposition.text for preposition in Prepositions)
"""The one-word prepositions, so a copular name ending in one attaches its object (``is_supported_by``)."""


def predicate_clause(
    name: str, subject: ClauseConstituent, *objects: ClauseConstituent
) -> Clause:
    """Build a predicate clause from a predicate's *name* and its operands — the boolean counterpart
    of :func:`value_function_phrase`.

    The name (CamelCase or snake_case) is read as the predicate. A copular name — its leading word a
    form of *"be"* (``IsReachable``) — uses the copula with the remaining words as the complement
    (*"<subject> is reachable"*). Any other name reads verb-first, so a wrapping ``Not`` negates it
    with do-support (``ConnectsTo`` → *"<subject> connects to <object>"*). The first operand is the
    subject; any further operands are trailing objects.

    A copular complement attaches trailing operands only through a final preposition
    (``is_supported_by`` → *"… is supported by <object>"*). With none, an adjective/noun complement
    cannot take them as objects and naming any single operand the subject would be a false claim, so
    the condition is stated to *hold given* all operands: *"one month holds given the begin and the end"*.

    :param name: The predicate's identifier — a class or function name.
    :param subject: The first operand, rendered as the clause's subject.
    :param objects: Any further operands, rendered as trailing objects.
    :return: The predicate clause.

    A bare literal :class:`Noun` (e.g. ``Noun("Robot")``) still needs the lowering passes
    (:func:`~…rendering.realization.realize_tree`) to choose its article before it can be flattened
    to text — the examples below realize the clause first, the same as a full query render does.

    >>> from krrood.entity_query_language.verbalization.fragments.base import (
    ...     flatten_fragment_to_plain_text,
    ... )
    >>> from krrood.entity_query_language.verbalization.rendering.realization import (
    ...     realize_tree,
    ... )
    >>> def render(fragment):
    ...     return flatten_fragment_to_plain_text(realize_tree(fragment))
    >>> render(predicate_clause("IsReachable", Noun("Robot")))
    'a Robot is reachable'
    >>> render(predicate_clause("ConnectsTo", Noun("body"), Noun("gripper")))
    'a body connects to a gripper'
    >>> render(predicate_clause("IsOneMonth", Noun.the("begin"), Noun.the("end")))
    'one month holds given the begin and the end'
    """
    head, *rest = camel_case_to_words(name).split()
    complement = [WordFragment(text=word) for word in rest]
    is_copular = morphology.verb_lemma(head) == _COPULA_LEMMA
    if is_copular and objects and (not rest or rest[-1] not in _PREPOSITION_WORDS):
        operands = oxford_comma(
            [
                Noun(subject).as_fragment(),
                *(Noun(obj).as_fragment() for obj in objects),
            ],
            Conjunctions.AND.as_fragment(),
        )
        return clause(
            Noun(PhraseFragment(parts=complement)),
            Verb("hold"),
            WordFragment(text="given"),
            operands,
        )
    predicate = (
        [Copula(), *complement]
        if is_copular
        else [Verb(morphology.verb_lemma(head)), *complement]
    )
    return clause(Noun(subject), *predicate, *(Noun(obj) for obj in objects))
