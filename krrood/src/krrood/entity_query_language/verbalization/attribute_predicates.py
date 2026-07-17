"""
Lexicalise a boolean attribute as a predicate clause.

A boolean attribute reads as a predicate: possessive (*"has milk"*), adjectival (*"is
operational"*), or verbal (*"produces milk"*). Which form an attribute takes is declared per field
via :class:`~krrood.patterns.boolean_predicate.BooleanPredicateSpec`
(:class:`~krrood.patterns.field_metadata.GrammarMetadata`), or inferred from the attribute name's
shape when unspecified. The positive clause is built here; its negation is *derived*, not
re-templated — the head verb/copula carries the ``negated`` flag and the morphology pass realises
do-support (*"does not have milk"*) or copula suppletion (*"is not operational"*).

Mirrors :mod:`~krrood.entity_query_language.verbalization.relational_attributes`: a name-shape
recognizer that renders an attribute as a predicate. It depends only on the parts-of-speech
vocabulary and morphology (no ``grammar`` import), so the chain assembler can call it without a
cycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

from typing_extensions import ClassVar, Dict, Optional, Type

from krrood.patterns.boolean_predicate import (
    AdjectivalPredicate,
    Article,
    BooleanPredicateSpec,
    PossessivePredicate,
    VerbalPredicate,
)
from krrood.patterns.field_metadata import GrammarMetadata
from krrood.patterns.specificity_ranking import concrete_subclasses

from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.exceptions import (
    UnknownBooleanPredicateError,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    Clause,
    NounPhrase,
    RoleFragment,
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    Logicals,
)
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    Adjective,
    Copula,
    Noun,
    Verb,
    clause,
)

#: Maps the declarative article choice to the fragment-level determiner feature — the single place
#: the ``patterns`` article enum meets the verbalization determiner system.
_ARTICLE_TO_DEFINITENESS: Dict[Article, Definiteness] = {
    Article.NONE: Definiteness.BARE,
    Article.INDEFINITE: Definiteness.INDEFINITE,
    Article.DEFINITE: Definiteness.DEFINITE,
}


class AttributePredicateRealizer(ABC):
    """
    Builds the positive predicate (finite head + complement) for one kind of boolean-
    predicate spec.

    The head is a verb or copula leaf the morphology pass inflects and negates; the
    complement is the adjective / possessed noun / verb object. Concrete realizers are
    discovered by spec type, so a new predicate kind is a new spec plus a new realizer —
    no dispatch edit (open/closed).
    """

    spec_type: ClassVar[Type[BooleanPredicateSpec]]
    """
    The spec class this realizer handles.
    """

    @abstractmethod
    def head(self, spec: BooleanPredicateSpec) -> VerbalizationFragment:
        """:return: The finite verb/copula leaf (affirmative, singular) the clause agrees and negates."""

    @abstractmethod
    def complement(
        self, spec: BooleanPredicateSpec, owner: Optional[type], attribute_name: str
    ) -> Optional[VerbalizationFragment]:
        """:return: The predicate complement (adjective / noun), or ``None`` for none."""


class AdjectivalRealizer(AttributePredicateRealizer):
    """
    *"is operational"* — a copula and a predicative adjective (the current default
    surface).
    """

    spec_type = AdjectivalPredicate

    def head(self, spec: AdjectivalPredicate) -> VerbalizationFragment:
        return Copula().as_fragment()

    def complement(
        self, spec: AdjectivalPredicate, owner: Optional[type], attribute_name: str
    ) -> VerbalizationFragment:
        if spec.adjective is not None:
            return Adjective(spec.adjective).as_fragment()
        return RoleFragment.for_attribute(owner, attribute_name)


class PossessiveRealizer(AttributePredicateRealizer):
    """
    *"has milk"* / *"has a backbone"* — the verb *have* and the possessed noun.
    """

    spec_type = PossessivePredicate

    def head(self, spec: PossessivePredicate) -> VerbalizationFragment:
        return Verb("have").as_fragment()

    def complement(
        self, spec: PossessivePredicate, owner: Optional[type], attribute_name: str
    ) -> VerbalizationFragment:
        return _object_noun(spec.noun, spec.article, owner, attribute_name)


class VerbalRealizer(AttributePredicateRealizer):
    """
    *"produces milk"* / *"breathes"* — a lexical verb and an optional object noun.
    """

    spec_type = VerbalPredicate

    def head(self, spec: VerbalPredicate) -> VerbalizationFragment:
        return Verb(spec.verb).as_fragment()

    def complement(
        self, spec: VerbalPredicate, owner: Optional[type], attribute_name: str
    ) -> Optional[VerbalizationFragment]:
        if spec.object_noun is None:
            return None
        return Noun(
            spec.object_noun,
            definiteness=_ARTICLE_TO_DEFINITENESS[spec.object_article],
        ).as_fragment()


def _object_noun(
    noun: Optional[str],
    article: Article,
    owner: Optional[type],
    attribute_name: str,
) -> VerbalizationFragment:
    """:return: The possessed-noun phrase, honouring an explicit *noun* or falling back to the
    attribute's (display) name, wrapped with the article the spec asks for."""
    definiteness = _ARTICLE_TO_DEFINITENESS[article]
    if noun is not None:
        return Noun(noun, definiteness=definiteness).as_fragment()
    return NounPhrase(
        head=RoleFragment.for_attribute(owner, attribute_name),
        definiteness=definiteness,
    )


#: One instance of each concrete realizer, keyed by the spec type it handles — the same
#: ``concrete_subclasses`` discovery the grammar rule families use.
_REALIZERS: Dict[Type[BooleanPredicateSpec], AttributePredicateRealizer] = {
    realizer.spec_type: realizer
    for realizer in (cls() for cls in concrete_subclasses(AttributePredicateRealizer))
}


def _realizer_for(spec: BooleanPredicateSpec) -> AttributePredicateRealizer:
    """:return: The realizer for *spec*'s type.

    :raises UnknownBooleanPredicateError: When no realizer handles the spec type.
    """
    realizer = _REALIZERS.get(type(spec))
    if realizer is None:
        raise UnknownBooleanPredicateError(spec=spec)
    return realizer


def default_boolean_predicate(attribute_name: str) -> BooleanPredicateSpec:
    """:return: The predicate inferred from *attribute_name*'s shape — adjectival for a
    participle/adjective-shaped name (*"completed"*, *"operational"*), else possessive (*"milk"* →
    *"has milk"*). Verbs are never inferred; a verbal reading needs an explicit spec.

    The participle test is deterministic; the adjective test is a best-effort suffix heuristic, so a
    suffix-less adjective (*"airborne"*) falls through to possessive and must be given an explicit
    :class:`~krrood.patterns.boolean_predicate.AdjectivalPredicate`.
    """
    last = attribute_name.split("_")[-1]
    if morphology.is_past_participle(last) or morphology.is_likely_adjective(last):
        return AdjectivalPredicate()
    return PossessivePredicate()


def resolve_boolean_predicate(
    owner: Optional[type], attribute_name: str
) -> BooleanPredicateSpec:
    """:return: The declared :class:`~krrood.patterns.boolean_predicate.BooleanPredicateSpec` for the
    field, or the :func:`default_boolean_predicate` heuristic when none is declared."""
    metadata = (
        GrammarMetadata.of_field(owner, attribute_name) if owner is not None else None
    )
    if metadata is not None and metadata.boolean_predicate is not None:
        return metadata.boolean_predicate
    return default_boolean_predicate(attribute_name)


def boolean_predicate_clause(
    subject: VerbalizationFragment,
    spec: BooleanPredicateSpec,
    owner: Optional[type],
    attribute_name: str,
    negated: bool = False,
    number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
) -> Clause:
    """
    :param subject: The already-rendered subject phrase (the navigation to the attribute owner).
    :param spec: How the attribute reads as a predicate.
    :param owner: The attribute's owner class (for the default noun/adjective surface).
    :param attribute_name: The attribute's name.
    :param negated: Whether the predicate is negated — realised as do-support / copula suppletion.
    :param number: The subject number the head agrees with (coreference re-agrees an in-scope subject).
    :return: The subject-led predicate clause (*"the Animal has milk"* / *"is not operational"*).
    """
    realizer = _realizer_for(spec)
    head = replace(realizer.head(spec), negated=negated, number=number)
    complement = realizer.complement(spec, owner, attribute_name)
    constituents = [subject, head]
    if complement is not None:
        constituents.append(complement)
    return clause(*constituents)


def boolean_alternative_clause(
    subject: VerbalizationFragment,
    spec: BooleanPredicateSpec,
    owner: Optional[type],
    attribute_name: str,
) -> Clause:
    """:return: The open-boolean predicate *"<subject> <head> either <complement> or not"* — for a
    boolean attribute left open (compared to a both-``True``-and-``False`` domain), so neither
    polarity is asserted (*"is either operational or not"*, *"has either milk or not"*).
    """
    realizer = _realizer_for(spec)
    head = realizer.head(spec)
    complement = realizer.complement(spec, owner, attribute_name)
    if complement is None:
        # An intransitive verb has nothing to coordinate, so "either" fronts the verb itself.
        return clause(subject, Logicals.EITHER, head, Conjunctions.OR, Logicals.NOT)
    return clause(
        subject, head, Logicals.EITHER, complement, Conjunctions.OR, Logicals.NOT
    )
