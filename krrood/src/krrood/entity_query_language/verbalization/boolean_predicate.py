"""
How a boolean attribute reads as a predicate, and how it builds its clause.

A boolean attribute reads as a predicate: possessive (*"has milk"*), adjectival (*"is operational"*),
or verbal (*"produces milk"*). Which form an attribute takes is declared per field via
:class:`~krrood.entity_query_language.verbalization.grammar_metadata.GrammarMetadata`, or inferred
from the attribute name's shape when unspecified.

Each :class:`BooleanPredicate` owns its own realisation: it contributes the finite ``head`` (the
verb/copula the morphology pass inflects and negates) and the ``predicate_object`` after it (a
predicative adjective, a possessed noun, an object noun). The positive clause is assembled in
:mod:`~krrood.entity_query_language.verbalization.attribute_predicates`; its negation is *derived* —
the head carries the ``negated`` flag and the morphology pass realises do-support (*"does not have
milk"*) or copula suppletion (*"is not operational"*).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from typing_extensions import Optional

from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    RoleFragment,
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import Definiteness
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    Adjective,
    Copula,
    Noun,
    Verb,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.core.mapped_variable import Attribute


@dataclass(frozen=True)
class BooleanPredicate(ABC):
    """
    How a boolean attribute reads as a predicate when it holds.

    A declarative hint attached to a field via
    :class:`~krrood.entity_query_language.verbalization.grammar_metadata.GrammarMetadata`. One of its
    concrete subclasses (:class:`AdjectivalPredicate`, :class:`PossessivePredicate`,
    :class:`VerbalPredicate`) is declared, never this class directly; each contributes the two
    constituents the clause is assembled from, so a new predicate kind is a new subclass with no
    dispatch to edit (open/closed).
    """

    @abstractmethod
    def head(self) -> VerbalizationFragment:
        """:return: The finite verb/copula leaf (affirmative, singular) the clause agrees with and
        negates — a copula for the adjectival reading, *have* for the possessive, the lexical verb
        for the verbal."""

    @abstractmethod
    def predicate_object(self, attribute: Attribute) -> Optional[VerbalizationFragment]:
        """:param attribute: The boolean attribute node, supplying the owner class and name for the
        default surface when the predicate names none.

        :return: The constituent the head takes — a predicative adjective (*"is **operational**"*), a
            possessed noun (*"has **milk**"*), or an object noun (*"produces **milk**"*) — or ``None``
            for an intransitive verb.
        """


@dataclass(frozen=True)
class AdjectivalPredicate(BooleanPredicate):
    """
    The attribute reads as a predicative adjective — *"is operational"*.
    """

    adjective: Optional[str] = None
    """
    The adjective word; ``None`` uses the attribute's own (display) name.
    """

    def head(self) -> VerbalizationFragment:
        return Copula().as_fragment()

    def predicate_object(self, attribute: Attribute) -> VerbalizationFragment:
        if self.adjective is not None:
            return Adjective(self.adjective).as_fragment()
        return RoleFragment.for_attribute(
            attribute._owner_class_, attribute._attribute_name_
        )


@dataclass(frozen=True)
class PossessivePredicate(BooleanPredicate):
    """
    The attribute reads as something the subject *has* — *"has milk"*, *"has a backbone"*.
    """

    noun: Optional[str] = None
    """
    The possessed noun; ``None`` uses the attribute's own (display) name.
    """

    definiteness: Definiteness = Definiteness.BARE
    """
    The determiner before the noun — bare for a mass/plural noun, indefinite for a count noun.
    """

    def head(self) -> VerbalizationFragment:
        return Verb("have").as_fragment()

    def predicate_object(self, attribute: Attribute) -> VerbalizationFragment:
        if self.noun is not None:
            return Noun(self.noun, definiteness=self.definiteness).as_fragment()
        return NounPhrase(
            head=RoleFragment.for_attribute(
                attribute._owner_class_, attribute._attribute_name_
            ),
            definiteness=self.definiteness,
        )


@dataclass(frozen=True)
class VerbalPredicate(BooleanPredicate):
    """
    The attribute reads as an action the subject performs — *"produces milk"*, *"breathes"*.
    """

    verb: str
    """
    The verb lemma (*"produce"*, *"breathe"*); conjugated and negated by the morphology pass.
    """

    object_noun: Optional[str] = None
    """
    An optional object noun (*"produces **milk**"*); ``None`` for an intransitive verb — one that
    takes no object, so the predicate is the verb alone (*"breathes"*).
    """

    object_definiteness: Definiteness = Definiteness.BARE
    """
    The determiner before :attr:`object_noun`, when it is present.
    """

    def head(self) -> VerbalizationFragment:
        return Verb(self.verb).as_fragment()

    def predicate_object(self, attribute: Attribute) -> Optional[VerbalizationFragment]:
        if self.object_noun is None:
            return None
        return Noun(
            self.object_noun, definiteness=self.object_definiteness
        ).as_fragment()
