from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from typing_extensions import Optional


class Article(Enum):
    """
    The determiner to realise before a predicate's object noun.

    A declarative choice (not a surface word) so the verbalization layer picks the concrete
    article; mass and plural nouns take none, a singular count noun takes the indefinite one.
    """

    NONE = auto()
    """No article — a mass or plural noun (*"has milk"*, *"has feathers"*)."""

    INDEFINITE = auto()
    """The indefinite article *a* / *an* — a singular count noun (*"has a backbone"*)."""

    DEFINITE = auto()
    """The definite article *the*."""


@dataclass(frozen=True)
class BooleanPredicateSpec:
    """
    How a boolean attribute reads as a predicate when it holds.

    A declarative, verbalization-agnostic hint attached to a field via
    :class:`~krrood.patterns.field_metadata.GrammarMetadata`; the verbalization layer maps it to a
    clause and derives the negation from it (do-support / copula suppletion). It lives in the
    ``patterns`` layer — carrying data only, no linguistic behaviour — so a domain class can declare
    it without depending on the verbalization subsystem.

    This is the sealed family's marker base: one of its concrete subclasses
    (:class:`AdjectivalPredicate`, :class:`PossessivePredicate`, :class:`VerbalPredicate`) is
    declared, never this class directly.
    """


@dataclass(frozen=True)
class AdjectivalPredicate(BooleanPredicateSpec):
    """The attribute reads as a predicative adjective — *"is operational"*."""

    adjective: Optional[str] = None
    """The adjective word; ``None`` uses the attribute's own (display) name."""


@dataclass(frozen=True)
class PossessivePredicate(BooleanPredicateSpec):
    """The attribute reads as something the subject *has* — *"has milk"*, *"has a backbone"*."""

    noun: Optional[str] = None
    """The possessed noun; ``None`` uses the attribute's own (display) name."""

    article: Article = Article.NONE
    """The determiner before the noun — bare for a mass/plural noun, indefinite for a count noun."""


@dataclass(frozen=True)
class VerbalPredicate(BooleanPredicateSpec):
    """The attribute reads as an action the subject performs — *"produces milk"*, *"breathes"*."""

    verb: str
    """The verb lemma (*"produce"*, *"breathe"*); conjugated and negated by the morphology pass."""

    object_noun: Optional[str] = None
    """An optional object noun (*"produces **milk**"*); ``None`` for an intransitive verb."""

    object_article: Article = Article.NONE
    """The determiner before :attr:`object_noun`, when it is present."""
