"""
The ways a leaf of the fragment tree is realized as surface words.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from typing_extensions import Callable, Generic, TypeVar

from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.fragments.base import (
    CardinalNumber,
    CollectionElementNoun,
    IndexOrdinal,
    RoleFragment,
    VerbalizationFragment,
    VerbFormFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.vocabulary.english import Copulas
from krrood.patterns.classproperty import classproperty
from krrood.patterns.specificity_ranking import SpecificityRankedAlternative
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

RealizedLeaf = TypeVar("RealizedLeaf", bound=VerbalizationFragment)
AgreeingLeaf = TypeVar("AgreeingLeaf", bound=WordFragment | RoleFragment)
PolarLeaf = TypeVar("PolarLeaf", bound=RoleFragment)

# %% the family


@dataclass(kw_only=True)
class LeafRealization(
    Generic[RealizedLeaf], SubClassSafeGeneric, SpecificityRankedAlternative, ABC
):
    """
    Realizes the leaves of its bound type as surface words.

    A new way of realizing a leaf is a new subclass: the most specific one that applies
    to a leaf realizes it.
    """

    @classproperty
    def leaf_type(cls) -> type:
        """
        The type of the leaves this realization handles.
        """
        return cls.get_type_of_generic_parameter(parameter=RealizedLeaf)

    @classproperty
    def guard(cls) -> Callable[..., bool]:
        """
        :meth:`applies`.
        """
        return cls.applies

    @classmethod
    def applies(cls, *, leaf: VerbalizationFragment) -> bool:
        """
        :return: Whether *leaf* is of :attr:`leaf_type`; a realization narrows this when
            some leaves of that type are realized differently.
        """
        return isinstance(leaf, cls.leaf_type)

    @classmethod
    @abstractmethod
    def realize(cls, *, leaf: RealizedLeaf) -> VerbalizationFragment:
        """
        :param leaf: A leaf this realization applies to.
        :return: The leaf realized as surface words.
        """


# %% words decided by a count or a position


@dataclass(kw_only=True)
class CardinalNumberRealization(LeafRealization[CardinalNumber]):
    """
    A count as its English cardinal word.
    """

    @classmethod
    def realize(cls, *, leaf: CardinalNumber) -> WordFragment:
        """
        >>> CardinalNumberRealization.realize(leaf=CardinalNumber(count=7)).text
        'seven'
        """
        return WordFragment(text=morphology.cardinal(leaf.count))


@dataclass(kw_only=True)
class IndexOrdinalRealization(LeafRealization[IndexOrdinal]):
    """
    A sequence index as its English ordinal phrase.
    """

    @classmethod
    def realize(cls, *, leaf: IndexOrdinal) -> WordFragment:
        """
        >>> IndexOrdinalRealization.realize(leaf=IndexOrdinal(index=-2)).text
        'second to last'
        """
        return WordFragment(text=morphology.index_ordinal(leaf.index))


# %% leaves that agree in number


@dataclass(kw_only=True)
class AgreeingRealization(LeafRealization[AgreeingLeaf], ABC):
    """
    A realization whose text depends on the grammatical number of the leaf, which is
    asked here and nowhere else.
    """

    @classmethod
    @abstractmethod
    def singular_text(cls, *, leaf: AgreeingLeaf) -> str:
        """
        :return: The text of the leaf agreeing with a singular subject.
        """

    @classmethod
    @abstractmethod
    def plural_text(cls, *, leaf: AgreeingLeaf) -> str:
        """
        :return: The text of the leaf agreeing with a plural subject.
        """

    @classmethod
    def realize(cls, *, leaf: AgreeingLeaf) -> AgreeingLeaf:
        """
        The leaf with the text that agrees with its number, which it then no longer
        needs to carry.
        """
        if leaf.number is GrammaticalNumber.PLURAL:
            text = cls.plural_text(leaf=leaf)
        else:
            text = cls.singular_text(leaf=leaf)
        return replace(leaf, text=text, number=GrammaticalNumber.SINGULAR)


@dataclass(kw_only=True)
class PolarRealization(LeafRealization[PolarLeaf], ABC):
    """
    A realization whose text depends on the grammatical number and the polarity of the
    leaf, which are asked here and nowhere else.
    """

    @classmethod
    @abstractmethod
    def affirmative_singular(cls, *, leaf: PolarLeaf) -> str:
        """
        :return: The text of the leaf agreeing with a singular subject.
        """

    @classmethod
    @abstractmethod
    def affirmative_plural(cls, *, leaf: PolarLeaf) -> str:
        """
        :return: The text of the leaf agreeing with a plural subject.
        """

    @classmethod
    @abstractmethod
    def negated_singular(cls, *, leaf: PolarLeaf) -> str:
        """
        :return: The negated text of the leaf agreeing with a singular subject.
        """

    @classmethod
    @abstractmethod
    def negated_plural(cls, *, leaf: PolarLeaf) -> str:
        """
        :return: The negated text of the leaf agreeing with a plural subject.
        """

    @classmethod
    def realize(cls, *, leaf: PolarLeaf) -> PolarLeaf:
        """
        The leaf with the text that agrees with its number and polarity, which it then
        no longer needs to carry.
        """
        plural = leaf.number is GrammaticalNumber.PLURAL
        if leaf.negated:
            text = (
                cls.negated_plural(leaf=leaf)
                if plural
                else cls.negated_singular(leaf=leaf)
            )
        else:
            text = (
                cls.affirmative_plural(leaf=leaf)
                if plural
                else cls.affirmative_singular(leaf=leaf)
            )
        return replace(
            leaf, text=text, number=GrammaticalNumber.SINGULAR, negated=False
        )


# %% nouns


@dataclass(kw_only=True)
class NounRealization(AgreeingRealization[WordFragment | RoleFragment]):
    """
    A noun, or any other text that is not a verb or a copula, inflected for its number.
    """

    @classmethod
    def applies(cls, *, leaf: VerbalizationFragment) -> bool:
        """
        Like :meth:`LeafRealization.applies`, but a verb or a copula is not a noun.
        """
        return super().applies(leaf=leaf) and not (
            isinstance(leaf, RoleFragment)
            and leaf.role in (SemanticRole.VERB, SemanticRole.OPERATOR)
        )

    @classmethod
    def singular_text(cls, *, leaf: WordFragment | RoleFragment) -> str:
        """
        >>> NounRealization.realize(leaf=WordFragment(text="Robot")).text
        'Robot'
        """
        return leaf.text

    @classmethod
    def plural_text(cls, *, leaf: WordFragment | RoleFragment) -> str:
        """
        >>> NounRealization.realize(leaf=WordFragment(text="Robot", number=GrammaticalNumber.PLURAL)).text
        'Robots'
        """
        return morphology.ensure_plural(leaf.text)


@dataclass(kw_only=True)
class CollectionElementNounRealization(NounRealization):
    """
    The noun naming one element of a collection, reduced from the collection's name to
    its singular before it is inflected for its number.
    """

    @classmethod
    def applies(cls, *, leaf: VerbalizationFragment) -> bool:
        """
        Like :meth:`NounRealization.applies`, but only for a
        :class:`CollectionElementNoun`.
        """
        return super().applies(leaf=leaf) and isinstance(leaf, CollectionElementNoun)

    @classmethod
    def singular_text(cls, *, leaf: CollectionElementNoun) -> str:
        """
        >>> CollectionElementNounRealization.realize(
        ...     leaf=CollectionElementNoun(text="tasks", role=SemanticRole.ATTRIBUTE)
        ... ).text
        'task'
        """
        return morphology.singular(leaf.text)

    @classmethod
    def plural_text(cls, *, leaf: CollectionElementNoun) -> str:
        """
        Like :meth:`NounRealization.plural_text`, of the singular of the leaf's text.
        """
        return morphology.ensure_plural(cls.singular_text(leaf=leaf))


# %% verbs


@dataclass(kw_only=True)
class VerbRealization(PolarRealization[RoleFragment]):
    """
    A verb agreeing with its subject in the present tense, or with do-support when
    negated.
    """

    @classmethod
    def applies(cls, *, leaf: VerbalizationFragment) -> bool:
        """
        Like :meth:`LeafRealization.applies`, but only for a leaf with the verb role.
        """
        return super().applies(leaf=leaf) and leaf.role is SemanticRole.VERB

    @classmethod
    def lemma(cls, *, leaf: RoleFragment) -> str:
        """
        :return: The base form of the verb that the leaf's text spells.
        """
        return leaf.text

    @classmethod
    def affirmative_singular(cls, *, leaf: RoleFragment) -> str:
        """
        >>> VerbRealization.realize(leaf=RoleFragment(text="work", role=SemanticRole.VERB)).text
        'works'
        """
        return morphology.third_person_singular(cls.lemma(leaf=leaf))

    @classmethod
    def affirmative_plural(cls, *, leaf: RoleFragment) -> str:
        return cls.lemma(leaf=leaf)

    @classmethod
    def negated_singular(cls, *, leaf: RoleFragment) -> str:
        """
        >>> VerbRealization.realize(
        ...     leaf=RoleFragment(text="work", role=SemanticRole.VERB, negated=True)
        ... ).text
        'does not work'
        """
        return f"does not {cls.lemma(leaf=leaf)}"

    @classmethod
    def negated_plural(cls, *, leaf: RoleFragment) -> str:
        return f"do not {cls.lemma(leaf=leaf)}"


@dataclass(kw_only=True)
class VerbFormRealization(VerbRealization):
    """
    A verb read off a name in any of its forms, reduced to its base form before it is
    inflected.
    """

    @classmethod
    def applies(cls, *, leaf: VerbalizationFragment) -> bool:
        """
        Like :meth:`VerbRealization.applies`, but only for a :class:`VerbFormFragment`.
        """
        return super().applies(leaf=leaf) and isinstance(leaf, VerbFormFragment)

    @classmethod
    def lemma(cls, *, leaf: VerbFormFragment) -> str:
        """
        >>> VerbFormRealization.lemma(leaf=VerbFormFragment(text="connects"))
        'connect'
        """
        return morphology.verb_lemma(leaf.text)


# %% copulas


@dataclass(kw_only=True)
class CopulaRealization(PolarRealization[RoleFragment]):
    """
    A copula agreeing in number with its subject and negated by suppletion (*"is not"*,
    *"are"*).

    An operator that is not a copula (*"greater than"*) keeps its text.
    """

    @classproperty
    def negative_forms(cls) -> dict[str, str]:
        """
        The negative form of each affirmative copula.
        """
        return {
            Copulas.IS.text: Copulas.IS_NOT.text,
            Copulas.ARE.text: Copulas.ARE_NOT.text,
        }

    @classproperty
    def plural_forms(cls) -> dict[str, str]:
        """
        The plural form of each singular copula.
        """
        return {
            Copulas.IS.text: Copulas.ARE.text,
            Copulas.IS_NOT.text: Copulas.ARE_NOT.text,
        }

    @classmethod
    def applies(cls, *, leaf: VerbalizationFragment) -> bool:
        """
        Like :meth:`LeafRealization.applies`, but only for a leaf with the operator
        role.
        """
        return super().applies(leaf=leaf) and leaf.role is SemanticRole.OPERATOR

    @classmethod
    def affirmative_singular(cls, *, leaf: RoleFragment) -> str:
        return leaf.text

    @classmethod
    def affirmative_plural(cls, *, leaf: RoleFragment) -> str:
        """
        >>> CopulaRealization.realize(
        ...     leaf=RoleFragment(text="is", role=SemanticRole.OPERATOR, number=GrammaticalNumber.PLURAL)
        ... ).text
        'are'
        """
        return cls.plural_forms.get(leaf.text, leaf.text)

    @classmethod
    def negated_singular(cls, *, leaf: RoleFragment) -> str:
        """
        >>> CopulaRealization.realize(
        ...     leaf=RoleFragment(text="is", role=SemanticRole.OPERATOR, negated=True)
        ... ).text
        'is not'
        """
        return cls.negative_forms.get(leaf.text, leaf.text)

    @classmethod
    def negated_plural(cls, *, leaf: RoleFragment) -> str:
        negative = cls.negated_singular(leaf=leaf)
        return cls.plural_forms.get(negative, negative)
