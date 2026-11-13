"""
Tests of specificity-ranked selection among the alternatives of a family.
"""

from abc import ABC

from typing_extensions import Callable

import pytest

from krrood.patterns.exceptions import AmbiguousRuleError
from krrood.patterns.classproperty import classproperty
from krrood.patterns.specificity_ranking import (
    SpecificityRankedAlternative,
    SpecificityRule,
    mro_depth,
)

# %% a family of rules that classify numbers


class NumberRule(SpecificityRule, ABC):
    """
    A family whose alternatives decide on integers.
    """


class NegativeNumberRule(NumberRule):
    """
    Applies to negative numbers.
    """

    @classmethod
    def applies(cls, *, number: int) -> bool:
        return number < 0


class NonNegativeNumberRule(NumberRule):
    """
    Applies to zero and positive numbers.
    """

    @classmethod
    def applies(cls, *, number: int) -> bool:
        return number >= 0


class ZeroRule(NonNegativeNumberRule):
    """
    A special case of non-negative numbers, which outranks the rule it refines.
    """

    @classmethod
    def applies(cls, *, number: int) -> bool:
        return number == 0


# %% a family of rules that covers only some numbers


class ParityRule(SpecificityRule, ABC):
    """
    A family whose only alternative decides on even numbers.
    """


class EvenNumberRule(ParityRule):
    """
    Applies to even numbers.
    """

    @classmethod
    def applies(cls, *, number: int) -> bool:
        return number % 2 == 0


# %% a family of rules whose guards overlap


class OverlappingRule(SpecificityRule, ABC):
    """
    A family whose alternatives both apply to every subject.
    """


class FirstOverlappingRule(OverlappingRule):
    """
    Applies to everything.
    """

    @classmethod
    def applies(cls, *, subject: object) -> bool:
        return True


class SecondOverlappingRule(OverlappingRule):
    """
    Also applies to everything.
    """

    @classmethod
    def applies(cls, *, subject: object) -> bool:
        return True


# %% a family ranked by the type it handles, with a guard of its own name


class TypeHandlingAlternative(SpecificityRankedAlternative, ABC):
    """
    A family whose alternatives each handle one type and are ranked by that type's
    depth.
    """

    @classproperty
    def handled_type(cls) -> type:
        """
        The type of the subjects this alternative handles.
        """
        return object

    @classproperty
    def guard(cls) -> Callable[..., bool]:
        """
        :meth:`handles`.
        """
        return cls.handles

    @classproperty
    def specificity(cls) -> int:
        """
        The depth of :attr:`handled_type`, not of the alternative itself.
        """
        return mro_depth(cls.handled_type)

    @classmethod
    def handles(cls, *, subject: object) -> bool:
        """
        :param subject: The subject to decide on.
        :return: Whether ``subject`` is of :attr:`handled_type`.
        """
        return isinstance(subject, cls.handled_type)


class IntegerHandlingAlternative(TypeHandlingAlternative):
    """
    Handles integers, booleans included.
    """

    @classproperty
    def handled_type(cls) -> type:
        """
        :class:`int`.
        """
        return int


class BooleanHandlingAlternative(TypeHandlingAlternative):
    """
    Handles booleans; a sibling of :class:`IntegerHandlingAlternative`, not a subclass.
    """

    @classproperty
    def handled_type(cls) -> type:
        """
        :class:`bool`.
        """
        return bool


# %% tests


def test_alternatives_are_the_concrete_rules_of_the_family():
    assert set(NumberRule.alternatives()) == {
        NegativeNumberRule,
        NonNegativeNumberRule,
        ZeroRule,
    }


@pytest.mark.parametrize(
    "number, rule",
    [(-3, NegativeNumberRule), (5, NonNegativeNumberRule), (0, ZeroRule)],
)
def test_most_specific_applying_rule_is_chosen(number, rule):
    assert NumberRule.most_applicable(number=number) is rule


def test_no_applying_rule_gives_none():
    assert ParityRule.most_applicable(number=3) is None


def test_equally_specific_applying_rules_are_rejected():
    with pytest.raises(AmbiguousRuleError) as collision:
        OverlappingRule.most_applicable(subject="anything")

    assert set(collision.value.candidates) == {
        FirstOverlappingRule,
        SecondOverlappingRule,
    }


@pytest.mark.parametrize(
    "subject, alternative",
    [(3, IntegerHandlingAlternative), (True, BooleanHandlingAlternative)],
)
def test_a_family_chooses_with_its_own_guard_and_specificity(subject, alternative):
    """
    A boolean goes to the alternative for :class:`bool`, whose type is deeper than
    :class:`int`, although neither alternative subclasses the other.
    """
    assert TypeHandlingAlternative.most_applicable(subject=subject) is alternative
