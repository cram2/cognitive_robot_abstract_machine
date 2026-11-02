from abc import ABC

import pytest

from krrood.patterns.exceptions import AmbiguousRuleError
from krrood.patterns.specificity_ranking import SpecificityRule

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
    def applies(cls, number: int) -> bool:
        return number < 0


class NonNegativeNumberRule(NumberRule):
    """
    Applies to zero and positive numbers.
    """

    @classmethod
    def applies(cls, number: int) -> bool:
        return number >= 0


class ZeroRule(NonNegativeNumberRule):
    """
    A special case of non-negative numbers, which outranks the rule it refines.
    """

    @classmethod
    def applies(cls, number: int) -> bool:
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
    def applies(cls, number: int) -> bool:
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
    def applies(cls, subject: object) -> bool:
        return True


class SecondOverlappingRule(OverlappingRule):
    """
    Also applies to everything.
    """

    @classmethod
    def applies(cls, subject: object) -> bool:
        return True


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
    assert NumberRule.most_applicable(number) is rule


def test_no_applying_rule_gives_none():
    assert ParityRule.most_applicable(3) is None


def test_equally_specific_applying_rules_are_rejected():
    with pytest.raises(AmbiguousRuleError) as collision:
        OverlappingRule.most_applicable("anything")

    assert set(collision.value.candidates) == {
        FirstOverlappingRule,
        SecondOverlappingRule,
    }
