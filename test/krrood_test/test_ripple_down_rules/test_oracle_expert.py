from dataclasses import dataclass
from enum import Enum

import pytest
from typing_extensions import Optional

from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.exceptions import (
    NoDistinguishingAttributeFound,
    OracleCannotInferConclusionWithoutTarget,
)
from krrood.ripple_down_rules.experts import Oracle
from krrood.ripple_down_rules.rdr import SingleClassRDR


class AnimalClass(Enum):
    """
    A minimal target enum mimicking the real-world Species enum, used to exercise Oracle
    without depending on the zoo dataset or the network.
    """

    mammal = "mammal"
    bird = "bird"
    arachnid = "arachnid"


@dataclass
class Animal:
    """
    A minimal case mimicking a real-world animal entity with a mix of simple attributes,
    used to exercise Oracle without depending on the zoo dataset or the network.
    """

    name: str
    has_fur: bool
    leg_count: int
    animal_class: Optional[AnimalClass] = None


@dataclass
class RuleWithCornerCase:
    """
    A minimal stand-in for a ripple-down-rule node, carrying only the one field
    (``corner_case``) that Oracle reads off the last evaluated rule.
    """

    corner_case: object


def make_case_query(case: Animal, target: AnimalClass) -> CaseQuery:
    return CaseQuery(case, "animal_class", (AnimalClass,), True, _target=target)


class TestOracleBuildsAConditionTrueForTheCaseAndFalseForTheCornerCase:

    def test_condition_is_true_for_the_case_and_false_for_the_corner_case(self):
        case = Animal(name="dog", has_fur=True, leg_count=4)
        corner_case = Animal(name="parrot", has_fur=False, leg_count=2)
        case_query = make_case_query(case, target=AnimalClass.mammal)
        last_evaluated_rule = RuleWithCornerCase(corner_case=corner_case)

        condition = Oracle().ask_for_conditions(case_query, last_evaluated_rule)

        assert condition(case) is True
        assert condition(corner_case) is False

    def test_picks_the_first_differing_attribute_in_declaration_order(self):
        case = Animal(name="dog", has_fur=True, leg_count=4)
        corner_case = Animal(name="cat", has_fur=True, leg_count=4)
        case_query = make_case_query(case, target=AnimalClass.mammal)
        last_evaluated_rule = RuleWithCornerCase(corner_case=corner_case)

        condition = Oracle().ask_for_conditions(case_query, last_evaluated_rule)

        assert "case.name" in condition.user_input

    def test_never_builds_a_condition_on_the_attribute_being_predicted(self):
        case = Animal(
            name="dog", has_fur=True, leg_count=4, animal_class=AnimalClass.mammal
        )
        corner_case = Animal(
            name="cat", has_fur=True, leg_count=4, animal_class=AnimalClass.bird
        )
        case_query = make_case_query(case, target=AnimalClass.mammal)
        last_evaluated_rule = RuleWithCornerCase(corner_case=corner_case)

        condition = Oracle().ask_for_conditions(case_query, last_evaluated_rule)

        assert "case.animal_class" not in condition.user_input
        assert "case.name" in condition.user_input


class TestOracleReturnsAnUnconditionalRuleWithNoCornerCase:

    def test_returns_an_unconditional_rule_when_no_rule_was_evaluated_yet(self):
        case = Animal(name="dog", has_fur=True, leg_count=4)
        case_query = make_case_query(case, target=AnimalClass.mammal)

        condition = Oracle().ask_for_conditions(case_query, last_evaluated_rule=None)

        assert condition(case) is True


class TestOracleRaisesWhenNoAttributeDistinguishesTheCaseFromItsCornerCase:

    def test_raises_when_the_case_and_corner_case_are_identical(self):
        case = Animal(name="dog", has_fur=True, leg_count=4)
        corner_case = Animal(name="dog", has_fur=True, leg_count=4)
        case_query = make_case_query(case, target=AnimalClass.mammal)
        last_evaluated_rule = RuleWithCornerCase(corner_case=corner_case)

        with pytest.raises(NoDistinguishingAttributeFound):
            Oracle().ask_for_conditions(case_query, last_evaluated_rule)


class TestOracleNeverGuessesAConclusion:

    def test_returns_the_known_target_as_the_conclusion(self):
        case = Animal(name="dog", has_fur=True, leg_count=4)
        case_query = make_case_query(case, target=AnimalClass.mammal)

        conclusion = Oracle().ask_for_conclusion(case_query)

        assert conclusion(case) == AnimalClass.mammal

    def test_raises_when_the_case_query_has_no_known_target(self):
        case = Animal(name="dog", has_fur=True, leg_count=4)
        case_query = CaseQuery(case, "animal_class", (AnimalClass,), True)

        with pytest.raises(OracleCannotInferConclusionWithoutTarget):
            Oracle().ask_for_conclusion(case_query)


class TestOracleFitsASingleClassRDRWithoutAnyPreRecordedAnswers:

    def test_fits_and_classifies_a_small_multi_class_dataset(self):
        cases = [
            Animal(name="dog", has_fur=True, leg_count=4),
            Animal(name="cat", has_fur=True, leg_count=4),
            Animal(name="parrot", has_fur=False, leg_count=2),
            Animal(name="eagle", has_fur=False, leg_count=2),
            Animal(name="spider", has_fur=False, leg_count=8),
        ]
        targets = [
            AnimalClass.mammal,
            AnimalClass.mammal,
            AnimalClass.bird,
            AnimalClass.bird,
            AnimalClass.arachnid,
        ]
        case_queries = [
            CaseQuery(case, "animal_class", (AnimalClass,), True, _target=target)
            for case, target in zip(cases, targets)
        ]

        scrdr = SingleClassRDR()
        scrdr.fit(case_queries, expert=Oracle())

        for case, target in zip(cases, targets):
            assert scrdr.classify(case) == target
