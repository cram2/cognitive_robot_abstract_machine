from dataclasses import dataclass

import pytest

from krrood.entity_query_language.backends import EntityQueryLanguageBackend
from krrood.entity_query_language.operators.causal import CausesEffect
from krrood.entity_query_language.exceptions import (
    CausesEffectRequiresEqualityComparator,
)
from krrood.entity_query_language.factories import a, and_, not_
from krrood.entity_query_language.operators.core_logical_operators import AND


@dataclass
class Pick:
    arm: float
    status: str


# %% construction


def test_causes_effect_accepts_a_literal_comparator():
    arm = a(Pick)(arm=..., status="idle").variable
    CausesEffect(arm.status == "SUCCESS")  # does not raise


def test_causes_effect_accepts_a_conjunction_of_literal_comparators():
    arm = a(Pick)(arm=..., status="idle").variable
    CausesEffect(and_(arm.status == "SUCCESS", arm.arm == 0.3))  # does not raise


def test_causes_effect_rejects_a_comparison_between_two_attributes():
    first_pick = a(Pick)(arm=..., status="idle").variable
    second_pick = a(Pick)(arm=..., status="idle").variable
    with pytest.raises(CausesEffectRequiresEqualityComparator):
        CausesEffect(first_pick.arm == second_pick.arm)


def test_causes_effect_rejects_an_inequality_comparator():
    arm = a(Pick)(arm=..., status="idle").variable
    with pytest.raises(CausesEffectRequiresEqualityComparator):
        CausesEffect(arm.arm > 0.3)


def test_causes_effect_rejects_an_ellipsis_valued_comparator():
    arm = a(Pick)(arm=..., status="idle").variable
    with pytest.raises(CausesEffectRequiresEqualityComparator):
        CausesEffect(arm.status == ...)


def test_causes_effect_rejects_a_set_valued_comparator():
    arm = a(Pick)(arm=..., status="idle").variable
    with pytest.raises(CausesEffectRequiresEqualityComparator):
        CausesEffect(arm.status == {"SUCCESS", "FAILURE"})


# %% transparent evaluation


def test_causes_effect_selects_the_same_instances_as_a_plain_where():
    matching = Pick(0.3, "SUCCESS")
    non_matching = Pick(0.7, "FAILURE")

    plain = a(Pick).from_([matching, non_matching])
    plain.where(plain.variable.status == "SUCCESS")

    wrapped = a(Pick).from_([matching, non_matching])
    wrapped.causes_effect(wrapped.variable.status == "SUCCESS")

    backend = EntityQueryLanguageBackend()
    assert list(plain.evaluate(backend=backend)) == list(
        wrapped.evaluate(backend=backend)
    )


def test_causes_effect_negation_still_selects_the_complement():
    matching = Pick(0.3, "SUCCESS")
    non_matching = Pick(0.7, "FAILURE")

    wrapped = a(Pick).from_([matching, non_matching])
    wrapped.where(not_(CausesEffect(wrapped.variable.status == "SUCCESS")))

    backend = EntityQueryLanguageBackend()
    assert list(wrapped.evaluate(backend=backend)) == [non_matching]


# %% Match.causes_effect sugar


def test_match_causes_effect_is_sugar_for_where_with_causes_effect():
    match = a(Pick)(arm=..., status=...)
    match.causes_effect(match.variable.status == "SUCCESS")
    [condition] = match._where_conditions_
    assert isinstance(condition, CausesEffect)


def test_match_causes_effect_ands_multiple_conditions():
    match = a(Pick)(arm=..., status=...)
    match.causes_effect(match.variable.status == "SUCCESS", match.variable.arm == 0.3)
    [condition] = match._where_conditions_
    assert isinstance(condition, CausesEffect)
    assert isinstance(condition._child_, AND)
