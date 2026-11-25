"""
Tests that every evaluation of a query reads its domain as it is at that evaluation, so
objects added after an earlier evaluation are found.
"""

from krrood.entity_query_language.factories import a, variable
from ...dataset.example_classes import KRROODPosition


# %% registry domain


def test_match_over_the_registry_finds_an_instance_created_after_an_evaluation():
    first = KRROODPosition(x=1.0, y=0.0, z=0.0)
    query = a(KRROODPosition)(y=0.0)
    assert query.tolist() == [first]

    second = KRROODPosition(x=2.0, y=0.0, z=0.0)

    assert query.tolist() == [first, second]


def test_variable_over_the_registry_finds_an_instance_created_after_an_evaluation():
    first = KRROODPosition(x=1.0, y=0.0, z=0.0)
    position = variable(KRROODPosition, domain=None)
    assert position.tolist() == [first]

    second = KRROODPosition(x=2.0, y=0.0, z=0.0)

    assert position.tolist() == [first, second]


# %% explicit domain


def test_match_over_a_list_finds_an_element_appended_after_an_evaluation():
    first = KRROODPosition(x=1.0, y=0.0, z=0.0)
    positions = [first]
    query = a(KRROODPosition).from_(domain=positions)
    assert query.tolist() == [first]

    second = KRROODPosition(x=2.0, y=0.0, z=0.0)
    positions.append(second)

    assert query.tolist() == [first, second]
