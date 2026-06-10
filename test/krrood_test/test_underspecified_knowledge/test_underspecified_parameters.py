import pytest

from krrood.parametrization.exceptions import InvalidEllipsis
from ..dataset.semantic_world_like_classes import Body
from krrood.entity_query_language.factories import (
    variable,
    underspecified,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from random_events.interval import singleton, reals

from ..dataset.example_classes import (
    KRROODPosition,
    TestEnum,
    EnumAction,
)


def test_enum_domain():
    """
    Test that a KRROOD variable with an Enum domain is correctly handled.
    """
    prob_q = underspecified(EnumAction)(
        obj=Body(name="body"),
        enum=variable(
            TestEnum, [TestEnum.OPTION_A, TestEnum.OPTION_B, TestEnum.OPTION_C]
        ),
    )
    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables

    assert len(variables) == 2
    assert len(parameters.truncation_assignments_from_krrood_variables) == 1


def test_invalid_ellipsis():
    prob_q = underspecified(EnumAction)(
        obj=...,
        enum=variable(
            TestEnum, [TestEnum.OPTION_A, TestEnum.OPTION_B, TestEnum.OPTION_C]
        ),
    )
    with pytest.raises(InvalidEllipsis):
        parameters = UnderspecifiedParameters(prob_q)


def test_assignments_for_conditioning():
    """
    Test that assignments_for_conditioning returns only literal facts.
    """
    prob_q = underspecified(KRROODPosition)(
        x=1.0, y=..., z=variable(float, domain=[2.0, 3.0])
    )
    parameters = UnderspecifiedParameters(prob_q)
    assignments = parameters.conditioning_assignments_from_literal_values

    variables = parameters.variables
    # The variable name for 'x' literal should be 'KRROODPosition.x'
    x_var = variables.get("KRROODPosition.x")

    assert x_var in assignments
    assert assignments[x_var] == 1.0
    assert len(assignments) == 1


def test_union_types_easy():
    prob_q = underspecified(KRROODPosition)(x=..., y=..., z=...)
    prob_q.resolve()
    prob_q.where(
        prob_q.variable.x < 5.0,
    )
    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables
    assert variables["KRROODPosition.x"].domain == reals()


def test_union_types():
    prob_q = underspecified(KRROODPosition)(
        x=..., y=..., z=variable(int, domain=[10, 20])
    )
    prob_q.resolve()
    prob_q.where(prob_q.variable.x < 5.0)

    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables
    assert variables["KRROODPosition.x"].domain == reals()
