from dataclasses import dataclass
from typing import List, Any

import pytest

from krrood.parametrization.exceptions import InvalidEllipsis
from ..dataset.semantic_world_like_classes import Body
from krrood.entity_query_language.factories import (
    variable,
    a,
    an,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from random_events.interval import singleton, reals

from ..dataset.example_classes import (
    KRROODPosition,
    TestEnum,
    ListOfEnum,
    EnumAction,
    ActionWithMissingAggregationsMixin,
)
from ..dataset.semantic_world_like_classes import Cabinet, Container, Body, Handle


def test_enum_domain():
    """
    Test that a KRROOD variable with an Enum domain is correctly handled.
    """
    prob_q = an(EnumAction)(
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
    prob_q = an(EnumAction)(
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
    prob_q = a(KRROODPosition)(x=1.0, y=..., z=variable(float, domain=[2.0, 3.0]))
    parameters = UnderspecifiedParameters(prob_q)
    assignments = parameters.conditioning_assignments_from_literal_values

    variables = parameters.variables
    # The variable name for 'x' literal should be 'KRROODPosition.x'
    x_var = variables.get("KRROODPosition.x")

    assert x_var in assignments
    assert assignments[x_var] == 1.0
    assert len(assignments) == 1


def test_union_types_easy():
    prob_q = a(KRROODPosition)(x=..., y=..., z=...)
    prob_q.resolve()
    prob_q.where(
        prob_q.variable.x < 5.0,
    )
    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables
    assert variables["KRROODPosition.x"].domain == reals()


def test_union_types():
    prob_q = a(KRROODPosition)(x=..., y=..., z=variable(int, domain=[10, 20]))
    prob_q.resolve()
    prob_q.where(prob_q.variable.x < 5.0)

    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables
    assert variables["KRROODPosition.x"].domain == reals()


def test_domain_object_with_exchangeable_parts_but_no_aggregation_mixin_is_skipped():
    """
    Prove that a domain object whose class has exchangeable parts but does not inherit
    from HasExchangeablePartAggregations produces no variables for those parts and
    raises no exception.
    """
    container = Container(name="container")
    cabinet = Cabinet(container=container)
    prob_q = an(ActionWithMissingAggregationsMixin)(
        domain_object=variable(Cabinet, [cabinet])
    )
    parameters = UnderspecifiedParameters(prob_q)
    assert not any("drawers" in name for name in parameters.variables)


def test_iterable_of_primitives_produces_indexed_variables():
    """
    Each primitive element in a list literal must produce a distinct conditioning
    variable named ``ClassName.field[index]`` in ``parameters.variables``.

    Regression test for a bug in
    ``_extract_variables_from_iterable_literal`` where the return value
    of ``_handle_compatible_type_literal`` was discarded, so primitive
    elements were silently dropped from the returned variable dict.

    A ``List[Any]`` field is used because ``issubclass(Any,
    compatible_types)`` returns ``False``, which bypasses the outer
    type-mismatch guard and lets the list reach
    ``_extract_variables_from_iterable_literal`` while still having
    float elements that satisfy ``isinstance(element,
    compatible_types)``.
    """

    @dataclass
    class FloatMeasurements:
        readings: List[Any]

    prob_q = a(FloatMeasurements)(readings=[1.0, 2.0, 3.0])
    parameters = UnderspecifiedParameters(prob_q)

    assert "FloatMeasurements.readings[0]" in parameters.variables
    assert "FloatMeasurements.readings[1]" in parameters.variables
    assert "FloatMeasurements.readings[2]" in parameters.variables


def test_list_of_enum_field_produces_indexed_variables():
    """
    Assigning a literal list of enum values to a ``List[EnumType]`` field must produce
    one conditioning variable per element, named ``ClassName.field[index]``.

    Regression test for a bug in the type-mismatch guard in
    ``_handle_literal_attribute_match``: the guard fired for any value that was not
    itself in ``compatible_types``, including lists. Because the EQL system strips the
    ``List[...]`` container and exposes only the element type as ``_type_``, a
    ``List[TestEnum]`` field and a bare ``TestEnum`` field were indistinguishable, so
    assigning a list to the former raised ``TypeError`` instead of forwarding to
    ``_extract_variables_from_iterable_literal``.
    """
    prob_q = a(ListOfEnum)(list_of_enum=[TestEnum.OPTION_A, TestEnum.OPTION_B])
    parameters = UnderspecifiedParameters(prob_q)

    assert "ListOfEnum.list_of_enum[0]" in parameters.variables
    assert "ListOfEnum.list_of_enum[1]" in parameters.variables
