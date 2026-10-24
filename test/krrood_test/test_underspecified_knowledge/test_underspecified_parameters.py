from dataclasses import dataclass
from typing import List, Any

import pytest

from krrood.parametrization.exceptions import (
    AttributeNotParameterized,
    InvalidEllipsis,
    ModelVariableNotBound,
    ValueOutsideVariableDomain,
)
from krrood.parametrization.model_registries import FullyFactorizedRegistry
from ..dataset.semantic_world_like_classes import Body
from krrood.entity_query_language.factories import (
    variable,
    a,
    an,
)
from krrood.entity_query_language.core.variable import Variable as KRROODVariable
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from random_events.interval import singleton, reals

from ..dataset.example_classes import (
    KRROODOrientation,
    KRROODPose,
    KRROODPosition,
    TestEnum,
    ListOfEnum,
    EnumAction,
    ActionWithMissingAggregationsMixin,
)
from ..dataset.semantic_world_like_classes import Cabinet, Container, Body, Handle
from ..pattern_variables import find_assigned_variable


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
    prob_q.where(
        prob_q.x < 5.0,
    )
    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables
    assert variables["KRROODPosition.x"].domain == reals()


def test_union_types():
    prob_q = a(KRROODPosition)(x=..., y=..., z=variable(int, domain=[10, 20]))
    prob_q.where(prob_q.x < 5.0)

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


# %% variable lookup by attribute


def test_variable_of_nested_attribute_is_the_variable_its_literal_conditions():
    x = 1.0
    y = 2.0
    pose = a(KRROODPose)(position=a(KRROODPosition)(x=x, y=y, z=...))
    parameters = UnderspecifiedParameters(pose)
    conditions = parameters.conditioning_assignments_from_literal_values

    assert conditions[parameters.variable_of(pose.position.x)] == x
    assert conditions[parameters.variable_of(pose.position.y)] == y


def test_variable_of_attribute_missing_from_the_pattern_raises():
    position = a(KRROODPosition)(x=1.0, y=...)
    parameters = UnderspecifiedParameters(position)

    with pytest.raises(AttributeNotParameterized):
        parameters.variable_of(position.z)


# %% translating between model samples and bindings


def test_model_sample_from_bindings_inverts_bindings_from_model_samples():
    query = an(EnumAction)(obj=Body(name="body"), enum=...)
    parameters = UnderspecifiedParameters(query)
    model = parameters.resolve_conditioned_and_truncated_model(
        FullyFactorizedRegistry().get_model(parameters)
    )
    samples = model.sample(1)
    [bindings] = parameters.bindings_from_model_samples(model.variables, samples)
    assert (
        parameters.model_sample_from_bindings(model.variables, bindings).tolist()
        == samples.tolist()
    )


def test_model_sample_from_bindings_rejects_an_unbound_model_variable():
    query = an(EnumAction)(obj=Body(name="body"), enum=...)
    parameters = UnderspecifiedParameters(query)
    model = FullyFactorizedRegistry().get_model(parameters)
    with pytest.raises(ModelVariableNotBound):
        parameters.model_sample_from_bindings(model.variables, {})


@dataclass
class PoseOverPositions:
    """
    A pose pattern whose position is a variable ranging over positions.
    """

    positions: list[KRROODPosition]
    """
    The positions the variable ranges over.
    """

    position: KRROODVariable
    """
    The variable the pattern assigns to the pose's position.
    """

    query: Match
    """
    The pose pattern.
    """


@pytest.fixture
def pose_over_two_positions() -> PoseOverPositions:
    """
    :return: A pose pattern whose position ranges over two positions.
    """
    positions = [KRROODPosition(1.0, 2.0, 3.0), KRROODPosition(4.0, 5.0, 6.0)]
    position = variable(KRROODPosition, positions)
    return PoseOverPositions(
        positions=positions,
        position=position,
        query=a(KRROODPose)(
            position=position, orientation=KRROODOrientation(0.0, 0.0, 0.0, 1.0)
        ),
    )


def test_model_sample_from_bindings_derives_the_features_of_a_bound_domain_object(
    pose_over_two_positions,
):
    positions = pose_over_two_positions.positions
    position = pose_over_two_positions.position
    parameters = UnderspecifiedParameters(pose_over_two_positions.query)
    model = parameters.resolve_conditioned_and_truncated_model(
        FullyFactorizedRegistry().get_model(parameters)
    )
    bindings = {position._id_: positions[1]}
    samples = parameters.model_sample_from_bindings(model.variables, bindings)
    [sample] = samples
    sample_values = {
        model_variable.name: value
        for model_variable, value in zip(model.variables, sample)
    }
    assert [
        sample_values["KRROODPose.position.x"],
        sample_values["KRROODPose.position.y"],
        sample_values["KRROODPose.position.z"],
    ] == [positions[1].x, positions[1].y, positions[1].z]
    assert list(parameters.bindings_from_model_samples(model.variables, samples)) == [
        bindings
    ]


# %% bindings outside the domain of a model variable


def test_model_sample_from_bindings_refuses_an_object_outside_the_domain(
    pose_over_two_positions,
):
    position = pose_over_two_positions.position
    parameters = UnderspecifiedParameters(pose_over_two_positions.query)
    identifier = parameters.variables["KRROODPose.position"]
    outside_position = KRROODPosition(7.0, 8.0, 9.0)
    with pytest.raises(ValueOutsideVariableDomain) as error:
        parameters.model_sample_from_bindings(
            [identifier], {position._id_: outside_position}
        )
    assert error.value.variable == identifier
    assert error.value.value is outside_position


def test_model_sample_from_bindings_refuses_a_feature_of_an_object_outside_the_domain(
    pose_over_two_positions,
):
    position = pose_over_two_positions.position
    parameters = UnderspecifiedParameters(pose_over_two_positions.query)
    feature = parameters.variables["KRROODPose.position.x"]
    outside_position = KRROODPosition(7.0, 8.0, 9.0)
    with pytest.raises(ValueOutsideVariableDomain) as error:
        parameters.model_sample_from_bindings(
            [feature], {position._id_: outside_position}
        )
    assert error.value.variable == feature
    assert error.value.value is outside_position


def test_model_sample_from_bindings_refuses_a_value_that_is_no_element_of_a_symbolic_variable():
    query = an(EnumAction)(obj=Body(name="body"), enum=...)
    parameters = UnderspecifiedParameters(query)
    enum = parameters.variables["EnumAction.enum"]
    with pytest.raises(ValueOutsideVariableDomain) as error:
        parameters.model_sample_from_bindings(
            [enum],
            {
                find_assigned_variable(
                    query, "EnumAction.enum"
                )._id_: TestEnum.OPTION_A.value
            },
        )
    assert error.value.variable == enum
    assert error.value.value == TestEnum.OPTION_A.value
