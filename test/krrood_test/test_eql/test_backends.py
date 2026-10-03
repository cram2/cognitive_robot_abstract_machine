import time
from copy import deepcopy
from datetime import datetime
from types import EllipsisType

import pytest
from sqlalchemy.orm import sessionmaker

import numpy as np
from probabilistic_model.distributions.distributions import DiracDeltaDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from random_events.variable import Continuous

from ..dataset.derived_attributes import Rectangle
from ..dataset.semantic_world_like_classes import Apple, Body
from krrood.entity_query_language.backends import (
    SQLAlchemyBackend,
    EntityQueryLanguageBackend,
    EntityQueryLanguageGenerativeBackend,
    ProbabilisticBackend,
)
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.exceptions import (
    SelectiveBackendCannotResolveEllipsisMatch,
)
from krrood.entity_query_language.factories import (
    variable,
    entity,
    a,
    an,
    variable_from,
)
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.query_graph import QueryGraph
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.entity_query_language.core.variable import Variable as KRROODVariable
from krrood.parametrization.model_registries import DictRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from random_events.interval import reals
from random_events.set import Set
from random_events.variable import Symbolic
from ..dataset.example_classes import (
    KRROODPose,
    KRROODPosition,
    KRROODPositions,
    KRROODOrientation,
    Atom,
    Element,
    TestEnum,
    NestedAction,
    EnumAction,
)
from ..dataset.ormatic_interface import *  # type: ignore
from ..dataset.value_comparisons import IsGreaterThan


def test_nested_action():

    apple = Apple("apple", 7)

    prob_q = a(NestedAction)(
        obj=variable(Apple, domain=[apple]),
        pose=a(KRROODPose)(
            position=a(KRROODPosition)(x=0.02, y=..., z=...),
            orientation=a(KRROODOrientation)(
                x=..., y=..., z=..., w=variable(float, domain=[0.0, 1.0])
            ),
        ),
    )

    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables
    names_of_actual_specified_parameters = [
        match.name_from_variable_access_path
        for match in parameters.statement._matches_with_variables_
        if (
            isinstance(match.assigned_variable, Literal)
            or isinstance(match.assigned_variable, KRROODVariable)
        )
        and not isinstance(match.assigned_value, EllipsisType)
        and not match.assigned_value is None
    ]

    assert names_of_actual_specified_parameters == [
        "NestedAction.obj",
        "NestedAction.pose.position.x",
        "NestedAction.pose.orientation.w",
    ]
    assert (
        variables["NestedAction.pose.position.x"].domain.simple_sets
        == reals().simple_sets
    )
    assert len(parameters.conditioning_assignments_from_literal_values) == 1

    assert 0.02 == (
        parameters.conditioning_assignments_from_literal_values.get(
            variables["NestedAction.pose.position.x"]
        )
    )


def test_selective_query_multiple_backends(session, database):

    p1 = KRROODPose(
        position=KRROODPosition(1, 0, 0), orientation=KRROODOrientation(0, 0, 0, 1)
    )
    p2 = KRROODPose(
        position=KRROODPosition(0, 1, 0), orientation=KRROODOrientation(0, 0, 0, 1)
    )

    python_domain = [p1, p2]

    daos = [to_dao(p1), to_dao(p2)]
    session.add_all(daos)
    session.commit()
    session_maker = sessionmaker(session.bind)

    pose_variable = variable(KRROODPose, python_domain)

    q = an(
        entity(pose_variable).where(
            pose_variable.position.x > 0.5,
        )
    )

    result = list(q.evaluate(backend=EntityQueryLanguageBackend()))
    assert len(result) == 1

    result = list(q.evaluate(backend=SQLAlchemyBackend(session_maker)))
    assert len(result) == 1


def test_probabilistic_backend_with_symbolic_expression():
    prob_q = a(KRROODPosition)(x=..., y=..., z=variable(int, domain=[1, 2, 3]))
    parameters = UnderspecifiedParameters(prob_q)
    assert parameters.variables["KRROODPosition.z"] == Symbolic(
        name="KRROODPosition.z", domain=Set.from_iterable([1, 2, 3])
    )


def test_underspecified_parameters_with_partly_symbolic_expression():
    prob_q = a(KRROODPosition)(x=..., y=..., z=variable(int, domain=[1, 2, 3]))
    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables
    assert len(variables) == 3
    assert variables["KRROODPosition.x"].domain == reals()
    assert variables["KRROODPosition.x"].is_numeric
    assert variables["KRROODPosition.y"].domain == reals()
    assert variables["KRROODPosition.y"].is_numeric
    assert variables["KRROODPosition.z"].domain == reals()
    assert variables["KRROODPosition.z"].is_numeric
    assert (
        len(parameters.truncation_assignments_from_krrood_variables[0].simple_sets) == 3
    )
    assert (
        len(
            parameters.truncation_assignments_from_krrood_variables[0]
            .simplify()
            .simple_sets
        )
        == 1
    )


def test_underspecified_parameters_with_full_symbolic_expression():
    prob_q = variable(KRROODPosition, domain=[KRROODPosition(1, 2, 3)])

    with pytest.raises(TypeError):
        UnderspecifiedParameters(prob_q)


def test_underspecified_parameters_with_only_underspecified():
    prob_q = a(KRROODPose)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
    )
    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables

    assert len(variables) == 7


def test_underspecified_parameters_with_only_literals():
    prob_q = a(KRROODPose)(
        position=KRROODPosition(1, 2, 3),
        orientation=KRROODOrientation(0, 0, 0, 1),
    )
    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables

    assert len(variables) == 7
    assert len(parameters.truncation_assignments_from_krrood_variables) == 0
    assert len(parameters.conditioning_assignments_from_literal_values) == 7


def test_enum_value_as_literal():
    prob_q = an(EnumAction)(
        obj=Body(
            "x",
        ),
        enum=TestEnum.OPTION_A,
    )
    pm_backend = ProbabilisticBackend(number_of_samples=10)
    values = list(prob_q.evaluate(backend=pm_backend))
    for value in values:
        assert value.enum == TestEnum.OPTION_A


def test_probabilistic_query_backend():
    prob_q = a(KRROODPose)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    prob_q.where(prob_q.position.x > 0.5)

    pm_backend = ProbabilisticBackend(number_of_samples=10)
    values = list(prob_q.evaluate(backend=pm_backend))
    for value in values:
        assert value.position.x > 0.5

    assert pm_backend.number_of_samples == len({v.position for v in values})


def test_generative_eql_backend():
    q = an(Atom)(
        element=...,
        type=variable_from([0, 1, 2]),
        charge=variable_from([0.0, 1.0, 2.0]),
        timestamp=datetime.datetime.now(),
    )
    # No explicit resolve(): the subject variable is available as soon as the pattern is
    # specified, so where() can reference it directly.
    q.where(q.type > q.charge)
    results = list(q.evaluate(backend=EntityQueryLanguageGenerativeBackend()))
    assert len(results) == 6
    for result in results:
        assert isinstance(result.element, Element)
        assert result.type > result.charge


def test_generative_backend_grounds_a_predicate_over_two_attributes_of_the_match():
    """
    A predicate in a match's where condition can take several attributes of the match,
    each standing for that attribute of the instance being checked.
    """
    values = [0.0, 1.0, 2.0]
    position = a(KRROODPosition)(
        x=variable_from(values), y=variable_from(values), z=0.0
    )
    position.where(IsGreaterThan(position.x, position.y))

    results = list(position.evaluate(backend=EntityQueryLanguageGenerativeBackend()))

    assert {(result.x, result.y) for result in results} == {
        (x, y) for x in values for y in values if x > y
    }


def test_generative_backend_grounds_a_predicate_taking_one_attribute_of_the_match_twice():
    """
    An attribute of the match filling two arguments of a predicate stands for that
    attribute of the instance being checked in both of them.
    """
    values = [0.0, 1.0, 2.0]
    position = a(KRROODPosition)(
        x=variable_from(values), y=variable_from(values), z=0.0
    )
    position.where(IsGreaterThan(position.x, position.x))

    results = list(position.evaluate(backend=EntityQueryLanguageGenerativeBackend()))

    assert results == []


def test_generative_backend_grounds_a_comparison_of_an_attribute_of_the_match_with_itself():
    """
    An attribute of the match on both sides of a comparison stands for that attribute of
    the instance being checked on both sides.
    """
    values = [0.0, 1.0, 2.0]
    position = a(KRROODPosition)(
        x=variable_from(values), y=variable_from(values), z=0.0
    )
    position.where(position.x > position.x)

    results = list(position.evaluate(backend=EntityQueryLanguageGenerativeBackend()))

    assert results == []


def test_selective_backend_rejects_match_with_ellipsis_attribute():
    q = a(KRROODPosition)(x=..., y=1.0, z=2.0)
    with pytest.raises(SelectiveBackendCannotResolveEllipsisMatch):
        list(q.evaluate(backend=EntityQueryLanguageBackend()))


def test_selective_backend_rejects_match_with_nested_ellipsis_attribute():
    q = a(KRROODPose)(
        position=a(KRROODPosition)(x=..., y=1.0, z=2.0),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    with pytest.raises(SelectiveBackendCannotResolveEllipsisMatch):
        list(q.evaluate(backend=EntityQueryLanguageBackend()))


def test_selective_backend_rejects_match_with_ellipsis_element_in_plain_list():
    q = a(KRROODPositions)(
        positions=[KRROODPosition(1, 2, 3)],
        some_strings=["a", ..., "c"],
    )
    with pytest.raises(SelectiveBackendCannotResolveEllipsisMatch):
        list(q.evaluate(backend=EntityQueryLanguageBackend()))


def test_selective_backend_accepts_match_without_ellipsis_attribute():
    apple = Apple("apple", 7)
    q = an(Apple)(name="apple", size=7).from_([apple])
    assert list(q.evaluate(backend=EntityQueryLanguageBackend())) == [apple]


# %% generated instances satisfy the pattern they were generated from


RECTANGLE_SIDES = [(3.0, 4.0), (2.0, 2.0)]
"""
The width and height of every rectangle the rectangle model can sample.
"""


@pytest.fixture
def rectangle_model() -> ProbabilisticCircuit:
    """
    :return: A model over the sides of a rectangle that samples each pair of
        :data:`RECTANGLE_SIDES` with equal probability, and knows nothing about the area
        those sides enclose.
    """
    width = Continuous("Rectangle.width")
    height = Continuous("Rectangle.height")
    model = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=model)
    for rectangle_width, rectangle_height in RECTANGLE_SIDES:
        sides = ProductUnit(probabilistic_circuit=model)
        sides.add_subcircuit(
            leaf(
                DiracDeltaDistribution(
                    variable=width, location=rectangle_width, density_cap=1.0
                ),
                model,
            )
        )
        sides.add_subcircuit(
            leaf(
                DiracDeltaDistribution(
                    variable=height, location=rectangle_height, density_cap=1.0
                ),
                model,
            )
        )
        root.add_subcircuit(sides, np.log(1 / len(RECTANGLE_SIDES)))
    return model


def test_probabilistic_backend_rejects_samples_contradicting_a_derived_attribute(
    rectangle_model,
):
    first_width, first_height = RECTANGLE_SIDES[0]
    query = a(Rectangle)(width=..., height=..., area=first_width * first_height)
    backend = ProbabilisticBackend(
        number_of_samples=20, model_registry=DictRegistry({Rectangle: rectangle_model})
    )
    rectangles = query.evaluate(backend=backend)
    assert {(rectangle.width, rectangle.height) for rectangle in rectangles} == {
        RECTANGLE_SIDES[0]
    }


def test_enumerating_backend_rejects_instances_contradicting_a_derived_attribute():
    query = a(Rectangle)(
        width=variable(int, [1, 2, 3, 4]), height=variable(int, [3, 4, 6]), area=12
    )
    rectangles = query.evaluate(backend=EntityQueryLanguageGenerativeBackend())
    assert {(rectangle.width, rectangle.height) for rectangle in rectangles} == {
        (2, 6),
        (3, 4),
        (4, 3),
    }


def test_enumerating_backend_keeps_instances_whose_factory_renames_a_stated_value():
    query = a(Rectangle.from_sides)(
        first_side=variable(int, [3, 4]), second_side=RECTANGLE_SIDES[0][1]
    )
    rectangles = query.evaluate(backend=EntityQueryLanguageGenerativeBackend())
    assert {(rectangle.width, rectangle.height) for rectangle in rectangles} == {
        (3, RECTANGLE_SIDES[0][1]),
        (4, RECTANGLE_SIDES[0][1]),
    }
