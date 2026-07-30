import numpy as np
import pytest

from krrood.entity_query_language.factories import (
    a,
    and_,
    for_all,
    not_,
    or_,
    variable,
)
from krrood.parametrization.exceptions import (
    WhereExpressionHasNoRandomEventRepresentation,
    WhereExpressionIsFirstOrder,
)
from krrood.parametrization.random_events_translator import (
    WhereExpressionToRandomEventTranslator,
)
from random_events.interval import closed, closed_open, open, singleton
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Continuous
from ..dataset.example_classes import (
    KRROODOrientation,
    KRROODPose,
    KRROODPosition,
)
from ..dataset.ormatic_interface import *  # type: ignore

# %% helpers


def underspecified_pose():
    """
    :return: A pose whose position and orientation are underspecified.
    """
    return a(KRROODPose)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
    )


def translate(condition) -> Event:
    """
    :param condition: The where condition to translate.
    :return: The random event that corresponds to the condition.
    """
    return WhereExpressionToRandomEventTranslator(condition).translate()


def translate_where_conditions(query) -> Event:
    """
    :param query: The query whose where conditions to translate.
    :return: The random event that corresponds to the where conditions.
    """
    return translate(and_(*query._where_conditions_))


def assert_equal_events(actual: Event, expected: Event) -> None:
    """
    Assert that two events describe the same set of points.

    :param actual: The event under test.
    :param expected: The event it is expected to equal.
    """
    assert (expected - actual).is_empty()
    assert (actual - expected).is_empty()


# %% conjunctions of comparators


def test_underspecification_with_where():
    pose = underspecified_pose()
    query = pose.where(
        pose.variable.position.y > 0.0,
        pose.variable.position.x == 0.0,
        pose.variable.position.y < 10.0,
        pose.variable.position.z >= -1.0,
        pose.variable.position.z <= 1.0,
        pose.variable.orientation.x != 1.0,
    )

    translated = translate_where_conditions(query)

    result_by_hand = SimpleEvent.from_data(
        {
            Continuous("KRROODPose.orientation.x"): ~singleton(1.0),
            Continuous("KRROODPose.position.y"): open(0.0, 10),
            Continuous("KRROODPose.position.z"): closed(-1.0, 1.0),
            Continuous("KRROODPose.position.x"): singleton(0.0),
        }
    )

    assert result_by_hand.as_composite_set() == translated


# %% disjunctions of conjunctions


def test_disjunction_of_conjunctions():
    pose_variable = underspecified_pose().variable

    translated = translate(
        or_(
            pose_variable.position.x == 0,
            and_(
                pose_variable.position.z >= -1,
                pose_variable.position.z <= 1,
                pose_variable.position.y < 10,
            ),
            and_(pose_variable.orientation.z > 0),
        )
    )

    variables = [
        Continuous("KRROODPose.position.x"),
        Continuous("KRROODPose.position.y"),
        Continuous("KRROODPose.position.z"),
        Continuous("KRROODPose.orientation.z"),
    ]
    [position_x, position_y, position_z, orientation_z] = variables

    fixed_x = SimpleEvent.from_data({position_x: singleton(0.0)})
    fixed_x.fill_missing_variables(variables)
    bounded_y_and_z = SimpleEvent.from_data(
        {
            position_z: closed(-1.0, 1.0),
            position_y: closed_open(-np.inf, 10.0),
        }
    )
    bounded_y_and_z.fill_missing_variables(variables)
    positive_orientation_z = SimpleEvent.from_data({orientation_z: open(0.0, np.inf)})
    positive_orientation_z.fill_missing_variables(variables)

    assert_equal_events(
        translated,
        Event.from_simple_sets(fixed_x, bounded_y_and_z, positive_orientation_z),
    )


# %% conditions that are not in disjunctive normal form


def test_where_condition_outside_disjunctive_normal_form():
    pose = underspecified_pose()
    pose_variable = pose.variable
    query = pose.where(
        or_(
            pose_variable.position.x > 0.0,
            pose_variable.position.y > 0.0,
        ),
        pose_variable.position.x < 10.0,
    )

    translated = translate_where_conditions(query)

    position_x = Continuous("KRROODPose.position.x")
    position_y = Continuous("KRROODPose.position.y")
    variables = [position_x, position_y]

    bounded_x = SimpleEvent.from_data({position_x: open(0.0, 10.0)})
    bounded_x.fill_missing_variables(variables)
    positive_y = SimpleEvent.from_data(
        {
            position_x: closed_open(-np.inf, 10.0),
            position_y: open(0.0, np.inf),
        }
    )
    positive_y.fill_missing_variables(variables)

    assert_equal_events(translated, Event.from_simple_sets(bounded_x, positive_y))


def test_disjunction_nested_in_conjunction_keeps_shared_variable():
    pose_variable = underspecified_pose().variable

    translated = translate(
        and_(
            pose_variable.position.x > 0.0,
            or_(
                pose_variable.position.x < 1.0,
                pose_variable.position.x > 5.0,
            ),
        )
    )

    position_x = Continuous("KRROODPose.position.x")
    lower_part = SimpleEvent.from_data({position_x: open(0.0, 1.0)})
    upper_part = SimpleEvent.from_data({position_x: open(5.0, np.inf)})

    assert_equal_events(translated, Event.from_simple_sets(lower_part, upper_part))


# %% negated conditions


def test_negation_equals_inverted_comparator():
    pose_variable = underspecified_pose().variable

    translated = translate(not_(pose_variable.position.x > 0.0))

    assert_equal_events(translated, translate(pose_variable.position.x <= 0.0))


def test_negated_conjunction_is_disjunction_of_negations():
    pose_variable = underspecified_pose().variable

    translated = translate(
        not_(
            and_(
                pose_variable.position.x > 0.0,
                pose_variable.position.y > 0.0,
            )
        )
    )

    assert_equal_events(
        translated,
        translate(
            or_(
                pose_variable.position.x <= 0.0,
                pose_variable.position.y <= 0.0,
            )
        ),
    )


# %% conditions without a random event representation


def test_quantified_where_condition():
    pose = underspecified_pose()
    position_variable = variable(KRROODPosition, [KRROODPosition(1.0, 2.0, 3.0)])
    query = pose.where(
        for_all(position_variable, position_variable.x > 0.0),
    )

    with pytest.raises(WhereExpressionIsFirstOrder):
        translate_where_conditions(query)


def test_comparison_between_two_variables_is_refused():
    pose_variable = underspecified_pose().variable

    with pytest.raises(WhereExpressionHasNoRandomEventRepresentation):
        translate(pose_variable.position.x > pose_variable.position.y)
