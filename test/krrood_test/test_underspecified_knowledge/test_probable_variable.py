import numpy as np

from krrood.entity_query_language.factories import (
    variable,
    entity,
    and_,
    or_,
    underspecified,
)
from krrood.parametrization.random_events_translator import (
    WhereExpressionToRandomEventTranslator,
    is_disjunctive_normal_form,
)
from random_events.interval import singleton, open, closed, closed_open
from random_events.product_algebra import SimpleEvent, Event
from random_events.variable import Continuous
from ..dataset.example_classes import Pose, Position, Orientation, Positions
from ..dataset.ormatic_interface import *  # type: ignore


def test_underspecification_with_where():
    underspecified_pose = underspecified(Pose)(
        position=underspecified(Position)(x=..., y=..., z=...),
        orientation=underspecified(Orientation)(x=..., y=..., z=..., w=...),
    )
    underspecified_pose.expression
    q = underspecified_pose.where(
        underspecified_pose.variable.position.y > 0.0,
        underspecified_pose.variable.position.x == 0.0,
        underspecified_pose.variable.position.y < 10.0,
        underspecified_pose.variable.position.z >= -1.0,
        underspecified_pose.variable.position.z <= 1.0,
        underspecified_pose.variable.orientation.x != 1.0,
    )

    t = WhereExpressionToRandomEventTranslator(
        and_(*q._where_conditions_),
    )
    r = t.translate()

    result_by_hand = SimpleEvent(
        {
            Continuous("Pose.orientation.x"): ~singleton(1.0),
            Continuous("Pose.position.y"): open(0.0, 10),
            Continuous("Pose.position.z"): closed(-1.0, 1.0),
            Continuous("Pose.position.x"): singleton(0.0),
        }
    )

    assert result_by_hand.as_composite_set() == r


def test_dnf_checking():
    pose_variable = variable(Pose, None)

    q1 = entity(pose_variable).where(
        and_(
            or_(
                pose_variable.position.y > 0,
                pose_variable.position.x == 0,
            ),
            or_(
                pose_variable.position.z >= -1,
                pose_variable.position.x == 0,
            ),
        )
    )

    assert not is_disjunctive_normal_form(q1._conditions_root_)

    underspecified_pose = underspecified(Pose)(
        position=underspecified(Position)(x=..., y=..., z=...),
        orientation=underspecified(Orientation)(x=..., y=..., z=..., w=...),
    )
    underspecified_pose.expression
    pose_variable = underspecified_pose.variable
    q2 = underspecified_pose.where(
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

    where_expression = and_(*q2._where_conditions_)
    assert is_disjunctive_normal_form(where_expression)

    t = WhereExpressionToRandomEventTranslator(
        where_expression,
    )
    translated = t.translate()

    variables = [
        Continuous("Pose.position.x"),
        Continuous("Pose.position.y"),
        Continuous("Pose.position.z"),
        Continuous("Pose.orientation.z"),
    ]
    [p_x, p_y, p_z, o_z] = variables

    e1 = SimpleEvent(
        {
            p_x: singleton(0.0),
        }
    )
    e1.fill_missing_variables(variables)
    e2 = SimpleEvent(
        {
            p_z: closed(-1.0, 1.0),
            p_y: closed_open(-np.inf, 10.0),
        }
    )
    e2.fill_missing_variables(variables)
    e3 = SimpleEvent({o_z: open(0.0, np.inf)})
    e3.fill_missing_variables(variables)

    result_by_hand = Event(e1, e2, e3)

    assert (result_by_hand - translated).is_empty()
    assert (translated - result_by_hand).is_empty()


def test_query_writing_with_match_and_copy():
    var = underspecified(Pose)(
        position=underspecified(Position)(x=0.1, y=..., z=...), orientation=None
    )

    obj = var.construct_instance()
    assert obj.position.x == 0.1
    assert obj.position.y == ...
    assert obj.position.z == ...
    assert obj.orientation is None


def test_probable_variable_with_concrete_kwarg():
    probable_pose = underspecified(Pose)

    prob_q = probable_pose(
        position=underspecified(Position)(x=..., y=..., z=...),
        orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    prob_q.expression
    prob_q.where(probable_pose.variable.position.x > 0.5)
    prob_q.expression.build()
    instance = prob_q.construct_instance()

    correct_instance = Pose(Position(..., ..., ...), Orientation(0.0, 0.0, 0.0, 1.0))

    assert instance == correct_instance
    assert len(list(prob_q.literals)) == 4


def test_new_underspecified_with_factory():

    probable_pose = underspecified(Pose)
    prob_q = probable_pose(
        position=underspecified(Position.from_abc)(a=..., b=..., c=...),
        orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    prob_q.expression
    prob_q = prob_q.where(probable_pose.variable.position.x > 0.5)
    prob_q.expression.build()
    r = prob_q.construct_instance()
    assert r == Pose(Position(..., ..., ...), Orientation(0.0, 0.0, 0.0, 1.0))


def test_underspecified_with_list():
    q = underspecified(Positions)(
        positions=[underspecified(Position)(x=1.0, y=..., z=...), Position(1, 2, 3)],
        some_strings=["a", "b"],
    )
    q.expression.build()

    for literal in q.literals:
        if literal.assigned_value is ...:
            literal.assigned_variable._value_ = 0.0

    q._update_kwargs_from_literal_values()

    assert q.kwargs["positions"][0].kwargs == {"x": 1.0, "y": 0.0, "z": 0.0}
    assert q.factory == Positions
    r = q.construct_instance()
    assert r == Positions([Position(1.0, 0.0, 0.0), Position(1, 2, 3)], ["a", "b"])
