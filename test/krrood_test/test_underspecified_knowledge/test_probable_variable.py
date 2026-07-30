from krrood.entity_query_language.factories import (
    a,
)
from ..dataset.example_classes import (
    KRROODPose,
    KRROODPosition,
    KRROODOrientation,
    KRROODPositions,
)
from ..dataset.ormatic_interface import *  # type: ignore


def test_query_writing_with_match_and_copy():
    var = a(KRROODPose)(
        position=a(KRROODPosition)(x=0.1, y=..., z=...), orientation=None
    )

    obj = var.construct_instance()
    assert obj.position.x == 0.1
    assert obj.position.y == ...
    assert obj.position.z == ...
    assert obj.orientation is None


def test_probable_variable_with_concrete_kwarg():
    prob_q = a(KRROODPose)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    prob_q.where(prob_q.variable.position.x > 0.5)
    instance = prob_q.construct_instance()

    correct_instance = KRROODPose(
        KRROODPosition(..., ..., ...), KRROODOrientation(0.0, 0.0, 0.0, 1.0)
    )

    assert instance == correct_instance
    assert len(list(prob_q.matches_with_variables)) == 4


def test_new_underspecified_with_factory():

    prob_q = a(KRROODPose)(
        position=a(KRROODPosition.from_abc, target_type=KRROODPosition)(
            a=..., b=..., c=...
        ),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    prob_q.where(prob_q.variable.position.x > 0.5)
    prob_q.expression.build()
    r = prob_q.construct_instance()
    assert r == KRROODPose(
        KRROODPosition(..., ..., ...), KRROODOrientation(0.0, 0.0, 0.0, 1.0)
    )


def test_underspecified_with_list():
    q = a(KRROODPositions)(
        positions=[
            a(KRROODPosition)(x=1.0, y=..., z=...),
            KRROODPosition(1, 2, 3),
        ],
        some_strings=["a", "b"],
    )

    for literal in q.matches_with_variables:
        if literal.assigned_value is ...:
            literal.assigned_variable._value_ = 0.0

    q._update_kwargs_from_literal_values()

    assert q.kwargs["positions"][0].kwargs == {"x": 1.0, "y": 0.0, "z": 0.0}
    assert q.factory == KRROODPositions
    r = q.construct_instance()
    assert r == KRROODPositions(
        [KRROODPosition(1.0, 0.0, 0.0), KRROODPosition(1, 2, 3)], ["a", "b"]
    )
