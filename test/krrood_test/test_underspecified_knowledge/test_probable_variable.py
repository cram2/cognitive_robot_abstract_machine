import pytest

from krrood.entity_query_language.exceptions import UnboundPatternVariable
from krrood.entity_query_language.factories import (
    a,
    variable,
)
from ..dataset.derived_attributes import Rectangle
from ..pattern_variables import find_assigned_variable
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
    prob_q.where(prob_q.position.x > 0.5)
    instance = prob_q.construct_instance()

    correct_instance = KRROODPose(
        KRROODPosition(..., ..., ...), KRROODOrientation(0.0, 0.0, 0.0, 1.0)
    )

    assert instance == correct_instance
    assert len(list(prob_q._matches_with_variables_)) == 4


def test_new_underspecified_with_factory():

    prob_q = a(KRROODPose)(
        position=a(KRROODPosition.from_abc, target_type=KRROODPosition)(
            a=..., b=..., c=...
        ),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    prob_q.where(prob_q.position.x > 0.5)
    prob_q._get_expression_().build()
    r = prob_q.construct_instance()
    assert r == KRROODPose(
        KRROODPosition(..., ..., ...), KRROODOrientation(0.0, 0.0, 0.0, 1.0)
    )


# %% constructing an instance from the bindings of its pattern's variables


def test_match_constructs_an_instance_from_bindings_of_its_nested_pattern():
    stated_y, stated_z, bound_x = 1.0, 2.0, 5.0
    orientation = KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0)
    query = a(KRROODPose)(
        position=a(KRROODPosition)(x=..., y=stated_y, z=stated_z),
        orientation=orientation,
    )
    x = find_assigned_variable(query, "KRROODPose.position.x")
    instance = query._construct_instance_from_bindings_({x._id_: bound_x})
    assert instance == KRROODPose(
        KRROODPosition(bound_x, stated_y, stated_z), orientation
    )


def test_match_constructs_list_elements_from_bindings():
    stated_x, bound_y, bound_z = 1.0, 0.0, 3.0
    stated_position = KRROODPosition(1, 2, 3)
    some_strings = ["a", "b"]
    query = a(KRROODPositions)(
        positions=[a(KRROODPosition)(x=stated_x, y=..., z=...), stated_position],
        some_strings=some_strings,
    )
    y = find_assigned_variable(query, "KRROODPositions.positions[0].y")
    z = find_assigned_variable(query, "KRROODPositions.positions[0].z")
    instance = query._construct_instance_from_bindings_(
        {y._id_: bound_y, z._id_: bound_z}
    )
    assert instance == KRROODPositions(
        [KRROODPosition(stated_x, bound_y, bound_z), stated_position], some_strings
    )


def test_constructing_from_bindings_leaves_the_pattern_unchanged():
    query = a(KRROODPosition)(x=..., y=1.0, z=2.0)
    x = find_assigned_variable(query, "KRROODPosition.x")
    query._construct_instance_from_bindings_({x._id_: 5.0})
    assert query._kwargs_ == {"x": ..., "y": 1.0, "z": 2.0}
    assert x._value_ is ...


def test_constructing_from_bindings_rejects_an_unbound_pattern_variable():
    query = a(Rectangle)(width=variable(int, [1, 2]), height=3)
    with pytest.raises(UnboundPatternVariable):
        query._construct_instance_from_bindings_({})
