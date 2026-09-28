"""
Tests for attaching simulator properties to an entity and looking up the one property of
a type it carries.
"""

from __future__ import annotations

import pytest

from semantic_digital_twin.adapters.multi_sim import (
    MujocoBody,
    MujocoCamera,
    MujocoGeom,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import DuplicateSimulatorPropertyError
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def body() -> Body:
    return Body(name=PrefixedName("body"))


def test_no_property_of_a_type_is_none(body):
    assert body.get_simulator_property_of_type(MujocoBody) is None


def test_the_attached_property_is_found(body):
    attached = MujocoBody(motion_capture=True)
    body.add_simulator_property(attached)

    assert body.get_simulator_property_of_type(MujocoBody) is attached


def test_a_property_of_another_type_is_ignored(body):
    body.add_simulator_property(MujocoGeom())

    assert body.get_simulator_property_of_type(MujocoBody) is None


def test_attaching_a_second_unique_property_of_one_type_raises(body):
    body.add_simulator_property(MujocoBody())

    with pytest.raises(DuplicateSimulatorPropertyError) as raised:
        body.add_simulator_property(MujocoBody())
    assert raised.value.property_type is MujocoBody
    assert body.simulator_additional_properties == [MujocoBody()]


def test_a_property_an_entity_may_carry_several_of_is_attached_again(body):
    """
    Cameras and lights are not unique per entity, so a second one is not a duplicate.
    """
    first, second = MujocoCamera(), MujocoCamera()

    body.add_simulator_property(first)
    body.add_simulator_property(second)

    assert body.simulator_additional_properties == [first, second]


def test_lookup_returns_the_first_of_two_properties_handed_in_whole(body):
    """
    Only attaching through ``add_simulator_property`` keeps a unique property from being
    attached twice; a list handed in whole is read as it is.
    """
    first, second = MujocoBody(), MujocoBody(motion_capture=True)
    body.simulator_additional_properties = [first, second]

    assert body.get_simulator_property_of_type(MujocoBody) is first
