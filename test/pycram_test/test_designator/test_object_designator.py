import pytest

from krrood.entity_query_language.entity import entity, variable, and_, contains
from krrood.entity_query_language.entity_result_processors import an, the

from pycram.designator import EQLObjectDesignator, NamedObject
from pycram.designators.object_designator import *
from pycram.language import SequentialPlan


def test_eql_designator(immutable_model_world):
    world, robot_view, context = immutable_model_world
    obj = variable(type_=Body, domain=world.bodies)
    milk_desig = EQLObjectDesignator(
        an(
            entity(obj).where(
                contains(obj.name.name, "milk"),
            )
        )
    )
    found_milks = list(milk_desig)
    assert 1 == len(found_milks)
    assert "milk.stl" == found_milks[0].name.name
    assert world.get_body_by_name("milk.stl") == found_milks[0]


def test_named_object(immutable_model_world):
    world, robot_view, context = immutable_model_world
    named_desig = NamedObject("milk.stl")
    plan = SequentialPlan(context, named_desig)
    found_milks = list(named_desig)
    assert 1 == len(found_milks)
    assert "milk.stl" == found_milks[0].name.name
    assert world.get_body_by_name("milk.stl") == found_milks[0]
