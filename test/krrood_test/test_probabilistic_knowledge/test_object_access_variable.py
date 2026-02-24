from random_events.set import Set

from krrood.entity_query_language.factories import variable_from
from krrood.ormatic.dao import to_dao
from random_events.variable import Continuous, Symbolic

from krrood.probabilistic_knowledge.object_access_variable import ObjectAccessVariable
from ..dataset.example_classes import Position, Positions
from ..dataset.ormatic_interface import (
    PositionDAO,
    PoseDAO,
    OrientationDAO,
    PositionsDAO,
)


def test_object_access_variable_flat():

    obj = PositionDAO()

    obj_variable = variable_from([obj])

    x_variable = ObjectAccessVariable(Continuous("x"), obj_variable.x)
    y_variable = ObjectAccessVariable(Continuous("y"), obj_variable.y)
    z_variable = ObjectAccessVariable(Continuous("z"), obj_variable.z)
    x_variable.set_value(obj, 10)
    y_variable.set_value(obj, 20)
    z_variable.set_value(obj, 30)
    assert obj.x == 10
    assert obj.y == 20
    assert obj.z == 30


def test_object_access_variable_nested():
    obj = PoseDAO()
    obj.position = PositionDAO()
    obj.orientation = OrientationDAO()
    obj_variable = variable_from([obj])

    x_variable = ObjectAccessVariable(Continuous("position.x"), obj_variable.position.x)
    w_variable = ObjectAccessVariable(Continuous("w"), obj_variable.orientation.w)
    x_variable.set_value(obj, 10)
    w_variable.set_value(obj, 30)
    assert obj.position.x == 10
    assert obj.orientation.w == 30


def test_object_access_variable_container():
    obj = Positions([Position(None, None, None), Position(None, None, None)], [None])
    dao: PositionsDAO = to_dao(obj)

    dao_variable = variable_from([dao])

    x0_variable = ObjectAccessVariable(
        Continuous("positions[0].x"),
        dao_variable.positions[0].target.x,
    )
    y1_variable = ObjectAccessVariable(
        Continuous("positions[1].y"), dao_variable.positions[1].target.y
    )
    str0_variable = ObjectAccessVariable(
        Symbolic("some_strings[0]", Set.from_iterable(["a", "b", "c"])),
        dao_variable.some_strings[0],
    )

    x0_variable.set_value(dao, 10)
    y1_variable.set_value(dao, 20)
    str0_variable.set_value(dao, "a")
    assert dao.positions[0].target.x == 10
    assert dao.positions[1].target.y == 20
    assert dao.some_strings[0] == "a"
