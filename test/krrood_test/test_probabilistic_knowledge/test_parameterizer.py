from __future__ import annotations

import datetime

from krrood.entity_query_language.factories import variable_from, variable
from krrood.ormatic.dao import to_dao
from krrood.probabilistic_knowledge.object_access_variable import ObjectAccessVariable
from krrood.probabilistic_knowledge.parameterizer import (
    DataAccessObjectParameterizer,
    Parameterization,
)
from random_events.set import Set
from random_events.variable import Continuous, Symbolic
from ..dataset.example_classes import (
    Position,
    Pose,
    Orientation,
    Positions,
    ListOfEnum,
    TestEnum,
    Atom,
    Element,
)


def test_parameterize_position():
    """
    Test parameterization of the Position class.
    """
    position = Position(..., ..., ...)
    dao = to_dao(position)
    position_dao_variable = variable(type(dao), [dao])
    parameterizer = DataAccessObjectParameterizer(dao)
    parameterization = parameterizer.parameterize()
    expected_variables = [
        ObjectAccessVariable(
            Continuous(position_dao_variable.x._name_),
            position_dao_variable.x,
        ),
        ObjectAccessVariable(
            Continuous(position_dao_variable.y._name_),
            position_dao_variable.y,
        ),
        ObjectAccessVariable(
            Continuous(position_dao_variable.z._name_),
            position_dao_variable.z,
        ),
    ]
    assert parameterization.variables == expected_variables


def test_parameterize_position_skip_none_field():
    """
    Test parameterization of the Position class.
    """
    position = Position(None, None, None)
    dao = to_dao(position)
    parameterizer = DataAccessObjectParameterizer(dao)
    parameterization = parameterizer.parameterize()
    assert parameterization.variables == []


def test_parameterize_orientation_mixed_none():
    """
    Test parameterization of the Orientation class.
    """
    orientation = Orientation(..., None, ..., None)
    dao = to_dao(orientation)
    orientation_dao_variable = variable(type(dao), [dao])
    parameterizer = DataAccessObjectParameterizer(dao)
    parameterization = parameterizer.parameterize()

    expected_variables = [
        ObjectAccessVariable(
            Continuous(orientation_dao_variable.x._name_),
            orientation_dao_variable.x,
        ),
        ObjectAccessVariable(
            Continuous(orientation_dao_variable.z._name_),
            orientation_dao_variable.z,
        ),
    ]

    assert parameterization.variables == expected_variables


def test_parameterize_pose():
    """
    Test parameterization of the Pose class.
    """
    pose = Pose(
        position=Position(..., ..., ...),
        orientation=Orientation(..., ..., ..., None),
    )

    dao = to_dao(pose)
    pose_dao_variable = variable(type(dao), [dao])

    parameterizer = DataAccessObjectParameterizer(dao)
    parameterization = parameterizer.parameterize()
    expected_variables = [
        ObjectAccessVariable(
            Continuous(pose_dao_variable.position.x._name_),
            pose_dao_variable.position.x,
        ),
        ObjectAccessVariable(
            Continuous(pose_dao_variable.position.y._name_),
            pose_dao_variable.position.y,
        ),
        ObjectAccessVariable(
            Continuous(pose_dao_variable.position.z._name_),
            pose_dao_variable.position.z,
        ),
        ObjectAccessVariable(
            Continuous(pose_dao_variable.orientation.x._name_),
            pose_dao_variable.orientation.x,
        ),
        ObjectAccessVariable(
            Continuous(pose_dao_variable.orientation.y._name_),
            pose_dao_variable.orientation.y,
        ),
        ObjectAccessVariable(
            Continuous(pose_dao_variable.orientation.z._name_),
            pose_dao_variable.orientation.z,
        ),
    ]

    assert parameterization.variables == expected_variables


def test_create_fully_factorized_distribution():
    """
    Test for a fully factorized distribution.
    """
    variables = [
        ObjectAccessVariable(Continuous("Variable.A"), variable_from([])),
        ObjectAccessVariable(Continuous("Variable.B"), variable_from([])),
    ]
    parameterization = Parameterization(variables)
    probabilistic_circuit = parameterization.create_fully_factorized_distribution()
    assert len(probabilistic_circuit.variables) == 2
    assert set(probabilistic_circuit.variables) == set(
        parameterization.random_events_variables
    )


def test_parameterize_object_with_given_values():
    """
    Test parameterization of a single object via Parameterizer.parameterize.
    """
    position = Position(x=1.0, y=2.0, z=3.0)
    dao = to_dao(position)
    parameterizer = DataAccessObjectParameterizer(dao)
    parameterization = parameterizer.parameterize()

    [x, y, z] = parameterization.variables

    result_by_hand = {
        x: 1.0,
        y: 2.0,
        z: 3.0,
    }

    assert parameterization.assignments == result_by_hand


def test_parameterize_nested_object():
    """
    Test parameterization of a nested object via Parameterizer.parameterize.
    """
    pose = Pose(
        position=Position(x=1.0, y=2.0, z=3.0),
        orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    dao = to_dao(pose)
    parameterizer = DataAccessObjectParameterizer(dao)
    parameterization = parameterizer.parameterize()

    [px, py, pz, ox, oy, oz, ow] = parameterization.variables

    result_by_hand = {
        px: 1.0,
        py: 2.0,
        pz: 3.0,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
        ow: 1.0,
    }
    assert parameterization.assignments == result_by_hand


def test_one_to_many_and_collection_of_builtins():
    p = Positions([Position(1, 2, 3), Position(4, 5, 6)], ["a", ...])
    dao = to_dao(p)
    parameters = DataAccessObjectParameterizer(dao).parameterize()

    [p0_x, p0_y, p0_z, p1_x, p1_y, p1_z] = parameters.variables

    result_by_hand = {
        p0_x: 1.0,
        p0_y: 2.0,
        p0_z: 3.0,
        p1_x: 4.0,
        p1_y: 5.0,
        p1_z: 6.0,
    }

    assert parameters.assignments == result_by_hand


def test_symbolic_variables():
    obj = ListOfEnum([..., ...])
    dao = to_dao(obj)
    parameterizer = DataAccessObjectParameterizer(dao)
    parameterization = parameterizer.parameterize()

    test_enum_set = Set.from_iterable(TestEnum)
    assert parameterization.random_events_variables == [
        Symbolic("ListOfEnumDAO.list_of_enum[0]", test_enum_set),
        Symbolic("ListOfEnumDAO.list_of_enum[1]", test_enum_set),
    ]
    assert parameterization.assignments_for_conditioning == {}


def test_not_follow_none_relationships():
    p = Pose(position=Position(..., ..., ...), orientation=None)
    dao = to_dao(p)
    parameterizer = DataAccessObjectParameterizer(dao)
    parameterization = parameterizer.parameterize()
    variables = [
        Continuous("PoseDAO.position.x"),
        Continuous("PoseDAO.position.y"),
        Continuous("PoseDAO.position.z"),
    ]

    assert parameterization.random_events_variables == variables
    assert parameterization.assignments == {}


def test_parameterize_object_from_sample():
    obj = Atom(..., ..., ..., datetime.datetime.now())
    dao = to_dao(obj)
    parameterizer = DataAccessObjectParameterizer(dao)
    parameterization = parameterizer.parameterize()
    distribution = parameterization.create_fully_factorized_distribution()
    sample = distribution.sample(1)[0]
    sample_dict = parameterization.create_assignment_from_variables_and_sample(
        distribution.variables, sample
    )

    parameterized_obj: Atom = parameterization.parameterize_object_with_sample(
        obj, sample_dict
    )
    assert parameterized_obj.timestamp == obj.timestamp
    assert isinstance(parameterized_obj.charge, float)
    assert isinstance(parameterized_obj.element, Element)
