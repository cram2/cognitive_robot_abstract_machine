from __future__ import annotations

import datetime

from random_events.set import Set
from random_events.variable import Continuous, Symbolic

from krrood.entity_query_language.factories import variable_from, variable
from krrood.ormatic.dao import to_dao
from krrood.probabilistic_knowledge.object_access_variable import ObjectAccessVariable
from krrood.probabilistic_knowledge.parameterizer import Parameterizer, Parameterization
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
    parameterizer = Parameterizer()
    parameterization = parameterizer.parameterize(position)
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
    parameterizer = Parameterizer()
    parameterization = parameterizer.parameterize(position)
    assert parameterization.variables == []


def test_parameterize_orientation_mixed_none():
    """
    Test parameterization of the Orientation class.
    """
    orientation = Orientation(..., None, ..., None)
    dao = to_dao(orientation)
    orientation_dao_variable = variable(type(dao), [dao])
    parameterizer = Parameterizer()
    parameterization = parameterizer.parameterize(orientation)

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

    parameterizer = Parameterizer()
    parameterization = parameterizer.parameterize(pose)
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
    parameterizer = Parameterizer()
    parameterization = parameterizer.parameterize(position)

    result_by_hand = {
        Continuous("PositionDAO.x"): 1.0,
        Continuous("PositionDAO.y"): 2.0,
        Continuous("PositionDAO.z"): 3.0,
    }

    assert parameterization.assignments_for_conditioning == result_by_hand


def test_parameterize_nested_object():
    """
    Test parameterization of a nested object via Parameterizer.parameterize.
    """
    pose = Pose(
        position=Position(x=1.0, y=2.0, z=3.0),
        orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    parameterizer = Parameterizer()
    parameterization = parameterizer.parameterize(pose)

    result_by_hand = {
        Continuous("PoseDAO.position.x"): 1.0,
        Continuous("PoseDAO.position.y"): 2.0,
        Continuous("PoseDAO.position.z"): 3.0,
        Continuous("PoseDAO.orientation.x"): 0.0,
        Continuous("PoseDAO.orientation.y"): 0.0,
        Continuous("PoseDAO.orientation.z"): 0.0,
        Continuous("PoseDAO.orientation.w"): 1.0,
    }
    assert parameterization.assignments_for_conditioning == result_by_hand


def test_one_to_many_and_collection_of_builtins():
    p = Positions([Position(1, 2, 3), Position(4, 5, 6)], ["a", ...])
    parameters = Parameterizer().parameterize(p)

    result_by_hand = {
        Continuous("PositionsDAO.positions[0].target.x"): 1.0,
        Continuous("PositionsDAO.positions[0].target.y"): 2.0,
        Continuous("PositionsDAO.positions[0].target.z"): 3.0,
        Continuous("PositionsDAO.positions[1].target.x"): 4.0,
        Continuous("PositionsDAO.positions[1].target.y"): 5.0,
        Continuous("PositionsDAO.positions[1].target.z"): 6.0,
    }

    assert parameters.assignments_for_conditioning == result_by_hand


def test_symbolic_variables():
    obj = ListOfEnum([..., ...])

    parameterizer = Parameterizer()
    parameterization = parameterizer.parameterize(obj)

    test_enum_set = Set.from_iterable(TestEnum)
    assert parameterization.random_events_variables == [
        Symbolic("ListOfEnumDAO.list_of_enum[0]", test_enum_set),
        Symbolic("ListOfEnumDAO.list_of_enum[1]", test_enum_set),
    ]
    assert parameterization.assignments_for_conditioning == {}


def test_not_follow_none_relationships():
    p = Pose(position=Position(..., ..., ...), orientation=None)
    parameterizer = Parameterizer()
    parameterization = parameterizer.parameterize(p)
    variables = [
        Continuous("PoseDAO.position.x"),
        Continuous("PoseDAO.position.y"),
        Continuous("PoseDAO.position.z"),
    ]

    assert parameterization.random_events_variables == variables
    assert parameterization.assignments == {}


def test_parameterize_object_from_sample():
    obj = Atom(..., ..., ..., datetime.datetime.now())
    parameterizer = Parameterizer()
    parameterization = parameterizer.parameterize(obj)
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
