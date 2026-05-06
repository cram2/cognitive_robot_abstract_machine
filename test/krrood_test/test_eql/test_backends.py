from copy import deepcopy
from datetime import datetime
from types import EllipsisType

import pytest
from sqlalchemy.orm import sessionmaker

from krrood.entity_query_language.backends import (
    SQLAlchemyBackend,
    EntityQueryLanguageBackend,
    ProbabilisticBackend,
    EntityQueryLanguageBackend,
)
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.factories import (
    variable,
    entity,
    an,
    underspecified,
    variable_from,
)
from krrood.entity_query_language.query_graph import QueryGraph
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.entity_query_language.core.variable import Variable as KRROODVariable
from krrood.parametrization.model_registries import DictRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection
from pycram.datastructures.grasp import GraspDescription
from pycram.robot_plans.actions.composite.transporting import MoveAndPickUpAction
from pycram.robot_plans.actions.core.misc import MoveToReach
from random_events.interval import Interval, reals, singleton
from random_events.set import Set
from random_events.variable import Symbolic, Variable
from semantic_digital_twin.orm.model import (
    PoseMapping,
    Point3Mapping,
    QuaternionMapping,
)
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.robots.pr2 import PR2
from ..dataset.example_classes import (
    KRROODPose,
    KRROODPosition,
    KRROODOrientation,
    Atom,
    Element,
)
from ..dataset.ormatic_interface import *  # type: ignore
from ..test_ripple_down_rules.test_results.datasets_physical_object_is_a_robot import (
    physical_object_is_a_robot_output__scrdr,
)


@pytest.fixture(scope="function")
def mutable_model_world(pr2_apartment_world):
    world = deepcopy(pr2_apartment_world)
    pr2 = PR2.from_world(world)
    return world, pr2, Context(world, pr2)


def test_nested_action(mutable_model_world):
    world, robot_view, context = mutable_model_world
    milk = world.get_body_by_name("milk.stl")

    milk_variable = variable_from([milk])

    manipulation_offset = 0.05
    prob_q = underspecified(MoveAndPickUpAction)(
        keep_joint_states=...,
        standing_position=underspecified(
            PoseMapping.from_point_mapping_quaternion_mapping
        )(
            position=underspecified(Point3Mapping)(
                x=..., y=..., z=..., reference_frame=None
            ),
            orientation=underspecified(QuaternionMapping)(
                x=..., y=..., z=..., w=..., reference_frame=None
            ),
            reference_frame=variable_from([robot_view.root]),
        ),
        object_designator=milk_variable,
        arm=...,
        grasp_description=underspecified(GraspDescription)(
            approach_direction=...,
            vertical_alignment=...,
            rotate_gripper=...,
            manipulation_offset=manipulation_offset,
            manipulator=variable(Manipulator, world.semantic_annotations),
        ),
    )
    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables
    names_of_actual_specified_parameters = [
        match.name_from_variable_access_path
        for match in parameters.statement.matches_with_variables
        if (
            isinstance(match.assigned_variable, Literal)
            or isinstance(match.assigned_variable, KRROODVariable)
        )
        and not isinstance(match.assigned_value, EllipsisType)
        and not match.assigned_value is None
    ]
    random_events_names = [
        name
        for name, variable in variables.items()
        if isinstance(variable, Symbolic)
        and name in names_of_actual_specified_parameters
    ]

    assert names_of_actual_specified_parameters == [
        "MoveAndPickUpAction.object_designator",
        "MoveAndPickUpAction.standing_position.reference_frame",
        "MoveAndPickUpAction.grasp_description.manipulation_offset",
        "MoveAndPickUpAction.grasp_description.manipulator",
    ]
    assert (
        variables[
            "MoveAndPickUpAction.grasp_description.manipulation_offset"
        ].domain.simple_sets
        == singleton(manipulation_offset).simple_sets
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

    python_backend = EntityQueryLanguageBackend()
    result = list(python_backend.evaluate(q))
    assert len(result) == 1

    database_backend = SQLAlchemyBackend(session_maker)
    result = list(database_backend.evaluate(q))
    assert len(result) == 1


def test_probabilistic_backend_with_symbolic_expression():
    prob_q = underspecified(KRROODPosition)(
        x=..., y=..., z=variable(int, domain=[1, 2, 3])
    )
    parameters = UnderspecifiedParameters(prob_q)
    assert parameters.variables["KRROODPosition.z"] == Symbolic(
        name="KRROODPosition.z", domain=Set.from_iterable([1, 2, 3])
    )


def test_underspecified_parameters_with_partly_symbolic_expression():
    prob_q = underspecified(KRROODPosition)(
        x=..., y=..., z=variable(int, domain=[1, 2, 3])
    )
    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables
    assert len(variables) == 3
    assert variables["KRROODPosition.x"].domain == reals()
    assert variables["KRROODPosition.x"].is_numeric
    assert variables["KRROODPosition.y"].domain == reals()
    assert variables["KRROODPosition.y"].is_numeric
    assert variables["KRROODPosition.z"].domain == Set.from_iterable([1, 2, 3])
    assert not variables["KRROODPosition.z"].is_numeric


def test_underspecified_parameters_with_full_symbolic_expression():
    prob_q = variable(KRROODPosition, domain=[KRROODPosition(1, 2, 3)])

    with pytest.raises(TypeError):
        UnderspecifiedParameters(prob_q)


def test_underspecified_parameters_with_only_underspecified():
    prob_q = underspecified(PoseMapping.from_point_mapping_quaternion_mapping)(
        position=underspecified(Point3Mapping)(
            x=..., y=..., z=..., reference_frame=None
        ),
        orientation=underspecified(QuaternionMapping)(
            x=..., y=..., z=..., w=..., reference_frame=None
        ),
        reference_frame=None,
    )
    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables

    assert len(variables) == 7


def test_underspecified_parameters_with_only_literals():
    prob_q = underspecified(PoseMapping.from_point_mapping_quaternion_mapping)(
        position=KRROODPosition(1, 2, 3),
        orientation=KRROODOrientation(0, 0, 0, 1),
        reference_frame=None,
    )
    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables

    assert len(variables) == 7


def test_enum_value_as_literal():
    prob_q = underspecified(MoveToReach)(
        target_pose=None,
        robot_x=...,
        robot_y=...,
        hip_rotation=...,
        grasp_description=underspecified(GraspDescription)(
            approach_direction=ApproachDirection.FRONT,
            vertical_alignment=...,
            manipulator=None,
            rotate_gripper=...,
        ),
    )
    pm_backend = ProbabilisticBackend(number_of_samples=10)
    values = list(pm_backend.evaluate(prob_q))
    for value in values:
        assert value.grasp_description.approach_direction == ApproachDirection.FRONT


def test_probabilistic_query_backend():
    prob_q = underspecified(KRROODPose)(
        position=underspecified(KRROODPosition)(x=..., y=..., z=...),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    prob_q.resolve()
    prob_q.where(prob_q.variable.position.x > 0.5)

    pm_backend = ProbabilisticBackend(number_of_samples=10)
    values = list(pm_backend.evaluate(prob_q))
    for value in values:
        assert value.position.x > 0.5

    assert pm_backend.number_of_samples == len({v.position for v in values})


def test_generative_eql_backend():
    q = underspecified(Atom)(
        element=...,
        type=variable_from([0, 1, 2]),
        charge=variable_from([0.0, 1.0, 2.0]),
        timestamp=datetime.datetime.now(),
    )
    q.resolve()
    q.where(q.variable.type > q.variable.charge)
    backend = EntityQueryLanguageBackend()
    results = list(backend.evaluate(q))
    assert len(results) == 6
    for result in results:
        assert isinstance(result.element, Element)
        assert result.type > result.charge
