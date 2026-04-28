from datetime import datetime

import pytest
from sqlalchemy.orm import sessionmaker

from krrood.entity_query_language.backends import (
    SQLAlchemyBackend,
    EntityQueryLanguageBackend,
    ProbabilisticBackend,
    EntityQueryLanguageBackend,
)
from krrood.entity_query_language.factories import (
    variable,
    entity,
    an,
    underspecified,
    variable_from,
)
from krrood.entity_query_language.query_graph import QueryGraph
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.model_registries import DictRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from pycram.datastructures.enums import ApproachDirection
from pycram.datastructures.grasp import GraspDescription
from pycram.robot_plans.actions.composite.transporting import MoveAndPickUpAction
from pycram.robot_plans.actions.core.misc import MoveToReach
from random_events.interval import Interval, reals
from random_events.set import Set
from random_events.variable import Symbolic
from semantic_digital_twin.orm.model import (
    PoseMapping,
    Point3Mapping,
    QuaternionMapping,
)
from ..dataset.example_classes import (
    KRROODPose,
    KRROODPosition,
    KRROODOrientation,
    Atom,
    Element,
)
from ..dataset.ormatic_interface import *  # type: ignore


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
