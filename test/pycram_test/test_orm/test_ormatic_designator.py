import time

import numpy as np
import pytest
from sqlalchemy import select

# The alternative mapping needs to be imported for the stretch to work properly
import pycram.alternative_motion_mappings.stretch_motion_mapping  # type: ignore
import pycram.alternative_motion_mappings.tiago_motion_mapping  # type: ignore
from krrood.ormatic.data_access_objects.helper import to_dao
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.motion_executor import simulated_robot
from pycram.orm.ormatic_interface import *  # type: ignore
from pycram.plans.factories import sequential, execute_single
from pycram.plans.plan import Plan
from pycram.robot_plans.actions.composite.transporting import TransportAction
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.spatial_types.spatial_types import Pose


@pytest.fixture()
def simple_plan(immutable_model_world):
    world, robot_view, context = immutable_model_world

    plan = sequential(
        [
            NavigateAction(
                Pose.from_xyz_quaternion(
                    1.6, 1.9, 0, 0, 0, 0, 1, reference_frame=world.root
                ),
                True,
            ),
            MoveTorsoAction(TorsoState.HIGH),
            ParkArmsAction(Arms.BOTH),
        ],
        context=context,
    ).plan
    return plan


def test_plan_serialization(pycram_testing_session, simple_plan):
    session = pycram_testing_session

    with simulated_robot:
        simple_plan.perform()

    dao = to_dao(simple_plan)
    session.add(dao)
    session.commit()

    result = session.scalars(
        select(ActionNodeDAO).join(NavigateActionDAO, ActionNodeDAO.designator)
    ).all()
    assert all(
        [
            r.execution_data.execution_start_pose is not None
            and r.execution_data.execution_end_pose is not None
            for r in result
        ]
    )

    motions = session.scalars(select(BaseMotionDAO)).all()
    assert len(motions) == 3


def test_replay_simple_plan(pycram_testing_session, simple_plan):

    with simulated_robot:
        simple_plan.perform()

    session = pycram_testing_session

    dao = to_dao(simple_plan)
    session.add(dao)
    session.commit()

    fetched_plan = session.scalars(select(PlanMappingDAO)).one()
    recreated_plan: Plan = fetched_plan.from_dao()

    # TODO: this does not work yet as semantic annotations cannot be copied.
    # recreated_plan.prepare_for_replay()
    # recreated_plan.replay()


@pytest.fixture
def complex_plan(mutable_model_world):
    world, robot_view, context = mutable_model_world
    context.evaluate_conditions = False

    plan = execute_single(
        TransportAction(
            object_designator=world.get_body_by_name("milk.stl"),
            target_location=Pose.from_xyz_quaternion(
                2.4, 2.8, 1, 0, 0, 0, 1, reference_frame=world.root
            ),
            arm=Arms.LEFT,
            grasp_description=GraspDescription(
                ApproachDirection.LEFT,
                VerticalAlignment.NoAlignment,
                robot_view.left_arm.manipulator,
            ),
        ),
        context=context,
    ).plan

    return plan


def test_execution_data_of_complex_plan(pycram_testing_session, complex_plan):

    with simulated_robot:
        complex_plan.perform()

    session = pycram_testing_session
    plan = complex_plan
    dao = to_dao(plan)
    session.add(dao)
    session.commit()

    pick_up_node = session.scalars(
        select(ActionNodeDAO).join(PickUpActionDAO, ActionNodeDAO.designator)
    ).one()
    place_node = session.scalars(
        select(ActionNodeDAO).join(PlaceActionDAO, ActionNodeDAO.designator)
    ).one()

    assert plan.initial_world is not None
    assert pick_up_node.execution_data is not None
    assert place_node.execution_data is not None


def test_replay_complex_plan_from_db(pycram_testing_session, complex_plan):

    with simulated_robot:
        complex_plan.perform()

    complex_plan.initial_world = None
    session = pycram_testing_session

    plan = complex_plan
    dao = to_dao(plan)
    session.add(dao)
    session.commit()

    fetched_plan = session.scalars(select(PlanMappingDAO)).one()

    recreated_plan = fetched_plan.from_dao()
