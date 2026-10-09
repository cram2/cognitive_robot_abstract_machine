import pytest
from sqlalchemy import select

# The alternative mapping needs to be imported for the stretch to work properly
import coraplex.alternative_motion_mappings.stretch_motion_mapping  # type: ignore
import coraplex.alternative_motion_mappings.tiago_motion_mapping  # type: ignore
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.exceptions import QueryCannotBePersisted
from coraplex.execution_environment import kinematically_simulated_robot
from coraplex.orm.ormatic_interface import *  # type: ignore
from coraplex.plans.factories import sequential, execute_single
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndPickUpAction,
    MoveAndPlaceAction,
    TransportAction,
)
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk


@pytest.fixture()
def simple_plan(pr2_apartment_context):
    world, robot_view, context = pr2_apartment_context

    plan = sequential(
        [
            NavigateAction(
                Pose.from_xyz_quaternion(
                    1.6, 1.9, 0, 0, 0, 0, 1, reference_frame=world.root
                ),
            ),
            MoveTorsoAction(TorsoState.HIGH),
            ParkArmsAction(context.robot.all_arms),
        ],
        context=context,
    ).plan
    return plan


@pytest.mark.skip("Execution Data is not recorded right now")
def test_plan_serialization(coraplex_testing_session, simple_plan):
    session = coraplex_testing_session

    with kinematically_simulated_robot:
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


def test_replay_simple_plan(coraplex_testing_session, simple_plan):

    with kinematically_simulated_robot:
        simple_plan.perform()

    session = coraplex_testing_session

    dao = to_dao(simple_plan)
    session.add(dao)
    session.commit()

    fetched_plan = session.scalars(select(PlanMappingDAO)).one()
    recreated_plan: Plan = fetched_plan.from_dao()

    # TODO: this does not work yet as semantic annotations cannot be copied.
    # recreated_plan.prepare_for_replay()
    # recreated_plan.replay()


@pytest.fixture
def complex_plan(pr2_apartment_context):
    """
    A plan transporting the milk with steps that are grounded already, standing where
    the transport described by queries grounds its steps to.
    """
    world, robot_view, context = pr2_apartment_context
    context.evaluate_conditions = False
    milk = world.get_semantic_annotations_by_type(Milk)[0]

    plan = execute_single(
        TransportAction(
            pick_up=MoveAndPickUpAction.from_standing_position(
                standing_position=Pose.from_xyz_rpy(
                    1.63, 1.98, 0.0, reference_frame=world.root
                ),
                grasp=milk.grasp_candidates()[0],
                arm=context.robot.left_arm,
            ),
            place=MoveAndPlaceAction.from_standing_position(
                standing_position=Pose.from_xyz_rpy(
                    1.8, 2.54, 0.0, reference_frame=world.root
                ),
                target_location=Pose.from_xyz_quaternion(
                    2.4, 2.8, 1, 0, 0, 0, 1, reference_frame=world.root
                ),
                object_designator=milk,
            ),
        ),
        context=context,
    ).plan

    return plan


@pytest.mark.skip("Execution Data is not recorded right now")
def test_execution_data_of_complex_plan(coraplex_testing_session, complex_plan):

    with kinematically_simulated_robot:
        complex_plan.perform()

    session = coraplex_testing_session
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


def test_replay_complex_plan_from_db(coraplex_testing_session, complex_plan):
    """
    A performed plan holding a transport is persisted and recreated from the database.
    """
    with kinematically_simulated_robot:
        complex_plan.perform()

    complex_plan.initial_world = None
    session = coraplex_testing_session

    plan = complex_plan
    dao = to_dao(plan)
    session.add(dao)
    session.commit()

    fetched_plan = session.scalars(select(PlanMappingDAO)).one()

    recreated_plan = fetched_plan.from_dao()


def test_a_plan_whose_transport_still_holds_queries_cannot_be_stored(
    pr2_apartment_context,
):
    """
    A step still described by a query has no value to store until it is grounded, so
    storing it is refused rather than writing something that cannot be read back.
    """
    world, robot_view, context = pr2_apartment_context
    transport = TransportAction.from_graspable_by_closest_grasps(
        world.get_semantic_annotations_by_type(Milk)[0],
        Pose.from_xyz_quaternion(2.4, 2.8, 1, 0, 0, 0, 1, reference_frame=world.root),
        context.robot.left_arm,
        context,
    )
    plan = execute_single(transport, context=context).plan

    with pytest.raises(QueryCannotBePersisted) as failure:
        to_dao(plan)

    assert failure.value.query in (transport.pick_up, transport.place)
