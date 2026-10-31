import pytest

from krrood.ormatic.data_access_objects.helper import (
    get_data_access_object_class,
    to_dao,
)
from sqlalchemy import select
from krrood.ormatic.exceptions import QueryCannotBePersisted
from coraplex.orm.ormatic_interface import *  # type: ignore
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndPickUpAction,
    MoveAndPlaceAction,
    TransportAction,
)
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from cramph.composites import Sequence
from cramph.node import StatechartNode
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from ...plan_running import context_of, simulated_executor, statechart_of
from cramph.context import ContextExtension


@pytest.fixture()
def simple_plan(pr2_apartment_context):
    world, robot_view, extensions = pr2_apartment_context

    plan = Sequence(
        nodes=[
            NavigateAction(
                target_location=Pose.from_xyz_quaternion(
                    1.6, 1.9, 0, 0, 0, 0, 1, reference_frame=world.root
                )
            ),
            MoveTorsoAction(torso_state=TorsoState.HIGH),
            ParkArmsAction(arms=robot_view.all_arms),
        ]
    )
    return plan


def _stored_and_loaded(session, plan: StatechartNode) -> StatechartNode:
    """
    :return: `plan`, written to the database and read back.
    """
    dao = to_dao(plan)
    session.add(dao)
    session.commit()
    database_id = dao.database_id
    session.expunge_all()
    return session.get(type(dao), database_id).from_dao()


def _executed(plan: StatechartNode, extensions: list[ContextExtension]) -> None:
    """
    Execute `plan` in simulation.
    """
    executor = simulated_executor(extensions)
    executor.compile(statechart_of(executor, plan))
    executor.execute()


def test_a_performed_plan_is_read_back_with_its_steps(
    coraplex_testing_session, pr2_apartment_context, simple_plan
) -> None:
    world, robot_view, extensions = pr2_apartment_context
    _executed(simple_plan, extensions)

    recreated_plan = _stored_and_loaded(coraplex_testing_session, simple_plan)

    assert type(recreated_plan) is Sequence
    assert [type(step) for step in recreated_plan.nodes] == [
        type(step) for step in simple_plan.nodes
    ]
    assert recreated_plan.nodes[1].torso_state == simple_plan.nodes[1].torso_state


def test_actions_of_every_shape_are_stored_as_actions(
    coraplex_testing_session, pr2_apartment_context, simple_plan
) -> None:
    """
    Reaching a goal, running steps in turn and running them side by side are shapes of
    an action, so asking for the stored actions finds all of them.
    """
    world, robot_view, extensions = pr2_apartment_context
    _executed(simple_plan, extensions)
    _stored_and_loaded(coraplex_testing_session, simple_plan)

    stored_actions = coraplex_testing_session.scalars(
        select(get_data_access_object_class(Action))
    ).all()

    assert sorted(type(action).__name__ for action in stored_actions) == sorted(
        f"{type(step).__name__}DAO" for step in simple_plan.nodes
    )


@pytest.fixture
def complex_plan(pr2_apartment_context):
    """
    A plan transporting the milk with steps that are grounded already, standing where
    the transport described by queries grounds its steps to.
    """
    world, robot_view, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]

    plan = TransportAction(
        pick_up=MoveAndPickUpAction.from_standing_position(
            standing_position=Pose.from_xyz_rpy(
                1.63, 1.98, 0.0, reference_frame=world.root
            ),
            grasp=milk.grasp_candidates()[0],
            arm=robot_view.left_arm,
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
    )

    return plan


def test_a_performed_transport_is_read_back(
    coraplex_testing_session, pr2_apartment_context, complex_plan
) -> None:
    """
    A performed plan holding a transport is persisted and recreated from the database.
    """
    world, robot_view, extensions = pr2_apartment_context
    _executed(complex_plan, extensions)

    recreated_plan = _stored_and_loaded(coraplex_testing_session, complex_plan)

    assert type(recreated_plan) is TransportAction
    assert type(recreated_plan.pick_up) is MoveAndPickUpAction
    assert type(recreated_plan.place) is MoveAndPlaceAction


def test_a_plan_whose_transport_still_holds_queries_cannot_be_stored(
    pr2_apartment_context,
):
    """
    A step still described by a query has no value to store until it is grounded, so
    storing it is refused rather than writing something that cannot be read back.
    """
    world, robot_view, extensions = pr2_apartment_context
    transport = TransportAction.from_graspable_by_closest_grasps(
        world.get_semantic_annotations_by_type(Milk)[0],
        Pose.from_xyz_quaternion(2.4, 2.8, 1, 0, 0, 0, 1, reference_frame=world.root),
        robot_view.left_arm,
        context_of(extensions),
    )

    with pytest.raises(QueryCannotBePersisted) as failure:
        to_dao(transport)

    assert failure.value.query in (transport.pick_up, transport.place)
