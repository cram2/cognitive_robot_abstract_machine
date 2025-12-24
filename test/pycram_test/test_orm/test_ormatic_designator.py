import logging
from copy import deepcopy

import numpy as np
import pytest
import rclpy
import sqlalchemy.sql.elements

from krrood.entity_query_language.symbol_graph import SymbolGraph
from krrood.ormatic.dao import to_dao
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
from semantic_digital_twin.robots.pr2 import PR2
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import (
    TorsoState,
    ApproachDirection,
    Arms,
    VerticalAlignment,
    GripperState,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PyCramPose, PoseStamped
from pycram.designator import NamedObject
from pycram.language import SequentialPlan, ParallelPlan
from pycram.orm.ormatic_interface import *
from pycram.process_module import simulated_robot
from pycram.robot_plans import (
    MoveTorsoActionDescription,
    ParkArmsAction,
    PlaceActionDescription,
    TransportAction,
    ParkArmsActionDescription,
    TransportActionDescription,
    NavigateActionDescription,
    PickUpActionDescription,
    SetGripperActionDescription,
    OpenActionDescription,
    CloseActionDescription,
    NavigateAction,
    PickUpAction,
    PlaceAction,
)
from pycram.testing import ApartmentWorldTestCase


class ORMaticBaseTestCaseMixin(ApartmentWorldTestCase):
    engine: sqlalchemy.engine
    session: Session

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.engine = create_engine("sqlite:///:memory:")

    def setUp(self):
        super().setUp()
        session = Session(engine)
        Base.metadata.create_all(bind=session.bind)

    def tearDown(self):
        super().tearDown()
        Base.metadata.drop_all(session.bind)
        session.expunge_all()
        session.close()


engine = create_engine("sqlite:///:memory:")


@pytest.fixture(scope="function")
def database():
    session = Session(engine)
    Base.metadata.create_all(bind=session.bind)
    yield session
    Base.metadata.drop_all(session.bind)
    session.expunge_all()
    session.close()


@pytest.fixture()
def test_simple_plan(immutable_model_world):
    world, robot_view, context = immutable_model_world

    with simulated_robot:
        plan = SequentialPlan(
            Context.from_world(world),
            NavigateActionDescription(
                PoseStamped.from_list([1.6, 1.9, 0], [0, 0, 0, 1], world.root),
                True,
            ),
            MoveTorsoActionDescription(TorsoState.HIGH),
            ParkArmsActionDescription(Arms.BOTH),
            # PickUpActionDescription(
            #     NamedObject("milk.stl"),
            #     Arms.LEFT,
            #     GraspDescription(
            #         ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
            #     ),
            # ),
            # PlaceActionDescription(
            #     NamedObject("milk.stl"),
            #     [
            #         PoseStamped.from_list(
            #             [2.3, 2.2, 1], [0, 0, 0, 1], world.root
            #         )
            #     ],
            #     [Arms.LEFT],
            # ),
        )
        plan.perform()
    return plan

def test_pose(database, test_simple_plan):
    session = database
    plan = test_simple_plan
    dao = to_dao(plan)
    session.add(dao)
    session.commit()
    result = session.scalars(select(PyCramPoseDAO)).all()
    assert len(result) > 0
    assert all([r.position is not None and r.orientation is not None for r in result])


def test_action_to_pose(database, test_simple_plan):
    session = database
    plan = test_simple_plan
    dao = to_dao(plan)
    session.add(dao)
    session.commit()
    # result = session.scalars(select(ActionDescriptionDAO)).all()
    result = session.scalars(
        select(ResolvedActionNodeMappingDAO).where(
            ResolvedActionNodeMappingDAO.designator_type == NavigateAction
        )
    ).all()
    assert all(
        [
            r.execution_data.execution_start_pose is not None
            and r.execution_data.execution_end_pose is not None
            for r in result
        ]
    )


def test_pose_vs_pose_stamped(database, test_simple_plan):
    session = database
    plan = test_simple_plan
    dao = to_dao(plan)
    session.add(dao)
    session.commit()
    pose_stamped_result = session.scalars(select(PoseStampedDAO)).all()
    pose_result = session.scalars(select(PyCramPoseDAO)).all()
    poses_from_pose_stamped_results = session.scalars(
        select(PyCramPoseDAO).where(
            PyCramPoseDAO.database_id.in_([r.pose_id for r in pose_stamped_result])
        )
    ).all()
    assert all([r.pose is not None for r in pose_stamped_result])
    assert all(
        [r.position is not None and r.orientation is not None for r in pose_result]
    )
    assert len(poses_from_pose_stamped_results) == len(pose_result)
    assert pose_stamped_result[0].pose_id == pose_result[0].database_id


def test_pose_creation(database, test_simple_plan):
    session = database
    plan = test_simple_plan
    pose = PyCramPose()
    pose.position.x = 1.0
    pose.position.y = 2.0
    pose.position.z = 3.0
    pose.orientation.x = 4.0
    pose.orientation.y = 5.0
    pose.orientation.z = 6.0
    pose.orientation.w = 7.0

    pose_dao = to_dao(pose)

    session.add(pose_dao.position)
    session.add(pose_dao.orientation)
    session.add(pose_dao)
    session.commit()

    with session.bind.connect() as conn:
        raw_pose = conn.execute(text("SELECT * FROM PyCramPoseDAO")).fetchall()

    pose_result = session.scalars(select(PyCramPoseDAO)).first()
    assert pose_result.position.x == 1.0
    assert pose_result.position.y == 2.0
    assert pose_result.position.z == 3.0
    assert pose_result.database_id == raw_pose[0][0]

# ORM Action Designator Tests


def test_code_designator_type(database, mutable_model_world):
    session = database
    world, robot_view, context = mutable_model_world
    action = SequentialPlan(
        context,
        NavigateActionDescription(
            PoseStamped.from_list([0.6, 0.4, 0], [0, 0, 0, 1], world.root),
            True,
        ),
    )
    with simulated_robot:
        action.perform()
    dao = to_dao(action)
    session.add(dao)
    session.commit()

    result = session.scalars(select(ResolvedActionNodeMappingDAO)).all()
    # motion = session.scalars(select(MoveMotionDAO)).all()
    assert result[0].designator_type == NavigateAction
    assert result[0].start_time < result[0].end_time
    # assertEqual(result[1].action.dtype, MoveMotion.__name__)


def test_inheritance(database, mutable_model_world):
    session = database
    world, robot_view, context = mutable_model_world
    with simulated_robot:
        sp = SequentialPlan(
            Context.from_world(world),
            NavigateActionDescription(
                PoseStamped.from_list([1.6, 1.9, 0], [0, 0, 0, 1], world.root),
                True,
            ),
            ParkArmsActionDescription(Arms.BOTH),
            PickUpActionDescription(
                world.get_body_by_name("milk.stl"),
                Arms.LEFT,
                GraspDescription(
                    ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
                ),
            ),
            NavigateActionDescription(
                PoseStamped.from_list([1.6, 2.3, 0], [0, 0, 0, 1], world.root),
                True,
            ),
            PlaceActionDescription(
                world.get_body_by_name("milk.stl"),
                PoseStamped.from_list([2.3, 2.5, 1.0], [0, 0, 0, 1], world.root),
                Arms.LEFT,
            ),
        )
        sp.perform()
    dao = to_dao(sp)
    session.add(dao)
    session.commit()

    result = session.scalars(select(ActionDescriptionDAO)).all()
    assert len(result) == 7


def test_parkArmsAction(database, mutable_model_world):
    session = database
    world, robot_view, context = mutable_model_world
    action = SequentialPlan(context, ParkArmsActionDescription(Arms.BOTH))
    with simulated_robot:
        action.perform()
    dao = to_dao(action)
    session.add(dao)
    session.commit()
    result = session.scalars(select(ParkArmsActionDAO)).all()
    assert 1 == len(result)
    assert type(result[0]).original_class() == ParkArmsAction


def test_transportAction(database, mutable_simple_pr2_world):
    session = database
    world, robot_view, context = mutable_simple_pr2_world
    action = SequentialPlan(
        Context.from_world(world),
        TransportActionDescription(
            world.get_body_by_name("milk.stl"),
            PoseStamped.from_list([1.7, 0.2, 1.07], [0, 0, 0, 1], world.root),
            Arms.LEFT,
        ),
    )
    with simulated_robot:
        action.perform()
    dao = to_dao(action)
    session.add(dao)
    session.commit()
    result = session.scalars(select(TransportActionDAO)).all()

    assert type(result[0]) == TransportActionDAO
    assert result[0].original_class() == TransportAction
    assert result[0].target_location is not None
    result = session.scalars(select(TransportActionDAO)).first()
    assert result is not None


def test_pickUpAction(database, mutable_model_world):
    session = database
    world, robot_view, context = mutable_model_world
    with simulated_robot:
        sp = SequentialPlan(
            Context.from_world(world),
            NavigateActionDescription(
                PoseStamped.from_list([1.6, 1.9, 0], [0, 0, 0, 1], world.root),
                True,
            ),
            ParkArmsActionDescription(Arms.BOTH),
            PickUpActionDescription(
                world.get_body_by_name("milk.stl"),
                Arms.LEFT,
                GraspDescription(
                    ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
                ),
            ),
            NavigateActionDescription(
                PoseStamped.from_list([1.6, 2.3, 0], [0, 0, 0, 1], world.root),
                True,
            ),
            PlaceActionDescription(
                world.get_body_by_name("milk.stl"),
                PoseStamped.from_list([2.3, 2.5, 1.0], [0, 0, 0, 1], world.root),
                Arms.LEFT,
            ),
        )
        sp.perform()
    dao = to_dao(sp)
    session.add(dao)
    session.commit()
    result = session.scalars(select(PickUpActionDAO)).first()


def test_setGripperAction(database, mutable_model_world):
    session = database
    world, robot_view, context = mutable_model_world
    action = SequentialPlan(
        context, SetGripperActionDescription(Arms.LEFT, GripperState.OPEN)
    )
    with simulated_robot:
        action.perform()
    dao = to_dao(action)
    session.add(dao)
    session.commit()
    result = session.scalars(select(SetGripperActionDAO)).all()
    assert result[0].gripper == Arms.LEFT
    assert result[0].motion == GripperState.OPEN


def test_open_and_closeAction(database, mutable_model_world):
    session = database
    world, robot_view, context = mutable_model_world
    with simulated_robot:
        sp = SequentialPlan(
            Context.from_world(world),
            ParkArmsActionDescription(Arms.BOTH),
            NavigateActionDescription(
                PoseStamped.from_list(
                    [1.81, 1.73, 0.0], [0.0, 0.0, 0.594, 0.804], world.root
                ),
                True,
            ),
            OpenActionDescription(
                world.get_body_by_name("handle_cab10_t"),
                arm=Arms.LEFT,
                grasping_prepose_distance=0.03,
            ),
            CloseActionDescription(
                world.get_body_by_name("handle_cab10_t"),
                arm=Arms.LEFT,
                grasping_prepose_distance=0.03,
            ),
        )
        sp.perform()
    dao = to_dao(sp)
    session.add(dao)
    session.commit()
    open_result = session.scalars(select(OpenActionDAO)).all()
    close_result = session.scalars(select(CloseActionDAO)).all()
    assert open_result is not None
    # can not do that yet with new mapping
    # assertEqual(open_result[0].object.name, "handle_cab10_t")
    assert close_result is not None
    # can not do that yet with new mapping
    # assertEqual(close_result[0].object.name, "handle_cab10_t")


def test_parallel_plan(database, mutable_model_world):
    session = database
    world, robot_view, context = mutable_model_world
    plan = ParallelPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
    )

    with simulated_robot:
        plan.perform()

    dao = to_dao(plan)
    session.add(dao)
    session.commit()

    park_result = session.scalars(select(ParkArmsActionDAO)).all()
    move_torso_result = session.scalars(select(MoveTorsoActionDAO)).all()

    assert park_result is not None
    assert move_torso_result is not None


# Exec Data Tests


@pytest.fixture
def complex_plan(mutable_model_world):
    world, robot_view, context = mutable_model_world
    with simulated_robot:
        sp = SequentialPlan(
            Context.from_world(world),
            NavigateActionDescription(
                PoseStamped.from_list([1.6, 1.9, 0], [0, 0, 0, 1], world.root),
                True,
            ),
            ParkArmsActionDescription(Arms.BOTH),
            PickUpActionDescription(
                world.get_body_by_name("milk.stl"),
                Arms.LEFT,
                GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.NoAlignment,
                    False,
                ),
            ),
            NavigateActionDescription(
                PoseStamped.from_list([1.6, 2.3, 0], [0, 0, 0, 1], world.root),
                True,
            ),
            MoveTorsoActionDescription(TorsoState.HIGH),
            PlaceActionDescription(
                world.get_body_by_name("milk.stl"),
                PoseStamped.from_list([2.3, 2.5, 1], [0, 0, 0, 1], world.root),
                Arms.LEFT,
            ),
        )

        sp.perform()
    return sp


def test_exec_creation(database, immutable_model_world):
    session = database
    world, robot_view, context = immutable_model_world
    plan = SequentialPlan(
        context,
        NavigateActionDescription(
            PoseStamped.from_list([0.6, 0.4, 0], [0, 0, 0, 1], world.root),
            True,
        ),
    )

    with simulated_robot:
        plan.perform()
    dao = to_dao(plan)
    session.add(dao)
    session.commit()
    exec_data = session.scalars(select(ExecutionDataDAO)).all()
    assert exec_data is not None


def test_exec_data_pose(database, immutable_model_world):
    session = database
    world, robot_view, context = immutable_model_world
    plan = SequentialPlan(
        context,
        NavigateActionDescription(
            PoseStamped.from_list([0.6, 0.4, 0], [0, 0, 0, 1], world.root),
            True,
        ),
    )

    with simulated_robot:
        plan.perform()

    dao = to_dao(plan)
    session.add(dao)
    session.commit()
    exec_data = session.scalars(select(ExecutionDataDAO)).all()[0]
    assert exec_data is not None
    assert (
        [1.5, 2.5, 0]
        == [
            exec_data.execution_start_pose.pose.position.x,
            exec_data.execution_start_pose.pose.position.y,
            exec_data.execution_start_pose.pose.position.z,
        ],
    )
    np.testing.assert_almost_equal(
        [0.6, 0.4, 0],
        [
            exec_data.execution_end_pose.pose.position.x,
            exec_data.execution_end_pose.pose.position.y,
            exec_data.execution_end_pose.pose.position.z,
        ],
        decimal=1,
    )


def test_manipulated_body_pose(database, complex_plan):
    session = database
    plan = complex_plan
    dao = to_dao(plan)
    session.add(dao)
    session.commit()

    # pick_up = session.scalars(select(PickUpActionDAO)).all()[0]
    pick_up_node = session.scalars(
        select(ResolvedActionNodeMappingDAO).where(
            ResolvedActionNodeMappingDAO.designator_type == PickUpAction
        )
    ).all()[0]
    place_node = session.scalars(
        select(ResolvedActionNodeMappingDAO).where(
            ResolvedActionNodeMappingDAO.designator_type == PlaceAction
        )
    ).all()[0]
    # place = session.scalars(select(PlaceActionDAO)).all()[0]
    assert (pick_up_node.execution_data.manipulated_body_pose_start) is not None
    assert (pick_up_node.execution_data.manipulated_body_pose_end) is not None
    start_pose_pick = PoseStampedDAO.from_dao(
        pick_up_node.execution_data.manipulated_body_pose_start
    )
    end_pose_pick = PoseStampedDAO.from_dao(
        pick_up_node.execution_data.manipulated_body_pose_end
    )
    start_pose_place = PoseStampedDAO.from_dao(
        place_node.execution_data.manipulated_body_pose_start
    )
    end_pose_place = PoseStampedDAO.from_dao(
        place_node.execution_data.manipulated_body_pose_end
    )

    # assertListEqual([2.37, 2, 1.05], start_pose_pick.position.to_list())
    np.testing.assert_almost_equal(
        [2.37, 2, 1.05], end_pose_pick.position.to_list(), decimal=1
    )
    # Check that the end_pose of pick_up and start pose of place are not equal because of navigate in between
    for pick, place in zip(
        end_pose_pick.position.to_list(), start_pose_place.position.to_list()
    ):
        assert pick != place
    np.testing.assert_almost_equal(
        [2.3, 2.5, 1], end_pose_place.position.to_list(), decimal=1
    )


def test_manipulated_body(database, complex_plan):
    session = database
    plan = complex_plan

    dao = to_dao(plan)
    session.add(dao)
    session.commit()

    pick_up_node = session.scalars(
        select(ResolvedActionNodeMappingDAO).where(
            ResolvedActionNodeMappingDAO.designator_type == PickUpAction
        )
    ).all()[0]
    assert (pick_up_node.execution_data.manipulated_body) is not None
    milk = BodyDAO.from_dao(pick_up_node.execution_data.manipulated_body)
    assert milk.name.name == "milk.stl"


def test_state(database, immutable_model_world):
    world, robot_view, context = immutable_model_world
    session = database
    plan = SequentialPlan(
        context,
        NavigateActionDescription(
            PoseStamped.from_list([0.6, 0.4, 0], [0, 0, 0, 1], world.root),
            True,
        ),
    )
    with simulated_robot:
        plan.perform()
    dao = to_dao(plan)
    session.add(dao)
    session.commit()
    navigate_node = session.scalars(
        select(ResolvedActionNodeMappingDAO).where(
            ResolvedActionNodeMappingDAO.designator_type == NavigateAction
        )
    ).all()[0]
    assert (navigate_node.execution_data.execution_start_world_state) is not None


# Relational Algebra Tests


def test_filtering(database, mutable_model_world):
    session = database
    world, robot_view, context = mutable_model_world
    with simulated_robot:
        sp = SequentialPlan(
            context,
            NavigateActionDescription(
                PoseStamped.from_list([1.6, 1.9, 0], [0, 0, 0, 1], world.root),
                True,
            ),
            ParkArmsActionDescription(Arms.BOTH),
            PickUpActionDescription(
                world.get_body_by_name("milk.stl"),
                Arms.LEFT,
                GraspDescription(
                    ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
                ),
            ),
            NavigateActionDescription(
                PoseStamped.from_list([1.6, 2.3, 0], [0, 0, 0, 1], world.root),
                True,
            ),
            PlaceActionDescription(
                world.get_body_by_name("milk.stl"),
                PoseStamped.from_list([2.3, 2.5, 1], [0, 0, 0, 1], world.root),
                Arms.LEFT,
            ),
        )
        sp.perform()
    dao = to_dao(sp)
    session.add(dao)
    session.commit()

    filtered_navigate_results = session.scalars(
        select(NavigateActionDAO).where(NavigateActionDAO.database_id == 1)
    ).all()
    assert 1 == len(filtered_navigate_results)
