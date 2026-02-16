import os
from copy import deepcopy
import pytest
import rclpy

from krrood.ormatic.dao import to_dao
from krrood.ormatic.utils import create_engine, drop_database
from pycram.motion_executor import simulated_robot
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)

# from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher

from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from sqlalchemy.orm import Session, session

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import (
    ApproachDirection,
    Arms,
    VerticalAlignment,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PyCramPose, PoseStamped
from pycram.language import SequentialPlan, ParallelPlan
from pycram.orm.ormatic_interface import *
from pycram.robot_plans import (
    MoveTorsoActionDescription,
    NavigateActionDescription,
    PickUpActionDescription,
    MoveAndPickUpActionDescription,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

rclpy.init()
uri = os.environ["SEMANTIC_DIGITAL_TWIN_DATABASE_URI"]
engine = sqlalchemy.create_engine(uri)
node = rclpy.create_node("simple_viz_node")


@pytest.fixture(scope="function")
def mutable_model_world(pr2_apartment_world):
    world = deepcopy(pr2_apartment_world)
    pr2 = PR2.from_world(world)
    return world, pr2, Context(world, pr2)


@pytest.fixture(scope="function")
def database():
    session = Session(engine)
    Base.metadata.create_all(bind=session.bind)
    yield session
    drop_database(engine)
    session.expunge_all()
    session.close()


def test_pick_up(database, mutable_model_world):
    world, robot_view, context = mutable_model_world
    viz_publisher = VizMarkerPublisher(world, node)
    with_tf = viz_publisher.with_tf_publisher()

    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        robot_view.left_arm.manipulator,
    )
    description = PickUpActionDescription(
        world.get_body_by_name("milk.stl"), [Arms.LEFT], [grasp_description]
    )

    plan = SequentialPlan(
        context,
        NavigateActionDescription(
            PoseStamped.from_list([1.7, 1.5, 0], [0, 0, 0, 1], world.root),
            True,
        ),
        MoveTorsoActionDescription([TorsoState.HIGH]),
        description,
    )
    with simulated_robot:
        plan.perform()

    session = database
    dao = to_dao(plan)
    session.add(dao)
    session.commit()


def test_navigate(database, mutable_model_world):
    world, robot_view, context = mutable_model_world

    description = NavigateActionDescription(None)

    plan = SequentialPlan(
        context,
        description,
    )
    params = plan.parameterize()
    p = params.create_fully_factorized_distribution()
    s = p.sample(10)

    print(s)
    for sample in s:
        print(dict(zip(p.variables, sample)))
    # with simulated_robot:
    #     plan.perform()
    #
    # session = database
    # dao = to_dao(plan)
    # session.add(dao)
    # session.commit()


def test_move_and_pick_up(database, mutable_model_world):
    world, robot_view, context = mutable_model_world

    description = MoveAndPickUpActionDescription(None, None, None, None, None)

    plan = SequentialPlan(
        context,
        description,
    )
    params = plan.parameterize()
    p = params.create_fully_factorized_distribution()
    s = p.sample(10)

    print(s)
    for sample in s:
        print(dict(zip(p.variables, sample)))
    # with simulated_robot:
    #     plan.perform()
    #
    # session = database
    # dao = to_dao(plan)
    # session.add(dao)
    # session.commit()
