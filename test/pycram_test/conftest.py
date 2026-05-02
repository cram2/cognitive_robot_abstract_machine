from copy import deepcopy
from functools import partial

import pytest
import rclpy
from sqlalchemy.orm import sessionmaker

from krrood.ormatic.utils import create_engine, drop_database
from pycram.datastructures.dataclasses import Context

from pycram.orm.ormatic_interface import Base

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch


@pytest.fixture(scope="session")
def viz_marker_publisher():
    rclpy.init()
    node = rclpy.create_node("test_viz_marker_publisher")
    # VizMarkerPublisher(world, node)  # Initialize the publisher
    yield partial(VizMarkerPublisher, node=node)
    rclpy.shutdown()


@pytest.fixture(scope="function")
def mutable_model_world(pr2_apartment_world):
    world = deepcopy(pr2_apartment_world)
    pr2 = PR2.from_world(world)
    return world, pr2, Context(world, pr2)


@pytest.fixture(scope="function")
def immutable_model_world(pr2_apartment_world):
    world = pr2_apartment_world
    pr2 = pr2_apartment_world.get_semantic_annotations_by_type(PR2)[0]
    state = deepcopy(world.state._data)
    yield world, pr2, Context(world, pr2)
    world.state._data[:] = state
    world.notify_state_change()


@pytest.fixture
def immutable_simple_pr2_world(simple_pr2_world_setup):
    world, robot_view, context = simple_pr2_world_setup
    state = deepcopy(world.state._data)
    yield world, robot_view, context
    world.state._data[:] = state
    world.notify_state_change()


@pytest.fixture
def mutable_simple_pr2_world(simple_pr2_world_setup):
    world, robot_view, context = simple_pr2_world_setup
    copy_world = deepcopy(world)
    robot_view = world.get_semantic_annotations_by_type(PR2)[0]
    return world, robot_view, Context(copy_world, robot_view)


@pytest.fixture(scope="function")
def pycram_testing_session():
    engine = create_engine("sqlite:///:memory:")
    session_maker = sessionmaker(engine)
    session = session_maker()
    Base.metadata.create_all(bind=session.bind)
    yield session
    drop_database(session.bind)
    session.close()
    engine.dispose()


@pytest.fixture(scope="function")
def immutable_stretch_apartment_world(stretch_apartment_world):
    context = Context(
        stretch_apartment_world,
        Stretch.from_world(stretch_apartment_world),
    )
    state = deepcopy(stretch_apartment_world.state._data)

    yield stretch_apartment_world, context.robot, context

    stretch_apartment_world.state._data[:] = state
