# %% ORM interfaces

# Built before the imports below, which read a mapped datastructure: pytest imports every
# conftest of a run before calling any hook, so a hook would fire too late. The build runs
# once per process and never on an xdist worker.
from ..orm_interface_build import regenerate_orm_interfaces

regenerate_orm_interfaces()


from copy import deepcopy
from functools import partial

import pytest

try:
    import rclpy
except ModuleNotFoundError:
    pass
from sqlalchemy.orm import sessionmaker

from krrood.ormatic.utils import create_engine, drop_database

try:
    from coraplex.datastructures.dataclasses import Context
except ModuleNotFoundError:
    pass

try:
    from coraplex.orm.ormatic_interface import Base
except ImportError:
    pass
try:
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )
except ModuleNotFoundError:
    pass
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import BoundingBox

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
    pr2 = world.get_semantic_annotations_by_type(PR2)[0]
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
def coraplex_testing_session():
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
    robot = stretch_apartment_world.get_semantic_annotations_by_type(Stretch)[0]
    context = Context(stretch_apartment_world, robot)
    state = deepcopy(stretch_apartment_world.state._data)

    yield stretch_apartment_world, robot, context

    stretch_apartment_world.state._data[:] = state
    stretch_apartment_world.notify_state_change()

@pytest.fixture
def whole_scene_region(immutable_model_world) -> BoundingBox:
    """
    A region large enough to contain everything in the apartment fixture.

    Lets a perception test say "look everywhere" without restating the extents.
    """
    world, _, _ = immutable_model_world
    return BoundingBox(
        origin=HomogeneousTransformationMatrix(reference_frame=world.root),
        min_x=-10,
        min_y=-10,
        min_z=-10,
        max_x=10,
        max_y=10,
        max_z=10,
    )
