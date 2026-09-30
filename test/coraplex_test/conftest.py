# %% ORM interfaces

# Built before the imports below, which read a mapped datastructure: pytest imports every
# conftest of a run before calling any hook, so a hook would fire too late. The build runs
# once per process and never on an xdist worker.
from ..orm_interface_build import regenerate_orm_interfaces

regenerate_orm_interfaces()


from functools import partial

import pytest

from semantic_digital_twin.predetermined_maps.building_floor import BuildingFloor
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World

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
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox

from .world_snapshot import WorldSnapshot


@pytest.fixture(scope="session")
def viz_marker_publisher():
    rclpy.init()
    node = rclpy.create_node("test_viz_marker_publisher")
    # VizMarkerPublisher(world, node)  # Initialize the publisher
    yield partial(VizMarkerPublisher, node=node)
    rclpy.shutdown()


# %% world rollback


@pytest.fixture(scope="function")
def pr2_apartment_context(pr2_apartment_world):
    """
    The shared PR2 apartment world, its robot and a context for both, returned to its
    initial model and state after the test.
    """
    snapshot = WorldSnapshot.capture(pr2_apartment_world)
    pr2 = pr2_apartment_world.get_semantic_annotations_by_type(PR2)[0]
    yield pr2_apartment_world, pr2, Context(pr2_apartment_world, pr2)
    snapshot.restore()


@pytest.fixture(scope="function")
def simple_pr2_context(simple_pr2_world_setup):
    """
    The shared PR2 world in the simple apartment, its robot and a context for both,
    returned to its initial model and state after the test.
    """
    world, robot_view, context = simple_pr2_world_setup
    snapshot = WorldSnapshot.capture(world)
    yield world, robot_view, context
    snapshot.restore()


@pytest.fixture(scope="function")
def stretch_apartment_context(stretch_apartment_world):
    """
    The shared Stretch apartment world, its robot and a context for both, returned to
    its initial model and state after the test.
    """
    snapshot = WorldSnapshot.capture(stretch_apartment_world)
    robot = stretch_apartment_world.get_semantic_annotations_by_type(Stretch)[0]
    yield stretch_apartment_world, robot, Context(stretch_apartment_world, robot)
    snapshot.restore()


# %% database session


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


# %% perception regions


@pytest.fixture
def whole_scene_region(pr2_apartment_context) -> VolumetricBoundingBox:
    """
    A region large enough to contain everything in the apartment fixture.

    Lets a perception test say "look everywhere" without restating the extents.
    """
    world, _, _ = pr2_apartment_context
    return VolumetricBoundingBox(
        origin=HomogeneousTransformationMatrix(reference_frame=world.root),
        min_x=-10,
        min_y=-10,
        min_z=-10,
        max_x=10,
        max_y=10,
        max_z=10,
    )
