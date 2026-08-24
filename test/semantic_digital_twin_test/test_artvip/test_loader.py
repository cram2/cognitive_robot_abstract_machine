import os
import time

import pytest
from huggingface_hub.errors import HfHubHTTPError

from semantic_digital_twin.adapters.artvip_dataset.loader import ArtVipDatasetLoader
from semantic_digital_twin.adapters.artvip_dataset.schema import (
    ArtVipCategory,
    ArtVipObject,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)

try:
    import pxr  # noqa: F401

    PXR_AVAILABLE = True
except ImportError:
    PXR_AVAILABLE = False

DOOR_OBJECT_NAME = "EKET_Cabinet_with_door_brown_walnut_effect_35x35x35cm"
DRAWERS_OBJECT_NAME = "EKET_Cabinet_with_2_drawers_white_70x35x35cm"

pytestmark = pytest.mark.skipif(
    not PXR_AVAILABLE, reason="usd-core (pxr) not installed"
)


def get_artvip_object(name: str) -> ArtVipObject | None:
    try:
        loader = ArtVipDatasetLoader()
        return loader.load(ArtVipCategory.IKEA_FURNITURE, name)
    except HfHubHTTPError:
        return None


@pytest.fixture(scope="session")
def door_object() -> ArtVipObject:
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker:
        worker_num = int(worker.removeprefix("gw"))
        time.sleep(worker_num)
    obj = get_artvip_object(DOOR_OBJECT_NAME)
    if obj is None:
        pytest.skip("ArtVIP dataset not available")
    return obj


@pytest.fixture(scope="session")
def drawers_object() -> ArtVipObject:
    obj = get_artvip_object(DRAWERS_OBJECT_NAME)
    if obj is None:
        pytest.skip("ArtVIP dataset not available")
    return obj


def test_door_cabinet_builds_a_tree_with_one_hinge(door_object):
    world = door_object.world
    assert len(world.bodies) == 3  # root + carcass + door
    assert len(world.connections) == 2
    assert world.root is not None

    [hinge] = [c for c in world.connections if isinstance(c, RevoluteConnection)]
    assert hinge.raw_dof.limits.lower.position == pytest.approx(-1.5707963267948966)
    assert hinge.raw_dof.limits.upper.position == pytest.approx(0.0)

    [fixed] = [c for c in world.connections if isinstance(c, FixedConnection)]
    assert fixed.parent is world.root


def test_drawer_cabinet_builds_a_tree_with_two_slides(drawers_object):
    world = drawers_object.world
    assert len(world.bodies) == 4  # root + carcass + 2 drawers
    assert len(world.connections) == 3
    assert world.root is not None

    slides = [c for c in world.connections if isinstance(c, PrismaticConnection)]
    assert len(slides) == 2
    for slide in slides:
        assert slide.raw_dof.limits.lower.position < slide.raw_dof.limits.upper.position


def test_every_body_has_mesh_geometry(door_object):
    world = door_object.world
    non_root_bodies = [body for body in world.bodies if body is not world.root]
    assert non_root_bodies
    for body in non_root_bodies:
        assert len(body.visual.shapes) > 0


def test_hinged_door_is_reachable_as_an_active_connection_from_its_body(door_object):
    # giskardpy's Open/Close goals (motion_statechart/goals/open_close.py) drive any
    # mechanism whose grasped part "hangs below an ActiveConnection1DOF" by calling
    # exactly this lookup on the grasped Body - this is the actual shape a manipulation
    # plan would query, not just that the World happens to contain a RevoluteConnection
    # somewhere.
    world = door_object.world
    [hinge] = [c for c in world.connections if isinstance(c, RevoluteConnection)]

    assert (
        hinge.child.get_first_parent_connection_of_type(ActiveConnection1DOF) is hinge
    )
