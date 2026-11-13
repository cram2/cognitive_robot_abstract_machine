"""
Tests for watching a body without collision geometry, such as a wall outlet of a kitchen
that is drawn but cannot be touched: nothing can touch or support it, and watching it
does not keep the detectors from judging the other bodies.
"""

from __future__ import annotations

import pytest

from segmind.datastructures.events import ContactEvent, SupportEvent
from segmind.detectors.atomic_event_detectors_nodes import ContactDetector
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

from .test_several_watched_bodies import _events_of, _ticked_while_nothing_moves

# %% a body without collision geometry among the watched ones


@pytest.fixture
def box_in_the_tray_beside_an_outlet(box_and_trays) -> tuple:
    """
    :return: The world of :func:`box_and_trays` with the box set down in the tray, the
        box, and an outlet without collision geometry free to move in the world.
    """
    world = box_and_trays.world
    outlet = Body(name=PrefixedName("outlet"))
    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world, parent=world.root, child=outlet
            )
        )
    box_and_trays.box.parent_connection.origin = box_and_trays.set_down_in_the_tray
    return world, box_and_trays.box, outlet


def test_a_body_without_collision_geometry_touches_nothing(
    box_in_the_tray_beside_an_outlet,
):
    world, box, outlet = box_in_the_tray_beside_an_outlet

    segmind_context = _ticked_while_nothing_moves(
        world, [ContactDetector, SupportDetector], [box, outlet]
    )

    assert _events_of(segmind_context, ContactEvent, outlet) == []
    assert _events_of(segmind_context, SupportEvent, outlet) == []


def test_watching_a_body_without_collision_geometry_leaves_the_others_judged(
    box_in_the_tray_beside_an_outlet,
):
    world, box, outlet = box_in_the_tray_beside_an_outlet

    segmind_context = _ticked_while_nothing_moves(
        world, [SupportDetector], [box, outlet]
    )

    assert len(_events_of(segmind_context, SupportEvent, box)) == 1
