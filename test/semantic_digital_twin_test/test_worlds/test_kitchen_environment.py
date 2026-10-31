"""
Geometric regression tests for the predefined kitchen environment.
"""

from __future__ import annotations

import numpy as np
import pytest
from semantic_digital_twin.predefined_maps.kitchen_environment import (
    KitchenEnvironment,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cooktop,
    CounterTop,
    KitchenIsland,
)
from semantic_digital_twin.world import World

# %% mesh bounds


def visual_world_bounds(world: World, body_name: str) -> np.ndarray:
    """
    Return a body's visual mesh bounds in the world root frame.
    """
    body = world.get_body_by_name(body_name)
    mesh = body.visual.combined_mesh.copy()
    mesh.apply_transform(world.compute_forward_kinematics_np(world.root, body))
    return mesh.bounds


# %% handle placement


def test_module_1_has_fixed_face_plate_above_door() -> None:
    """
    Module 1 should have a 14.3 cm face plate separated from its door.
    """
    world = KitchenEnvironment().get_world()
    face_plate_bounds = visual_world_bounds(world, "module_1_face_plate")
    door_bounds = visual_world_bounds(world, "module_1_door")

    assert np.isclose(np.ptp(face_plate_bounds[:, 2]), 0.143)
    assert np.isclose(face_plate_bounds[0, 2] - door_bounds[1, 2], 0.005)


@pytest.mark.parametrize("body_name", ["module_1_face_plate", "module_1_door"])
def test_module_1_front_piece_is_59_5_centimeters_wide(body_name: str) -> None:
    """
    Each Module 1 front piece should be narrower than the cabinet carcass.
    """
    world = KitchenEnvironment().get_world()
    body = world.get_body_by_name(body_name)

    assert np.isclose(body.visual.combined_mesh.extents[1], 0.595)


def test_module_1_handle_top_is_four_centimeters_below_door_top() -> None:
    """
    The Module 1 handle should be inset from the shortened door's top edge.
    """
    world = KitchenEnvironment().get_world()
    handle_bounds = visual_world_bounds(world, "module_1_handle")
    door_bounds = visual_world_bounds(world, "module_1_door")

    assert np.isclose(door_bounds[1, 2] - handle_bounds[1, 2], 0.04)


@pytest.mark.parametrize(
    ("handle_name", "door_name"),
    [
        ("dishwasher_handle", "dishwasher_door"),
        ("oven_handle", "oven_door"),
        ("module_1_handle", "module_1_door"),
        ("oven_cabinet_handle", "oven_cabinet_door"),
    ],
)
def test_horizontal_door_handle_fits_within_door_width(
    handle_name: str, door_name: str
) -> None:
    """
    A full-width horizontal handle should not extend beyond either door edge.
    """
    world = KitchenEnvironment().get_world()
    handle_bounds = visual_world_bounds(world, handle_name)
    door_bounds = visual_world_bounds(world, door_name)

    assert handle_bounds[0, 0] >= door_bounds[0, 0]
    assert handle_bounds[1, 0] <= door_bounds[1, 0]


def test_fridge_door_handle_is_near_free_edge() -> None:
    """
    The vertical fridge handle should be inset from the edge opposite its hinge.
    """
    world = KitchenEnvironment().get_world()
    handle_bounds = visual_world_bounds(world, "fridge_door_handle")
    door_bounds = visual_world_bounds(world, "fridge_door")

    handle_center_x = handle_bounds[:, 0].mean()
    assert np.isclose(door_bounds[1, 0] - handle_center_x, 0.03)


# %% the kitchen island


def test_the_sideboard_is_a_kitchen_island_built_from_its_corpus_under_its_worktop() -> (
    None
):
    """
    The sideboard is the kitchen island: its corpus is the cabinet built into it and its
    worktop is the island's counter top, as in the other modelled kitchens.
    """
    world = KitchenEnvironment().get_world()
    [island] = world.get_semantic_annotations_by_type(KitchenIsland)

    assert island.cabinets == [
        world.get_semantic_annotation_by_name("sideboard_cabinet")
    ]
    assert island.counter_top is world.get_semantic_annotation_by_name("sideboard")
    assert isinstance(island.counter_top, CounterTop)


def test_the_island_worktop_still_carries_its_cooktop() -> None:
    """
    Naming the worktop a counter top must not lose what stands on it.
    """
    world = KitchenEnvironment().get_world()
    worktop = world.get_semantic_annotation_by_name("sideboard")

    assert worktop.get_objects_of_type(Cooktop) == [
        world.get_semantic_annotation_by_name("sideboard_cooktop")
    ]
