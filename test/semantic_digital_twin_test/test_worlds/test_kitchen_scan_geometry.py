"""
Check measured kitchen placements and cabinet-to-wall relationships.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest

from semantic_digital_twin.predefined_maps.kitchen_environment import KitchenEnvironment
from semantic_digital_twin.world import World

from .test_kitchen_environment import visual_world_bounds


# %% Measured geometry
@pytest.fixture
def measured_kitchen() -> World:
    """
    Construct the predefined kitchen with its default scan measurements.
    """
    return KitchenEnvironment().get_world()


@pytest.mark.parametrize(
    ("body_name", "front_coordinate"),
    [
        ("module_1_door", -1.894),
        ("dishwasher_door", -1.894),
        ("oven_door", -1.871),
        ("fridge_door", -1.889),
        ("counter_top", -1.890),
        ("east_wall", -2.480),
    ],
)
def test_observed_run_faces_match_scan_planes(
    measured_kitchen: World, body_name: str, front_coordinate: float
) -> None:
    """
    Observed faces occupy their measured positions in the registered root frame.
    """
    body = measured_kitchen.get_body_by_name(body_name)
    mesh = body.visual.combined_mesh.copy()
    mesh.apply_transform(
        measured_kitchen.compute_forward_kinematics_np(measured_kitchen.root, body)
    )
    assert mesh.bounds[1, 1] == pytest.approx(front_coordinate, abs=1e-6)


def test_island_corpus_uses_measured_horizontal_placement(
    measured_kitchen: World,
) -> None:
    """
    The island moves as a rigid assembly while retaining floor contact.
    """
    body = measured_kitchen.get_body_by_name("sideboard_cabinet")
    pose = measured_kitchen.compute_forward_kinematics_np(measured_kitchen.root, body)
    np.testing.assert_allclose(pose[:3, 3], (3.52835439, 0.46312429, 0.4125), atol=1e-7)
    assert np.arctan2(pose[1, 0], pose[0, 0]) == pytest.approx(
        np.pi / 2 + np.radians(0.3315143615)
    )


@pytest.mark.parametrize("body_name", ["refrigerator", "oven_tower", "counter_top"])
def test_run_backs_remain_flush_with_wall_interior(
    measured_kitchen: World, body_name: str
) -> None:
    """
    Depth corrections preserve the existing flush-back construction constraint.
    """
    body = measured_kitchen.get_body_by_name(body_name)
    wall = measured_kitchen.get_body_by_name("east_wall")
    bounds = []
    for item in (body, wall):
        mesh = item.visual.combined_mesh.copy()
        mesh.apply_transform(
            measured_kitchen.compute_forward_kinematics_np(measured_kitchen.root, item)
        )
        bounds.append(mesh.bounds)
    assert bounds[0][0, 1] == pytest.approx(bounds[1][1, 1], abs=1e-6)


# %% Refrigerator-side wall junctions
@pytest.mark.parametrize(
    ("body_name", "axis", "side", "coordinate"),
    [("south_wall1", 0, 1, 0.0315), ("south_wall2", 1, 0, -1.4812)],
)
def test_refrigerator_walls_match_observed_planes(
    measured_kitchen: World,
    body_name: str,
    axis: int,
    side: int,
    coordinate: float,
) -> None:
    """
    The two exposed wall faces occupy their measured scan planes.
    """
    bounds = visual_world_bounds(measured_kitchen, body_name)
    assert bounds[side, axis] == pytest.approx(coordinate, abs=1e-6)


def test_refrigerator_wall_meets_short_return(measured_kitchen: World) -> None:
    """
    The side wall reaches the return face without an open junction.
    """
    wall = visual_world_bounds(measured_kitchen, "south_wall1")
    short_return = visual_world_bounds(measured_kitchen, "south_wall2")
    assert wall[1, 1] == pytest.approx(short_return[0, 1], abs=1e-6)
    assert wall[1, 0] == pytest.approx(short_return[1, 0], abs=1e-6)


def test_refrigerator_wall_overlaps_east_wall(measured_kitchen: World) -> None:
    """
    The side wall remains connected to the wall behind the kitchen run.
    """
    wall = visual_world_bounds(measured_kitchen, "south_wall1")
    east = visual_world_bounds(measured_kitchen, "east_wall")
    assert wall[0, 1] < east[1, 1]
    assert east[0, 1] < wall[0, 1]
    assert wall[0, 0] < east[0, 0] < wall[1, 0]
