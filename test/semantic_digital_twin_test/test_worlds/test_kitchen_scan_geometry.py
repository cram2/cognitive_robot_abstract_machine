"""
Check measured kitchen placements and cabinet-to-wall relationships.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest

from semantic_digital_twin.predefined_maps.kitchen_environment import KitchenEnvironment
from semantic_digital_twin.world import World
from semantic_digital_twin.spatial_types.spatial_types import Point3

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


# %% Windowed oven-side wall
def test_oven_wall_matches_main_wall_plane(measured_kitchen: World) -> None:
    """
    The wall follows the exposed plaster rather than the cable channel.
    """
    bounds = visual_world_bounds(measured_kitchen, "north_wall")
    assert bounds[0, 0] == pytest.approx(4.986, abs=1e-6)


def test_east_wall_reaches_recessed_window_wall(measured_kitchen: World) -> None:
    """
    The corrected wall remains joined to the kitchen-run wall.
    """
    east = visual_world_bounds(measured_kitchen, "east_wall")
    north = visual_world_bounds(measured_kitchen, "north_wall")
    assert east[1, 0] == pytest.approx(north[0, 0], abs=1e-6)


def test_window_wall_remains_connected_to_curved_corner(
    measured_kitchen: World,
) -> None:
    """
    The shifted wall overlaps the existing curved wall at its far end.
    """
    junction = Point3(4.99, 5.55, 1.5, reference_frame=measured_kitchen.root)
    wall = measured_kitchen.get_body_by_name("north_wall")
    corner = measured_kitchen.get_body_by_name("north_west_wall_body")
    assert wall.collision.as_bounding_box_collection_in_frame(
        measured_kitchen.root
    ).contains(junction)
    corner_point = measured_kitchen.transform(junction, corner).to_np()[:3]
    assert corner.collision.combined_mesh.contains(np.array([corner_point]))[0]


def test_oven_wall_has_three_semantic_window_openings(measured_kitchen: World) -> None:
    """
    Each scanned window bay has an aperture in the wall annotation.
    """
    wall = measured_kitchen.get_semantic_annotation_by_name("north_wall")
    assert len(wall.apertures) == 3


@pytest.mark.parametrize("along_wall_y", [-1.28, 1.46, 4.21])
def test_window_recess_contains_no_front_wall_material(
    measured_kitchen: World, along_wall_y: float
) -> None:
    """
    Window openings remove wall collision material at the room-facing plane.
    """
    wall = measured_kitchen.get_body_by_name("north_wall")
    collision = wall.collision.as_bounding_box_collection_in_frame(
        measured_kitchen.root
    )
    assert not collision.contains(
        Point3(5.0, along_wall_y, 1.2, reference_frame=measured_kitchen.root)
    )


@pytest.mark.parametrize("window_index", [0, 1, 2])
def test_recessed_windows_remain_closed_collision_surfaces(
    measured_kitchen: World, window_index: int
) -> None:
    """
    A recessed panel represents each closed window behind its aperture.
    """
    bounds = visual_world_bounds(measured_kitchen, f"north_window_{window_index}_panel")
    assert bounds[0, 0] == pytest.approx(5.129, abs=1e-6)
    assert bounds[1, 0] > bounds[0, 0]


def test_first_window_sill_matches_observed_top_and_front(
    measured_kitchen: World,
) -> None:
    """
    The kitchen sill projects in front of the wall at the measured height.
    """
    bounds = visual_world_bounds(measured_kitchen, "north_window_0_sill")
    assert bounds[1, 2] == pytest.approx(0.619, abs=1e-6)
    assert bounds[0, 0] == pytest.approx(4.945, abs=1e-6)


def test_cable_channel_leaves_the_lower_wall_strip_exposed(
    measured_kitchen: World,
) -> None:
    """
    The channel is raised above the floor with a separate surface beneath it.
    """
    channel = measured_kitchen.get_body_by_name("north_wall_cable_channel")
    lower_strip = measured_kitchen.get_body_by_name("north_wall_lower_strip")
    channel_mesh = channel.collision.combined_mesh
    lower_mesh = lower_strip.collision.combined_mesh
    point = np.array([[4.95, -1.0, 0.04]])
    assert not channel_mesh.contains(point)[0]
    assert lower_mesh.contains(point)[0]
    assert channel_mesh.contains(np.array([[4.95, -1.0, 0.15]]))[0]
