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
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

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


# %% Handle mounting positions
@pytest.mark.parametrize(
    ("handle_name", "axis", "coordinate"),
    [
        ("fridge_door_handle", 0, 1.0595),
        ("fridge_drawer_handle", 2, 0.455),
        ("module_1_handle", 2, 0.6245),
        ("counter_drawer_0_handle", 2, 0.3385),
        ("counter_drawer_1_handle", 2, 0.6275),
        ("counter_drawer_2_handle", 2, 0.769),
        ("oven_side_handle_left", 0, 3.306),
        ("oven_side_handle_right", 0, 4.206),
        ("oven_cabinet_handle", 2, 0.674),
        ("oven_center_drawer_handle", 2, 0.822),
        ("sideboard_drawer_0_0_handle", 2, 0.348),
        ("sideboard_drawer_0_1_handle", 2, 0.637),
        ("sideboard_drawer_1_0_handle", 2, 0.342),
        ("sideboard_drawer_1_1_handle", 2, 0.636),
        ("sideboard_drawer_2_0_handle", 2, 0.342),
        ("sideboard_drawer_2_1_handle", 2, 0.634),
        ("dishwasher_handle", 2, 0.789),
        ("oven_handle", 2, 1.3265),
    ],
)
def test_handle_mount_matches_reviewed_scan_candidate(
    measured_kitchen: World,
    handle_name: str,
    axis: int,
    coordinate: float,
) -> None:
    """
    Handle mounts occupy reviewed transverse positions in the closed kitchen.
    """
    handle = measured_kitchen.get_body_by_name(handle_name)
    pose = measured_kitchen.compute_forward_kinematics_np(measured_kitchen.root, handle)
    assert pose[axis, 3] == pytest.approx(coordinate, abs=1e-6)


@pytest.mark.parametrize(
    ("handle_name", "parent_name"),
    [
        ("fridge_door_handle", "fridge_door"),
        ("counter_drawer_0_handle", "counter_drawer_0"),
        ("sideboard_drawer_1_1_handle", "sideboard_drawer_1_1"),
        ("oven_side_handle_left", "oven_side_drawer_left"),
        ("oven_side_handle_right", "oven_side_drawer_right"),
    ],
)
def test_handle_retains_its_mount_when_parent_opens(
    measured_kitchen: World,
    handle_name: str,
    parent_name: str,
) -> None:
    """
    A corrected handle follows its door or drawer with a fixed mounting pose.
    """
    handle = measured_kitchen.get_body_by_name(handle_name)
    parent = measured_kitchen.get_body_by_name(parent_name)
    assert handle.parent_connection.parent is parent
    before = measured_kitchen.compute_forward_kinematics_np(parent, handle)
    root_before = measured_kitchen.compute_forward_kinematics_np(
        measured_kitchen.root, handle
    )
    connection = parent.get_first_parent_connection_of_type(ActiveConnection1DOF)
    connection.position = 0.1
    after = measured_kitchen.compute_forward_kinematics_np(parent, handle)
    root_after = measured_kitchen.compute_forward_kinematics_np(
        measured_kitchen.root, handle
    )
    np.testing.assert_allclose(after, before, atol=1e-9)
    assert not np.allclose(root_after, root_before)


# %% Oven-side drawer fronts
@pytest.mark.parametrize(
    ("drawer_name", "front_y"),
    [("oven_side_drawer_left", -1.871), ("oven_side_drawer_right", -1.869)],
)
def test_oven_side_drawer_front_matches_reviewed_scan_plane(
    measured_kitchen: World,
    drawer_name: str,
    front_y: float,
) -> None:
    """
    The two pull-out fronts occupy their reviewed outward positions.
    """
    bounds = visual_world_bounds(measured_kitchen, drawer_name)
    assert bounds[1, 1] == pytest.approx(front_y, abs=1e-6)


# %% Sofa placement
def test_sofa_uses_scan_fitted_horizontal_placement(measured_kitchen: World) -> None:
    """
    The sofa follows its observed front and side without changing floor contact.
    """
    sofa = measured_kitchen.get_body_by_name("sofa")
    root_T_sofa = measured_kitchen.compute_forward_kinematics_np(
        measured_kitchen.root, sofa
    )
    np.testing.assert_allclose(
        root_T_sofa[:3, 3], (3.70718969, 1.23349163, 0.34), atol=1e-7
    )
    assert np.arctan2(root_T_sofa[1, 0], root_T_sofa[0, 0]) == pytest.approx(
        4.719151098618833 - 2 * np.pi
    )


def test_sofa_envelope_matches_observed_cushions(measured_kitchen: World) -> None:
    """
    The refined envelope includes the cushion overhang and taller backrest.
    """
    sofa = measured_kitchen.get_body_by_name("sofa")
    np.testing.assert_allclose(
        sofa.collision.combined_mesh.extents, (0.97, 1.68, 0.842), atol=1e-6
    )


def test_sofa_seating_space_is_clear_above_cushions(measured_kitchen: World) -> None:
    """
    The seating cavity is free while the sloping back remains solid.
    """
    sofa = measured_kitchen.get_body_by_name("sofa")
    points = np.array(
        [
            [-0.2, 0.3, 0.60 - 0.34],
            [0.22, -0.3, 0.75 - 0.34],
            [-0.2, -0.77, 0.60 - 0.34],
        ]
    )
    np.testing.assert_array_equal(
        sofa.collision.combined_mesh.contains(points), [False, True, True]
    )


def test_sofa_support_region_tracks_the_seat_instead_of_the_back(
    measured_kitchen: World,
) -> None:
    """
    Semantic support occupies the gently inclined seating area.
    """
    sofa = measured_kitchen.get_semantic_annotation_by_name("sofa")
    surface = sofa.supporting_surface
    pose = measured_kitchen.compute_forward_kinematics_np(
        measured_kitchen.root, surface
    )
    assert pose[2, 3] == pytest.approx(0.435)
    assert surface.parent_connection.parent is sofa.root
    np.testing.assert_allclose(surface.area.combined_mesh.extents, (0.40, 1.30, 0.005))


def test_sofa_back_does_not_intersect_island_worktop(measured_kitchen: World) -> None:
    """
    The taller backrest remains clear of the island's rear overhang.
    """
    sofa = measured_kitchen.get_body_by_name("sofa")
    worktop = measured_kitchen.get_body_by_name("sideboard")
    meshes = []
    for body in (sofa, worktop):
        mesh = body.collision.combined_mesh.copy()
        mesh.apply_transform(
            measured_kitchen.compute_forward_kinematics_np(measured_kitchen.root, body)
        )
        meshes.append(mesh)
    assert meshes[0].intersection(meshes[1], engine="manifold").volume == pytest.approx(
        0, abs=1e-9
    )


def test_sofa_stays_behind_the_island_corpus(measured_kitchen: World) -> None:
    """
    The fitted sofa box does not penetrate the island's rear supporting plane.
    """
    cabinet = measured_kitchen.get_body_by_name("sideboard_cabinet")
    root_T_cabinet = measured_kitchen.compute_forward_kinematics_np(
        measured_kitchen.root, cabinet
    )
    normal = root_T_cabinet[:3, 0]
    rear_coordinate = (
        normal @ root_T_cabinet[:3, 3] + cabinet.collision.combined_mesh.bounds[1, 0]
    )
    sofa = measured_kitchen.get_body_by_name("sofa")
    root_T_sofa = measured_kitchen.compute_forward_kinematics_np(
        measured_kitchen.root, sofa
    )
    vertices = (
        sofa.collision.combined_mesh.vertices @ root_T_sofa[:3, :3].T
        + root_T_sofa[:3, 3]
    )
    assert (vertices @ normal).min() >= rear_coordinate - 1e-6
