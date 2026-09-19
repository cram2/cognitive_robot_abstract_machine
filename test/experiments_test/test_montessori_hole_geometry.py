import numpy as np
import pytest

from experiments.montessori.hole_geometry import (
    PlanarPoint,
    PlanarSize,
    PolygonMeasurement,
    cut_board_mesh,
    detect_hole_footprints,
)
from experiments.montessori.semantics import MontessoriShapeCategory
from semantic_digital_twin.world_description.geometry import Scale


def test_detect_hole_footprints_finds_one_footprint_per_hole_category():
    footprints = detect_hole_footprints()

    categories = [footprint.category for footprint in footprints]
    assert sorted(categories, key=str) == sorted(
        [
            MontessoriShapeCategory.CUBE,
            MontessoriShapeCategory.CYLINDER,
            MontessoriShapeCategory.CYLINDER,
            MontessoriShapeCategory.TRIANGULAR_PRISM,
            MontessoriShapeCategory.RECTANGULAR_PRISM,
            MontessoriShapeCategory.DISK,
        ],
        key=str,
    )


def test_detect_hole_footprints_places_each_footprint_within_the_board_bounds():
    footprints = detect_hole_footprints()

    for footprint in footprints:
        # the board's lid measures roughly 0.11 x 0.282 meters
        assert abs(footprint.center.x) < 0.055
        assert abs(footprint.center.y) < 0.141
        assert 0 < footprint.size.x < 0.11
        assert 0 < footprint.size.y < 0.282


def test_detect_hole_footprints_is_ordered_by_ascending_y_position():
    footprints = detect_hole_footprints()

    y_positions = [footprint.center.y for footprint in footprints]
    assert y_positions == sorted(y_positions)


def test_hole_footprint_boundary_matches_its_bounding_box_size():
    footprints = detect_hole_footprints()

    for footprint in footprints:
        boundary_x = [point.x for point in footprint.boundary]
        boundary_y = [point.y for point in footprint.boundary]
        assert max(boundary_x) - min(boundary_x) == pytest.approx(
            footprint.size.x, abs=1e-6
        )
        assert max(boundary_y) - min(boundary_y) == pytest.approx(
            footprint.size.y, abs=1e-6
        )


def test_circular_hole_footprint_extrudes_to_a_round_not_rectangular_solid():
    footprints = detect_hole_footprints()
    circular_footprints = [
        footprint
        for footprint in footprints
        if footprint.category == MontessoriShapeCategory.CYLINDER
    ]
    circular_footprint = circular_footprints[0]

    solid = circular_footprint.extrude(0.01)

    assert solid.is_watertight
    bounding_box_volume = circular_footprint.size.x * circular_footprint.size.y * 0.01
    # a circle inscribed in its bounding square fills only pi/4 of its area; a
    # rectangular cut would fill all of it
    assert solid.volume == pytest.approx(bounding_box_volume * 3.14159 / 4, rel=0.05)


def test_cut_board_mesh_cuts_a_watertight_board_with_true_hole_shapes():
    footprints = detect_hole_footprints()
    board_scale = Scale(0.13, 0.30, 0.08)

    board_mesh = cut_board_mesh(board_scale, footprints)

    assert board_mesh.is_watertight
    uncut_volume = board_scale.x * board_scale.y * board_scale.z
    assert 0 < board_mesh.volume < uncut_volume

    # slicing through the cut board must reveal a round cross-section (many
    # vertices) for the circular holes, not a 4-8 vertex rectangular notch
    section = board_mesh.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
    loop_lengths = sorted(len(loop) for loop in section.discrete)
    assert loop_lengths[-1] > 20


def test_hole_footprint_names_the_parts_of_its_position_and_size():
    [footprint, *_] = detect_hole_footprints()

    assert isinstance(footprint.center, PlanarPoint)
    assert isinstance(footprint.size, PlanarSize)
    assert all(isinstance(point, PlanarPoint) for point in footprint.boundary)


def test_polygon_measurement_reads_a_squares_area_and_middle():
    square = np.array([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)])

    measurement = PolygonMeasurement.of(square)

    assert measurement.area == pytest.approx(2.0)
    assert measurement.centroid == PlanarPoint(pytest.approx(1.0), pytest.approx(0.5))


def test_polygon_measurement_balances_a_triangle_at_a_third_of_its_height():
    triangle = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])

    measurement = PolygonMeasurement.of(triangle)

    assert measurement.area == pytest.approx(0.5)
    assert measurement.centroid == PlanarPoint(
        pytest.approx(1 / 3), pytest.approx(1 / 3)
    )
