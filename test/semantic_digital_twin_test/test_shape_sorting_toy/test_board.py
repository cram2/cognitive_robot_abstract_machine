"""
Tests for reading the shape-sorting board's holes and parts off its mesh.
"""

from __future__ import annotations

import math

import pytest

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CubePiece,
    CylinderPiece,
    DiskPiece,
    RectangularPrismPiece,
    SpherePiece,
    TriangularPrismPiece,
)
from semantic_digital_twin.shape_sorting_toy.board import (
    HoleFootprint,
    HoleShapeClassifier,
    ShapeSortingBoardMesh,
)
from semantic_digital_twin.world_description.geometry import Polygon2D, Scale

# %% the board's mesh


@pytest.fixture(scope="module")
def board_mesh() -> ShapeSortingBoardMesh:
    return ShapeSortingBoardMesh()


def test_the_board_has_one_hole_per_kind_of_piece(board_mesh):
    kinds = [footprint.piece_kind for footprint in board_mesh.hole_footprints]

    assert sorted(kinds, key=lambda kind: kind.__name__) == sorted(
        [
            CubePiece,
            CylinderPiece,
            SpherePiece,
            TriangularPrismPiece,
            RectangularPrismPiece,
            DiskPiece,
        ],
        key=lambda kind: kind.__name__,
    )


def test_every_hole_lies_on_the_lid(board_mesh):
    lid_minimum, lid_maximum = board_mesh.lid.bounds[:, :2]

    for footprint in board_mesh.hole_footprints:
        outline = footprint.outline_on_board.bounding_box
        assert lid_minimum[0] < outline.min_x < outline.max_x < lid_maximum[0]
        assert lid_minimum[1] < outline.min_y < outline.max_y < lid_maximum[1]


def test_holes_are_ordered_from_minus_to_plus_y(board_mesh):
    y_positions = [
        float(footprint.center.y) for footprint in board_mesh.hole_footprints
    ]

    assert y_positions == sorted(y_positions)


def test_a_hole_boundary_is_centred_on_the_hole(board_mesh):
    for footprint in board_mesh.hole_footprints:
        assert footprint.boundary.centroid.to_np() == pytest.approx(
            [0.0, 0.0], abs=1e-9
        )


def test_the_hole_cut_for_a_kind_of_piece_is_found(board_mesh):
    assert (
        board_mesh.matching_hole_footprint(piece_kind=DiskPiece).piece_kind is DiskPiece
    )


def test_a_circular_hole_extrudes_to_a_round_solid(board_mesh):
    footprint = board_mesh.matching_hole_footprint(piece_kind=CylinderPiece)

    solid = footprint.boundary.extrude(thickness=0.01)

    assert solid.is_watertight
    assert solid.volume == pytest.approx(
        footprint.boundary.bounding_box.area * 0.01 * math.pi / 4, rel=0.05
    )


def test_a_hole_marker_is_as_thick_as_the_marker_thickness(board_mesh):
    [footprint, *_] = board_mesh.hole_footprints

    assert footprint.marker().extents[2] == pytest.approx(
        HoleFootprint.marker_thickness
    )


def test_a_cut_blank_has_the_true_hole_shapes_cut_through_it(board_mesh):
    blank_scale = Scale(0.13, 0.30, 0.08)

    board = board_mesh.cut_blank(blank_scale=blank_scale)

    assert board.is_watertight
    assert 0 < board.volume < blank_scale.x * blank_scale.y * blank_scale.z
    section = board.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
    assert max(len(loop) for loop in section.discrete) > (
        board_mesh.hole_classifier.circle_vertex_count
    )


def test_the_lid_sits_on_top_of_the_board(board_mesh):
    assert board_mesh.bottom < board_mesh.lid_bottom < board_mesh.lid_top
    assert board_mesh.height == pytest.approx(board_mesh.lid_top - board_mesh.bottom)


def test_the_board_has_a_compartment_under_each_drawer(board_mesh):
    compartments = board_mesh.compartments()

    assert len(compartments) == 3
    assert all(
        left.upper <= right.lower for left, right in zip(compartments, compartments[1:])
    )


# %% naming the piece a hole is cut for


@pytest.mark.parametrize(
    "outline, expected",
    [
        (Polygon2D.circle(diameter=0.03), CylinderPiece),
        (Polygon2D.circle(diameter=0.04), SpherePiece),
        (Polygon2D.equilateral_triangle(side=0.04), TriangularPrismPiece),
        (Polygon2D.rectangle(width=0.004, length=0.045), DiskPiece),
        (Polygon2D.rectangle(width=0.02, length=0.04), RectangularPrismPiece),
        (Polygon2D.rectangle(width=0.03, length=0.03), CubePiece),
    ],
)
def test_the_classifier_names_the_piece_an_outline_is_cut_for(outline, expected):
    assert HoleShapeClassifier().classify(outline=outline) is expected
