"""
Tests for reading the shape-sorting board's holes and parts off its mesh.
"""

from __future__ import annotations

import math

import pytest

from semantic_digital_twin.shape_sorting_toy.semantic_annotations import (
    CubePiece,
    CylinderPiece,
    DiskPiece,
    RectangularPrismPiece,
    SpherePiece,
    TriangularPrismPiece,
)
from krrood.patterns.role import Role
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.shape_sorting_toy.board import (
    BoardResource,
    HoleShapeClassifier,
    ShapeSortingBoardMesh,
)
from semantic_digital_twin.shape_sorting_toy.semantic_annotations import (
    ShapeSortingBoard,
    ShapeSortingHole,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.semantic_annotations import (
    PartSpecificationBinding,
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import (
    Box,
    Mesh,
    Polygon2D,
    Scale,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% the board's mesh


@pytest.fixture(scope="module")
def board_mesh() -> ShapeSortingBoardMesh:
    return ShapeSortingBoardMesh(role_taker=Mesh.from_file(str(BoardResource.MESH)))


@pytest.fixture(scope="module")
def board(board_mesh) -> ShapeSortingBoard:
    """
    A board standing at the origin of a world of its own, with the holes read off its
    mesh.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("world_root")))
    return SemanticAnnotationWithRootSpecification(
        name="board",
        semantic_annotation_type=ShapeSortingBoard,
        root_specification=BodySpecification(
            name="board",
            shapes=ShapeCollection([Box(scale=Scale(0.1, 0.3, 0.01))]),
        ),
        part_bindings=[
            PartSpecificationBinding(
                field_name="apertures",
                specifications=board_mesh.hole_specifications(),
            )
        ],
    ).spawn(world)


def test_the_board_mesh_is_a_role_of_its_mesh(board_mesh):
    assert Role.has_role(
        role_taker=board_mesh.role_taker, role_types=ShapeSortingBoardMesh
    )


def test_the_board_has_one_hole_per_kind_of_piece(board):
    kinds = [hole.piece_kind for hole in board.apertures]

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


def test_every_hole_is_a_shape_sorting_hole(board):
    assert all(isinstance(hole, ShapeSortingHole) for hole in board.apertures)


def test_every_hole_lies_on_the_lid(board_mesh):
    lid_minimum, lid_maximum = board_mesh.lid.bounds[:, :2]

    for outline in board_mesh.hole_outlines:
        bounding_box = outline.bounding_box
        assert lid_minimum[0] < bounding_box.min_x < bounding_box.max_x < lid_maximum[0]
        assert lid_minimum[1] < bounding_box.min_y < bounding_box.max_y < lid_maximum[1]


def test_holes_are_ordered_from_minus_to_plus_y(board_mesh):
    y_positions = [float(outline.centroid.y) for outline in board_mesh.hole_outlines]

    assert y_positions == sorted(y_positions)


def test_a_hole_stands_over_its_outlines_centroid_on_the_lids_top(board_mesh):
    for outline, specification in zip(
        board_mesh.hole_outlines, board_mesh.hole_specifications()
    ):
        position = specification.root_specification.parent_T_self.position
        assert position.to_np()[:3] == pytest.approx(
            [float(outline.centroid.x), float(outline.centroid.y), board_mesh.lid_top]
        )


def test_a_hole_has_the_outline_it_was_read_with(board_mesh, board):
    for outline, hole in zip(board_mesh.hole_outlines, board.apertures):
        assert hole.footprint.bounding_box.dimensions == pytest.approx(
            outline.bounding_box.dimensions, abs=1e-6
        )


def test_a_hole_is_as_thick_as_the_marker_thickness(board_mesh):
    [specification, *_] = board_mesh.hole_specifications()
    [shape] = specification.root_specification.shapes.shapes

    assert shape.mesh.extents[2] == pytest.approx(board_mesh.hole_marker_thickness)


def test_a_circular_hole_extrudes_to_a_round_solid(board_mesh):
    outline = next(
        outline
        for outline in board_mesh.hole_outlines
        if board_mesh.hole_classifier.classify(outline=outline) is CylinderPiece
    )

    solid = outline.extrude(thickness=0.01)

    assert solid.is_watertight
    assert solid.volume == pytest.approx(
        outline.bounding_box.area * 0.01 * math.pi / 4, rel=0.05
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
