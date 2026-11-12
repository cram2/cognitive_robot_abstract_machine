"""
Tests for the shape-sorting toy's annotations: its pieces, the holes through its board,
and which hole a piece is sorted into.
"""

from __future__ import annotations

import math

import pytest

from krrood_spatial_math.spatial_types import Point3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import NoMatchingHoleError
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CubePiece,
    CylinderPiece,
    DiskPiece,
    Drawer,
    RectangularPrismPiece,
    ShapeSortingBoard,
    ShapeSortingHole,
    ShapeSortingPiece,
    SpherePiece,
    TriangularPrismPiece,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% building pieces and holes


@pytest.fixture
def world() -> World:
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("root")))
    return world


def _piece(
    world: World,
    *,
    name: str,
    kind: type[ShapeSortingPiece],
    scale: Scale = Scale(0.03, 0.03, 0.03),
) -> ShapeSortingPiece:
    """
    A piece of the given kind whose collision geometry is a box of the given size, added
    to a world already open for modification.
    """
    body = Body.from_shape_collection(
        PrefixedName(name), ShapeCollection([Box(scale=scale)])
    )
    world.add_connection(FixedConnection(parent=world.root, child=body))
    piece = kind(name=PrefixedName(name), root=body, grasp_height=0.01)
    world.add_semantic_annotation(piece)
    return piece


def _hole(
    world: World, *, name: str, kind: type[ShapeSortingPiece], scale: Scale
) -> ShapeSortingHole:
    """
    A hole cut for the given kind of piece, its region a box of the given size, added to
    a world already open for modification.
    """
    hole = ShapeSortingHole.create_with_new_region_in_world(
        name=PrefixedName(name), world=world, scale=scale
    )
    hole.piece_kind = kind
    return hole


def _board(world: World) -> ShapeSortingBoard:
    return ShapeSortingBoard.create_with_new_body_in_world(
        name=PrefixedName("board"), world=world, scale=Scale(0.3, 0.3, 0.1)
    )


# %% kinds of piece


@pytest.mark.parametrize(
    "kind",
    [
        CubePiece,
        CylinderPiece,
        DiskPiece,
        SpherePiece,
        TriangularPrismPiece,
        RectangularPrismPiece,
    ],
)
def test_each_kind_of_piece_is_found_as_a_shape_sorting_piece(world, kind):
    with world.modify_world():
        piece = _piece(world, name="piece", kind=kind)

    assert world.get_semantic_annotations_by_type(ShapeSortingPiece) == [piece]


def test_a_piece_reads_its_footprint_and_height_off_its_body(world):
    with world.modify_world():
        piece = _piece(
            world,
            name="prism",
            kind=RectangularPrismPiece,
            scale=Scale(0.02, 0.04, 0.03),
        )

    assert piece.footprint.bounding_box.dimensions == pytest.approx([0.02, 0.04])
    assert piece.height == pytest.approx(0.03)


# %% inserting a piece


def test_a_piece_hovers_over_its_hole_at_the_given_offset(world):
    with world.modify_world():
        piece = _piece(world, name="cube", kind=CubePiece)
        hole = _hole(
            world, name="square_hole", kind=CubePiece, scale=Scale(0.03, 0.03, 0.001)
        )

    pose = piece.insertion_pose_relative_to_hole(
        hole=hole, horizontal_offset=Point3(0.01, -0.02, 0.0), hover_height=0.03
    )

    assert pose.reference_frame is hole.root
    assert pose.position.to_np()[:3] == pytest.approx([0.01, -0.02, 0.03])


@pytest.mark.parametrize(
    "hole_scale, expected_yaw",
    [(Scale(0.042, 0.022, 0.001), math.pi / 2), (Scale(0.022, 0.042, 0.001), 0.0)],
)
def test_a_piece_is_turned_onto_its_hole_when_inserted(world, hole_scale, expected_yaw):
    with world.modify_world():
        piece = _piece(
            world,
            name="prism",
            kind=RectangularPrismPiece,
            scale=Scale(0.02, 0.04, 0.03),
        )
        hole = _hole(world, name="hole", kind=RectangularPrismPiece, scale=hole_scale)

    pose = piece.insertion_pose_relative_to_hole(
        hole=hole, horizontal_offset=Point3(0.0, 0.0, 0.0), hover_height=0.03
    )

    _, _, yaw = pose.rotation_matrix.rpy
    assert float(yaw) % math.pi == pytest.approx(
        expected_yaw, abs=piece.footprint.alignment_angle_step
    )


def test_a_disk_tips_onto_its_edge_to_pass_through_its_slot(world):
    with world.modify_world():
        disk = _piece(world, name="disk", kind=DiskPiece)
        slot = _hole(
            world, name="slot", kind=DiskPiece, scale=Scale(0.005, 0.048, 0.001)
        )

    pose = disk.insertion_pose_relative_to_hole(
        hole=slot, horizontal_offset=Point3(0.0, 0.0, 0.0), hover_height=0.03
    )

    assert [float(angle) for angle in pose.rotation_matrix.rpy] == pytest.approx(
        [0.0, math.pi / 2, 0.0]
    )


# %% which hole a piece goes through


def test_a_piece_too_large_for_a_hole_of_its_kind_does_not_fit_through_it(world):
    with world.modify_world():
        small_hole = _hole(
            world, name="small", kind=CylinderPiece, scale=Scale(0.032, 0.032, 0.001)
        )
        large_hole = _hole(
            world, name="large", kind=CylinderPiece, scale=Scale(0.04, 0.04, 0.001)
        )
        large_piece = _piece(
            world, name="cylinder", kind=CylinderPiece, scale=Scale(0.034, 0.034, 0.03)
        )

    assert not large_piece.fits_through(hole=small_hole)
    assert large_piece.fits_through(hole=large_hole)


def test_a_piece_does_not_fit_through_a_hole_cut_for_another_kind(world):
    with world.modify_world():
        hole = _hole(
            world, name="round", kind=CylinderPiece, scale=Scale(0.04, 0.04, 0.001)
        )
        cube = _piece(world, name="cube", kind=CubePiece)

    assert not cube.fits_through(hole=hole)


def test_hole_for_returns_the_smallest_hole_a_piece_fits_through(world):
    with world.modify_world():
        board = _board(world)
        small_hole = _hole(
            world, name="small", kind=CylinderPiece, scale=Scale(0.032, 0.032, 0.001)
        )
        large_hole = _hole(
            world, name="large", kind=CylinderPiece, scale=Scale(0.04, 0.04, 0.001)
        )
        board.add(small_hole)
        board.add(large_hole)
        small_piece = _piece(
            world,
            name="small_piece",
            kind=CylinderPiece,
            scale=Scale(0.0272, 0.0272, 0.03),
        )
        large_piece = _piece(
            world,
            name="large_piece",
            kind=CylinderPiece,
            scale=Scale(0.034, 0.034, 0.03),
        )

    assert board.hole_for(piece=small_piece) is small_hole
    assert board.hole_for(piece=large_piece) is large_hole


def test_hole_for_raises_when_no_hole_fits_the_piece(world):
    with world.modify_world():
        board = _board(world)
        board.add(
            _hole(
                world, name="round", kind=CylinderPiece, scale=Scale(0.04, 0.04, 0.001)
            )
        )
        cube = _piece(world, name="cube", kind=CubePiece)

    with pytest.raises(NoMatchingHoleError):
        board.hole_for(piece=cube)


# %% the board


def test_a_board_collects_its_drawers_and_holes(world):
    with world.modify_world():
        board = _board(world)
        drawer = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer"), world=world
        )
        hole = _hole(
            world,
            name="triangle",
            kind=TriangularPrismPiece,
            scale=Scale(0.03, 0.03, 0.001),
        )
        board.add(drawer)
        board.add(hole)

    assert board.drawers == [drawer]
    assert board.apertures == [hole]


def test_mounting_a_hole_leaves_the_boards_geometry_uncut(world):
    with world.modify_world():
        board = _board(world)
    volume_before = board.root.collision.combined_mesh.volume

    with world.modify_world():
        board.add(
            _hole(world, name="square", kind=CubePiece, scale=Scale(0.03, 0.03, 0.2))
        )

    assert board.root.collision.combined_mesh.volume == pytest.approx(volume_before)
