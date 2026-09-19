import math

import pytest

from experiments.montessori.exceptions import NoMatchingHoleError
from experiments.montessori.semantics import (
    CubeShape,
    CylinderShape,
    DiskShape,
    MontessoriShape,
    MontessoriShapeCategory,
    RectangularPrismShape,
    ShapeSortingBoard,
    ShapeSortingHole,
    SphereShape,
    TriangularPrismShape,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
from semantic_digital_twin.spatial_types.spatial_types import Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def world():
    world = World()
    root = Body(name=PrefixedName("root"))
    with world.modify_world():
        world.add_body(root)
    return world


@pytest.mark.parametrize(
    "shape_class,category",
    [
        (CubeShape, MontessoriShapeCategory.CUBE),
        (CylinderShape, MontessoriShapeCategory.CYLINDER),
        (DiskShape, MontessoriShapeCategory.DISK),
        (SphereShape, MontessoriShapeCategory.SPHERE),
        (TriangularPrismShape, MontessoriShapeCategory.TRIANGULAR_PRISM),
        (RectangularPrismShape, MontessoriShapeCategory.RECTANGULAR_PRISM),
    ],
)
def test_montessori_shape_subclass_reports_its_shape_category(
    world, shape_class, category
):
    body = Body(name=PrefixedName("piece"))
    with world.modify_world():
        world.add_body(body)
        world.add_connection(FixedConnection(parent=world.root, child=body))
        shape = shape_class(name=PrefixedName("piece"), root=body)
        world.add_semantic_annotation(shape)

    assert world.get_semantic_annotations_by_type(MontessoriShape) == [shape]
    assert shape.shape_category == category


def test_montessori_shape_computes_its_insertion_pose_relative_to_the_hole(world):
    body = Body(name=PrefixedName("piece"))
    with world.modify_world():
        world.add_body(body)
        world.add_connection(FixedConnection(parent=world.root, child=body))
        shape = CubeShape(name=PrefixedName("piece"), root=body)
        hole = ShapeSortingHole.create_with_new_region_in_world(
            name=PrefixedName("square_hole"),
            world=world,
            scale=Scale(0.03, 0.03, 0.001),
        )
        hole.shape_category = MontessoriShapeCategory.CUBE

    pose = shape.insertion_pose_relative_to_hole(
        hole, horizontal_offset=Point3(0.01, -0.02, 0.0), hover_height=0.03
    )

    assert pose.reference_frame is hole.root
    position = pose.to_position()
    assert float(position.x) == pytest.approx(0.01)
    assert float(position.y) == pytest.approx(-0.02)
    assert float(position.z) == pytest.approx(0.03)


def test_disk_shape_tips_onto_its_edge_to_pass_through_its_slot(world):
    """
    A disk's matching hole is a narrow slot, not a coin-shaped opening, so unlike
    every other shape it must be rotated onto its edge to fit through (see
    experiments.montessori.hole_geometry._classify_hole_shape).
    """
    body = Body(name=PrefixedName("piece"))
    with world.modify_world():
        world.add_body(body)
        world.add_connection(FixedConnection(parent=world.root, child=body))
        shape = DiskShape(name=PrefixedName("piece"), root=body)
        hole = ShapeSortingHole.create_with_new_region_in_world(
            name=PrefixedName("disk_hole"),
            world=world,
            scale=Scale(0.005, 0.048, 0.001),
        )
        hole.shape_category = MontessoriShapeCategory.DISK

    pose = shape.insertion_pose_relative_to_hole(
        hole, horizontal_offset=Point3(0.0, 0.0, 0.0), hover_height=0.03
    )

    roll, pitch, yaw = pose.to_rotation_matrix().to_rpy()
    assert float(pitch) == pytest.approx(math.pi / 2)
    assert float(roll) == pytest.approx(0.0)
    assert float(yaw) == pytest.approx(0.0)


def _shape_with_cross_section(
    world: World, name: str, shape_class, size: float
) -> MontessoriShape:
    """
    A :class:`MontessoriShape` of the given class whose collision geometry is a flat
    square box, so :attr:`MontessoriShape.cross_section_size` equals ``size``.
    """
    body = Body.from_shape_collection(
        PrefixedName(name), ShapeCollection([Box(scale=Scale(size, size, 0.03))])
    )
    world.add_body(body)
    world.add_connection(FixedConnection(parent=world.root, child=body))
    return shape_class(name=PrefixedName(name), root=body)


def test_fits_through_rejects_a_shape_too_large_for_an_otherwise_matching_hole(world):
    """
    The board has two circular holes of different sizes, both categorized
    MontessoriShapeCategory.CYLINDER; matching category alone is not enough to tell
    them apart, so fits_through must also check size.
    """
    with world.modify_world():
        small_hole = ShapeSortingHole.create_with_new_region_in_world(
            name=PrefixedName("small_circular_hole"),
            world=world,
            scale=Scale(0.032, 0.032, 0.001),
        )
        small_hole.shape_category = MontessoriShapeCategory.CYLINDER

        large_hole = ShapeSortingHole.create_with_new_region_in_world(
            name=PrefixedName("large_circular_hole"),
            world=world,
            scale=Scale(0.04, 0.04, 0.001),
        )
        large_hole.shape_category = MontessoriShapeCategory.CYLINDER

        small_shape = _shape_with_cross_section(
            world, "small_cylinder", CylinderShape, 0.0272
        )
        large_shape = _shape_with_cross_section(
            world, "large_cylinder", CylinderShape, 0.034
        )

    assert small_shape.fits_through(small_hole)
    assert small_shape.fits_through(large_hole)
    assert not large_shape.fits_through(small_hole)
    assert large_shape.fits_through(large_hole)


def test_hole_for_returns_the_smallest_hole_a_shape_actually_fits_through(world):
    with world.modify_world():
        board = ShapeSortingBoard.create_with_new_body_in_world(
            name=PrefixedName("board"), world=world, scale=Scale(0.3, 0.3, 0.1)
        )
        small_hole = ShapeSortingHole.create_with_new_region_in_world(
            name=PrefixedName("small_circular_hole"),
            world=world,
            scale=Scale(0.032, 0.032, 0.001),
        )
        small_hole.shape_category = MontessoriShapeCategory.CYLINDER
        large_hole = ShapeSortingHole.create_with_new_region_in_world(
            name=PrefixedName("large_circular_hole"),
            world=world,
            scale=Scale(0.04, 0.04, 0.001),
        )
        large_hole.shape_category = MontessoriShapeCategory.CYLINDER
        board.add(small_hole)
        board.add(large_hole)

        small_shape = _shape_with_cross_section(
            world, "small_cylinder", CylinderShape, 0.0272
        )
        large_shape = _shape_with_cross_section(
            world, "large_cylinder", CylinderShape, 0.034
        )

    assert board.hole_for(small_shape) is small_hole
    assert board.hole_for(large_shape) is large_hole


def test_hole_for_raises_when_the_board_has_no_hole_of_the_shapes_category(world):
    with world.modify_world():
        board = ShapeSortingBoard.create_with_new_body_in_world(
            name="board", world=world, scale=Scale(0.3, 0.3, 0.1)
        )
        hole = ShapeSortingHole.create_with_new_region_in_world(
            name="circular_hole",
            world=world,
            scale=Scale(0.04, 0.04, 0.001),
        )
        hole.shape_category = MontessoriShapeCategory.CYLINDER
        board.add(hole)

        cube = _shape_with_cross_section(world, "cube", CubeShape, 0.03)

    with pytest.raises(NoMatchingHoleError):
        board.hole_for(cube)


def test_shape_sorting_hole_stores_its_shape_category(world):
    with world.modify_world():
        hole = ShapeSortingHole.create_with_new_region_in_world(
            name=PrefixedName("square_hole"),
            world=world,
            scale=Scale(0.03, 0.03, 0.001),
        )
        hole.shape_category = MontessoriShapeCategory.CUBE

    assert hole.shape_category == MontessoriShapeCategory.CUBE


def test_shape_sorting_board_collects_drawers_and_shape_holes(world):
    with world.modify_world():
        board = ShapeSortingBoard.create_with_new_body_in_world(
            name=PrefixedName("board"), world=world, scale=Scale(0.3, 0.3, 0.1)
        )
        drawer = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer"), world=world
        )
        hole = ShapeSortingHole.create_with_new_region_in_world(
            name=PrefixedName("triangle_hole"),
            world=world,
            scale=Scale(0.03, 0.03, 0.001),
        )
        hole.shape_category = MontessoriShapeCategory.TRIANGULAR_PRISM
        board.add(drawer)
        board.add(hole)

    assert board.drawers == [drawer]
    assert board.apertures == [hole]
