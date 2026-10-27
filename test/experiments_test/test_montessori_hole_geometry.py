import pytest

from experiments.montessori.hole_geometry import (
    HoleShapeClassifier,
    ShapeSortingBoardMesh,
)
from experiments.montessori.semantics import (
    CubeShape,
    CylinderShape,
    DiskShape,
    RectangularPrismShape,
    SphereShape,
    TriangularPrismShape,
)
from semantic_digital_twin.spatial_types.spatial_types import Point2
from semantic_digital_twin.world_description.geometry import (
    PlanarBoundingBox,
    Polygon2D,
    Scale,
)


def test_hole_footprints_has_one_footprint_per_hole():
    footprints = ShapeSortingBoardMesh().hole_footprints

    categories = [footprint.category for footprint in footprints]
    assert sorted(categories, key=lambda category: category.__name__) == sorted(
        [
            CubeShape,
            CylinderShape,
            SphereShape,
            TriangularPrismShape,
            RectangularPrismShape,
            DiskShape,
        ],
        key=lambda category: category.__name__,
    )


def test_hole_footprints_lie_within_the_board_bounds():
    footprints = ShapeSortingBoardMesh().hole_footprints

    for footprint in footprints:
        # the board's lid measures roughly 0.11 x 0.282 meters
        center_x, center_y = footprint.center.to_np()
        bounding_box = footprint.boundary.bounding_box
        assert abs(center_x) < 0.055
        assert abs(center_y) < 0.141
        assert 0 < bounding_box.depth < 0.11
        assert 0 < bounding_box.width < 0.282


def test_hole_footprints_are_ordered_by_ascending_y_position():
    footprints = ShapeSortingBoardMesh().hole_footprints

    y_positions = [float(footprint.center.y) for footprint in footprints]
    assert y_positions == sorted(y_positions)


def test_circular_hole_footprint_extrudes_to_a_round_not_rectangular_solid():
    footprints = ShapeSortingBoardMesh().hole_footprints
    circular_footprints = [
        footprint for footprint in footprints if footprint.category == CylinderShape
    ]
    circular_footprint = circular_footprints[0]

    solid = circular_footprint.extrude(0.01)

    assert solid.is_watertight
    bounding_box_volume = circular_footprint.boundary.bounding_box.area * 0.01
    # a circle inscribed in its bounding square fills only pi/4 of its area; a
    # rectangular cut would fill all of it
    assert solid.volume == pytest.approx(bounding_box_volume * 3.14159 / 4, rel=0.05)


def test_cut_blank_cuts_a_watertight_board_with_true_hole_shapes():
    board_scale = Scale(0.13, 0.30, 0.08)

    board_mesh = ShapeSortingBoardMesh().cut_blank(board_scale)

    assert board_mesh.is_watertight
    uncut_volume = board_scale.x * board_scale.y * board_scale.z
    assert 0 < board_mesh.volume < uncut_volume

    # slicing through the cut board must reveal a round cross-section (many
    # vertices) for the circular holes, not a 4-8 vertex rectangular notch
    section = board_mesh.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
    loop_lengths = sorted(len(loop) for loop in section.discrete)
    assert loop_lengths[-1] > 20


def test_hole_footprint_names_the_parts_of_its_position_and_outline():
    [footprint, *_] = ShapeSortingBoardMesh().hole_footprints

    assert isinstance(footprint.center, Point2)
    assert isinstance(footprint.boundary, Polygon2D)
    assert isinstance(footprint.boundary.bounding_box, PlanarBoundingBox)


def test_hole_footprint_boundary_is_centred_on_the_hole():
    for footprint in ShapeSortingBoardMesh().hole_footprints:
        assert footprint.boundary.centroid.to_np() == pytest.approx(
            [0.0, 0.0], abs=1e-9
        )


def test_hole_marker_is_as_thick_as_the_marker_thickness():
    [footprint, *_] = ShapeSortingBoardMesh().hole_footprints

    assert footprint.marker().extents[2] == pytest.approx(footprint.MARKER_THICKNESS)


# %% naming the piece a hole is cut for


@pytest.mark.parametrize(
    "outline, expected",
    [
        (Polygon2D.circle(0.03), CylinderShape),
        (Polygon2D.circle(0.04), SphereShape),
        (Polygon2D.equilateral_triangle(0.04), TriangularPrismShape),
        (Polygon2D.rectangle(0.004, 0.045), DiskShape),
        (Polygon2D.rectangle(0.02, 0.04), RectangularPrismShape),
        (Polygon2D.rectangle(0.03, 0.03), CubeShape),
    ],
)
def test_hole_shape_classifier_names_the_piece_an_outline_is_cut_for(outline, expected):
    assert HoleShapeClassifier().classify(outline) is expected
