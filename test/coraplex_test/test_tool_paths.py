import numpy as np
import pytest

from coraplex.datastructures.enums import (
    CuttingTechnique,
    MixingPattern,
    SlicingPriority,
    ToolPathSegmentKind,
    WipingTechnique,
)
from coraplex.robot_plans.actions.composite.tool_paths import (
    SliceAnchorPlacement,
    ToolPath,
    ToolPathSegment,
    _constrain_to_convex_hull_xy,
    _local_bounding_box,
    build_container_path,
    build_cutting_path,
    build_surface_path,
    clamp_to_cylinder_xy,
    ramp,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

BOX_SIZE = (0.2, 0.2, 0.1)


@pytest.fixture
def box_body():
    world = World()
    shape_collection = ShapeCollection([Box(scale=Scale(*BOX_SIZE))])
    body = Body(
        name=PrefixedName("tool_path_test_box"),
        collision=shape_collection,
        visual=shape_collection,
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
    return body


def _sample(path):
    return path.sample(frame=HomogeneousTransformationMatrix())


def test_local_bounding_box_returns_box_extents(box_body):
    bounding_box = _local_bounding_box(box_body)
    assert bounding_box.min_x == pytest.approx(-0.1)
    assert bounding_box.min_y == pytest.approx(-0.1)
    assert bounding_box.min_z == pytest.approx(-0.05)
    assert bounding_box.max_x == pytest.approx(0.1)
    assert bounding_box.max_y == pytest.approx(0.1)
    assert bounding_box.max_z == pytest.approx(0.05)


def test_ramp_is_linear_and_saturates():
    assert ramp(0.0, tau_end=0.5, d_max=2.0) == 0.0
    assert ramp(0.25, tau_end=0.5, d_max=2.0) == pytest.approx(1.0)
    assert ramp(0.5, tau_end=0.5, d_max=2.0) == pytest.approx(2.0)
    assert ramp(0.9, tau_end=0.5, d_max=2.0) == pytest.approx(2.0)


def test_container_path_spiral_stays_inside_cylinder(box_body):
    path = build_container_path(box_body, pattern=MixingPattern.SPIRAL)
    _, points, _ = _sample(path)

    radius_xy = 0.5 * min(BOX_SIZE[0], BOX_SIZE[1])
    radii = np.linalg.norm(points[:, :2], axis=1)
    assert np.all(radii <= radius_xy + 1e-9)
    assert np.all(points[:, 2] >= -0.05)
    assert np.all(points[:, 2] <= 0.05)
    assert [segment.kind for segment in path.segments] == [ToolPathSegmentKind.SPIRAL]


def test_container_path_stir_matches_requested_duration(box_body):
    mix_duration = 5.0
    path = build_container_path(
        box_body, pattern=MixingPattern.STIR, mix_duration=mix_duration
    )
    assert path.duration == pytest.approx(mix_duration)
    assert [segment.kind for segment in path.segments] == [ToolPathSegmentKind.STIR]


def test_cutting_path_descends_to_cut_depth(box_body):
    path = build_cutting_path(box_body, technique=CuttingTechnique.SLICE)
    times, points, segment_ids = _sample(path)

    size_z = BOX_SIZE[2]
    expected_cut_depth = -0.05 + max(0.015, 0.20 * size_z)
    assert points[:, 2].min() == pytest.approx(expected_cut_depth)
    assert segment_ids.max() == 2


def test_cutting_path_has_three_segments_per_cut(box_body):
    path = build_cutting_path(
        box_body, technique=CuttingTechnique.SAW, number_of_cuts_on_local_x_axis=2
    )
    assert len(path.segments) == 6
    assert [segment.kind for segment in path.segments] == [
        ToolPathSegmentKind.APPROACH,
        ToolPathSegmentKind.SAW,
        ToolPathSegmentKind.RETRACT,
        ToolPathSegmentKind.APPROACH,
        ToolPathSegmentKind.SAW,
        ToolPathSegmentKind.RETRACT,
    ]
    assert [segment.cut_index for segment in path.segments] == [0, 0, 0, 1, 1, 1]


def test_cutting_path_halving_cuts_center_at_half_height(box_body):
    path = build_cutting_path(box_body, technique=CuttingTechnique.HALVING)
    _, points, _ = _sample(path)

    assert [segment.kind for segment in path.segments] == [
        ToolPathSegmentKind.APPROACH,
        ToolPathSegmentKind.DESCEND,
        ToolPathSegmentKind.RETRACT,
    ]
    np.testing.assert_allclose(np.unique(points[:, 0]), [0.0], atol=1e-9)
    assert points[:, 2].min() == pytest.approx(0.0)


def test_slice_anchors_thickness_only_derives_cut_count():
    placement = SliceAnchorPlacement(
        interval_start=0.0, interval_end=0.18, slice_thickness=0.05
    )
    np.testing.assert_allclose(placement.compute_anchor_positions(), [0.05, 0.10, 0.15])


def test_slice_anchors_cut_count_only_fills_interval():
    placement = SliceAnchorPlacement(
        interval_start=0.0, interval_end=0.18, number_of_cuts=4
    )
    np.testing.assert_allclose(
        placement.compute_anchor_positions(), [0.045, 0.09, 0.135, 0.18]
    )


def test_slice_anchors_use_both_parameters_when_they_fit():
    placement = SliceAnchorPlacement(
        interval_start=0.0,
        interval_end=0.18,
        slice_thickness=0.05,
        number_of_cuts=1,
    )
    np.testing.assert_allclose(placement.compute_anchor_positions(), [0.05])


def test_slice_anchors_thickness_priority_reduces_cut_count():
    placement = SliceAnchorPlacement(
        interval_start=0.0,
        interval_end=0.18,
        slice_thickness=0.05,
        number_of_cuts=20,
        priority=SlicingPriority.THICKNESS,
    )
    np.testing.assert_allclose(placement.compute_anchor_positions(), [0.05, 0.10, 0.15])


def test_slice_anchors_cut_count_priority_shrinks_thickness():
    placement = SliceAnchorPlacement(
        interval_start=0.0,
        interval_end=0.18,
        slice_thickness=5.0,
        number_of_cuts=20,
        priority=SlicingPriority.CUT_COUNT,
    )
    anchor_positions = placement.compute_anchor_positions()
    assert len(anchor_positions) == 20
    np.testing.assert_allclose(np.diff(anchor_positions), 0.009)
    assert anchor_positions[-1] == pytest.approx(0.18)


def test_slice_anchors_oversized_thickness_clamps_to_single_cut():
    placement = SliceAnchorPlacement(
        interval_start=0.0,
        interval_end=0.18,
        slice_thickness=5.0,
        priority=SlicingPriority.THICKNESS,
    )
    np.testing.assert_allclose(placement.compute_anchor_positions(), [0.18])


def test_cutting_path_spaces_cuts_by_slice_thickness(box_body):
    path = build_cutting_path(
        box_body,
        technique=CuttingTechnique.SLICE,
        slice_thickness=0.05,
        number_of_cuts_on_local_x_axis=2,
    )
    _, points, _ = _sample(path)

    # usable X interval starts at -0.1 + margin of 0.01
    np.testing.assert_allclose(np.unique(points[:, 0]), [-0.04, 0.01], atol=1e-9)


def test_surface_path_stays_inside_xy_bounds(box_body):
    expected_kinds = {
        WipingTechnique.WIPE: ToolPathSegmentKind.SPIRAL,
        WipingTechnique.SHEAR: ToolPathSegmentKind.SHEAR,
        WipingTechnique.SPREAD: ToolPathSegmentKind.RASTER,
    }
    for technique, expected_kind in expected_kinds.items():
        path = build_surface_path(box_body, technique=technique)
        _, points, _ = _sample(path)
        assert np.all(points[:, 0] >= -0.1 - 1e-9)
        assert np.all(points[:, 0] <= 0.1 + 1e-9)
        assert np.all(points[:, 1] >= -0.1 - 1e-9)
        assert np.all(points[:, 1] <= 0.1 + 1e-9)
        assert [segment.kind for segment in path.segments] == [expected_kind]


def test_clamp_to_cylinder_xy_clamps_radius_and_height():
    clamped = clamp_to_cylinder_xy([1.0, 0.0, 1.0], radius=0.5, z_min=0.0, z_max=0.4)
    np.testing.assert_allclose(clamped, [0.5, 0.0, 0.4])

    inside = clamp_to_cylinder_xy([0.1, 0.1, 0.2], radius=0.5, z_min=0.0, z_max=0.4)
    np.testing.assert_allclose(inside, [0.1, 0.1, 0.2])


def test_constrain_to_convex_hull_moves_outside_points_onto_hull():
    unit_square_hull = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float
    )
    outside_point = np.array([2.0, 0.5, 0.3])
    constrained = _constrain_to_convex_hull_xy(outside_point, unit_square_hull)
    np.testing.assert_allclose(constrained, [1.0, 0.5, 0.3])

    inside_point = np.array([0.5, 0.5, 0.3])
    unchanged = _constrain_to_convex_hull_xy(inside_point, unit_square_hull)
    np.testing.assert_allclose(unchanged, inside_point)


def test_tool_path_sample_stitches_segments_without_duplicates():
    first = ToolPathSegment(
        kind=ToolPathSegmentKind.SWEEP,
        duration=1.0,
        local_curve=lambda tau: np.array([tau, 0.0, 0.0]),
    )
    second = ToolPathSegment(
        kind=ToolPathSegmentKind.SPIRAL,
        duration=1.0,
        local_curve=lambda tau: np.array([1.0, tau, 0.0]),
    )
    path = ToolPath([first, second])

    times, points, segment_ids = _sample(path)

    assert np.all(np.diff(times) > 0)
    assert set(np.unique(segment_ids)) == {0, 1}
    assert path.duration == pytest.approx(2.0)
    assert times[-1] == pytest.approx(2.0)
