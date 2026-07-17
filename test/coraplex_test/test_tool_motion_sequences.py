import numpy as np
import pytest

from coraplex.datastructures.enums import (
    CuttingTechnique,
    MixingPattern,
    SlicingPriority,
    WipingTechnique,
)
from coraplex.robot_plans.actions.composite.tool_motion_sequences import (
    DEFAULT_SAMPLE_DT,
    MotionSegment,
    MotionSequence,
    SliceAnchorPlacement,
    _constrain_to_convex_hull_xy,
    body_local_aabb,
    build_container_sequence,
    build_cutting_sequence,
    build_surface_sequence,
    clamp_to_cylinder_xy,
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
        name=PrefixedName("tool_motion_test_box"),
        collision=shape_collection,
        visual=shape_collection,
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
    return body


def _sample(sequence):
    return sequence.sample(
        frame=HomogeneousTransformationMatrix(), dt=DEFAULT_SAMPLE_DT
    )


def test_body_local_aabb_returns_box_extents(box_body):
    mins, maxs = body_local_aabb(box_body)
    np.testing.assert_allclose(mins, [-0.1, -0.1, -0.05])
    np.testing.assert_allclose(maxs, [0.1, 0.1, 0.05])


def test_container_sequence_spiral_stays_inside_cylinder(box_body):
    sequence = build_container_sequence(box_body, pattern=MixingPattern.SPIRAL)
    _, points, _ = _sample(sequence)

    radius_xy = 0.5 * min(BOX_SIZE[0], BOX_SIZE[1])
    radii = np.linalg.norm(points[:, :2], axis=1)
    assert np.all(radii <= radius_xy + 1e-9)
    assert np.all(points[:, 2] >= -0.05)
    assert np.all(points[:, 2] <= 0.05)


def test_container_sequence_stir_matches_requested_duration(box_body):
    mix_duration_s = 5.0
    sequence = build_container_sequence(
        box_body, pattern=MixingPattern.STIR, mix_duration_s=mix_duration_s
    )
    assert sequence.duration_s == pytest.approx(mix_duration_s)


def test_cutting_sequence_descends_to_cut_depth(box_body):
    sequence = build_cutting_sequence(box_body, technique=CuttingTechnique.SLICE)
    times, points, phase_ids = _sample(sequence)

    size_z = BOX_SIZE[2]
    expected_cut_depth = -0.05 + max(0.015, 0.20 * size_z)
    assert points[:, 2].min() == pytest.approx(expected_cut_depth)
    assert phase_ids.max() == 2


def test_cutting_sequence_has_three_phases_per_cut(box_body):
    sequence = build_cutting_sequence(
        box_body, technique=CuttingTechnique.SAW, num_cuts_x=2
    )
    assert len(sequence.phases) == 6


def test_slice_anchors_thickness_only_derives_cut_count():
    placement = SliceAnchorPlacement(
        interval_start=0.0, interval_end=0.18, slice_thickness=0.05
    )
    np.testing.assert_allclose(
        placement.compute_anchor_positions(), [0.05, 0.10, 0.15]
    )


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
    np.testing.assert_allclose(
        placement.compute_anchor_positions(), [0.05, 0.10, 0.15]
    )


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


def test_cutting_sequence_spaces_cuts_by_slice_thickness(box_body):
    sequence = build_cutting_sequence(
        box_body,
        technique=CuttingTechnique.SLICE,
        slice_thickness=0.05,
        num_cuts_x=2,
    )
    _, points, _ = _sample(sequence)

    # usable X interval starts at -0.1 + margin of 0.01
    np.testing.assert_allclose(np.unique(points[:, 0]), [-0.04, 0.01], atol=1e-9)


def test_surface_sequence_stays_inside_xy_bounds(box_body):
    for technique in (
        WipingTechnique.WIPE,
        WipingTechnique.SHEAR,
        WipingTechnique.SPREAD,
    ):
        sequence = build_surface_sequence(box_body, technique=technique)
        _, points, _ = _sample(sequence)
        assert np.all(points[:, 0] >= -0.1 - 1e-9)
        assert np.all(points[:, 0] <= 0.1 + 1e-9)
        assert np.all(points[:, 1] >= -0.1 - 1e-9)
        assert np.all(points[:, 1] <= 0.1 + 1e-9)


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


def test_motion_sequence_sample_stitches_phases_without_duplicates():
    first = MotionSegment(
        name="first",
        duration_s=1.0,
        local_curve=lambda tau: np.array([tau, 0.0, 0.0]),
    )
    second = MotionSegment(
        name="second",
        duration_s=1.0,
        local_curve=lambda tau: np.array([1.0, tau, 0.0]),
    )
    sequence = MotionSequence([first, second])

    times, points, phase_ids = _sample(sequence)

    assert np.all(np.diff(times) > 0)
    assert set(np.unique(phase_ids)) == {0, 1}
    assert sequence.duration_s == pytest.approx(2.0)
    assert times[-1] == pytest.approx(2.0)
