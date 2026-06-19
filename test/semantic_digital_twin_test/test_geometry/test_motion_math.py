import types

import numpy as np
import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description import geometry, motion_presets
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import (
    Box,
    Scale,
    ShearProfile,
    ShearXYProfile,
    Sphere,
    SpiralProfile,
    SweepProfile,
    body_local_aabb,
    clamp_to_aabb,
    clamp_to_cylinder_xy,
    constant_orientation,
    fixed_rpy,
    make_constrained_curve,
    oscillatory_shear_local_profiled,
    oscillatory_shear_xy_profiled,
    planar_raster_xy,
    planar_spiral_xy,
    planar_sweep_x,
    points_world_to_body,
    ramp,
    rot_x,
    rot_y,
    rot_z,
    rpy_matrix,
    sample_local_curve,
    tilt_about_local_y,
)
from semantic_digital_twin.world_description.motion_presets import (
    MotionSegment,
    MotionSequence,
    build_container_sequence,
    build_cutting_sequence,
    build_default_sequence,
    build_pouring_sequence,
    build_surface_sequence,
    cutting_depth_metrics,
    mixing_bowl_metrics,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


class DummyFrame:
    def __init__(self, matrix):
        self._matrix = np.asarray(matrix, dtype=float)

    def to_np(self):
        return self._matrix


class DummyBody:
    """A plain object that exposes no `combined_mesh`, used to drive motion-preset
    builders through `body_local_aabb` without needing a real World/Body."""

    def __init__(self, mins, maxs):
        self.mins = np.asarray(mins, dtype=float)
        self.maxs = np.asarray(maxs, dtype=float)


@pytest.fixture
def patch_body_local_aabb(monkeypatch):
    def _patch(body_to_aabb):
        def fake(body, use_visual=False, apply_shape_scale=False):
            return body_to_aabb(body)

        monkeypatch.setattr(geometry, "body_local_aabb", fake)

    return _patch


def _make_box_world(scale_xyz, offset_xyz=(1.0, 2.0, 0.5)):
    """Build a two-body world: root -[fixed, offset]-> target (box collision)."""
    w = World()
    root = Body(name=PrefixedName("root"))
    target = Body.from_shape_collection(
        PrefixedName("target"), ShapeCollection([Box(scale=Scale(*scale_xyz))])
    )
    with w.modify_world():
        w.add_connection(
            FixedConnection(
                parent=root,
                child=target,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=offset_xyz[0],
                    y=offset_xyz[1],
                    z=offset_xyz[2],
                    reference_frame=root,
                ),
            )
        )
    return w, target


# --------------------------------------------------------------------------
# Rotation matrices and curve profiles
# --------------------------------------------------------------------------


def test_rot_matrices_and_rpy():
    assert rot_x(0.0) == pytest.approx(np.eye(3))
    assert rot_y(0.0) == pytest.approx(np.eye(3))
    assert rot_z(0.0) == pytest.approx(np.eye(3))
    assert rot_x(np.pi / 2) == pytest.approx(
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), abs=1e-9
    )
    assert rot_y(np.pi / 2) == pytest.approx(
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), abs=1e-9
    )
    assert rot_z(np.pi / 2) == pytest.approx(
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), abs=1e-9
    )
    assert rpy_matrix() == pytest.approx(np.eye(3))
    assert rpy_matrix(roll=np.pi / 2) == pytest.approx(rot_x(np.pi / 2), abs=1e-9)
    assert rpy_matrix(
        roll=0.1, pitch=0.2, yaw=0.3
    ) == pytest.approx(rot_z(0.3) @ rot_y(0.2) @ rot_x(0.1))


def test_constant_and_fixed_orientation():
    curve = constant_orientation()
    assert curve(0.3) == pytest.approx(np.eye(3))

    curve = fixed_rpy(yaw=np.pi / 2)
    assert curve(0.0) == pytest.approx(rot_z(np.pi / 2), abs=1e-9)
    assert curve(1.0) == pytest.approx(rot_z(np.pi / 2), abs=1e-9)


def test_tilt_about_local_y_ramp_hold_and_retract():
    profile = tilt_about_local_y(max_angle=np.pi / 2, ramp_in=0.3, hold_until=0.7)

    assert profile(0.0) == pytest.approx(np.eye(3))
    assert profile(0.15) == pytest.approx(rot_y(np.pi / 4), abs=1e-9)
    # Inside the hold plateau (ramp_in < tau <= hold_until): angle stays at max.
    assert profile(0.5) == pytest.approx(rot_y(np.pi / 2), abs=1e-9)
    assert profile(1.0) == pytest.approx(np.eye(3), abs=1e-9)

    # Values outside [0, 1] are clipped before evaluating the profile.
    assert profile(-1.0) == pytest.approx(np.eye(3))
    assert profile(2.0) == pytest.approx(np.eye(3))


def test_ramp_and_spiral_profiles():
    assert ramp(-0.1, tau_end=0.5, d_max=0.3) == 0.0
    assert ramp(0.25, tau_end=0.5, d_max=0.3) == pytest.approx(0.15)
    assert ramp(0.75, tau_end=0.5, d_max=0.3) == pytest.approx(0.3)

    spiral = planar_spiral_xy(0.5, r0=0.1, r1=0.5, cycles=1.0)
    assert spiral == pytest.approx(np.array([-0.3, 0.0, 0.0]))

    sweep = planar_sweep_x(0.25, length=0.2, cycles=1.0)
    assert sweep == pytest.approx(np.array([0.2, 0.0, 0.0]))


def test_raster_sampling_and_constraints():
    lane0 = planar_raster_xy(0.125, width=2.0, height=4.0, lanes=3)
    lane1 = planar_raster_xy(0.5, width=2.0, height=4.0, lanes=3)
    lane_last = planar_raster_xy(1.0, width=2.0, height=4.0, lanes=3)

    assert lane0 == pytest.approx(np.array([-0.25, -2.0, 0.0]))
    assert lane1 == pytest.approx(np.array([0.0, 0.0, 0.0]))
    # tau=1.0 exercises the "lane clamped to last lane" branch.
    assert lane_last == pytest.approx(np.array([1.0, 2.0, 0.0]))

    sampled = sample_local_curve(
        lambda tau: np.array([tau, tau**2, -tau], dtype=float),
        [0.0, 0.5, 1.0],
    )
    assert sampled == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [0.5, 0.25, -0.5], [1.0, 1.0, -1.0]])
    )

    clamped_box = clamp_to_aabb(
        np.array([2.0, -2.0, 0.1]),
        mins=np.array([-1.0, -1.0, -1.0]),
        maxs=np.array([1.0, 1.0, 1.0]),
        margin=0.2,
    )
    assert clamped_box == pytest.approx(np.array([0.8, -0.8, 0.1]))

    # Inside the radius: no clamping happens (r_xy <= r branch).
    clamped_inside = clamp_to_cylinder_xy(
        np.array([0.1, 0.1, 0.0]), radius=4.0, z_min=-1.0, z_max=1.0
    )
    assert clamped_inside == pytest.approx(np.array([0.1, 0.1, 0.0]))

    clamped_cyl = clamp_to_cylinder_xy(
        np.array([3.0, 4.0, 2.0]), radius=4.0, z_min=-1.0, z_max=1.0, margin=0.5
    )
    assert clamped_cyl == pytest.approx(np.array([2.1, 2.8, 0.5]))

    constrained = make_constrained_curve(
        lambda tau: np.array([tau, tau, tau], dtype=float),
        lambda q: q * 2.0,
    )
    assert constrained(0.25) == pytest.approx(np.array([0.5, 0.5, 0.5]))


def test_shear_profiles():
    local = oscillatory_shear_local_profiled(
        0.25,
        ShearProfile(depth_max=0.4, depth_ramp_end=0.5, shear_amp=0.3, shear_cycles=1.0),
    )
    xy = oscillatory_shear_xy_profiled(
        0.0, ShearXYProfile(shear_amp=0.2, shear_cycles=1.0)
    )

    assert local == pytest.approx(np.array([0.3, 0.0, -0.2]))
    assert xy == pytest.approx(np.array([0.0, 0.2, 0.0]))


def test_spiral_and_sweep_profile_dataclasses_hold_parameters():
    spiral = SpiralProfile(r0=0.0, r1=0.5, cycles=2.0)
    sweep = SweepProfile(length=0.1, cycles=3.0)
    assert (spiral.r0, spiral.r1, spiral.cycles) == (0.0, 0.5, 2.0)
    assert (sweep.length, sweep.cycles) == (0.1, 3.0)


# --------------------------------------------------------------------------
# MotionSegment / MotionSequence
# --------------------------------------------------------------------------


def test_motion_segment_sampling_applies_frame_transform():
    segment = MotionSegment(
        name="line",
        duration_s=1.0,
        local_curve=lambda tau: np.array([tau, 2.0 * tau, 0.0], dtype=float),
    )
    frame = np.array(
        [
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    times, pts = segment.sample(frame, dt=0.5, t0=1.0)

    assert times == pytest.approx(np.array([1.0, 1.5, 2.0]))
    assert pts == pytest.approx(
        np.array([[10.0, 20.0, 30.0], [9.0, 20.5, 30.0], [8.0, 21.0, 30.0]])
    )


def test_motion_sequence_concatenates_without_duplicate_boundary():
    seq = MotionSequence(
        [
            MotionSegment(
                name="first",
                duration_s=1.0,
                local_curve=lambda tau: np.array([tau, 0.0, 0.0], dtype=float),
            ),
            MotionSegment(
                name="second",
                duration_s=1.0,
                local_curve=lambda tau: np.array([1.0, tau, 0.0], dtype=float),
            ),
        ]
    )

    times, pts, phase_ids = seq.sample(DummyFrame(np.eye(4)), dt=0.5, t0=2.0)

    assert seq.duration_s == pytest.approx(2.0)
    assert times == pytest.approx(np.array([2.0, 2.5, 3.0, 3.5, 4.0]))
    assert pts == pytest.approx(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    assert np.array_equal(phase_ids, np.array([0, 0, 0, 1, 1]))


def test_motion_segment_and_sequence_pose_sampling():
    segment = MotionSegment(
        name="tilt_line",
        duration_s=1.0,
        local_curve=lambda tau: np.array([tau, 0.0, 0.0], dtype=float),
        local_orientation_curve=tilt_about_local_y(
            max_angle=np.pi / 2, ramp_in=0.5, hold_until=0.5
        ),
    )
    frame = DummyFrame(np.eye(4))

    times, positions, rotations = segment.sample_poses(frame.to_np(), dt=0.5, t0=0.0)
    assert times == pytest.approx(np.array([0.0, 0.5, 1.0]))
    assert positions == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )
    assert rotations[0] == pytest.approx(np.eye(3))
    assert rotations[1] == pytest.approx(rot_y(np.pi / 2), abs=1e-9)
    assert rotations[2] == pytest.approx(np.eye(3), abs=1e-9)

    seq = MotionSequence(
        [
            MotionSegment(
                name="first",
                duration_s=1.0,
                local_curve=lambda tau: np.array([tau, 0.0, 0.0], dtype=float),
                local_orientation_curve=fixed_rpy(yaw=np.pi / 2),
            ),
            MotionSegment(
                name="second",
                duration_s=1.0,
                local_curve=lambda tau: np.array([1.0, tau, 0.0], dtype=float),
            ),
        ]
    )

    sampled = seq.sample_poses(DummyFrame(np.eye(4)), dt=0.5, t0=2.0)
    assert sampled.times == pytest.approx(np.array([2.0, 2.5, 3.0, 3.5, 4.0]))
    assert sampled.positions == pytest.approx(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    assert sampled.rotations[0] == pytest.approx(rot_z(np.pi / 2), abs=1e-9)
    assert sampled.rotations[-1] == pytest.approx(np.eye(3))
    assert np.array_equal(sampled.phase_ids, np.array([0, 0, 0, 1, 1]))


# --------------------------------------------------------------------------
# _shape_scale_xyz
# --------------------------------------------------------------------------


def test_shape_scale_xyz_branches():
    box = Box(scale=Scale(1.0, 2.0, 3.0))
    assert geometry._shape_scale_xyz(box) == pytest.approx([1.0, 2.0, 3.0])

    # Shapes without a `scale` attribute (e.g. Sphere) yield None.
    assert geometry._shape_scale_xyz(Sphere(radius=0.2)) is None

    # Array-like scale without x/y/z attributes is coerced via np.asarray.
    array_like = types.SimpleNamespace(scale=(4.0, 5.0, 6.0))
    assert geometry._shape_scale_xyz(array_like) == pytest.approx([4.0, 5.0, 6.0])

    # Scale that cannot be reshaped to length 3 falls back to None.
    unreshapeable = types.SimpleNamespace(scale="not-a-vector")
    assert geometry._shape_scale_xyz(unreshapeable) is None


# --------------------------------------------------------------------------
# body_local_aabb / points_world_to_body / metrics on a real World
# --------------------------------------------------------------------------


def test_body_local_aabb_collision_and_visual():
    w, target = _make_box_world(scale_xyz=(0.2, 0.4, 0.1))

    mins, maxs = body_local_aabb(target)
    assert mins == pytest.approx([-0.1, -0.2, -0.05])
    assert maxs == pytest.approx([0.1, 0.2, 0.05])

    # visual and collision share the same shape collection here, so the
    # use_visual=True branch produces the same result.
    mins_v, maxs_v = body_local_aabb(target, use_visual=True)
    assert mins_v == pytest.approx(mins)
    assert maxs_v == pytest.approx(maxs)


def test_body_local_aabb_apply_shape_scale_multiplies_bounds_when_scale_exceeds_one():
    w = World()
    big_box = Body.from_shape_collection(
        PrefixedName("big_box"), ShapeCollection([Box(scale=Scale(2.0, 1.0, 1.0))])
    )
    with w.modify_world():
        w.add_body(big_box)

    mins_unscaled, maxs_unscaled = body_local_aabb(big_box, apply_shape_scale=False)
    assert mins_unscaled == pytest.approx([-1.0, -0.5, -0.5])
    assert maxs_unscaled == pytest.approx([1.0, 0.5, 0.5])

    # max_scale = max(1, |scale|) elementwise; only the x-axis (scale=2) changes.
    mins_scaled, maxs_scaled = body_local_aabb(big_box, apply_shape_scale=True)
    assert mins_scaled == pytest.approx([-2.0, -0.5, -0.5])
    assert maxs_scaled == pytest.approx([2.0, 0.5, 0.5])


def test_body_local_aabb_apply_shape_scale_skips_shapes_without_scale():
    # A Sphere has no `scale` attribute; it must be skipped (not crash) when
    # computing the max scale alongside a Box that does have one.
    w = World()
    body = Body.from_shape_collection(
        PrefixedName("mixed_shapes"),
        ShapeCollection(
            [
                Box(scale=Scale(2.0, 1.0, 1.0)),
                Sphere(radius=0.1),
            ]
        ),
    )
    with w.modify_world():
        w.add_body(body)

    mins_scaled, maxs_scaled = body_local_aabb(body, apply_shape_scale=True)
    assert mins_scaled[0] == pytest.approx(-2.0)
    assert maxs_scaled[0] == pytest.approx(2.0)


def test_body_local_aabb_apply_shape_scale_is_noop_when_scale_below_one():
    w, target = _make_box_world(scale_xyz=(0.2, 0.4, 0.1))

    mins, maxs = body_local_aabb(target, apply_shape_scale=False)
    mins_scaled, maxs_scaled = body_local_aabb(target, apply_shape_scale=True)
    assert mins_scaled == pytest.approx(mins)
    assert maxs_scaled == pytest.approx(maxs)


def test_body_local_aabb_unions_multiple_shapes():
    w = World()
    body = Body.from_shape_collection(
        PrefixedName("multi_shape"),
        ShapeCollection(
            [
                Box(
                    scale=Scale(0.2, 0.2, 0.2),
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=-1.0),
                ),
                Box(
                    scale=Scale(0.2, 0.2, 0.2),
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0),
                ),
            ]
        ),
    )
    with w.modify_world():
        w.add_body(body)

    mins, maxs = body_local_aabb(body)
    assert mins == pytest.approx([-1.1, -0.1, -0.1])
    assert maxs == pytest.approx([1.1, 0.1, 0.1])


def test_points_world_to_body_applies_inverse_offset():
    w, target = _make_box_world(scale_xyz=(0.2, 0.2, 0.1), offset_xyz=(1.0, 2.0, 0.5))

    points_world = np.array([[1.0, 2.0, 0.6], [1.0, 2.0, 0.4]])
    points_body = points_world_to_body(points_world, w, target)

    assert points_body == pytest.approx(np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]]))


def test_cutting_depth_metrics_detects_entry_from_above():
    w, target = _make_box_world(scale_xyz=(0.2, 0.2, 0.1))

    points_world = np.array([[1.0, 2.0, 0.6], [1.0, 2.0, 0.4]])
    metrics = cutting_depth_metrics(points_world, w, target)

    assert metrics["z_top"] == pytest.approx(0.05)
    assert metrics["z_cut"] == pytest.approx(-0.045)
    assert metrics["inside_xy_ratio"] == pytest.approx(1.0)
    assert metrics["has_entry_from_above"] is True


def test_cutting_depth_metrics_no_entry_when_points_stay_outside_xy():
    w, target = _make_box_world(scale_xyz=(0.2, 0.2, 0.1))

    # Far outside the bread's XY footprint, even though above the top surface.
    points_world = np.array([[5.0, 5.0, 10.0]])
    metrics = cutting_depth_metrics(points_world, w, target)

    assert metrics["inside_xy_ratio"] == pytest.approx(0.0)
    assert metrics["has_entry_from_above"] is False


def test_mixing_bowl_metrics_success_and_failure():
    w, target = _make_box_world(scale_xyz=(0.4, 0.4, 0.2))

    # Points centered and near the top of the bowl, close to the wall: success.
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    radius = 0.95 * (0.5 * min(0.4, 0.4) - 0.005)
    near_wall_xy = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    points_world = np.column_stack(
        [
            near_wall_xy[:, 0] + 1.0,
            near_wall_xy[:, 1] + 2.0,
            np.full(len(angles), 0.5),
        ]
    )
    metrics = mixing_bowl_metrics(points_world, w, target)
    assert metrics["inside_ratio"] == pytest.approx(1.0)
    assert metrics["near_interior_ratio"] == pytest.approx(1.0)
    assert metrics["mixing_success"] is True

    # Points far above the bowl: failure.
    points_world_outside = np.array([[1.0, 2.0, 5.0]])
    metrics_fail = mixing_bowl_metrics(points_world_outside, w, target)
    assert metrics_fail["inside_ratio"] == pytest.approx(0.0)
    assert metrics_fail["mixing_success"] is False


# --------------------------------------------------------------------------
# Convex hull helpers used by build_surface_sequence
# --------------------------------------------------------------------------


def test_convex_hull_xy_handles_degenerate_and_normal_cases():
    single = geometry._convex_hull_xy(np.array([[0.0, 0.0]]))
    assert single.shape == (1, 2)
    assert single[0] == pytest.approx(np.array([0.0, 0.0]))

    two_points = geometry._convex_hull_xy(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert two_points[0] == pytest.approx(np.array([0.0, 0.0]))
    assert two_points[1] == pytest.approx(np.array([1.0, 1.0]))

    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    hull = geometry._convex_hull_xy(square)
    assert len(hull) == 4
    assert set(map(tuple, np.round(hull, 6))) == {
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    }


def test_point_in_convex_polygon_xy():
    assert geometry._point_in_convex_polygon_xy(
        np.array([0.0, 0.0]), np.array([[0.0, 0.0], [1.0, 0.0]])
    )

    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert geometry._point_in_convex_polygon_xy(np.array([0.5, 0.5]), square)
    assert not geometry._point_in_convex_polygon_xy(np.array([2.0, 2.0]), square)
    # On the boundary: should still count as inside (cross == 0 is skipped).
    assert geometry._point_in_convex_polygon_xy(np.array([0.5, 0.0]), square)


def test_project_point_to_segment_xy():
    degenerate = geometry._project_point_to_segment_xy(
        np.array([5.0, 5.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0])
    )
    assert degenerate == pytest.approx([1.0, 1.0])

    projected = geometry._project_point_to_segment_xy(
        np.array([0.5, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
    )
    assert projected == pytest.approx([0.5, 0.0])

    clamped_start = geometry._project_point_to_segment_xy(
        np.array([-1.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
    )
    assert clamped_start == pytest.approx([0.0, 0.0])


def test_constrain_to_convex_hull_xy():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    inside = geometry._constrain_to_convex_hull_xy(
        np.array([0.5, 0.5, 0.3]), square
    )
    assert inside == pytest.approx([0.5, 0.5, 0.3])

    outside = geometry._constrain_to_convex_hull_xy(
        np.array([2.0, 0.5, 0.3]), square
    )
    assert outside == pytest.approx([1.0, 0.5, 0.3])

    # Fewer than 3 hull points: constraint is a no-op.
    degenerate_hull = geometry._constrain_to_convex_hull_xy(
        np.array([2.0, 0.5, 0.3]), np.array([[0.0, 0.0], [1.0, 0.0]])
    )
    assert degenerate_hull == pytest.approx([2.0, 0.5, 0.3])


def test_top_surface_hull_xy_branches():
    assert geometry._top_surface_hull_xy(types.SimpleNamespace()) is None
    assert (
        geometry._top_surface_hull_xy(
            types.SimpleNamespace(combined_mesh=types.SimpleNamespace(is_empty=True))
        )
        is None
    )

    empty_mesh = types.SimpleNamespace(
        is_empty=False,
        face_normals=np.empty((0, 3)),
        faces=np.empty((0, 3), dtype=int),
        vertices=np.empty((0, 3)),
    )
    assert (
        geometry._top_surface_hull_xy(types.SimpleNamespace(combined_mesh=empty_mesh))
        is None
    )

    no_upward_faces = types.SimpleNamespace(
        is_empty=False,
        face_normals=np.array([[0.0, 0.0, -1.0]]),
        faces=np.array([[0, 1, 2]]),
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert (
        geometry._top_surface_hull_xy(
            types.SimpleNamespace(combined_mesh=no_upward_faces)
        )
        is None
    )

    too_few_vertices = types.SimpleNamespace(
        is_empty=False,
        face_normals=np.array([[0.0, 0.0, 1.0]]),
        faces=np.array([[0, 0, 1]]),
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    assert (
        geometry._top_surface_hull_xy(
            types.SimpleNamespace(combined_mesh=too_few_vertices)
        )
        is None
    )

    box_mesh = types.SimpleNamespace(
        is_empty=False,
        face_normals=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        faces=np.array([[0, 1, 2], [3, 4, 5]]),
        vertices=np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
                [1.0, 1.0, -1.0],
            ]
        ),
    )
    hull = geometry._top_surface_hull_xy(types.SimpleNamespace(combined_mesh=box_mesh))
    assert hull is not None
    assert len(hull) == 3


# --------------------------------------------------------------------------
# Motion presets (use a monkeypatched body_local_aabb + DummyBody, matching
# the original thesis-branch test design so the math is tested in isolation
# from the heavier real-World geometry pipeline).
# --------------------------------------------------------------------------


def test_duration_scale_uses_aabb_diagonal_and_validates_reference(
    patch_body_local_aabb,
):
    body = DummyBody(mins=[0.0, 0.0, 0.0], maxs=[3.0, 4.0, 0.0])
    patch_body_local_aabb(lambda b: (b.mins, b.maxs))

    scale = motion_presets._duration_scale_from_body(body, reference_size=2.5)
    assert scale == pytest.approx(2.0)

    scale_debug = motion_presets._duration_scale_from_body(
        body, reference_size=2.5, debug=True
    )
    assert scale_debug == pytest.approx(2.0)

    with pytest.raises(ValueError, match="reference_size must be positive"):
        motion_presets._duration_scale_from_body(body, reference_size=0.0)


def test_build_default_sequence_returns_three_named_phases():
    seq = build_default_sequence()

    assert [phase.name for phase in seq.phases] == [
        "planar_spiral",
        "oscillatory_shear",
        "planar_sweep",
    ]
    assert seq.duration_s == pytest.approx(5.0)


def test_build_container_sequence_supports_spiral_and_stir_patterns(
    patch_body_local_aabb,
):
    bowl = DummyBody(mins=[-0.2, -0.3, 0.1], maxs=[0.2, 0.3, 0.5])
    patch_body_local_aabb(lambda b: (b.mins, b.maxs))

    spiral = build_container_sequence(bowl, pattern="spiral", debug=True)
    stir = build_container_sequence(bowl, pattern="loop", mix_duration_s=5.5)
    stir_default_duration = build_container_sequence(
        bowl, pattern="loop", mix_duration_s=0.0
    )

    assert [phase.name for phase in spiral.phases] == ["planar_spiral_bowl"]
    spiral_start = spiral.phases[0].local_curve(0.0)
    spiral_edge = spiral.phases[0].local_curve(1.0)
    assert spiral_start == pytest.approx(np.array([0.0, 0.0, 0.495]))
    assert np.linalg.norm(spiral_edge[:2]) <= 0.195 + 1e-9
    assert 0.105 <= spiral_edge[2] <= 0.495

    assert [phase.name for phase in stir.phases] == ["continuous_stir_bowl"]
    assert stir.phases[0].duration_s == pytest.approx(5.5)
    stir_point = stir.phases[0].local_curve(0.25)
    assert np.linalg.norm(stir_point[:2]) <= 0.195 + 1e-9
    assert stir_point[2] == pytest.approx(0.495)

    # mix_duration_s=0.0 falls back to the same stir-base duration as mix_duration_s=None.
    stir_none = build_container_sequence(bowl, pattern="loop", mix_duration_s=None)
    assert stir_default_duration.phases[0].duration_s == pytest.approx(
        stir_none.phases[0].duration_s
    )

    with pytest.raises(ValueError, match="Unknown pattern"):
        build_container_sequence(bowl, pattern="unknown")


def test_build_surface_sequence_selects_expected_patterns(patch_body_local_aabb):
    surface = DummyBody(mins=[-0.4, -0.2, 0.0], maxs=[0.6, 0.2, 0.1])
    patch_body_local_aabb(lambda b: (b.mins, b.maxs))

    spiral = build_surface_sequence(surface, pattern="planar_spiral", debug=True)
    shear = build_surface_sequence(surface, pattern="shear")
    raster = build_surface_sequence(surface, pattern="surface_cover")
    wipe_via_technique = build_surface_sequence(surface, technique="wipe", pattern=None)

    assert [phase.name for phase in spiral.phases] == ["planar_spiral_surface"]
    assert [phase.name for phase in shear.phases] == ["oscillatory_shear_surface"]
    assert [phase.name for phase in raster.phases] == ["planar_raster_surface"]
    assert [phase.name for phase in wipe_via_technique.phases] == [
        "planar_spiral_surface"
    ]

    center = np.array([0.1, 0.0, 0.115])
    assert spiral.phases[0].local_curve(0.0) == pytest.approx(center)
    assert shear.phases[0].local_curve(0.0) == pytest.approx(
        center + np.array([0.0, 0.063, 0.0])
    )
    assert raster.phases[0].local_curve(0.0) == pytest.approx(
        center + np.array([-0.45, -0.18, 0.0])
    )

    with pytest.raises(ValueError, match="Unknown pattern"):
        build_surface_sequence(surface, pattern="zigzag")
    with pytest.raises(ValueError, match="Unknown surface technique"):
        build_surface_sequence(surface, technique="zigzag", pattern=None)


def test_build_surface_sequence_applies_top_surface_hull_constraint(
    patch_body_local_aabb, monkeypatch
):
    surface = DummyBody(mins=[-0.4, -0.4, 0.0], maxs=[0.4, 0.4, 0.1])
    patch_body_local_aabb(lambda b: (b.mins, b.maxs))
    hull = np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]])
    monkeypatch.setattr(geometry, "_top_surface_hull_xy", lambda body: hull)

    seq = build_surface_sequence(surface, pattern="planar_spiral", debug=True)
    # The spiral would normally reach far beyond the tiny hull; the
    # hull constraint must clamp it back onto the hull boundary.
    far_point = seq.phases[0].local_curve(1.0)
    assert abs(far_point[0] - 0.1) <= 0.05 + 1e-9
    assert abs(far_point[1] - 0.0) <= 0.05 + 1e-9


def test_build_cutting_sequence_covers_slice_saw_and_halving(patch_body_local_aabb):
    food = DummyBody(mins=[0.0, 0.0, 0.0], maxs=[0.3, 0.2, 0.1])
    patch_body_local_aabb(lambda b: (b.mins, b.maxs))

    slice_seq = build_cutting_sequence(
        food, technique="slice", num_cuts_x=2, slice_thickness=0.04, debug=True
    )
    saw_seq = build_cutting_sequence(food, technique="saw")
    halving_seq = build_cutting_sequence(food, technique="halving")
    # Slice thickness so large that x_max_anchor <= x_anchor for multiple cuts.
    crowded_seq = build_cutting_sequence(
        food, technique="slice", num_cuts_x=3, slice_thickness=10.0
    )

    assert [phase.name for phase in slice_seq.phases] == [
        "cut_approach_x0",
        "cut_descend_x0",
        "cut_retract_x0",
        "cut_approach_x1",
        "cut_descend_x1",
        "cut_retract_x1",
    ]
    assert len(saw_seq.phases) == 3
    assert [phase.name for phase in saw_seq.phases] == [
        "cut_approach_x0",
        "oscillatory_shear_x0",
        "cut_retract_x0",
    ]
    assert [phase.name for phase in halving_seq.phases] == [
        "cut_approach_x0",
        "cut_descend_x0",
        "cut_retract_x0",
    ]
    assert len(crowded_seq.phases) == 9
    x_values = {
        round(float(phase.local_curve(0.0)[0]), 6)
        for phase in crowded_seq.phases
        if phase.name.startswith("cut_approach")
    }
    assert len(x_values) == 1  # all anchors collapsed to the same x.

    slice_first = slice_seq.phases[0].local_curve(0.0)
    slice_last = slice_seq.phases[3].local_curve(0.0)
    assert slice_first == pytest.approx(np.array([0.03, 0.1, 0.145]))
    assert slice_last == pytest.approx(np.array([0.27, 0.1, 0.145]))

    saw_mid = saw_seq.phases[1].local_curve(0.25)
    assert saw_mid == pytest.approx(np.array([0.025, 0.1, 0.06607142857142857]))

    halving_mid = halving_seq.phases[1].local_curve(1.0)
    assert halving_mid == pytest.approx(np.array([0.15, 0.1, 0.05]))

    with pytest.raises(ValueError, match="Unknown cutting technique"):
        build_cutting_sequence(food, technique="dice")


def test_build_pouring_sequence_generates_pose_aware_phases(patch_body_local_aabb):
    source = DummyBody(mins=[-0.05, -0.03, 0.0], maxs=[0.05, 0.03, 0.12])
    target = DummyBody(mins=[0.20, -0.04, 0.0], maxs=[0.32, 0.04, 0.10])
    patch_body_local_aabb(lambda b: (b.mins, b.maxs))

    seq = build_pouring_sequence(
        source,
        target_body=target,
        pour_height=0.08,
        approach_distance=0.06,
        retreat_distance=0.10,
        max_tilt=np.pi / 3,
        debug=True,
    )
    seq_no_target = build_pouring_sequence(source, debug=True)

    assert [phase.name for phase in seq.phases] == [
        "pour_approach",
        "pour_tilt_in",
        "pour_hold",
        "pour_tilt_out_retreat",
    ]
    # Without a target body, the anchor is derived from the source body itself.
    assert seq_no_target.phases[0].local_curve(1.0)[2] == pytest.approx(0.12 + 0.10)

    anchor = np.array([0.26, 0.0, 0.18])
    approach = np.array([0.20, 0.0, 0.18])
    retreat = np.array([0.16, 0.0, 0.18])

    assert seq.phases[0].local_curve(0.0) == pytest.approx(approach)
    assert seq.phases[0].local_curve(1.0) == pytest.approx(anchor)
    assert seq.phases[1].local_curve(0.5) == pytest.approx(anchor)
    assert seq.phases[2].local_curve(0.5) == pytest.approx(anchor)
    assert seq.phases[3].local_curve(1.0) == pytest.approx(retreat)

    sampled = seq.sample_poses(DummyFrame(np.eye(4)), dt=1.0)
    assert sampled.positions[0] == pytest.approx(approach)
    assert sampled.positions[-1] == pytest.approx(retreat)
    assert sampled.rotations[0] == pytest.approx(np.eye(3))
    assert np.any(
        np.all(np.isclose(sampled.rotations, rot_y(np.pi / 3)), axis=(1, 2))
    )
    assert sampled.rotations[-1] == pytest.approx(np.eye(3))

    with pytest.raises(ValueError, match="Unsupported pouring tilt axis"):
        build_pouring_sequence(source, tilt_axis="x")
