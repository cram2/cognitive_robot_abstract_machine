from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import Callable, List, Optional, Tuple

from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.datastructures.enums import (
    CuttingTechnique,
    MixingPattern,
    SlicingPriority,
    ToolPathSegmentKind,
    WipingTechnique,
)

LocalCurve = Callable[[float], np.ndarray]
"""
A curve mapping a normalized time in [0, 1] to a local 3D point.
"""


def _local_bounding_box(body: Body, use_visual: bool = False) -> BoundingBox:
    """
    :param body: The body whose bounding box is computed.
    :param use_visual: Use the visual geometry instead of the collision geometry.
    :return: The local axis-aligned bounding box of the body in its own frame.
    """
    geometry = body.visual if use_visual else body.collision
    return geometry.as_bounding_box_collection_in_frame(body).bounding_box()


@dataclass
class ToolPathSegment:
    """
    A local curve of one geometric pattern over a fixed duration.
    """

    kind: ToolPathSegmentKind
    """
    The geometric pattern this segment follows.
    """

    duration: float
    """
    Duration of the segment in seconds.
    """

    local_curve: LocalCurve
    """
    Curve mapping normalized time to a local 3D point.
    """

    cut_index: Optional[int] = None
    """
    Index of the cut this segment belongs to, if the segment is part of a multi-cut
    path.
    """

    def sample(
        self, frame_np: np.ndarray, dt: float, t0: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample the segment's curve in the given frame.

        :param frame_np: Homogeneous 4x4 transformation of the frame the curve is
            expressed in.
        :param dt: Sampling period in seconds.
        :param t0: Start time of the segment.
        :return: Sample times and the sampled points as an Nx3 array.
        """
        number_of_samples = max(2, int(np.ceil(self.duration / float(dt))) + 1)
        times = np.linspace(t0, t0 + self.duration, number_of_samples)
        tau = (times - t0) / self.duration

        frame = np.asarray(frame_np, dtype=float)
        rotation = frame[:3, :3]
        translation = frame[:3, 3]

        local_points = np.array(
            [self.local_curve(float(u)) for u in tau], dtype=float
        ).reshape(-1, 3)
        points = local_points @ rotation.T + translation

        return times, points


@dataclass
class ToolPath:
    """
    An ordered list of tool path segments forming one continuous path.
    """

    segments: List[ToolPathSegment]
    """
    The ordered segments of this path.
    """

    @property
    def duration(self) -> float:
        """
        :return: Total duration across all segments in seconds.
        """
        return float(sum(segment.duration for segment in self.segments))

    def sample(
        self, frame: HomogeneousTransformationMatrix, dt: float = 0.01, t0: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample all segments into one concatenated path.

        :param frame: The frame the path's curves are expressed in.
        :param dt: Sampling period in seconds.
        :param t0: Start time of the path.
        :return: Sample times, sampled points as an Nx3 array, and the segment index of
            every sample.
        """
        all_times, all_points, all_segment_ids = [], [], []
        t = float(t0)

        for segment_index, segment in enumerate(self.segments):
            times, points = segment.sample(frame.to_np(), dt=dt, t0=t)
            if all_times:
                times = times[1:]
                points = points[1:]

            all_times.append(times)
            all_points.append(points)
            all_segment_ids.append(np.full(len(times), segment_index, dtype=int))
            t += segment.duration

        return (
            np.concatenate(all_times),
            np.vstack(all_points),
            np.concatenate(all_segment_ids),
        )


def ramp(tau: float, tau_end: float, d_max: float) -> float:
    """
    :param tau: Normalized time in [0, 1].
    :param tau_end: Normalized time at which the ramp reaches its maximum.
    :param d_max: Maximum value of the ramp.
    :return: A linear ramp from 0 to ``d_max`` over ``tau_end``.
    """
    if tau <= 0.0:
        return 0.0
    if tau >= tau_end:
        return float(d_max)
    return float(d_max) * (tau / tau_end)


def planar_spiral_xy(tau: float, r0: float, r1: float, cycles: float) -> np.ndarray:
    """
    :param tau: Normalized time in [0, 1].
    :param r0: Radius of the spiral at the start of the motion.
    :param r1: Radius of the spiral at the end of the motion.
    :param cycles: Number of revolutions over the motion.
    :return: A point on a planar spiral in XY with linearly growing radius.
    """
    radius = r0 + (r1 - r0) * tau
    angle = 2.0 * np.pi * cycles * tau
    return np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0], dtype=float)


def planar_sweep_x(tau: float, length: float, cycles: float) -> np.ndarray:
    """
    :param tau: Normalized time in [0, 1].
    :param length: Amplitude of the sweep in meters.
    :param cycles: Number of sweep oscillations over the motion.
    :return: A point on a sinusoidal sweep along the X axis.
    """
    sweep = float(length) * np.sin(2.0 * np.pi * float(cycles) * tau)
    return np.array([sweep, 0.0, 0.0], dtype=float)


def planar_raster_xy(tau: float, width: float, height: float, lanes: int) -> np.ndarray:
    """
    :param tau: Normalized time in [0, 1].
    :param width: Width of the covered rectangle along X in meters.
    :param height: Height of the covered rectangle along Y in meters.
    :param lanes: Number of parallel lanes of the raster scan.
    :return: A point on a raster scan covering a rectangle in XY.
    """
    rectangle_width = float(width)
    rectangle_height = float(height)
    number_of_lanes = max(2, int(lanes))
    lane_progress = float(np.clip(tau, 0.0, 1.0)) * number_of_lanes
    lane = min(int(np.floor(lane_progress)), number_of_lanes - 1)
    local_t = lane_progress - lane

    x_start = -0.5 * rectangle_width
    x_end = 0.5 * rectangle_width
    if lane % 2 == 0:
        x = x_start + (x_end - x_start) * local_t
    else:
        x = x_end - (x_end - x_start) * local_t

    y = -0.5 * rectangle_height + (rectangle_height * lane / float(number_of_lanes - 1))
    return np.array([x, y, 0.0], dtype=float)


@dataclass(frozen=True)
class ShearProfile:
    """
    Parameters for an oscillatory shear motion with a monotone depth profile.
    """

    depth_max: float
    """
    Maximum depth of the shear motion.
    """

    depth_ramp_end: float
    """
    Normalized time at which the maximum depth is reached.
    """

    shear_amp: float
    """
    Amplitude of the shear oscillation.
    """

    shear_cycles: float
    """
    Number of shear oscillations over the motion.
    """


@dataclass(frozen=True)
class ShearXYProfile:
    """
    Parameters for an oscillatory shear motion in the XY plane.
    """

    shear_amp: float
    """
    Amplitude of the shear oscillation.
    """

    shear_cycles: float
    """
    Number of shear oscillations over the motion.
    """


def oscillatory_shear_local_profiled(tau: float, profile: ShearProfile) -> np.ndarray:
    """
    :param tau: Normalized time in [0, 1].
    :param profile: Depth and oscillation parameters of the shear motion.
    :return: A point on an oscillatory shear curve with a monotone depth profile.
    """
    depth = ramp(tau, tau_end=profile.depth_ramp_end, d_max=profile.depth_max)
    shear = float(profile.shear_amp) * np.sin(
        2.0 * np.pi * float(profile.shear_cycles) * tau
    )
    return np.array([shear, 0.0, -depth], dtype=float)


def oscillatory_shear_xy_profiled(tau: float, profile: ShearXYProfile) -> np.ndarray:
    """
    :param tau: Normalized time in [0, 1].
    :param profile: Oscillation parameters of the shear motion.
    :return: A point on an oscillatory shear curve in the XY plane with no depth
        change.
    """
    angle = 2.0 * np.pi * float(profile.shear_cycles) * tau
    amplitude = float(profile.shear_amp)
    return np.array(
        [amplitude * np.sin(angle), amplitude * np.cos(angle), 0.0], dtype=float
    )


def clamp_to_cylinder_xy(
    q_local: np.ndarray,
    radius: float,
    z_min: float,
    z_max: float,
    margin: float = 0.0,
) -> np.ndarray:
    """
    :param q_local: The local 3D point to clamp.
    :param radius: Radius of the cylinder in meters.
    :param z_min: Lower Z bound in meters.
    :param z_max: Upper Z bound in meters.
    :param margin: Safety margin in meters kept from the cylinder walls and Z bounds.
    :return: The point clamped to a vertical cylinder in XY and to the Z bounds.
    """
    point = np.asarray(q_local, dtype=float).reshape(3)
    clamped_radius = float(radius) - float(margin)
    xy = point[:2]
    radius_xy = np.linalg.norm(xy)
    if radius_xy > clamped_radius and radius_xy > 1e-9:
        xy = (clamped_radius / radius_xy) * xy
    z = np.clip(point[2], float(z_min) + float(margin), float(z_max) - float(margin))
    return np.array([xy[0], xy[1], z], dtype=float)


def make_constrained_curve(
    local_curve: LocalCurve, constraint_fn: Callable[[np.ndarray], np.ndarray]
) -> LocalCurve:
    """
    :param local_curve: The curve to constrain.
    :param constraint_fn: Function that maps a curve point to its constrained
        counterpart.
    :return: The curve wrapped so every point respects the constraint function.
    """
    return lambda tau: constraint_fn(local_curve(float(tau)))


def _cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    :param o: Origin point of the two spanning vectors.
    :param a: End point of the first vector.
    :param b: End point of the second vector.
    :return: The 2D cross product of the vectors ``o -> a`` and ``o -> b``.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    """
    :param points_xy: The 2D points to compute the hull of, as an Nx2 array.
    :return: The convex hull of 2D points, ordered counter-clockwise.
    """
    points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if len(points) <= 1:
        return points
    points = np.unique(np.round(points, decimals=8), axis=0)
    if len(points) <= 2:
        return points
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    lower = []
    for point in points:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in points[::-1]:
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _point_in_convex_polygon_xy(point_xy: np.ndarray, hull_xy: np.ndarray) -> bool:
    """
    :param point_xy: The 2D point to test.
    :param hull_xy: The convex polygon as an Nx2 array of ordered vertices.
    :return: True if the point lies inside or on the boundary of the polygon.
    """
    hull = np.asarray(hull_xy, dtype=float).reshape(-1, 2)
    point = np.asarray(point_xy, dtype=float).reshape(2)
    if len(hull) < 3:
        return True
    eps = 1e-9
    previous_sign = 0.0
    for i in range(len(hull)):
        a = hull[i]
        b = hull[(i + 1) % len(hull)]
        cross = _cross_2d(a, b, point)
        if abs(cross) <= eps:
            continue
        sign = np.sign(cross)
        if previous_sign == 0.0:
            previous_sign = sign
        elif sign != previous_sign:
            return False
    return True


def _project_point_to_segment_xy(
    point_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray
) -> np.ndarray:
    """
    :param point_xy: The 2D point to project.
    :param a_xy: Start point of the segment.
    :param b_xy: End point of the segment.
    :return: The closest point to ``point_xy`` on the segment.
    """
    point = np.asarray(point_xy, dtype=float).reshape(2)
    a = np.asarray(a_xy, dtype=float).reshape(2)
    b = np.asarray(b_xy, dtype=float).reshape(2)
    segment = b - a
    squared_length = float(np.dot(segment, segment))
    if squared_length <= 1e-12:
        return a.copy()
    t = float(np.dot(point - a, segment) / squared_length)
    t = np.clip(t, 0.0, 1.0)
    return a + t * segment


def _constrain_to_convex_hull_xy(
    q_local: np.ndarray, hull_xy: np.ndarray
) -> np.ndarray:
    """
    :param q_local: The local 3D point to constrain.
    :param hull_xy: The convex hull as an Nx2 array of ordered vertices.
    :return: The point moved to the closest hull edge if it lies outside the hull.
    """
    point = np.asarray(q_local, dtype=float).reshape(3)
    if len(hull_xy) < 3 or _point_in_convex_polygon_xy(point[:2], hull_xy):
        return point

    best_xy = None
    best_distance = float("inf")
    for i in range(len(hull_xy)):
        a = hull_xy[i]
        b = hull_xy[(i + 1) % len(hull_xy)]
        candidate_xy = _project_point_to_segment_xy(point[:2], a, b)
        distance = float(np.linalg.norm(point[:2] - candidate_xy))
        if distance < best_distance:
            best_distance = distance
            best_xy = candidate_xy

    return np.array([best_xy[0], best_xy[1], point[2]], dtype=float)


def _top_surface_hull_xy(
    surface_body: Body, upward_threshold: float = 0.6
) -> Optional[np.ndarray]:
    """
    :param surface_body: The body whose top surface is computed.
    :param upward_threshold: Minimum Z component of a face normal for the face to
        count as upward-facing.
    :return: The convex hull in XY of the body's upward-facing mesh faces, or None if
        the body has no usable mesh.
    """
    mesh = surface_body.combined_mesh
    if mesh is None or mesh.is_empty:
        return None

    normals = np.asarray(mesh.face_normals, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(normals) == 0 or len(faces) == 0 or len(vertices) == 0:
        return None

    upward_mask = normals[:, 2] > float(upward_threshold)
    if not upward_mask.any():
        return None

    upward_faces = faces[upward_mask]
    top_vertices = vertices[np.unique(upward_faces.reshape(-1))]
    if len(top_vertices) < 3:
        return None

    return _convex_hull_xy(top_vertices[:, :2])


def _duration_scale_from_body(
    body: Body, reference_size: float = 0.10, use_visual_aabb: bool = False
) -> float:
    """
    :param body: The body the motion acts on.
    :param reference_size: Reference size in meters the diagonal is compared against.
    :param use_visual_aabb: Use the visual geometry instead of the collision geometry.
    :return: A duration scaling factor derived from the body's bounding box diagonal
        relative to a reference size.
    """
    bounding_box = _local_bounding_box(body, use_visual=use_visual_aabb)
    diagonal = float(
        np.linalg.norm([bounding_box.depth, bounding_box.width, bounding_box.height])
    )
    reference = float(reference_size)
    if reference <= 0.0:
        raise ValueError("reference_size must be positive")
    return max(diagonal, 1e-6) / reference


def build_container_path(
    container_body: Body,
    pattern: MixingPattern = MixingPattern.SPIRAL,
    mix_duration: Optional[float] = None,
    reference_size: float = 0.10,
    approach_clearance: float = 0.02,
) -> ToolPath:
    """
    Build a mixing tool path sized to a container-like object.

    :param container_body: The container whose contents are mixed.
    :param pattern: The mixing pattern to use.
    :param mix_duration: Total duration in seconds of the stirring pattern; only used
        for :attr:`MixingPattern.STIR`.
    :param reference_size: Reference size in meters that scales the motion duration.
    :param approach_clearance: Extra vertical clearance in meters when approaching the
        container from above.
    :return: The mixing tool path in the container's frame.
    """
    bounding_box = _local_bounding_box(container_body)
    radius_xy = 0.5 * min(bounding_box.depth, bounding_box.width)
    z_min, z_max = bounding_box.min_z, bounding_box.max_z

    center_y = 0.5 * (bounding_box.min_y + bounding_box.max_y)
    surface_margin = 0.005
    start_offset = np.array(
        [0.0, center_y, z_max + approach_clearance - surface_margin],
        dtype=float,
    )
    duration_scale = _duration_scale_from_body(
        container_body, reference_size=reference_size
    )
    spiral_r1 = 0.75 * radius_xy

    def _container_constraint(q_local: np.ndarray) -> np.ndarray:
        """
        :param q_local: The local 3D point to constrain.
        :return: The point clamped to the inside of the container.
        """
        return clamp_to_cylinder_xy(
            q_local, radius=radius_xy, z_min=z_min, z_max=z_max, margin=0.005
        )

    def _with_offset(curve: LocalCurve) -> LocalCurve:
        """
        :param curve: The curve to shift and constrain.
        :return: The curve shifted to the start offset and kept inside the container.
        """
        return make_constrained_curve(
            lambda tau: curve(tau) + start_offset, _container_constraint
        )

    if pattern is MixingPattern.SPIRAL:
        return ToolPath(
            [
                ToolPathSegment(
                    kind=ToolPathSegmentKind.SPIRAL,
                    duration=2.0 * duration_scale,
                    local_curve=_with_offset(
                        lambda tau: planar_spiral_xy(
                            tau, r0=0.00, r1=spiral_r1, cycles=2.0
                        )
                    ),
                )
            ]
        )

    stir_amplitude = max(0.005, 0.55 * radius_xy)
    stir_base_duration = max(1.0, 2.0 * duration_scale)
    if mix_duration is not None and float(mix_duration) > 0.0:
        total_duration = float(mix_duration)
    else:
        total_duration = stir_base_duration
    stir_loops = max(1, int(np.ceil(total_duration / stir_base_duration)))

    return ToolPath(
        [
            ToolPathSegment(
                kind=ToolPathSegmentKind.STIR,
                duration=total_duration,
                local_curve=_with_offset(
                    lambda tau: oscillatory_shear_xy_profiled(
                        tau,
                        ShearXYProfile(
                            shear_amp=stir_amplitude,
                            shear_cycles=stir_loops,
                        ),
                    )
                ),
            )
        ]
    )


def build_surface_path(
    surface_body: Body,
    technique: WipingTechnique = WipingTechnique.WIPE,
    reference_size: float = 0.10,
    approach_clearance: float = 0.02,
) -> ToolPath:
    """
    Build a planar wiping tool path on a surface.

    :param surface_body: The surface to act on.
    :param technique: The wiping technique to use.
    :param reference_size: Reference size in meters that scales the motion duration.
    :param approach_clearance: Extra vertical clearance in meters when approaching the
        surface from above.
    :return: The wiping tool path in the surface's frame.
    """
    bounding_box = _local_bounding_box(surface_body, use_visual=True)
    size_x = bounding_box.depth
    size_y = bounding_box.width
    radius_xy = 0.45 * min(size_x, size_y)

    center_x = 0.5 * (bounding_box.min_x + bounding_box.max_x)
    center_y = 0.5 * (bounding_box.min_y + bounding_box.max_y)
    surface_margin = 0.005
    start_offset = np.array(
        [
            center_x,
            center_y,
            bounding_box.max_z + approach_clearance - surface_margin,
        ],
        dtype=float,
    )
    duration_scale = _duration_scale_from_body(
        surface_body, reference_size=reference_size, use_visual_aabb=True
    )
    spiral_r1 = 0.9 * radius_xy
    shear_amplitude = 0.35 * radius_xy
    raster_width = 0.9 * size_x
    raster_height = 0.9 * size_y
    lane_spacing = max(0.02, 0.25 * float(reference_size))
    raster_lanes = max(4, int(np.ceil(raster_height / max(lane_spacing, 1e-6))))

    hull_xy = _top_surface_hull_xy(surface_body)

    def _with_hull_constraint(curve: LocalCurve) -> LocalCurve:
        """
        :param curve: The curve to shift and constrain.
        :return: The curve shifted to the start offset and kept on the surface.
        """
        if hull_xy is None:
            return lambda tau: curve(tau) + start_offset
        return make_constrained_curve(
            lambda tau: curve(tau) + start_offset,
            lambda q_local: _constrain_to_convex_hull_xy(q_local, hull_xy),
        )

    if technique is WipingTechnique.WIPE:
        return ToolPath(
            [
                ToolPathSegment(
                    kind=ToolPathSegmentKind.SPIRAL,
                    duration=2.0 * duration_scale,
                    local_curve=_with_hull_constraint(
                        lambda tau: planar_spiral_xy(
                            tau, r0=0.00, r1=spiral_r1, cycles=2.0
                        )
                    ),
                )
            ]
        )
    if technique is WipingTechnique.SHEAR:
        return ToolPath(
            [
                ToolPathSegment(
                    kind=ToolPathSegmentKind.SHEAR,
                    duration=1.5 * duration_scale,
                    local_curve=_with_hull_constraint(
                        lambda tau: oscillatory_shear_xy_profiled(
                            tau,
                            ShearXYProfile(
                                shear_amp=shear_amplitude,
                                shear_cycles=5.0,
                            ),
                        )
                    ),
                )
            ]
        )
    return ToolPath(
        [
            ToolPathSegment(
                kind=ToolPathSegmentKind.RASTER,
                duration=2.0 * duration_scale,
                local_curve=_with_hull_constraint(
                    lambda tau: planar_raster_xy(
                        tau,
                        width=raster_width,
                        height=raster_height,
                        lanes=raster_lanes,
                    )
                ),
            )
        ]
    )


@dataclass
class SliceAnchorPlacement:
    """
    Places cut anchors along an axis interval from a requested slice thickness and
    number of cuts.

    Anchors are spaced by the slice thickness, starting one thickness after the
    interval start, so every slice between two neighbouring cuts (and before the first
    cut) has the requested thickness. A parameter left as ``None`` is derived from the
    other one and the interval length.

    .. note:: If both parameters are requested but do not fit the interval together,
        :attr:`priority` decides which one is kept while the other is adjusted.
    """

    interval_start: float
    """
    Lower bound of the usable cutting interval in meters.
    """

    interval_end: float
    """
    Upper bound of the usable cutting interval in meters.
    """

    slice_thickness: Optional[float] = None
    """
    Requested thickness of each slice in meters.

    Derived from :attr:`number_of_cuts` if None.
    """

    number_of_cuts: Optional[int] = None
    """
    Requested number of cuts.

    Derived from :attr:`slice_thickness` if None.
    """

    priority: SlicingPriority = SlicingPriority.THICKNESS
    """
    Parameter that is kept when the requested thickness and number of cuts conflict.
    """

    default_slice_thickness: float = 0.03
    """
    Slice thickness in meters used when neither parameter is requested.
    """

    minimum_slice_thickness: float = 1e-4
    """
    Lower bound on the slice thickness in meters to keep anchors distinct.
    """

    def compute_anchor_positions(self) -> List[float]:
        """
        :return: The anchor positions along the axis, ordered from interval start to
            interval end.
        """
        usable_length = max(0.0, self.interval_end - self.interval_start)
        thickness, count = self._resolve_parameters(usable_length)
        return [
            self.interval_start + cut_index * thickness
            for cut_index in range(1, count + 1)
        ]

    def _resolve_parameters(self, usable_length: float) -> Tuple[float, int]:
        """
        :param usable_length: Length of the usable cutting interval in meters.
        :return: The feasible ``(slice_thickness, number_of_cuts)`` pair after filling
            in missing parameters and applying :attr:`priority` on conflicts.
        """
        thickness = self.slice_thickness
        count = self.number_of_cuts
        if thickness is None and count is None:
            thickness = self.default_slice_thickness
            count = 1
        if count is None:
            count = int(usable_length / max(thickness, self.minimum_slice_thickness))
        if thickness is None:
            thickness = usable_length / max(count, 1)
        thickness = max(float(thickness), self.minimum_slice_thickness)
        count = max(1, int(count))
        if count * thickness > usable_length + 1e-9:
            if self.priority is SlicingPriority.THICKNESS:
                count = max(1, int(usable_length / thickness))
            else:
                thickness = usable_length / count
        thickness = min(thickness, usable_length)
        return thickness, count


def build_cutting_path(
    food_body: Body,
    technique: CuttingTechnique = CuttingTechnique.SAW,
    slice_thickness: Optional[float] = None,
    number_of_cuts_on_local_x_axis: Optional[int] = None,
    reference_size: float = 0.10,
    slicing_priority: SlicingPriority = SlicingPriority.THICKNESS,
    approach_clearance: float = 0.02,
) -> ToolPath:
    """
    Build a cutting tool path sized to a food object.

    :param food_body: The object to cut.
    :param technique: The cutting technique to use.
    :param slice_thickness: Thickness of each slice in meters. Derived from
        ``number_of_cuts_on_local_x_axis`` if None.
    :param number_of_cuts_on_local_x_axis: Number of cuts along the object's local X
        axis. Derived from ``slice_thickness`` if None.
    :param reference_size: Reference size in meters that scales the motion duration.
    :param slicing_priority: Parameter that is kept when ``slice_thickness`` and
        ``number_of_cuts_on_local_x_axis`` do not both fit the object.
    :param approach_clearance: Extra vertical clearance in meters when approaching the
        object from above.
    :return: The cutting tool path in the food object's frame.
    """
    bounding_box = _local_bounding_box(food_body, use_visual=True)
    size_x = bounding_box.depth
    size_y = bounding_box.width
    size_z = bounding_box.height
    z_surface = bounding_box.max_z

    duration_scale = _duration_scale_from_body(food_body, reference_size=reference_size)

    margin_x = min(0.01, 0.15 * size_x)
    margin_y = min(0.01, 0.10 * size_y)
    z_clearance = max(0.01, 0.25 * size_z) + approach_clearance
    z_top = z_surface + z_clearance
    cut_floor_clearance = max(0.015, 0.20 * size_z)
    cut_floor_clearance = min(cut_floor_clearance, max(0.003, size_z - 0.005))
    z_cut = bounding_box.min_z + cut_floor_clearance
    center_y = 0.5 * (bounding_box.min_y + bounding_box.max_y)

    if technique is CuttingTechnique.HALVING:
        x_anchors = [0.5 * (bounding_box.min_x + bounding_box.max_x)]
        z_cut = 0.5 * (bounding_box.min_z + bounding_box.max_z)
    else:
        x_anchors = SliceAnchorPlacement(
            interval_start=bounding_box.min_x + margin_x,
            interval_end=bounding_box.max_x - margin_x,
            slice_thickness=slice_thickness,
            number_of_cuts=number_of_cuts_on_local_x_axis,
            priority=slicing_priority,
        ).compute_anchor_positions()

    y_min = bounding_box.min_y + margin_y
    y_max = bounding_box.max_y - margin_y
    saw_cycles = max(2.0, round(size_y / max(reference_size, 1e-6)))
    shear_depth_max = z_surface - z_cut
    shear_amplitude = 0.5 * max(y_max - y_min, 0.0)

    def _approach_segment(cut_index: int, x_value: float) -> ToolPathSegment:
        """
        :param cut_index: Index of the cut the segment belongs to.
        :param x_value: X position of the cut in the object's frame.
        :return: A segment approaching the object's surface from above.
        """
        return ToolPathSegment(
            kind=ToolPathSegmentKind.APPROACH,
            duration=0.8 * duration_scale,
            local_curve=lambda tau, x=x_value: np.array(
                [x, center_y, z_top + (z_surface - z_top) * float(tau)], dtype=float
            ),
            cut_index=cut_index,
        )

    def _retract_segment(cut_index: int, x_value: float) -> ToolPathSegment:
        """
        :param cut_index: Index of the cut the segment belongs to.
        :param x_value: X position of the cut in the object's frame.
        :return: A segment retracting from the cut depth back above the object.
        """
        return ToolPathSegment(
            kind=ToolPathSegmentKind.RETRACT,
            duration=0.8 * duration_scale,
            local_curve=lambda tau, x=x_value: np.array(
                [x, center_y, z_cut + (z_top - z_cut) * float(tau)], dtype=float
            ),
            cut_index=cut_index,
        )

    def _descend_segment(cut_index: int, x_value: float) -> ToolPathSegment:
        """
        :param cut_index: Index of the cut the segment belongs to.
        :param x_value: X position of the cut in the object's frame.
        :return: A segment pressing straight down to the cut depth.
        """
        return ToolPathSegment(
            kind=ToolPathSegmentKind.DESCEND,
            duration=1.0 * duration_scale,
            local_curve=lambda tau, x=x_value: np.array(
                [x, center_y, z_surface + (z_cut - z_surface) * float(tau)],
                dtype=float,
            ),
            cut_index=cut_index,
        )

    def _saw_segment(cut_index: int, x_value: float) -> ToolPathSegment:
        """
        :param cut_index: Index of the cut the segment belongs to.
        :param x_value: X position of the cut in the object's frame.
        :return: A segment sawing down to the cut depth with a shear oscillation.
        """

        def _saw_curve(tau: float, x: float = x_value) -> np.ndarray:
            """
            :param tau: Normalized time in [0, 1].
            :param x: X position of the cut in the object's frame.
            :return: A point on the sawing curve.
            """
            shear_point = oscillatory_shear_local_profiled(
                tau,
                ShearProfile(
                    depth_max=shear_depth_max,
                    depth_ramp_end=0.7,
                    shear_amp=shear_amplitude,
                    shear_cycles=saw_cycles,
                ),
            )
            return np.array(
                [x, center_y + shear_point[0], z_surface + shear_point[2]],
                dtype=float,
            )

        return ToolPathSegment(
            kind=ToolPathSegmentKind.SAW,
            duration=5.0 * duration_scale,
            local_curve=_saw_curve,
            cut_index=cut_index,
        )

    segments: List[ToolPathSegment] = []
    for cut_index, x_value in enumerate(x_anchors):
        segments.append(_approach_segment(cut_index, x_value))
        if technique is CuttingTechnique.SAW:
            segments.append(_saw_segment(cut_index, x_value))
        else:
            segments.append(_descend_segment(cut_index, x_value))
        segments.append(_retract_segment(cut_index, x_value))

    return ToolPath(segments)
