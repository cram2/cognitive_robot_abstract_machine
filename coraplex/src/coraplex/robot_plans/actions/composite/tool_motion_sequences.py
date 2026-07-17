from __future__ import annotations

import numpy as np
from typing_extensions import Callable, List, Optional, Tuple

from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.datastructures.enums import (
    CuttingTechnique,
    MixingPattern,
    SlicingPriority,
    WipingTechnique,
)
from dataclasses import dataclass
from typing_extensions import ClassVar

DEFAULT_SAMPLE_DT = 0.01
"""
Default sampling period in seconds for motion sequences.
"""

APPROACH_Z_EXTRA_CLEARANCE_M = 0.02
"""
Extra vertical clearance in meters when approaching a surface from above.
"""

LocalCurve = Callable[[float], np.ndarray]
"""
A curve mapping a normalized time in [0, 1] to a local 3D point.
"""


def body_local_aabb(
    body: Body, use_visual: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param body: The body whose bounding box is computed.
    :param use_visual: Use the visual geometry instead of the collision geometry.
    :return: The local axis-aligned bounding box ``(mins, maxs)`` of the body in its
        own frame.
    """
    geometry = body.visual if use_visual else body.collision
    bounding_box = geometry.as_bounding_box_collection_in_frame(body).bounding_box()
    mins = np.array(
        [bounding_box.min_x, bounding_box.min_y, bounding_box.min_z], dtype=float
    )
    maxs = np.array(
        [bounding_box.max_x, bounding_box.max_y, bounding_box.max_z], dtype=float
    )
    return mins, maxs


class MotionSegment:
    """
    A local motion curve over a fixed duration.
    """

    def __init__(self, name: str, duration_s: float, local_curve: LocalCurve):
        self.name = str(name)
        """
        Name of the segment.
        """
        self.duration_s = float(duration_s)
        """
        Duration of the segment in seconds.
        """
        self.local_curve = local_curve
        """
        Curve mapping normalized time to a local 3D point.
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
        number_of_samples = max(2, int(np.ceil(self.duration_s / float(dt))) + 1)
        times = np.linspace(t0, t0 + self.duration_s, number_of_samples)
        tau = (times - t0) / self.duration_s

        frame = np.asarray(frame_np, dtype=float)
        rotation = frame[:3, :3]
        translation = frame[:3, 3]

        local_points = np.array(
            [self.local_curve(float(u)) for u in tau], dtype=float
        ).reshape(-1, 3)
        points = local_points @ rotation.T + translation

        return times, points


class MotionSequence:
    """
    An ordered list of motion segments forming one continuous motion.
    """

    def __init__(self, phases: List[MotionSegment]):
        self.phases = list(phases)
        """
        The ordered motion segments of this sequence.
        """

    @property
    def duration_s(self) -> float:
        """
        :return: Total duration across all phases in seconds.
        """
        return float(sum(phase.duration_s for phase in self.phases))

    def sample(
        self, frame: HomogeneousTransformationMatrix, dt: float, t0: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample all phases into one concatenated sequence.

        :param frame: The frame the sequence's curves are expressed in.
        :param dt: Sampling period in seconds.
        :param t0: Start time of the sequence.
        :return: Sample times, sampled points as an Nx3 array, and the phase index of
            every sample.
        """
        all_times, all_points, all_phase_ids = [], [], []
        t = float(t0)

        for phase_index, phase in enumerate(self.phases):
            times, points = phase.sample(frame.to_np(), dt=dt, t0=t)
            if all_times:
                times = times[1:]
                points = points[1:]

            all_times.append(times)
            all_points.append(points)
            all_phase_ids.append(np.full(len(times), phase_index, dtype=int))
            t += phase.duration_s

        return (
            np.concatenate(all_times),
            np.vstack(all_points),
            np.concatenate(all_phase_ids),
        )


def ramp(tau: float, tau_end: float, d_max: float) -> float:
    """
    :return: A linear ramp from 0 to ``d_max`` over ``tau_end``.
    """
    if tau <= 0.0:
        return 0.0
    if tau >= tau_end:
        return float(d_max)
    return float(d_max) * (tau / tau_end)


def planar_spiral_xy(tau: float, r0: float, r1: float, cycles: float) -> np.ndarray:
    """
    :return: A point on a planar spiral in XY with linearly growing radius.
    """
    radius = r0 + (r1 - r0) * tau
    angle = 2.0 * np.pi * cycles * tau
    return np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0], dtype=float)


def planar_sweep_x(tau: float, length: float, cycles: float) -> np.ndarray:
    """
    :return: A point on a sinusoidal sweep along the X axis.
    """
    sweep = float(length) * np.sin(2.0 * np.pi * float(cycles) * tau)
    return np.array([sweep, 0.0, 0.0], dtype=float)


def planar_raster_xy(tau: float, width: float, height: float, lanes: int) -> np.ndarray:
    """
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
    :return: A point on an oscillatory shear curve with a monotone depth profile.
    """
    depth = ramp(tau, tau_end=profile.depth_ramp_end, d_max=profile.depth_max)
    shear = float(profile.shear_amp) * np.sin(
        2.0 * np.pi * float(profile.shear_cycles) * tau
    )
    return np.array([shear, 0.0, -depth], dtype=float)


def oscillatory_shear_xy_profiled(tau: float, profile: ShearXYProfile) -> np.ndarray:
    """
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
    :return: The curve wrapped so every point respects the constraint function.
    """
    return lambda tau: constraint_fn(local_curve(float(tau)))


def _cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    """
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
    :return: A duration scaling factor derived from the body's bounding box diagonal
        relative to a reference size.
    """
    mins, maxs = body_local_aabb(body, use_visual=use_visual_aabb)
    extents = maxs - mins
    diagonal = float(np.linalg.norm(extents))
    reference = float(reference_size)
    if reference <= 0.0:
        raise ValueError("reference_size must be positive")
    return max(diagonal, 1e-6) / reference


def build_container_sequence(
    container_body: Body,
    pattern: MixingPattern = MixingPattern.SPIRAL,
    mix_duration_s: Optional[float] = None,
    reference_size: float = 0.10,
) -> MotionSequence:
    """
    Build a mixing sequence sized to a container-like object.

    :param container_body: The container whose contents are mixed.
    :param pattern: The mixing pattern to use.
    :param mix_duration_s: Total duration of the stirring pattern; only used for
        :attr:`MixingPattern.STIR`.
    :param reference_size: Reference size in meters that scales the motion duration.
    :return: The mixing motion sequence in the container's frame.
    """
    mins, maxs = body_local_aabb(container_body)
    size_x = maxs[0] - mins[0]
    size_y = maxs[1] - mins[1]
    radius_xy = 0.5 * min(size_x, size_y)
    z_min, z_max = mins[2], maxs[2]

    center_y = 0.5 * (mins[1] + maxs[1])
    surface_margin = 0.005
    start_offset = np.array(
        [0.0, center_y, maxs[2] + APPROACH_Z_EXTRA_CLEARANCE_M - surface_margin],
        dtype=float,
    )
    duration_scale = _duration_scale_from_body(
        container_body, reference_size=reference_size
    )
    spiral_r1 = 0.75 * radius_xy

    def _container_constraint(q_local: np.ndarray) -> np.ndarray:
        return clamp_to_cylinder_xy(
            q_local, radius=radius_xy, z_min=z_min, z_max=z_max, margin=0.005
        )

    def _with_offset(curve: LocalCurve) -> LocalCurve:
        return make_constrained_curve(
            lambda tau: curve(tau) + start_offset, _container_constraint
        )

    if pattern is MixingPattern.SPIRAL:
        return MotionSequence(
            [
                MotionSegment(
                    name="planar_spiral_container",
                    duration_s=2.0 * duration_scale,
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
    if mix_duration_s is not None and float(mix_duration_s) > 0.0:
        total_duration = float(mix_duration_s)
    else:
        total_duration = stir_base_duration
    stir_loops = max(1, int(np.ceil(total_duration / stir_base_duration)))

    return MotionSequence(
        [
            MotionSegment(
                name="continuous_stir_container",
                duration_s=total_duration,
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


def build_surface_sequence(
    surface_body: Body,
    technique: WipingTechnique = WipingTechnique.WIPE,
    reference_size: float = 0.10,
) -> MotionSequence:
    """
    Build a planar wiping sequence on a surface.

    :param surface_body: The surface to act on.
    :param technique: The wiping technique to use.
    :param reference_size: Reference size in meters that scales the motion duration.
    :return: The wiping motion sequence in the surface's frame.
    """
    mins, maxs = body_local_aabb(surface_body, use_visual=True)
    size_x = maxs[0] - mins[0]
    size_y = maxs[1] - mins[1]
    radius_xy = 0.45 * min(size_x, size_y)

    center_x = 0.5 * (mins[0] + maxs[0])
    center_y = 0.5 * (mins[1] + maxs[1])
    surface_margin = 0.005
    start_offset = np.array(
        [
            center_x,
            center_y,
            maxs[2] + APPROACH_Z_EXTRA_CLEARANCE_M - surface_margin,
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
        if hull_xy is None:
            return lambda tau: curve(tau) + start_offset
        return make_constrained_curve(
            lambda tau: curve(tau) + start_offset,
            lambda q_local: _constrain_to_convex_hull_xy(q_local, hull_xy),
        )

    if technique is WipingTechnique.WIPE:
        return MotionSequence(
            [
                MotionSegment(
                    name="planar_spiral_surface",
                    duration_s=2.0 * duration_scale,
                    local_curve=_with_hull_constraint(
                        lambda tau: planar_spiral_xy(
                            tau, r0=0.00, r1=spiral_r1, cycles=2.0
                        )
                    ),
                )
            ]
        )
    if technique is WipingTechnique.SHEAR:
        return MotionSequence(
            [
                MotionSegment(
                    name="oscillatory_shear_surface",
                    duration_s=1.5 * duration_scale,
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
    return MotionSequence(
        [
            MotionSegment(
                name="planar_raster_surface",
                duration_s=2.0 * duration_scale,
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
    Requested thickness of each slice in meters. Derived from :attr:`number_of_cuts`
    if None.
    """
    number_of_cuts: Optional[int] = None
    """
    Requested number of cuts. Derived from :attr:`slice_thickness` if None.
    """
    priority: SlicingPriority = SlicingPriority.THICKNESS
    """
    Parameter that is kept when the requested thickness and number of cuts conflict.
    """

    DEFAULT_SLICE_THICKNESS: ClassVar[float] = 0.03
    """
    Slice thickness in meters used when neither parameter is requested.
    """

    MINIMUM_SLICE_THICKNESS: ClassVar[float] = 1e-4
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
            thickness = self.DEFAULT_SLICE_THICKNESS
            count = 1
        if count is None:
            count = int(usable_length / max(thickness, self.MINIMUM_SLICE_THICKNESS))
        if thickness is None:
            thickness = usable_length / max(count, 1)
        thickness = max(float(thickness), self.MINIMUM_SLICE_THICKNESS)
        count = max(1, int(count))
        if count * thickness > usable_length + 1e-9:
            if self.priority is SlicingPriority.THICKNESS:
                count = max(1, int(usable_length / thickness))
            else:
                thickness = usable_length / count
        thickness = min(thickness, usable_length)
        return thickness, count


def build_cutting_sequence(
    food_body: Body,
    technique: CuttingTechnique = CuttingTechnique.SAW,
    slice_thickness: Optional[float] = None,
    num_cuts_x: Optional[int] = None,
    reference_size: float = 0.10,
    slicing_priority: SlicingPriority = SlicingPriority.THICKNESS,
) -> MotionSequence:
    """
    Build a cutting sequence sized to a food object.

    :param food_body: The object to cut.
    :param technique: The cutting technique to use.
    :param slice_thickness: Thickness of each slice in meters. Derived from
        ``num_cuts_x`` if None.
    :param num_cuts_x: Number of cuts along the object's X axis. Derived from
        ``slice_thickness`` if None.
    :param reference_size: Reference size in meters that scales the motion duration.
    :param slicing_priority: Parameter that is kept when ``slice_thickness`` and
        ``num_cuts_x`` do not both fit the object.
    :return: The cutting motion sequence in the food object's frame.
    """
    mins, maxs = body_local_aabb(food_body, use_visual=True)
    size_x = maxs[0] - mins[0]
    size_y = maxs[1] - mins[1]
    size_z = maxs[2] - mins[2]

    duration_scale = _duration_scale_from_body(food_body, reference_size=reference_size)

    margin_x = min(0.01, 0.15 * size_x)
    margin_y = min(0.01, 0.10 * size_y)
    z_clearance = max(0.01, 0.25 * size_z) + APPROACH_Z_EXTRA_CLEARANCE_M
    z_top = maxs[2] + z_clearance
    cut_floor_clearance = max(0.015, 0.20 * size_z)
    cut_floor_clearance = min(cut_floor_clearance, max(0.003, size_z - 0.005))
    z_cut = mins[2] + cut_floor_clearance
    center_y = 0.5 * (mins[1] + maxs[1])

    if technique is CuttingTechnique.HALVING:
        x_anchors = [0.5 * (mins[0] + maxs[0])]
        z_cut = 0.5 * (mins[2] + maxs[2])
    else:
        x_anchors = SliceAnchorPlacement(
            interval_start=mins[0] + margin_x,
            interval_end=maxs[0] - margin_x,
            slice_thickness=slice_thickness,
            number_of_cuts=num_cuts_x,
            priority=slicing_priority,
        ).compute_anchor_positions()

    y_min = mins[1] + margin_y
    y_max = maxs[1] - margin_y
    saw_cycles = max(2.0, round(size_y / max(reference_size, 1e-6)))
    shear_depth_max = maxs[2] - z_cut
    shear_amplitude = 0.5 * max(y_max - y_min, 0.0)

    def _approach_segment(cut_index: int, x_val: float) -> MotionSegment:
        return MotionSegment(
            name=f"cut_approach_x{cut_index}",
            duration_s=0.8 * duration_scale,
            local_curve=lambda tau, x=x_val: np.array(
                [x, center_y, z_top + (maxs[2] - z_top) * float(tau)], dtype=float
            ),
        )

    def _retract_segment(cut_index: int, x_val: float) -> MotionSegment:
        return MotionSegment(
            name=f"cut_retract_x{cut_index}",
            duration_s=0.8 * duration_scale,
            local_curve=lambda tau, x=x_val: np.array(
                [x, center_y, z_cut + (z_top - z_cut) * float(tau)], dtype=float
            ),
        )

    def _descend_segment(cut_index: int, x_val: float) -> MotionSegment:
        return MotionSegment(
            name=f"cut_descend_x{cut_index}",
            duration_s=1.0 * duration_scale,
            local_curve=lambda tau, x=x_val: np.array(
                [x, center_y, maxs[2] + (z_cut - maxs[2]) * float(tau)], dtype=float
            ),
        )

    def _saw_segment(cut_index: int, x_val: float) -> MotionSegment:
        def _saw_curve(tau: float, x: float = x_val) -> np.ndarray:
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
                [x, center_y + shear_point[0], maxs[2] + shear_point[2]], dtype=float
            )

        return MotionSegment(
            name=f"oscillatory_shear_x{cut_index}",
            duration_s=5.0 * duration_scale,
            local_curve=_saw_curve,
        )

    phases: List[MotionSegment] = []
    for cut_index, x_val in enumerate(x_anchors):
        phases.append(_approach_segment(cut_index, x_val))
        if technique is CuttingTechnique.SAW:
            phases.append(_saw_segment(cut_index, x_val))
        else:
            phases.append(_descend_segment(cut_index, x_val))
        phases.append(_retract_segment(cut_index, x_val))

    return MotionSequence(phases)
