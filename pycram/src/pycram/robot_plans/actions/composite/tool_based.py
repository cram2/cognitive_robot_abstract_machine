from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from typing_extensions import Optional, Any
from visualization_msgs.msg import MarkerArray

from pycram.robot_plans.actions.composite.thesis_math.metrics import (
    points_world_to_body,
    cutting_depth_metrics,
    mixing_bowl_metrics,
)
from pycram.robot_plans.actions.composite.thesis_math.motion_models import (
    MotionSegment,
    MotionSequence,
)


from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body
from .thesis_math.motion_presets import (
    build_container_sequence,
    build_surface_sequence,
    build_cutting_sequence,
    build_pouring_sequence,
)
from .thesis_math.motion_profiles import planar_spiral_xy, planar_sweep_x
from .thesis_math.world_utils import body_local_aabb
from ... import (
    MoveTCPWaypointsAlignedMotion,
    MoveTCPWaypointsMotion,
)

from ....datastructures.enums import (
    Arms,
    MovementType,
)
from ....plans.factories import sequential
from ....robot_plans.actions.base import ActionDescription
from ....robot_plans.motions.gripper import MoveToolCenterPointMotion
from ....tf_transformations import quaternion_from_matrix
from ....view_manager import ViewManager

logger = logging.getLogger(__name__)
DEFAULT_SAMPLE_DT = 0.01

_MARKER_GROUP_COUNTER = 0


def _norm_topic(topic: str) -> str:
    """Ensure the topic name starts with a slash."""
    t = str(topic)
    return t if t.startswith("/") else "/" + t


def _color(r, g, b, a=1.0):
    """Create a ColorRGBA message."""
    c = ColorRGBA()
    c.r = float(r)
    c.g = float(g)
    c.b = float(b)
    c.a = float(a)
    return c


def _as_points(P):
    """Convert an array of points into a list of ROS Point messages."""
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    pts = []
    for i in range(P.shape[0]):
        p = Point()
        p.x = float(P[i, 0])
        p.y = float(P[i, 1])
        p.z = float(P[i, 2])
        pts.append(p)
    return pts


def _phase_color(k, a=1.0):
    """Pick a color from a fixed palette for a phase index."""
    palette = [
        (1.0, 0.2, 0.2),
        (0.2, 1.0, 0.2),
        (0.2, 0.2, 1.0),
        (1.0, 0.8, 0.2),
        (0.8, 0.2, 1.0),
        (0.2, 1.0, 1.0),
    ]
    r, g, b = palette[int(k) % len(palette)]
    return _color(r, g, b, a)


def _heat_color(value, alpha=0.85):
    """Map a normalized scalar in [0, 1] to a readable heatmap color."""
    v = float(np.clip(value, 0.0, 1.0))
    if v <= 0.25:
        t = v / 0.25
        return _color(0.10, 0.25 + 0.55 * t, 0.95, alpha)
    if v <= 0.50:
        t = (v - 0.25) / 0.25
        return _color(0.10, 0.80 + 0.15 * t, 0.95 - 0.55 * t, alpha)
    if v <= 0.75:
        t = (v - 0.50) / 0.25
        return _color(0.10 + 0.90 * t, 0.95, 0.40 - 0.30 * t, alpha)
    t = (v - 0.75) / 0.25
    return _color(1.00, 0.95 - 0.70 * t, 0.10, alpha)


def _quaternion_to_rotation_matrix(quaternion):
    """Convert an xyzw quaternion to a 3x3 rotation matrix."""
    x, y, z, w = np.asarray(quaternion, dtype=float)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _transform_point(position, orientation, local_point):
    """Apply a pose defined by position and quaternion to a local 3D point."""
    position_vec = np.asarray(position, dtype=float).reshape(-1)
    if position_vec.size == 4:
        position_vec = position_vec[:3]
    elif position_vec.size != 3:
        raise ValueError(
            f"position must contain 3 values (xyz) or 4 homogeneous values, got {position_vec.size}"
        )

    local_point_vec = np.asarray(local_point, dtype=float).reshape(-1)
    if local_point_vec.size != 3:
        raise ValueError(
            f"local_point must contain exactly 3 values, got {local_point_vec.size}"
        )

    world_point = (
        position_vec + _quaternion_to_rotation_matrix(orientation) @ local_point_vec
    )
    point = Point()
    point.x = float(world_point[0])
    point.y = float(world_point[1])
    point.z = float(world_point[2])
    return point


def _pose_to_position_and_orientation_lists(pose):
    """Extract xyz and xyzw lists from pose-like objects with either old or new APIs."""
    if pose is None:
        raise ValueError("pose must not be None")

    position = getattr(pose, "position", None)
    if position is None and hasattr(pose, "to_position"):
        position = pose.to_position()

    orientation = getattr(pose, "orientation", None)
    if orientation is None and hasattr(pose, "to_quaternion"):
        orientation = pose.to_quaternion()

    if position is None or orientation is None:
        raise AttributeError(
            "pose must provide position/orientation attributes or "
            "to_position()/to_quaternion() methods"
        )
    if hasattr(position, "to_list"):
        position_values = position.to_list()
    else:
        position_values = [
            getattr(position, axis)
            for axis in ("x", "y", "z")
            if hasattr(position, axis)
        ]

    if hasattr(orientation, "to_list"):
        orientation_values = orientation.to_list()
    else:
        orientation_values = [
            getattr(orientation, axis)
            for axis in ("x", "y", "z", "w")
            if hasattr(orientation, axis)
        ]

    position_values = np.asarray(position_values, dtype=float).reshape(-1)
    orientation_values = np.asarray(orientation_values, dtype=float).reshape(-1)

    if position_values.size == 4:
        position_values = position_values[:3]
    elif position_values.size != 3:
        raise ValueError(
            f"pose position must contain 3 values (xyz) or 4 homogeneous values, got {position_values.size}"
        )

    if orientation_values.size != 4:
        raise ValueError(
            f"pose orientation must contain 4 quaternion values (xyzw), got {orientation_values.size}"
        )

    return position_values.tolist(), orientation_values.tolist()


def _next_marker_group(prefix="rviz"):
    global _MARKER_GROUP_COUNTER
    _MARKER_GROUP_COUNTER += 1
    return f"{prefix}_{_MARKER_GROUP_COUNTER}"


class MotionSequenceRviz:
    def __init__(
        self,
        P,
        phase_id=None,
        frame_id="map",
        topic="phase_sequence_markers",
        node=None,
        line_width=0.01,
        alpha=1.0,
        republish_hz=2.0,
        marker_ns=None,
    ):
        """
        Initialization method of the MotionSequenceRviz class.

        This method initializes a MotionSequenceRviz object with attributes related
        to markers' visualization in RViz. It subscribes to a specific ROS topic,
        sets up a publisher for MarkerArray messages, and optionally starts a timer
        to regularly republish the markers. The input data for markers, including
        their positions, ids, and visualization properties (such as line width
        and transparency), are configured through the parameters.

        Raises
        ------
        ValueError
            If the `node` parameter is not provided because the MotionSequenceRviz
            class requires a ROS node to operate.

        Parameters
        ----------
        P : numpy.ndarray
            A NumPy array representing the set of 3D points for visualization.
            It must have a shape (-1, 3) indicating that it's a list of XYZ
            coordinates.
        phase_id : numpy.ndarray, optional
            An optional NumPy array of integers representing IDs associated with
            the phases of the points. Default is None.
        frame_id : str, optional
            The coordinate frame in which the markers are defined. Defaults to "map".
        topic : str, optional
            The ROS topic on which MarkerArray messages will be published. Defaults
            to "phase_sequence_markers".
        node : Any
            The ROS node used to create the MarkerArray publisher and other required
            functionality. This must be provided explicitly.
        line_width : float, optional
            The width of the lines between points, used for visualization. Defaults
            to 0.01.
        alpha : float, optional
            The transparency of the markers, defined between 0 (completely
            transparent) and 1 (fully opaque). Defaults to 1.0.
        republish_hz : float, optional
            Frequency in Hz for republishing markers to the topic. If None or set to
            a non-positive value, the markers are not republished. Defaults to 2.0.
        """
        self.P = np.asarray(P, dtype=float).reshape(-1, 3)
        self.phase_id = (
            None if phase_id is None else np.asarray(phase_id, dtype=int).reshape(-1)
        )
        self.frame_id = str(frame_id)
        self.topic = _norm_topic(topic)
        self.line_width = float(line_width)
        self.alpha = float(alpha)
        self.marker_ns = (
            str(marker_ns) if marker_ns is not None else _next_marker_group("phase_seq")
        )

        if node is None:
            raise ValueError(
                "node must be provided when using MotionSequenceRviz directly in your system"
            )

        self.node = node
        self.pub = self.node.create_publisher(MarkerArray, self.topic, 10)

        self.node.get_logger().info(
            f"Publishing MarkerArray on {self.topic} in frame '{self.frame_id}'"
        )

        self.publish_once()

        self._timer = None
        if republish_hz is not None and float(republish_hz) > 0.0:
            period = 1.0 / float(republish_hz)
            self._timer = self.node.create_timer(period, self.publish_once)

    def publish_once(self):
        """Publish the current sequence once."""
        now = self.node.get_clock().now().to_msg()
        arr = MarkerArray()

        if self.phase_id is None:
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = now
            m.ns = self.marker_ns
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = self.line_width
            m.color = _color(1.0, 1.0, 1.0, self.alpha)
            m.points = _as_points(self.P)
            arr.markers.append(m)
        else:
            start = 0
            mid = 0
            while start < self.P.shape[0]:
                pid = int(self.phase_id[start])
                end = start + 1
                while end < self.P.shape[0] and int(self.phase_id[end]) == pid:
                    end += 1

                m = Marker()
                m.header.frame_id = self.frame_id
                m.header.stamp = now
                m.ns = self.marker_ns
                m.id = mid
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.pose.orientation.w = 1.0
                m.scale.x = self.line_width
                m.color = _phase_color(pid, a=self.alpha)
                m.points = _as_points(self.P[start:end, :])
                arr.markers.append(m)

                mid += 1
                start = end

        self.pub.publish(arr)

def publish_points_sequence(
    node,
    points,
    frame_id="map",
    topic="points_sequence",
    line_width=0.01,
    color=(1.0, 1.0, 1.0),
    alpha=1.0,
    phase_id=None,
    republish_hz=None,
    clear_existing=False,
    marker_ns=None,
):
    """
    Publish a single LINE_STRIP marker for a sequence of points.

    Parameters
    ----------
    node : Any
        ROS node used to create the publisher.
    points : array-like
        Nx3 points in the given frame.
    frame_id : str
        Coordinate frame for the marker.
    topic : str
        Topic name for the MarkerArray.
    line_width : float
        Line width for the strip.
    color : tuple
        RGB tuple with values in [0, 1].
    alpha : float
        Transparency in [0, 1].
    phase_id : array-like, optional
        Per-point phase ids; consecutive equal ids are colored as segments.
    """
    if node is None:
        return None

    pts = _as_points(points)
    phase_id_arr = (
        None if phase_id is None else np.asarray(phase_id, dtype=int).reshape(-1)
    )
    topic = _norm_topic(topic)
    marker_ns = (
        str(marker_ns)
        if marker_ns is not None
        else _next_marker_group("points_sequence")
    )

    pub = node.create_publisher(MarkerArray, topic, 10)

    # Keep publisher (and optional timer) alive by storing on the node.
    if not hasattr(node, "_rviz_publishers"):
        node._rviz_publishers = []

    arr = MarkerArray()
    markers = []

    if phase_id_arr is None:
        m = Marker()
        m.header.frame_id = str(frame_id)
        m.ns = marker_ns
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = float(line_width)
        m.color = _color(
            float(color[0]), float(color[1]), float(color[2]), float(alpha)
        )
        m.points = pts
        markers.append(m)
    else:
        start = 0
        mid = 0
        while start < len(pts):
            pid = int(phase_id_arr[start])
            end = start + 1
            while end < len(pts) and int(phase_id_arr[end]) == pid:
                end += 1

            m = Marker()
            m.header.frame_id = str(frame_id)
            m.ns = marker_ns
            m.id = mid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = float(line_width)
            m.color = _phase_color(pid, a=float(alpha))
            m.points = pts[start:end]
            markers.append(m)

            mid += 1
            start = end

    if clear_existing:
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
    arr.markers.extend(markers)

    def _publish_once():
        now = node.get_clock().now().to_msg()
        for marker in markers:
            marker.header.stamp = now
        pub.publish(arr)

    _publish_once()

    timer = None
    if republish_hz is not None and float(republish_hz) > 0.0:
        period = 1.0 / float(republish_hz)
        timer = node.create_timer(period, _publish_once)

    node._rviz_publishers.append((pub, timer))
    return pub


def _logging_helper_apply_fields(action: Any, fields: dict) -> None:
    for key, value in fields.items():
        setattr(action, key, value)


def _logging_helper_collect_identity_fields(action: Any) -> dict:
    tool_root_name = None
    if action.tool is not None and getattr(action.tool, "root", None) is not None:
        tool_root_name = str(getattr(action.tool.root, "name", action.tool.root))

    container_obj = getattr(action, "container", None)
    container_name = (
        "no container given"
        if container_obj is None
        else str(getattr(container_obj, "name", container_obj))
    )

    return {
        "logged_tool_root_name": tool_root_name,
        "logged_container_name": container_name,
    }


def _logging_helper_collect_target_intersection_fields(
    action: Any, points_world: np.ndarray
) -> dict:
    container = getattr(action, "container", None)
    if container is None:
        return {}

    points_body = points_world_to_body(points_world, action.world, container)
    mins, maxs = body_local_aabb(container, use_visual=False, apply_shape_scale=True)
    inside = (
        (points_body[:, 0] >= mins[0])
        & (points_body[:, 0] <= maxs[0])
        & (points_body[:, 1] >= mins[1])
        & (points_body[:, 1] <= maxs[1])
        & (points_body[:, 2] >= mins[2])
        & (points_body[:, 2] <= maxs[2])
    )
    inside_ratio = float(np.mean(inside)) if len(inside) > 0 else 0.0
    return {"logged_target_intersection_success": bool(inside_ratio >= 0.5)}


def _logging_helper_collect_cutting_fields(
    action: Any, points_world: np.ndarray
) -> dict:
    if (
        action.__class__.__name__ != "CuttingAction"
        or getattr(action, "container", None) is None
    ):
        return {}

    metrics = cutting_depth_metrics(
        points_world=points_world,
        world=action.world,
        bread_body=action.container,
        apply_shape_scale=True,
    )
    return {
        "has_entry_from_above": metrics.get("has_entry_from_above"),
    }


def _logging_helper_collect_mixing_fields(
    action: Any, points_world: np.ndarray
) -> dict:
    if (
        action.__class__.__name__ != "MixingAction"
        or getattr(action, "container", None) is None
    ):
        return {}

    metrics = mixing_bowl_metrics(
        points_world=points_world,
        world=action.world,
        bowl_body=action.container,
        apply_shape_scale=True,
    )
    return {"mixing_bowl_metrics": metrics}


def _logging_helper_collect_container_fields(
    action: Any, points_world: np.ndarray
) -> dict:
    fields = {}
    fields.update(
        _logging_helper_collect_target_intersection_fields(action, points_world)
    )
    fields.update(_logging_helper_collect_cutting_fields(action, points_world))
    fields.update(_logging_helper_collect_mixing_fields(action, points_world))
    return fields


@dataclass
class GeneralizedActionPlan(ActionDescription):
    """
    Base class for tool-based motion sequences over a container.
    """

    arm: Arms
    """
    Arm used for the motion.
    """

    tool: HasRootBody
    """
    Tool body used to estimate the tip offset.
    """

    clear_viz: bool = False
    """
    If viz should be cleared
    """

    pointer_stride: int = 1
    """
    Keep every Nth waypoint for execution (testing downsampling).
    """

    logged_target_intersection_success: Optional[bool] = None
    """
    Optional DB-logged boolean derived from target_intersection_success.
    """

    logged_tool_root_name: Optional[str] = None
    """
    Optional DB-logged tool root name captured during execute().
    """

    logged_container_name: Optional[str] = None
    """
    Optional DB-logged container name captured during execute().
    Uses "no container given" if no container is present.
    """

    logged_action_name: Optional[str] = None
    """
    Optional DB-logged concrete action class name captured during execute().
    """

    logged_technique: Optional[str] = None
    """
    Optional DB-logged technique value captured during execute() when available.
    """

    logged_waypoint_count: Optional[int] = None
    """
    Number of TCP waypoints generated for this action.
    """

    logged_stopped_waypoint_index: Optional[int] = None
    """
    Nearest waypoint index to the tool frame after execution or failure.
    """

    logged_stopped_waypoint_fraction: Optional[float] = None
    """
    Nearest waypoint index normalized by the final waypoint index.
    """

    logged_stopped_waypoint_x: Optional[float] = None
    logged_stopped_waypoint_y: Optional[float] = None
    logged_stopped_waypoint_z: Optional[float] = None
    logged_stopped_distance_m: Optional[float] = None
    logged_motion_progress_note: Optional[str] = None

    def execute(self) -> None:
        sampled = self._sample_motion()
        P = self._extract_points_for_logging(sampled)
        self.logged_action_name = self.__class__.__name__
        self.logged_technique = self._logged_technique_value()
        _logging_helper_apply_fields(
            self, _logging_helper_collect_identity_fields(self)
        )

        publish_points_sequence(
            node=self.plan.context.ros_node,
            points=P,
            frame_id="map",
            topic="/point_sequence",
            phase_id=self._extract_phase_ids_for_logging(sampled),
            republish_hz=2.0,
            clear_existing=self.clear_viz,
        )
        self.robot.full_body_controlled = True
        waypoints = self._build_waypoints(sampled)
        self._last_waypoints = waypoints
        self._postprocess_waypoints_for_logging(waypoints)
        self._update_waypoint_progress_for_logging(
            waypoints, note="planned_not_started"
        )

        _logging_helper_apply_fields(
            self, _logging_helper_collect_container_fields(self, P)
        )

        try:
            self.add_subplan(sequential([self._build_motion(waypoints)])).perform()
            self._update_waypoint_progress_for_logging(waypoints, note="completed")
        except Exception as exc:
            self._update_waypoint_progress_for_logging(waypoints, note="failed")
            if self._accept_execution_failure_as_success(waypoints, exc):
                self.logged_motion_progress_note = "accepted_failure_as_success"
                return

            collision_contacts = None
            try:
                collision_contacts = len(
                    self.world.collision_manager.compute_collisions().contacts
                )
            except Exception:
                collision_contacts = None

            msg = str(exc)
            if (
                "No waypoints provided to MoveTCPWaypointsAlignedMotion" in msg
                or "No aligned waypoint tasks generated" in msg
                or "No waypoints left after applying pointer_stride" in msg
                or "No pose waypoints left after applying pointer_stride" in msg
            ):
                raise ValueError(
                    "Aligned motion failed: no waypoint sequence to execute "
                    f"(waypoints={len(waypoints)}, collisions_now={collision_contacts})."
                ) from exc

            raise RuntimeError(
                "Aligned motion failed during execution "
                f"(waypoints={len(waypoints)}, collisions_now={collision_contacts}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _update_waypoint_progress_for_logging(self, waypoints, note: str) -> None:
        self.logged_waypoint_count = len(waypoints) if waypoints is not None else 0
        self.logged_motion_progress_note = note
        if not waypoints:
            return

        try:
            tool_xyz = self._current_tool_xyz()
            waypoint_xyz = np.asarray(
                [self._waypoint_xyz(waypoint) for waypoint in waypoints],
                dtype=float,
            )
            distances = np.linalg.norm(waypoint_xyz - tool_xyz, axis=1)
            stopped_index = int(np.argmin(distances))
            stopped_xyz = waypoint_xyz[stopped_index]
            final_index = max(1, len(waypoints) - 1)
            self.logged_stopped_waypoint_index = stopped_index
            self.logged_stopped_waypoint_fraction = float(stopped_index / final_index)
            self.logged_stopped_waypoint_x = float(stopped_xyz[0])
            self.logged_stopped_waypoint_y = float(stopped_xyz[1])
            self.logged_stopped_waypoint_z = float(stopped_xyz[2])
            self.logged_stopped_distance_m = float(distances[stopped_index])
        except Exception as exc:
            self.logged_motion_progress_note = (
                f"{note}:progress_lookup_failed:{type(exc).__name__}"
            )

    def _current_tool_xyz(self) -> np.ndarray:
        tool_frame = None
        if self.tool is not None:
            try:
                tool_frame = self.tool.get_tool_frame()
            except Exception:
                tool_frame = None
        if tool_frame is None:
            tool_frame = ViewManager().get_end_effector_view(
                self.arm, self.robot
            ).tool_frame
        tool_point = self.world.transform(
            tool_frame.global_pose.to_position(), self.world.root
        )
        return np.asarray(tool_point.to_np(), dtype=float).reshape(-1)[:3]

    def _waypoint_xyz(self, waypoint) -> np.ndarray:
        if hasattr(waypoint, "to_position"):
            waypoint = waypoint.to_position()
        world_point = self.world.transform(waypoint, self.world.root)
        return np.asarray(world_point.to_np(), dtype=float).reshape(-1)[:3]

    def _accept_execution_failure_as_success(self, waypoints, exc: Exception) -> bool:
        return False

    def _sample_motion(self):
        return self._sample_points()

    def _extract_points_for_logging(self, sampled) -> np.ndarray:
        _, points, _ = sampled
        return np.asarray(points, dtype=float)

    def _extract_phase_ids_for_logging(self, sampled) -> np.ndarray:
        _, _, ids = sampled
        return ids

    def _logged_technique_value(self):
        return getattr(self, "technique", None)

    def _build_waypoints(self, sampled):
        _, points, _ = sampled
        stride = max(1, int(self.pointer_stride))
        return self._to_waypoints(points, stride)

    def _postprocess_waypoints_for_logging(self, waypoints) -> None:
        if self.__class__.__name__ == "CuttingAction":
            self.db_debug_waypoint_count = float(len(waypoints))

    def _build_motion(self, waypoints):
        alignment_target = self._resolve_alignment_target()

        alignment_pairs = (
            self.tool.tool_alignment(alignment_target)
            if (self.tool is not None and alignment_target is not None)
            else []
        )
        try:
            tip = self.tool.get_tool_frame()
        except Exception:
            tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame

        return MoveTCPWaypointsAlignedMotion(
            waypoints,
            self.arm,
            allow_gripper_collision=True,
            alignment_pairs=alignment_pairs,
            tip=tip,
        )

    def _sample_points(selfs):
        raise NotImplementedError

    def _to_waypoints(self, points: np.ndarray, stride: int) -> list[Point3]:
        waypoints = [
            Point3(x=p[0], y=p[1], z=p[2], reference_frame=self.world.root)
            for p in points
        ][::stride]
        if not waypoints:
            raise ValueError("No waypoints left after applying pointer_stride.")
        return waypoints

    def _resolve_alignment_target(self):
        if hasattr(self, "surface_body") and self.surface_body is not None:
            return self.surface_body
        if hasattr(self, "container") and self.container is not None:
            return self.container
        if hasattr(self, "target_pose") and self.target_pose is not None:
            return self.target_pose
        return None


@dataclass
class MixingAction(GeneralizedActionPlan):
    """
    Execute a mixing motion sequence around a container.
    """

    motion_timeout_ticks = 300

    container: Body = None
    """
    The container (e.g., bowl) to operate in.
    """

    mix_duration_s: float = 0.0
    """
    Total mixing time in seconds for a continuous connected stir loop.
    If <= 0, the default short pattern is used.
    """

    mixing_bowl_metrics: Optional[dict] = None
    """
    Optional DB-logged metrics dict populated during execute().
    """

    def _sample_points(self):
        pattern = "stir" if float(self.mix_duration_s) > 0.0 else "spiral"
        seq = build_container_sequence(
            self.container,
            use_visual_aabb=True,
            apply_shape_scale=True,
            pattern=pattern,
            mix_duration_s=(
                self.mix_duration_s if float(self.mix_duration_s) > 0.0 else None
            ),
        )
        return seq.sample(frame=self.container.global_pose, dt=DEFAULT_SAMPLE_DT)


@dataclass(kw_only=True)
class WipingAction(GeneralizedActionPlan):
    """
    Execute a planar wiping motion around a target pose.
    """

    motion_timeout_ticks = 300

    final_waypoint_success_tolerance_m: float = 0.08
    """
    Accept a timeout as successful if the tool is already this close to the final wipe waypoint.
    """

    container: Optional[Body] = None
    """
    Optional alias for surface_body (backward compatibility).
    """
    target_pose: Optional[Pose] = None
    """
    Center pose for the wiping patch.
    """
    technique: str = "wipe"
    """
    Surface-contact technique variant.
    """
    length: float = 0.20
    """
    Sweep length for the wiping motion.
    """

    cycles: float = 1.0
    """
    Number of sweep cycles.
    """
    _resolved_surface_body: Optional[Body] = field(default=None, init=False, repr=False)
    _resolved_alignment_target: Optional[Body | Pose] = field(
        default=None, init=False, repr=False
    )

    def _target_pose_to_spatial(self):
        if self.target_pose is None:
            return None
        if getattr(self.target_pose, "frame_id", None) is None:
            logger.warning(
                "WipingAction target_pose has no frame_id; defaulting to world root."
            )
            self.target_pose.frame_id = self.world.root
        return self.target_pose

    def _resolved_technique(self) -> str:
        return str(self.technique).lower().strip()

    def _resolve_surface_body(self) -> Body:
        if self._resolved_surface_body is not None:
            return self._resolved_surface_body

        if self.container is None:
            if self.target_pose is None:
                raise ValueError(
                    "WipingAction requires either container or target_pose."
                )
            resolved = getattr(self.target_pose, "frame_id", None)
        else:
            resolved = self.container
        self._resolved_surface_body = resolved
        return resolved

    def _resolve_alignment_target(self):
        if self._resolved_alignment_target is not None:
            return self._resolved_alignment_target
        if self.container is None and self.target_pose is not None:
            self._resolved_alignment_target = self._resolve_pose_alignment_target()
            return self._resolved_alignment_target
        return self._resolve_surface_body()

    def _resolve_pose_alignment_target(self):
        return self.target_pose

    def _sample_points(self):
        if self.container is not None:
            seq = build_surface_sequence(
                self.container,
                use_visual_aabb=True,
                apply_shape_scale=True,
                technique=self._resolved_technique(),
            )
            return seq.sample(frame=self.container.global_pose, dt=DEFAULT_SAMPLE_DT)

        if self.target_pose is None:
            raise ValueError(
                "WipingAction requires either container or target_pose for sampling."
            )

        t_pose = self._target_pose_to_spatial()
        technique = self._resolved_technique()
        if technique == "spread":
            segment = MotionSegment(
                name="spread_patch",
                duration_s=2.0,
                local_curve=lambda tau: planar_sweep_x(
                    tau, length=float(self.length), cycles=max(1.0, float(self.cycles))
                ),
            )
        else:
            segment = MotionSegment(
                name="wipe_patch",
                duration_s=2.0,
                local_curve=lambda tau: planar_spiral_xy(
                    tau, r0=0.00, r1=0.12, cycles=2.5
                ),
            )
        seq = MotionSequence([segment])
        return seq.sample(frame=t_pose, dt=DEFAULT_SAMPLE_DT)

    def _accept_execution_failure_as_success(self, waypoints, exc: Exception) -> bool:
        if not self._is_timeout_failure(exc) or not waypoints:
            return False

        sponge = getattr(getattr(self, "tool", None), "root", None)
        if sponge is None:
            print("[wipe timeout check] no sponge root on WipingAction.tool")
            return False

        try:
            raw_last_xyz = np.asarray(waypoints[-1].to_np(), dtype=float).reshape(-1)[
                :3
            ]
            raw_sponge_xyz = np.asarray(
                sponge.global_pose.to_position().to_np(), dtype=float
            ).reshape(-1)[:3]
            print(
                "[wipe timeout check] raw last_point_xyz="
                f"{np.round(raw_last_xyz, 4).tolist()} "
                f"sponge_global_xyz={np.round(raw_sponge_xyz, 4).tolist()}"
            )
            tip_point = self.world.transform(
                sponge.global_pose.to_position(), self.world.root
            )
            tip_xyz = np.asarray(tip_point.to_np(), dtype=float).reshape(-1)[:3]
            goal_point = self.world.transform(waypoints[-1], self.world.root)
            goal_xyz = np.asarray(goal_point.to_np(), dtype=float).reshape(-1)[:3]
            distance = float(np.linalg.norm(tip_xyz - goal_xyz))
        except Exception as transform_exc:
            print(
                "[wipe timeout check] failed to compare sponge and last point: "
                f"{type(transform_exc).__name__}: {transform_exc}"
            )
            return False

        print(
            "[wipe timeout check] transformed last_point_xyz="
            f"{np.round(goal_xyz, 4).tolist()} "
            f"sponge_xyz={np.round(tip_xyz, 4).tolist()} "
            f"distance={distance:.3f}m "
            f"tolerance={float(self.final_waypoint_success_tolerance_m):.3f}m"
        )
        logger.warning(
            "WipingAction timeout check: last_point_xyz=%s sponge_xyz=%s distance=%.3fm",
            np.round(goal_xyz, 4).tolist(),
            np.round(tip_xyz, 4).tolist(),
            distance,
        )

        if distance > float(self.final_waypoint_success_tolerance_m):
            return False

        logger.warning(
            "Accepting WipingAction timeout as success because sponge reached final waypoint "
            "(distance=%.3fm, tolerance=%.3fm).",
            distance,
            self.final_waypoint_success_tolerance_m,
        )
        return True

    def _accept_motion_timeout_as_success(self, exc: Exception) -> bool:
        waypoints = getattr(self, "_last_waypoints", [])
        return self._accept_execution_failure_as_success(waypoints, exc)

    @staticmethod
    def _is_timeout_failure(exc: Exception) -> bool:
        current = exc
        while current is not None:
            if isinstance(current, TimeoutError):
                return True
            msg = str(current)
            if (
                "Timeout reached while waiting for end of motion" in msg
                or "Motion stalled while waiting for end of motion" in msg
                or "Hard timeout reached while waiting for end of motion" in msg
            ):
                return True
            current = current.__cause__ or current.__context__
        return False


@dataclass
class CuttingAction(GeneralizedActionPlan):
    """
    Execute a cutting motion sequence on a food object.
    """

    motion_timeout_ticks = 100

    container: Body = None
    """
    The object to cut.
    """
    technique: str = "saw"
    """
    Cutting trajectory variant.
    """

    slice_thickness: float = 0.03
    """
    Target slice thickness used to place the cut anchor.
    """

    num_cuts_x: int = 1
    """
    Number of repeated cut passes distributed across local X.
    """

    db_debug_waypoint_count: Optional[float] = None
    """
    Optional DB-logged test metric set during execute().
    """

    has_entry_from_above: Optional[bool] = None
    """
    Optional DB-logged cutting flag populated during execute().
    """

    def _sample_points(self):
        seq = build_cutting_sequence(
            self.container,
            use_visual_aabb=True,
            apply_shape_scale=True,
            technique=self.technique,
            slice_thickness=self.slice_thickness,
            num_cuts_x=self.num_cuts_x,
        )
        return seq.sample(frame=self.container.global_pose, dt=DEFAULT_SAMPLE_DT)


@dataclass
class PouringAction(GeneralizedActionPlan):
    """
    Execute a simple pose-aware pouring motion around a target container or anchor.
    """

    container: Body = None
    """
    Source container that is being tilted.
    """

    target_container: Optional[Body] = None
    """
    Optional target container/body over which the pour should happen.
    """

    pour_height: float = 0.10
    """
    Vertical offset above the target top surface during pouring.
    """

    pour_offset_xy: tuple[float, float] = (0.0, 0.0)
    """
    Additional local XY offset for the pour anchor.
    """

    approach_distance: float = 0.08
    """
    Approach distance before the pour anchor.
    """

    retreat_distance: float = 0.08
    """
    Retreat distance after the pour anchor.
    """

    max_tilt: float = float(np.deg2rad(65.0))
    """
    Maximum tilt angle during the pouring phase.
    """

    tilt_axis: str = "y"
    """
    Local tilt axis for the pouring rotation profile.
    """

    def _sample_pose_sequence(self):
        seq = build_pouring_sequence(
            self.container,
            target_body=self.target_container,
            use_visual_aabb=True,
            apply_shape_scale=True,
            pour_height=self.pour_height,
            pour_offset_xy=self.pour_offset_xy,
            approach_distance=self.approach_distance,
            retreat_distance=self.retreat_distance,
            max_tilt=self.max_tilt,
            tilt_axis=self.tilt_axis,
        )
        return seq.sample_poses(frame=self.container.global_pose, dt=DEFAULT_SAMPLE_DT)

    def _to_pose_waypoints(
        self, positions: np.ndarray, rotations: np.ndarray, stride: int
    ) -> list[Pose]:
        poses = []
        for p, r in zip(positions[::stride], rotations[::stride]):
            rotation_matrix = np.asarray(r, dtype=float)
            if rotation_matrix.shape == (3, 3):
                homogeneous_rotation = np.eye(4, dtype=float)
                homogeneous_rotation[:3, :3] = rotation_matrix
            elif rotation_matrix.shape == (4, 4):
                homogeneous_rotation = rotation_matrix
            else:
                raise ValueError(
                    "Expected pouring waypoint rotation to have shape (3, 3) or (4, 4), "
                    f"got {rotation_matrix.shape}."
                )

            quat = quaternion_from_matrix(homogeneous_rotation)
            poses.append(
                Pose.from_xyz_quaternion(
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    float(quat[0]),
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                    reference_frame=self.world.root,
                )
            )
        if not poses:
            raise ValueError("No pose waypoints left after applying pointer_stride.")
        return poses

    def _sample_motion(self):
        return self._sample_pose_sequence()

    def _extract_points_for_logging(self, sampled) -> np.ndarray:
        return np.asarray(sampled.positions, dtype=float)

    def _extract_phase_ids_for_logging(self, sampled) -> np.ndarray:
        return sampled.phase_ids

    def _logged_technique_value(self):
        return "pour"

    def _build_waypoints(self, sampled):
        stride = max(1, int(self.pointer_stride))
        return self._to_pose_waypoints(sampled.positions, sampled.rotations, stride)

    def _build_motion(self, waypoints):
        return MoveTCPWaypointsMotion(
            waypoints,
            self.arm,
            allow_gripper_collision=True,
        )


@dataclass
class SimplePouringAction(ActionDescription):
    """
    Execute a simple Cartesian pour over a target container.
    """

    object_designator: Body
    """
    Target container over which the held object should be poured.
    """

    arm: Arms
    """
    Arm that is holding the pouring object.
    """

    offset_x: float = 0.009
    """
    World-space X offset from the target container center.
    """

    offset_y: float = -0.125
    """
    World-space Y offset from the target container center.
    """

    offset_z: float = 0.17
    """
    World-space Z offset above the target container.
    """

    tilt_degrees: float = -110.0
    """
    Local roll applied after reaching the pre-pour pose.
    """

    def _approach_pose(self) -> Pose:
        target_pose = self.object_designator.global_pose
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        tip_quat = tip.global_pose.to_quaternion().to_np()

        return Pose.from_xyz_quaternion(
            float(target_pose.x) + float(self.offset_x),
            float(target_pose.y) + float(self.offset_y),
            float(target_pose.z) + float(self.offset_z),
            float(tip_quat[0]),
            float(tip_quat[1]),
            float(tip_quat[2]),
            float(tip_quat[3]),
            reference_frame=self.world.root,
        )

    def _tilted_pose(self, approach_pose: Pose) -> Pose:
        tilt = Pose.from_xyz_rpy(
            roll=float(np.deg2rad(self.tilt_degrees)),
            reference_frame=self.world.root,
        ).to_homogeneous_matrix()
        return (approach_pose.to_homogeneous_matrix() @ tilt).to_pose()

    def execute(self) -> None:
        approach_pose = self._approach_pose()
        tilted_pose = self._tilted_pose(approach_pose)

        self.add_subplan(
            sequential(
                [
                    MoveToolCenterPointMotion(
                        approach_pose,
                        self.arm,
                        allow_gripper_collision=True,
                        movement_type=MovementType.CARTESIAN,
                    ),
                    MoveToolCenterPointMotion(
                        tilted_pose,
                        self.arm,
                        allow_gripper_collision=True,
                        movement_type=MovementType.CARTESIAN,
                    ),
                ]
            )
        ).perform()
