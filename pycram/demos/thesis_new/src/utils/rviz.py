import numpy as np

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from semantic_digital_twin.spatial_computations.raytracer import RayTracer

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

class CostmapHeatmapRviz:
    def __init__(
        self,
        costmap,
        *,
        node,
        topic="costmap_heatmap",
        frame_id="map",
        marker_ns=None,
        marker_type=Marker.CUBE_LIST,
        z_offset=0.02,
        z_scale=0.08,
        xy_scale=None,
        cell_height=0.003,
        alpha=0.85,
        republish_hz=2.0,
        min_value=1e-9,
        min_normalized_value=0.0,
        sample_stride=1,
    ):
        """
        Publish a costmap as a colored RViz heatmap using a list marker.
        """
        if node is None:
            raise ValueError("node must be provided when publishing a costmap heatmap")

        self.costmap = costmap
        self.node = node
        self.topic = _norm_topic(topic)
        self.frame_id = str(frame_id)
        self.marker_ns = (
            str(marker_ns)
            if marker_ns is not None
            else _next_marker_group("costmap_heat")
        )
        self.marker_type = int(marker_type)
        self.z_offset = float(z_offset)
        self.z_scale = float(z_scale)
        self.xy_scale = xy_scale
        self.cell_height = float(cell_height)
        self.alpha = float(alpha)
        self.min_value = float(min_value)
        self.min_normalized_value = float(min_normalized_value)
        self.sample_stride = max(int(sample_stride), 1)
        self.pub = self.node.create_publisher(MarkerArray, self.topic, 10)

        self.node.get_logger().info(
            f"Publishing costmap heatmap on {self.topic} in frame '{self.frame_id}'"
        )

        self.publish_once()

        self._timer = None
        if republish_hz is not None and float(republish_hz) > 0.0:
            self._timer = self.node.create_timer(
                1.0 / float(republish_hz), self.publish_once
            )

    def set_costmap(self, costmap):
        self.costmap = costmap

    def _costmap_to_points_and_colors(self):
        if self.costmap is None:
            return [], []

        map_data = np.asarray(self.costmap.map, dtype=float)
        valid_mask = np.isfinite(map_data) & (map_data > self.min_value)
        if not np.any(valid_mask):
            return [], []

        values = map_data[valid_mask]
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        denom = v_max - v_min

        origin_position, origin_orientation = _pose_to_position_and_orientation_lists(
            self.costmap.origin
        )
        corner_offset = np.array(
            [
                -self.costmap.height * self.costmap.resolution / 2.0,
                -self.costmap.width * self.costmap.resolution / 2.0,
                self.z_offset,
            ],
            dtype=float,
        )

        points = []
        colors = []
        for row, col in np.argwhere(valid_mask):
            if (row % self.sample_stride) != 0 or (col % self.sample_stride) != 0:
                continue
            normalized = 1.0 if denom <= 0.0 else (map_data[row, col] - v_min) / denom
            if normalized < self.min_normalized_value:
                continue

            local_point = corner_offset + np.array(
                [
                    (float(row) + 0.5) * self.costmap.resolution,
                    (float(col) + 0.5) * self.costmap.resolution,
                    normalized * self.z_scale,
                ],
                dtype=float,
            )
            points.append(
                _transform_point(origin_position, origin_orientation, local_point)
            )

            colors.append(_heat_color(normalized, alpha=self.alpha))

        return points, colors

    def publish_once(self):
        now = self.node.get_clock().now().to_msg()
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = now
        marker.ns = self.marker_ns
        marker.id = 0
        marker.type = self.marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        default_xy_scale = float(self.costmap.resolution) if self.costmap else 0.01
        xy_scale = (
            float(self.xy_scale) if self.xy_scale is not None else default_xy_scale
        )
        marker.scale.x = xy_scale
        marker.scale.y = xy_scale
        marker.scale.z = self.cell_height

        marker.points, marker.colors = self._costmap_to_points_and_colors()

        arr = MarkerArray()
        arr.markers.append(marker)
        self.pub.publish(arr)


class CameraVisiblePointsRviz:
    def __init__(
        self,
        world,
        camera_pose,
        *,
        node,
        frame_id="map",
        topic="camera_visible_points",
        marker_ns=None,
        resolution=128,
        min_distance=0.0,
        max_distance=np.inf,
        point_scale=0.01,
        alpha=0.9,
        show_rays=False,
        ray_stride=8,
        ray_alpha=0.2,
        origin_scale=0.03,
        republish_hz=2.0,
    ):
        """
        Publish visible raytracer hit points from a camera pose as an RViz POINTS marker.
        """
        if node is None:
            raise ValueError(
                "node must be provided when publishing visible camera points"
            )

        self.world = world
        self.camera_pose = camera_pose
        self.node = node
        self.frame_id = str(frame_id)
        self.topic = _norm_topic(topic)
        self.marker_ns = (
            str(marker_ns)
            if marker_ns is not None
            else _next_marker_group("camera_visible_points")
        )
        self.resolution = int(resolution)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.point_scale = float(point_scale)
        self.alpha = float(alpha)
        self.show_rays = bool(show_rays)
        self.ray_stride = max(int(ray_stride), 1)
        self.ray_alpha = float(ray_alpha)
        self.origin_scale = float(origin_scale)
        self.pub = self.node.create_publisher(MarkerArray, self.topic, 10)
        self.ray_tracer = RayTracer(world)

        self.node.get_logger().info(
            f"Publishing visible camera points on {self.topic} in frame '{self.frame_id}'"
        )

        self.publish_once()

        self._timer = None
        if republish_hz is not None and float(republish_hz) > 0.0:
            self._timer = self.node.create_timer(
                1.0 / float(republish_hz), self.publish_once
            )

    def set_camera_pose(self, camera_pose):
        self.camera_pose = camera_pose

    def _visible_points_and_colors(self):
        ray_origins, ray_directions, _ = self.ray_tracer.create_camera_rays(
            self.camera_pose, resolution=self.resolution
        )
        target_points = ray_origins + ray_directions * 10.0
        points, index_ray, _ = self.ray_tracer.ray_test(
            ray_origins,
            target_points,
            multiple_hits=True,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
        )

        if len(index_ray) == 0:
            return np.zeros((0, 3), dtype=float), [], np.zeros((0, 3), dtype=float)

        unique_index = np.unique(index_ray, return_index=True)[1]
        points = np.asarray(points, dtype=float)[unique_index]
        index_ray = np.asarray(index_ray, dtype=int)[unique_index]
        hit_origins = np.asarray(ray_origins[index_ray], dtype=float)
        depths = np.linalg.norm(points - hit_origins, axis=1)

        if len(depths) == 0:
            return np.zeros((0, 3), dtype=float), [], np.zeros((0, 3), dtype=float)

        d_min = float(np.min(depths))
        d_max = float(np.max(depths))
        denom = d_max - d_min
        normalized = np.ones_like(depths) if denom <= 0.0 else (depths - d_min) / denom
        colors = [_heat_color(value, alpha=self.alpha) for value in normalized]
        return points, colors, hit_origins

    def publish_once(self):
        now = self.node.get_clock().now().to_msg()
        points_np, colors, hit_origins = self._visible_points_and_colors()
        arr = MarkerArray()

        points_marker = Marker()
        points_marker.header.frame_id = self.frame_id
        points_marker.header.stamp = now
        points_marker.ns = self.marker_ns
        points_marker.id = 0
        points_marker.type = Marker.POINTS
        points_marker.action = Marker.ADD
        points_marker.pose.orientation.w = 1.0
        points_marker.scale.x = self.point_scale
        points_marker.scale.y = self.point_scale
        points_marker.points = _as_points(points_np)
        points_marker.colors = colors
        arr.markers.append(points_marker)

        origin_position, _ = _pose_to_position_and_orientation_lists(self.camera_pose)
        origin_marker = Marker()
        origin_marker.header.frame_id = self.frame_id
        origin_marker.header.stamp = now
        origin_marker.ns = self.marker_ns
        origin_marker.id = 1
        origin_marker.type = Marker.SPHERE
        origin_marker.action = Marker.ADD
        origin_marker.pose.orientation.w = 1.0
        origin_marker.pose.position.x = float(origin_position[0])
        origin_marker.pose.position.y = float(origin_position[1])
        origin_marker.pose.position.z = float(origin_position[2])
        origin_marker.scale.x = self.origin_scale
        origin_marker.scale.y = self.origin_scale
        origin_marker.scale.z = self.origin_scale
        origin_marker.color = _color(1.0, 1.0, 1.0, 1.0)
        arr.markers.append(origin_marker)

        if self.show_rays and len(points_np) > 0:
            ray_marker = Marker()
            ray_marker.header.frame_id = self.frame_id
            ray_marker.header.stamp = now
            ray_marker.ns = self.marker_ns
            ray_marker.id = 2
            ray_marker.type = Marker.LINE_LIST
            ray_marker.action = Marker.ADD
            ray_marker.pose.orientation.w = 1.0
            ray_marker.scale.x = self.point_scale * 0.35
            ray_points = []
            ray_colors = []
            for i in range(0, len(points_np), self.ray_stride):
                ray_points.extend(_as_points([hit_origins[i], points_np[i]]))
                ray_colors.extend(
                    [
                        _color(1.0, 1.0, 1.0, self.ray_alpha),
                        _color(1.0, 1.0, 1.0, self.ray_alpha),
                    ]
                )
            ray_marker.points = ray_points
            ray_marker.colors = ray_colors
            arr.markers.append(ray_marker)

        self.pub.publish(arr)

