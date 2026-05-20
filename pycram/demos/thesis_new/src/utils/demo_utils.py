import time

import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile

from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from tf2_msgs.msg import TFMessage

from krrood.ormatic.data_access_objects.helper import to_dao
from pycram.datastructures.enums import Arms
from pycram.locations.costmaps import OccupancyCostmap, RingCostmap
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
    ShapeSource,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.robot_mixins import SpecifiesLeftRightArm
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3
from semantic_digital_twin.world_description.connections import FixedConnection

from demos.thesis_new.src.utils.experiment_logging import body_name
from demos.thesis_new.src.utils.rviz import CostmapHeatmapRviz
from pycram.tf_transformations import quaternion_from_euler

KITCHEN_NAVIGATION_X_MAX = 1.5
KITCHEN_ORIGIN_EXCLUSION_RADIUS_M = 0.75
RVIZ_MARKER_TOPICS = (
    "/semworld/viz_marker",
    "/point_sequence",
    "/pycram/wipe_targets",
    "/debug/costmap/occupancy",
    "/debug/costmap/ring",
    "/debug/costmap/final",
)
DEMO_CAMERA_TARGET_FRAME = "demo_camera_target"


def _load_thesis_world_setup():
    try:
        from thesis_new.src.world_setup import (
            resolve_environment_name,
            resolve_robot_name_from_annotation,
        )
    except ImportError:
        from ..world_setup import (
            resolve_environment_name,
            resolve_robot_name_from_annotation,
        )
    return resolve_environment_name, resolve_robot_name_from_annotation


def setup_experiment_runtime(world, node_name):
    rclpy.init()
    node = rclpy.create_node(node_name)
    _clear_rviz_marker_topics(node)
    # Let RViz process DELETEALL before republishing the next world snapshot.
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.05)
    tf_wrapper = TFWrapper(node=node)
    TFPublisher(node=node, _world=world)
    VizMarkerPublisher(
        _world=world, node=node, shape_source=ShapeSource.VISUAL_WITH_COLLISION_BACKUP
    )
    robot = get_primary_robot(world)
    # tf_wrapper.wait_for_transform(
    #     "apartment/apartment_root",
    #     str(robot.root.name),
    #     timeout=RclpyDuration(seconds=1.0),
    #     time=Time(),
    # )
    return node


def _clear_rviz_marker_topics(node):
    clear_array = MarkerArray()
    clear_marker = Marker()
    clear_marker.action = Marker.DELETEALL
    clear_array.markers.append(clear_marker)

    volatile_qos = QoSProfile(depth=10)
    transient_local_qos = QoSProfile(
        depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL
    )
    publishers = [
        (
            node.create_publisher(MarkerArray, topic, volatile_qos),
            node.create_publisher(MarkerArray, topic, transient_local_qos),
        )
        for topic in RVIZ_MARKER_TOPICS
    ]
    for _ in range(3):
        for volatile_pub, transient_pub in publishers:
            volatile_pub.publish(clear_array)
            transient_pub.publish(clear_array)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.05)

    # Give ROS a brief chance to flush the clear messages before the node is destroyed.
    time.sleep(0.1)


def publish_demo_camera_target(
    node,
    world,
    target_pose_fn,
    *,
    camera_pose_fn=None,
    frame_name=DEMO_CAMERA_TARGET_FRAME,
):
    tf_pub = node.create_publisher(TFMessage, "tf", 10)

    if not hasattr(node, "_demo_camera_tf_handles"):
        node._demo_camera_tf_handles = []

    def _publish_camera_target():
        target_pose = target_pose_fn()
        target_xyz = np.asarray(target_pose.to_position().to_np(), dtype=float).reshape(
            -1
        )[:3]
        if camera_pose_fn is not None:
            camera_pose = camera_pose_fn()
            quat = np.asarray(camera_pose.to_quaternion().to_np(), dtype=float).reshape(
                -1
            )[:4]
        else:
            quat = np.asarray(
                quaternion_from_euler(0.0, 0.0, 0.0), dtype=float
            ).reshape(-1)[:4]

        transform = TransformStamped()
        transform.header.stamp = node.get_clock().now().to_msg()
        transform.header.frame_id = str(world.root.name)
        transform.child_frame_id = frame_name
        transform.transform.translation.x = float(target_xyz[0])
        transform.transform.translation.y = float(target_xyz[1])
        transform.transform.translation.z = float(target_xyz[2] + 0.12)
        transform.transform.rotation.x = float(quat[0])
        transform.transform.rotation.y = float(quat[1])
        transform.transform.rotation.z = float(quat[2])
        transform.transform.rotation.w = float(quat[3])
        tf_pub.publish(TFMessage(transforms=[transform]))

    _publish_camera_target()
    timer = node.create_timer(0.2, _publish_camera_target)
    node._demo_camera_tf_handles.append((tf_pub, timer))
    return frame_name


def shutdown_experiment_runtime(node):
    node.destroy_node()
    rclpy.shutdown()


def build_navigation_costmaps(
    robot,
    world,
    target_pose,
    *,
    width=200,
    height=200,
    resolution=0.02,
    ring_std=15,
    ring_distance=0.4,
    obstacle_clearance=None,
    number_of_samples=200,
):
    # Navigation costmaps represent feasible base poses on the floor plane, so the
    # map origin must not inherit the manipulated object's height.
    ground_pose = Pose(
        position=Point3(
            target_pose.to_position().x,
            target_pose.to_position().y,
            0.0,
        ),
        reference_frame=target_pose.reference_frame,
    )

    if obstacle_clearance is None:
        base_bb = robot.base.bounding_box
        obstacle_clearance = (base_bb.depth / 2 + base_bb.width / 2) / 2

    occupancy = OccupancyCostmap(
        distance_to_obstacle=obstacle_clearance,
        world=world,
        robot_view=robot,
        width=width,
        height=height,
        resolution=resolution,
        origin=ground_pose,
    )
    ring = RingCostmap(
        resolution=resolution,
        width=width,
        height=height,
        std=ring_std,
        distance=ring_distance,
        world=world,
        origin=ground_pose,
    )
    final_map = occupancy + ring
    final_map.number_of_samples = number_of_samples
    return occupancy, ring, final_map


def update_navigation_costmap_debug_publishers(
    node,
    world,
    publishers,
    occupancy,
    ring,
    final_map,
    *,
    namespace_prefix,
):
    if not publishers:
        publishers["occupancy"] = CostmapHeatmapRviz(
            occupancy,
            node=node,
            topic="/debug/costmap/occupancy",
            frame_id=str(world.root.name),
            marker_ns=f"{namespace_prefix}_occupancy_costmap",
            z_scale=0.04,
        )
        publishers["ring"] = CostmapHeatmapRviz(
            ring,
            node=node,
            topic="/debug/costmap/ring",
            frame_id=str(world.root.name),
            marker_ns=f"{namespace_prefix}_ring_costmap",
            z_scale=0.10,
            min_normalized_value=0.12,
        )
        publishers["final"] = CostmapHeatmapRviz(
            final_map,
            node=node,
            topic="/debug/costmap/final",
            frame_id=str(world.root.name),
            marker_ns=f"{namespace_prefix}_final_costmap",
            marker_type=Marker.SPHERE_LIST,
            z_offset=0.025,
            z_scale=0.14,
            xy_scale=0.042,
            cell_height=0.042,
            alpha=0.72,
            min_normalized_value=0.20,
            sample_stride=2,
        )
        return publishers

    publishers["occupancy"].set_costmap(occupancy)
    publishers["occupancy"].publish_once()
    publishers["ring"].set_costmap(ring)
    publishers["ring"].publish_once()
    publishers["final"].set_costmap(final_map)
    publishers["final"].publish_once()
    return publishers


def resolve_navigation_target(location_designator, *, description):
    try:
        first_candidate = next(iter(location_designator))
    except RuntimeError as exc:
        if "No values in the iterable" not in str(exc):
            raise
        first_candidate = None
    except StopIteration:
        first_candidate = None
    if first_candidate is not None:
        return [first_candidate]
    raise RuntimeError(f"No collision-free navigation pose found for {description}.")


def is_excluded_kitchen_pose(pose, *, environment_name=None):
    resolve_environment_name, _ = _load_thesis_world_setup()
    if resolve_environment_name(environment_name) not in {
        "kitchen",
        "test-kitchen-chat",
    }:
        return False
    position = pose.to_position()
    x = float(position.x)
    y = float(position.y)
    return (
        x > KITCHEN_NAVIGATION_X_MAX
        or float((x * x + y * y) ** 0.5) < KITCHEN_ORIGIN_EXCLUSION_RADIUS_M
    )


def resolve_navigation_target_for_environment(
    location_designator, *, description, environment_name=None
):
    saw_candidate = False
    try:
        for candidate in location_designator:
            saw_candidate = True
            if is_excluded_kitchen_pose(candidate, environment_name=environment_name):
                continue
            return [candidate]
    except RuntimeError as exc:
        if "No values in the iterable" not in str(exc):
            raise

    if saw_candidate:
        raise RuntimeError(
            f"No collision-free navigation pose found for {description} after environment filtering."
        )
    raise RuntimeError(f"No collision-free navigation pose found for {description}.")


def collect_named_targets(world, prefix):
    targets = []
    for body in getattr(world, "bodies", []):
        name = body_name(body)
        if name.startswith(prefix):
            targets.append(body)
    targets.sort(key=body_name)
    return targets


def set_entity_global_pose(world, entity, world_T_entity):
    parent_connection = entity.parent_connection
    parent_T_entity = world.transform(world_T_entity, parent_connection.parent)
    with world.modify_world():
        try:
            parent_connection.origin = parent_T_entity
        except NotImplementedError:
            parent_connection.parent_T_connection_expression = parent_T_entity


def iter_visual_shapes(body):
    seen = set()
    for owner in (body, getattr(body, "root", None)):
        if owner is None:
            continue
        geom = getattr(owner, "visual", None)
        if geom is None:
            continue
        for shape in getattr(geom, "shapes", []):
            sid = id(shape)
            if sid in seen:
                continue
            seen.add(sid)
            yield shape


def highlight_current_target(
    world,
    targets,
    active_target,
    *,
    default_color,
    active_color,
    failed_color,
    success_color,
    failed_targets=None,
    successful_targets=None,
):
    failed_targets = failed_targets or set()
    successful_targets = successful_targets or set()
    for target in targets:
        if target is active_target:
            color = active_color
        elif target in failed_targets:
            color = failed_color
        elif target in successful_targets:
            color = success_color
        else:
            color = default_color
        for shape in iter_visual_shapes(target):
            shape.color = color
    for callback in world.get_world_model_manager().model_change_callbacks:
        if isinstance(callback, VizMarkerPublisher):
            callback.notify()


def attach_bimanual_tools(
    world,
    parse_stl_fn,
    *,
    mesh_parts,
    right_name,
    left_name,
    right_pose_kwargs,
    left_pose_kwargs,
    tool_cls,
):
    right_tool = parse_stl_fn(*mesh_parts)
    right_tool.root.name.name = right_name
    left_tool = parse_stl_fn(*mesh_parts)
    left_tool.root.name.name = left_name

    l_tip, r_tip = get_bimanual_tool_frames(world)

    with world.modify_world():
        right_connection = FixedConnection(
            parent=r_tip,
            child=right_tool.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=r_tip, **right_pose_kwargs
            ),
        )
        left_connection = FixedConnection(
            parent=l_tip,
            child=left_tool.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=l_tip, **left_pose_kwargs
            ),
        )
        world.merge_world(right_tool, right_connection)
        world.merge_world(left_tool, left_connection)

    right_body = world.get_body_by_name(right_name)
    left_body = world.get_body_by_name(left_name)
    return tool_cls(root=right_body), tool_cls(root=left_body)


def get_available_arm_tool_frames(world):
    robot = get_primary_robot(world)
    if not getattr(robot, "arms", None):
        raise ValueError(f"Robot '{body_name(robot.root)}' does not expose any arms.")

    if isinstance(robot, SpecifiesLeftRightArm) and len(robot.arms) == 2:
        return [
            (Arms.RIGHT, robot.right_arm.manipulator.tool_frame),
            (Arms.LEFT, robot.left_arm.manipulator.tool_frame),
        ]

    if len(robot.arms) == 1:
        return [(Arms.LEFT, robot.arms[0].manipulator.tool_frame)]

    raise ValueError(
        f"Robot '{body_name(robot.root)}' exposes unsupported arm configuration with {len(robot.arms)} arms."
    )


def get_available_arms(world):
    return [arm for arm, _ in get_available_arm_tool_frames(world)]


def get_park_arms_argument(world):
    available_arms = get_available_arms(world)
    if len(available_arms) == 2:
        return Arms.BOTH
    return available_arms[0]


def attach_available_tools(
    world,
    parse_stl_fn,
    *,
    mesh_parts,
    right_name,
    left_name,
    right_pose_kwargs,
    left_pose_kwargs,
    tool_cls,
):
    available_frames = get_available_arm_tool_frames(world)
    tool_by_arm = {}

    with world.modify_world():
        for arm, tip in available_frames:
            if arm == Arms.RIGHT:
                tool_world = parse_stl_fn(*mesh_parts)
                tool_world.root.name.name = right_name
                tool_name = right_name
                pose_kwargs = right_pose_kwargs
            else:
                tool_world = parse_stl_fn(*mesh_parts)
                tool_world.root.name.name = left_name
                tool_name = left_name
                pose_kwargs = left_pose_kwargs

            connection = FixedConnection(
                parent=tip,
                child=tool_world.root,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=tip, **pose_kwargs
                ),
            )
            world.merge_world(tool_world, connection)
            tool_by_arm[arm] = tool_cls(root=connection.child)

    return [(arm, tool_by_arm[arm]) for arm, _ in available_frames]


def get_primary_robot(world):
    robots = world.get_semantic_annotations_by_type(AbstractRobot)
    if not robots:
        raise RuntimeError("No semantic robot found in world.")
    return robots[0]


def get_primary_robot_name(world):
    _, resolve_robot_name_from_annotation = _load_thesis_world_setup()
    return resolve_robot_name_from_annotation(get_primary_robot(world))


def get_bimanual_tool_frames(world):
    robot = get_primary_robot(world)
    if not isinstance(robot, SpecifiesLeftRightArm):
        raise ValueError(
            f"Robot '{body_name(robot.root)}' does not expose distinct left/right arms."
        )
    if len(robot.arms) != 2:
        raise ValueError(
            f"Robot '{body_name(robot.root)}' requires exactly two arms for this demo, found {len(robot.arms)}."
        )
    return robot.left_arm.manipulator.tool_frame, robot.right_arm.manipulator.tool_frame


def commit_plan_to_db(session, current_plan):
    try:
        dao = to_dao(current_plan)
        session.add(dao)
        session.commit()
        print("commited")
    except Exception as exc:
        session.rollback()
        print(f"[DB] commit failed: {type(exc).__name__}: {exc}")
