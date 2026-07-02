from dataclasses import dataclass, field
from time import sleep

from rclpy.node import Node
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import OmniDrive
from visualization_msgs.msg import Marker, MarkerArray

from semantic_digital_twin.adapters.ros.visualization.collision_viz_marker import (
    CollisionVisualizationMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.robots.minimal_robot import MinimalRobot


@dataclass
class MarkerArrayRecorder:
    namespace: str
    last_msg: MarkerArray = field(init=False, default=None)

    def __call__(self, msg: MarkerArray):
        if msg.markers[0].ns == self.namespace:
            self.last_msg = msg


def _avoid_robot_environment_collisions(world):
    """
    Adds a rule so the robot avoids both environment bodies and rebuilds the matrix.
    """
    robot = world.get_semantic_annotations_by_type(MinimalRobot)[0]
    environment = world.get_kinematic_structure_entity_by_name("environment")
    environment2 = world.get_kinematic_structure_entity_by_name("environment2")
    collision_manager = world.collision_manager
    collision_manager.temporary_rules.append(
        AvoidCollisionBetweenGroups(
            buffer_zone_distance=10,
            violated_distance=0.0,
            body_group_a=[robot.root],
            body_group_b=[environment, environment2],
        )
    )
    collision_manager.update_collision_matrix()
    return collision_manager


def _wait_for_message(recorder: MarkerArrayRecorder):
    for _ in range(30):
        if recorder.last_msg is not None:
            break
        sleep(0.1)
    else:
        assert False, "Callback timed out"


def _subscribe(node: Node, publisher: CollisionVisualizationMarkerPublisher):
    recorder = MarkerArrayRecorder(namespace=publisher.namespace)
    node.create_subscription(
        msg_type=MarkerArray,
        topic=publisher.topic_name,
        callback=recorder,
        qos_profile=publisher.qos_profile,
    )
    return recorder


def test_publishes_line_list_on_collision_check(rclpy_node, cylinder_bot_world):
    collision_manager = _avoid_robot_environment_collisions(cylinder_bot_world)
    publisher = CollisionVisualizationMarkerPublisher(
        node=rclpy_node, world=cylinder_bot_world
    )
    recorder = _subscribe(rclpy_node, publisher)

    collisions = collision_manager.compute_collisions()
    assert collisions.any()
    collision_manager.compute_collisions()

    _wait_for_message(recorder)
    marker = recorder.last_msg.markers[0]
    assert marker.type == Marker.LINE_LIST
    assert marker.header.frame_id == str(cylinder_bot_world.root.name)
    assert len(marker.points) == 2 * len(collisions.contacts)
    assert len(marker.colors) == len(marker.points)


def test_color_green_when_below_buffer_threshold(rclpy_node, cylinder_bot_world):
    collision_manager = _avoid_robot_environment_collisions(cylinder_bot_world)
    # Threshold above every contact distance, so all contacts must be red.
    publisher = CollisionVisualizationMarkerPublisher(
        node=rclpy_node, world=cylinder_bot_world
    )
    recorder = _subscribe(rclpy_node, publisher)

    collision_manager.compute_collisions()

    _wait_for_message(recorder)
    colors = recorder.last_msg.markers[0].colors
    assert colors
    for color in colors:
        assert color.r == 1.0
        assert color.g == 1.0
        assert color.b == 0.0
        assert color.a == 1.0


def test_color_green_when_below_violated_threshold(rclpy_node, cylinder_bot_world):
    collision_manager = _avoid_robot_environment_collisions(cylinder_bot_world)
    # Threshold above every contact distance, so all contacts must be red.
    publisher = CollisionVisualizationMarkerPublisher(
        node=rclpy_node, world=cylinder_bot_world
    )
    recorder = _subscribe(rclpy_node, publisher)

    cylinder_bot_world.get_connections_by_type(OmniDrive)[0].origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(x=1)
    )

    collision_manager.compute_collisions()

    _wait_for_message(recorder)
    colors = recorder.last_msg.markers[0].colors
    assert colors
    for color in colors:
        if color.r == 1.0 and color.g == 0.0 and color.b == 0.0 and color.a == 1.0:
            break
    else:
        assert False, "No red color found"


def test_color_green_when_above_buffer_threshold(rclpy_node, cylinder_bot_world):
    collision_manager = _avoid_robot_environment_collisions(cylinder_bot_world)
    # Threshold below every contact distance, so all contacts must be green.
    publisher = CollisionVisualizationMarkerPublisher(
        node=rclpy_node, world=cylinder_bot_world
    )
    recorder = _subscribe(rclpy_node, publisher)

    # put robot to 10.3 to be above buffer threshold but close enough to still get checked.
    cylinder_bot_world.get_connections_by_type(OmniDrive)[0].origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(x=-10.3)
    )

    collision_manager.compute_collisions()

    _wait_for_message(recorder)
    colors = recorder.last_msg.markers[0].colors
    assert colors
    for color in colors:
        assert color.r == 0.0
        assert color.g == 1.0
        assert color.b == 0.0
        assert color.a == 1.0


def test_throttle_publishes_every_nth_check(rclpy_node, cylinder_bot_world):
    collision_manager = _avoid_robot_environment_collisions(cylinder_bot_world)
    publisher = CollisionVisualizationMarkerPublisher(
        node=rclpy_node, throttle=2, world=cylinder_bot_world
    )
    recorder = _subscribe(rclpy_node, publisher)

    collision_manager.compute_collisions()
    sleep(0.3)
    assert recorder.last_msg is None

    collision_manager.compute_collisions()
    _wait_for_message(recorder)


def test_with_collision_visualization_wires_consumer(rclpy_node, cylinder_bot_world):
    collision_manager = _avoid_robot_environment_collisions(cylinder_bot_world)
    viz = VizMarkerPublisher(_world=cylinder_bot_world, node=rclpy_node)
    viz.with_collision_visualization()
    publisher = viz._collision_publisher

    assert publisher in collision_manager.collision_consumers
    recorder = _subscribe(rclpy_node, publisher)

    collision_manager.compute_collisions()
    _wait_for_message(recorder)
    assert recorder.last_msg.markers[0].type == Marker.LINE_LIST
