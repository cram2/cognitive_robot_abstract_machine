from semantic_digital_twin.adapters.ros import tf_publisher
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.pose_publisher import (
    PosePublisher,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


def test_pose_publisher(rclpy_node, cylinder_bot_world):
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, 1, 1, reference_frame=cylinder_bot_world.root
    )

    publisher = PosePublisher(pose=pose, node=rclpy_node, lifetime=0)
    viz_marker = VizMarkerPublisher(_world=cylinder_bot_world, node=rclpy_node)
    tf_publisher = TFPublisher(node=rclpy_node, _world=cylinder_bot_world)


def test_marker_array(rclpy_node, cylinder_bot_world):
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, 1, 1, reference_frame=cylinder_bot_world.root
    )

    publisher = PosePublisher(pose=pose, node=rclpy_node, lifetime=0)

    marker = publisher._create_marker_array()

    assert len(marker.markers) == 3

    assert marker.markers[0].header.frame_id == str(cylinder_bot_world.root.name)
    assert marker.markers[0].points[1].x == 0.5

    assert marker.markers[1].points[1].y == 0.5
    assert marker.markers[2].points[1].z == 0.5
