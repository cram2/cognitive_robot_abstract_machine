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


def test_publish_pose_convenience(rclpy_node, cylinder_bot_world):
    from semantic_digital_twin.adapters.ros.visualization.pose_publisher import (
        publish_pose,
    )

    pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, 1, 1, reference_frame=cylinder_bot_world.root
    )

    publisher = publish_pose(pose=pose, node=rclpy_node, lifetime=0)
    assert isinstance(publisher, PosePublisher)
    assert publisher.pose == pose


def test_publish_pose_with_semdt_pose(rclpy_node, cylinder_bot_world):
    from semantic_digital_twin.adapters.ros.visualization.pose_publisher import (
        publish_pose,
    )
    from semantic_digital_twin.spatial_types import Pose as SemDTPose

    pose = SemDTPose.from_xyz_rpy(1, 1, 1, reference_frame=cylinder_bot_world.root)

    publisher = publish_pose(pose=pose, node=rclpy_node, lifetime=0)
    assert isinstance(publisher, PosePublisher)
    assert isinstance(publisher.pose, HomogeneousTransformationMatrix)
    # Pose is not a subclass of HomogeneousTransformationMatrix, so this check is valid
    assert not isinstance(publisher.pose, SemDTPose)
    assert publisher.pose.reference_frame == cylinder_bot_world.root


def test_pose_publisher_direct_with_pose(rclpy_node, cylinder_bot_world):
    from semantic_digital_twin.spatial_types import Pose as SemDTPose

    pose = SemDTPose.from_xyz_rpy(1, 1, 1, reference_frame=cylinder_bot_world.root)
    publisher = PosePublisher(pose=pose, node=rclpy_node, lifetime=0)
    assert isinstance(publisher.pose, HomogeneousTransformationMatrix)
    assert not isinstance(publisher.pose, SemDTPose)
