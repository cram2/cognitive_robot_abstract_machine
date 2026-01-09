from itertools import chain, repeat

from unittest.mock import patch, MagicMock

import rclpy.publisher
from tf2_msgs.msg import TFMessage
from pycram.datastructures.enums import ExecutionType
from pycram.ros_utils.tf_broadcaster import TFBroadcaster


def test_initialization(immutable_model_world, rclpy_node):
    world, robot_view, context = immutable_model_world
    node = rclpy_node
    broadcaster = TFBroadcaster(world, node)
    assert broadcaster.tf_publisher is not None
    assert broadcaster.tf_static_publisher is not None
    assert isinstance(broadcaster.tf_publisher, rclpy.publisher.Publisher)
    assert isinstance(broadcaster.tf_static_publisher, rclpy.publisher.Publisher)
    assert broadcaster.interval == 0.1
    assert broadcaster.odom_frame == "odom"
    assert broadcaster.projection_namespace == ExecutionType.SIMULATED


def test_update_static_odom_calls_publish_pose(immutable_model_world, rclpy_node):
    world, robot_view, context = immutable_model_world
    node = rclpy_node
    broadcaster = TFBroadcaster(world, node)
    broadcaster.tf_publisher = MagicMock()
    broadcaster.tf_static_publisher = MagicMock()
    broadcaster.projection_namespace = ExecutionType.SIMULATED
    broadcaster._publish_pose = MagicMock()
    broadcaster.odom_frame = "odom"

    mock_pose = MagicMock()
    with patch(
        "pycram.ros_utils.tf_broadcaster.PoseStamped.from_list",
        return_value=mock_pose,
    ):
        broadcaster._update_static_odom()
        broadcaster._publish_pose.assert_called_once_with(
            "odom", mock_pose, static=True
        )


def test_update_objects_publishes_all_objects_and_links(
    immutable_model_world, rclpy_node
):
    world, robot_view, context = immutable_model_world
    node = rclpy_node
    broadcaster = TFBroadcaster(world, node)
    # broadcaster.world = MagicMock()
    broadcaster.tf_publisher = MagicMock()
    broadcaster.tf_static_publisher = MagicMock()
    broadcaster.projection_namespace = ExecutionType.SIMULATED

    mock_obj = MagicMock()
    mock_obj.tf_frame = "obj1"
    mock_obj.link_name_to_id = {"link1": 0}
    mock_obj.get_link_tf_frame.return_value = "obj1/link1"
    # broadcaster.world.objects = [mock_obj]

    broadcaster._publish_pose = MagicMock()
    broadcaster._update_objects()

    assert broadcaster._publish_pose.call_count == len(world.bodies)
    assert (
        broadcaster._publish_pose.call_args_list[0][0][0] == "apartment/apartment_root"
    )


def test_update_objects_with_no_objects(immutable_model_world, rclpy_node):
    world, robot_view, context = immutable_model_world
    node = rclpy_node
    broadcaster = TFBroadcaster(world, node)
    broadcaster.world = MagicMock()
    broadcaster.world.objects = []
    broadcaster._publish_pose = MagicMock()

    broadcaster._update_objects()
    broadcaster._publish_pose.assert_not_called()


def test_publish_pose_calls_correct_publisher(immutable_model_world, rclpy_node):
    world, robot_view, context = immutable_model_world
    node = rclpy_node
    broadcaster = TFBroadcaster(world, node)
    broadcaster.tf_publisher = MagicMock()
    broadcaster.tf_static_publisher = MagicMock()
    broadcaster.projection_namespace = ExecutionType.SIMULATED

    mock_pose = MagicMock()
    mock_pose.frame_id = "base"
    mock_transform = MagicMock()
    mock_transform.child_frame_id = "child"
    mock_transform.frame_id = "base"
    mock_pose.to_transform_stamped.return_value = mock_transform

    with patch("pycram.ros_utils.tf_broadcaster.TFMessage", return_value=TFMessage()):
        broadcaster._publish_pose("child", mock_pose, static=False)

    broadcaster.tf_publisher.publish.assert_called_once()


def test_publish_loop_runs_once(immutable_model_world, rclpy_node):
    world, robot_view, context = immutable_model_world
    node = rclpy_node
    broadcaster = TFBroadcaster(world, node)
    broadcaster.kill_event = MagicMock()
    broadcaster.kill_event.is_set.side_effect = chain([False], repeat(True))
    broadcaster.update = MagicMock()
    broadcaster.interval = 0.01

    broadcaster._publish()

    broadcaster.update.assert_called_once()


def test_stop_publishing(immutable_model_world, rclpy_node):
    world, robot_view, context = immutable_model_world
    node = rclpy_node
    broadcaster = TFBroadcaster(world, node)
    broadcaster.kill_event = MagicMock()
    broadcaster.thread = MagicMock()

    broadcaster._stop_publishing()
    broadcaster.kill_event.set.assert_called_once()
    broadcaster.thread.join.assert_called_once()
