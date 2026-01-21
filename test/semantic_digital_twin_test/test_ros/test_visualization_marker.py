from dataclasses import dataclass, field
from time import sleep

from visualization_msgs.msg import MarkerArray

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.viz_marker import VizMarkerPublisher
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import OmniDrive


def test_visualization_marker(rclpy_node, box_bot_world):
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(node=rclpy_node, world=box_bot_world)
    viz = VizMarkerPublisher(world=box_bot_world, node=rclpy_node)

    @dataclass
    class Callback:
        last_msg: MarkerArray = field(init=False, default=None)

        def __call__(self, msg: MarkerArray):
            self.last_msg = msg

    callback = Callback()

    sub = rclpy_node.create_subscription(
        msg_type=MarkerArray,
        topic=viz.topic_name,
        callback=callback,
        qos_profile=viz.qos_profile,
    )
    for i in range(30):
        if callback.last_msg is not None:
            break
        sleep(0.1)
    else:
        assert False, "Callback timed out"
    assert len(callback.last_msg.markers) == 2
    callback.last_msg = None

    drive = box_bot_world.get_connections_by_type(OmniDrive)[0]
    new_pose = HomogeneousTransformationMatrix.from_xyz_rpy(1, 1)
    drive.origin = new_pose

    for i in range(30):
        transform = tf_wrapper.lookup_transform("map", "bot")
        sleep(0.1)
        if (
            transform.transform.translation.x == 1
            and transform.transform.translation.y == 1
        ):
            break
    else:
        assert False, "TF lookup timed out"
    assert callback.last_msg is None
