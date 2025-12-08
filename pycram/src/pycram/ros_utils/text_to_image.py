from typing import Optional
import logging
import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from std_msgs.msg import String
from ..ros.ros2.publisher import create_publisher
from rclpy.qos import QoSProfile, ReliabilityPolicy

logger = logging.getLogger(__name__)


class TextImagePublisher:
    def __init__(self, topic_name: str = "/head_display/text_to_image"):
        self.is_init = False
        self.topic_name = topic_name
        self.node: Optional[Node] = None
        self.publisher: Optional[Publisher] = None
        self._init_interface()


    def _init_interface(self):
        """
        Initializes the ROS node and publisher once.
        """
        if self.is_init:
            return

        if not rclpy.ok():
            rclpy.init()

        reliable_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.node = rclpy.create_node("text_publisher_node")
        self.publisher = self.node.create_publisher(String, self.topic_name, reliable_qos)

        self.first_screen()
        self.is_init = True

        logger.info("TextImagePublisher initialized")


    def publish_text(self, text: str):
        """
        Publishes a text message to the display of HSR.
        """
        self._init_interface()

        msg = String()
        msg.data = text
        self.publisher.publish(msg)

        logger.info("Published new text to image display")


    def first_screen(self):
        """
        Triggers the first screen update.
        The initial image takes a few seconds to appear.
        """

        if not self.publisher:
            return

        msg = String()
        msg.data = ""

        for _ in range(8):
            self.publisher.publish(msg)


    def shutdown(self):
        """
        Shuts down the ROS node.
        """
        if self.node:
            self.node.destroy_node()

        logger.info("TextImagePublisher shut down")
