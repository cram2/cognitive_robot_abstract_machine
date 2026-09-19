"""
Watch the camera's own streams, with no perception behind them.

Run it with::

    python -m experiments.montessori.perception.watch_camera

Unlike :mod:`~experiments.montessori.perception.node` this needs nothing but the camera:
no world to fetch, no transform tree, no robot. That makes it the thing to reach for when
the question is whether frames are arriving at all, since anything it fails to draw was
never received.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

from experiments.montessori.perception.camera import (
    CameraTopic,
    decode_compressed_color_image,
    decode_compressed_depth_image,
)
from experiments.montessori.perception.viewer import CameraFrameViewer

# %% watching

NODE_NAME = "montessori_camera_watch"
"""
Name this node registers under.
"""

SPIN_SECONDS = 0.05
"""
How long each turn of the loop waits for a message before drawing again.
"""


@dataclass
class CameraStreamWatcher:
    """
    Draws each camera stream as its frames arrive.

    Each stream is drawn on its own, so a stream that has stopped shows as an empty
    window while the other keeps updating.
    """

    node: Node
    """
    The node the streams are subscribed on.
    """

    viewer: CameraFrameViewer = field(default_factory=CameraFrameViewer)
    """
    Where the frames are shown.
    """

    def __post_init__(self) -> None:
        self.node.create_subscription(
            CompressedImage, CameraTopic.COLOR, self._on_color, qos_profile_sensor_data
        )
        self.node.create_subscription(
            CompressedImage, CameraTopic.DEPTH, self._on_depth, qos_profile_sensor_data
        )

    def _on_color(self, message: CompressedImage) -> None:
        """
        Hand the newest colour image to the viewer.

        :param message: The colour image, as the camera compressed it.
        """
        self.viewer.show_color(
            decode_compressed_color_image(message.data, message.format)
        )

    def _on_depth(self, message: CompressedImage) -> None:
        """
        Hand the newest depth image to the viewer.

        :param message: The depth image, as the camera compressed it.
        """
        self.viewer.show_depth(
            decode_compressed_depth_image(message.data, message.format)
        )


def main() -> None:
    """
    Draw the camera's streams until interrupted.
    """
    rclpy.init()
    node = rclpy.create_node(NODE_NAME)
    watcher = CameraStreamWatcher(node=node)
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=SPIN_SECONDS)
        watcher.viewer.refresh()


if __name__ == "__main__":
    main()
