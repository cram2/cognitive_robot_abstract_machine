from __future__ import annotations

from dataclasses import dataclass

from sensor_msgs.msg import LaserScan

from semantic_digital_twin.adapters.ros.latest_message_subscriber import (
    LatestMessageSubscriber,
)
from semantic_digital_twin.adapters.ros.ros2_to_semdt_converters import (
    LaserScanToSemDTConverter,
)
from semantic_digital_twin.adapters.sensors.lidar import Lidar, LidarSource
from semantic_digital_twin.datastructures.lidar_reading import LidarReading
from semantic_digital_twin.exceptions import NoLaserScanReceived


@dataclass
class SubscribedLidarSource(LidarSource, LatestMessageSubscriber[LaserScan]):
    """
    A source that reports what a real scanner publishes on a ROS 2 topic.

    The scanner itself decides what it sweeps, so reading it adopts the pattern the
    received scan was taken with.
    """

    @property
    def received_scan(self) -> LaserScan:
        """
        :return: The most recently received scan.
        :raises NoLaserScanReceived: If no scan has arrived yet.
        """
        if self.latest_message is None:
            raise NoLaserScanReceived(self.topic_name)
        return self.latest_message

    def get_lidar_reading(self, lidar: Lidar) -> LidarReading:
        reading = LaserScanToSemDTConverter.convert(
            self.received_scan, lidar.root._world
        )
        lidar.scan_pattern = reading.scan_pattern
        return reading
