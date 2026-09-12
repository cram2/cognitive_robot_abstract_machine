from __future__ import annotations

from dataclasses import dataclass, field

from rclpy.node import Node
from rclpy.subscription import Subscription
from sensor_msgs.msg import LaserScan
from typing_extensions import List, Optional, Self

from semantic_digital_twin.adapters.ros.ros2_to_semdt_converters import (
    LaserScanToSemDTConverter,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.laser_reading import LaserReading
from semantic_digital_twin.datastructures.scan_pattern import ScanPattern
from semantic_digital_twin.exceptions import NoLaserScanReceived, UselessConceptError
from semantic_digital_twin.robots.robot_parts import Laser
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class SubscribedLaser(Laser):
    """
    A laser that reports what a real scanner publishes on a ROS 2 topic.

    The scanner itself decides what it sweeps, so a received scan replaces the pattern
    this laser was built with.
    """

    node: Node = field(kw_only=True)
    """
    The node the scans are received on.
    """

    topic_name: str = field(kw_only=True)
    """
    The topic the scans are published on.
    """

    latest_scan: Optional[LaserScan] = field(default=None, init=False)
    """
    The most recently received scan, or ``None`` while none has arrived.
    """

    subscription: Subscription = field(init=False, repr=False)
    """
    The subscription the scans arrive through.
    """

    def __post_init__(self):
        super().__post_init__()
        self.subscription = self.node.create_subscription(
            LaserScan,
            topic=self.topic_name,
            callback=self.store_scan,
            qos_profile=10,
        )

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        A subscribed laser needs the node and topic its scans arrive on, which a robot
        description does not carry.

        :raises UselessConceptError: Always.
        """
        raise UselessConceptError(
            reason="A SubscribedLaser needs the node and topic its scans arrive on, which a robot description does not carry"
        )

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    def store_scan(self, scan: LaserScan) -> None:
        """
        Keeps a received scan as the one this laser reports, and adopts the pattern it
        was taken with.

        :param scan: The scan that was received.
        """
        self.latest_scan = scan
        self.scan_pattern = ScanPattern(
            minimum_angle=scan.angle_min,
            maximum_angle=scan.angle_max,
            angle_increment=scan.angle_increment,
            minimum_range=scan.range_min,
            maximum_range=scan.range_max,
        )

    @property
    def received_scan(self) -> LaserScan:
        """
        :return: The most recently received scan.
        :raises NoLaserScanReceived: If no scan has arrived yet.
        """
        if self.latest_scan is None:
            raise NoLaserScanReceived(self.topic_name)
        return self.latest_scan

    def get_laser_reading(self) -> LaserReading:
        return LaserScanToSemDTConverter.convert(self.received_scan, self.root._world)
