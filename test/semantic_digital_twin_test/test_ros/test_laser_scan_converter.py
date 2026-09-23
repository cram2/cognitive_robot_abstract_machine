from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from sensor_msgs.msg import LaserScan
from typing_extensions import Self

from semantic_digital_twin.adapters.ros.lidar import SubscribedLidarSource
from semantic_digital_twin.adapters.ros.exceptions import LaserScanBeamCountMismatch
from semantic_digital_twin.adapters.ros.msg_converter import Ros2ToSemDTConverter
from semantic_digital_twin.adapters.ros.ros2_to_semdt_converters import (
    LaserScanToSemDTConverter,
)
from semantic_digital_twin.adapters.sensors.lidar import Lidar, LidarSource
from semantic_digital_twin.datastructures.scan_pattern import ScanPattern
from semantic_digital_twin.exceptions import NoLaserScanReceived
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% scan messages under test

LASER_FRAME_NAME = "base_laser_link"
"""
Name of the body the scans below are expressed in.
"""

RANGES = [1.0, 2.0, 3.0]
"""
Measured distances of the beams of :func:`laser_scan`.
"""

DECLARED_PATTERN = ScanPattern(
    minimum_angle=-np.pi / 2,
    maximum_angle=np.pi / 2,
    angle_increment=np.pi / 2,
    minimum_range=0.2,
    maximum_range=5.0,
)
"""
The pattern a lidar is built with, chosen to differ from the one :func:`laser_scan`
declares.
"""


def laser_scan(ranges: list[float] = None) -> LaserScan:
    """
    :param ranges: The measured distances, defaulting to :data:`RANGES`.
    :return: A scan of three beams spanning a right angle around the forward axis.
    """
    scan = LaserScan()
    scan.header.frame_id = LASER_FRAME_NAME
    scan.angle_min = -np.pi / 4
    scan.angle_max = np.pi / 4
    scan.angle_increment = np.pi / 4
    scan.range_min = 0.1
    scan.range_max = 10.0
    scan.ranges = list(RANGES if ranges is None else ranges)
    return scan


@pytest.fixture
def world_with_laser_body() -> World:
    """
    A world whose root body carries the name the scan messages refer to.
    """
    return World.create_with_root_body(LASER_FRAME_NAME)


# %% message conversion


def test_converted_scan_keeps_the_ranges_of_the_message(world_with_laser_body):
    reading = LaserScanToSemDTConverter.convert(laser_scan(), world_with_laser_body)

    assert reading.ranges.tolist() == RANGES


def test_converted_scan_is_expressed_in_the_body_named_by_its_header(
    world_with_laser_body,
):
    reading = LaserScanToSemDTConverter.convert(laser_scan(), world_with_laser_body)

    assert reading.reference_frame is world_with_laser_body.root


def test_converted_scan_carries_the_pattern_the_message_declares(
    world_with_laser_body,
):
    scan = laser_scan()

    reading = LaserScanToSemDTConverter.convert(scan, world_with_laser_body)

    assert reading.scan_pattern == ScanPattern(
        minimum_angle=scan.angle_min,
        maximum_angle=scan.angle_max,
        angle_increment=scan.angle_increment,
        minimum_range=scan.range_min,
        maximum_range=scan.range_max,
    )


def test_scan_whose_range_count_disagrees_with_its_angles_is_rejected(
    world_with_laser_body,
):
    with pytest.raises(LaserScanBeamCountMismatch):
        LaserScanToSemDTConverter.convert(
            laser_scan(ranges=[1.0, 2.0]), world_with_laser_body
        )


def test_converter_is_found_by_the_registry(world_with_laser_body):
    scan = laser_scan()

    assert Ros2ToSemDTConverter.get_to_converter(scan) is LaserScanToSemDTConverter


# %% the lidar a subscribed source feeds

TOPIC_NAME = "/scan"
"""
The topic the subscribed sources below listen on.
"""


@dataclass(eq=False)
class RootMountedLidar(Lidar):
    """
    A lidar mounted on the world's root body, sweeping :data:`DECLARED_PATTERN` until a
    real scan says otherwise.
    """

    @classmethod
    def with_source(
        cls, robot_root: KinematicStructureEntity, source: LidarSource
    ) -> Self:
        return cls(root=robot_root, scan_pattern=DECLARED_PATTERN, source=source)


def subscribed_lidar(node, world: World) -> RootMountedLidar:
    """
    :return: A lidar on the world's root body, reading what arrives on
        :data:`TOPIC_NAME`.
    """
    return RootMountedLidar.with_source(
        world.root, SubscribedLidarSource(node=node, topic_name=TOPIC_NAME)
    )


def test_a_lidar_given_a_subscribed_source_listens_on_the_given_topic(
    rclpy_node, world_with_laser_body
):
    lidar = subscribed_lidar(rclpy_node, world_with_laser_body)

    assert isinstance(lidar.source, SubscribedLidarSource)
    assert lidar.source.topic_name == TOPIC_NAME
    assert lidar.source.node is rclpy_node


def test_subscribed_source_reports_the_reading_of_its_latest_scan(
    rclpy_node, world_with_laser_body
):
    scan = laser_scan()
    lidar = subscribed_lidar(rclpy_node, world_with_laser_body)
    lidar.source.buffer_message(scan)

    expected = LaserScanToSemDTConverter.convert(scan, world_with_laser_body)
    reading = lidar.get_lidar_reading()

    np.testing.assert_array_equal(reading.ranges, expected.ranges)
    assert reading.scan_pattern == expected.scan_pattern
    assert reading.reference_frame is expected.reference_frame


def test_reading_a_subscribed_source_adopts_the_pattern_of_its_latest_scan(
    rclpy_node, world_with_laser_body
):
    scan = laser_scan()
    lidar = subscribed_lidar(rclpy_node, world_with_laser_body)
    lidar.source.buffer_message(scan)

    lidar.get_lidar_reading()

    assert lidar.scan_pattern == ScanPattern(
        minimum_angle=scan.angle_min,
        maximum_angle=scan.angle_max,
        angle_increment=scan.angle_increment,
        minimum_range=scan.range_min,
        maximum_range=scan.range_max,
    )


def test_subscribed_lidar_source_reads_laser_scan_messages():
    assert SubscribedLidarSource.message_type() is LaserScan


def test_closing_a_subscribed_source_destroys_its_subscription(
    rclpy_node, world_with_laser_body
):
    lidar = subscribed_lidar(rclpy_node, world_with_laser_body)
    subscription = lidar.source.subscription

    lidar.source.close()

    assert subscription not in rclpy_node.subscriptions


def test_a_lidar_sweeps_its_declared_pattern_until_a_scan_arrives(
    rclpy_node, world_with_laser_body
):
    lidar = subscribed_lidar(rclpy_node, world_with_laser_body)

    assert lidar.scan_pattern == DECLARED_PATTERN


def test_a_subscribed_source_without_a_scan_cannot_be_read(
    rclpy_node, world_with_laser_body
):
    lidar = subscribed_lidar(rclpy_node, world_with_laser_body)

    with pytest.raises(NoLaserScanReceived):
        lidar.get_lidar_reading()
