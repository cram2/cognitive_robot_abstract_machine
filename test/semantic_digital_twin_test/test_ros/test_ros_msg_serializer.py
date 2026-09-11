# %% imports
from geometry_msgs.msg import WrenchStamped
from rclpy.duration import Duration
from rclpy.qos import QoSProfile

from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.adapters.ros.ros_msg_serializer import (
    Ros2MessageJSONSerializer,
)


# %% ROS 2 message serialization
def test_ros2_message_serializer_accepted_kwargs_cached_property():
    accepted_kwargs = Ros2MessageJSONSerializer._ACCEPTED_CONVERT_KWARGS
    assert isinstance(accepted_kwargs, set)
    assert "kind" in accepted_kwargs
    assert "strict_mode" in accepted_kwargs
    assert "check_missing_fields" in accepted_kwargs
    assert Ros2MessageJSONSerializer._ACCEPTED_CONVERT_KWARGS is accepted_kwargs
    assert Ros2MessageJSONSerializer()._ACCEPTED_CONVERT_KWARGS is accepted_kwargs


def test_ros2_message_serialization_round_trip():
    message = WrenchStamped()
    message.wrench.force.x = 20.0
    message.wrench.torque.z = 5.0
    message.header.frame_id = "sensor_frame"

    serialized = to_json(message)
    deserialized = from_json(serialized)

    assert deserialized == message


def test_ros2_message_deserialization_ignores_extraneous_kwargs():
    message = WrenchStamped()
    message.wrench.force.x = 20.0

    serialized = to_json(message)
    deserialized = from_json(
        serialized,
        unexpected_tracker="some_tracker_instance",
        another_unsupported_argument=42,
    )

    assert deserialized == message


# %% QoSProfile serialization
def test_qos_profile_serialization_round_trip():
    qos = QoSProfile(depth=10)
    serialized = to_json(qos)
    deserialized = from_json(serialized)

    assert deserialized == qos


# %% Duration serialization
def test_duration_serialization_round_trip():
    duration = Duration(nanoseconds=123456789)
    serialized = to_json(duration)
    deserialized = from_json(serialized)

    assert deserialized == duration
