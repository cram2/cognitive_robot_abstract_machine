import inspect
from dataclasses import dataclass

from rclpy_message_converter.message_converter import (
    convert_ros_message_to_dictionary,
    convert_dictionary_to_ros_message,
)
from typing_extensions import Dict, Type, Any

from krrood.adapters.exceptions import JSON_TYPE_NAME
from krrood.adapters.json_serializer import ExternalClassJSONSerializer
from krrood.utils import get_full_class_name
from semantic_digital_twin.adapters.ros.utils import is_ros2_message_class


@dataclass
class Ros2MessageJSONSerializer(ExternalClassJSONSerializer[None]):
    """
    Json serializer for ROS2 messages.
    Since there is no common superclass for ROS2 messages, we need to rely on checking class fields instead.
    That's also why T is set to None.
    """

    @classmethod
    def to_json(cls, obj: Any) -> Dict[str, Any]:
        return {
            JSON_TYPE_NAME: get_full_class_name(obj.__class__),
            "data": convert_ros_message_to_dictionary(obj),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type, **kwargs) -> Any:
        return convert_dictionary_to_ros_message(clazz, data["data"], **kwargs)

    @classmethod
    def matches_generic_type(cls, clazz: Type):
        return is_ros2_message_class(clazz)
