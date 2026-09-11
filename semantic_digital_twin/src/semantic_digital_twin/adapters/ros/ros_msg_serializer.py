import inspect
from dataclasses import dataclass
from functools import lru_cache

from rclpy.duration import Duration
from rclpy.qos import QoSProfile
from rclpy_message_converter.message_converter import (
    convert_ros_message_to_dictionary,
    convert_dictionary_to_ros_message,
)
from typing_extensions import Dict, Type, Any

from krrood.adapters.exceptions import JSON_TYPE_NAME
from krrood.adapters.json_serializer import (
    ExternalClassJSONSerializer,
    to_json,
    from_json,
)
from krrood.ormatic.utils import classproperty
from krrood.utils import get_full_class_name
from semantic_digital_twin.adapters.ros.utils import is_ros2_message_class

# %% ROS 2 message serializer


@dataclass
class Ros2MessageJSONSerializer(ExternalClassJSONSerializer[None]):
    """
    Json serializer for ROS2 messages.

    Since there is no common superclass for ROS2 messages, we need to rely on checking
    class fields instead. That's also why T is set to None.
    """

    @classproperty
    @lru_cache
    def _ACCEPTED_CONVERT_KWARGS(cls) -> set[str]:
        """
        Accepted keyword arguments for convert_dictionary_to_ros_message.
        """
        return set(
            inspect.signature(convert_dictionary_to_ros_message).parameters.keys()
        )

    @classmethod
    def to_json(cls, obj: Any, **kwargs) -> Dict[str, Any]:
        """
        Serialize a ROS 2 message into a JSON-compatible dictionary.
        """
        return {
            JSON_TYPE_NAME: get_full_class_name(obj.__class__),
            "data": convert_ros_message_to_dictionary(obj),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type, **kwargs: Any) -> Any:
        """
        Deserialize a JSON-compatible dictionary into a ROS 2 message.
        """
        valid_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in cls._ACCEPTED_CONVERT_KWARGS
        }
        return convert_dictionary_to_ros_message(clazz, data["data"], **valid_kwargs)

    @classmethod
    def matches_generic_type(cls, clazz: Type) -> bool:
        """
        Check if the given class is a ROS 2 message class.
        """
        return is_ros2_message_class(clazz)


# %% QoS profile serializer


@dataclass
class QoSProfileJSONSerializer(ExternalClassJSONSerializer[QoSProfile]):
    """
    A serializer class for converting a QoSProfile instance to and from JSON format.

    All fields of the QoSProfile are saved in its `__slots__` attribute with a `_`
    prefix.
    """

    @classmethod
    def to_json(cls, obj: QoSProfile, **kwargs) -> Dict[str, Any]:
        """
        Serialize a QoSProfile into a JSON-compatible dictionary.
        """
        return {
            JSON_TYPE_NAME: get_full_class_name(obj.__class__),
            **{
                field_name: to_json(getattr(obj, field_name), **kwargs)
                for field_name in obj.__slots__
            },
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type[QoSProfile], **kwargs) -> Any:
        """
        Deserialize a JSON-compatible dictionary into a QoSProfile.
        """
        return clazz(
            **{
                field_name[1:]: from_json(data[field_name])
                for field_name in clazz.__slots__
            }
        )


# %% Duration serializer


@dataclass
class DurationJSONSerializer(ExternalClassJSONSerializer[Duration]):
    """
    Serializer for converting Duration objects to and from JSON format.
    """

    @classmethod
    def to_json(cls, obj: Duration, **kwargs) -> Dict[str, Any]:
        """
        Serialize a Duration into a JSON-compatible dictionary.
        """
        return {
            JSON_TYPE_NAME: get_full_class_name(obj.__class__),
            "nanoseconds": obj.nanoseconds,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type, **kwargs) -> Any:
        """
        Deserialize a JSON-compatible dictionary into a Duration.
        """
        return clazz(nanoseconds=data["nanoseconds"])
