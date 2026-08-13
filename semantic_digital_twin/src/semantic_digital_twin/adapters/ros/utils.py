import inspect

from typing_extensions import Type, Any


def is_ros2_message(obj: Any) -> bool:
    """
    Determines if the given object is a ROS 2 message.

    :param obj: The object to be checked.
    :return: True if the object is from a ROS 2 message module; False otherwise.
    """
    return hasattr(inspect.getmodule(obj), "rosidl_parser")


def is_ros2_message_class(clazz: Type) -> bool:
    """
    Checks if a class is a ROS2 message based on its slots and field types and hope nothing else has the same.
    :param clazz: class to check
    :return: whether it's a ros2 message
    """
    return inspect.isclass(clazz) and is_ros2_message(clazz)
