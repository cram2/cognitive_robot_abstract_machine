from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Type

from typing_extensions import Generic, TypeVar, Any, get_args

from krrood.utils import recursive_subclasses, DataclassException
from ...world import World

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


@dataclass
class ROS2ConversionError(DataclassException):
    """Base class for errors that occur during ROS2 message conversion."""

    message: str = field(kw_only=True, default=None, init=False)


@dataclass
class CannotConvertSemDTToRos2Error(ROS2ConversionError):
    """Raised when a semDT object cannot be converted to a ROS2 message."""

    data_type: Type = field(kw_only=True)

    def __post_init__(self):
        self.message = f"Cannot convert {self.data_type.__name__} to ROS2 message."


@dataclass
class CannotConvertRos2ToSemDTError(ROS2ConversionError):
    """Raised when a ROS2 message cannot be converted to a semDT object."""

    data_type: Type = field(kw_only=True)

    def __post_init__(self):
        self.message = f"Cannot convert {self.data_type.__name__} to our semDT type."


@dataclass
class Ros2ToSemDTConverter(ABC, Generic[InputType, OutputType]):
    """
    Base class for converters that convert ROS2 messages to their semDT representation.
    If you want to add a new converter, subclass this class and override the convert method.
    No registration is necessary.
    """

    @classmethod
    @property
    def input_type(cls) -> Type[InputType]:
        """
        The semDT type for which this converter handles conversion.
        """
        return get_args(cls.__orig_bases__[0])[0]

    @classmethod
    @property
    def output_type(cls) -> Type[InputType]:
        """
        The ROS2 message type for which this converter handles conversion.
        """
        return get_args(cls.__orig_bases__[0])[1]

    @classmethod
    def can_convert(cls, data: Any) -> bool:
        """
        Checks whether this converter can convert the given ROS2 message.
        Override this if you want to customize the conversion check.
        :param data: The ROS2 message to check conversion for.
        :return: True if this converter can handle the conversion, False otherwise.
        """
        return cls.input_type == type(data)

    @classmethod
    def get_to_converter(cls, input_obj: Any) -> Type[Ros2ToSemDTConverter]:
        """
        Recursively checks all subclasses of Ros2ToSemDTConverter to find the converter for the given semDT type.
        :param our_type: The semDT type for which to find the ROS2 converter.
        :return: The Ros2ToSemDTConverter subclass that handles conversion from the given semDT type.
        """
        for sub_class in recursive_subclasses(cls):
            if sub_class.can_convert(input_obj):
                return sub_class
        raise CannotConvertSemDTToRos2Error(data_type=type(input_obj))

    @classmethod
    def convert(cls, data: InputType, world: World) -> OutputType:
        """
        Converts the given ROS2 message to its semDT representation.
        :param data: The ROS2 message to convert.
        :param world: The world in which the semDT object exists.
        :return: The semDT representation of the given ROS2 message.
        """
        return cls.get_to_converter(data).convert(data, world)


@dataclass
class SemDTToRos2Converter(ABC, Generic[InputType, OutputType]):
    """
    Base class for converters that convert semDT objects to their ROS2 message representation.
    If you want to add a new converter, subclass this class and override the convert method.
    No registration is necessary.
    """

    @classmethod
    @property
    def input_type(cls) -> Type[InputType]:
        """
        The semDT type for which this converter handles conversion.
        """
        return get_args(cls.__orig_bases__[0])[0]

    @classmethod
    @property
    def output_type(cls) -> Type[InputType]:
        """
        The ROS2 message type for which this converter handles conversion.
        """
        return get_args(cls.__orig_bases__[0])[1]

    @classmethod
    def can_convert(cls, obj: Any) -> bool:
        """
        Checks whether this converter can convert the given semDT object.
        Override this if you want to customize the conversion check.
        :param obj: The semDT object to check conversion for.
        :return: True if this converter can handle the conversion, False otherwise.
        """
        return cls.input_type == type(obj)

    @classmethod
    def get_to_converter(cls, input_obj: Any) -> Type[SemDTToRos2Converter]:
        """
        Recursively checks all subclasses of SemDTToRos2Converter to find the converter for the given semDT type.
        :param input_type: The semDT type for which to find the ROS2 converter.
        :return: The SemDTToRos2Converter subclass that handles conversion from the given semDT type.
        """
        for sub_class in recursive_subclasses(cls):
            if sub_class.can_convert(input_obj):
                return sub_class
        raise CannotConvertRos2ToSemDTError(data_type=type(input_obj))

    @classmethod
    def convert(cls, data: InputType) -> OutputType:
        """
        Converts the given semDT object to its ROS2 message representation.
        Subclasses should override this method.
        :param data: The semDT object to convert.
        :return: The ROS2 message representation of the given semDT object.
        """
        return cls.get_to_converter(data).convert(data)
