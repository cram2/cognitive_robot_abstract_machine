from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

from typing_extensions import ClassVar, Generic, TypeVar, Any, get_args

import semantic_digital_twin.adapters.ros as ros_package
from krrood.utils import recursive_subclasses
from semantic_digital_twin.adapters.ros.exceptions import (
    CannotConvertRos2ToSemDTError,
    CannotConvertSemDTToRos2Error,
    ROS2ConversionError,
)
from semantic_digital_twin.world import World

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


# %% finding the converter for a message


@dataclass
class MessageConverter(ABC, Generic[InputType, OutputType]):
    """
    Base class for converters between ROS2 messages and their semDT representation.

    If you want to add a new converter, subclass one of the two directions below and
    override the convert method. No registration is necessary.
    """

    converter_module_suffix: ClassVar[str] = "_converters"
    """
    The name ending that marks a module of this package as defining converters.
    """

    _converter_modules_loaded: ClassVar[bool] = False
    """
    Whether the modules defining the converters have already been imported.
    """

    @classmethod
    @property
    def input_type(cls) -> Type[InputType]:
        """
        The type this converter reads.
        """
        return get_args(cls.__orig_bases__[0])[0]

    @classmethod
    @property
    def output_type(cls) -> Type[OutputType]:
        """
        The type this converter writes.
        """
        return get_args(cls.__orig_bases__[0])[1]

    @classmethod
    @abstractmethod
    def conversion_error(cls) -> Type[ROS2ConversionError]:
        """
        :return: The error raised when no converter of this direction fits.
        """

    @classmethod
    def can_convert(cls, data: Any) -> bool:
        """
        Checks whether this converter can convert the given object.

        Override this if you want to customize the conversion check.

        :param data: The object to check conversion for.
        :return: True if this converter can handle the conversion, False otherwise.
        """
        return cls.input_type == type(data)

    @classmethod
    def _load_converter_modules(cls) -> None:
        """
        Imports every module of this package whose name ends in
        :attr:`converter_module_suffix`, so that the converters they define are
        discoverable.

        ..note:: The import happens here rather than at module level because those
            modules import this one.
        """
        if MessageConverter._converter_modules_loaded:
            return
        MessageConverter._converter_modules_loaded = True
        for module in pkgutil.iter_modules(
            ros_package.__path__, ros_package.__name__ + "."
        ):
            if module.name.endswith(cls.converter_module_suffix):
                importlib.import_module(module.name)

    @classmethod
    def get_to_converter(cls, input_obj: Any) -> Type[MessageConverter]:
        """
        :param input_obj: The object to find a converter for.
        :return: The subclass of this direction that converts the given object.
        :raises ROS2ConversionError: If no converter of this direction fits.
        """
        cls._load_converter_modules()
        for sub_class in recursive_subclasses(cls):
            if sub_class.can_convert(input_obj):
                return sub_class
        raise cls.conversion_error()(data_type=type(input_obj))


@dataclass
class Ros2ToSemDTConverter(MessageConverter[InputType, OutputType], ABC):
    """
    Base class for converters that convert ROS2 messages to their semDT representation.
    """

    @classmethod
    def conversion_error(cls) -> Type[ROS2ConversionError]:
        return CannotConvertRos2ToSemDTError

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
class SemDTToRos2Converter(MessageConverter[InputType, OutputType], ABC):
    """
    Base class for converters that convert semDT objects to their ROS2 message
    representation.
    """

    @classmethod
    def conversion_error(cls) -> Type[ROS2ConversionError]:
        return CannotConvertSemDTToRos2Error

    @classmethod
    def convert(cls, data: InputType) -> OutputType:
        """
        Converts the given semDT object to its ROS2 message representation.

        :param data: The semDT object to convert.
        :return: The ROS2 message representation of the given semDT object.
        """
        return cls.get_to_converter(data).convert(data)
