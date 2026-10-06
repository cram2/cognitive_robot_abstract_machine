from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from krrood.exceptions import DataclassException


@dataclass
class ROS2ConversionError(DataclassException):
    """
    Base class for errors that occur during ROS2 message conversion.
    """


@dataclass
class CannotConvertSemDTToRos2Error(ROS2ConversionError):
    """
    Raised when a semDT object cannot be converted to a ROS2 message.
    """

    data_type: Type = field(kw_only=True)
    """
    The type of the object that could not be converted.
    """

    def error_message(self) -> str:
        return f"Cannot convert {self.data_type.__name__} to ROS2 message."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class CannotConvertRos2ToSemDTError(ROS2ConversionError):
    """
    Raised when a ROS2 message cannot be converted to a semDT object.
    """

    data_type: Type = field(kw_only=True)
    """
    The type of the message that could not be converted.
    """

    def error_message(self) -> str:
        return f"Cannot convert {self.data_type.__name__} to our semDT type."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class LaserScanBeamCountMismatch(ROS2ConversionError):
    """
    Raised when a laser scan holds a different number of measurements than the beams its
    own angles describe.
    """

    beam_count: int = field(kw_only=True)
    """
    How many beams the scan's angles describe.
    """

    range_count: int = field(kw_only=True)
    """
    How many measurements the scan holds.
    """

    def error_message(self) -> str:
        return f"Laser scan describes {self.beam_count} beams but holds {self.range_count} measurements."

    def suggest_correction(self) -> str:
        return "check that the scan's angle_min, angle_max and angle_increment match its ranges."
