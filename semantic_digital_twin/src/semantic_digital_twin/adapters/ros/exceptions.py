from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import UsageError

if TYPE_CHECKING:
    from semantic_digital_twin.adapters.ros.messages import MetaData


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

    data_type: type = field(kw_only=True)
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

    data_type: type = field(kw_only=True)
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


# %% world updates that cannot be applied


@dataclass
class WorldUpdateReferencesUnknownEntityError(UsageError):
    """
    Raised when an update refers to an entity this world never received, which leaves
    everything the missing update carried out of reach.
    """

    publisher: MetaData
    """
    The synchronizer whose update could not be applied.
    """

    entity_id: UUID
    """
    The entity the update refers to.

    Only its id is known, because the update that created it never arrived.
    """

    entity_name: PrefixedName | None
    """
    The name the update calls that entity, or ``None`` where it carries none.
    """

    def error_message(self) -> str:
        named_entity = (
            f"'{self.entity_name}' ({self.entity_id})"
            if self.entity_name is not None
            else f"'{self.entity_id}'"
        )
        return (
            f"The update of '{self.publisher.node_name}' refers to the entity "
            f"{named_entity}, which this world never received."
        )

    def suggest_correction(self) -> str:
        return (
            "Create the synchronizer of a world before modifying that world: changes made "
            "before it exists reach nobody, and every later change that builds on them "
            "cannot be applied."
        )
