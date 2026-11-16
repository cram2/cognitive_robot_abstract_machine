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


# %% services


@dataclass
class ServiceUnavailableError(DataclassException):
    """
    Raised when a service does not become available in time.
    """

    service_name: str
    """
    The service that was waited for.
    """

    def error_message(self) -> str:
        return f'The service "{self.service_name}" is not available.'

    def suggest_correction(self) -> str:
        return "Make sure the node offering the service is running."


# %% topics and messages


@dataclass
class TopicNotFoundError(DataclassException):
    """
    Raised when no topic of a message type is found on the ROS 2 graph.
    """

    message_type_name: str
    """
    The name ROS 2 gives the message type that was looked for, e.g.
    ``nav_msgs/msg/Odometry``.
    """

    node_name: str | None = None
    """
    The node whose topics were searched, any node if ``None``.
    """

    def error_message(self) -> str:
        if self.node_name is None:
            return f'No topic of type "{self.message_type_name}" was found.'
        return (
            f'The node "{self.node_name}" has no topic of type '
            f'"{self.message_type_name}".'
        )

    def suggest_correction(self) -> str:
        return (
            "Make sure the node that publishes or subscribes to the topic is running, "
            "or name the topic explicitly."
        )


@dataclass
class AmbiguousTopicError(DataclassException):
    """
    Raised when a single topic of a message type was expected but several were found.
    """

    message_type_name: str
    """
    The name ROS 2 gives the message type that was looked for, e.g.
    ``nav_msgs/msg/Odometry``.
    """

    topic_names: list[str]
    """
    The topics of that type that were found.
    """

    def error_message(self) -> str:
        return (
            f'Expected one topic of type "{self.message_type_name}", but found '
            f"{len(self.topic_names)}: {self.topic_names}."
        )

    def suggest_correction(self) -> str:
        return "Name the topic explicitly."


@dataclass
class NoMessageReceivedError(DataclassException):
    """
    Raised when no message arrives on a topic in time.
    """

    topic: str
    """
    The topic that was waited on.
    """

    def error_message(self) -> str:
        return f'No message arrived on "{self.topic}" in time.'

    def suggest_correction(self) -> str:
        return "Make sure a node publishes on the topic, or wait longer."
