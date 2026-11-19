"""
Exceptions raised by core robot plan actions.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from coraplex.robot_plans.actions.core.misc import DetectAction
    from krrood_spatial_math.spatial_types import Vector3
    from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
    from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
    from semantic_digital_twin.world_description.connections import (
        ActiveConnection1DOF,
        ScrewConnection,
    )
    from semantic_digital_twin.world_description.world_entity import Body

# %% detecting


@dataclass
class DetectionTargetMissing(DataclassException):
    """
    Raised when a detection is given neither the type of object to look for nor the box
    to look in.
    """

    instance: DetectAction
    """
    The detection that has no target.
    """

    def error_message(self) -> str:
        return f"{self.instance} has neither an object type nor a detection box."

    def suggest_correction(self) -> str:
        return "provide the type of object to detect, a box to detect in, or both."


# %% sending the fingers to a width


@dataclass
class WidthBeyondGripperOpening(DataclassException):
    """
    Raised when the fingers of a gripper are sent to a width wider than the gripper
    opens.
    """

    gripper: HasTwoFingers
    """
    The gripper whose fingers were sent.
    """

    width: float
    """
    How far apart the fingertips' facing surfaces were sent to stand, in metres.
    """

    def error_message(self) -> str:
        return (
            f"{self.gripper.name} opens only {self.gripper.separation_when_open:.3f} m "
            f"wide, but its fingers were sent {self.width:.3f} m apart."
        )

    def suggest_correction(self) -> str:
        return (
            "grasp the object across a side narrower than the gripper opens, or with a "
            "gripper that opens wider."
        )


# %% screwing


@dataclass(kw_only=True)
class MissingSingleDofConnection(DataclassException):
    """
    Raised when a screw body has no single-degree-of-freedom parent connection to drive.
    """

    body: Body
    """
    The body expected to hang from a single-DoF parent connection.
    """

    def error_message(self) -> str:
        return f"{self.body.name} has no single-DoF parent connection to drive."

    def suggest_correction(self) -> str:
        return "connect the screw body to its counterpart with a ScrewConnection."


@dataclass(kw_only=True)
class UnsupportedScrewConnection(DataclassException):
    """
    Raised when joint-driven screwing is given a connection that couples no translation
    to its rotation.
    """

    connection: ActiveConnection1DOF
    """
    The parent connection found on the screw body.
    """

    def error_message(self) -> str:
        return (
            f"{type(self.connection).__name__} couples no translation to its rotation; "
            "joint-driven screwing needs a ScrewConnection."
        )

    def suggest_correction(self) -> str:
        return "use a ScrewConnection between the screw body and its counterpart."


@dataclass(kw_only=True)
class MissingScrewDisplacement(DataclassException):
    """
    Raised when a screw joint goal is requested without either turns or travel distance.
    """

    connection: ScrewConnection
    """
    The screw connection whose goal displacement was omitted.
    """

    def error_message(self) -> str:
        return f"No turns or travel_distance was given for screw connection {self.connection.name}."

    def suggest_correction(self) -> str:
        return "specify either turns or travel_distance."


@dataclass(kw_only=True)
class NegativeScrewTurns(DataclassException):
    """
    Raised when a non-negative revolution count for screwing inwards is given as
    negative.
    """

    turns: float
    """
    The negative revolution count that was given.
    """

    def error_message(self) -> str:
        return f"turns must not be negative, got {self.turns}."

    def suggest_correction(self) -> str:
        return "pass non-negative turns, or pass a signed travel_distance."


@dataclass(kw_only=True)
class ZeroAlignmentAxis(DataclassException):
    """
    Raised when aligning local +Z onto a vector of zero length.
    """

    axis: Vector3
    """
    The zero-length vector passed for alignment.
    """

    def error_message(self) -> str:
        return "Cannot align local +Z with the zero vector."

    def suggest_correction(self) -> str:
        return "pass a non-zero direction vector."


@dataclass(kw_only=True)
class NonToolGraspForScrewing(DataclassException):
    """
    Raised when a screwing action receives a tool grasp whose graspable is not a
    :class:`~semantic_digital_twin.semantic_annotations.semantic_annotations.Tool`.
    """

    graspable: HasGraspCandidates
    """
    The graspable passed as the held tool.
    """

    def error_message(self) -> str:
        return (
            f"Expected tool_grasp.graspable to be a Tool, got "
            f"{type(self.graspable).__name__}."
        )

    def suggest_correction(self) -> str:
        return "pass a GraspCandidate on a Tool such as a Screwdriver."
