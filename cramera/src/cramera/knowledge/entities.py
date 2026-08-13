"""
The recorded episode's entity model: robot parts, objects, episodes and joint motion.
"""

from __future__ import annotations

from dataclasses import dataclass

from cramera.knowledge.entity import NamedEntity

from typing_extensions import Optional

from coraplex.datastructures.enums import Arms
from semantic_digital_twin.spatial_types import Point3

from cramera.knowledge.enums import JointRegion


@dataclass(unsafe_hash=True)
class Gripper(NamedEntity):
    """
    An end effector of the recorded robot.
    """

    side: Optional[Arms]
    """
    Which arm the gripper belongs to, or None for a robot that does not specify a left
    and a right arm.
    """

    opening_metres: Optional[float] = None
    """
    Maximum opening width in metres as recorded by the onboarder, or None when the
    bundle does not report one.
    """


@dataclass(unsafe_hash=True)
class Arm(NamedEntity):
    """
    A manipulator of the recorded robot.
    """

    side: Optional[Arms]
    """
    Which arm this is, or None for a robot that does not specify a left and a right arm.
    """

    robot: str
    """
    Name of the robot this arm belongs to.
    """

    gripper: Gripper
    """
    The end effector mounted on this arm.
    """


@dataclass(unsafe_hash=True)
class Robot(NamedEntity):
    """
    The robot that executed the recorded episode.
    """

    arm_count: int
    """
    Number of annotated arms.
    """


@dataclass(eq=False)
class BenchObject(NamedEntity):
    """
    A loose object (or named location) in the scene.
    """

    kind: str
    """
    ``object`` for graspable things, ``location`` for named areas.
    """

    label: str
    """
    Human-readable display name.
    """

    height_metres: Optional[float]
    """
    Object height in metres as recorded by the onboarder, or None when the bundle does
    not report one (the object's shapes carry no measurable size).
    """

    position: Point3
    """
    Spawn position recorded at frame 0 of the episode.
    """

    def _comparison_key(self) -> tuple:
        """
        Field values, with :attr:`position` reduced to its plain coordinates.

        :class:`Point3` compares by identity and hashes its CasADi expression, neither
        of which reflects the recorded coordinates, so equality/hashing here must read
        the coordinates out explicitly instead of deferring to :class:`Point3` itself.
        """
        return (
            self.name,
            self.kind,
            self.label,
            self.height_metres,
            tuple(self.position.to_np().tolist()),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BenchObject):
            return NotImplemented
        return self._comparison_key() == other._comparison_key()

    def __hash__(self) -> int:
        return hash(self._comparison_key())


@dataclass(unsafe_hash=True)
class ActionEpisode(NamedEntity):
    """
    One executed plan segment of the recording.
    """

    index: int
    """
    Position of the episode in execution order.
    """

    start_frame: int
    """
    First trajectory frame of the episode.
    """

    end_frame: int
    """
    Frame after the last trajectory frame of the episode.
    """

    duration_seconds: float
    """
    Episode duration in seconds.
    """

    performed_by: Optional[Arm]
    """
    The arm that performed the manipulation, if any.
    """

    picks: Optional[BenchObject]
    """
    The object the episode picks up, if any.
    """

    places_at: Optional[BenchObject]
    """
    The location the object is placed at, if any.
    """


@dataclass(unsafe_hash=True)
class JointMotion(NamedEntity):
    """
    Per-joint motion statistics over the whole recorded trajectory.
    """

    region: JointRegion
    """
    Region of the robot/scene the joint belongs to.
    """

    minimum_radians: float
    """
    Smallest recorded joint position (radians or metres).
    """

    maximum_radians: float
    """
    Largest recorded joint position (radians or metres).
    """

    range_radians: float
    """Travelled range, ``maximum_radians - minimum_radians``."""
