"""
The recorded episode's entity model: robot parts, objects, episodes and joint motion.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Optional

from cram_viz.knowledge.enums import ArmSide


@dataclass(unsafe_hash=True)
class Position:
    """
    A world position in metres.
    """

    x: float
    """
    World x coordinate.
    """

    y: float
    """
    World y coordinate.
    """

    z: float
    """
    World z coordinate.
    """

    def __repr__(self) -> str:
        return "(%.2f, %.2f, %.2f)" % (self.x, self.y, self.z)


@dataclass(unsafe_hash=True)
class Gripper:
    """
    An end effector of the recorded robot.
    """

    name: str
    """
    Part name from the scene's robot annotation.
    """

    side: ArmSide
    """
    Body side the gripper belongs to.
    """

    opening_m: Optional[float] = None
    """
    Maximum opening width in metres as recorded by the onboarder, or None when the
    bundle does not report one.
    """


@dataclass(unsafe_hash=True)
class Arm:
    """
    A manipulator of the recorded robot.
    """

    name: str
    """
    Part name from the scene's robot annotation.
    """

    side: ArmSide
    """
    Body side of the arm.
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
class Robot:
    """
    The robot that executed the recorded episode.
    """

    name: str
    """
    Robot name from the scene bundle.
    """

    arm_count: int
    """
    Number of annotated arms.
    """


@dataclass(unsafe_hash=True)
class BenchObject:
    """
    A loose object (or named location) in the scene.
    """

    name: str
    """
    Object identifier, e.g. ``milk``.
    """

    kind: str
    """
    ``object`` for graspable things, ``location`` for named areas.
    """

    label: str
    """
    Human-readable display name.
    """

    height_m: Optional[float]
    """
    Object height in metres as recorded by the onboarder, or None when the bundle does
    not report one (the object's shapes carry no measurable size).
    """

    position: Position
    """
    Spawn position recorded at frame 0 of the episode.
    """


@dataclass(unsafe_hash=True)
class ActionEpisode:
    """
    One executed plan segment of the recording.
    """

    name: str
    """
    Segment step name, e.g. ``transport_milk``.
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

    duration_s: float
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
class JointMotion:
    """
    Per-joint motion statistics over the whole recorded trajectory.
    """

    name: str
    """
    Joint name (without the model prefix).
    """

    arm_side: ArmSide
    """
    Body side the joint belongs to.
    """

    min_rad: float
    """
    Smallest recorded joint position (radians or metres).
    """

    max_rad: float
    """
    Largest recorded joint position (radians or metres).
    """

    range_rad: float
    """Travelled range, ``max_rad - min_rad``."""
