"""
The semantic_digital_twin robot-part annotations of a world, in the form the recorded
scene bundles and the live bridge both publish them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Any, Dict, List, Optional

from coraplex.datastructures.enums import Arms
from semantic_digital_twin.robots.hsrb import HSRBArm, HSRBGripper
from semantic_digital_twin.robots.pr2 import (
    PR2LeftArm,
    PR2RightArm,
    PR2LeftGripper,
    PR2RightGripper,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot, AbstractRobotPart
from semantic_digital_twin.robots.stretch import StretchArm, StretchGripper
from semantic_digital_twin.robots.tracy import (
    TracyLeftArm,
    TracyRightArm,
    TracyLeftGripper,
    TracyRightGripper,
)

# %% the published shape of a robot part


ArmSide = Arms  # Preserve the published import name without a second side enum.


class RobotPartRole(StrEnum):
    """
    What a robot part is, as far as the viewer and the knowledge base care.
    """

    ARM = "arm"
    """
    A :class:`semantic_digital_twin.robots.robot_parts.Arm` annotation.
    """

    END_EFFECTOR = "end_effector"
    """
    A :class:`semantic_digital_twin.robots.robot_parts.EndEffector` annotation.
    """

    SENSOR = "sensor"


class LegacyRobotPart(StrEnum):
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    CENTER_ARM = "center_arm"
    ARM = "arm"
    LEFT_GRIPPER = "left_gripper"
    RIGHT_GRIPPER = "right_gripper"
    GRIPPER = "gripper"

    @classmethod
    def of_name(cls, name: str) -> LegacyRobotPart | None:
        if name in cls:
            return cls(name)
        return {
            PR2LeftArm.__name__: cls.LEFT_ARM,
            PR2RightArm.__name__: cls.RIGHT_ARM,
            PR2LeftGripper.__name__: cls.LEFT_GRIPPER,
            PR2RightGripper.__name__: cls.RIGHT_GRIPPER,
            TracyLeftArm.__name__: cls.LEFT_ARM,
            TracyRightArm.__name__: cls.RIGHT_ARM,
            TracyLeftGripper.__name__: cls.LEFT_GRIPPER,
            TracyRightGripper.__name__: cls.RIGHT_GRIPPER,
            StretchArm.__name__: cls.ARM,
            StretchGripper.__name__: cls.GRIPPER,
            HSRBArm.__name__: cls.ARM,
            HSRBGripper.__name__: cls.GRIPPER,
        }.get(name)

    @property
    def side(self) -> Arms | None:
        return {
            self.LEFT_ARM: Arms.LEFT,
            self.LEFT_GRIPPER: Arms.LEFT,
            self.RIGHT_ARM: Arms.RIGHT,
            self.RIGHT_GRIPPER: Arms.RIGHT,
        }.get(self)

    @property
    def arm(self) -> LegacyRobotPart | None:
        return {
            self.LEFT_GRIPPER: self.LEFT_ARM,
            self.RIGHT_GRIPPER: self.RIGHT_ARM,
            self.GRIPPER: self.ARM,
        }.get(self)

    @property
    def role(self) -> RobotPartRole:
        return RobotPartRole.END_EFFECTOR if self.arm is not None else RobotPartRole.ARM


@dataclass
class RobotPartAnnotation:
    """
    One robot-part annotation of a world, reduced to what survives serialization.

    The knowledge base and the viewer never see the sem_dt annotation objects
    themselves, so this carries the facts they would otherwise have to guess from part
    and link names.
    """

    name: str
    """
    The sem_dt annotation class name, e.g. ``PR2LeftArm``.
    """

    role: RobotPartRole
    """
    Whether the part is an arm, end effector, or sensor.
    """

    side: Optional[Arms]
    """
    Which arm of the robot the part belongs to, or None for a robot that does not
    specify a left and a right arm.
    """

    links: List[str] = field(default_factory=list)
    """
    Link names of the part, stripped of their model-name prefix.

    An arm's links exclude those of its own end effector.
    """

    attached_to: Optional[str] = None
    """
    For an end effector, the name of the arm carrying it; None for an arm.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The annotation in the JSON shape written to ``scene.json`` and served live.
        """
        return {
            "name": self.name,
            "role": self.role.value,
            "side": self.side.name.lower() if self.side is not None else None,
            "links": list(self.links),
            "attachedTo": self.attached_to,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> RobotPartAnnotation:
        """
        Read back an annotation written by :meth:`to_payload`.

        :param payload: One entry of a bundle's ``robot.partAnnotations`` list.
        """
        side = payload.get("side")
        return cls(
            name=payload["name"],
            role=RobotPartRole(payload["role"]),
            side=Arms[side.upper()] if side is not None else None,
            links=list(payload.get("links") or []),
            attached_to=payload.get("attachedTo"),
        )

    @classmethod
    def of_recording(cls, robot: Dict[str, Any]) -> List[RobotPartAnnotation]:
        if "partAnnotations" in robot:
            return [
                cls.from_payload(payload) for payload in robot["partAnnotations"] or []
            ]
        annotations = []
        parts = robot.get("parts") or {}
        legacy_parts = {name: LegacyRobotPart.of_name(name) for name in parts}
        for name, links in parts.items():
            part = legacy_parts[name]
            if part is None:
                continue
            attached_to = None
            if part.arm is not None:
                attached_to = next(
                    (
                        name
                        for name, candidate in legacy_parts.items()
                        if candidate is part.arm
                    ),
                    None,
                )
                if (
                    attached_to is None
                    and part is LegacyRobotPart.GRIPPER
                    and LegacyRobotPart.CENTER_ARM in parts
                ):
                    attached_to = LegacyRobotPart.CENTER_ARM.value
            annotations.append(
                cls(
                    name=name,
                    role=part.role,
                    side=part.side,
                    links=list(links),
                    attached_to=attached_to,
                )
            )
        return annotations

    @staticmethod
    def link_names(part: AbstractRobotPart) -> List[str]:
        """
        A robot part's link names, stripped of their model-name prefix.

        :param part: The robot part whose link names are read.
        """
        names = []
        for body in part.bodies or []:
            name = str(body.name)
            names.append(name.split("/", 1)[1] if "/" in name else name)
        return names

    @staticmethod
    def _arm_sides(robot: AbstractRobot) -> Dict[int, Arms]:
        """
        The side of every arm the robot names as its left or its right one, keyed by arm
        identity.

        Robots that do not specify a left and a right arm contribute nothing, which is
        what leaves a one-armed robot's arm sideless.

        :param robot: The robot whose arm annotations are read.
        """
        sides = {}
        left_arm = robot.get_left_arm_if_specified()
        if left_arm is not None:
            sides[id(left_arm)] = Arms.LEFT
        right_arm = robot.get_right_arm_if_specified()
        if right_arm is not None:
            sides[id(right_arm)] = Arms.RIGHT
        return sides

    @classmethod
    def of_robot(cls, robot: AbstractRobot) -> List[RobotPartAnnotation]:
        """
        Recorded arms, their end effectors, and the robot's native sensors.

        :param robot: The robot annotation of the world being recorded or served.
        """
        sides = cls._arm_sides(robot)
        annotations = []
        for arm in robot.get_arms():
            arm_name = type(arm).__name__
            side = sides.get(id(arm))
            end_effector = arm.end_effector
            end_effector_links = (
                cls.link_names(end_effector) if end_effector is not None else []
            )
            annotations.append(
                cls(
                    name=arm_name,
                    role=RobotPartRole.ARM,
                    side=side,
                    links=sorted(set(cls.link_names(arm)) - set(end_effector_links)),
                )
            )
            if end_effector is not None:
                annotations.append(
                    cls(
                        name=type(end_effector).__name__,
                        role=RobotPartRole.END_EFFECTOR,
                        side=side,
                        links=sorted(set(end_effector_links)),
                        attached_to=arm_name,
                    )
                )
        annotations.extend(
            cls(
                name=type(sensor).__name__,
                role=RobotPartRole.SENSOR,
                side=None,
                links=sorted(set(cls.link_names(sensor))),
            )
            for sensor in robot.get_sensors()
        )
        return annotations


# %% identifying a model within a world
