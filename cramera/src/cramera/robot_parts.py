"""
The semantic_digital_twin robot-part annotations of a world, in the form the recorded
scene bundles and the live bridge both publish them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Any, Dict, List, Optional

from semantic_digital_twin.robots.robot_parts import AbstractRobot, AbstractRobotPart

# %% the published shape of a robot part


class ArmSide(StrEnum):
    """
    Which of a robot's two arms a part belongs to.

    semantic_digital_twin encodes handedness structurally, through
    :meth:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.get_left_arm_if_specified`,
    rather than as an enum. coraplex's :class:`~coraplex.datastructures.enums.Arms` is
    not reused here because :mod:`cramera.live.bridge` reads this module and has to
    stay importable outside a demo environment.
    """

    LEFT = "left"
    RIGHT = "right"


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
    Whether the part is an arm or an end effector.
    """

    side: Optional[ArmSide]
    """
    Which arm of the robot the part belongs to, or None for a robot that does not
    specify a left and a right arm.
    """

    links: List[str] = field(default_factory=list)
    """
    Link names of the part, stripped of their model-name prefix. An arm's links exclude
    those of its own end effector.
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
            "side": self.side.value if self.side is not None else None,
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
            side=ArmSide(side) if side else None,
            links=list(payload.get("links") or []),
            attached_to=payload.get("attachedTo"),
        )

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
    def _arm_sides(robot: AbstractRobot) -> Dict[int, ArmSide]:
        """
        The side of every arm the robot names as its left or its right one, keyed by arm
        identity.

        Robots that do not specify a left and a right arm contribute nothing, which is what
        leaves a one-armed robot's arm sideless.

        :param robot: The robot whose arm annotations are read.
        """
        sides = {}
        left_arm = robot.get_left_arm_if_specified()
        if left_arm is not None:
            sides[id(left_arm)] = ArmSide.LEFT
        right_arm = robot.get_right_arm_if_specified()
        if right_arm is not None:
            sides[id(right_arm)] = ArmSide.RIGHT
        return sides

    @classmethod
    def of_robot(cls, robot: AbstractRobot) -> List[RobotPartAnnotation]:
        """
        Every arm of a robot and the end effector it carries, in publication order.

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
        return annotations


# %% reading them off a world's robot
