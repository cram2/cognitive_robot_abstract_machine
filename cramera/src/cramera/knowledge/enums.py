"""
Enums shared across the knowledge package.
"""

from __future__ import annotations

from enum import StrEnum

from typing_extensions import Optional, Tuple, Union


class JointRegion(StrEnum):
    """
    Which region of the robot/scene a joint belongs to, as inferred from its name.

    Arms and grippers use :class:`coraplex.datastructures.enums.Arms` instead, since
    they always name a specific arm; a joint can also belong to the robot's body or to
    the environment, which :class:`~coraplex.datastructures.enums.Arms` has no member
    for.
    """

    LEFT = "left"
    RIGHT = "right"
    BODY = "body"
    ENVIRONMENT = "environment"


class NodeGroup(StrEnum):
    """
    Colour group of a graph-panel node.
    """

    ROBOT = "robot"
    OBJECT = "object"
    EVENT = "event"
    ROOT = "root"
    PACKAGE = "package"
    SUBPACKAGE = "subpackage"
    PLAN = "plan"
    PYTHON_CLASS = "python_class"
    EXTERNAL_CLASS = "external_class"
    OTHER = "other"


class EdgeKind(StrEnum):
    """
    Rendering kind of a graph-panel edge.
    """

    PROPERTY = "property"
    TYPE = "type"


class LabelledGroup(StrEnum):
    """
    A colour group that carries the text its legend row shows.

    Keeping both on the member is what stops a view from listing its groups twice: once
    to classify nodes and once to build the legend.
    """

    def __new__(cls, value: str, label: str) -> "LabelledGroup":
        member = str.__new__(cls, value)
        member._value_ = value
        member.label = label
        return member

    @classmethod
    def legend(cls) -> Tuple["LabelledGroup", ...]:
        """
        Every group of this kind, in declaration order, for a view's legend.
        """
        return tuple(cls)


class KinematicChainGroup(LabelledGroup):
    """
    Colour group of a link in the robot's kinematic tree.

    Separate from :class:`NodeGroup`, whose members name ontological categories of the
    knowledge graph: a right arm is not an "event", it just needs a colour of its own.
    """

    BASE = ("base", "Base / torso")
    LEFT_ARM = ("left_arm", "Left arm")
    RIGHT_ARM = ("right_arm", "Right arm")
    GRIPPER = ("gripper", "Grippers")
    SENSOR = ("sensor", "Head / sensors")


class PlanNodeGroup(LabelledGroup):
    """
    Colour group of a node in the executed plan tree.

    Separate from :class:`NodeGroup` for the same reason as
    :class:`KinematicChainGroup`: a motion is not a robot and a condition is not a goal,
    they are kinds of plan node that each need a colour of their own.
    """

    ACTION = ("action", "Action")
    MOTION = ("motion", "Motion")
    CONDITION = ("condition", "Condition")
    ATTACHMENT = ("attachment", "Attach / detach")
    OTHER = ("other_plan_node", "Other plan node")

    @classmethod
    def of_plan_node_kind(cls, kind: Optional[str]) -> "PlanNodeGroup":
        """
        The group a coraplex plan-node class belongs to.

        :param kind: The plan node's own class name, as recorded or observed live.
        """
        return {
            "ActionNode": cls.ACTION,
            "MotionNode": cls.MOTION,
            "ConditionNode": cls.CONDITION,
            "AttachNode": cls.ATTACHMENT,
            "DetachNode": cls.ATTACHMENT,
        }.get(kind or "", cls.OTHER)


ColourGroup = Union[NodeGroup, KinematicChainGroup, PlanNodeGroup]
"""
Any colour group a graph-panel node can carry: an ontological one from the knowledge
graph, or a kinematic-chain one from the URDF tree.
"""
