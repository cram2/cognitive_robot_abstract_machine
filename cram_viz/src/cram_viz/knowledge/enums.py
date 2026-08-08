"""
Enums shared across the knowledge package.
"""

from __future__ import annotations

from enum import Enum


class ArmSide(str, Enum):
    """
    Which body side a joint/part belongs to, as inferred from its name.
    """

    LEFT = "left"
    RIGHT = "right"
    BODY = "body"
    ENVIRONMENT = "environment"
    UNKNOWN = "unknown"


class NodeGroup(str, Enum):
    """
    Colour group of a graph-panel node.
    """

    ROBOT = "robot"
    OBJECT = "object"
    EVENT = "event"
    ROOT = "root"
    CONCEPT = "concept"
    KLASS = "klass"
    GOAL = "goal"
    PYCLASS = "pyclass"
    UPPER = "upper"
    OTHER = "ind"


class EdgeKind(str, Enum):
    """
    Rendering kind of a graph-panel edge.
    """

    PROP = "prop"
    TYPE = "type"
