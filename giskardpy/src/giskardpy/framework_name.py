"""
The name of the framework that ticks the nodes of giskardpy.
"""

from __future__ import annotations

from enum import StrEnum


class MotionFramework(StrEnum):
    """
    The frameworks of giskardpy, as
    :attr:`~cramph.node.StatechartNode.ticking_framework` names them.
    """

    GISKARD = "giskard"
    """
    Ticks the motion statechart nodes, see
    :class:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode`.
    """
