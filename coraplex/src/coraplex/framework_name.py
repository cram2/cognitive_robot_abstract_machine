"""
The name of the framework that ticks the nodes of coraplex.
"""

from __future__ import annotations

from enum import StrEnum


class PlanFramework(StrEnum):
    """
    The frameworks of coraplex, as :attr:`~cramph.node.StatechartNode.ticking_framework`
    names them.
    """

    CORAPLEX = "coraplex"
    """
    Ticks the actions of robot plans, see
    :class:`~coraplex.robot_plans.actions.base.Action`.
    """
