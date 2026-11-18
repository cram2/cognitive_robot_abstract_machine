"""
When a motion reads the forward kinematics a goal is given relative to.
"""

from enum import Enum


class GoalBindingPolicy(Enum):
    """
    This policy should be used together with
    :class:`~giskardpy.motion_statechart.forward_kinematics_binding.ForwardKinematicsBinding`.
    """

    Bind_at_build = 1
    """
    Forward kinematics is only computed once at build time.
    """

    Bind_on_start = 2
    """
    Forward kinematics is computed during on_start of the MotionStatechartNode.
    """
