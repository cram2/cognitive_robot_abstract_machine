from __future__ import annotations

import abc
from abc import ABC
from dataclasses import field

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    NodeArtifacts,
    velocity_convergence_expression,
)
from giskardpy.utils.decorators import dataclass


@dataclass
class ThreadedPayloadMonitor(MotionStatechartNode, ABC):
    """
    A monitor which executes its __call__ function when start_condition becomes True.

    Subclass this and implement __init__.py and __call__. The __call__ method should
    change self.state to True when it's done. Calls __call__ in a separate thread. Use
    for expensive operations
    """

    state: ObservationStateValues = field(
        init=False, default=ObservationStateValues.UNKNOWN
    )

    @abc.abstractmethod
    def __call__(self):
        pass


@dataclass
class LocalMinimumReached(MotionStatechartNode):
    """
    Checks if the robot has reached a local minimum in the trajectory, by checking if
    all velocities are below a degree of freedoms' max velocity
    *`joint_convergence_threshold`.
    """

    joint_convergence_threshold: float = 0.01
    """
    If a degree of freedom velocity is below its maximum velocity * this value, it is
    considered as not moving.
    """

    minimum_threshold: float = 0.01
    """
    Minimum value for degree of freedom velocity * joint_convergence_threshold.
    """

    maximum_threshold: float = 0.06
    """
    Maximum value for degree of freedom velocity * joint_convergence_threshold.
    """

    windows_size: int = 1
    """
    Windows size for joint convergence check.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(
            observation=velocity_convergence_expression(
                context=context,
                joint_convergence_threshold=self.joint_convergence_threshold,
                minimum_threshold=self.minimum_threshold,
                maximum_threshold=self.maximum_threshold,
            )
        )
