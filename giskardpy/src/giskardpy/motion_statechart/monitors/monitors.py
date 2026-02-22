from __future__ import annotations

import abc
from abc import ABC
from dataclasses import field

import krrood.symbolic_math.symbolic_math as sm
from ..context import MotionStatechartContext
from ..data_types import ObservationStateValues
from ..graph_node import MotionStatechartNode, NodeArtifacts
from ...utils.decorators import dataclass


@dataclass
class ThreadedPayloadMonitor(MotionStatechartNode, ABC):
    """
    A monitor which executes its __call__ function when start_condition becomes True.
    Subclass this and implement __init__.py and __call__. The __call__ method should change self.state to True when
    it's done.
    Calls __call__ in a separate thread. Use for expensive operations
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
    Checks if the robot has reached a local minimum in the trajectory,
    by checking if all velocities are below a degree of freedoms' max velocity *`joint_convergence_threshold`.
    """

    joint_convergence_threshold: float = 0.01
    """
    if a degree of freedom velocity is below its maximum velocity * this value, it is considered as not moving. 
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
        artifacts = NodeArtifacts()

        ref = []
        symbols = []
        for dof in context.world.active_degrees_of_freedom:
            velocity_limit = dof.limits.upper.velocity
            velocity_limit *= self.joint_convergence_threshold
            velocity_limit = min(
                max(self.minimum_threshold, velocity_limit), self.maximum_threshold
            )
            ref.append(velocity_limit)
            symbols.append(dof.variables.velocity)
        ref = sm.Vector(ref)
        vel_symbols = sm.Vector(symbols)

        dt = (
            context.qp_controller_config.control_dt
            or context.qp_controller_config.mpc_dt
        )
        traj_longer_than_1_sec = context.control_cycle_variable * dt > 1
        artifacts.observation = sm.trinary_logic_and(
            traj_longer_than_1_sec, sm.logic_all(sm.abs(vel_symbols) < ref)
        )
        return artifacts
