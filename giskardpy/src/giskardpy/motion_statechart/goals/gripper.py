from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta

from cramph.composites import Parallel
from cramph.node import StatechartNode
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.mixins import (
    HasFingerStallToleranceParameters,
    HasGoalParameter,
    HasJointVelocityLimitParameter,
    HasThresholdParameter,
    HasTwoFingerGripperParameters,
)
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.monitors.progress_monitors import Stalled
from giskardpy.motion_statechart.tasks.cartesian_tasks import DistanceAlongAxis
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointVelocityLimit,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.robots.gripper_configuration import (
    GripperConfiguration,
)
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

# %% bringing a gripper into a configuration


class MovesGripperToConfiguration(ABC):
    """
    A node bringing a gripper into a configuration, built for any configuration the same
    way, so which kind of node does it can be chosen without knowing what each one
    needs.
    """

    @classmethod
    @abstractmethod
    def moving_gripper_to(
        cls,
        configuration: GripperConfiguration,
        finger_stall_minimum_time: float | None = None,
        tolerate_finger_stall: bool = False,
    ) -> StatechartNode:
        """
        :param configuration: The configuration to bring the gripper of its end effector
            into.
        :param finger_stall_minimum_time: See :attr:`~giskardpy.motion_statechart.mixins.HasFingerStallToleranceParameters.finger_stall_minimum_time`.
        :param tolerate_finger_stall: See :attr:`~giskardpy.motion_statechart.mixins.HasFingerStallToleranceParameters.tolerate_finger_stall`.
        :return: A new node bringing the gripper into `configuration`.
        """


# %% finger goals


@dataclass(eq=False, repr=False, kw_only=True)
class MoveFingers(
    Parallel,
    MovesGripperToConfiguration,
    HasJointVelocityLimitParameter,
    HasFingerStallToleranceParameters,
    ABC,
):
    """
    Drive the fingers of an end effector to a goal.

    Fingers that close on an object stop short of their goal, so the goal can be told to
    settle for fingers that have stopped approaching it instead.
    """

    nodes: list[StatechartNode] = field(default_factory=list, init=False)
    """
    The finger goal, together with the speed cap if one was asked for.
    """

    @classmethod
    def moving_gripper_to(
        cls,
        configuration: GripperConfiguration,
        finger_stall_minimum_time: float | None = None,
        tolerate_finger_stall: bool = False,
    ) -> MoveFingers:
        """
        :return: A new goal driving the fingers into `configuration` at its finger
            velocity: to the width its fingertips stand apart in its joint state for a
            two-finger gripper, to its joint state for any other hand.
        """
        end_effector = configuration.end_effector
        if isinstance(end_effector, HasTwoFingers):
            return MoveGripper(
                gripper=end_effector,
                width=end_effector.separation_at(configuration.joint_state),
                max_joint_velocity=configuration.finger_velocity,
                finger_stall_minimum_time=finger_stall_minimum_time,
                tolerate_finger_stall=tolerate_finger_stall,
            )
        return MoveFingersToJointState(
            goal=configuration.joint_state,
            max_joint_velocity=configuration.finger_velocity,
            finger_stall_minimum_time=finger_stall_minimum_time,
            tolerate_finger_stall=tolerate_finger_stall,
        )

    @property
    @abstractmethod
    def finger_connections(self) -> list[ActiveConnection1DOF]:
        """
        :return: The connections that move the fingers.
        """

    @abstractmethod
    def create_finger_task(self) -> MotionStatechartNode:
        """
        :return: A new task driving the fingers to the goal.
        """

    def expand(self) -> None:
        """
        Create the finger goal and the speed cap, after the nodes this goal was handed.
        """
        self.add_node(self._finger_goal())
        if self.max_joint_velocity is not None:
            self.add_node(
                JointVelocityLimit(
                    connections=self.finger_connections,
                    max_joint_velocity=self.max_joint_velocity,
                )
            )

    def _finger_goal(self) -> StatechartNode:
        """
        :return: The node that is done once the fingers arrived, which with
            :attr:`tolerate_finger_stall` is also done once they stopped short of it.

        The stall monitor sits beside the finger task rather than replacing its
        observation, because fingers that stopped short have not reached the goal.
        """
        finger_task = self.create_finger_task()
        if not self.tolerate_finger_stall:
            return finger_task
        return Parallel(
            nodes=[finger_task, self._stall_monitor(finger_task)], minimum_success=1
        )

    def _stall_monitor(self, finger_task: MotionStatechartNode) -> Stalled:
        """
        :param finger_task: The task driving the fingers to the goal.
        :return: A monitor that observes `finger_task` no longer approaching its goal for
            long enough.
        """
        return Stalled(
            monitored_node=finger_task,
            timeout=timedelta(
                seconds=(
                    LocalMinimumReached.minimum_time
                    if self.finger_stall_minimum_time is None
                    else self.finger_stall_minimum_time
                )
            ),
        )


@dataclass(eq=False, repr=False, kw_only=True)
class MoveGripper(MoveFingers, HasTwoFingerGripperParameters, HasThresholdParameter):
    """
    Open or close a two-finger gripper until its fingertips' facing surfaces stand
    :attr:`width` apart, with an error in metres.
    """

    threshold: float = 0.001
    """
    A millimetre by default.
    """

    @property
    def finger_connections(self) -> list[ActiveConnection1DOF]:
        return self.gripper.finger_connections

    def create_finger_task(self) -> DistanceAlongAxis:
        """
        :return: A new task bringing the facing points of the fingertips :attr:`width`
            apart along the axis the fingers close along.
        """
        fingertips = self.gripper.fingertips
        return DistanceAlongAxis(
            name=f"{self.name}/width",
            root_link=fingertips.tool_frame,
            reference_feature=fingertips.finger_P_facing_point,
            controlled_feature=fingertips.thumb_P_facing_point,
            axis=fingertips.tool_V_finger_to_thumb,
            distance=self.width,
            threshold=self.threshold,
        )


@dataclass(eq=False, repr=False, kw_only=True)
class MoveFingersToJointState(MoveFingers, HasGoalParameter[JointState]):
    """
    Drive the fingers of a hand to the joint state :attr:`goal`, for a hand whose
    fingers do not close to a single width.
    """

    @property
    def finger_connections(self) -> list[ActiveConnection1DOF]:
        return list(self.goal.connections)

    def create_finger_task(self) -> JointPositionList:
        """
        :return: A new task driving the fingers to :attr:`goal`.
        """
        return JointPositionList(goal=self.goal)
