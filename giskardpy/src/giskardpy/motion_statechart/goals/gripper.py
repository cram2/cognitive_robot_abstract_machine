from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from cramph.composites import Parallel
from cramph.node import StatechartNode
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
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
        stall_minimum_time: float | None = None,
        tolerate_stall: bool = False,
    ) -> StatechartNode:
        """
        :param configuration: The configuration to bring the gripper of its end effector
            into.
        :param stall_minimum_time: See :attr:`MoveFingers.stall_minimum_time`.
        :param tolerate_stall: See :attr:`MoveFingers.tolerate_stall`.
        :return: A new node bringing the gripper into `configuration`.
        """


# %% finger goals


@dataclass(eq=False, repr=False, kw_only=True)
class MoveFingers(Parallel, MovesGripperToConfiguration, ABC):
    """
    Drive the fingers of an end effector to a goal.

    Fingers that close on an object stop short of their goal, so the goal can be told to
    settle for fingers that have stopped moving instead.
    """

    tolerate_stall: bool = False
    """
    Whether fingers that have stopped moving count as done, even short of their goal.
    """

    stall_minimum_time: float | None = None
    """
    How long the fingers must stand still before a stall counts, in seconds.

    None keeps :attr:`~giskardpy.motion_statechart.monitors.monitors.LocalMinimumReached.minimum_time`.
    """

    finger_velocity: float | None = None
    """
    Maximum speed of the finger joints, in meters per second.

    None leaves the speed capped only by the joints' own limits.
    """

    nodes: list[StatechartNode] = field(default_factory=list, init=False)
    """
    The finger goal, together with the speed cap if one was asked for.
    """

    @classmethod
    def moving_gripper_to(
        cls,
        configuration: GripperConfiguration,
        stall_minimum_time: float | None = None,
        tolerate_stall: bool = False,
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
                finger_velocity=configuration.finger_velocity,
                stall_minimum_time=stall_minimum_time,
                tolerate_stall=tolerate_stall,
            )
        return MoveFingersToJointState(
            goal_state=configuration.joint_state,
            finger_velocity=configuration.finger_velocity,
            stall_minimum_time=stall_minimum_time,
            tolerate_stall=tolerate_stall,
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
        if self.finger_velocity is not None:
            self.add_node(
                JointVelocityLimit(
                    connections=self.finger_connections,
                    max_velocity=self.finger_velocity,
                )
            )

    def _finger_goal(self) -> StatechartNode:
        """
        :return: The node that is done once the fingers arrived, which with
            :attr:`tolerate_stall` is also done once they stopped moving short of it.

        The stall monitor sits beside the finger task rather than replacing its
        observation, because fingers standing still does not mean the goal was reached.
        """
        finger_task = self.create_finger_task()
        if not self.tolerate_stall:
            return finger_task
        return Parallel(nodes=[finger_task, self._stall_monitor()], minimum_success=1)

    def _stall_monitor(self) -> LocalMinimumReached:
        """
        :return: A monitor that observes the fingers, and only those, standing still for
            long enough.
        """
        return LocalMinimumReached(
            degrees_of_freedom=[
                connection.raw_dof for connection in self.finger_connections
            ],
            minimum_time=(
                LocalMinimumReached.minimum_time
                if self.stall_minimum_time is None
                else self.stall_minimum_time
            ),
            measure_from_own_start=True,
        )


@dataclass(eq=False, repr=False, kw_only=True)
class MoveGripper(MoveFingers):
    """
    Open or close a two-finger gripper until its fingertips' facing surfaces stand
    :attr:`width` apart.
    """

    gripper: HasTwoFingers
    """
    The gripper whose fingers move.
    """

    width: float
    """
    How far apart the fingertips' facing surfaces are to stand, in metres; negative
    where they are to overlap, as a fully closed gripper's may.
    """

    threshold: float = 0.001
    """
    How close, in metres, the fingertips have to come to :attr:`width` to count as
    arrived.
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
            reference_point=fingertips.finger_P_facing_point,
            tip_point=fingertips.thumb_P_facing_point,
            axis=fingertips.tool_V_finger_to_thumb,
            distance=self.width,
            threshold=self.threshold,
        )


@dataclass(eq=False, repr=False, kw_only=True)
class MoveFingersToJointState(MoveFingers):
    """
    Drive the fingers of a hand to a joint state, for a hand whose fingers do not close
    to a single width.
    """

    goal_state: JointState
    """
    The joint state to drive the fingers to.
    """

    @property
    def finger_connections(self) -> list[ActiveConnection1DOF]:
        return list(self.goal_state.connections)

    def create_finger_task(self) -> JointPositionList:
        """
        :return: A new task driving the fingers to :attr:`goal_state`.
        """
        return JointPositionList(goal_state=self.goal_state)
