from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.ros2_nodes.griplink import (
    GriplinkFlexActionServerTask,
    GriplinkPresetActionServerTask,
)
from griplink_interfaces.action import Flexgrip, Flexrelease, Grip, Release
from semantic_digital_twin.datastructures.robots.gripper_configuration import (
    TGripperConfiguration,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.griplink_gripper import (
    GriplinkEndpoint,
    GriplinkFlexConfiguration,
    GriplinkPresetConfiguration,
)

from coraplex.datastructures.enums import ExecutionType
from coraplex.exceptions import NoGriplinkEndpoint
from coraplex.plans.executables import GiskardExecutable
from coraplex.robot_plans import MoveGripperMotion
from coraplex.robot_plans.motions.base import AlternativeMotion

logger = logging.getLogger(__name__)


# %% DAiSy griplink motions
@dataclass
class DAiSyGripperMotion(MoveGripperMotion[TGripperConfiguration]):
    """
    Moves a griplink gripper of real DAiSy on its griplink action server, or commands a
    joint position goal for semi-real and simulated execution.

    Concrete motions are alternative motions for DAiSy and read the griplink endpoints
    the gripper's semantic annotation declares.
    """

    execution_type: tuple[ExecutionType, ...] = (
        ExecutionType.REAL,
        ExecutionType.SEMI_REAL,
        ExecutionType.SIMULATED,
    )
    """
    Execution types this alternative applies to.

    Real execution drives the griplink action server; semi-real and simulated execution
    fall back to the joint position goal the configuration describes.
    """

    _commanded_actions: ClassVar[frozenset[type]]
    """
    Griplink actions this motion's task builds goals for.

    The gripper declares an endpoint for every state it serves, so the endpoint lookup
    accepts only endpoints whose action this motion executes.
    """

    def perform(self):
        """
        Logs the performed motion; the action server execution itself is started by the
        motion chart.
        """
        logger.info(f"Performing action {self.__class__.__name__}")

    @property
    def _motion_chart(self) -> MotionStatechartNode:
        """
        :return: The joint position goal for semi-real and simulated execution, or the
            griplink action server task for real execution.
        """
        if (
            GiskardExecutable.execution_type == ExecutionType.SEMI_REAL
            or GiskardExecutable.execution_type == ExecutionType.SIMULATED
        ):
            return super()._motion_chart

        return Parallel([self._action_server_task])

    @property
    def _griplink_endpoint(self) -> GriplinkEndpoint:
        """
        :return: The endpoint the griplink server for this motion's gripper and state
            listens on.
        :raises NoGriplinkEndpoint: If the gripper declares no endpoint for the state
            this motion commands, or the endpoint's action is not one this motion
            executes.
        """
        state_type = self.configuration.joint_state.state_type
        endpoint = self.configuration.end_effector.griplink_endpoint(state_type)
        if endpoint is None or endpoint.message_type not in self._commanded_actions:
            raise NoGriplinkEndpoint(
                end_effector=self.configuration.end_effector, state_type=state_type
            )
        return endpoint

    @property
    @abstractmethod
    def _action_server_task(self) -> MotionStatechartNode:
        """
        The griplink action server task this motion builds its chart from.

        :return: The task configured from this motion's configuration and endpoint.
        """


# %% DAiSy grip motion
@dataclass
class DAiSyGripMotion(
    AlternativeMotion[DAiSy], DAiSyGripperMotion[GriplinkPresetConfiguration]
):
    """
    Uses the griplink action server to grip or release with the griplink grippers of
    real DAiSy, or a joint position goal for semi-real execution.
    """

    _commanded_actions = frozenset((Grip, Release))

    @property
    def _action_server_task(self) -> GriplinkPresetActionServerTask:
        """
        :return: The griplink task executing the preset of this motion's configuration.
        """
        endpoint = self._griplink_endpoint
        return GriplinkPresetActionServerTask(
            action_topic=endpoint.action_topic,
            message_type=endpoint.message_type,
            grip_preset=self.configuration.grip_preset,
        )


# %% DAiSy flex grip motion
@dataclass
class DAiSyFlexGripMotion(
    AlternativeMotion[DAiSy], DAiSyGripperMotion[GriplinkFlexConfiguration]
):
    """
    Uses flex grip and release motions for the griplink grippers of real DAiSy, or a
    joint position goal for semi-real execution.
    """

    _commanded_actions = frozenset((Flexgrip, Flexrelease))

    @property
    def _action_server_task(self) -> GriplinkFlexActionServerTask:
        """
        :return: The griplink task executing the commanded opening width of this
            motion's configuration.
        """
        endpoint = self._griplink_endpoint
        return GriplinkFlexActionServerTask(
            action_topic=endpoint.action_topic,
            message_type=endpoint.message_type,
            grip_position=self.configuration.grip_position,
            grip_force=self.configuration.grip_force,
            grip_velocity=self.configuration.grip_velocity,
            grip_acceleration=self.configuration.grip_acceleration,
        )
