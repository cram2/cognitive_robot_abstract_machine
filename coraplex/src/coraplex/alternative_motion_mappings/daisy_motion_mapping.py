from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.ros2_nodes.griplink import (
    GriplinkFlexActionServerTask,
    GriplinkPresetActionServerTask,
)
from griplink_interfaces.action import Flexgrip, Flexrelease, Grip, Release
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.robots.gripper_configurations import (
    GriplinkGripperConfiguration,
)
from semantic_digital_twin.datastructures.robots.gripper_specification import (
    GriplinkFlexSpecification,
    GriplinkPresetSpecification,
    TGripperSpecification,
)
from semantic_digital_twin.robots.daisy import (
    DAiSy,
    DAiSyLeftGripper,
    DAiSyRightGripper,
)
from semantic_digital_twin.robots.robot_parts import EndEffector

from coraplex.datastructures.enums import ExecutionType
from coraplex.exceptions import NoGriplinkEndpoint
from coraplex.plans.executables import GiskardExecutable
from coraplex.robot_plans import MoveGripperMotion
from coraplex.robot_plans.motions.base import AlternativeMotion

logger = logging.getLogger(__name__)


# %% griplink endpoint resolution


@dataclass(frozen=True)
class GriplinkEndpoint:
    """
    A griplink action server endpoint a single gripper is reached on.
    """

    action_topic: str
    """
    ROS action topic the griplink server for one gripper listens on.
    """

    message_type: type
    """
    Griplink action message type this endpoint expects (``Grip``/``Release``/
    ``Flexgrip``/``Flexrelease``).
    """


# %% DAiSy griplink motions
@dataclass
class DAiSyGripperMotion(MoveGripperMotion[TGripperSpecification]):
    """
    Moves a griplink gripper of real DAiSy on its griplink action server, or commands a
    joint position goal for semi-real and simulated execution.

    Concrete motions are alternative motions for DAiSy and declare the griplink
    endpoints of the states they command.
    """

    execution_type: tuple[ExecutionType, ...] = (
        ExecutionType.REAL,
        ExecutionType.SEMI_REAL,
        ExecutionType.SIMULATED,
    )
    """
    Execution types this alternative applies to.

    Real execution drives the griplink action server; semi-real and simulated execution
    fall back to the joint position goal the specification describes.
    """

    _griplink_endpoints: ClassVar[
        Mapping[type[EndEffector], Mapping[GripperState, GriplinkEndpoint]]
    ]
    """
    Griplink endpoints the griplink server of each DAiSy gripper listens on, by the
    state the endpoint commands; concrete motions declare the table for the states they
    command.
    """

    def perform(self):
        """
        Logs the performed motion; the action server execution itself is started by the
        motion chart.
        """
        logger.info(f"Performing action {self.__class__.__name__}")
        return

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
        :raises NoGriplinkEndpoint: If the gripper or state has no endpoint in
            :attr:`_griplink_endpoints`.
        """
        state_type = self.specification.joint_state.state_type
        gripper_type = type(self.specification.end_effector)
        endpoint = self._griplink_endpoints.get(gripper_type, {}).get(state_type)
        if endpoint is None:
            raise NoGriplinkEndpoint(
                end_effector=self.specification.end_effector, state_type=state_type
            )
        return endpoint

    @property
    @abstractmethod
    def _action_server_task(self) -> MotionStatechartNode:
        """
        The griplink action server task this motion builds its chart from.

        :return: The task configured from this motion's specification and endpoint.
        """


# %% DAiSy grip motion
@dataclass
class DAiSyGripMotion(
    AlternativeMotion[DAiSy], DAiSyGripperMotion[GriplinkPresetSpecification]
):
    """
    Uses the griplink action server to grip or release with the griplink grippers of
    real DAiSy, or a joint position goal for semi-real execution.
    """

    _griplink_endpoints = {
        DAiSyLeftGripper: {
            GripperState.OPEN: GriplinkEndpoint(
                action_topic="/left_gripper/release", message_type=Release
            ),
            GripperState.CLOSE: GriplinkEndpoint(
                action_topic="/left_gripper/grip", message_type=Grip
            ),
        },
        DAiSyRightGripper: {
            GripperState.OPEN: GriplinkEndpoint(
                action_topic="/right_gripper/release", message_type=Release
            ),
            GripperState.CLOSE: GriplinkEndpoint(
                action_topic="/right_gripper/grip", message_type=Grip
            ),
        },
    }

    @property
    def _action_server_task(self) -> GriplinkPresetActionServerTask:
        """
        :return: The griplink task executing the preset of this motion's specification.
        """
        configuration: GriplinkGripperConfiguration = self.specification.configuration
        endpoint = self._griplink_endpoint
        return GriplinkPresetActionServerTask(
            action_topic=endpoint.action_topic,
            message_type=endpoint.message_type,
            grip_preset=configuration.grip_preset,
        )


# %% DAiSy flex grip motion
@dataclass
class DAiSyFlexGripMotion(
    AlternativeMotion[DAiSy], DAiSyGripperMotion[GriplinkFlexSpecification]
):
    """
    Uses flex grip and release motions for the griplink grippers of real DAiSy, or a
    joint position goal for semi-real execution.
    """

    _griplink_endpoints = {
        DAiSyLeftGripper: {
            GripperState.FLEXCLOSE: GriplinkEndpoint(
                action_topic="/left_gripper/flexgrip", message_type=Flexgrip
            ),
            GripperState.FLEXOPEN: GriplinkEndpoint(
                action_topic="/left_gripper/flexrelease", message_type=Flexrelease
            ),
        },
        DAiSyRightGripper: {
            GripperState.FLEXCLOSE: GriplinkEndpoint(
                action_topic="/right_gripper/flexgrip", message_type=Flexgrip
            ),
            GripperState.FLEXOPEN: GriplinkEndpoint(
                action_topic="/right_gripper/flexrelease", message_type=Flexrelease
            ),
        },
    }

    @property
    def _action_server_task(self) -> GriplinkFlexActionServerTask:
        """
        :return: The griplink task executing the commanded opening width of this
            motion's specification.
        """
        configuration: GriplinkGripperConfiguration = self.specification.configuration
        endpoint = self._griplink_endpoint
        return GriplinkFlexActionServerTask(
            action_topic=endpoint.action_topic,
            message_type=endpoint.message_type,
            grip_position=configuration.grip_position,
            grip_force=configuration.grip_force,
            grip_speed=configuration.grip_speed,
            grip_acceleration=configuration.grip_acceleration,
        )
