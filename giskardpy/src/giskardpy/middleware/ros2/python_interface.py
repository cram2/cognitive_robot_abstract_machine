from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Thread
from time import sleep
from typing import Dict, List

import rclpy
from json_msgs.action import JsonAction
from json_msgs.action._json_action import JsonAction_Result
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.exceptions import NoActiveGoalToCancelError
from giskardpy.middleware.ros2.motion_goal import MotionGoal
from giskardpy.middleware.ros2.ros2_interface import MyActionClient
from giskardpy.middleware.ros2.world_updates import ClientWorldUpdates
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
    LifeCycleState,
    ObservationState,
)
from rclpy import Context, Parameter, Future
from rclpy.action.client import ClientGoalHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World


@dataclass
class GiskardWrapper:
    """
    Python wrapper for the ROS interface of Giskard.

    :param node_handle: node used to talk to Giskard
    :param giskard_node_name: node name of Giskard
    :param world: world to keep in step with the world of Giskard, fetched from Giskard
        when not given
    """

    node_handle: Node
    giskard_node_name: str = "giskard"
    _goal_handle: ClientGoalHandle | None = None
    _goal_result: JsonAction_Result | None = None
    _result_future: Future | None = None
    world: World = None
    _client: MyActionClient = None

    world_updates: ClientWorldUpdates = field(init=False, default=None)
    """
    Keeps this world in step with the world Giskard controls around a goal.
    """

    _motion_statechart: MotionStatechart | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self):
        if self.world is None:
            self.node_handle.get_logger().info(
                "No world provided, fetching from service"
            )
            self.world = fetch_world_from_service(self.node_handle, timeout_seconds=300)
            self.node_handle.get_logger().info("world synced")
            WorldSynchronizer(_world=self.world, node=self.node_handle)
        self.world_updates = ClientWorldUpdates(
            world_synchronizer=WorldSynchronizer.of_world(self.world)
        )
        giskard_topic = f"{self.giskard_node_name}/command"
        self._client = MyActionClient(self.node_handle, JsonAction, giskard_topic)
        sleep(0.3)

    @property
    def robot_name(self) -> PrefixedName:
        return self.robot.name

    @property
    def robot(self) -> AbstractRobot:
        return self.world.get_semantic_annotations_by_type(AbstractRobot)[0]

    def execute_async(self, motion_statechart: MotionStatechart) -> Future:
        self._motion_statechart = motion_statechart
        motion_statechart.sanity_check()
        return self._send_action_goal_async(motion_statechart)

    def execute(self, motion_statechart: MotionStatechart):
        """
        Executes a MotionStatechart and syncs its state with the result of Giskard.

        A goal that fails raises the exception that made Giskard abort it, for example
        :class:`WorldModelModifiedDuringMotionError` when another process modified the
        world model while the motion was running.

        :param motion_statechart: statechart to execute
        """
        motion_statechart.sanity_check()
        result = self._send_action_goal(motion_statechart)
        self._take_over_result(result, motion_statechart)

    def _take_over_result(
        self, result: JsonAction_Result, motion_statechart: MotionStatechart
    ) -> None:
        """
        Copy the final states of a finished goal into the given motion statechart.

        Only reached for a goal that succeeded; a failed one raises while its result is
        awaited.

        :param result: result of the finished goal
        :param motion_statechart: statechart the goal was built from
        """
        result_json = json.loads(result.result.result)
        self.world_updates.wait_for_the_changes_of_a_goal(result_json)
        parsed_life_cycle_state = LifeCycleState.from_json(
            result_json["life_cycle_state"], motion_statechart=motion_statechart
        )
        parsed_observation_state = ObservationState.from_json(
            result_json["observation_state"], motion_statechart=motion_statechart
        )
        motion_statechart.life_cycle_state.data = parsed_life_cycle_state.data
        motion_statechart.observation_state.data = parsed_observation_state.data
        assert motion_statechart.is_end_motion()

    def _create_goal_message(
        self, motion_statechart: MotionStatechart
    ) -> JsonAction.Goal:
        """
        Wrap the motion statechart into a goal that names the change of this world it
        was built on.

        :param motion_statechart: statechart to send to Giskard
        :return: action goal message holding the serialized motion goal
        """
        goal_msg = JsonAction.Goal()
        goal = MotionGoal.for_motion_statechart(
            motion_statechart,
            required_position=self.world_updates.required_position(),
        )
        goal_msg.goal = json.dumps(goal.to_json())
        return goal_msg

    def _send_action_goal_async(self, motion_statechart: MotionStatechart) -> Future:
        return self._client.send_goal_async(
            self._create_goal_message(motion_statechart)
        )

    def _send_action_goal(
        self, motion_statechart: MotionStatechart
    ) -> JsonAction_Result:
        return self._client.send_goal(self._create_goal_message(motion_statechart))

    def cancel_goal_async(self) -> Future:
        """
        Stops the goal that was last sent to Giskard.

        :return: future that completes once the cancel request was answered
        """
        try:
            future = self._client._goal_handle.cancel_goal_async()
        except AttributeError as e:
            raise NoActiveGoalToCancelError()
        return future

    async def get_result(self):
        """
        Wait for the goal sent with :func:`execute_async` and sync its final states.

        A goal that fails raises the exception that made Giskard abort it, for example
        :class:`WorldModelModifiedDuringMotionError` when another process modified the
        world model while the motion was running.
        """
        result = await self._client.get_result()
        self._take_over_result(result, self._motion_statechart)

    def get_end_motion_reason(
        self, move_result: JsonAction_Result | None = None, show_all: bool = False
    ) -> Dict[str, bool]:
        """
        Analyzes a MoveResult msg to return a list of all monitors that hindered the
        EndMotion Monitors from becoming active.

        Uses the last received MoveResult msg from execute() or projection() when not
        explicitly given.
        :param move_result: the move_result msg to analyze
        :param show_all: returns the state of all monitors when show_all==True
        :return: Dict with monitor name as key and True or False as value
        """
        ...


@dataclass
class GiskardWrapperNode(GiskardWrapper):
    is_spinning: bool = False
    node_name: str = "giskard_client"
    giskard_node_name: str = "giskard"
    avoid_name_conflict: bool = True
    context: Context | None = field(kw_only=True, default=None)
    cli_args: List[str] | None = field(kw_only=True, default=None)
    namespace: str | None = field(kw_only=True, default=None)
    use_global_arguments: bool = field(kw_only=True, default=True)
    enable_rosout: bool = field(kw_only=True, default=True)
    start_parameter_services: bool = field(kw_only=True, default=True)
    parameter_overrides: List[Parameter] | None = field(kw_only=True, default=None)
    allow_undeclared_parameters: bool = field(kw_only=True, default=False)
    automatically_declare_parameters_from_overrides: bool = field(
        kw_only=True, default=False
    )
    enable_logger_service: bool = field(kw_only=True, default=False)
    node_handle: Node = field(init=False)

    def __post_init__(self):
        self.node_handle = Node(
            self.node_name,
            context=self.context,
            cli_args=self.cli_args,
            namespace=self.namespace,
            use_global_arguments=self.use_global_arguments,
            enable_rosout=self.enable_rosout,
            start_parameter_services=self.start_parameter_services,
            parameter_overrides=self.parameter_overrides,
            allow_undeclared_parameters=self.allow_undeclared_parameters,
            automatically_declare_parameters_from_overrides=self.automatically_declare_parameters_from_overrides,
        )
        rospy.executor.add_node(self.node_handle)
        self.is_spinning = False
        super().__post_init__()

    def __spin(self):
        self.my_executor = MultiThreadedExecutor()
        self.my_executor.add_node(self.node_handle)
        self.is_spinning = True
        while rclpy.ok():
            self.my_executor.spin_once(timeout_sec=1)
        self.is_spinning = False

    def spin_in_background(self):
        self.spinner = Thread(
            target=self.__spin, daemon=False, name="background giskard wrapper spinner"
        )
        self.spinner.start()
