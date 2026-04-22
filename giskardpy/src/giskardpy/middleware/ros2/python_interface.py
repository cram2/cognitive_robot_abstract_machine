from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Thread
from time import sleep
from typing import Dict, Optional, List

import rclpy
from json_msgs.action import JsonAction
from json_msgs.action._json_action import JsonAction_Result
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.exceptions import ExecutionException
from giskardpy.middleware.ros2.ros2_interface import MyActionClient
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
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    ModelSynchronizer,
    StateSynchronizer,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World


@dataclass
class GiskardWrapper:
    """
    Python wrapper for the ROS interface of Giskard.
    :param giskard_node_name: node name of Giskard
    """

    node_handle: Node
    giskard_node_name: str = "giskard"
    _goal_handle: Optional[ClientGoalHandle] = None
    _goal_result: Optional[JsonAction_Result] = None
    _result_future: Optional[Future] = None
    world: World = None
    _client: MyActionClient = None
    _motion_statechart: MotionStatechart = field(init=False)

    def __post_init__(self):
        if self.world is None:
            self.node_handle.get_logger().info(
                "No world provided, fetching from service"
            )
            self.world = fetch_world_from_service(self.node_handle, timeout_seconds=300)
            self.node_handle.get_logger().info("world synced")
            self.model_synchronizer = ModelSynchronizer(
                _world=self.world, node=self.node_handle, synchronous=True
            )
            self.state_synchronizer = StateSynchronizer(
                _world=self.world, node=self.node_handle
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
        """
        motion_statechart.sanity_check()
        result = self._send_action_goal(motion_statechart)
        result_json = json.loads(result.result.result)
        parsed_life_cycle_state = LifeCycleState.from_json(
            result_json["life_cycle_state"], motion_statechart=motion_statechart
        )
        parsed_observation_state = ObservationState.from_json(
            result_json["observation_state"], motion_statechart=motion_statechart
        )
        motion_statechart.life_cycle_state.data = parsed_life_cycle_state.data
        motion_statechart.observation_state.data = parsed_observation_state.data
        assert motion_statechart.is_end_motion()

    def _send_action_goal_async(self, motion_statechart: MotionStatechart) -> Future:
        goal_msg = JsonAction.Goal()
        goal_msg.goal = json.dumps(motion_statechart.to_json())
        return self._client.send_goal_async(goal_msg)

    def _send_action_goal(
        self, motion_statechart: MotionStatechart
    ) -> JsonAction_Result:
        goal_msg = JsonAction.Goal()
        goal_msg.goal = json.dumps(motion_statechart.to_json())
        return self._client.send_goal(goal_msg)

    def cancel_goal_async(self) -> Future:
        """
        Stops the goal that was last sent to Giskard.
        """
        try:
            future = self._client._goal_handle.cancel_goal_async()
        except AttributeError as e:
            raise ExecutionException(
                "Can't cancel goals, because there is no active one"
            )
        return future

    async def get_result(self):
        result = await self._client.get_result()

        result_json = json.loads(result.result.result)
        parsed_life_cycle_state = LifeCycleState.from_json(
            result_json["life_cycle_state"], motion_statechart=self._motion_statechart
        )
        parsed_observation_state = ObservationState.from_json(
            result_json["observation_state"], motion_statechart=self._motion_statechart
        )
        self._motion_statechart.life_cycle_state.data = parsed_life_cycle_state.data
        self._motion_statechart.observation_state.data = parsed_observation_state.data
        assert self._motion_statechart.is_end_motion()

    def get_end_motion_reason(
        self, move_result: Optional[JsonAction_Result] = None, show_all: bool = False
    ) -> Dict[str, bool]:
        """
        Analyzes a MoveResult msg to return a list of all monitors that hindered the EndMotion Monitors from becoming active.
        Uses the last received MoveResult msg from execute() or projection() when not explicitly given.
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
    context: Optional[Context] = field(kw_only=True, default=None)
    cli_args: Optional[List[str]] = field(kw_only=True, default=None)
    namespace: Optional[str] = field(kw_only=True, default=None)
    use_global_arguments: bool = field(kw_only=True, default=True)
    enable_rosout: bool = field(kw_only=True, default=True)
    start_parameter_services: bool = field(kw_only=True, default=True)
    parameter_overrides: Optional[List[Parameter]] = field(kw_only=True, default=None)
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
