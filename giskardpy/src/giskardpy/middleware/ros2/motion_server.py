from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List

import rclpy
from json_msgs.action import JsonAction

from giskardpy.data_types.exceptions import DontPrintStackTrace
from giskardpy.executor import Executor, RealTimePacer
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.action_server import ActionServerHandler
from giskardpy.middleware.ros2.control_loop import ControlLoop
from giskardpy.middleware.ros2.exceptions import (
    ExecutionCanceledException,
    RequiredWorldUpdateNotReceivedError,
)
from giskardpy.middleware.ros2.feedback_publisher import ActionFeedbackPublisher
from giskardpy.middleware.ros2.cycle_counter import CycleCounter
from giskardpy.middleware.ros2.input_synchronization import WorldStateInputs
from giskardpy.middleware.ros2.motion_goal import MotionGoal
from giskardpy.middleware.ros2.post_goal_plotters import PostGoalPlotter
from giskardpy.middleware.ros2.world_updates import IncomingWorldUpdates
from krrood.adapters.json_serializer import to_json
from semantic_digital_twin.adapters.ros.messages import StreamPosition
from semantic_digital_twin.adapters.ros.world_synchronizer import PublicationProgress
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.world import World


@dataclass
class MotionServer:
    """
    The goal lifecycle of Giskard.

    While idle, the server keeps the world in sync with the outside and waits for a
    goal. An accepted goal is parsed, executed by the control loop and always answered,
    even if it fails.
    """

    executor: Executor
    """
    Compiles and ticks the motion statecharts of incoming goals.
    """

    action_server: ActionServerHandler
    """
    Receives goals and returns their results.
    """

    control_loop: ControlLoop
    """
    Executes a compiled motion statechart.
    """

    world_updates: IncomingWorldUpdates
    """
    Applies the world updates of other processes that the control loop could not.
    """

    world_synchronizer: PublicationProgress
    """
    Reports how far the changes of this world were published to the other processes.
    """

    feedback_publisher: ActionFeedbackPublisher
    """
    Reports the state of the motion statechart to the action client.
    """

    inputs: WorldStateInputs
    """
    Writes the state of the robot into the world while waiting for a goal.
    """

    cycle_counter: CycleCounter
    """
    Ticked once per idle cycle and, through the control loop, once per control cycle.
    """

    idle_frequency: float = 20.0
    """
    Frequency in hertz at which the idle loop runs.
    """

    world_update_timeout: float = 30.0
    """
    Seconds a goal waits for the change of the world it was built on.
    """

    post_goal_plotters: List[PostGoalPlotter] = field(default_factory=list)
    """
    Debug plots that are written once a goal is finished.
    """

    idle_pacer: RealTimePacer = field(init=False)
    """
    Paces the idle loop to ``idle_frequency``.
    """

    _published_sequence_number_before_goal: int = field(init=False, default=0)
    """
    How far this world had published when the running goal was accepted.
    """

    def __post_init__(self):
        self.idle_pacer = RealTimePacer()
        self.idle_pacer.target_frequency = self.idle_frequency

    @property
    def world(self) -> World:
        return self.executor.context.world

    # %% waiting for goals

    def live(self) -> None:
        """
        Run the idle loop until ROS shuts down.
        """
        rospy.node.get_logger().info("giskard is ready")
        while rclpy.ok():
            self.run_idle_cycle()
            self.idle_pacer.sleep()

    def run_idle_cycle(self) -> None:
        """
        Apply everything that happened outside of Giskard and execute a goal if one is
        waiting.
        """
        if self.world.world_is_being_modified:
            return
        self.world_updates.apply_all()
        self.inputs.synchronize_and_announce()
        self.cycle_counter.tick()
        if not self.action_server.has_goal():
            return
        self.action_server.accept_goal()
        self.execute_goal()

    # %% executing goals

    def execute_goal(self) -> None:
        """
        Execute the accepted goal and answer the client, whatever happens.
        """
        self._published_sequence_number_before_goal = (
            self.world_synchronizer.published_sequence_number
        )
        error: Exception | None = None
        try:
            goal = MotionGoal.from_json(json.loads(self.action_server.goal_msg.goal))
            self.wait_for_required_world_updates(goal.required_position)
            self.compile_goal(goal)
            self.control_loop.run()
        except Exception as exception:
            if not isinstance(
                exception, (DontPrintStackTrace, ExecutionCanceledException)
            ):
                traceback.print_exc()
            error = exception
        finally:
            self.finish_goal(error)

    def wait_for_required_world_updates(
        self, required_position: StreamPosition | None
    ) -> None:
        """
        Wait until the world contains the change the goal was built on.

        :raises RequiredWorldUpdateNotReceivedError: If that change does not arrive
            within ``world_update_timeout``.
        """
        if required_position is None:
            return
        deadline = time.monotonic() + self.world_update_timeout
        while True:
            self.world_updates.apply_all()
            if self.world_updates.has_applied(required_position):
                return
            if time.monotonic() >= deadline:
                raise RequiredWorldUpdateNotReceivedError(
                    current_sequence_number=self.world_synchronizer.published_sequence_number,
                    publisher_name=required_position.origin.node_name,
                    awaited_sequence_number=required_position.sequence_number,
                    timeout=self.world_update_timeout,
                )
            self.idle_pacer.sleep()

    def compile_goal(self, goal: MotionGoal) -> None:
        """
        Turn the goal message into a compiled motion statechart.
        """
        rospy.node.get_logger().info(
            f"Parsing goal #{self.action_server.goal_id} message."
        )
        tracker = WorldEntityWithIDKwargsTracker.from_world(self.world)
        kwargs = tracker.create_kwargs()
        kwargs["world"] = self.world
        motion_statechart = goal.parse_motion_statechart(**kwargs)
        self.executor.compile(motion_statechart)
        self.feedback_publisher.publish_structure()
        rospy.node.get_logger().info("Done parsing goal message.")

    def finish_goal(self, error: Exception | None) -> None:
        """
        Stop the robot, clean up the motion statechart and answer the client.

        The client is answered even if cleaning up or plotting fails, so that a failure
        here cannot make it wait forever.
        """
        try:
            self.control_loop.stop()
            if self.executor.motion_statechart is not None:
                self.executor.motion_statechart.cleanup_nodes(
                    context=self.executor.context
                )
            self.feedback_publisher.publish()
            self.write_debug_plots()
        finally:
            self.action_server.result_msg = self.create_result(error)
            self.action_server.send_result()

    def create_result(self, error: Exception | None) -> JsonAction.Result:
        """
        Mark the goal as canceled, aborted or succeeded and describe its final state.

        A failed goal also reports the error itself, because the ROS action status alone
        cannot tell a client whether sending the goal again would help. The error is
        serialized so that the client can rebuild and raise the very same exception.
        """
        match error:
            case ExecutionCanceledException():
                self.action_server.set_canceled()
                rospy.node.get_logger().warning("Goal canceled by user.")
            case None:
                self.action_server.set_succeeded()
                rospy.node.get_logger().info("Goal succeeded.")
            case _:
                self.action_server.set_aborted()
                rospy.node.get_logger().error(f"Goal aborted: {error}")
        states = self.create_states()
        if error is not None:
            states["error"] = to_json(error)
        published_position = self.published_position_of_goal()
        if published_position is not None:
            states["published_position"] = to_json(published_position)
        result = JsonAction.Result()
        result.result = json.dumps(states)
        return result

    def published_position_of_goal(self) -> StreamPosition | None:
        """
        The position this world published up to while the goal was running, or ``None``
        if the goal published nothing.
        """
        if (
            self.world_synchronizer.published_sequence_number
            == self._published_sequence_number_before_goal
        ):
            return None
        return self.world_synchronizer.latest_published_position

    def create_states(self) -> Dict[str, Any]:
        """
        Collect the final life cycle and observation state of the motion statechart.

        A goal whose statechart could not be compiled has no states to report.
        """
        if self.executor.motion_statechart is None:
            return {}
        return self.feedback_publisher.create_states()

    def write_debug_plots(self) -> None:
        """
        Write the configured debug plots of the finished goal.

        A plot is a diagnostic, so a plotter that fails is reported and skipped. Letting
        it raise would end the loop that serves goals, leaving every later client
        waiting for a result that no one is going to produce.
        """
        if self.executor.motion_statechart is None:
            return
        for plotter in self.post_goal_plotters:
            try:
                plotter.plot(self.action_server.goal_id)
            except Exception:
                rospy.node.get_logger().error(
                    f"{type(plotter).__name__} failed to plot goal "
                    f"#{self.action_server.goal_id}:\n{traceback.format_exc()}"
                )
