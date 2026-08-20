from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass, field
from typing import List

import rclpy
from json_msgs.action import JsonAction
from sqlalchemy.orm import sessionmaker

from giskardpy.data_types.exceptions import NoControlledJointsError
from giskardpy.executor import Executor
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.action_server import ActionServerHandler
from giskardpy.middleware.ros2.control_loop import ControlLoop
from giskardpy.middleware.ros2.feedback_publisher import ActionFeedbackPublisher
from giskardpy.middleware.ros2.cycle_counter import CycleCounter
from giskardpy.middleware.ros2.input_synchronization import WorldStateInputs
from giskardpy.middleware.ros2.motion_server import MotionServer
from giskardpy.middleware.ros2.post_goal_plotters import (
    GoalGanttChartPlotter,
    GoalTrajectoryPlotter,
    MotionStatechartPlotter,
    PostGoalPlotter,
)
from giskardpy.middleware.ros2.robot_interface_config import RobotInterfaceConfig
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.middleware.ros2.world_updates import IncomingWorldUpdates
from giskardpy.model.world_config import WorldConfig
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.ros_executor import Ros2Executor
from krrood.ormatic.utils import create_engine
from krrood.utils import clear_memoization_cache
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.collision_viz_marker import (
    CollisionVisualizationMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    WorldSynchronizer,
    ModelReloadSynchronizer,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world_description.connections import ActiveConnection

logger = logging.getLogger(__name__)


@dataclass
class Giskard:
    """
    The main Class of Giskard.

    Instantiate it with appropriate configs for you setup and then call giskard.live()
    :param world_config: A world configuration. Use a predefined one or implement your
        own WorldConfig class.
    :param robot_interface_config: How Giskard talk to the robot. You probably have to
        implement your own RobotInterfaceConfig.
    :param collision_avoidance_config: default is no collision avoidance or implement
        your own collision_avoidance_config.
    :param server_config: how goals are executed, default is standalone mode
    :param qp_controller_config: default is good for almost all cases
    :param additional_goal_package_paths: specify paths that Giskard needs to import to
        find your custom Goals. Giskard will run 'from <additional path> import *' for
        each additional path in the list.
    :param additional_monitor_package_paths: specify paths that Giskard needs to import
        to find your custom Monitors. Giskard will run 'from <additional path> import *'
        for each additional path in the list.
    """

    world_config: WorldConfig
    server_config: GiskardServerConfig
    robot_interface_config: RobotInterfaceConfig
    qp_controller_config: QPControllerConfig = field(default_factory=QPControllerConfig)
    executor: Executor = field(init=False)
    motion_server: MotionServer = field(init=False)
    world_synchronizer: WorldSynchronizer = field(init=False)
    tf_publisher: TFPublisher = field(init=False)
    viz_marker_publisher: VizMarkerPublisher = field(init=False)
    collision_marker_publisher: CollisionVisualizationMarkerPublisher = field(
        init=False
    )
    model_reload_synchronizer: ModelReloadSynchronizer = field(init=False)
    world_fetcher: FetchWorldServer = field(init=False)

    def setup(self):
        """
        Initialize the world, the ros interfaces and the motion server.

        You usually don't need to call this.
        """
        with self.world_config.world.modify_world():
            self.world_config.setup_world()
            clear_memoization_cache(self.world_config.world)
            self.executor = Ros2Executor(
                ros_node=rospy.node,
                context=MotionStatechartContext(
                    world=self.world_config.world,
                    qp_controller_config=self.qp_controller_config,
                ),
                pacer=self.server_config.create_pacer(),
            )

        self.setup_world_model_ros_interface()
        self.motion_server = self.create_motion_server()
        self.robot_interface_config.attach(self)
        self.robot_interface_config.setup()
        self.sanity_check()

    def create_motion_server(self) -> MotionServer:
        """
        Build the goal lifecycle around the executor.
        """
        world = self.world_config.world
        action_server = ActionServerHandler(
            action_name=f"{rospy.node.get_name()}/command", action_type=JsonAction
        )
        feedback_publisher = ActionFeedbackPublisher(
            executor=self.executor, action_server=action_server
        )
        cycle_counter = CycleCounter()
        world_updates = IncomingWorldUpdates(
            world_synchronizer=self.world_synchronizer,
            model_reload_synchronizer=self.model_reload_synchronizer,
        )
        control_loop = ControlLoop(
            executor=self.executor,
            action_server=action_server,
            feedback_publisher=feedback_publisher,
            inputs=WorldStateInputs(world=world),
            cycle_counter=cycle_counter,
            world_updates=world_updates,
        )
        return MotionServer(
            executor=self.executor,
            action_server=action_server,
            control_loop=control_loop,
            world_updates=world_updates,
            world_synchronizer=self.world_synchronizer,
            feedback_publisher=feedback_publisher,
            inputs=WorldStateInputs(world=world),
            cycle_counter=cycle_counter,
            idle_frequency=self.server_config.idle_frequency,
            post_goal_plotters=self.create_post_goal_plotters(),
        )

    def create_post_goal_plotters(self) -> List[PostGoalPlotter]:
        """
        Create the debug plotters that are configured and let them record what they
        need.
        """
        if not self.server_config.debug_mode:
            return []
        plotters: List[PostGoalPlotter] = []
        if self.server_config.plot_trajectory:
            plotters.append(GoalTrajectoryPlotter(executor=self.executor))
        if self.server_config.plot_gantt_chart:
            plotters.append(GoalGanttChartPlotter(executor=self.executor))
        if self.server_config.plot_motion_statechart:
            plotters.append(MotionStatechartPlotter(executor=self.executor))
        for plotter in plotters:
            plotter.start_recording()
        return plotters

    def setup_world_model_ros_interface(self):
        try:
            semantic_digital_twin_database_uri = os.environ.get(
                "SEMANTIC_DIGITAL_TWIN_DATABASE_URI"
            )
            assert (
                semantic_digital_twin_database_uri is not None
            ), "Please set the SEMANTIC_DIGITAL_TWIN_DATABASE_URI environment variable."

            engine = create_engine(semantic_digital_twin_database_uri)
            session = sessionmaker(bind=engine)()

            self.model_reload_synchronizer = ModelReloadSynchronizer(
                node=rospy.node,
                _world=self.world_config.world,
                session=session,
                defer_incoming_reloads=True,
            )
        except AssertionError as e:
            logger.warning(
                f'Model reload synchronization not available because "SEMANTIC_DIGITAL_TWIN_DATABASE_URI" is not set.'
            )
            self.model_reload_synchronizer = None

        # Deferring only the incoming direction: the control loop decides when a foreign
        # update may touch the world it is controlling, while this world's own changes
        # keep being published through the normal callbacks.
        self.world_synchronizer = WorldSynchronizer(
            _world=self.world_config.world,
            node=rospy.node,
            defer_incoming_updates=True,
        )
        self.world_fetcher = FetchWorldServer(
            node=rospy.node, world=self.world_config.world
        )
        self.tf_publisher = TFPublisher.create_with_ignore_existing_tf(
            node=rospy.node, world=self.world_config.world
        )
        self.viz_marker_publisher = VizMarkerPublisher(
            node=rospy.node, _world=self.world_config.world
        )
        self.collision_marker_publisher = CollisionVisualizationMarkerPublisher(
            node=rospy.node, throttle=5, world=self.world_config.world
        )

    def close_world_model_ros_interface(self):
        """
        Deregister everything Giskard attached to the world and destroy its ros
        entities.

        The world outlives the ros node whenever Giskard is torn down while the world is
        kept, and a callback that publishes on a destroyed node fails. Nothing may
        therefore stay registered on the world.
        """
        self.world_synchronizer.close()
        if self.model_reload_synchronizer is not None:
            self.model_reload_synchronizer.close()
        self.world_fetcher.close()
        self.tf_publisher.stop()
        self.viz_marker_publisher.stop()

    def sanity_check(self):
        self._controlled_joints_sanity_check()

    @property
    def robot(self) -> AbstractRobot:
        return self.robots[0]

    @property
    def robots(self) -> List[AbstractRobot]:
        return self.world_config.world.get_semantic_annotations_by_type(AbstractRobot)

    def _controlled_joints_sanity_check(self):
        world = self.world_config.world
        movable_joints = world.get_connections_by_type(ActiveConnection)
        controlled_joints = self.robot.controlled_connections
        non_controlled_joints = set(movable_joints).difference(set(controlled_joints))
        if len(controlled_joints) == 0 and len(world.connections) > 0:
            raise NoControlledJointsError()
        if len(non_controlled_joints) > 0:
            rospy.node.get_logger().info(
                f"The following joints are non-fixed according to the urdf, "
                f"but not flagged as controlled: {[c.name for c in non_controlled_joints]}."
            )

    def live(self):
        """
        Start Giskard and wait for goals until ROS shuts down.
        """
        try:
            self.setup()
            self.motion_server.live()
            rospy.spinner_thread.join()
        except Exception:
            traceback.print_exc()
        finally:
            self.close_world_model_ros_interface()
            if rclpy.ok():
                rclpy.try_shutdown()
