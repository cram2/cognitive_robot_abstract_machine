from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import json

from typing_extensions import Callable, Self

import rclpy
from json_msgs.action import JsonAction
from std_msgs.msg import String
from sqlalchemy.orm import sessionmaker

from giskardpy.data_types.exceptions import NoControlledJointsError
from semantic_digital_twin.adapters.ros import rospy
from giskardpy.middleware.ros2.action_server import ActionServerHandler
from giskardpy.middleware.ros2.child_choices import (
    ChildChoiceMessage,
    ChildSentByClient,
    child_choices_topic,
)
from giskardpy.middleware.ros2.client_presence import ClientWatchdog, HeartbeatPresence
from giskardpy.middleware.ros2.control_loop import ControlLoop
from giskardpy.middleware.ros2.feedback_publisher import ActionFeedbackPublisher
from giskardpy.middleware.ros2.graceful_shutdown import GracefulShutdownSignals
from giskardpy.middleware.ros2.cycle_counter import CycleCounter
from semantic_digital_twin.input_synchronization import WorldStateInputs
from giskardpy.middleware.ros2.motion_server import MotionServer
from giskardpy.middleware.ros2.post_goal_plotters import (
    GoalGanttChartPlotter,
    GoalTrajectoryPlotter,
    MotionStatechartPlotter,
    PostGoalPlotter,
)
from giskardpy.middleware.ros2.robot_interface_config import (
    RobotInterfaceConfig,
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.middleware.ros2.world_updates import (
    IncomingWorldUpdates,
    OutgoingWorldUpdates,
)
from giskardpy.model.world_config import WorldConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.ormatic.utils import create_engine
from krrood.patterns.caching import clear_memoization_cache
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
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from cramph.composites import ChildChooserAccess
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor

logger = logging.getLogger(__name__)


@dataclass
class Giskard:
    """
    A motion server for one robot.

    Instantiate it with the configs of the setup and call :meth:`live`.
    """

    world_config: WorldConfig
    """
    Which robot description is loaded and how the robot is attached to the world.
    """

    server_config: GiskardServerConfig
    """
    The goal lifecycle and the debug output of the motion server.
    """

    robot_interface_config: RobotInterfaceConfig
    """
    How Giskard talks to the robot, which also decides whether a motion is simulated or
    executed.
    """

    qp_controller_config: QPControllerConfig = field(default_factory=QPControllerConfig)
    """
    The control frequency and the tuning of the controller.
    """

    executor: StatechartExecutor = field(init=False)
    """
    Runs the motion statecharts of the goals.
    """

    motion_server: MotionServer | None = field(init=False, default=None)
    """
    Accepts goals and executes them.
    """

    world_synchronizer: WorldSynchronizer | None = field(init=False, default=None)
    """
    Shares changes of the world with other processes.
    """

    tf_publisher: TFPublisher | None = field(init=False, default=None)
    """
    Publishes the frames of the world on tf.
    """

    viz_marker_publisher: VizMarkerPublisher | None = field(init=False, default=None)
    """
    Publishes the world as visualization markers.
    """

    collision_marker_publisher: CollisionVisualizationMarkerPublisher = field(
        init=False
    )
    """
    Publishes the closest points of the collision checks as visualization markers.
    """

    model_reload_synchronizer: ModelReloadSynchronizer | None = field(
        init=False, default=None
    )
    """
    Reloads the world from the database when another process asks for it, ``None``
    without a database.
    """

    world_fetcher: FetchWorldServer | None = field(init=False, default=None)
    """
    Serves the whole world to processes that join later.
    """

    @classmethod
    def standalone(
        cls, world_config: WorldConfig, qp_controller_config: QPControllerConfig
    ) -> Self:
        """
        :param world_config: Which robot description is loaded and how the robot is
            attached to the world.
        :param qp_controller_config: The control frequency and the tuning of the
            controller.
        :return: A Giskard that simulates the robot instead of commanding it, with
            debug output enabled.
        """
        return cls(
            world_config=world_config,
            robot_interface_config=StandAloneRobotInterfaceConfig(),
            server_config=GiskardServerConfig(debug_mode=True),
            qp_controller_config=qp_controller_config,
        )

    @classmethod
    def launch(
        cls, create_giskard: Callable[[], Giskard], node_name: str = "giskard"
    ) -> None:
        """
        Run a Giskard as a ros node until ROS shuts down.

        :param create_giskard: Builds the Giskard to run. It is called once the ros node
            exists, so it can read from topics.
        :param node_name: The name of the ros node.
        """
        rospy.init_node(node_name)
        create_giskard().live()

    def setup(self):
        """
        Initialize the world, the ros interfaces and the motion server.

        You usually don't need to call this.
        """
        node_logger = rospy.get_node().get_logger()
        node_logger.info("Building the world.")
        with self.world_config.world.modify_world():
            self.world_config.setup_world()
            clear_memoization_cache(self.world_config.world)
            self.executor = StatechartExecutor(
                context=StatechartContext(world=self.world_config.world),
                pacer=self.robot_interface_config.create_pacer(),
                extensions=[
                    RosNodeAccess(rospy.get_node()),
                    MotionControl(qp_controller_config=self.qp_controller_config),
                ],
            )

        self.setup_world_model_ros_interface()
        self.motion_server = self.create_motion_server()
        node_logger.info("Connecting to the robot.")
        self.robot_interface_config.attach(self)
        self.robot_interface_config.connect()
        self.sanity_check()

    def create_motion_server(self) -> MotionServer:
        """
        Build the goal lifecycle around the executor.
        """
        world = self.world_config.world
        action_server = ActionServerHandler(
            action_name=f"{rospy.get_node().get_name()}/command", action_type=JsonAction
        )
        client_watchdog = ClientWatchdog(
            presence=HeartbeatPresence(node=rospy.get_node()),
        )
        outgoing_world_updates = OutgoingWorldUpdates(
            world_synchronizer=self.world_synchronizer
        )
        feedback_publisher = ActionFeedbackPublisher(
            executor=self.executor,
            action_server=action_server,
            outgoing_world_updates=outgoing_world_updates,
        )
        cycle_counter = CycleCounter()
        world_updates = IncomingWorldUpdates(
            world_synchronizer=self.world_synchronizer,
            model_reload_synchronizer=self.model_reload_synchronizer,
        )
        self.receive_child_choices(world_updates, action_server)
        control_loop = ControlLoop(
            executor=self.executor,
            action_server=action_server,
            client_watchdog=client_watchdog,
            feedback_publisher=feedback_publisher,
            inputs=WorldStateInputs(world=world, reapplies_inputs=True),
            cycle_counter=cycle_counter,
            world_updates=world_updates,
        )
        return MotionServer(
            executor=self.executor,
            action_server=action_server,
            control_loop=control_loop,
            client_watchdog=client_watchdog,
            world_updates=world_updates,
            outgoing_world_updates=outgoing_world_updates,
            feedback_publisher=feedback_publisher,
            inputs=WorldStateInputs(world=world),
            cycle_counter=cycle_counter,
            idle_frequency=self.server_config.idle_frequency,
            post_goal_plotters=self.create_post_goal_plotters(),
        )

    def receive_child_choices(
        self, world_updates: IncomingWorldUpdates, action_server: ActionServerHandler
    ) -> None:
        """
        Let the nodes of a goal that choose their child run the children the client
        sends for them.
        """
        chooser = ChildSentByClient(
            world_updates=world_updates, action_server=action_server
        )
        self.executor.context.add_extension(ChildChooserAccess(choosers=[chooser]))
        node = rospy.get_node()
        node.create_subscription(
            String,
            child_choices_topic(node.get_name()),
            lambda message: chooser.receive(
                ChildChoiceMessage.from_json(json.loads(message.data))
            ),
            10,
        )

    def create_post_goal_plotters(self) -> list[PostGoalPlotter]:
        """
        Create the debug plotters that are configured and let them record what they
        need.
        """
        if not self.server_config.debug_mode:
            return []
        plotters: list[PostGoalPlotter] = []
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
            assert semantic_digital_twin_database_uri is not None, (
                "Please set the SEMANTIC_DIGITAL_TWIN_DATABASE_URI environment variable."
            )

            engine = create_engine(semantic_digital_twin_database_uri)
            session = sessionmaker(bind=engine)()

            self.model_reload_synchronizer = ModelReloadSynchronizer(
                node=rospy.get_node(),
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
        rospy.get_node().get_logger().info("Publishing the world.")
        self.world_synchronizer = WorldSynchronizer(
            _world=self.world_config.world,
            node=rospy.get_node(),
            defer_incoming_updates=True,
        )
        self.world_fetcher = FetchWorldServer(
            node=rospy.get_node(), world=self.world_config.world
        )
        rospy.get_node().get_logger().info(
            "Listening for frames that are already published on tf."
        )
        self.tf_publisher = TFPublisher.create_with_ignore_existing_tf(
            node=rospy.get_node(), world=self.world_config.world
        )
        rospy.get_node().get_logger().info("Starting the visualization.")
        self.viz_marker_publisher = VizMarkerPublisher(
            node=rospy.get_node(), _world=self.world_config.world
        )
        self.collision_marker_publisher = CollisionVisualizationMarkerPublisher(
            node=rospy.get_node(), throttle=5, world=self.world_config.world
        )

    def close_world_model_ros_interface(self):
        """
        Deregister everything Giskard attached to the world and destroy its ros
        entities.

        The world outlives the ros node whenever Giskard is torn down while the world is
        kept, and a callback that publishes on a destroyed node fails. Nothing may
        therefore stay registered on the world.

        .. note:: Giskard can fail before all of them exist, so only the ones that were
            created are closed.
        """
        if self.world_synchronizer is not None:
            self.world_synchronizer.close()
        if self.model_reload_synchronizer is not None:
            self.model_reload_synchronizer.close()
        if self.world_fetcher is not None:
            self.world_fetcher.close()
        if self.motion_server is not None:
            self.motion_server.client_watchdog.presence.stop()
        if self.tf_publisher is not None:
            self.tf_publisher.stop()
        if self.viz_marker_publisher is not None:
            self.viz_marker_publisher.stop()

    def sanity_check(self):
        self._controlled_joints_sanity_check()

    @property
    def robot(self) -> AbstractRobot:
        return self.robots[0]

    @property
    def robots(self) -> list[AbstractRobot]:
        return self.world_config.world.get_semantic_annotations_by_type(AbstractRobot)

    def _controlled_joints_sanity_check(self):
        world = self.world_config.world
        movable_joints = world.get_connections_by_type(ActiveConnection)
        controlled_joints = [joint for joint in movable_joints if joint.is_controlled]
        non_controlled_joints = set(movable_joints).difference(set(controlled_joints))
        if len(controlled_joints) == 0 and len(world.connections) > 0:
            raise NoControlledJointsError()
        if len(non_controlled_joints) > 0:
            rospy.get_node().get_logger().info(
                f"The following joints are non-fixed according to the urdf, "
                f"but not flagged as controlled: {[c.name for c in non_controlled_joints]}."
            )

    def live(self):
        """
        Start Giskard and wait for goals until ROS shuts down.

        An error is raised after Giskard has cleaned up, so that a robot that could not
        be set up makes the process fail.
        """
        with GracefulShutdownSignals():
            try:
                self.setup()
                self.motion_server.live()
                rospy.spinner_thread.join()
            except KeyboardInterrupt:
                rospy.get_node().get_logger().info(
                    "Giskard was interrupted, shutting down."
                )
            finally:
                self.close_world_model_ros_interface()
                if rclpy.ok():
                    rclpy.try_shutdown()
