from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass, field
from typing import List

import rclpy
from sqlalchemy.orm import sessionmaker

from giskardpy.data_types.exceptions import SetupException
from giskardpy.executor import Executor, SimulationPacer
from giskardpy.model.world_config import WorldConfig
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.ros_executor import Ros2Executor
from giskardpy.middleware.ros2.behavior_tree_config import (
    BehaviorTreeConfig,
    StandAloneBTConfig,
)
from giskardpy.middleware.ros2.robot_interface_config import RobotInterfaceConfig
from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from krrood.ormatic.utils import create_engine
from krrood.utils import clear_memoization_cache
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    ModelSynchronizer,
    StateSynchronizer,
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
    :param world_config: A world configuration. Use a predefined one or implement your own WorldConfig class.
    :param robot_interface_config: How Giskard talk to the robot. You probably have to implement your own RobotInterfaceConfig.
    :param collision_avoidance_config: default is no collision avoidance or implement your own collision_avoidance_config.
    :param behavior_tree_config: default is open loop mode
    :param qp_controller_config: default is good for almost all cases
    :param additional_goal_package_paths: specify paths that Giskard needs to import to find your custom Goals.
                                          Giskard will run 'from <additional path> import *' for each additional
                                          path in the list.
    :param additional_monitor_package_paths: specify paths that Giskard needs to import to find your custom Monitors.
                                          Giskard will run 'from <additional path> import *' for each additional
                                          path in the list.
    """

    world_config: WorldConfig
    behavior_tree_config: BehaviorTreeConfig
    robot_interface_config: RobotInterfaceConfig
    qp_controller_config: QPControllerConfig = field(default_factory=QPControllerConfig)
    executor: Executor = field(init=False)
    model_synchronizer: ModelSynchronizer = field(init=False)
    state_synchronizer: StateSynchronizer = field(init=False)
    tf_publisher: TFPublisher = field(init=False)
    viz_marker_publisher: VizMarkerPublisher = field(init=False)
    model_reload_synchronizer: ModelReloadSynchronizer = field(init=False)
    world_fetcher: FetchWorldServer = field(init=False)

    def __post_init__(self):
        GiskardBlackboard().giskard = self

    def setup(self):
        """
        Initialize the behavior tree and world. You usually don't need to call this.
        """
        with self.world_config.world.modify_world():
            self.world_config.setup_world()
            clear_memoization_cache(self.world_config.world)
            if isinstance(self.behavior_tree_config, StandAloneBTConfig):
                real_time_factor = None
            else:
                real_time_factor = 1.0
            self.executor = Ros2Executor(
                ros_node=rospy.node,
                context=MotionStatechartContext(
                    world=self.world_config.world,
                    qp_controller_config=self.qp_controller_config,
                ),
                pacer=SimulationPacer(real_time_factor=real_time_factor),
            )

            self.behavior_tree_config.setup()

            self.robot_interface_config.setup()

        self.sanity_check()
        self.setup_world_model_ros_interface()
        GiskardBlackboard().tree.setup(rospy.node)

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
            )
        except AssertionError as e:
            logger.warning(
                f'Model reload synchronization not available because "SEMANTIC_DIGITAL_TWIN_DATABASE_URI" is not set.'
            )
            self.model_reload_synchronizer = None

        self.model_synchronizer = ModelSynchronizer(
            _world=self.world_config.world, node=rospy.node
        )
        self.model_synchronizer.pause()
        self.state_synchronizer = StateSynchronizer(
            _world=self.world_config.world, node=rospy.node
        )
        self.state_synchronizer.pause()
        self.world_fetcher = FetchWorldServer(
            node=rospy.node, world=self.world_config.world
        )
        self.tf_publisher = TFPublisher.create_with_ignore_existing_tf(
            node=rospy.node, world=self.world_config.world
        )
        self.tf_publisher.pause()
        self.viz_marker_publisher = VizMarkerPublisher(
            node=rospy.node, _world=self.world_config.world
        )

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
            raise SetupException("No joints are flagged as controlled.")
        if len(non_controlled_joints) > 0:
            rospy.node.get_logger().info(
                f"The following joints are non-fixed according to the urdf, "
                f"but not flagged as controlled: {[c.name for c in non_controlled_joints]}."
            )

    def live(self):
        """
        Start Giskard.
        """
        try:
            self.setup()
            GiskardBlackboard().tree.live()
        except Exception as e:
            traceback.print_exc()
            rclpy.shutdown()
