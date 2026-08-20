"""
Scaffolding shared by real-world robot demonstrations.

A demonstration acquires a world, populates it with the objects it manipulates and
performs one plan against it. The parts that do not vary between demonstrations -- ROS
session ownership, fetching the world from a running controller and wrapping execution in
the right environment -- live here, so a demonstration only writes its own scene and plan.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from typing_extensions import ClassVar, List, Type

from coraplex.alternative_motion_mapping import AlternativeMotion
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.plans.plan_node import PlanNode
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World

# %% ros session

WORLD_FETCH_TIMEOUT_SECONDS = 300
"""
How long to wait for a controller to serve its world.

Matches giskardpy's own client (``giskardpy/middleware/ros2/python_interface.py``), which
waits this long for the same race: the world-fetcher server is still parsing the URDF and
starting up when a shorter budget would expire.
"""

SPIN_THREAD_JOIN_TIMEOUT_SECONDS = 5.0
"""
How long to wait for the executor's spin thread to exit after :meth:`Executor.shutdown`.

A daemon thread still running rclpy's native code when the interpreter starts finalizing
gets torn down mid-call, which crashes the process rather than raising a catchable
exception. Joining it here ensures it has actually left that code first.
"""


@dataclass
class RobotDemonstrationRosSession:
    """
    A ROS node whose executor is spun on a background thread.

    ..note:: Service responses are delivered by the executor, so a node that is not spun
        blocks every call until its timeout.
    """

    node: Node
    """
    The node that service calls and synchronizers are attached to.
    """

    executor: SingleThreadedExecutor
    """
    Executor delivering this node's callbacks.
    """

    owns_context: bool
    """
    Whether this session initialized rclpy and must therefore shut it down again.
    """

    spin_thread: threading.Thread = field(init=False)
    """
    Thread running :attr:`executor`'s spin loop, joined again in :meth:`stop`.
    """

    @classmethod
    def start(cls, node_name: str) -> RobotDemonstrationRosSession:
        """
        Create a node and spin its executor, initializing rclpy if nothing else has.

        :param node_name: Name to register the node under.
        """
        owns_context = not rclpy.ok()
        if owns_context:
            rclpy.init()
        node = rclpy.create_node(node_name)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        session = cls(node=node, executor=executor, owns_context=owns_context)
        session.spin_thread = threading.Thread(
            target=executor.spin, daemon=True, name=f"{node_name}-executor"
        )
        session.spin_thread.start()
        return session

    def fetch_world(
        self, timeout_seconds: float = WORLD_FETCH_TIMEOUT_SECONDS
    ) -> World:
        """
        Fetch the world that a controller is serving.

        :param timeout_seconds: How long to wait for the service to answer.
        """
        return fetch_world_from_service(node=self.node, timeout_seconds=timeout_seconds)

    def stop(self) -> None:
        """
        Destroy the node, and release the ROS context if this session created it.
        """
        self.executor.shutdown()
        self.spin_thread.join(timeout=SPIN_THREAD_JOIN_TIMEOUT_SECONDS)
        self.node.destroy_node()
        if self.owns_context and rclpy.ok():
            rclpy.shutdown()

    def __enter__(self) -> RobotDemonstrationRosSession:
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.stop()


# %% demonstrations


@dataclass
class RobotDemonstration(ABC):
    """
    A robot demonstration that runs either in simulation or against a real controller.

    A real run takes its world from the controller and keeps it synchronized, so the
    objects the demonstration spawns reach the controller before the plan needs them. A
    simulated run builds its own world and publishes it to Rviz, so its progress can be
    watched the same way a real run's can.
    """

    used_robot: Type[AbstractRobot]
    """
    The robot this demonstration uses.
    """

    ros_node_name: ClassVar[str] = "robot_demonstration"
    """
    Name of the node a real run registers.
    """

    execution_type: ExecutionType = ExecutionType.SIMULATED
    """
    Whether the plan drives the real robot or a simulated one.
    """

    collision_avoidance: bool = False
    """
    Whether collision avoidance is added to every motion state chart of this run.
    """

    repetitions: int = 1
    """
    How often the plan is performed against the scene.

    Repeating only makes sense for a plan that leaves the scene as it found it, such as
    one carrying an object away and back again.
    """

    ros_session: RobotDemonstrationRosSession | None = field(init=False, default=None)
    """
    Session held for the duration of a real run, and ``None`` in simulation.
    """

    @abstractmethod
    def build_simulated_world(self) -> World:
        """
        Build the world for a simulated run, without contacting a controller.
        """

    @abstractmethod
    def is_scene_populated(self, world: World) -> bool:
        """
        Implementations should define a condition which without a doubt define the scene was already populated
        before, populating a scene in a different process multiple times, for example because one restarts the demo in
        the current process.
        """

    @abstractmethod
    def populate_scene(self, world: World) -> None:
        """
        Add the objects and furniture this demonstration acts on to ``world``.
        """

    @abstractmethod
    def build_context(self, world: World) -> Context:
        """
        Build the plan context, resolving the robot in ``world``.
        """

    @abstractmethod
    def build_plan(self, context: Context) -> PlanNode:
        """
        Build the plan this demonstration performs.
        """

    @property
    def ros_node(self) -> Node | None:
        """
        The node a real run drives the controller through.
        """
        if self.ros_session is None:
            return None
        return self.ros_session.node

    @property
    def alternative_motion_mappings(self) -> List[Type[AlternativeMotion]]:
        """
        Every alternative motion mapping known to coraplex, for every robot.

        Resolution filters by ``used_robot`` and execution type, so handing over the
        full set is always safe and needs no per-robot selection here.
        """
        return AlternativeMotion.discover_all()

    def acquire_world(self) -> World:
        """
        Obtain the world to act on: from the running controller for a real run, and from
        this demonstration's own description otherwise.
        """
        self.ros_session = RobotDemonstrationRosSession.start(self.ros_node_name)

        if self.execution_type is not ExecutionType.REAL:
            world = self.build_simulated_world()
            viz = VizMarkerPublisher(node=self.ros_node, _world=world)
            viz.with_tf_publisher()
            return world

        world = self.ros_session.fetch_world()
        WorldSynchronizer(_world=world, node=self.ros_session.node)
        return world

    def run(self) -> World:
        """
        Acquire a world, populate it if needed, and perform the plan against it.

        :return: The world the demonstration acted on.
        """
        world = self.acquire_world()
        try:
            if not self.is_scene_populated(world):
                self.populate_scene(world)
            for _ in range(self.repetitions):
                plan = self.build_plan(self.build_context(world))
                with ExecutionEnvironment(
                    execution_type=self.execution_type,
                    collision_avoidance=self.collision_avoidance,
                ):
                    plan.perform()
        finally:
            self.tear_down()
        return world

    def tear_down(self) -> None:
        """
        Release the ROS session if this demonstration started the ROS context.

        A session running inside a context somebody else owns is left alone: that owner
        decides when its nodes go away, and destroying this one early can drop world
        modifications that have not reached the controller yet.
        """
        if self.ros_session is None or not self.ros_session.owns_context:
            return
        self.ros_session.stop()
        self.ros_session = None
