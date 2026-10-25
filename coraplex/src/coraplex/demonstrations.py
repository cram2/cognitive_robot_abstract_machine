"""
Scaffolding shared by real-world robot demonstrations.

A demonstration acquires a world holding its robot and its scene, and performs one plan
against it. The parts that do not vary between demonstrations -- ROS session ownership,
fetching the world from a running controller, merging the scene into it and wrapping
execution in the right environment -- live here, so a demonstration only writes its own
scene and plan.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from typing_extensions import List, Type

from coraplex.alternative_motion_mapping import AlternativeMotion
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType, VisualizationBackend
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.visualization import VisualizationSession, WorldVisualization
from coraplex.plans.plan_node import PlanNode
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection

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

    The scene, the map the robot acts in together with the objects the plan acts on, is
    built the same way for both kinds of run, by :meth:`build_scene`, and holds no robot.

    A real run takes its world, with the robot in it, from the controller and keeps it
    synchronized, and merges the scene into it unless the controller already holds it,
    so the scene reaches the controller before the plan needs it. A simulated run puts
    the robot into the scene instead, and publishes it to Rviz, so its progress can be
    watched the same way a real run's can.
    """

    used_robot: Type[AbstractRobot]
    """
    The robot this demonstration uses.
    """

    ros_node_name: str = "robot_demonstration"
    """
    Name of the node a real run registers.
    """

    execution_type: ExecutionType = ExecutionType.KINEMATICALLY_SIMULATED
    """
    Whether the plan drives the real robot or a simulated one.
    """

    collision_avoidance: bool = False
    """
    Whether collision avoidance is added to every motion state chart of this run.
    """

    event_segmentation: bool = True
    """
    Whether the events of this run are segmented while the plan is performed, as
    :meth:`segment_events` describes.
    """

    debug: bool = False
    """
    Whether the plan runs in debug mode, logging debug messages and publishing every
    copy of the world a candidate is tried in.
    """

    repetitions: int = 1
    """
    How often the plan is performed against the scene.

    Repeating only makes sense for a plan that leaves the scene as it found it, such as
    one carrying an object away and back again.
    """

    default_visualization_backend: VisualizationBackend = VisualizationBackend.RVIZ
    """Renderer used unless explicitly selected through the environment."""

    visualization: WorldVisualization | None = field(init=False, default=None)
    """The visualization owned by this simulated demonstration."""

    ros_session: RobotDemonstrationRosSession | None = field(init=False, default=None)
    """
    Session held for the duration of a real run, and ``None`` in simulation.
    """

    @abstractmethod
    def scene_specification(self) -> WorldSpecification:
        """
        :return: The map this demonstration acts in, without any robot, see
            :meth:`build_scene`.
        """

    @abstractmethod
    def robot_specification(self) -> RobotSpecification:
        """
        How a simulated run puts :attr:`used_robot` into the scene. A real run takes the
        robot from the controller instead.

        :return: The robot to put into the scene, with where it starts and how fast its
            drive is.
        """

    @abstractmethod
    def is_scene_populated(self, world: World) -> bool:
        """
        Whether ``world`` already holds the scene, for instance because the controller
        serving it was given the scene by an earlier run, in which case a real run does
        not merge it in a second time.

        :param world: The world the controller serves.
        :return: Whether ``world`` holds the scene already.
        """

    def populate_scene(self, world: World) -> None:
        """
        Add what the plan needs to the freshly built map in ``world``, such as the
        objects it acts on and the semantic annotations of the furniture, see
        :meth:`build_scene`.

        Adds nothing unless a demonstration says what it needs.

        :param world: The scene, which holds no robot yet.
        """

    @abstractmethod
    def build_context(self, world: World) -> Context:
        """
        Build the plan context, resolving the robot in ``world``, in debug mode when
        :attr:`debug` is set.
        """

    @abstractmethod
    def build_plan(self, context: Context) -> PlanNode:
        """
        Build the plan this demonstration performs.
        """

    def segment_events(self, world: World) -> AbstractContextManager:
        """
        Segment what happens in ``world`` into events while the plan is performed.

        Segments nothing unless a demonstration says what it wants detected.

        :param world: The world the plan is performed in.
        :return: A context manager that segments the events while it is entered.
        """
        return nullcontext()

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

    def build_scene(self) -> World:
        """
        Build the scene: the map of :meth:`scene_specification`, populated by
        :meth:`populate_scene`.

        :return: A world holding the scene and no robot.
        """
        scene = self.scene_specification().to_domain_object()
        self.populate_scene(scene)
        return scene

    def build_simulated_world(self) -> World:
        """
        Build the world for a simulated run, without contacting a controller: the scene,
        with the robot of :meth:`robot_specification` put into it.

        :return: A world holding the scene and the robot.
        """
        world = self.build_scene()
        self.robot_specification().spawn(world)
        return world

    def fetch_real_world(self) -> World:
        """
        Fetch the world the controller serves and keep it synchronized with it, and
        merge the scene into it unless :meth:`is_scene_populated` finds it there already.

        The scene's root is fixed to the root of the fetched world, which is the map frame
        the robot localizes in.

        :return: The world the controller serves, holding the scene.
        """
        world = self.ros_session.fetch_world()
        WorldSynchronizer(_world=world, node=self.ros_session.node)
        if not self.is_scene_populated(world):
            scene = self.build_scene()
            world.merge_world(
                scene,
                root_connection=FixedConnection(parent=world.root, child=scene.root),
            )
        return world

    def acquire_world(self) -> World:
        """
        Obtain the world to act on: built by :meth:`build_simulated_world` for a
        simulated run, and by :meth:`fetch_real_world` otherwise.

        :return: The world the plan is performed in.
        """
        self.ros_session = RobotDemonstrationRosSession.start(self.ros_node_name)
        VisualizationSession.register(self.stop_visualization)

        if self.execution_type is not ExecutionType.REAL:
            world = self.build_simulated_world()
            self.visualization = WorldVisualization.from_environment(
                world,
                default_backend=self.default_visualization_backend,
                ros_node=self.ros_node,
                collision_visualization=True,
            ).start()
            return world
        return self.fetch_real_world()

    def run(self) -> World:
        """
        Acquire a world, and perform the plan against it.

        :return: The world the demonstration acted on.
        """
        try:
            world = self.acquire_world()
            for _ in range(self.repetitions):
                context = self.build_context(world)
                plan = self.build_plan(context)
                if self.visualization is not None:
                    self.visualization.attach_plan(plan)
                event_segmentation = (
                    self.segment_events(world)
                    if self.event_segmentation
                    else nullcontext()
                )
                with ExecutionEnvironment(
                    execution_type=self.execution_type,
                    collision_avoidance=self.collision_avoidance,
                ), event_segmentation:
                    plan.perform()
        finally:
            self.tear_down()
        return world

    def tear_down(self) -> None:
        """
        Release owned ROS resources after execution.

        An explicitly selected browser viewer remains available for inspection until
        :meth:`stop_visualization`. A borrowed ROS session is left to its owner.
        """
        if self.visualization is not None:
            self.visualization.finish_execution()
            if not self.visualization.is_rendering:
                self.visualization = None
            elif VisualizationSession.is_active():
                return
        if self.ros_session is None or not self.ros_session.owns_context:
            return
        self.ros_session.stop()
        self.ros_session = None

    def stop_visualization(self) -> None:
        """Close the retained viewer and executor, preserving a borrowed ROS context."""
        if self.visualization is not None:
            self.visualization.stop()
            self.visualization = None
        if self.ros_session is not None:
            self.ros_session.stop()
            self.ros_session = None
