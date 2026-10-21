"""
The RoboKudo perception process.

A :class:`RoboKudo` runs one analysis engine on the semantic digital twin perception
perceives into. That world is either a world of its own or the world other processes,
such as the one executing robot motions, share; a shared world receives their changes
and every change perception makes is published to them.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import timedelta
from timeit import default_timer as timer

import rclpy
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Camera
from semantic_digital_twin.world import World
from typing_extensions import TYPE_CHECKING, List, Optional, Self, Type

# robokudo.pipeline must be imported before robokudo.garden, whose import otherwise
# trips the circular import between robokudo.pipeline and robokudo.annotators.core.
import robokudo.pipeline
from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.camera_placement import CameraPlacement, SharedWorldCameraPlacement
from robokudo.defs import LOGGING_IDENTIFIER_MAIN_EXECUTABLE
from robokudo.exceptions import MultipleRobotsInWorldError, RoboKudoAlreadyRunningError
from robokudo.garden import grow_tree
from robokudo.identifier import BBIdentifier
from robokudo.perceived_objects import PerceivedObjects
from robokudo.pipeline import Pipeline
from robokudo.utils.tree import setup_with_descendants_rk

if TYPE_CHECKING:
    from py_trees_ros.trees import BehaviourTree
    from rclpy.node import Node

DEFAULT_WORLD_FETCH_TIMEOUT = timedelta(seconds=10)
"""
How long to wait for another process to serve the shared world.
"""

DEFAULT_TICK_RATE = 5.0
"""
How often the behaviour tree is ticked per second.
"""

# %% perception process


@dataclass
class RoboKudo:
    """
    The RoboKudo perception process, running an analysis engine on a world.

    The engine's pipelines perceive into :attr:`world`; annotators reach it through
    their pipeline. Only one RoboKudo runs per process, since the blackboard its
    behaviour tree uses is shared by the whole process.
    """

    node: Node
    """
    The node perception communicates through.
    """

    analysis_engine: AnalysisEngineInterface
    """
    The engine whose behaviour tree perceives.
    """

    world: World = field(default_factory=World)
    """
    The world perception perceives into; an empty world of its own by default.
    """

    perceived_objects: PerceivedObjects = field(init=False)
    """
    The objects perception believes in, kept as bodies in :attr:`world`.
    """

    synchronizer: Optional[WorldSynchronizer] = field(init=False, default=None)
    """
    Shares the world with other processes; None while perception keeps its world to
    itself.
    """

    tree: Optional[BehaviourTree] = field(init=False, default=None)
    """
    The behaviour tree of :attr:`analysis_engine`; None until :meth:`start`.
    """

    applied_cas_id: Optional[int] = field(init=False, default=None)
    """
    The id of the last CAS of :attr:`pipeline` whose objects were put into the world.
    """

    pipeline: Optional[Pipeline] = field(init=False, default=None)
    """
    The pipeline :attr:`analysis_engine` perceives with, whose finished passes are put
    into the world; None until :meth:`start`, and for an engine without a pipeline.
    """

    def __post_init__(self) -> None:
        self.perceived_objects = PerceivedObjects(world=self.world)

    @classmethod
    def joining_shared_world(
        cls,
        node: Node,
        analysis_engine: AnalysisEngineInterface,
        fetch_timeout: timedelta = DEFAULT_WORLD_FETCH_TIMEOUT,
    ) -> Self:
        """
        Create a RoboKudo that perceives into the world other processes share.

        The world is fetched from the process serving it, then kept in sync in both
        directions: changes of the other processes are applied to it and changes
        perception makes are published.

        :param node: The node to fetch and synchronize the world through. It has to be
            spun by an executor.
        :param analysis_engine: The engine whose behaviour tree perceives.
        :param fetch_timeout: How long to wait for a process serving the world.
        :return: The RoboKudo sharing the world.
        """
        world = fetch_world_from_service(
            node, timeout_seconds=fetch_timeout.total_seconds()
        )
        robokudo = cls(node=node, analysis_engine=analysis_engine, world=world)
        robokudo.synchronizer = WorldSynchronizer(node=node, _world=world)
        return robokudo

    @classmethod
    def from_blackboard(cls) -> RoboKudo:
        """
        The RoboKudo running in this process.

        :return: The RoboKudo stored on the blackboard.
        """
        return Blackboard().get(BBIdentifier.ROBOKUDO)

    @property
    def camera(self) -> Optional[Camera]:
        """
        The default camera of the robot in the perception world; None if the world holds
        no robot.

        :raises MultipleRobotsInWorldError: If the world holds several robots.
        """
        robots = self.world.get_semantic_annotations_by_type(AbstractRobot)
        if not robots:
            return None
        if len(robots) > 1:
            raise MultipleRobotsInWorldError(robot_count=len(robots))
        return robots[0].get_default_camera()

    # %% lifecycle

    def start(
        self, include_gui: bool = True, visualizer_types: Optional[List[Type]] = None
    ) -> None:
        """
        Become the RoboKudo of this process and build the behaviour tree, whose
        pipelines perceive into :attr:`world`. Their cameras are placed without adding
        world frames while :attr:`world` is shared.

        :param include_gui: Whether to show the visualizers.
        :param visualizer_types: The visualizers to show; all of them when None.
        :raises RoboKudoAlreadyRunningError: If another RoboKudo runs in this process.
        """
        blackboard = Blackboard()
        if blackboard.exists(BBIdentifier.ROBOKUDO):
            raise RoboKudoAlreadyRunningError()
        blackboard.set(BBIdentifier.ROBOKUDO, self)

        engine_tree = self.analysis_engine.implementation()
        if isinstance(engine_tree, Pipeline):
            self.pipeline = engine_tree
        camera_placement = (
            CameraPlacement(world=self.world)
            if self.synchronizer is None
            else SharedWorldCameraPlacement(world=self.world)
        )
        for behaviour in engine_tree.iterate():
            if isinstance(behaviour, Pipeline):
                behaviour.world = self.world
                behaviour.camera_placement = camera_placement
        self.tree = grow_tree(
            engine_tree,
            node=self.node,
            include_gui=include_gui,
            visualizer_types=visualizer_types,
        )
        setup_with_descendants_rk(self.tree)

    def run(self, tick_rate: float = DEFAULT_TICK_RATE) -> None:
        """
        Tick the behaviour tree until its first top-level child fails or the ROS context
        of :attr:`node` shuts down, as it does on SIGINT and SIGTERM.

        An exception raised while ticking is logged and ticking goes on.

        :param tick_rate: How often to tick per second.
        """
        logger = logging.getLogger(LOGGING_IDENTIFIER_MAIN_EXECUTABLE)
        logger.info(f"Running AE named '{self.analysis_engine.name()}'...")
        Blackboard().set("CAS", None)

        interval = 1.0 / tick_rate
        tick_count = 0
        last_tick = time.monotonic()
        while rclpy.ok(context=self.node.context):
            elapsed = time.monotonic() - last_tick
            if elapsed < interval:
                time.sleep(interval - elapsed)
                continue
            last_tick = time.monotonic()
            try:
                logger.debug(f"--------- Tick {tick_count} ---------")
                start = timer()
                self.tick()
                logger.debug(f"Tick took {timer() - start:.4f} seconds")
                if (
                    self.tree.root.children
                    and self.tree.root.children[0].status == Status.FAILURE
                ):
                    return
                tick_count += 1
            except Exception as exception:
                logger.error(f"Exception: {exception}")
                logger.error("Traceback:\n" + traceback.format_exc())

    def tick(self) -> None:
        """
        Tick the behaviour tree once, and put the objects found by a pass of
        :attr:`pipeline` that finished successfully into the world, once per pass.
        """
        self.tree.tick()
        if self.pipeline is None or self.pipeline.status != Status.SUCCESS:
            return
        cas = self.pipeline.cas
        if cas.cas_id == self.applied_cas_id:
            return
        self.perceived_objects.apply(cas)
        self.applied_cas_id = cas.cas_id

    def close(self) -> None:
        """
        Stop sharing the world and make room for another RoboKudo in this process.
        """
        if self.synchronizer is not None:
            self.synchronizer.close()
            self.synchronizer = None
        blackboard = Blackboard()
        if (
            blackboard.exists(BBIdentifier.ROBOKUDO)
            and blackboard.get(BBIdentifier.ROBOKUDO) is self
        ):
            blackboard.unset(BBIdentifier.ROBOKUDO)
