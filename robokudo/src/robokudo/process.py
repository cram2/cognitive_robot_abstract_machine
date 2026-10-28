"""
The RoboKudo perception process.

A :class:`RoboKudo` runs one analysis engine on the semantic digital twin perception
perceives into. That world is the world other processes, such as the one executing robot
motions, share, if one of them serves it; a shared world receives their changes and
every change perception makes is published to them. Without one, perception keeps an
empty world of its own.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import timedelta
from threading import Thread
from timeit import default_timer as timer

import rclpy
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from rclpy.executors import SingleThreadedExecutor
from semantic_digital_twin.adapters.ros.world_fetcher import (
    fetch_world_from_service,
    find_world_fetcher_service,
)
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.world import World
from typing_extensions import TYPE_CHECKING, List, Optional, Type

# robokudo.pipeline must be imported before robokudo.garden, whose import otherwise
# trips the circular import between robokudo.pipeline and robokudo.annotators.core.
import robokudo.pipeline
from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.defs import LOGGING_IDENTIFIER_MAIN_EXECUTABLE
from robokudo.exceptions import (
    EngineRootIsNotAPipelineError,
    RoboKudoAlreadyRunningError,
)
from robokudo.garden import grow_tree
from robokudo.identifier import BBIdentifier
from robokudo.perceived_objects import PerceivedObjects
from robokudo.pipeline import Pipeline
from robokudo.utils.tree import find_parent_of_type, setup_with_descendants_rk

if TYPE_CHECKING:
    from py_trees_ros.trees import BehaviourTree
    from rclpy.node import Node

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
    The node perception communicates through, served by :attr:`executor` until
    :meth:`close`.
    """

    analysis_engine: AnalysisEngineInterface
    """
    The engine whose behaviour tree perceives.
    """

    world_discovery_timeout: timedelta = timedelta(seconds=2)
    """
    How long to look for a process serving a shared world before perceiving into a world
    of its own.
    """

    world: World = field(init=False)
    """
    The world perception perceives into: the shared world, or an empty world of its own
    if no process serves one.
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

    executor: SingleThreadedExecutor = field(init=False)
    """
    Serves the services, actions and subscriptions of :attr:`node`.
    """

    executor_thread: Thread = field(init=False)
    """
    The thread :attr:`executor` serves :attr:`node` in.
    """

    def __post_init__(self) -> None:
        """
        Serve :attr:`node` in the background and join the world a process serves,
        keeping it in sync in both directions: changes of the other processes are applied
        to it and changes perception makes are published. Without such a process,
        perceive into an empty world of its own.

        .. note:: No other executor may serve :attr:`node`.
        """
        self.executor = SingleThreadedExecutor(context=self.node.context)
        self.executor.add_node(self.node)
        self.executor_thread = Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        service = find_world_fetcher_service(
            self.node, timeout_seconds=self.world_discovery_timeout.total_seconds()
        )
        if service is None:
            logging.getLogger(LOGGING_IDENTIFIER_MAIN_EXECUTABLE).warning(
                "No process serves a shared world; perceiving into a world of its own."
            )
            self.world = World()
        else:
            self.world = fetch_world_from_service(self.node)
            self.synchronizer = WorldSynchronizer(node=self.node, _world=self.world)
        self.perceived_objects = PerceivedObjects(world=self.world)

    @classmethod
    def from_blackboard(cls) -> RoboKudo:
        """
        The RoboKudo running in this process.

        :return: The RoboKudo stored on the blackboard.
        """
        return Blackboard().get(BBIdentifier.ROBOKUDO)

    @property
    def pipeline(self) -> Pipeline:
        """
        The pipeline :attr:`analysis_engine` perceives with: the one of :attr:`tree` no
        other pipeline encloses.

        Its finished passes are put into the world.
        """
        return next(
            behaviour
            for behaviour in self.tree.root.iterate()
            if isinstance(behaviour, Pipeline)
            and find_parent_of_type(behaviour, Pipeline) is None
        )

    # %% lifecycle

    def start(
        self, include_gui: bool = True, visualizer_types: Optional[List[Type]] = None
    ) -> None:
        """
        Become the RoboKudo of this process and build the behaviour tree, whose
        pipelines perceive into :attr:`world`.

        :param include_gui: Whether to show the visualizers.
        :param visualizer_types: The visualizers to show; all of them when None.
        :raises RoboKudoAlreadyRunningError: If another RoboKudo runs in this process.
        :raises EngineRootIsNotAPipelineError: If the tree of :attr:`analysis_engine` is
            not rooted at a pipeline.
        """
        blackboard = Blackboard()
        if blackboard.exists(BBIdentifier.ROBOKUDO):
            raise RoboKudoAlreadyRunningError()
        engine_tree = self.analysis_engine.implementation()
        if not isinstance(engine_tree, Pipeline):
            raise EngineRootIsNotAPipelineError(
                analysis_engine_name=self.analysis_engine.name(),
                root_type=type(engine_tree),
            )
        blackboard.set(BBIdentifier.ROBOKUDO, self)

        for behaviour in engine_tree.iterate():
            if isinstance(behaviour, Pipeline):
                behaviour.world = self.world
        self.tree = grow_tree(
            engine_tree,
            node=self.node,
            include_gui=include_gui,
            visualizer_types=visualizer_types,
        )
        setup_with_descendants_rk(self.tree)

    def run(self, tick_rate: float = 5.0) -> None:
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
        :attr:`pipeline` that finished successfully in this tick into the world.
        """
        self.tree.tick()
        pipeline = self.pipeline
        if pipeline.status != Status.SUCCESS:
            return
        self.perceived_objects.apply(pipeline.cas)

    def close(self) -> None:
        """
        Stop sharing the world and serving :attr:`node`, and make room for another
        RoboKudo in this process.
        """
        if self.synchronizer is not None:
            self.synchronizer.close()
            self.synchronizer = None
        self.executor.shutdown()
        self.executor_thread.join()
        blackboard = Blackboard()
        if (
            blackboard.exists(BBIdentifier.ROBOKUDO)
            and blackboard.get(BBIdentifier.ROBOKUDO) is self
        ):
            blackboard.unset(BBIdentifier.ROBOKUDO)
