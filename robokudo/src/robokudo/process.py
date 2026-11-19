"""
The RoboKudo perception process.

A :class:`RoboKudo` runs one analysis engine on the semantic digital twin perception
perceives into: the world another process serves, kept in sync with it in both
directions, or else an empty world of its own.
"""

from __future__ import annotations

import logging
import time
import traceback
from contextlib import ExitStack
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
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher, TfFrameNames
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.world import World
from typing_extensions import TYPE_CHECKING

# robokudo.pipeline must be imported before robokudo.garden, whose import otherwise
# trips the circular import between robokudo.pipeline and robokudo.annotators.core.
import robokudo.pipeline
from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.defs import LOGGING_IDENTIFIER_MAIN_EXECUTABLE, PACKAGE_NAME
from robokudo.exceptions import (
    CameraPoseOutsideWorldError,
    EngineRootIsNotAPipelineError,
    RoboKudoAlreadyRunningError,
)
from robokudo.garden import grow_tree
from robokudo.identifier import BBIdentifier
from robokudo.perception_beliefs import ObjectMatcher, PerceptionBelief
from robokudo.pipeline import Pipeline
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.tree import setup_with_descendants_rk

if TYPE_CHECKING:
    from py_trees_ros.trees import BehaviourTree
    from rclpy.node import Node

# %% perception process


@dataclass(kw_only=True)
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

    world_fetch_timeout: timedelta = timedelta(seconds=300)
    """
    How long the process serving a shared world may take to send it.
    """

    publishes_own_world: bool = False
    """
    Whether to publish a world of its own to RViz, as markers and as the tf frames of
    the objects in it.

    A shared world is published by the process serving it.
    """

    minimum_class_confidence: float = 0.5
    """
    How confident a classification has to be for an object to be believed of the class
    it names; a less confident one is believed a :class:`PerceivedObject`.
    """

    world: World = field(init=False)
    """
    The world perception perceives into: the shared world, or an empty world of its own
    if no process serves one.
    """

    world_publisher: VizMarkerPublisher | None = field(init=False, default=None)
    """
    Publishes a world of its own, from the first pass on that gave it a world frame, if
    :attr:`publishes_own_world`; None until then.
    """

    synchronizer: WorldSynchronizer | None = field(init=False, default=None)
    """
    Shares the world with other processes; None while perception keeps its world to
    itself.
    """

    tree: BehaviourTree | None = field(init=False, default=None)
    """
    The behaviour tree of :attr:`analysis_engine`; None until :meth:`start`.
    """

    pipeline: Pipeline | None = field(init=False, default=None)
    """
    The pipeline at the root of the tree of :attr:`analysis_engine`, whose finished
    passes are put into the world; None until :meth:`start`.
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
        keeping it in sync in both directions, or else perceive into an empty world of
        its own. If joining fails, :attr:`node` is no longer served.

        .. note:: No other executor may serve :attr:`node`.

        :raises TimeoutError: If the process serving a shared world does not send it
            within :attr:`world_fetch_timeout`.
        """
        self.executor = SingleThreadedExecutor(context=self.node.context)
        self.executor.add_node(self.node)
        self.executor_thread = Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        with ExitStack() as closing_on_failure:
            # A failed start returns no RoboKudo whose close() a caller could call.
            closing_on_failure.callback(self.close)
            service = find_world_fetcher_service(
                self.node, timeout_seconds=self.world_discovery_timeout.total_seconds()
            )
            if service is None:
                logging.getLogger(LOGGING_IDENTIFIER_MAIN_EXECUTABLE).warning(
                    "No process serves a shared world; perceiving into a world of its own."
                )
                self.world = World()
            else:
                self.world = fetch_world_from_service(
                    self.node,
                    timeout_seconds=self.world_fetch_timeout.total_seconds(),
                )
                self.synchronizer = WorldSynchronizer(node=self.node, _world=self.world)
            closing_on_failure.pop_all()

    # %% lifecycle

    def start(
        self, *, include_gui: bool = True, visualizer_types: list[type] | None = None
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
        self.pipeline = engine_tree

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

    def run(self, *, tick_rate: float = 5.0) -> None:
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
                self.tick_tree()
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

    def tick_tree(self) -> None:
        """
        Tick the behaviour tree once, and put the objects found by a pass of
        :attr:`pipeline` that finished successfully in this tick into the world.

        Publishing a world of its own starts once it has a world frame, as
        :attr:`publishes_own_world` asks.
        """
        self.tree.tick()
        if self.pipeline.status != Status.SUCCESS:
            return
        self.update_world_with_perception_beliefs(pipeline=self.pipeline)
        if (
            self.publishes_own_world
            and self.synchronizer is None
            and self.world_publisher is None
            and self.world.root is not None
        ):
            self.start_publishing_own_world()

    def start_publishing_own_world(self) -> None:
        """
        Publish :attr:`world` as markers and as tf frames named under the package name,
        joined to the unprefixed world frame.

        Its frames thus never claim those of the tf tree perception reads the camera
        poses from, which also holds its cameras.
        """
        self.world_publisher = VizMarkerPublisher(
            _world=self.world,
            node=self.node,
            tf_publisher=TFPublisher(
                _world=self.world,
                node=self.node,
                frame_names=TfFrameNames(prefix=f"{PACKAGE_NAME}/"),
            ),
        )

    def update_world_with_perception_beliefs(self, *, pipeline: Pipeline) -> None:
        """
        Form a belief about each object the last pass of ``pipeline`` found and match
        the beliefs to the perceivable objects of :attr:`world`. A matched object is
        updated as :meth:`PerceptionBelief.update_known_object` describes; any other
        belief is added to :attr:`world`. The whole pass is one modification of
        :attr:`world`: no other thread changes :attr:`world` meanwhile, and the changes
        are published once it has ended.

        A pass without a camera pose in the world is left out, with a warning logged.

        :param pipeline: A pipeline whose last pass finished successfully.
        :raises CameraPoseOutsideWorldError: If the camera pose of the pass is measured
            against a body that is not part of :attr:`world`.
        """
        matcher = ObjectMatcher()
        cas = pipeline.cas
        observation = cas.camera_observation
        if (
            observation is None
            or observation.world_T_camera is None
            or observation.world_T_camera.reference_frame is None
        ):
            logging.getLogger(LOGGING_IDENTIFIER_MAIN_EXECUTABLE).warning(
                "Leaving out a pass without a camera pose in the world."
            )
            return
        world_T_camera = observation.world_T_camera
        with self.world.modify_world():
            if (
                world_T_camera.reference_frame
                not in self.world.kinematic_structure_entities
            ):
                raise CameraPoseOutsideWorldError(
                    reference_frame_name=str(world_T_camera.reference_frame.name)
                )
            beliefs = [
                belief
                for hypothesis in cas.filter_annotations_by_type(ObjectHypothesis)
                if (
                    belief := PerceptionBelief.from_hypothesis(
                        hypothesis=hypothesis,
                        world_T_camera=world_T_camera,
                        minimum_class_confidence=self.minimum_class_confidence,
                    )
                )
                is not None
            ]
            matches = matcher.match(beliefs=beliefs, world=self.world)
            for belief in beliefs:
                known_object = matches.get(belief)
                if known_object is None:
                    belief.add_to_world(world=self.world)
                    continue
                belief.update_known_object(known_object=known_object, world=self.world)

    def close(self) -> None:
        """
        Stop sharing and publishing the world and serving :attr:`node`, and make room
        for another RoboKudo in this process.
        """
        if self.world_publisher is not None:
            self.world_publisher.stop()
            self.world_publisher.tf_publisher.stop()
            self.world_publisher = None
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
