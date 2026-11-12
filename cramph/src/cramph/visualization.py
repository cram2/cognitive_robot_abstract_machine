"""
Optional world visualization, added to the executor that runs a statechart.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Self

from cramph.exceptions import RvizUnavailableError
from cramph.executor import ExecutorExtension
from semantic_digital_twin.adapters.rerun import RerunAdapter, RerunMode

if TYPE_CHECKING:
    from rclpy.node import Node

    from cramph.executor import Executor
    from semantic_digital_twin.world import World

try:
    from semantic_digital_twin.adapters.ros.tf_publisher import (
        TFPublisher,
        TfFrameNames,
    )
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )
except ImportError:
    TFPublisher = None
    TfFrameNames = None
    VizMarkerPublisher = None


# %% visualizing the world of an executor
@dataclass
class WorldVisualization(ExecutorExtension, ABC):
    """
    Shows the world of the executor it is added to while that executor runs a
    statechart: from the moment the statechart is compiled until the run stops.
    """

    _cleanup: ExitStack = field(default_factory=ExitStack, init=False, repr=False)
    """
    Releases what the running visualization acquired, in reverse acquisition order.
    """

    @property
    @abstractmethod
    def is_rendering(self) -> bool:
        """
        Whether this visualization is started and not yet stopped.
        """

    @abstractmethod
    def _start(self, world: World, cleanup: ExitStack) -> None:
        """
        Acquire what showing `world` needs, registering its release on `cleanup` as soon
        as each part is acquired.

        :param world: The world to show.
        :param cleanup: Releases the acquired parts, also if a later part fails.
        """

    def start(self, world: World) -> Self:
        """
        Start showing `world`, unless this visualization already shows a world.

        A failing start releases what it acquired before it failed.

        :param world: The world to show.
        :return: This visualization.
        """
        if self.is_rendering:
            return self
        with ExitStack() as cleanup:
            self._start(world, cleanup)
            self._cleanup = cleanup.pop_all()
        return self

    def stop(self) -> None:
        """
        Release what this visualization acquired, keeping borrowed resources.
        """
        self._cleanup.close()

    def after_compile(self, executor: Executor) -> None:
        """
        Start showing the world of `executor`.
        """
        self.start(executor.context.world)

    def after_run(self, executor: Executor) -> None:
        """
        Stop showing the world of `executor`.
        """
        self.stop()


# %% native RViz publishing
@dataclass
class RvizVisualization(WorldVisualization):
    """
    Publishes the world as ROS markers for RViz.
    """

    ros_node: Node = field(kw_only=True)
    """
    The ROS node the markers are published through, which stays with its owner.
    """

    collision_visualization: bool = field(default=False, kw_only=True)
    """
    Whether collision results are published as well.
    """

    frame_prefix: str = field(default="", kw_only=True)
    """
    Put in front of every tf frame of the world, so a copy can be shown next to the
    world it copies.

    Empty publishes the frames under their own names.
    """

    marker_topic: str | None = field(default=None, kw_only=True)
    """
    The topic the markers are published on, or ``None`` for the marker publisher's own.
    """

    marker_alpha: float | None = field(default=None, kw_only=True)
    """
    The opacity the markers are drawn with, in [0.0, 1.0], or ``None`` for the marker
    publisher's own.
    """

    publisher: VizMarkerPublisher | None = field(default=None, init=False)
    """
    The marker publisher, while this visualization runs.
    """

    @property
    def is_rendering(self) -> bool:
        return self.publisher is not None

    def _start(self, world: World, cleanup: ExitStack) -> None:
        """
        Start publishing markers through :attr:`ros_node`.

        :raises RvizUnavailableError: If ROS is not installed.
        """
        if VizMarkerPublisher is None:
            raise RvizUnavailableError()
        tf_publisher = self._prefixed_tf_publisher(world, cleanup)
        self.publisher = VizMarkerPublisher(
            _world=world,
            node=self.ros_node,
            topic_name=self.marker_topic or VizMarkerPublisher.topic_name,
            alpha=(
                VizMarkerPublisher.alpha
                if self.marker_alpha is None
                else self.marker_alpha
            ),
            tf_publisher=tf_publisher,
        )
        cleanup.callback(self._stop_publisher, self.publisher)
        if self.collision_visualization:
            self.publisher.with_collision_visualization()

    def _prefixed_tf_publisher(
        self, world: World, cleanup: ExitStack
    ) -> TFPublisher | None:
        """
        :param world: The world whose frames are published.
        :param cleanup: Stops the tf publisher with this visualization.
        :return: A tf publisher of `world` under :attr:`frame_prefix`, or ``None`` to
            let the marker publisher find or start the world's own.
        """
        if not self.frame_prefix:
            return None
        tf_publisher = TFPublisher(
            node=self.ros_node,
            _world=world,
            frame_names=TfFrameNames(prefix=self.frame_prefix),
        )
        cleanup.callback(tf_publisher.stop)
        return tf_publisher

    def _stop_publisher(self, publisher: VizMarkerPublisher) -> None:
        """
        Stop publishing markers.

        :param publisher: The running marker publisher.
        """
        try:
            publisher.stop()
        finally:
            self.publisher = None


# %% native Rerun recording
@dataclass
class RerunVisualization(WorldVisualization):
    """
    Records the world with the native Rerun adapter.
    """

    mode: RerunMode = field(default=RerunMode.SPAWN, kw_only=True)
    """
    Where the recording is sent.
    """

    target: str | None = field(default=None, kw_only=True)
    """
    The Rerun output file or server.
    """

    adapter: RerunAdapter | None = field(default=None, init=False)
    """
    The Rerun adapter, while this visualization runs.
    """

    @property
    def is_rendering(self) -> bool:
        return self.adapter is not None

    def _start(self, world: World, cleanup: ExitStack) -> None:
        self.adapter = RerunAdapter(
            _world=world,
            mode=self.mode,
            target=self.target,
            state_history=True,
        )
        cleanup.callback(self._stop_adapter, self.adapter)

    def _stop_adapter(self, adapter: RerunAdapter) -> None:
        """
        Stop recording.

        :param adapter: The running Rerun adapter.
        """
        try:
            adapter.stop()
        finally:
            self.adapter = None
