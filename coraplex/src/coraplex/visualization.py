"""
Optional world visualization selected without changing a robot plan.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import ExitStack
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from functools import partial
from importlib.metadata import entry_points
from types import TracebackType

from typing_extensions import TYPE_CHECKING, ClassVar, Self

from coraplex.datastructures.enums import VisualizationBackend, VisualizationOption
from coraplex.exceptions import (
    UnknownVisualizationOption,
    VisualizationBackendUnavailable,
)
from coraplex.plans.plan_node import PlanNode
from semantic_digital_twin.adapters.rerun import RerunAdapter, RerunMode

if TYPE_CHECKING:
    from rclpy.node import Node

    from coraplex.plans.plan import Plan
    from coraplex.plans.plan_callbacks import PlanCallback
    from semantic_digital_twin.world import World

try:
    import rclpy
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )
except ImportError:
    rclpy = None
    VizMarkerPublisher = None


# %% provider contract
@dataclass
class PlanVisualization(ABC):
    """
    A visualization provider observing a world and its executed plans.
    """

    world: World
    """
    The world presented by this provider.
    """

    @abstractmethod
    def start(self) -> Self:
        """
        Start serving this world and return the provider.
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Stop serving and release resources owned by this provider.
        """

    @abstractmethod
    def plan_callback(self, plan: Plan) -> PlanCallback:
        """
        Create an execution observer for a plan.

        :param plan: The plan to observe.
        :return: A callback registered by the visualization owner.
        """


# %% visualization owner
@dataclass
class VisualizationSession:
    """
    Close visualizations acquired in a context when execution leaves that context.
    """

    _current: ClassVar[ContextVar[VisualizationSession | None]] = ContextVar(
        "visualization_session", default=None
    )
    """
    The cleanup scope active in the current execution context.
    """

    _cleanup: ExitStack = field(default_factory=ExitStack, init=False)
    """
    Resource cleanup callbacks in reverse acquisition order.
    """

    _token: Token[VisualizationSession | None] = field(init=False)
    """
    The previous scope restored when this context exits.
    """

    def __enter__(self) -> Self:
        """
        Make this session the owner of subsequently started visualizations.
        """
        self._token = self._current.set(self)
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Close acquired resources and restore the enclosing session.

        :param exception_type: The exception type raised inside the scope, if any.
        :param exception: The original exception propagated after cleanup.
        :param traceback: The traceback associated with that exception.
        """
        try:
            self._cleanup.close()
        finally:
            self._current.reset(self._token)

    @classmethod
    def is_active(cls) -> bool:
        """
        Return whether the current context owns visualization cleanup.
        """
        return cls._current.get() is not None

    @classmethod
    def register(cls, cleanup: Callable[[], None]) -> None:
        """
        Register cleanup in the active session, if one exists.

        :param cleanup: Release an acquired resource without requiring its caller to
            retain it.
        """
        current = cls._current.get()
        if current is not None:
            current._cleanup.callback(cleanup)


@dataclass
class WorldVisualization(ABC):
    """
    Own a selected renderer and the execution observers attached to it.
    """

    world: World
    """
    The observed world.
    """

    backend: ClassVar[VisualizationBackend]
    """
    The explicitly selected renderer.
    """

    _cleanup: ExitStack = field(default_factory=ExitStack, init=False, repr=False)
    """
    Resource cleanup callbacks in reverse acquisition order.
    """

    @classmethod
    def from_environment(
        cls,
        world: World,
        default_backend: VisualizationBackend = VisualizationBackend.NONE,
        *,
        ros_node: Node | None = None,
        collision_visualization: bool = False,
    ) -> WorldVisualization:
        """
        Read optional renderer settings while preserving the supplied default.

        :param world: The world to visualize.
        :param default_backend: Renderer used without an explicit setting.
        :param ros_node: An existing ROS node available for RViz publishing.
        :param collision_visualization: Whether to publish RViz collision results.
        """
        backend = (
            os.environ.get(VisualizationOption.BACKEND, default_backend.value)
            .strip()
            .lower()
        )
        if backend not in {member.value for member in VisualizationBackend}:
            raise UnknownVisualizationOption(VisualizationOption.BACKEND, backend)
        constructors = {
            VisualizationBackend.NONE: partial(HeadlessVisualization, world),
            VisualizationBackend.RVIZ: partial(
                RvizVisualization,
                world,
                ros_node=ros_node,
                collision_visualization=collision_visualization,
            ),
            VisualizationBackend.RERUN: partial(RerunVisualization, world),
            VisualizationBackend.CRAMERA: partial(PluginVisualization, world),
        }
        visualization = constructors[VisualizationBackend(backend)]()
        visualization._configure_from_environment()
        return visualization

    def _configure_from_environment(self) -> None:
        pass

    @property
    @abstractmethod
    def is_rendering(self) -> bool:
        """:return: Whether this owner has a started renderer."""

    @abstractmethod
    def _start(self, cleanup: ExitStack) -> None:
        pass

    def start(self) -> Self:
        """
        Start the selected renderer once.

        :return: This visualization owner.
        """
        if self.is_rendering:
            return self
        with ExitStack() as cleanup:
            self._start(cleanup)
            self._cleanup = cleanup.pop_all()
        if self.is_rendering:
            VisualizationSession.register(self.stop)
        return self

    def stop(self) -> None:
        """
        Remove owned observers and renderers, retaining borrowed ROS resources.
        """
        self._cleanup.close()

    def attach_plan(self, plan: Plan | PlanNode) -> None:
        pass

    def finish_execution(self) -> None:
        self.stop()


# %% headless execution
@dataclass
class HeadlessVisualization(WorldVisualization):
    backend: ClassVar[VisualizationBackend] = VisualizationBackend.NONE
    """
    Headless execution without a renderer.
    """

    @property
    def is_rendering(self) -> bool:
        return False

    def _start(self, cleanup: ExitStack) -> None:
        pass


# %% native RViz publishing
@dataclass
class RvizVisualization(WorldVisualization):
    backend: ClassVar[VisualizationBackend] = VisualizationBackend.RVIZ
    """
    Native ROS marker publishing.
    """

    ros_node: Node | None = field(default=None, kw_only=True)
    """
    A borrowed ROS node, or a node created for an RViz renderer.
    """

    collision_visualization: bool = field(default=False, kw_only=True)
    """
    Whether the RViz renderer also publishes native collision results.
    """

    publisher: VizMarkerPublisher | None = field(default=None, init=False)
    """
    The owned RViz marker publisher.
    """

    @property
    def is_rendering(self) -> bool:
        return self.publisher is not None

    def _start(self, cleanup: ExitStack) -> None:
        """
        Start native marker publishing, borrowing an existing ROS node if supplied.
        """
        if VizMarkerPublisher is None:
            raise VisualizationBackendUnavailable(self.backend)
        if self.ros_node is None:
            if not rclpy.ok():
                rclpy.init()
                cleanup.callback(self._shutdown_context)
            self.ros_node = rclpy.create_node("coraplex_visualization")
            cleanup.callback(self._destroy_node, self.ros_node)
        self.publisher = VizMarkerPublisher(_world=self.world, node=self.ros_node)
        cleanup.callback(self._stop_publisher, self.publisher)
        if self.collision_visualization:
            self.publisher.with_collision_visualization()

    def _shutdown_context(self) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def _destroy_node(self, node: Node) -> None:
        try:
            node.destroy_node()
        finally:
            self.ros_node = None

    def _stop_publisher(self, publisher: VizMarkerPublisher) -> None:
        try:
            publisher.stop()
        finally:
            self.publisher = None


# %% native Rerun recording
@dataclass
class RerunVisualization(WorldVisualization):
    backend: ClassVar[VisualizationBackend] = VisualizationBackend.RERUN
    """
    Native Rerun recording.
    """

    mode: RerunMode = field(default=RerunMode.SPAWN, kw_only=True)
    """
    Where the native Rerun adapter sends its recording.
    """

    target: str | None = field(default=None, kw_only=True)
    """
    The Rerun output file or server.
    """

    adapter: RerunAdapter | None = field(default=None, init=False)
    """
    The owned native Rerun adapter.
    """

    def _configure_from_environment(self) -> None:
        mode = (
            os.environ.get(VisualizationOption.RERUN_MODE, RerunMode.SPAWN.value)
            .strip()
            .lower()
        )
        if mode not in {member.value for member in RerunMode}:
            raise UnknownVisualizationOption(VisualizationOption.RERUN_MODE, mode)
        self.mode = RerunMode(mode)
        self.target = os.environ.get(VisualizationOption.RERUN_TARGET)

    @property
    def is_rendering(self) -> bool:
        return self.adapter is not None

    def _start(self, cleanup: ExitStack) -> None:
        self.adapter = RerunAdapter(
            _world=self.world,
            mode=self.mode,
            target=self.target,
            state_history=True,
        )
        cleanup.callback(self._stop_adapter, self.adapter)

    def _stop_adapter(self, adapter: RerunAdapter) -> None:
        try:
            adapter.stop()
        finally:
            self.adapter = None


# %% installed plan visualization
@dataclass
class PluginVisualization(WorldVisualization):
    backend: ClassVar[VisualizationBackend] = VisualizationBackend.CRAMERA
    """
    The installed browser visualization.
    """

    provider: PlanVisualization | None = field(default=None, init=False)
    """
    The optional installed browser visualization provider.
    """

    _callbacks: list[PlanCallback] = field(default_factory=list, init=False)
    """
    Plan callbacks registered by this owner.
    """

    @property
    def is_rendering(self) -> bool:
        return self.provider is not None

    def _start(self, cleanup: ExitStack) -> None:
        providers = entry_points(
            group=VisualizationOption.PROVIDER_GROUP, name=self.backend.value
        )
        if len(providers) != 1:
            raise VisualizationBackendUnavailable(self.backend)
        provider_type = next(iter(providers)).load()
        if not isinstance(provider_type, type) or not issubclass(
            provider_type, PlanVisualization
        ):
            raise VisualizationBackendUnavailable(self.backend)
        self.provider = provider_type(world=self.world)
        cleanup.callback(self._stop_provider, self.provider)
        cleanup.callback(self._remove_callbacks)
        self.provider.start()

    def attach_plan(self, plan: Plan | PlanNode) -> None:
        """
        Observe a plan through the running optional provider.

        :param plan: A plan or its root node.
        """
        if self.provider is None:
            return
        observed_plan = plan.plan if isinstance(plan, PlanNode) else plan
        if any(callback.plan is observed_plan for callback in self._callbacks):
            return
        callback = self.provider.plan_callback(observed_plan)
        observed_plan.node_callbacks.append(callback)
        self._callbacks.append(callback)

    def _remove_callbacks(self) -> None:
        for callback in self._callbacks:
            callback.plan.node_callbacks[:] = [
                registered
                for registered in callback.plan.node_callbacks
                if registered is not callback
            ]
        self._callbacks.clear()

    def _stop_provider(self, provider: PlanVisualization) -> None:
        try:
            provider.stop()
        finally:
            self.provider = None

    def finish_execution(self) -> None:
        pass
