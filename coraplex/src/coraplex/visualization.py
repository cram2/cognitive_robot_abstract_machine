from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum

import rerun
from typing_extensions import TYPE_CHECKING, Optional, Self, Type, TypeVar

from coraplex.datastructures.enums import TaskStatus, VisualizationBackend
from coraplex.exceptions import (
    UnknownVisualizationOption,
    VisualizationBackendUnavailable,
)
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import DesignatorNode, MotionNode, PlanNode
from semantic_digital_twin.adapters.rerun import RerunAdapter, RerunMode
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    import rclpy.node

    from coraplex.plans.plan import Plan
    from cramera.live.visualization import LiveVisualization

logger = logging.getLogger(__name__)

try:
    import rclpy
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )
except ImportError:
    rclpy = None
    VizMarkerPublisher = None
    logger.info(
        "Could not import VizMarkerPublisher. This is probably because you are not running ROS."
    )


# %% environment configuration

VISUALIZATION_BACKEND_VARIABLE = "CORAPLEX_VISUALIZATION"
"""
Environment variable naming the :class:`VisualizationBackend` member to use.
"""

RERUN_MODE_VARIABLE = "CORAPLEX_RERUN_MODE"
"""
Environment variable naming the :class:`RerunMode` member to use.
"""

RERUN_TARGET_VARIABLE = "CORAPLEX_RERUN_TARGET"
"""
Environment variable holding the Rerun gRPC URL or ``.rrd`` file path.
"""

EnumType = TypeVar("EnumType", bound=Enum)


def _enum_from_environment(
    variable: str, enum_type: Type[EnumType], default: EnumType
) -> EnumType:
    """
    Read an enum member from an environment variable by its case-insensitive name.

    :param variable: The environment variable to read.
    :param enum_type: The enum the value must name a member of.
    :param default: The member returned when the variable is unset.
    :raises UnknownVisualizationOption: When the value names no member.
    """
    value = os.environ.get(variable)
    if value is None:
        return default
    member_name = value.strip().upper()
    if member_name not in enum_type.__members__:
        raise UnknownVisualizationOption(
            environment_variable=variable,
            value=value,
            valid_values=[name.lower() for name in enum_type.__members__],
        )
    return enum_type[member_name]


# %% plan events on the Rerun timeline


@dataclass
class RerunPlanCallback(PlanCallback):
    """
    Logs plan node starts and ends as text entries on the adapter's timeline, so
    scrubbing the recording shows what the robot was doing alongside its motion.
    """

    adapter: RerunAdapter = field(kw_only=True)
    """
    The adapter whose recording and timeline the entries are logged to.
    """

    def on_start(self, node: PlanNode):
        self._log(node, "started", rerun.TextLogLevel.INFO)

    def on_end(self, node: PlanNode):
        level = (
            rerun.TextLogLevel.ERROR
            if node.status == TaskStatus.FAILED
            else rerun.TextLogLevel.INFO
        )
        self._log(node, "ended", level)
        # Pin the exact poses at the action boundary despite the state log stride.
        self.adapter.log_current_state()

    @staticmethod
    def _node_label(node: PlanNode) -> str:
        """
        A short label for a node: the designator's class name, nested under its owning
        action for motions.
        """
        if isinstance(node, MotionNode) and node.parent_action_node is not None:
            return f"{node.parent_action_node!r}/{node!r}"
        if isinstance(node, DesignatorNode):
            return repr(node)
        return type(node).__name__

    def _log(self, node: PlanNode, event: str, level: rerun.TextLogLevel) -> None:
        """
        Log one text entry for a node at the world's current state version.
        """
        label = self._node_label(node)
        rerun.set_time(
            self.adapter.timeline,
            sequence=node.plan.world.state.version,
            recording=self.adapter.recording,
        )
        rerun.log(
            f"{self.adapter.event_log_entity_path}/{label}",
            rerun.TextLog(f"{label} {event}", level=level),
            recording=self.adapter.recording,
        )


# %% canonical entry point


@dataclass
class WorldVisualization:
    """
    The canonical way to watch a world in 3D while a plan runs.

    Select a backend, :meth:`start` it, and optionally :meth:`attach_plan` so executed
    actions and motions appear on the same timeline as the robot's motion.
    """

    world: World
    """
    The world to visualize.
    """

    backend: VisualizationBackend = VisualizationBackend.NONE
    """
    The renderer to use.
    """

    rerun_mode: RerunMode = field(default=RerunMode.SPAWN, kw_only=True)
    """
    Where the Rerun recording streams to.
    """

    rerun_target: Optional[str] = field(default=None, kw_only=True)
    """
    GRPC URL or ``.rrd`` file path for the ``CONNECT`` and ``SAVE`` Rerun modes.
    """

    state_log_stride: int = field(default=5, kw_only=True)
    """
    Log every N-th world state version to Rerun; action boundaries are always logged.
    """

    rerun_adapter: Optional[RerunAdapter] = field(init=False, default=None)
    """
    The adapter streaming the world to Rerun, when the backend is ``RERUN``.
    """

    ros_node: Optional[rclpy.node.Node] = field(init=False, default=None)
    """
    The node publishing markers and TF, when the backend is ``RVIZ``.
    """

    cramera_visualization: Optional[LiveVisualization] = field(init=False, default=None)
    """
    The live bridge serving the world to the browser, when the backend is ``CRAMERA``.
    """

    @classmethod
    def from_environment(
        cls,
        world: World,
        default_backend: VisualizationBackend = VisualizationBackend.NONE,
    ) -> Self:
        """
        Build a visualization configured by the ``CORAPLEX_*`` environment variables.

        :param world: The world to visualize.
        :param default_backend: The backend used when the environment does not select
            one.
        """
        return cls(
            world=world,
            backend=_enum_from_environment(
                VISUALIZATION_BACKEND_VARIABLE, VisualizationBackend, default_backend
            ),
            rerun_mode=_enum_from_environment(
                RERUN_MODE_VARIABLE, RerunMode, RerunMode.SPAWN
            ),
            rerun_target=os.environ.get(RERUN_TARGET_VARIABLE),
        )

    def start(self) -> Self:
        """
        Start the selected backend and return this visualization.

        :raises VisualizationBackendUnavailable: When the selected backend cannot run in
            this environment.
        """
        match self.backend:
            case VisualizationBackend.NONE:
                pass
            case VisualizationBackend.RERUN:
                self._start_rerun()
            case VisualizationBackend.RVIZ:
                self._start_rviz()
            case VisualizationBackend.CRAMERA:
                self._start_cramera()
        return self

    def _start_rerun(self) -> None:
        self.rerun_adapter = RerunAdapter(
            _world=self.world,
            application_id="coraplex",
            mode=self.rerun_mode,
            target=self.rerun_target,
            state_history=True,
            state_log_stride=self.state_log_stride,
        )

    def _start_rviz(self) -> None:
        if VizMarkerPublisher is None:
            raise VisualizationBackendUnavailable(
                backend=self.backend,
                reason="rclpy is not importable; RViz needs a ROS 2 environment",
            )
        rclpy.init()
        self.ros_node = rclpy.create_node("viz_marker")
        VizMarkerPublisher(_world=self.world, node=self.ros_node).with_tf_publisher()

    def _start_cramera(self) -> None:
        try:
            from cramera.live.visualization import LiveVisualization
        except ImportError:
            raise VisualizationBackendUnavailable(
                backend=self.backend,
                reason="cramera is not importable; install the cramera package",
            )
        self.cramera_visualization = LiveVisualization(world=self.world).start()

    def attach_plan(self, plan: Plan) -> None:
        """
        Publish the plan's execution to the running backend.

        On Rerun the plan's node starts and ends appear as text entries on the
        recording's timeline; on cramera the plan tree, its per-node progress and the
        executing motion statechart appear in the viewer's panels. Does nothing for
        backends without a plan display.
        """
        if self.rerun_adapter is not None:
            plan.node_callbacks.append(
                RerunPlanCallback(adapter=self.rerun_adapter, plan=plan)
            )
        if self.cramera_visualization is not None:
            plan.node_callbacks.append(self.cramera_visualization.plan_callback(plan))

    def stop(self) -> None:
        """
        Detach the backend from the world and release its resources.
        """
        if self.rerun_adapter is not None:
            self.rerun_adapter.stop()
            self.rerun_adapter = None
        if self.ros_node is not None:
            self.ros_node.destroy_node()
        if self.cramera_visualization is not None:
            self.cramera_visualization.stop()
            self.cramera_visualization = None
            self.ros_node = None
