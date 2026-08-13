"""
The cramera backend of :class:`coraplex.visualization.WorldVisualization`.

Binds the live bridge to the world object itself: world state and model changes reach
the viewer through the world's own callbacks, and plan execution through a
:class:`~coraplex.plans.plan_callbacks.PlanCallback`. No parser or executor hooks are
involved — whatever world a demo builds, however it builds it, is what the viewer
shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional, TYPE_CHECKING

from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import MotionNode, PlanNode
from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
    StateChangeCallback,
)
from semantic_digital_twin.world import World

from cramera.live.bridge import BRIDGE, Bridge
from cramera.live.http import DEFAULT_PORT, serve
from cramera.live.ros_markers import RosMarkerListener

if TYPE_CHECKING:
    from coraplex.plans.plan import Plan
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart

# %% world synchronization


@dataclass(eq=False)
class WorldStateSync(StateChangeCallback):
    """
    Publishes a world snapshot to the bridge whenever the world's state changes.
    """

    bridge: Bridge = field(kw_only=True)
    """
    The bridge the snapshots are published to.
    """

    def on_state_change(self, **kwargs) -> None:
        self.bridge.snapshot()


@dataclass(eq=False)
class WorldModelSync(ModelChangeCallback):
    """
    Refreshes the bridge's body and geometry catalogs when the world model changes.
    """

    bridge: Bridge = field(kw_only=True)
    """
    The bridge whose catalogs are refreshed.
    """

    def on_model_change(self, **kwargs) -> None:
        self.bridge.observe_model_change()


# %% plan synchronization


@dataclass
class BridgePlanCallback(PlanCallback):
    """
    Feeds a plan's execution into the live bridge: per-node progress as its nodes start
    and end, and the executing motion statechart on every executor tick.
    """

    bridge: Bridge = field(kw_only=True)
    """
    The bridge the plan's execution is published to.
    """

    def on_start(self, node: PlanNode) -> None:
        if isinstance(node, MotionNode):
            self.bridge.observe_motion_started(node)
            return
        self.bridge.snapshot_plan()

    def on_end(self, node: PlanNode) -> None:
        if isinstance(node, MotionNode):
            self.bridge.observe_motion_ended(node)
            return
        self.bridge.snapshot_plan()

    def on_motion_tick(self, statechart: MotionStatechart) -> None:
        self.bridge.observe_motion_tick(statechart)


# %% the backend


@dataclass
class LiveVisualization:
    """
    Serves a world to the cramera browser viewer while a demo runs.
    """

    world: World
    """
    The world served to the viewer.
    """

    port: int = DEFAULT_PORT
    """
    Port of the bridge's HTTP endpoints.
    """

    bridge: Bridge = field(default_factory=lambda: BRIDGE)
    """
    The bridge translating between the world and the viewer.
    """

    state_sync: Optional[WorldStateSync] = field(init=False, default=None)
    """
    The callback publishing state changes, while started.
    """

    model_sync: Optional[WorldModelSync] = field(init=False, default=None)
    """
    The callback refreshing the catalogs on model changes, while started.
    """

    marker_listener: Optional[RosMarkerListener] = field(init=False, default=None)
    """
    The ROS marker subscription feeding the debug overlay, when ROS is available.
    """

    def start(self) -> LiveVisualization:
        """
        Attach the bridge to the world and start serving the viewer.

        Reuses the already running HTTP server when one exists, so starting a second
        visualization in the same process rebinds the world instead of failing on the
        port.

        :return: This visualization.
        """
        self.bridge.attach(self.world)
        self.bridge.snapshot()
        self.state_sync = WorldStateSync(_world=self.world, bridge=self.bridge)
        self.model_sync = WorldModelSync(_world=self.world, bridge=self.bridge)
        self.marker_listener = RosMarkerListener.start_if_available(self.bridge)
        self.bridge.marker_listener = self.marker_listener
        if self.bridge.live_server is None:
            self.bridge.live_server = serve(self.bridge, self.port)
        return self

    def plan_callback(self, plan: Plan) -> BridgePlanCallback:
        """
        The callback that publishes the plan's execution to the viewer.

        Also publishes the plan's tree immediately, so the viewer shows it before the
        first node runs.

        :param plan: The plan about to be performed.
        :return: The callback to append to the plan's ``node_callbacks``.
        """
        self.bridge.begin_plan(plan)
        return BridgePlanCallback(bridge=self.bridge, plan=plan)

    def stop(self) -> None:
        """
        Detach from the world and stop serving the viewer.
        """
        if self.state_sync is not None:
            self.state_sync.stop()
            self.state_sync = None
        if self.model_sync is not None:
            self.model_sync.stop()
            self.model_sync = None
        if self.marker_listener is not None:
            self.marker_listener.stop()
            self.marker_listener = None
            self.bridge.marker_listener = None
        if self.bridge.live_server is not None:
            self.bridge.live_server.shutdown()
            self.bridge.live_server = None
