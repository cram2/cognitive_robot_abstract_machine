"""
Publish native CRAM world and plan state to the browser viewer.

World callbacks publish geometry and poses. Plan callbacks and native motion histories
publish execution progress.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from functools import partial

from typing_extensions import Any, Callable, Optional, TYPE_CHECKING

from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import MotionNode, PlanNode
from coraplex.visualization import PlanVisualization, VisualizationSession
from giskardpy.motion_statechart.motion_statechart import (
    StateHistoryObserver,
)
from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
    StateChangeCallback,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import MeshFileStorage

from cramera.live.bridge import Bridge
from cramera.live.http import DEFAULT_PORT, serve
from cramera.live.recording import Recording
from cramera.live.recording_bundle import finalize_recording
from cramera.live.ros_markers import RosMarkerListener
from cramera.logging_setup import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from coraplex.plans.plan import Plan
    from giskardpy.motion_statechart.motion_statechart import StateHistory

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

    def on_state_change(self, **kwargs: Any) -> None:
        """
        Publish and record the world's updated state.

        :param kwargs: Metadata provided by the native callback dispatcher.
        """
        self.bridge.snapshot()
        if self.bridge.recording is not None:
            self.bridge.recording.append(
                self.bridge.state,
                self.bridge.running_step(),
                self.bridge.executing_statechart(),
            )


@dataclass(eq=False)
class WorldModelSync(ModelChangeCallback):
    """
    Refreshes the bridge's body and geometry catalogs when the world model changes.
    """

    bridge: Bridge = field(kw_only=True)
    """
    The bridge whose catalogs are refreshed.
    """

    def on_model_change(self, **kwargs: Any) -> None:
        """
        Refresh the published model after a structural change.

        :param kwargs: Metadata provided by the native callback dispatcher.
        """
        self.bridge.observe_model_change()


# %% plan synchronization


@dataclass
class BridgePlanCallback(PlanCallback, StateHistoryObserver):
    """
    Publish plan progress and changes recorded by its native motion histories.
    """

    bridge: Bridge = field(kw_only=True)
    """
    The bridge the plan's execution is published to.
    """

    _histories: list[StateHistory] = field(default_factory=list, init=False, repr=False)
    """
    The motion histories subscribed to during this plan's execution.
    """

    def on_start(self, node: PlanNode) -> None:
        """
        Publish execution of the started node.

        :param node: The plan node that started.
        """
        if isinstance(node, MotionNode):
            self.bridge.observe_motion_started(node)
            chart = node.motion_statechart
            if chart is not None:
                if not any(history is chart.history for history in self._histories):
                    chart.history.add_observer(self)
                    self._histories.append(chart.history)
                self.bridge.observe_chart(chart)
        self.bridge.snapshot_plan()

    def on_end(self, node: PlanNode) -> None:
        """
        Publish the completed node's final status.

        :param node: The plan node that completed.
        """
        plan_ended = self.plan is not None and node is self.plan.root
        if isinstance(node, MotionNode):
            self.bridge.observe_motion_ended(node)
            if plan_ended:
                self.bridge.observe_chart(node.motion_statechart)
        else:
            self.bridge.snapshot_plan()
        if (
            not isinstance(node, MotionNode) or plan_ended
        ) and self.bridge.recording is not None:
            self.bridge.recording.update_statechart(self.bridge.executing_statechart())
        if plan_ended:
            self.stop()

    def on_state_change(self, history: StateHistory) -> None:
        """
        Publish the chart and plan after a native history snapshot changes.

        :param history: The subscribed history containing the changed state.
        """
        self.bridge.observe_chart(
            history.history[-1].life_cycle_state.motion_statechart
        )
        self.bridge.snapshot_plan()

    def stop(self) -> None:
        """
        Remove this plan's motion history subscriptions.
        """
        for history in self._histories:
            history.remove_observer(self)
        self._histories.clear()


def _finalize_recording_at_exit(
    bridge: Bridge, recording: Optional[Recording] = None
) -> None:
    """
    Best-effort safety net: write the current recording to disk if the process is about
    to exit without the viewer ever sending ``/recording/stop``.

    A demo run directly (rather than through ``cramera-live``, which stays up for
    inspection after the demo finishes) has no long-lived process left for the browser
    to ask, so the recording would otherwise be lost the moment the script's main body
    returns.

    :param bridge: The bridge whose geometry belongs to the recording.
    :param recording: The session's capture, or the bridge's current capture.
    """
    if recording is None:
        recording = bridge.recording
    if recording is None:
        return
    try:
        finalize_recording(bridge, recording)
    except Exception:
        # boundary guard: the interpreter is tearing down and the world may be in a
        # partial state; losing the recording is better than a traceback on every exit
        logger.exception("could not finalize the live recording at exit")


# %% the backend


@dataclass
class LiveVisualization(PlanVisualization):
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

    bridge: Bridge = field(default_factory=Bridge)
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

    _recording: Optional[Recording] = field(init=False, default=None)
    """
    The capture owned by this visualization session.
    """

    _exit_callback: Optional[Callable[[], None]] = field(init=False, default=None)
    """
    The registered finalizer for this session's capture.
    """

    _plan_callbacks: list[BridgePlanCallback] = field(
        default_factory=list, init=False, repr=False
    )
    """
    The callbacks whose history subscriptions belong to this session.
    """

    def start(self) -> LiveVisualization:
        """
        Attach the bridge to the world and start serving the viewer.

        :return: This visualization.
        """
        if self.state_sync is not None:
            return self
        if self.bridge.recording is not None:
            finalize_recording(self.bridge, self.bridge.recording)
        try:
            self.bridge.attach(self.world)
            self._recording = Recording()
            self.bridge.recording = self._recording
            self._recording.start()
            MeshFileStorage()
            self._exit_callback = partial(
                _finalize_recording_at_exit, self.bridge, self._recording
            )
            atexit.register(self._exit_callback)
            self.bridge.snapshot()
            self.state_sync = WorldStateSync(_world=self.world, bridge=self.bridge)
            self.model_sync = WorldModelSync(_world=self.world, bridge=self.bridge)
            self.marker_listener = RosMarkerListener.start_if_available(self.bridge)
            self.bridge.marker_listener = self.marker_listener
            if self.bridge.live_server is None:
                self.bridge.live_server = serve(self.bridge, self.port)
        except BaseException:
            self.stop()
            raise
        VisualizationSession.register(self.stop)
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
        callback = BridgePlanCallback(bridge=self.bridge, plan=plan)
        self._plan_callbacks.append(callback)
        return callback

    def stop(self) -> None:
        """
        Finalize this session's recording and release its callbacks and server.
        """
        for callback in self._plan_callbacks:
            callback.stop()
        self._plan_callbacks.clear()
        if self._exit_callback is not None:
            atexit.unregister(self._exit_callback)
            self._exit_callback = None
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
            self.bridge.live_server.server_close()
            self.bridge.live_server = None
        if self._recording is not None:
            finalize_recording(self.bridge, self._recording)
            self._recording = None
            self.bridge.recording = None
