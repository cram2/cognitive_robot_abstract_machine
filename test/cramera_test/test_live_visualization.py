"""
Tests of the cramera visualization backend: binding the bridge to a world through the
world's own callbacks and publishing plan execution through plan callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.plans.plan_node import MotionNode

from cramera import paths
from cramera.live import visualization as visualization_module
from cramera.live.bridge import Bridge, MoveRequest, TaskStatusName
from cramera.live.recording import Recording, RecordingState
from cramera.live.visualization import (
    BridgePlanCallback,
    LiveVisualization,
    WorldModelSync,
    WorldStateSync,
)

from .test_live_bridge import (
    PlanWithRoot,
    ReportedStatus,
    make_chart,
    make_plan_node,
    nodes_by_kind,
)

# %% fixtures


@pytest.fixture()
def world() -> World:
    """
    A real world with one free-floating body, so state changes can be triggered.
    """
    world = World()
    root = Body(name=PrefixedName("root", prefix="world"))
    milk = Body(name=PrefixedName("milk.stl", prefix="world"))
    with world.modify_world():
        world.add_body(root)
        world.add_connection(
            Connection6DoF.create_with_dofs(parent=root, child=milk, world=world)
        )
    return world


@dataclass
class ServerRecorder:
    """
    Stands in for the HTTP server: records whether it was shut down.
    """

    shut_down: bool = False

    def shutdown(self):
        self.shut_down = True


# %% world synchronization


class TestWorldSync:
    def test_a_state_change_publishes_a_snapshot(self, world):
        bridge = Bridge()
        bridge.attach(world)
        sync = WorldStateSync(_world=world, bridge=bridge)
        before = bridge.get_state()["sequenceNumber"]

        sync.on_state_change()

        assert bridge.get_state()["sequenceNumber"] == before + 1

    def test_the_sync_registers_with_the_worlds_state_callbacks(self, world):
        sync = WorldStateSync(_world=world, bridge=Bridge())

        assert sync in world.state.state_change_callbacks

    def test_a_state_change_appends_to_an_active_recording(self, world):
        bridge = Bridge()
        bridge.attach(world)
        bridge.recording = Recording()
        bridge.recording.start()
        sync = WorldStateSync(_world=world, bridge=bridge)

        sync.on_state_change()

        assert bridge.recording.frame_count() == 1

    def test_a_recorded_tick_carries_the_action_the_plan_is_performing(self, world):
        """
        The tick's action is what names its stretch of the replay timeline, so it has to
        be captured as the tick is buffered — afterwards the plan has moved on.
        """
        bridge = Bridge()
        bridge.attach(world)
        bridge.recording = Recording()
        bridge.recording.start()
        performing = "TransportAction"
        bridge.running_step = lambda: performing
        sync = WorldStateSync(_world=world, bridge=bridge)

        sync.on_state_change()
        performing = "ParkArmsAction"
        sync.on_state_change()

        assert [frame.step for frame in bridge.recording.stop()] == [
            "TransportAction",
            "ParkArmsAction",
        ]

    def test_a_state_change_without_a_recording_does_not_fail(self, world):
        bridge = Bridge()
        bridge.attach(world)
        sync = WorldStateSync(_world=world, bridge=bridge)

        sync.on_state_change()  # bridge.recording is None — must not raise

    def test_a_model_change_refreshes_the_bundle_signature(self, world):
        bridge = Bridge()
        bridge.attach(world)
        sync = WorldModelSync(_world=world, bridge=bridge)
        before = bridge.bundle_signature()

        with world.modify_world():
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=Body(name=PrefixedName("bench", prefix="laboratory")),
                )
            )
        sync.on_model_change()

        assert bridge.bundle_signature() != before


# %% plan synchronization


class TestBridgePlanCallback:
    def test_a_motion_start_is_pinned_as_running(self):
        bridge = Bridge()
        motion = make_plan_node("MotionNode")
        root = make_plan_node("SequentialNode", children=[motion])
        bridge.begin_plan(PlanWithRoot(root=root))
        callback = BridgePlanCallback(bridge=bridge)

        callback.on_start(_as_motion_node(motion))
        bridge.snapshot_plan()

        assert nodes_by_kind(bridge)["MotionNode"]["status"] == TaskStatusName.RUNNING

    def test_a_motion_end_pins_the_reported_status(self):
        bridge = Bridge()
        motion = make_plan_node("MotionNode")
        root = make_plan_node("SequentialNode", children=[motion])
        bridge.begin_plan(PlanWithRoot(root=root))
        callback = BridgePlanCallback(bridge=bridge)
        motion.status = ReportedStatus(name=TaskStatusName.FAILED)

        callback.on_end(_as_motion_node(motion))
        motion.status = ReportedStatus(name=TaskStatusName.CREATED)

        assert nodes_by_kind(bridge)["MotionNode"]["status"] == TaskStatusName.FAILED

    def test_a_queued_move_is_applied_before_the_tick(self, world):
        """
        A drag queued while the demo was idle reaches the world before the control cycle
        it precedes, so the object stays where the viewer put it instead of being
        published back at its old pose.
        """
        bridge = Bridge()
        bridge.attach(world)
        callback = BridgePlanCallback(bridge=bridge)
        bridge.queue_move(MoveRequest(object_key="milk.stl", position=[1.0, 2.0, 3.0]))

        callback.before_motion_tick(make_chart())

        moved = world.get_body_by_name("milk.stl")
        assert moved.global_pose.to_position().to_np()[:3].tolist() == [1.0, 2.0, 3.0]

    def test_a_motion_tick_publishes_the_statechart(self):
        bridge = Bridge()
        callback = BridgePlanCallback(bridge=bridge)

        callback.on_motion_tick(make_chart())

        assert bridge.get_chart()["nodes"] != []

    def test_a_non_motion_node_republishes_the_plan(self):
        bridge = Bridge()
        action = make_plan_node("ActionNode", status=TaskStatusName.RUNNING)
        bridge._plan = PlanWithRoot(root=action)
        callback = BridgePlanCallback(bridge=bridge)

        callback.on_start(action)

        assert nodes_by_kind(bridge)["ActionNode"]["status"] == TaskStatusName.RUNNING


def _as_motion_node(mimic):
    """
    Give a plan-node mimic the class identity the callback routes motions by.

    The subclass is deliberately named ``MotionNode`` so the serialized kind matches,
    and it overrides the graph-reading ``parent_action_node`` property with the plain
    attribute the mimic carries.

    :param mimic: The node mimic to re-brand.
    """
    mimic.__class__ = type(
        "MotionNode",
        (MotionNode,),
        {
            "__init__": object.__init__,
            # the real properties read the plan graph; the mimic stands alone
            "parent_action_node": None,
            "children": (),
        },
    )
    return mimic


# %% the backend


class TestLiveVisualization:
    def test_start_attaches_and_serves(self, world, monkeypatch):
        bridge = Bridge()
        server = ServerRecorder()
        monkeypatch.setattr(
            visualization_module, "serve", lambda passed_bridge, port: server
        )
        live = LiveVisualization(world=world, bridge=bridge)

        assert live.start() is live

        assert bridge.world is world
        assert bridge.live_server is server
        assert live.state_sync in world.state.state_change_callbacks
        assert bridge.recording.state is RecordingState.RECORDING

    def test_start_replaces_an_earlier_unsaved_recording(self, world, monkeypatch):
        """
        Re-attaching (e.g. a rerun in the same process) starts a fresh capture — a
        single in-flight slot, the same precedent the ``__live__`` throwaway bundle
        already sets for a second attach.
        """
        bridge = Bridge()
        monkeypatch.setattr(
            visualization_module, "serve", lambda passed_bridge, port: ServerRecorder()
        )
        stale = Recording()
        stale.start()
        stale.stop()
        bridge.recording = stale

        LiveVisualization(world=world, bridge=bridge).start()

        assert bridge.recording is not stale
        assert bridge.recording.state is RecordingState.RECORDING

    def test_start_registers_an_atexit_safety_net(self, world, monkeypatch, tmp_path):
        """
        A demo run directly (not through ``cramera-live``) has no long-lived process
        left for the browser to send ``/recording/stop`` to — the interpreter exits the
        moment the script's main body returns.

        The registered callback must finalize whatever the bridge is currently recording
        at that point.
        """
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        bridge = Bridge()
        monkeypatch.setattr(
            visualization_module, "serve", lambda passed_bridge, port: ServerRecorder()
        )
        registered = []
        monkeypatch.setattr(
            visualization_module.atexit,
            "register",
            lambda fn, *args: registered.append((fn, args)),
        )

        LiveVisualization(world=world, bridge=bridge).start()
        bridge.recording.append(bridge.state)

        [(callback, args)] = registered
        callback(*args)

        assert bridge.recording.scene_name == paths.RECORDING_SCENE_NAME
        assert (
            tmp_path / "scenes" / paths.RECORDING_SCENE_NAME / "scene.json"
        ).is_file()

    def test_the_atexit_safety_net_is_silent_with_nothing_to_finalize(
        self, world, monkeypatch
    ):
        bridge = Bridge()  # never attached — bridge.recording is None
        visualization_module._finalize_recording_at_exit(bridge)  # must not raise

    def test_the_atexit_safety_net_swallows_a_bundling_failure(
        self, world, monkeypatch, capsys
    ):
        """
        The interpreter is already tearing down at this point; a bundling failure must
        not surface as an unhandled exception on every ordinary demo exit.
        """
        bridge = Bridge()
        bridge.attach(world)
        bridge.recording = Recording()
        bridge.recording.start()
        bridge.recording.append(bridge.state)
        monkeypatch.setattr(
            visualization_module,
            "finalize_recording",
            lambda *a: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        visualization_module._finalize_recording_at_exit(bridge)  # must not raise

    def test_start_reuses_a_running_server(self, world, monkeypatch):
        bridge = Bridge()
        existing = ServerRecorder()
        bridge.live_server = existing

        def fail(*arguments):
            raise AssertionError("serve() must not be called while a server runs")

        monkeypatch.setattr(visualization_module, "serve", fail)

        LiveVisualization(world=world, bridge=bridge).start()

        assert bridge.live_server is existing

    def test_stop_detaches_the_callbacks_and_the_server(self, world, monkeypatch):
        bridge = Bridge()
        server = ServerRecorder()
        monkeypatch.setattr(
            visualization_module, "serve", lambda passed_bridge, port: server
        )
        live = LiveVisualization(world=world, bridge=bridge).start()

        state_sync = live.state_sync
        live.stop()

        assert server.shut_down is True
        assert bridge.live_server is None
        assert state_sync not in world.state.state_change_callbacks

    def test_plan_callback_publishes_the_plan_tree(self, world, monkeypatch):
        bridge = Bridge()
        monkeypatch.setattr(
            visualization_module, "serve", lambda passed_bridge, port: ServerRecorder()
        )
        live = LiveVisualization(world=world, bridge=bridge).start()
        plan = PlanWithRoot(root=make_plan_node("SequentialNode"))

        callback = live.plan_callback(plan)

        assert isinstance(callback, BridgePlanCallback)
        assert callback.bridge is bridge
        assert nodes_by_kind(bridge)["SequentialNode"] is not None
