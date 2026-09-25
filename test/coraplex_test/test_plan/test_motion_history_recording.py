"""
Native motion execution preserves its final state in a Cramera recording.
"""

from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import MotionNode
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from cramera.live.bridge import Bridge
from cramera.live.chart_observer import ChartObserver
from cramera.live.recording import Recording
from cramera.live.visualization import BridgePlanCallback, WorldStateSync
from giskardpy.motion_statechart.data_types import LifeCycleValues
from semantic_digital_twin.datastructures.definitions import TorsoState


# %% recording native execution
def test_native_motion_recording_retains_completed_chart(immutable_model_world) -> None:
    """
    A real torso motion records its final native chart and releases its observers.
    """
    world, robot, context = immutable_model_world
    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context=context).plan
    bridge = Bridge()
    bridge.attach(world)
    bridge.begin_plan(plan)
    recording = Recording()
    bridge.recording = recording
    recording.start()
    synchronization = WorldStateSync(_world=world, bridge=bridge)
    callback = BridgePlanCallback(plan=plan, bridge=bridge)
    plan.node_callbacks.append(callback)

    try:
        with simulated_robot:
            plan.perform()

        frames = recording.stop()
        motions = [node for node in plan.all_nodes if isinstance(node, MotionNode)]
        assert motions
        assert {motion.status for motion in motions} == {LifeCycleValues.SUCCEEDED}
        assert frames
        chart = motions[-1].motion_statechart
        expected = ChartObserver(title=bridge.chart_state.title).snapshot(chart)
        assert frames[-1].statechart == expected
        assert bridge.chart_state == expected
        assert all(
            motion.motion_statechart.history.observers == [] for motion in motions
        )
    finally:
        callback.stop()
        synchronization.stop()
        plan.node_callbacks[:] = [
            registered
            for registered in plan.node_callbacks
            if registered is not callback
        ]
        recording.stop()
