"""
Tests of the payload a client sends Giskard as a motion goal.
"""

import json

import pytest

from giskardpy.middleware.ros2.motion_goal import (
    DebugOutput,
    MotionGoal,
    MotionGoalPayloadKey,
)
from giskardpy.motion_statechart.graph_node import EndMotion
from cramph.context import StatechartContext
from cramph.monitors import CountSimulationTimeSeconds
from cramph.statechart import Statechart
from krrood_synchronization.messages import MessageOrigin, Sender, StreamPosition
from semantic_digital_twin.world import World

# %% the payload a client sends


def create_client() -> Sender:
    """
    The identity a client names itself with.
    """
    return Sender(node_name="client", process_id=3)


def create_motion_statechart() -> Statechart:
    """
    Build a motion statechart that ends after some simulated time.
    """
    motion_statechart = Statechart(context=StatechartContext(world=World()))
    motion_statechart.add_node(
        node=(counter := CountSimulationTimeSeconds(seconds=0.5))
    )
    motion_statechart.add_node(node=EndMotion.when_true(node=counter))
    return motion_statechart


class TestMotionGoalPayload:
    """
    The goal travels as json, so what the client puts in has to be what Giskard reads
    out.
    """

    def test_the_motion_statechart_survives_the_round_trip(self):
        """
        The motion statechart of a goal arrives as the json it was sent as.
        """
        motion_statechart = create_motion_statechart()
        goal = MotionGoal.from_motion_statechart(
            motion_statechart=motion_statechart, client=create_client()
        )

        restored = MotionGoal.from_json(data=json.loads(json.dumps(goal.to_json())))

        assert restored.motion_statechart_json_data == motion_statechart.to_json()

    def test_a_goal_names_the_client_that_waits_for_it(self):
        """
        Giskard stops a motion whose client is gone, which it can only do for a client
        the goal names.
        """
        client = create_client()
        goal = MotionGoal.from_motion_statechart(
            motion_statechart=create_motion_statechart(), client=client
        )

        restored = MotionGoal.from_json(data=json.loads(json.dumps(goal.to_json())))

        assert restored.client == client

    def test_a_goal_built_on_a_change_names_it(self):
        """
        A goal names the change of the client's world it was built on.
        """
        position = StreamPosition(
            origin=MessageOrigin(sender=create_client()), sequence_number=11
        )
        goal = MotionGoal.from_motion_statechart(
            motion_statechart=create_motion_statechart(),
            client=create_client(),
            required_position=position,
        )

        restored = MotionGoal.from_json(data=json.loads(json.dumps(goal.to_json())))

        assert restored.required_position == position

    def test_a_goal_built_on_nothing_requires_nothing(self):
        """
        A goal built without a change of the world requires none.
        """
        goal = MotionGoal.from_motion_statechart(
            motion_statechart=create_motion_statechart(), client=create_client()
        )

        restored = MotionGoal.from_json(data=json.loads(json.dumps(goal.to_json())))

        assert restored.required_position is None

    @pytest.mark.parametrize(
        "debug_output",
        [
            DebugOutput(publishes_expression_markers=True),
            DebugOutput(plots_trajectory=True),
            DebugOutput(publishes_expression_markers=True, plots_trajectory=True),
        ],
    )
    def test_a_goal_says_what_giskard_produces_to_debug_it(
        self, *, debug_output: DebugOutput
    ):
        """
        The debug output of a goal survives its round trip through json.
        """
        goal = MotionGoal.from_motion_statechart(
            motion_statechart=create_motion_statechart(),
            client=create_client(),
            debug_output=debug_output,
        )

        restored = MotionGoal.from_json(data=json.loads(json.dumps(goal.to_json())))

        assert restored.debug_output == debug_output

    def test_a_goal_asks_for_no_debug_output_unless_told_to(self):
        """
        A goal built without a debug output asks for the default :class:`DebugOutput`.
        """
        goal = MotionGoal.from_motion_statechart(
            motion_statechart=create_motion_statechart(), client=create_client()
        )

        assert goal.debug_output == DebugOutput()

    def test_a_client_that_does_not_send_the_debug_output_gets_the_default(self):
        """
        A payload without :attr:`MotionGoalPayloadKey.DEBUG_OUTPUT` asks for the default
        :class:`DebugOutput`.
        """
        goal = MotionGoal.from_motion_statechart(
            motion_statechart=create_motion_statechart(),
            client=create_client(),
            debug_output=DebugOutput(publishes_expression_markers=True),
        )
        payload = json.loads(json.dumps(goal.to_json()))
        del payload[MotionGoalPayloadKey.DEBUG_OUTPUT]

        restored = MotionGoal.from_json(data=payload)

        assert restored.debug_output == DebugOutput()
