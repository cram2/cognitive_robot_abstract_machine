import json

from giskardpy.middleware.ros2.motion_goal import MotionGoal
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountSimulationTimeSeconds,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.adapters.ros.messages import MetaData, StreamPosition

# %% the payload a client sends


def create_client() -> MetaData:
    """
    The identity a client names itself with.
    """
    return MetaData(node_name="client", process_id=3)


def create_motion_statechart() -> MotionStatechart:
    """
    Build a motion statechart that ends after some simulated time.
    """
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(counter := CountSimulationTimeSeconds(seconds=0.5))
    motion_statechart.add_node(EndMotion.when_true(counter))
    return motion_statechart


class TestMotionGoalPayload:
    """
    The goal travels as json, so what the client puts in has to be what Giskard reads
    out.
    """

    def test_the_motion_statechart_survives_the_round_trip(self):
        motion_statechart = create_motion_statechart()
        goal = MotionGoal.for_motion_statechart(
            motion_statechart, client=create_client()
        )

        restored = MotionGoal.from_json(json.loads(json.dumps(goal.to_json())))

        assert restored.motion_statechart_json_data == motion_statechart.to_json()

    def test_a_goal_names_the_client_that_waits_for_it(self):
        """
        Giskard stops a motion whose client is gone, which it can only do for a client
        the goal names.
        """
        client = create_client()
        goal = MotionGoal.for_motion_statechart(
            create_motion_statechart(), client=client
        )

        restored = MotionGoal.from_json(json.loads(json.dumps(goal.to_json())))

        assert restored.client == client

    def test_a_goal_built_on_a_change_names_it(self):
        position = StreamPosition(origin=create_client(), sequence_number=11)
        goal = MotionGoal.for_motion_statechart(
            create_motion_statechart(),
            client=create_client(),
            required_position=position,
        )

        restored = MotionGoal.from_json(json.loads(json.dumps(goal.to_json())))

        assert restored.required_position == position

    def test_a_goal_built_on_nothing_requires_nothing(self):
        goal = MotionGoal.for_motion_statechart(
            create_motion_statechart(), client=create_client()
        )

        restored = MotionGoal.from_json(json.loads(json.dumps(goal.to_json())))

        assert restored.required_position is None
