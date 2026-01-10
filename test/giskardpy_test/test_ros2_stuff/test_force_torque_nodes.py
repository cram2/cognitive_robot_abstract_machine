import json

from geometry_msgs.msg import WrenchStamped

from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.ros2_nodes.force_torque_monitor import (
    ForceImpactMonitor,
)
from giskardpy.motion_statechart.ros2_nodes.topic_monitor import (
    PublishOnStart,
    WaitForMessage,
)
from giskardpy.ros_executor import Ros2Executor
from semantic_digital_twin.world import World


def test_force_impact_node(rclpy_node):
    topic_name = "force_torque_topic"

    msg_below = WrenchStamped()

    msg_above = WrenchStamped()
    msg_above.wrench.force.x = 20.0

    msc = MotionStatechart()
    msc.add_node(
        parallel := Parallel(
            [
                ForceImpactMonitor(topic_name=topic_name, threshold=10),
                Sequence(
                    nodes=[
                        PublishOnStart(topic_name=topic_name, msg=msg_below),
                        WaitForMessage(topic_name=topic_name, msg_type=WrenchStamped),
                        PublishOnStart(topic_name=topic_name, msg=msg_above),
                    ]
                ),
            ]
        )
    )
    msc.add_node(EndMotion.when_true(parallel))

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    msc_copy = MotionStatechart.from_json(new_json_data)

    kin_sim = Ros2Executor(world=World(), ros_node=rclpy_node)
    kin_sim.compile(motion_statechart=msc_copy)

    ft_node = msc_copy.nodes[0].nodes[0]

    kin_sim.tick_until_end(timeout=5_000)
    msc_copy.draw("muh.pdf")
    assert (
        msc_copy.history.get_observation_history_of_node(ft_node)[0]
        == ObservationStateValues.UNKNOWN
    )
    assert (
        ObservationStateValues.FALSE
        in msc_copy.history.get_observation_history_of_node(ft_node)
    )
    assert (
        msc_copy.history.get_observation_history_of_node(ft_node)[-1]
        == ObservationStateValues.TRUE
    )
