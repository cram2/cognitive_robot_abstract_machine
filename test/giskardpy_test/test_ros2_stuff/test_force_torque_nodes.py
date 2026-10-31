import json

from geometry_msgs.msg import WrenchStamped

from cramph.data_types import ObservationStateValues
from cramph.composites import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from cramph.statechart import Statechart
from giskardpy.motion_statechart.ros2_nodes.force_torque_monitor import (
    ForceImpactMonitor,
)
from giskardpy.motion_statechart.ros2_nodes.topic_monitor import (
    PublishOnStart,
    WaitForMessage,
)
from semantic_digital_twin.world import World
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor


def test_force_impact_node(rclpy_node):
    topic_name = "force_torque_topic"

    msg_below = WrenchStamped()

    msg_above = WrenchStamped()
    msg_above.wrench.force.x = 20.0

    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(
        parallel := Parallel(
            nodes=[
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
    statechart.add_node(EndMotion.when_true(parallel))

    json_data = statechart.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    executor = StatechartExecutor(
        context=StatechartContext(world=World()),
        extensions=[RosNodeAccess(rclpy_node), MotionControl()],
    )
    statechart_copy = Statechart.from_json(new_json_data, context=executor.context)
    executor.compile(statechart=statechart_copy)

    ft_node = statechart_copy.nodes[0].nodes[0]

    executor.tick_until_end(timeout=5_000)
    statechart_copy.draw("muh.pdf")
    assert (
        statechart_copy.state_history.get_observation_history_of_node(ft_node)[0]
        == ObservationStateValues.UNKNOWN
    )
    assert (
        ObservationStateValues.FALSE
        in statechart_copy.state_history.get_observation_history_of_node(ft_node)
    )
    assert (
        statechart_copy.state_history.get_observation_history_of_node(ft_node)[-1]
        == ObservationStateValues.TRUE
    )
