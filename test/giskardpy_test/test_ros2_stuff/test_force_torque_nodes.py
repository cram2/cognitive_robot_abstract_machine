"""
Tests of the statechart nodes that watch a force-torque sensor.
"""

import dataclasses
import json
from pathlib import Path

from geometry_msgs.msg import WrenchStamped

from cramph.data_types import ObservationStateValues
from cramph.composites import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from cramph.statechart import Statechart
from giskardpy.motion_statechart.ros2_nodes.force_torque_monitor import (
    ForceImpactMonitor,
)
from giskardpy.motion_statechart.ros2_nodes.topics import (
    PublishOnStart,
    WaitForMessage,
)
from semantic_digital_twin.world import World
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor


def test_a_force_impact_monitor_names_its_threshold_as_a_force():
    """
    The value is the force above which the monitor fires, not a goal error.
    """
    field_names = {field.name for field in dataclasses.fields(ForceImpactMonitor)}
    assert "force_threshold" in field_names
    assert "threshold" not in field_names


def test_force_impact_node(rclpy_node):
    """
    A force impact monitor observes false while the measured force stays below its
    threshold, and true once a larger force arrives.
    """
    topic_name = "force_torque_topic"

    msg_below = WrenchStamped()

    msg_above = WrenchStamped()
    msg_above.wrench.force.x = 20.0

    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(
        node=(
            parallel := Parallel(
                nodes=[
                    ForceImpactMonitor(topic_name=topic_name, force_threshold=10),
                    Sequence(
                        nodes=[
                            PublishOnStart(topic_name=topic_name, msg=msg_below),
                            WaitForMessage(
                                topic_name=topic_name, msg_type=WrenchStamped
                            ),
                            PublishOnStart(topic_name=topic_name, msg=msg_above),
                        ]
                    ),
                ]
            )
        )
    )
    statechart.add_node(node=EndMotion.when_true(node=parallel))

    json_data = statechart.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    executor = StatechartExecutor(
        context=StatechartContext(world=World()),
        extensions=[RosNodeAccess(ros_node=rclpy_node), MotionControl()],
    )
    statechart_copy = Statechart.from_json(data=new_json_data, context=executor.context)
    executor.compile(statechart=statechart_copy)

    ft_node = statechart_copy.nodes[0].nodes[0]

    executor.tick_until_end(timeout=5_000)
    statechart_copy.draw(file_name=Path("muh.pdf"))
    assert (
        statechart_copy.state_history.get_observation_history_of_node(node=ft_node)[0]
        == ObservationStateValues.UNKNOWN
    )
    assert (
        ObservationStateValues.FALSE
        in statechart_copy.state_history.get_observation_history_of_node(node=ft_node)
    )
    assert (
        statechart_copy.state_history.get_observation_history_of_node(node=ft_node)[-1]
        == ObservationStateValues.TRUE
    )
