"""
A visualization releases whatever it acquired, however its start or stop fails.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from rclpy.node import Node

import cramph.visualization as visualization_module
from cramph.visualization import RvizVisualization
from cramph.exceptions import TickLimitReachedError

from .test_optional_visualization import (
    VisualizationRecordingItsRuns,
    an_ending_statechart,
    an_executor,
)


# %% partial acquisition
def test_failed_publisher_creation_leaves_the_node_usable(
    monkeypatch: pytest.MonkeyPatch, rclpy_node: Node
) -> None:
    monkeypatch.setattr(
        visualization_module, "VizMarkerPublisher", Mock(side_effect=RuntimeError())
    )
    visualization = RvizVisualization(ros_node=rclpy_node)
    executor = an_executor(visualization)
    with pytest.raises(RuntimeError):
        executor.compile(an_ending_statechart(executor.context))
    assert rclpy_node.context.ok()
    assert not visualization.is_rendering


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_collision_setup_failure_stops_the_publisher(
    monkeypatch: pytest.MonkeyPatch,
    rclpy_node: Node,
    failure_type: type[BaseException],
) -> None:
    failure = failure_type()
    publisher = Mock()
    publisher.with_collision_visualization.side_effect = failure
    monkeypatch.setattr(
        visualization_module, "VizMarkerPublisher", Mock(return_value=publisher)
    )
    visualization = RvizVisualization(ros_node=rclpy_node, collision_visualization=True)
    executor = an_executor(visualization)
    with pytest.raises(failure_type) as caught:
        executor.compile(an_ending_statechart(executor.context))
    assert caught.value is failure
    publisher.stop.assert_called_once_with()
    assert not visualization.is_rendering


# %% cleanup failures and reuse
def test_publisher_stop_failure_still_ends_the_visualization(
    monkeypatch: pytest.MonkeyPatch, rclpy_node: Node
) -> None:
    publisher = Mock()
    publisher.stop.side_effect = RuntimeError()
    monkeypatch.setattr(
        visualization_module, "VizMarkerPublisher", Mock(return_value=publisher)
    )
    visualization = RvizVisualization(ros_node=rclpy_node)
    executor = an_executor(visualization)
    executor.compile(an_ending_statechart(executor.context))
    with pytest.raises(RuntimeError):
        visualization.stop()
    assert not visualization.is_rendering
    visualization.stop()
    publisher.stop.assert_called_once_with()


def test_a_failed_run_still_stops_the_visualization() -> None:
    visualization = VisualizationRecordingItsRuns()
    executor = an_executor(visualization)
    statechart = an_ending_statechart(executor.context)
    executor.compile(statechart)

    with pytest.raises(TickLimitReachedError):
        executor.tick_until_end(timeout=0)

    assert visualization.stops == 1
    assert not visualization.is_rendering
