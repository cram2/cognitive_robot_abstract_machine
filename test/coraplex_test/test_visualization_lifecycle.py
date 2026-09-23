from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import rclpy
from rclpy.node import Node
from typing_extensions import Self

import coraplex.visualization as visualization_module
from coraplex.datastructures.enums import VisualizationBackend, VisualizationOption
from coraplex.exceptions import VisualizationBackendUnavailable
from coraplex.plans.factories import sequential
from coraplex.visualization import WorldVisualization
from semantic_digital_twin.world import World

from .test_optional_visualization import ObservedScene, installed_scene


# %% partial acquisition
@dataclass
class FailsDuringStart(ObservedScene):
    def start(self) -> Self:
        self.started = True
        raise RuntimeError()


def test_failed_publisher_creation_releases_owned_ros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(VisualizationOption.BACKEND, VisualizationBackend.RVIZ)
    monkeypatch.setattr(
        visualization_module, "VizMarkerPublisher", Mock(side_effect=RuntimeError)
    )
    selected = WorldVisualization.from_environment(World())
    assert not rclpy.ok()
    try:
        with pytest.raises(RuntimeError):
            selected.start()
        assert not rclpy.ok()
        assert selected.ros_node is None
        assert not selected.is_rendering
    finally:
        selected.stop()


def test_failed_node_creation_releases_owned_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(VisualizationOption.BACKEND, VisualizationBackend.RVIZ)
    monkeypatch.setattr(rclpy, "create_node", Mock(side_effect=RuntimeError))
    selected = WorldVisualization.from_environment(World())
    assert not rclpy.ok()
    try:
        with pytest.raises(RuntimeError):
            selected.start()
        assert not rclpy.ok()
    finally:
        selected.stop()


def test_failed_plugin_start_stops_partially_started_provider(
    monkeypatch: pytest.MonkeyPatch,
    installed_scene: Mock,
) -> None:
    monkeypatch.setenv(VisualizationOption.BACKEND, VisualizationBackend.CRAMERA)
    scene = FailsDuringStart(World())
    installed_scene.return_value[0].load.return_value = FailsDuringStart
    monkeypatch.setattr(
        FailsDuringStart, "__new__", staticmethod(lambda cls, **arguments: scene)
    )
    selected = WorldVisualization.from_environment(scene.world)
    with pytest.raises(RuntimeError):
        selected.start()
    assert scene.started
    assert scene.stopped
    assert not selected.is_rendering


# %% cleanup failures and reuse
@pytest.mark.parametrize("failure", [RuntimeError(), KeyboardInterrupt()])
def test_collision_setup_failure_closes_publisher_and_owned_ros(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    publisher = Mock()
    publisher.with_collision_visualization.side_effect = failure
    monkeypatch.setattr(
        visualization_module, "VizMarkerPublisher", Mock(return_value=publisher)
    )
    selected = visualization_module.RvizVisualization(
        World(), collision_visualization=True
    )
    assert not rclpy.ok()
    with pytest.raises(type(failure)) as caught:
        selected.start()
    assert caught.value is failure
    publisher.stop.assert_called_once_with()
    assert selected.ros_node is None
    assert not selected.is_rendering
    assert not rclpy.ok()


def test_publisher_stop_failure_still_releases_owned_ros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = Mock()
    publisher.stop.side_effect = RuntimeError()
    monkeypatch.setattr(
        visualization_module, "VizMarkerPublisher", Mock(return_value=publisher)
    )
    selected = visualization_module.RvizVisualization(World()).start()
    with pytest.raises(RuntimeError):
        selected.stop()
    assert selected.ros_node is None
    assert not selected.is_rendering
    assert not rclpy.ok()
    selected.stop()
    publisher.stop.assert_called_once_with()


def test_failed_publisher_start_preserves_borrowed_node(
    monkeypatch: pytest.MonkeyPatch,
    rclpy_node: Node,
) -> None:
    monkeypatch.setattr(
        visualization_module, "VizMarkerPublisher", Mock(side_effect=RuntimeError())
    )
    selected = visualization_module.RvizVisualization(World(), ros_node=rclpy_node)
    with pytest.raises(RuntimeError):
        selected.start()
    assert selected.ros_node is rclpy_node
    assert rclpy_node.context.ok()
    assert not selected.is_rendering


def test_plugin_stop_failure_detaches_observers_and_allows_restart(
    monkeypatch: pytest.MonkeyPatch,
    installed_scene: Mock,
) -> None:
    selected = visualization_module.PluginVisualization(World()).start()
    provider = selected.provider
    failure = Mock(side_effect=RuntimeError())
    monkeypatch.setattr(provider, "stop", failure)
    plan = sequential([]).plan
    selected.attach_plan(plan)
    with pytest.raises(RuntimeError):
        selected.stop()
    assert plan.node_callbacks == []
    assert not selected.is_rendering
    selected.start()
    try:
        assert selected.provider is not provider
        selected.attach_plan(plan)
        assert selected.provider.plans == [plan]
    finally:
        selected.stop()
    failure.assert_called_once_with()
    assert plan.node_callbacks == []


def test_plugin_entry_point_must_load_a_provider_class(installed_scene: Mock) -> None:
    installed_scene.return_value[0].load.return_value = None
    with pytest.raises(VisualizationBackendUnavailable) as caught:
        visualization_module.PluginVisualization(World()).start()
    assert caught.value.backend is VisualizationBackend.CRAMERA


def test_stopped_plugin_does_not_attach_observers(installed_scene: Mock) -> None:
    selected = visualization_module.PluginVisualization(World())
    plan = sequential([]).plan
    selected.attach_plan(plan)
    assert plan.node_callbacks == []
    installed_scene.assert_not_called()
