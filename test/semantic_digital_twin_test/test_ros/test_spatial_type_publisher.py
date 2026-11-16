"""
Tests publishing spatial types as RViz markers.
"""

from dataclasses import dataclass, field

import numpy as np
import pytest
from visualization_msgs.msg import Marker, MarkerArray

from krrood.symbolic_math.symbolic_math import FloatVariable, SymbolicMathType
from semantic_digital_twin.adapters.ros.visualization import (
    spatial_type_marker_renderer,
)
from semantic_digital_twin.adapters.ros.visualization.spatial_type_marker_renderer import (
    SpatialTypeMarkers,
    SpatialTypeVisualization,
)
from semantic_digital_twin.adapters.ros.visualization.spatial_type_publisher import (
    CompiledSpatialTypeVisualizations,
    EvaluatedSpatialTypeVisualization,
    EvaluatedSpatialTypePublisher,
    SpatialTypePublisher,
)
from krrood_spatial_math.spatial_types import Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom


@dataclass
class FakeMarkerPublisher:
    """
    Captures published marker arrays so tests can assert without a live RViz.
    """

    published: list[MarkerArray] = field(default_factory=list)

    def publish(self, marker_array: MarkerArray) -> None:
        self.published.append(marker_array)


def capture(publisher: SpatialTypePublisher) -> FakeMarkerPublisher:
    """
    Replace the ROS publisher with one that records the published marker arrays.
    """
    fake = FakeMarkerPublisher()
    publisher.marker_publisher.publisher = fake
    return fake


def first_moving_dof(world: World) -> DegreeOfFreedom:
    return world.active_degrees_of_freedom[0]


# %% publishing evaluated spatial types


def test_marker_publisher_draws_markers_from_the_given_values(rclpy_node):
    """
    The marker publisher draws the given values and evaluates nothing.
    """
    marker_publisher = EvaluatedSpatialTypePublisher(node=rclpy_node)
    fake = FakeMarkerPublisher()
    marker_publisher.publisher = fake
    unresolvable = FloatVariable(name="unresolvable")

    marker_publisher.publish(
        evaluated_visualizations=[
            EvaluatedSpatialTypeVisualization(
                spatial_type_markers=SpatialTypeMarkers.from_visualization(
                    visualization=SpatialTypeVisualization(
                        spatial_type=Point3(x=unresolvable)
                    ),
                    root_frame_name="map",
                ),
                rendered_value=np.array([4.0, 5.0, 6.0]),
            )
        ]
    )

    position = fake.published[-1].markers[0].pose.position
    assert (position.x, position.y, position.z) == (4, 5, 6)


def test_marker_publisher_deletes_markers_no_longer_published(rclpy_node):
    """
    Markers of the previous call that are not published again are deleted.
    """
    marker_publisher = EvaluatedSpatialTypePublisher(node=rclpy_node)
    fake = FakeMarkerPublisher()
    marker_publisher.publisher = fake
    visualization = SpatialTypeVisualization(spatial_type=Point3(), namespace="removed")
    marker_publisher.publish(
        evaluated_visualizations=[
            EvaluatedSpatialTypeVisualization(
                spatial_type_markers=SpatialTypeMarkers.from_visualization(
                    visualization=visualization, root_frame_name="map"
                ),
                rendered_value=np.zeros(3),
            )
        ]
    )

    marker_publisher.publish(evaluated_visualizations=[])

    markers = fake.published[-1].markers
    assert [(marker.action, marker.ns) for marker in markers] == [
        (Marker.DELETE, "removed")
    ]


def test_closing_the_marker_publisher_removes_its_publisher_from_the_node(rclpy_node):
    """
    A closed marker publisher leaves no publisher behind on its node.
    """
    marker_publisher = EvaluatedSpatialTypePublisher(node=rclpy_node)

    marker_publisher.close()

    assert marker_publisher.publisher not in list(rclpy_node.publishers)


# %% evaluating spatial types


def test_compiled_visualizations_evaluate_the_current_variable_values(
    cylinder_bot_world, monkeypatch
):
    """
    Every evaluation reads the current variable values without recompiling.
    """
    dof = first_moving_dof(cylinder_bot_world)
    compiled_visualizations = CompiledSpatialTypeVisualizations(
        visualizations=[
            SpatialTypeVisualization(
                spatial_type=Point3(
                    x=dof.variables.position, reference_frame=cylinder_bot_world.root
                )
            )
        ],
        root_frame_name=str(cylinder_bot_world.root.name),
    )
    monkeypatch.setattr(
        SymbolicMathType, "evaluate", lambda self: pytest.fail("evaluate recompiles")
    )

    cylinder_bot_world.state[dof.id].position = 1.5
    [evaluated] = compiled_visualizations.evaluate()

    assert evaluated.rendered_value.tolist() == [1.5, 0, 0]


def test_compiled_visualizations_return_values_later_evaluations_leave_unchanged(
    cylinder_bot_world,
):
    """
    The returned values are copies, so another thread may read them while the next
    evaluation runs.
    """
    dof = first_moving_dof(cylinder_bot_world)
    compiled_visualizations = CompiledSpatialTypeVisualizations(
        visualizations=[
            SpatialTypeVisualization(
                spatial_type=Point3(
                    x=dof.variables.position, reference_frame=cylinder_bot_world.root
                )
            )
        ],
        root_frame_name=str(cylinder_bot_world.root.name),
    )
    cylinder_bot_world.state[dof.id].position = 1.5
    [first] = compiled_visualizations.evaluate()

    cylinder_bot_world.state[dof.id].position = 2.5
    compiled_visualizations.evaluate()

    assert first.rendered_value.tolist() == [1.5, 0, 0]


# %% live world-state visualization


def test_publish_once_publishes_expected_markers(rclpy_node, cylinder_bot_world):
    point = Point3(1, 2, 3, reference_frame=cylinder_bot_world.root)

    publisher = SpatialTypePublisher.publish_once(
        spatial_type=point, node=rclpy_node, world=cylinder_bot_world
    )
    fake = capture(publisher)
    publisher.publish()

    markers = fake.published[-1].markers
    assert len(markers) == 1
    assert markers[0].type == Marker.SPHERE
    assert (markers[0].pose.position.x, markers[0].pose.position.y) == (1, 2)


def test_markers_update_on_state_change(rclpy_node, cylinder_bot_world):
    """
    A world state change republishes the markers at the new value.
    """
    dof = first_moving_dof(cylinder_bot_world)
    point = Point3(x=dof.variables.position, reference_frame=cylinder_bot_world.root)

    publisher = SpatialTypePublisher(node=rclpy_node, _world=cylinder_bot_world)
    fake = capture(publisher)
    publisher.add(request=SpatialTypeVisualization(spatial_type=point))
    initial_x = fake.published[-1].markers[0].pose.position.x

    cylinder_bot_world.state[dof.id].position = initial_x + 2.0
    cylinder_bot_world.notify_state_change()

    updated_x = fake.published[-1].markers[0].pose.position.x
    assert updated_x == initial_x + 2.0


def test_set_requests_replaces_previous(rclpy_node, cylinder_bot_world):
    """
    Replacing the requests deletes the markers of the replaced ones.
    """
    publisher = SpatialTypePublisher(node=rclpy_node, _world=cylinder_bot_world)
    publisher.add(
        request=SpatialTypeVisualization(
            spatial_type=Point3(0, 0, 0, reference_frame=cylinder_bot_world.root),
            namespace="a",
        )
    )
    fake = capture(publisher)

    publisher.set_requests(
        requests=[
            SpatialTypeVisualization(
                spatial_type=Point3(1, 1, 1, reference_frame=cylinder_bot_world.root),
                namespace="b",
            )
        ]
    )

    markers = fake.published[-1].markers
    added = [marker for marker in markers if marker.action == Marker.ADD]
    deleted = [marker for marker in markers if marker.action == Marker.DELETE]
    assert [marker.ns for marker in added] == ["b"]
    assert [marker.ns for marker in deleted] == ["a"]


def test_add_all_publishes_all_requests(rclpy_node, cylinder_bot_world):
    """
    Every added request is published.
    """
    publisher = SpatialTypePublisher(node=rclpy_node, _world=cylinder_bot_world)
    fake = capture(publisher)

    publisher.add_all(
        requests=[
            SpatialTypeVisualization(
                spatial_type=Point3(0, 0, 0, reference_frame=cylinder_bot_world.root),
                namespace="a",
            ),
            SpatialTypeVisualization(
                spatial_type=Point3(1, 1, 1, reference_frame=cylinder_bot_world.root),
                namespace="b",
            ),
        ]
    )

    markers = fake.published[-1].markers
    assert {marker.ns for marker in markers} == {"a", "b"}


def test_clear_deletes_previously_published_markers(rclpy_node, cylinder_bot_world):
    """
    Clearing deletes every published marker and forgets the requests.
    """
    publisher = SpatialTypePublisher(node=rclpy_node, _world=cylinder_bot_world)
    fake = capture(publisher)
    publisher.add(
        request=SpatialTypeVisualization(
            spatial_type=Point3(0, 0, 0, reference_frame=cylinder_bot_world.root)
        )
    )
    published_identities = {
        (marker.ns, marker.id) for marker in fake.published[-1].markers
    }

    publisher.clear()

    markers = fake.published[-1].markers
    assert all(marker.action == Marker.DELETE for marker in markers)
    assert {(marker.ns, marker.id) for marker in markers} == published_identities
    assert publisher._compiled_visualizations.visualizations == []


def test_publish_deletes_only_removed_request_markers(rclpy_node, cylinder_bot_world):
    """
    Only the markers of removed requests are deleted.
    """
    publisher = SpatialTypePublisher(node=rclpy_node, _world=cylinder_bot_world)
    fake = capture(publisher)
    publisher.add_all(
        requests=[
            SpatialTypeVisualization(
                spatial_type=Point3(0, 0, 0, reference_frame=cylinder_bot_world.root),
                namespace="keep",
            ),
            SpatialTypeVisualization(
                spatial_type=Point3(1, 1, 1, reference_frame=cylinder_bot_world.root),
                namespace="drop",
            ),
        ]
    )

    publisher.set_requests(
        requests=[
            SpatialTypeVisualization(
                spatial_type=Point3(0, 0, 0, reference_frame=cylinder_bot_world.root),
                namespace="keep",
            )
        ]
    )

    markers = fake.published[-1].markers
    deleted = {marker.ns for marker in markers if marker.action == Marker.DELETE}
    assert deleted == {"drop"}
    assert all(marker.action == Marker.ADD for marker in markers if marker.ns == "keep")


def test_stop_deregisters_from_state_callbacks(rclpy_node, cylinder_bot_world):
    """
    Stopping removes the publisher from the world's state change callbacks.
    """
    publisher = SpatialTypePublisher(node=rclpy_node, _world=cylinder_bot_world)

    assert publisher in cylinder_bot_world.state.state_change_callbacks
    publisher.stop()
    assert publisher not in cylinder_bot_world.state.state_change_callbacks


def test_state_change_reuses_the_compiled_function(
    rclpy_node, cylinder_bot_world, monkeypatch
):
    """
    A state change evaluates the function compiled on registration instead of
    recompiling.
    """
    dof = first_moving_dof(cylinder_bot_world)
    point = Point3(x=dof.variables.position, reference_frame=cylinder_bot_world.root)
    publisher = SpatialTypePublisher(node=rclpy_node, _world=cylinder_bot_world)
    fake = capture(publisher)
    publisher.add(request=SpatialTypeVisualization(spatial_type=point))
    compiled_function = publisher._compiled_visualizations
    monkeypatch.setattr(
        SymbolicMathType, "evaluate", lambda self: pytest.fail("evaluate recompiles")
    )

    cylinder_bot_world.state[dof.id].position = 1.5
    cylinder_bot_world.notify_state_change()

    assert publisher._compiled_visualizations is compiled_function
    assert fake.published[-1].markers[0].pose.position.x == 1.5


def test_state_change_builds_no_markers(rclpy_node, cylinder_bot_world, monkeypatch):
    """
    A state change writes the new value into the markers built on registration.
    """
    dof = first_moving_dof(cylinder_bot_world)
    point = Point3(x=dof.variables.position, reference_frame=cylinder_bot_world.root)
    publisher = SpatialTypePublisher(node=rclpy_node, _world=cylinder_bot_world)
    fake = capture(publisher)
    publisher.add(request=SpatialTypeVisualization(spatial_type=point, label="point"))

    def fail_building_markers(**kwargs) -> Marker:
        """
        Fail the test instead of building a marker on a state change.
        """
        pytest.fail("a state change built markers")

    monkeypatch.setattr(spatial_type_marker_renderer, "Marker", fail_building_markers)

    cylinder_bot_world.state[dof.id].position = 1.5
    cylinder_bot_world.notify_state_change()

    assert [marker.pose.position.x for marker in fake.published[-1].markers] == [
        1.5,
        1.5,
    ]
