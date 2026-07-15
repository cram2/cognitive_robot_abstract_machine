from dataclasses import dataclass, field

from visualization_msgs.msg import Marker, MarkerArray

from semantic_digital_twin.adapters.ros.visualization.spatial_type_marker_renderer import (
    SpatialTypeVisualization,
)
from semantic_digital_twin.adapters.ros.visualization.spatial_type_publisher import (
    SpatialTypePublisher,
)
from semantic_digital_twin.spatial_types import Point3
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
    fake = FakeMarkerPublisher()
    publisher.publisher = fake
    return fake


def first_moving_dof(world: World) -> DegreeOfFreedom:
    return world.active_degrees_of_freedom[0]


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
    dof = first_moving_dof(cylinder_bot_world)
    point = Point3(x=dof.variables.position, reference_frame=cylinder_bot_world.root)

    publisher = SpatialTypePublisher(node=rclpy_node, _world=cylinder_bot_world)
    fake = capture(publisher)
    publisher.add(SpatialTypeVisualization(spatial_type=point))
    initial_x = fake.published[-1].markers[0].pose.position.x

    cylinder_bot_world.state[dof.id].position = initial_x + 2.0
    cylinder_bot_world.notify_state_change()

    updated_x = fake.published[-1].markers[0].pose.position.x
    assert updated_x == initial_x + 2.0


def test_set_requests_replaces_previous(rclpy_node, cylinder_bot_world):
    publisher = SpatialTypePublisher(node=rclpy_node, _world=cylinder_bot_world)
    publisher.add(
        SpatialTypeVisualization(
            spatial_type=Point3(0, 0, 0, reference_frame=cylinder_bot_world.root),
            namespace="a",
        )
    )
    fake = capture(publisher)

    publisher.set_requests(
        [
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
    publisher = SpatialTypePublisher(node=rclpy_node, _world=cylinder_bot_world)
    fake = capture(publisher)

    publisher.add_all(
        [
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
    publisher = SpatialTypePublisher(node=rclpy_node, _world=cylinder_bot_world)
    fake = capture(publisher)
    publisher.add(
        SpatialTypeVisualization(
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
    assert publisher._requests == []


def test_publish_deletes_only_removed_request_markers(rclpy_node, cylinder_bot_world):
    publisher = SpatialTypePublisher(node=rclpy_node, _world=cylinder_bot_world)
    fake = capture(publisher)
    publisher.add_all(
        [
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
        [
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
    publisher = SpatialTypePublisher(node=rclpy_node, _world=cylinder_bot_world)

    assert publisher in cylinder_bot_world.state.state_change_callbacks
    publisher.stop()
    assert publisher not in cylinder_bot_world.state.state_change_callbacks
