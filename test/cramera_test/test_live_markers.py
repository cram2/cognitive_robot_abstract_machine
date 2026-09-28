"""
Tests of the debug-marker overlay: RViz-style store semantics, the exclusion of the
world-geometry markers, and frame resolution into world coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any, List

from cramera.live.bridge import Bridge
from cramera.live.markers import MarkerEntry, MarkerStore

# %% mimics of the visualization_msgs vocabulary


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class MimicPose:
    position: Point = field(default_factory=Point)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class Color:
    r: float = 1.0
    g: float = 0.5
    b: float = 0.0
    a: float = 1.0


@dataclass
class Header:
    frame_id: str = ""


@dataclass
class MimicMarker:
    """
    A ``visualization_msgs`` marker, as the store reads it duck-typed.
    """

    ns: str = "debug"
    id: int = 0
    type: int = 2  # SPHERE
    action: int = 0  # ADD
    header: Header = field(default_factory=Header)
    pose: MimicPose = field(default_factory=MimicPose)
    scale: Point = field(default_factory=lambda: Point(0.1, 0.1, 0.1))
    color: Color = field(default_factory=Color)
    points: List[Any] = field(default_factory=list)
    text: str = ""


# %% the store


class TestMarkerStore:
    def test_an_added_marker_is_kept(self):
        store = MarkerStore()

        assert store.observe([MimicMarker()]) is True

        entry = store.entries[("debug", 0)]
        assert entry.kind == "sphere"
        assert entry.color == "#ff8000"
        assert entry.scale == [0.1, 0.1, 0.1]

    def test_republishing_the_same_marker_changes_nothing(self):
        store = MarkerStore()
        store.observe([MimicMarker()])

        assert store.observe([MimicMarker()]) is False
        assert store.revision == 1

    def test_a_delete_removes_one_marker(self):
        store = MarkerStore()
        store.observe([MimicMarker(id=1), MimicMarker(id=2)])

        store.observe([MimicMarker(id=1, action=2)])

        assert list(store.entries) == [("debug", 2)]

    def test_delete_all_clears_the_topic(self):
        store = MarkerStore()
        store.observe([MimicMarker(id=1), MimicMarker(id=2)])

        store.observe([MimicMarker(action=3)])

        assert store.entries == {}

    def test_an_unsupported_marker_type_is_skipped(self):
        store = MarkerStore()

        assert store.observe([MimicMarker(type=10)]) is False  # MESH_RESOURCE
        assert store.entries == {}

    def test_a_line_marker_keeps_its_points(self):
        marker = MimicMarker(type=4, points=[Point(0, 0, 0), Point(1, 0, 0)])
        entry = MarkerEntry.from_message(marker)

        assert entry.kind == "line_strip"
        assert entry.points == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


# %% the bridge overlay


def marker_world() -> World:
    """
    A world with one robot-ish body at a known pose, for exclusion and frames.
    """
    world = World()
    root = Body(name=PrefixedName("root", prefix="world"))
    hand = Body(name=PrefixedName("hand", prefix="robot"))
    with world.modify_world():
        world.add_body(root)
        world.add_connection(
            FixedConnection(
                parent=root,
                child=hand,
                parent_T_connection_expression=(
                    HomogeneousTransformationMatrix.from_xyz_rpy(1.0, 2.0, 0.5)
                ),
            )
        )
    return world


class TestBridgeMarkerOverlay:
    def test_markers_reach_the_state_and_endpoint_payloads(self):
        bridge = Bridge()
        bridge.attach(marker_world())

        bridge.observe_ros_markers("/semworld/viz_marker", [MimicMarker()])
        bridge.snapshot()

        payload = bridge.get_markers()
        assert payload["version"] == 1
        assert [marker["ns"] for marker in payload["markers"]] == ["debug"]
        assert bridge.get_state()["markersVersion"] == 1

    def test_world_geometry_markers_stay_out(self):
        """
        The robot and environment markers RViz shows are the world model itself, which
        the scene already renders — their namespaces are the world's body names and must
        not reach the overlay.
        """
        bridge = Bridge()
        bridge.attach(marker_world())

        bridge.observe_ros_markers(
            "/semworld/viz_marker",
            [MimicMarker(ns="robot/hand"), MimicMarker(ns="debug", id=7)],
        )
        bridge.snapshot()

        assert [marker["ns"] for marker in bridge.get_markers()["markers"]] == ["debug"]

    def test_a_body_frame_anchors_the_marker_to_the_body(self):
        bridge = Bridge()
        bridge.attach(marker_world())
        marker = MimicMarker(header=Header(frame_id="robot/hand"))
        marker.pose.position = Point(0.1, 0.0, 0.0)

        bridge.observe_ros_markers("/semworld/viz_marker", [marker])
        bridge.snapshot()

        pose = bridge.get_markers()["markers"][0]["pose"]
        assert pose[:3] == [1.1, 2.0, 0.5]

    def test_an_unknown_frame_reads_as_the_world(self):
        bridge = Bridge()
        bridge.attach(marker_world())
        marker = MimicMarker(header=Header(frame_id="map"))
        marker.pose.position = Point(0.3, 0.0, 0.0)

        bridge.observe_ros_markers("/semworld/viz_marker", [marker])
        bridge.snapshot()

        assert bridge.get_markers()["markers"][0]["pose"][:3] == [0.3, 0.0, 0.0]

    def test_an_unchanged_marker_set_is_not_rebuilt(self):
        bridge = Bridge()
        bridge.attach(marker_world())
        bridge.observe_ros_markers("/semworld/viz_marker", [MimicMarker()])
        bridge.snapshot()
        first = bridge.get_markers()

        bridge.snapshot()

        assert bridge.get_markers() is first


# %% the viewer's marker settings


@dataclass
class RecordingListener:
    """
    A marker listener mimic recording what the settings ask of it.
    """

    topics: List[str] = field(default_factory=lambda: ["/semworld/viz_marker"])

    def subscribe(self, topic: str) -> None:
        if topic not in self.topics:
            self.topics.append(topic)

    def unsubscribe(self, topic: str) -> None:
        if topic in self.topics:
            self.topics.remove(topic)

    def subscribed_topics(self) -> List[str]:
        return sorted(self.topics)

    def available_marker_topics(self) -> List[str]:
        return ["/coraplex/viz_marker", "/semworld/viz_marker"]


class TestMarkerTopicSettings:
    def test_without_ros_the_settings_report_it(self):
        payload = Bridge().marker_topics_payload()

        assert payload == {"ok": True, "ros": False, "subscribed": [], "available": []}

    def test_the_settings_list_subscribed_and_available_topics(self):
        bridge = Bridge()
        bridge.marker_listener = RecordingListener()

        payload = bridge.marker_topics_payload()

        assert payload["subscribed"] == ["/semworld/viz_marker"]
        assert payload["available"] == ["/coraplex/viz_marker", "/semworld/viz_marker"]

    def test_watching_a_topic_subscribes_it(self):
        bridge = Bridge()
        bridge.marker_listener = RecordingListener()

        payload = bridge.set_marker_topic("/coraplex/viz_marker", True)

        assert payload["subscribed"] == ["/coraplex/viz_marker", "/semworld/viz_marker"]

    def test_dropping_a_topic_clears_its_markers(self):
        bridge = Bridge()
        bridge.attach(marker_world())
        bridge.marker_listener = RecordingListener()
        bridge.observe_ros_markers("/semworld/viz_marker", [MimicMarker()])
        bridge.snapshot()
        version_before = bridge.get_markers()["version"]

        bridge.set_marker_topic("/semworld/viz_marker", False)
        bridge.snapshot()

        payload = bridge.get_markers()
        assert payload["markers"] == []
        assert payload["version"] > version_before

    def test_a_topic_must_be_absolute(self):
        bridge = Bridge()
        bridge.marker_listener = RecordingListener()

        assert bridge.set_marker_topic("nope", True)["ok"] is False

    def test_without_ros_watching_is_refused(self):
        assert Bridge().set_marker_topic("/x", True)["ok"] is False
