"""
Unit tests for the live transform graph: what changed, when, and who wrote it.

The tracker is exercised against a real :class:`~semantic_digital_twin.world.World`,
since the whole point of the view is which of the world's connections actually move — a
mimic of a connection would only prove that the mimic was read.
"""

from __future__ import annotations

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Dict, Tuple

from cramera.live.bridge import Bridge, MoveRequest
from cramera.live.transforms import (
    ConnectionActivity,
    ConnectionKind,
    TransformFreshness,
    TransformGraph,
    TransformWriter,
)

from .test_live_bridge import make_free_floating_object

# %% a world whose three connection kinds are all present at once
HINGE_NAME = PrefixedName("drawer_joint", prefix="kitchen")
"""
Name of the actuated connection the tests drive.
"""

SHELF_NAME = PrefixedName("shelf_joint", prefix="kitchen")
"""
Name of the fixed connection the tests expect to stay static.
"""


def make_kitchen() -> Tuple[World, RevoluteConnection]:
    """
    A world with one actuated hinge and one fixed shelf, both under the root.
    """
    world = World()
    root = Body(name=PrefixedName("root", prefix="kitchen"))
    drawer = Body(name=PrefixedName("drawer", prefix="kitchen"))
    shelf = Body(name=PrefixedName("shelf", prefix="kitchen"))
    with world.modify_world():
        world.add_body(root)
        hinge = RevoluteConnection.create_with_dofs(
            world, root, drawer, name=HINGE_NAME, axis=Vector3.Z()
        )
        world.add_connection(hinge)
        world.add_connection(FixedConnection(parent=root, child=shelf, name=SHELF_NAME))
    return world, hinge


def make_draggable_object() -> Tuple[World, Connection6DoF, Body]:
    """
    A world whose one free-floating body is mesh-named, so the bridge's own discovery
    publishes it and a queued viewer move reaches it.
    """
    world = World()
    root = Body(name=PrefixedName("root", prefix="world"))
    milk = Body(name=PrefixedName("milk.stl", prefix="world"))
    with world.modify_world():
        world.add_body(root)
        degrees_of_freedom = [
            DegreeOfFreedom(name=PrefixedName(component, prefix="milk"))
            for component in ("x", "y", "z", "qx", "qy", "qz", "qw")
        ]
        for degree_of_freedom in degrees_of_freedom:
            world.add_degree_of_freedom(degree_of_freedom)
        x, y, z, qx, qy, qz, qw = degrees_of_freedom
        connection = Connection6DoF(
            parent=root, child=milk, x=x, y=y, z=z, qx=qx, qy=qy, qz=qz, qw=qw
        )
        world.add_connection(connection)
        world.state[qw.id].position = 1.0
    return world, connection, milk


def observe(
    graph: TransformGraph, world: World, now: float
) -> Dict[str, ConnectionActivity]:
    """
    One observation of every connection in ``world``, indexed by connection name.

    :param graph: The tracker doing the observing.
    :param world: The world whose connections are read.
    :param now: The timestamp the observation is stamped with.
    """
    snapshot = graph.observe(list(world.connections), world, now)
    return {activity.name: activity for activity in snapshot.activities}


class TestConnectionKind:
    def test_an_actuated_connection_is_classified_by_its_degrees_of_freedom(self):
        world, hinge = make_kitchen()

        assert ConnectionKind.of_connection(hinge) is ConnectionKind.ACTUATED

    def test_a_fixed_connection_is_static(self):
        world, _ = make_kitchen()
        shelf = next(
            connection
            for connection in world.connections
            if str(connection.name) == str(SHELF_NAME)
        )

        assert ConnectionKind.of_connection(shelf) is ConnectionKind.FIXED

    def test_a_free_floating_connection_is_classified_as_free(self):
        _, connection, _ = make_free_floating_object()

        assert ConnectionKind.of_connection(connection) is ConnectionKind.FREE


class TestChangeDetection:
    """
    The tracker's one job: stamping a connection with the moment its transform last
    changed, so the viewer can tell a joint that is executing from one that is not.
    """

    def test_a_connection_nothing_has_written_carries_no_change_time(self):
        world, _ = make_kitchen()
        graph = TransformGraph()

        activities = observe(graph, world, 100.0)

        assert activities[str(HINGE_NAME)].changed_at is None

    def test_a_moved_joint_is_stamped_with_the_time_it_moved(self):
        world, hinge = make_kitchen()
        graph = TransformGraph()
        observe(graph, world, 100.0)

        hinge.position = 0.7
        activities = observe(graph, world, 101.5)

        assert activities[str(HINGE_NAME)].changed_at == 101.5

    def test_an_unmoved_joint_keeps_the_time_of_its_last_change(self):
        world, hinge = make_kitchen()
        graph = TransformGraph()
        observe(graph, world, 100.0)
        hinge.position = 0.7
        observe(graph, world, 101.5)

        activities = observe(graph, world, 109.0)

        assert activities[str(HINGE_NAME)].changed_at == 101.5

    def test_a_fixed_connection_never_gains_a_change_time(self):
        world, hinge = make_kitchen()
        graph = TransformGraph()
        observe(graph, world, 100.0)
        hinge.position = 0.7

        activities = observe(graph, world, 101.5)

        assert activities[str(SHELF_NAME)].changed_at is None


class TestWriterAttribution:
    def test_a_joint_the_demo_moved_is_attributed_to_the_demo(self):
        world, hinge = make_kitchen()
        graph = TransformGraph()
        observe(graph, world, 100.0)

        hinge.position = 0.7
        activities = observe(graph, world, 101.0)

        assert activities[str(HINGE_NAME)].writer is TransformWriter.DEMO

    def test_a_connection_the_viewer_dragged_is_attributed_to_the_viewer(self):
        world, connection, _ = make_free_floating_object()
        graph = TransformGraph()
        observe(graph, world, 100.0)

        graph.note_viewer_write(str(connection.name))
        activities = observe(graph, world, 101.0)

        assert activities[str(connection.name)].writer is TransformWriter.VIEWER

    def test_a_viewer_drag_ages_instead_of_being_stamped_again(self):
        """
        The drag is one change, not a change per tick: later observations of the same
        untouched connection must keep the moment it was dragged.
        """
        world, connection, _ = make_free_floating_object()
        graph = TransformGraph()
        observe(graph, world, 100.0)
        graph.note_viewer_write(str(connection.name))
        observe(graph, world, 101.0)

        activities = observe(graph, world, 102.0)

        assert activities[str(connection.name)].changed_at == 101.0

    def test_an_untouched_connection_has_no_writer(self):
        world, _ = make_kitchen()
        graph = TransformGraph()

        activities = observe(graph, world, 100.0)

        assert activities[str(HINGE_NAME)].writer is TransformWriter.NOBODY


class TestFreshness:
    """
    The freshness a connection's age maps to — what the viewer draws as the ring of the
    frame the connection carries.
    """

    def test_a_connection_without_degrees_of_freedom_is_static(self):
        world, _ = make_kitchen()
        graph = TransformGraph()

        activities = observe(graph, world, 100.0)

        assert activities[str(SHELF_NAME)].freshness(100.0) is TransformFreshness.STATIC

    def test_a_joint_that_never_moved_is_stale(self):
        world, _ = make_kitchen()
        graph = TransformGraph()

        activities = observe(graph, world, 100.0)

        assert activities[str(HINGE_NAME)].freshness(100.0) is TransformFreshness.STALE

    @pytest.mark.parametrize(
        "elapsed, expected",
        [
            (0.0, TransformFreshness.MOVING),
            (ConnectionActivity.MOVING_SECONDS / 2, TransformFreshness.MOVING),
            (ConnectionActivity.MOVING_SECONDS * 2, TransformFreshness.SETTLED),
            (ConnectionActivity.SETTLED_SECONDS * 2, TransformFreshness.STALE),
        ],
    )
    def test_freshness_follows_the_age_of_the_last_change(self, elapsed, expected):
        world, hinge = make_kitchen()
        graph = TransformGraph()
        observe(graph, world, 100.0)
        hinge.position = 0.7
        activities = observe(graph, world, 101.0)

        assert activities[str(HINGE_NAME)].freshness(101.0 + elapsed) is expected


class TestSnapshotPayload:
    def test_the_signature_survives_a_joint_moving(self):
        world, hinge = make_kitchen()
        graph = TransformGraph()
        before = graph.observe(list(world.connections), world, 100.0)

        hinge.position = 0.7
        after = graph.observe(list(world.connections), world, 101.0)

        assert after.signature == before.signature

    def test_a_new_connection_changes_the_signature(self):
        world, _ = make_kitchen()
        graph = TransformGraph()
        before = graph.observe(list(world.connections), world, 100.0)
        with world.modify_world():
            world.add_connection(
                FixedConnection(
                    parent=world.root, child=Body(name=PrefixedName("cup", "kitchen"))
                )
            )

        after = graph.observe(list(world.connections), world, 101.0)

        assert after.signature != before.signature

    def test_the_payload_reports_the_age_of_every_change(self):
        world, hinge = make_kitchen()
        graph = TransformGraph()
        graph.observe(list(world.connections), world, 100.0)
        hinge.position = 0.7
        snapshot = graph.observe(list(world.connections), world, 101.0)

        payload = snapshot.to_payload(103.5)

        hinge_payload = next(
            entry
            for entry in payload["connections"]
            if entry["name"] == str(HINGE_NAME)
        )
        assert hinge_payload["ageSeconds"] == 2.5

    def test_a_connection_that_never_changed_reports_no_age(self):
        world, _ = make_kitchen()
        graph = TransformGraph()

        payload = graph.observe(list(world.connections), world, 100.0).to_payload(105.0)

        assert all(entry["ageSeconds"] is None for entry in payload["connections"])

    def test_every_connection_is_reported_with_its_parent_and_child(self):
        world, _ = make_kitchen()
        graph = TransformGraph()

        payload = graph.observe(list(world.connections), world, 100.0).to_payload(100.0)

        shelf = next(
            entry
            for entry in payload["connections"]
            if entry["name"] == str(SHELF_NAME)
        )
        assert (shelf["parent"], shelf["child"]) == (
            str(world.root.name),
            "kitchen/shelf",
        )


class TestBridgeIntegration:
    """
    What the viewer polls: the bridge publishes the transform graph of the world it is
    attached to, and attributes its own applied drags to the viewer.
    """

    def test_a_fresh_bridge_publishes_nothing(self):
        assert Bridge().get_transforms()["connections"] == []

    def test_the_attached_worlds_connections_are_published(self):
        world, _ = make_kitchen()
        bridge = Bridge()
        bridge.world = world

        bridge.bind()
        bridge.snapshot()

        names = {entry["name"] for entry in bridge.get_transforms()["connections"]}
        assert names == {str(HINGE_NAME), str(SHELF_NAME)}

    def test_a_joint_the_demo_moves_is_published_as_moving(self):
        world, hinge = make_kitchen()
        bridge = Bridge()
        bridge.world = world
        bridge.bind()
        bridge.snapshot()

        hinge.position = 0.9
        bridge.snapshot()

        hinge_payload = next(
            entry
            for entry in bridge.get_transforms()["connections"]
            if entry["name"] == str(HINGE_NAME)
        )
        assert hinge_payload["freshness"] == TransformFreshness.MOVING
        assert hinge_payload["writer"] == TransformWriter.DEMO

    def test_an_applied_viewer_drag_is_published_as_written_by_the_viewer(self):
        world, connection, _ = make_draggable_object()
        bridge = Bridge()
        bridge.world = world
        bridge.bind()
        bridge.snapshot()

        bridge.queue_move(MoveRequest(object_key="milk.stl", position=[1.0, 0.0, 0.0]))
        bridge.apply_moves()
        bridge.snapshot()

        dragged = next(
            entry
            for entry in bridge.get_transforms()["connections"]
            if entry["name"] == str(connection.name)
        )
        assert dragged["writer"] == TransformWriter.VIEWER
