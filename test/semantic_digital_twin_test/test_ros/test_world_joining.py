"""
Tests of joining a world another process serves, so that no update published while it is
fetched is lost or applied twice.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pytest
from rclpy.node import Node

from krrood.adapters.json_serializer import from_json, to_json
from krrood_synchronization.messages import MessageOrigin, Sender, StreamPosition
from semantic_digital_twin.adapters.ros.messages import (
    SnapshotField,
    WorldModelSnapshot,
)
from semantic_digital_twin.adapters.ros.world_fetcher import (
    FetchWorldServer,
    join_shared_world,
)
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

from .test_world_synchronizer import wait_for_condition

# %% served worlds


def add_body(world: World, name: str) -> None:
    """
    Add a body named ``name`` to ``world``, fixed to its root, in one modification.
    """
    with world.modify_world():
        body = Body(name=PrefixedName(name))
        world.add_body(body)
        world.add_connection(FixedConnection(parent=world.root, child=body))


def remove_body(world: World, name: str) -> None:
    """
    Remove the body named ``name`` and the connection to its parent from ``world``, in
    one modification.
    """
    with world.modify_world():
        body = world.get_body_by_name(name)
        world.remove_connection(body.parent_connection)
        world.remove_kinematic_structure_entity(body)


def served_world() -> World:
    """
    :return: A world of two bodies connected by a movable connection, and a body fixed
        to its root that a server may remove.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        child = Body(name=PrefixedName("child"))
        world.add_body(root)
        world.add_body(child)
        world.add_connection(
            Connection6DoF.create_with_dofs(parent=root, child=child, world=world)
        )
    add_body(world, "removable")
    return world


def body_names(world: World) -> list[str]:
    """
    :return: The names of the bodies of ``world``, sorted.
    """
    return sorted(body.name.name for body in world.bodies)


@dataclass(kw_only=True)
class ServerModifyingBeforeTheSnapshot(FetchWorldServer):
    """
    Serves a world from which a body is removed, and the removal published, right before
    the snapshot of a request is taken.
    """

    def get_payload_as_json(self) -> str:
        """
        :meth:`FetchWorldServer.get_payload_as_json` after removing the removable body.
        """
        remove_body(self.world, "removable")
        return super().get_payload_as_json()


@dataclass(kw_only=True)
class ServerModifyingAfterTheSnapshot(FetchWorldServer):
    """
    Serves a world that is modified, and the modification published, right after the
    snapshot of a request is taken and before it is answered.
    """

    def get_payload_as_json(self) -> str:
        """
        :meth:`FetchWorldServer.get_payload_as_json`, adding a body after the snapshot.
        """
        payload = super().get_payload_as_json()
        add_body(self.world, "added_after_the_snapshot")
        return payload


@dataclass(kw_only=True)
class ServingProcess:
    """
    The world a process serves, the synchronizer publishing its changes and the server
    answering requests for it.
    """

    world: World
    """
    The served world.
    """

    synchronizer: WorldSynchronizer
    """
    Publishes the changes of :attr:`world`.
    """

    server: FetchWorldServer
    """
    Answers requests for :attr:`world`.
    """

    def close(self) -> None:
        """
        Stop serving and publishing :attr:`world`.
        """
        self.server.close()
        self.synchronizer.close()


def serve(node: Node, server_type: type[FetchWorldServer]) -> ServingProcess:
    """
    :return: A process serving :func:`served_world` through a server of
        ``server_type``.
    """
    world = served_world()
    synchronizer = WorldSynchronizer(node=node, _world=world)
    return ServingProcess(
        world=world,
        synchronizer=synchronizer,
        server=server_type(node=node, world=world),
    )


@pytest.fixture
def serving(rclpy_node: Node) -> Iterator[ServingProcess]:
    """
    A process serving its world through a plain server.
    """
    process = serve(rclpy_node, FetchWorldServer)
    yield process
    process.close()


def caught_up(joined_world: World, serving_process: ServingProcess) -> bool:
    """
    :return: Whether ``joined_world`` applied everything the serving process published.
    """
    return WorldSynchronizer.of_world(joined_world).has_applied(
        position=serving_process.synchronizer.latest_published_position
    )


# %% joining a served world


class TestJoiningAServedWorld:
    """
    A joined world starts as the served world and keeps in step with it.
    """

    def test_joined_world_is_the_served_world(
        self, *, rclpy_node: Node, serving: ServingProcess
    ) -> None:
        """
        The joined world holds the bodies and positions of the served world.
        """
        serving.world.state._data[0, :] = np.arange(len(serving.world.state))
        serving.world.notify_state_change()

        joined_world = join_shared_world(rclpy_node)

        assert body_names(joined_world) == body_names(serving.world)
        assert (
            joined_world.state.to_uuid_position_dict()
            == serving.world.state.to_uuid_position_dict()
        )
        WorldSynchronizer.of_world(joined_world).close()

    def test_joining_publishes_nothing(
        self, *, rclpy_node: Node, serving: ServingProcess
    ) -> None:
        """
        Taking in the served world is not announced to the other processes.
        """
        joined_world = join_shared_world(rclpy_node)

        assert WorldSynchronizer.of_world(joined_world).published_sequence_number == 0
        WorldSynchronizer.of_world(joined_world).close()

    def test_change_after_joining_reaches_the_joined_world(
        self, *, rclpy_node: Node, serving: ServingProcess
    ) -> None:
        """
        A modification of the served world after the join is applied to the joined one.
        """
        joined_world = join_shared_world(rclpy_node)

        add_body(serving.world, "added_after_joining")

        assert wait_for_condition(lambda: caught_up(joined_world, serving))
        assert body_names(joined_world) == body_names(serving.world)
        WorldSynchronizer.of_world(joined_world).close()


# %% updates published while the world is fetched


class TestUpdatesPublishedWhileFetching:
    """
    An update published while the world is fetched is applied exactly once, whether the
    snapshot contains it or not.
    """

    def test_update_the_snapshot_contains_is_not_applied_again(
        self, *, rclpy_node: Node
    ) -> None:
        """
        A removal published before the snapshot is taken is not applied to the joined
        world a second time, where it would refer to a body that is already gone.
        """
        serving_process = serve(rclpy_node, ServerModifyingBeforeTheSnapshot)

        joined_world = join_shared_world(rclpy_node)

        assert wait_for_condition(lambda: caught_up(joined_world, serving_process))
        assert body_names(joined_world) == body_names(serving_process.world)
        WorldSynchronizer.of_world(joined_world).close()
        serving_process.close()

    def test_update_published_after_the_snapshot_reaches_the_joined_world(
        self, *, rclpy_node: Node
    ) -> None:
        """
        A modification published after the snapshot is taken, before the joining process
        has the answer, is applied to the joined world.
        """
        serving_process = serve(rclpy_node, ServerModifyingAfterTheSnapshot)

        joined_world = join_shared_world(rclpy_node)

        assert wait_for_condition(lambda: caught_up(joined_world, serving_process))
        assert body_names(joined_world) == body_names(serving_process.world)
        WorldSynchronizer.of_world(joined_world).close()
        serving_process.close()


# %% stream positions of a snapshot


class TestSnapshotStreamPositions:
    """
    A snapshot names how far it caught up with each publisher.
    """

    def test_snapshot_keeps_its_stream_positions_through_json(self) -> None:
        """
        The stream positions survive serializing the snapshot.
        """
        position = StreamPosition(
            origin=MessageOrigin(sender=Sender(node_name="publisher", process_id=1)),
            sequence_number=3,
        )
        snapshot = WorldModelSnapshot(
            modifications=[], ids=[], states=[], stream_positions=[position]
        )

        assert from_json(to_json(snapshot)).stream_positions == [position]

    def test_served_snapshot_names_what_the_server_published(
        self, *, serving: ServingProcess
    ) -> None:
        """
        The snapshot of a served world names the position of the last change its
        synchronizer published.
        """
        add_body(serving.world, "published_before_serving")

        payload = json.loads(serving.server.get_payload_as_json())

        assert serving.synchronizer.latest_published_position in (
            WorldModelSnapshot.stream_positions_from_json(json_data=payload)
        )

    def test_snapshot_without_stream_positions_names_none(self) -> None:
        """
        A snapshot served by an older server, without stream positions, names none.
        """
        payload = WorldModelSnapshot(modifications=[], ids=[], states=[]).to_json()
        del payload[SnapshotField.STREAM_POSITIONS]

        assert WorldModelSnapshot.stream_positions_from_json(json_data=payload) == []
