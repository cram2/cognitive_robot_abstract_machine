"""
Serving a world to other processes, and finding, fetching or joining a world another
process serves.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass, field
from time import time, sleep

from rclpy.node import Node
from typing_extensions import Any
from rclpy.service import Service
from std_srvs.srv import Trigger

from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.adapters.ros.messages import WorldModelSnapshot
from semantic_digital_twin.adapters.ros.ros2_node import HasROS2Node
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer


@dataclass
class FetchWorldServer(HasROS2Node):
    """
    A ros service that allows other processes to fetch the entire world modification
    list from this world.

    The modification list is sent via a JSON string message.
    """

    world: World
    """
    The world to fetch modifications from.
    """

    service_name: str = "/semantic_digital_twin/fetch_world"
    """
    The name of the service.
    """

    service: Service | None = field(default=None, init=False)
    """
    The ROS service object.
    """

    def __post_init__(self):
        """
        Initialize the ROS service.
        """
        self.service = self.node.create_service(
            Trigger, self.service_name, self.service_callback
        )

    def service_callback(self, request: Trigger.Request, response: Trigger.Response):
        """
        Handle service requests to fetch the world modification list.

        :param request: The service request (empty for Trigger service).
        :param response: The service response containing success status and message.
        :return: The populated response.
        """
        # Return a payload that contains both the modification blocks and the current state snapshot
        modifications_json = self.get_payload_as_json()
        response.success = True
        response.message = modifications_json
        return response

    def get_modifications_as_json(self) -> str:
        """
        Serialize all world modification blocks to JSON.

        :return: JSON string containing all modification blocks.
        """
        modifications_list = [
            block.to_json()
            for block in self.world.get_world_model_manager().model_modification_blocks
        ]
        return json.dumps(modifications_list)

    def get_payload_as_json(self) -> str:
        """
        Serialize the served world as a :class:`WorldModelSnapshot`, naming the stream
        positions the world contains.
        """
        model_manager = self.world.get_world_model_manager()
        # Every modification that ended before the snapshot has been published once its
        # turn in the publication order is free, so the positions match the content.
        with self.world._world_lock, model_manager.publishing_in_order():
            snapshot = WorldModelSnapshot(
                modifications=list(model_manager.model_modification_blocks),
                ids=list(self.world.state.keys()),
                states=list(self.world.state.positions),
                stream_positions=[
                    position
                    for synchronizer in WorldSynchronizer.all_callbacks_of_this_type_from_world(
                        self.world
                    )
                    for position in synchronizer.contained_positions
                ],
            )

        return json.dumps(snapshot.to_json())

    def close(self):
        """
        Clean up the service.
        """
        if self.service is not None:
            self.node.destroy_service(self.service)
            self.service = None


@dataclass
class NoServiceFoundError(Exception):
    service_suffix: str


def find_world_fetcher_service(
    node: Node,
    service_suffix: str = "fetch_world",
    timeout_seconds: float = 10.0,
) -> str | None:
    """
    Find a WorldFetcher service in the ROS2 network.

    :param node: The ROS2 node to use for discovery.
    :param service_suffix: The suffix (last part behind '/') of the WorldFetcher
        services to look for.
    :param timeout_seconds: How long to wait for a service to be discovered.
    :return: The name of the first WorldFetcher service discovered, or None if none was
        discovered within ``timeout_seconds``.
    """
    deadline = time() + timeout_seconds
    while True:
        matching_services = [
            name
            for name, srv_type in node.get_service_names_and_types()
            if name.endswith(service_suffix) and srv_type == ["std_srvs/srv/Trigger"]
        ]
        if matching_services:
            return matching_services[0]
        if time() >= deadline:
            return None
        sleep(0.1)


def fetch_world_snapshot_json(
    node: Node,
    service_suffix: str = "fetch_world",
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """
    Fetch the snapshot of the world the first WorldFetcher service found serves.

    :param node: The ROS2 node to use for communication.
    :param service_suffix: The suffix (last part behind '/') of the WorldFetcher
        services to look for.
    :param timeout_seconds: Maximum time to wait for service availability and response.
    :return: The serialized :class:`WorldModelSnapshot`.
    """
    deadline = time() + timeout_seconds
    chosen_service = find_world_fetcher_service(node, service_suffix, timeout_seconds)
    if chosen_service is None:
        raise NoServiceFoundError(service_suffix)
    remaining = deadline - time()

    client = node.create_client(Trigger, chosen_service)

    service_available = client.wait_for_service(timeout_sec=remaining)
    if not service_available:
        raise TimeoutError(
            f"WorldFetcher service '{chosen_service}' not available after {timeout_seconds} seconds"
        )

    response = client.call(Trigger.Request(), timeout_sec=remaining)
    if response is None:
        raise TimeoutError(
            f"WorldFetcher service '{chosen_service}' did not respond after {remaining} seconds"
        )
    return json.loads(response.message)


def fetch_world_from_service(
    node: Node,
    service_suffix: str = "fetch_world",
    timeout_seconds: float = 10.0,
) -> World:
    """
    Fetch a world from any WorldFetcher Service.

    The fetched world is not kept in step with the served one; :func:`join_shared_world`
    does that.

    :param node: The ROS2 node to use for communication.
    :param service_suffix: The suffix (last part behind '/') of the WorldFetcher
        services to look for.
    :param timeout_seconds: Maximum time to wait for service availability and response.
    :return: The fetched world.
    """
    world = World()
    WorldModelSnapshot.apply_to_json_snapshot_to_world(
        world,
        fetch_world_snapshot_json(node, service_suffix, timeout_seconds),
        **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs(),
    )
    return world


def join_shared_world(
    node: Node,
    service_suffix: str = "fetch_world",
    timeout_seconds: float = 10.0,
) -> World:
    """
    Fetch the world a WorldFetcher service serves and keep it in step with the served
    world through a :class:`WorldSynchronizer`, see :meth:`WorldSynchronizer.of_world`.
    No update published while the world is fetched is lost or applied twice, and taking
    in the fetched world is not published.

    :param node: The ROS2 node to use for communication.
    :param service_suffix: The suffix (last part behind '/') of the WorldFetcher
        services to look for.
    :param timeout_seconds: Maximum time to wait for service availability and response.
    :return: The joined world.
    """
    world = World()
    synchronizer = WorldSynchronizer(
        node=node, _world=world, defer_incoming_updates=True
    )
    with ExitStack() as closing_on_failure:
        closing_on_failure.callback(synchronizer.close)
        snapshot_json = fetch_world_snapshot_json(node, service_suffix, timeout_seconds)
        synchronizer.pause()
        WorldModelSnapshot.apply_to_json_snapshot_to_world(
            world,
            snapshot_json,
            **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs(),
        )
        synchronizer.update_previous_world_state()
        synchronizer.resume()
        for position in WorldModelSnapshot.stream_positions_from_json(
            json_data=snapshot_json
        ):
            synchronizer.record_applied(position=position)
        synchronizer.stop_deferring_incoming_updates()
        closing_on_failure.pop_all()
    return world
