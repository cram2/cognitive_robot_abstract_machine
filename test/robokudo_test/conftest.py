import os
from collections.abc import Iterator
from dataclasses import dataclass
from threading import Thread

import py_trees
import pytest
import rclpy
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from py_trees.blackboard import Blackboard
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

import robokudo.defs
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.descriptors.camera_configs.config_mongodb_playback import (
    MongoCameraConfig,
)
from robokudo.identifier import BBIdentifier

from ._perception_world import FrameName, SharedWorld, world_with_a_head_camera

# %% Mongo storage preservation


@dataclass(frozen=True)
class MongoStorageSnapshot:
    """
    Availability and contents of RoboKudo's default storage database.
    """

    is_available: bool
    """
    Whether MongoDB could be reached for this snapshot.
    """

    database_exists: bool
    """
    Whether RoboKudo's default storage database exists.
    """

    cas_document_count: int | None
    """
    Number of persisted CAS documents when the database exists.
    """


@dataclass(frozen=True)
class MongoStoragePreservationCheck:
    """
    Capture the default RoboKudo storage database without modifying it.
    """

    connection_timeout_milliseconds: int = 1_000
    """
    Maximum wait for an unavailable MongoDB server.
    """

    def snapshot(self) -> MongoStorageSnapshot:
        """
        Read the availability and contents of the default storage database.
        """
        mongo_client = MongoClient(
            host=os.getenv("RK_MONGO_HOST", "localhost"),
            port=int(os.getenv("RK_MONGO_PORT", 27017)),
            serverSelectionTimeoutMS=self.connection_timeout_milliseconds,
        )
        database_name = MongoCameraConfig().db_name
        try:
            if database_name not in mongo_client.list_database_names():
                return MongoStorageSnapshot(
                    is_available=True,
                    database_exists=False,
                    cas_document_count=None,
                )
            return MongoStorageSnapshot(
                is_available=True,
                database_exists=True,
                cas_document_count=mongo_client[database_name].cas.count_documents({}),
            )
        except ServerSelectionTimeoutError:
            return MongoStorageSnapshot(
                is_available=False,
                database_exists=False,
                cas_document_count=None,
            )
        finally:
            mongo_client.close()


@pytest.fixture(autouse=True, scope="session")
def preserve_initial_robokudo_storage_database():
    """
    Verify that an existing default RoboKudo storage database survives the suite.
    """
    preservation_check = MongoStoragePreservationCheck()
    initial_snapshot = preservation_check.snapshot()
    yield

    if not initial_snapshot.database_exists:
        return

    final_snapshot = preservation_check.snapshot()
    if not final_snapshot.is_available:
        return

    assert final_snapshot.database_exists
    assert final_snapshot.cas_document_count == initial_snapshot.cas_document_count


@pytest.fixture(scope="session", autouse=True)
def ros_default():
    # RoboKudo keeps its own ROS lifecycle fixture on purpose.
    # We do not use test/conftest.py::rclpy_node here because:
    # 1) RoboKudo tests need a session-wide ROS context for many tests/files.
    # 2) Several tests create their own additional Nodes (sometimes multiple per test).
    # 3) We want explicit control over init/shutdown order in this suite.
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    # init once (default/global context)
    if not rclpy.ok():
        rclpy.init()
    yield
    # shutdown once, but don't fail if something already shut it down
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except RuntimeError:
        pass


@pytest.fixture
def node(ros_default):
    node_registry = ROSNodeRegistry()
    n = Node(robokudo.defs.TEST_ROS_NODE_NAME)
    node_registry.register(n)
    yield n
    node_registry.clear(n)
    n.destroy_node()


@pytest.fixture
def other_process_node(ros_default: None) -> Iterator[Node]:
    """
    The node of another process, such as the one executing robot motions, which an
    executor spins in the background so that its services, actions and subscriptions are
    served while a test blocks on them.
    """
    node = Node("other_process")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = Thread(target=executor.spin, daemon=True)
    thread.start()
    yield node
    executor.shutdown()
    thread.join()
    node.destroy_node()


@pytest.fixture
def shared_world(other_process_node: Node) -> Iterator[SharedWorld]:
    """
    A world another process serves and synchronizes, holding only the world frame.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName(FrameName.WORLD)))
    served_world = SharedWorld.served_through(other_process_node, world)
    yield served_world
    served_world.close()


@pytest.fixture
def shared_world_with_a_robot(other_process_node: Node) -> Iterator[SharedWorld]:
    """
    A world another process serves and synchronizes, holding a robot whose camera hangs
    below its head, which that process moves.
    """
    served_world = SharedWorld.served_through(
        other_process_node, world_with_a_head_camera()
    )
    yield served_world
    served_world.close()


def clear_blackboard() -> None:
    """
    Destroy blackboard-owned ROS nodes before clearing shared state.
    """
    blackboard = Blackboard()
    if blackboard.exists(BBIdentifier.QUERY_SERVER):
        query_server = blackboard.get(BBIdentifier.QUERY_SERVER)
        query_server.destroy()
    Blackboard.clear()


@pytest.fixture(autouse=True)
def isolate_blackboard() -> Iterator[None]:
    """
    Give each RoboKudo test an isolated behavior-tree blackboard.
    """
    clear_blackboard()
    yield
    clear_blackboard()


# %% annotator visualization output


@pytest.fixture
def annotator_output_buffer() -> AnnotatorOutputPerPipelineMap:
    """
    Provide the blackboard buffer annotators write their visualization output to.
    """
    buffer = AnnotatorOutputPerPipelineMap()
    Blackboard().set("annotator_output_pipeline_map_buffer", buffer)
    return buffer
