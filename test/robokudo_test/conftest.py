import os
from dataclasses import dataclass

import py_trees
import pytest
import rclpy
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from rclpy.node import Node

import robokudo.defs
import robokudo.io.ros
from robokudo.descriptors.camera_configs.config_mongodb_playback import (
    MongoCameraConfig,
)

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
    n = Node(robokudo.defs.TEST_ROS_NODE_NAME)
    robokudo.io.ros.register_node(n)
    yield n
    robokudo.io.ros.clear_node(n)
    n.destroy_node()


@pytest.fixture(autouse=True)
def cleanup_after_test():
    yield


# Move this to general test/conftest.py?
# @pytest.fixture(autouse=True, scope="function")
# def cleanup_after_test(request):
#     if request.node.get_closest_marker("skip_heavy_cleanup"):
#         yield
#         return
#
#     # heavy setup
#     SymbolGraph.clear()
#     class_diagram = ClassDiagram(
#         recursive_subclasses(Symbol) + [World],
#         introspector=DescriptorAwareIntrospector(),
#     )
#     SymbolGraph(_class_diagram=class_diagram)
#     yield
#     SymbolGraph.clear()
