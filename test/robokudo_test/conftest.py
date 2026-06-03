import py_trees
import pytest
import rclpy
from rclpy.node import Node

import robokudo.defs


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
    yield n
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
