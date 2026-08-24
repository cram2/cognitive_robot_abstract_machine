import pytest

from robokudo.exceptions import RoboKudoROSNodeMissing
from robokudo.io import ros

# %% node registry


class RegisteredNode:
    """
    Mimic a ROS node for registry lifecycle tests.
    """


def test_node_registry_reuses_its_single_instance() -> None:
    assert ros.RoboKudoNodeRegistry() is ros.RoboKudoNodeRegistry()


def setup_function() -> None:
    ros.clear_node()


def teardown_function() -> None:
    ros.clear_node()


def test_register_node_makes_node_available():
    node = RegisteredNode()

    ros.register_node(node)

    assert ros.get_node() is node


def test_clear_node_removes_registered_node():
    node = RegisteredNode()
    ros.register_node(node)

    ros.clear_node(node)

    with pytest.raises(RoboKudoROSNodeMissing):
        ros.get_node()


def test_clear_node_without_node_removes_registered_node():
    node = RegisteredNode()
    ros.register_node(node)

    ros.clear_node()

    with pytest.raises(RoboKudoROSNodeMissing):
        ros.get_node()


def test_clear_node_does_not_remove_another_registered_node():
    registered_node = RegisteredNode()
    other_node = RegisteredNode()
    ros.register_node(registered_node)

    ros.clear_node(other_node)

    assert ros.get_node() is registered_node


def test_get_node_raises_when_no_node_is_registered():
    with pytest.raises(RoboKudoROSNodeMissing):
        ros.get_node()
