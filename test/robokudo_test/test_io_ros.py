from robokudo.exceptions import RoboKudoROSNodeMissing
from robokudo.io import ros

# %% node registry


class RegisteredNode:
    """
    Mimic a ROS node for registry lifecycle tests.
    """


def setup_function():
    ros._rk_node = None


def teardown_function():
    ros._rk_node = None


def test_register_node_makes_node_available():
    node = RegisteredNode()

    ros.register_node(node)

    assert ros.get_node() is node


def test_clear_node_removes_registered_node():
    node = RegisteredNode()
    ros.register_node(node)

    ros.clear_node(node)

    assert ros._rk_node is None


def test_clear_node_does_not_remove_another_registered_node():
    registered_node = RegisteredNode()
    other_node = RegisteredNode()
    ros.register_node(registered_node)

    ros.clear_node(other_node)

    assert ros.get_node() is registered_node


def test_get_node_raises_when_no_node_is_registered():
    try:
        ros.get_node()
    except RoboKudoROSNodeMissing as error:
        assert "RoboKudo ROS node is not initialized" in str(error)
    else:
        raise AssertionError("get_node() must fail without a registered node")
