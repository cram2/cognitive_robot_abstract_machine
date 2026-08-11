import logging

import pytest

from coraplex.datastructures.dataclasses import Context

# %% debug validation


def test_debug_requires_a_ros_node(immutable_model_world):
    """
    Debug output is visualized over ROS, so a context constructed in debug mode without
    a node is rejected at construction rather than failing later during execution.
    """
    world, robot, _ = immutable_model_world

    with pytest.raises(ValueError):
        Context(world, robot, _debug=True)


def test_debug_raises_the_coraplex_log_level(immutable_model_world, rclpy_node):
    """
    Constructing a context in debug mode lowers the package's log level, so debug
    messages are emitted without the caller touching logging.
    """
    world, robot, _ = immutable_model_world
    coraplex_logger = logging.getLogger("coraplex")
    previous_level = coraplex_logger.level

    try:
        Context(world, robot, ros_node=rclpy_node, _debug=True)
        assert coraplex_logger.level == logging.DEBUG
    finally:
        coraplex_logger.setLevel(previous_level)


def test_default_context_logs_at_info(immutable_model_world):
    """
    Without debug mode the package logs at info level.
    """
    world, robot, _ = immutable_model_world
    coraplex_logger = logging.getLogger("coraplex")
    previous_level = coraplex_logger.level

    try:
        context = Context(world, robot)
        assert not context.debug
        assert coraplex_logger.level == logging.INFO
    finally:
        coraplex_logger.setLevel(previous_level)
