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


# %% segmenting what happens while a plan runs


def test_a_run_is_watched_unless_it_says_otherwise(immutable_model_world):
    """
    Watching a run costs a detector tick beside every step of the plan, so a run that
    only has to perform its plan can say so.
    """
    world, robot, _ = immutable_model_world

    assert Context(world, robot).segment_events
    assert not Context(world, robot, segment_events=False).segment_events
