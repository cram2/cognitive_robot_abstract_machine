import logging

import pytest

from coraplex.datastructures.dataclasses import Context

from ...conftest import SAMPLING_SEED

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


# %% repeatable location draws

WORLD_FIXTURES_WITH_A_CONTEXT = [
    "mutable_model_world",
    "immutable_model_world",
    "mutable_simple_pr2_world",
    "immutable_simple_pr2_world",
    "apartment_world_pr2_copy_with_context",
]
"""
The shared fixtures that hand a test a plan context to run its actions in.
"""


@pytest.mark.parametrize("world_fixture", WORLD_FIXTURES_WITH_A_CONTEXT)
def test_a_shared_fixture_fixes_the_draws_its_context_makes(world_fixture, request):
    """
    A location draws its candidates from a costmap rather than ranking it, so a test
    handed an unseeded context would stand somewhere else every run.
    """
    _, _, context = request.getfixturevalue(world_fixture)

    assert context.sampling_seed == SAMPLING_SEED
