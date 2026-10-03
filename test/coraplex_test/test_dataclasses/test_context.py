import logging
from copy import deepcopy

import pytest

from coraplex.datastructures.dataclasses import Context

from ...conftest import SAMPLING_SEED

# %% debug validation


def test_debug_requires_a_ros_node(pr2_apartment_context):
    """
    Debug output is visualized over ROS, so a context constructed in debug mode without
    a node is rejected at construction rather than failing later during execution.
    """
    world, robot, _ = pr2_apartment_context

    with pytest.raises(ValueError):
        Context(world, robot, _debug=True)


def test_debug_raises_the_coraplex_log_level(pr2_apartment_context, rclpy_node):
    """
    Constructing a context in debug mode lowers the package's log level, so debug
    messages are emitted without the caller touching logging.
    """
    world, robot, _ = pr2_apartment_context
    coraplex_logger = logging.getLogger("coraplex")
    previous_level = coraplex_logger.level

    try:
        Context(world, robot, ros_node=rclpy_node, _debug=True)
        assert coraplex_logger.level == logging.DEBUG
    finally:
        coraplex_logger.setLevel(previous_level)


def test_default_context_logs_at_info(pr2_apartment_context):
    """
    Without debug mode the package logs at info level.
    """
    world, robot, _ = pr2_apartment_context
    coraplex_logger = logging.getLogger("coraplex")
    previous_level = coraplex_logger.level

    try:
        context = Context(world, robot)
        assert not context.debug
        assert coraplex_logger.level == logging.INFO
    finally:
        coraplex_logger.setLevel(previous_level)


# %% repeatable location samples

WORLD_FIXTURES_WITH_A_CONTEXT = [
    "pr2_apartment_context",
    "simple_pr2_context",
    "stretch_apartment_context",
    "apartment_world_pr2_copy_with_context",
]
"""
The shared fixtures that hand a test a plan context to run its actions in.
"""


@pytest.mark.parametrize("world_fixture", WORLD_FIXTURES_WITH_A_CONTEXT)
def test_a_shared_fixture_fixes_the_samples_its_context_makes(world_fixture, request):
    """
    A location samples its candidates from a costmap rather than ranking it, so a test
    handed an unseeded context would stand somewhere else every run.
    """
    _, _, context = request.getfixturevalue(world_fixture)

    assert context.sampling_seed == SAMPLING_SEED


# %% copying something that belongs to a run


def test_a_copied_context_is_the_same_context(pr2_apartment_context, rclpy_node):
    """
    A context is the run an object belongs to, holding the world it acts on and the ROS
    node it talks through, so copying an object that refers to it keeps referring to
    that run rather than trying to copy the run itself.
    """
    world, robot, context = pr2_apartment_context
    context.ros_node = rclpy_node

    assert deepcopy(context) is context
