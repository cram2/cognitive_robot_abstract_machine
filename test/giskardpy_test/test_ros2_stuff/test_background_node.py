"""
Tests for Giskard's ownership of the process-local ROS node lifecycle.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from semantic_digital_twin.adapters.ros import background_node
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry


@pytest.fixture()
def initialized_background_node() -> Iterator[ROSNodeRegistry]:
    """
    Start Giskard's ROS runtime and stop it after the test.
    """
    registry = ROSNodeRegistry()
    registry.clear()
    background_node.init_node("giskard_registry_test")
    yield registry
    background_node.shutdown()
    registry.clear()


def test_init_node_registers_giskards_node(
    initialized_background_node: ROSNodeRegistry,
) -> None:
    assert background_node.get_node() is initialized_background_node.get()


def test_init_node_does_not_replace_an_initialized_node(
    initialized_background_node: ROSNodeRegistry,
) -> None:
    initialized_node = initialized_background_node.get()

    background_node.init_node("replacement")

    assert initialized_background_node.get() is initialized_node


def test_shutdown_clears_the_node_registration() -> None:
    registry = ROSNodeRegistry()
    registry.clear()
    background_node.init_node("giskard_registry_test")

    background_node.shutdown()

    assert not registry.has_node()
