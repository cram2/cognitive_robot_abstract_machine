from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.exceptions import ROSNodeNotRegisteredError


@dataclass
class LifecycleOwnedNode:
    """
    Records whether something outside its lifecycle owner tries to destroy it.
    """

    destruction_count: int = 0
    """
    The number of times destruction was requested.
    """

    def destroy_node(self) -> None:
        """
        Record a request to destroy this node.
        """
        self.destruction_count += 1


@pytest.fixture
def registry() -> Iterator[ROSNodeRegistry]:
    """
    Provide an empty process registry and release its singleton after the test.
    """
    ROSNodeRegistry.clear_instance()
    registry = ROSNodeRegistry()
    yield registry
    ROSNodeRegistry.clear_instance()


# %% process-wide access


def test_registry_is_a_process_singleton(registry: ROSNodeRegistry) -> None:
    assert ROSNodeRegistry() is registry


def test_new_registry_has_no_node(registry: ROSNodeRegistry) -> None:
    assert not registry.has_node()


def test_get_without_registration_raises_specific_error(
    registry: ROSNodeRegistry,
) -> None:
    with pytest.raises(ROSNodeNotRegisteredError):
        registry.get()


# %% registration


def test_registered_node_is_returned(registry: ROSNodeRegistry) -> None:
    node = LifecycleOwnedNode()

    registry.register(node)

    assert registry.get() is node


def test_register_replaces_the_previous_node(registry: ROSNodeRegistry) -> None:
    previous_node = LifecycleOwnedNode()
    current_node = LifecycleOwnedNode()
    registry.register(previous_node)

    registry.register(current_node)

    assert registry.get() is current_node


# %% clearing a registration


def test_clear_without_expected_node_removes_registration(
    registry: ROSNodeRegistry,
) -> None:
    registry.register(LifecycleOwnedNode())

    registry.clear()

    assert not registry.has_node()


def test_clear_matching_node_removes_registration(registry: ROSNodeRegistry) -> None:
    node = LifecycleOwnedNode()
    registry.register(node)

    registry.clear(node)

    assert not registry.has_node()


def test_clear_previous_node_preserves_replacement(registry: ROSNodeRegistry) -> None:
    previous_node = LifecycleOwnedNode()
    current_node = LifecycleOwnedNode()
    registry.register(previous_node)
    registry.register(current_node)

    registry.clear(previous_node)

    assert registry.get() is current_node


def test_clear_does_not_destroy_node(registry: ROSNodeRegistry) -> None:
    node = LifecycleOwnedNode()
    registry.register(node)

    registry.clear(node)

    assert node.destruction_count == 0
