"""
Process-local access to an application-owned ROS node.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krrood.singleton import SingletonMeta
from rclpy.node import Node

from semantic_digital_twin.exceptions import ROSNodeNotRegisteredError

# %% shared node access


@dataclass
class ROSNodeRegistry(metaclass=SingletonMeta):
    """
    Stores the shared ROS node borrowed from this process's lifecycle owner.

    Exactly one node may be registered per process. Consumers may create and use ROS
    entities through that node while its registration is stable. They must not destroy
    the node, change its executor ownership, or shut down its ROS context.

    The lifecycle owner serializes calls to :meth:`register` and :meth:`clear`. It must
    stop users of the node before clearing the registration, and clear it before
    destroying the node. This registry does not initialize ROS, create or spin an
    executor, destroy the node, or shut down its context.
    """

    _node: Node | None = field(default=None, init=False, repr=False)
    """
    The application-owned node currently shared within this process.
    """

    def has_node(self) -> bool:
        """
        Return whether a shared node is registered.

        :return: Whether :meth:`get` can return a node.
        """
        return self._node is not None

    def register(self, node: Node) -> None:
        """
        Register the application-owned node for shared access.

        Replaces an earlier registration without changing either node's lifecycle.

        :param node: The node to share within this process.
        """
        self._node = node

    def get(self) -> Node:
        """
        Return the registered shared node.

        :return: The registered node.
        :raises ROSNodeNotRegisteredError: If no node is registered.
        """
        if self._node is None:
            raise ROSNodeNotRegisteredError()
        return self._node

    def clear(self, expected_node: Node | None = None) -> None:
        """
        Remove the registration without changing the node's lifecycle.

        When an expected node is supplied, the registration is removed only if it still
        refers to that node. This prevents an earlier lifecycle owner from clearing a
        replacement registered later.

        :param expected_node: The node that must be registered, or ``None`` to clear any
            registration.
        """
        if expected_node is None or self._node is expected_node:
            self._node = None
