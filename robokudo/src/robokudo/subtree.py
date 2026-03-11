from __future__ import annotations

from typing_extensions import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from py_trees.behaviour import Behaviour


class SubtreeInterface(Protocol):
    """
    Interface for Subtrees to implement. This will be loaded by the RK runner.
    Please do NOT instantiate this class directly.
    The idea is to define a subclass of this interface which is providing an implementation of your desired behaviour tree.
    See src/robokudo/descriptors/analysis_engines for examples.
    """

    def name(self) -> str:
        """Return the name of this subtree."""
        pass

    def implementation(self) -> Behaviour:
        """Return the subtree implementation."""
        pass
