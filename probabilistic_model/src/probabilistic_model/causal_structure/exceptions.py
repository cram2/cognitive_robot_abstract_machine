"""
What can go wrong reading a causal graph or choosing an adjustment set from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from probabilistic_model.exceptions import ProbabilisticModelException
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from probabilistic_model.causal_structure.graph import GraphNode
    from probabilistic_model.causal_structure.identification import (
        InvalidAdjustmentReason,
    )

# %% reading a graph


@dataclass
class UnknownGraphNode(ProbabilisticModelException):
    """
    Raised when a graph is asked for a node it does not have.
    """

    name: str
    """
    The name asked for.
    """

    known_names: list[str]
    """
    The names of the graph's nodes.
    """

    def error_message(self) -> str:
        return (
            f"The graph has no node named '{self.name}'. Its nodes: {self.known_names}."
        )

    def suggest_correction(self) -> str:
        return "Name a node by the name of its variable."


@dataclass
class NotAPartAttribute(ProbabilisticModelException):
    """
    Raised when a node that does not stand for an attribute of a collection's parts is
    asked for the variable of one part.
    """

    node: GraphNode
    """
    The node asked.
    """

    def error_message(self) -> str:
        return (
            f"The node '{self.node.variable.name}' is a {self.node.level} node, not an "
            f"attribute of a collection's parts."
        )

    def suggest_correction(self) -> str:
        return "Ask a part attribute node for the variable of one part."


# %% the graphs an equivalence class stands for


@dataclass
class CyclicGraph(ProbabilisticModelException):
    """
    Raised when a graph whose edges are all directed has a directed cycle.
    """

    cycle: list[GraphNode]
    """
    The nodes on one cycle, in order.
    """

    def error_message(self) -> str:
        return f"The graph has the directed cycle {' -> '.join(node.name for node in self.cycle)}."

    def suggest_correction(self) -> str:
        return "Reverse or remove an edge on the cycle."


@dataclass
class InconsistentGraph(ProbabilisticModelException):
    """
    Raised when no way of orienting a graph's undirected edges gives a directed acyclic
    graph with the graph's unshielded colliders.
    """

    undirected_edge_count: int
    """
    How many undirected edges were oriented.
    """

    def error_message(self) -> str:
        return (
            f"None of the orientations of the graph's {self.undirected_edge_count} "
            f"undirected edges is acyclic and keeps its colliders."
        )

    def suggest_correction(self) -> str:
        return "Check the directed edges the background knowledge requires."


@dataclass
class EquivalenceClassTooLarge(ProbabilisticModelException):
    """
    Raised when a graph has more undirected edges than the graphs it stands for can be
    listed over.
    """

    undirected_edge_count: int
    """
    How many undirected edges the graph has.
    """

    maximum_undirected_edges: int
    """
    How many it may have.
    """

    def error_message(self) -> str:
        return (
            f"The graph has {self.undirected_edge_count} undirected edges, more than "
            f"the {self.maximum_undirected_edges} its member graphs are listed over."
        )

    def suggest_correction(self) -> str:
        return (
            "Orient edges with background knowledge, or raise the graph's "
            "maximum_undirected_edges."
        )


# %% adjustment sets


@dataclass
class InvalidAdjustmentSet(ProbabilisticModelException):
    """
    Raised when a declared adjustment set does not identify the effect of the cause on
    the effect in a graph the causal graph stands for.
    """

    cause: GraphNode
    """
    The cause.
    """

    effect: GraphNode
    """
    The effect.
    """

    declared: frozenset[GraphNode]
    """
    The declared adjustment set.
    """

    reason: InvalidAdjustmentReason
    """
    Why the set fails the adjustment criterion there.
    """

    member_index: int
    """
    The position of a member graph the set fails in.
    """

    def error_message(self) -> str:
        return (
            f"Adjusting for {sorted(node.name for node in self.declared)} does not "
            f"identify the effect of '{self.cause.name}' on '{self.effect.name}': "
            f"{self.reason} in member graph "
            f"{self.member_index}."
        )

    def suggest_correction(self) -> str:
        return "Leave the adjustment set to be chosen from the learned graph."
