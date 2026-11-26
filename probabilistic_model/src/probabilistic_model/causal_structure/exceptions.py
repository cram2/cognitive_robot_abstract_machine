"""
What can go wrong reading a causal graph or choosing an adjustment set from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from probabilistic_model.exceptions import ProbabilisticModelException
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from probabilistic_model.causal_structure.graph import GraphEdge, GraphNode
    from probabilistic_model.causal_structure.learning import KnowledgeKind
    from probabilistic_model.causal_structure.identification import (
        AdjustmentSet,
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


@dataclass(kw_only=True)
class NotADirectedEdge(ProbabilisticModelException):
    """
    Raised when an edge that is not directed is asked for its parent or its child.
    """

    edge: GraphEdge
    """
    The edge asked.
    """

    def error_message(self) -> str:
        return f"The edge {self.edge} is not directed, so it has no parent or child."

    def suggest_correction(self) -> str:
        return "Ask only an edge whose is_directed holds for its parent or child."


@dataclass(kw_only=True)
class NotAnEndOfTheEdge(ProbabilisticModelException):
    """
    Raised when an edge is asked about a node at neither of its ends.
    """

    node: GraphNode
    """
    The node asked about.
    """

    edge: GraphEdge
    """
    The edge asked.
    """

    def error_message(self) -> str:
        return f"The node '{self.node.name}' is not an end of the edge {self.edge}."

    def suggest_correction(self) -> str:
        return "Ask an edge about one of its two nodes."


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
    Raised when a graph stands for more directed acyclic graphs than are listed.
    """

    maximum_member_graphs: int
    """
    How many are listed.
    """

    def error_message(self) -> str:
        return (
            f"The graph stands for more than {self.maximum_member_graphs} directed "
            f"acyclic graphs."
        )

    def suggest_correction(self) -> str:
        return (
            "Orient edges with domain causal knowledge, or raise the graph's "
            "maximum_member_graphs."
        )


# %% learning a graph


@dataclass
class TooFewVariablesToScore(ProbabilisticModelException):
    """
    Raised when a graph is to be learned by its score over fewer variables than the
    score is defined for.
    """

    variable_count: int
    """
    How many variables the graph is to be learned over.
    """

    minimum_variable_count: int
    """
    How many the score needs.
    """

    def error_message(self) -> str:
        return (
            f"The graph score needs at least {self.minimum_variable_count} variables "
            f"that vary, but the table has {self.variable_count}."
        )

    def suggest_correction(self) -> str:
        return "Learn the graph of this table with the Peter-Clark algorithm instead."


@dataclass(kw_only=True)
class UnsupportedKnowledgeUnderLatentConfounding(ProbabilisticModelException):
    """
    Raised when knowledge is given to a learner that allows hidden confounders which it
    would turn into causal claims the data does not support.
    """

    kind: KnowledgeKind
    """
    The kind of knowledge given.
    """

    def error_message(self) -> str:
        return (
            f"The learner allows hidden confounders and cannot use {self.kind}: it "
            f"would orient edges as if no hidden variable caused both their ends."
        )

    def suggest_correction(self) -> str:
        return self.kind.correction_under_latent_confounding


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

    declared: AdjustmentSet
    """
    The declared adjustment set.
    """

    reason: InvalidAdjustmentReason
    """
    Why the set fails the adjustment criterion there.
    """

    member_index: int | None
    """
    The position of a member graph the set fails in; ``None`` for a graph that stands
    for no list of member graphs.
    """

    def error_message(self) -> str:
        where = (
            "the graph"
            if self.member_index is None
            else f"member graph {self.member_index}"
        )
        return (
            f"Adjusting for {self.declared.names} does not "
            f"identify the effect of '{self.cause.name}' on '{self.effect.name}': "
            f"{self.reason} in {where}."
        )

    def suggest_correction(self) -> str:
        return "Leave the adjustment set to be chosen from the learned graph."


@dataclass(kw_only=True)
class NoAdjustmentCriterionForGraph(ProbabilisticModelException):
    """
    Raised when an adjustment set is to be chosen from a kind of graph no adjustment
    criterion checks.
    """

    graph_kind: str
    """
    The name of the graph's class.
    """

    def error_message(self) -> str:
        return f"No adjustment criterion checks a {self.graph_kind}."

    def suggest_correction(self) -> str:
        return (
            "Bind a subclass of AdjustmentCriterion to the graph's class, or choose "
            "the set from a CausalGraph or an AncestralGraph."
        )


@dataclass(kw_only=True)
class QuerySpansSeveralParts(ProbabilisticModelException):
    """
    Raised when a causal query over a structure learned one table per collection names
    attributes of two collections, or of two parts of one collection.
    """

    variable_names: list[str]
    """
    The names of the query's cause, effect and declared confounders.
    """

    parts: list[str]
    """
    The parts they reach, each as its collection and position.
    """

    def error_message(self) -> str:
        return (
            f"The variables {self.variable_names} are attributes of the parts "
            f"{self.parts}, but each learned graph holds one part of one collection."
        )

    def suggest_correction(self) -> str:
        return (
            "Ask about the attributes of one part at a time, or learn the structure "
            "with a learner that assumes no hidden confounders."
        )
