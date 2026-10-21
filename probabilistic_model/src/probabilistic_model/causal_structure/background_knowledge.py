"""
What is known about a domain's causal structure before any data is seen: which variables
come before which, which edges must or must not be there, and which variables only copy
another.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from probabilistic_model.causal_structure.exceptions import UnknownGraphNode

# %% edges known in advance


@dataclass(frozen=True)
class KnownEdge:
    """
    A directed edge known before any data is seen.
    """

    parent: str
    """
    The name of the cause.
    """

    child: str
    """
    The name of what it causes.
    """


@dataclass(frozen=True)
class DeterministicCopy:
    """
    A variable that is a deterministic function of another, such as a band read off a
    distance.

    Testing it for independence breaks the assumptions the tests rely on, so it is left
    out of the tests and restored as a child of its source.
    """

    copy: str
    """
    The name of the variable that copies.
    """

    source: str
    """
    The name of the variable it copies.
    """


# %% the knowledge


@dataclass
class BackgroundKnowledge:
    """
    Domain knowledge about a causal structure that no data supplies.

    Every entry names a variable by the name the learned graph gives it, which is the
    name a query's parameters give it; a name the learner does not know raises instead
    of being ignored.
    """

    tiers: list[list[str]] = field(default_factory=list)
    """
    The variables in temporal order, earliest first: a variable is never caused by one
    in a later tier.
    """

    required_edges: list[KnownEdge] = field(default_factory=list)
    """
    The edges that must be in the graph.
    """

    forbidden_edges: list[KnownEdge] = field(default_factory=list)
    """
    The edges that must not be in the graph.
    """

    deterministic_copies: list[DeterministicCopy] = field(default_factory=list)
    """
    The variables that only copy another.
    """

    def named_variables(self) -> set[str]:
        """
        :return: Every variable name the knowledge mentions.
        """
        return (
            set(itertools.chain.from_iterable(self.tiers))
            | {
                name
                for edge in self.required_edges + self.forbidden_edges
                for name in (edge.parent, edge.child)
            }
            | {
                name
                for copy in self.deterministic_copies
                for name in (copy.copy, copy.source)
            }
        )

    def check_names(self, known_names: list[str]) -> None:
        """
        :param known_names: The names of the variables the graph is learned over.
        :raises UnknownGraphNode: If the knowledge names a variable not among them.
        """
        for name in sorted(self.named_variables() - set(known_names)):
            raise UnknownGraphNode(name, known_names)

    def restricted_to(self, names: list[str]) -> BackgroundKnowledge:
        """
        :param names: The names of the variables one table is learned over.
        :return: The knowledge about those variables alone.
        """
        kept = set(names)
        return BackgroundKnowledge(
            tiers=[
                [name for name in tier if name in kept]
                for tier in self.tiers
                if any(name in kept for name in tier)
            ],
            required_edges=[
                edge
                for edge in self.required_edges
                if {edge.parent, edge.child} <= kept
            ],
            forbidden_edges=[
                edge
                for edge in self.forbidden_edges
                if {edge.parent, edge.child} <= kept
            ],
            deterministic_copies=[
                copy
                for copy in self.deterministic_copies
                if {copy.copy, copy.source} <= kept
            ],
        )

    def merged_with(self, other: BackgroundKnowledge) -> BackgroundKnowledge:
        """
        :param other: More knowledge.
        :return: Both, with the other's tiers placed after this knowledge's.
        """
        return BackgroundKnowledge(
            tiers=self.tiers + other.tiers,
            required_edges=self.required_edges + other.required_edges,
            forbidden_edges=self.forbidden_edges + other.forbidden_edges,
            deterministic_copies=self.deterministic_copies + other.deterministic_copies,
        )
