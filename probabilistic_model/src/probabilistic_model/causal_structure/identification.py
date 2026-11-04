"""
Choosing the set of variables to adjust for, so that adjusting for it gives the effect
of a cause on an effect, from a causal graph: the adjustment criterion of Shpitser,
VanderWeele and Robins (2010) and Perković et al. (2015), checked in every directed
acyclic graph the causal graph stands for.
"""

from __future__ import annotations

import itertools
from abc import ABC
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property

import pandas as pd

from probabilistic_model.causal_structure.discretization import Discretizer
from probabilistic_model.causal_structure.exceptions import (
    InvalidAdjustmentSet,
)
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    GraphNode,
    MemberGraph,
)

# %% the adjustment criterion in one graph


def name_of(node: GraphNode) -> str:
    """
    :param node: A node.
    :return: Its name, by which nodes are ordered wherever an order is needed.
    """
    return node.name


class InvalidAdjustmentReason(StrEnum):
    """
    Why a set of variables fails the adjustment criterion in a directed acyclic graph.
    """

    CONTAINS_CAUSE_OR_EFFECT = "contains_cause_or_effect"
    """
    The set holds the cause or the effect itself.
    """

    CONTAINS_FORBIDDEN_NODE = "contains_forbidden_node"
    """
    The set holds a descendant of a node on a causal path from the cause to the effect.
    """

    LEAVES_NON_CAUSAL_PATH_OPEN = "leaves_non_causal_path_open"
    """
    The set leaves a path from the cause to the effect open that is not causal.
    """


@dataclass
class AdjustmentCheck:
    """
    The adjustment criterion for one cause and one effect in one directed acyclic graph.
    """

    member_graph: MemberGraph
    """
    The directed acyclic graph.
    """

    cause: GraphNode
    """
    The cause.
    """

    effect: GraphNode
    """
    The effect.
    """

    @cached_property
    def causal_path_nodes(self) -> set[GraphNode]:
        """
        :return: The nodes other than the cause on a directed path from the cause to
            the effect, the effect included if there is such a path.
        """
        return self.member_graph.descendants(self.cause) & (
            self.member_graph.ancestors(self.effect) | {self.effect}
        )

    @cached_property
    def forbidden_nodes(self) -> set[GraphNode]:
        """
        :return: The cause, and every node on a causal path with its descendants.
        """
        return {self.cause}.union(
            *(
                self.member_graph.descendants(node) | {node}
                for node in self.causal_path_nodes
            )
        )

    @cached_property
    def proper_backdoor_graph(self) -> MemberGraph:
        """
        :return: The graph without the edges from the cause into a node on a causal
            path, which leaves only the paths that are not causal to connect the two.
        """
        return self.member_graph.without_edges(
            (self.cause, node)
            for node in self.causal_path_nodes
            if self.member_graph.has_edge(self.cause, node)
        )

    def failure_reason(
        self, adjustment: frozenset[GraphNode]
    ) -> InvalidAdjustmentReason | None:
        """
        :param adjustment: The nodes to adjust for.
        :return: The first condition of the adjustment criterion the set fails, or
            ``None`` if it meets them all.
        """
        if {self.cause, self.effect} & adjustment:
            return InvalidAdjustmentReason.CONTAINS_CAUSE_OR_EFFECT
        if self.forbidden_nodes & adjustment:
            return InvalidAdjustmentReason.CONTAINS_FORBIDDEN_NODE
        if not self.proper_backdoor_graph.is_directionally_separated(
            {self.cause}, {self.effect}, set(adjustment)
        ):
            return InvalidAdjustmentReason.LEAVES_NON_CAUSAL_PATH_OPEN
        return None


# %% what a graph says about an effect


class Identification(ABC):
    """
    What a causal graph says about the effect of a cause on an effect.
    """


@dataclass
class IdentifiedEffect(Identification):
    """
    The effect is identified by adjusting for one set of variables.
    """

    adjustment: frozenset[GraphNode]
    """
    The nodes to adjust for.
    """


@dataclass
class PartiallyIdentifiedEffect(Identification):
    """
    The graphs the causal graph stands for disagree on the cause's parents, so the
    effect is one of several, one per parent set (Maathuis, Kalisch and Bühlmann, 2009).
    """

    candidate_adjustments: list[frozenset[GraphNode]]
    """
    Each distinct parent set of the cause that does not hold the effect, the smallest
    first.
    """

    includes_member_without_causal_path: bool
    """
    Whether the effect is a parent of the cause in some member graph, where the cause
    has no effect on it at all.
    """


class UnidentifiedReason(StrEnum):
    """
    Why a causal graph leaves an effect unidentified.
    """

    NO_VALID_SET_AMONG_ELIGIBLE = "no_valid_set_among_eligible"
    """
    The graphs agree on the cause's parents, but no set of variables the model can
    adjust for, small enough to be tried, meets the criterion in all of them.
    """

    PARENT_SET_NOT_ELIGIBLE = "parent_set_not_eligible"
    """
    The graphs disagree on the cause's parents, and one of the parent sets holds a
    variable the model cannot adjust for.
    """

    NO_MEMBER_GRAPH = "no_member_graph"
    """
    No directed acyclic graph fits the graph: its undirected edges cannot be oriented
    without a cycle or a collider it does not have.
    """


@dataclass
class UnidentifiedEffect(Identification):
    """
    The effect is not identified.
    """

    reason: UnidentifiedReason
    """
    Why.
    """


# %% choosing an adjustment set


@dataclass
class AdjustmentSelector:
    """
    Chooses the set of variables to adjust for from a causal graph, and checks a set
    declared for it.
    """

    graph: CausalGraph
    """
    The causal graph.
    """

    maximum_adjustment_size: int = 3
    """
    How many variables an adjustment set may hold.
    """

    minimum_rows_per_cell: int = 10
    """
    How many training rows a combination of the cause's and the adjustment set's
    values needs to count as supported.
    """

    discretizer: Discretizer = field(default_factory=Discretizer)
    """
    How the training rows are cut into combinations of values.
    """

    def identify(
        self,
        cause: GraphNode,
        effect: GraphNode,
        eligible: set[GraphNode],
        training_rows: pd.DataFrame | None = None,
    ) -> Identification:
        """
        :param cause: The cause.
        :param effect: The effect.
        :param eligible: The nodes the model can adjust for.
        :param training_rows: The rows the model was fitted on, whose support ranks the
            valid adjustment sets.
        :return: The best valid adjustment set; else, if the member graphs disagree on
            the cause's parents, one candidate per parent set; else a refusal.
        """
        if not self.graph.has_member_graphs:
            return UnidentifiedEffect(UnidentifiedReason.NO_MEMBER_GRAPH)
        checks = self.checks_for(cause, effect)
        forbidden = set().union(*(check.forbidden_nodes for check in checks))
        valid = [
            adjustment
            for adjustment in self.candidate_adjustments(
                sorted(eligible - {cause, effect} - forbidden, key=name_of)
            )
            if all(check.failure_reason(adjustment) is None for check in checks)
        ]
        if valid:
            return IdentifiedEffect(
                min(
                    valid,
                    key=lambda adjustment: (
                        self.under_supported_cells(cause, adjustment, training_rows),
                        len(adjustment),
                        sorted(node.name for node in adjustment),
                    ),
                )
            )
        return self.parent_set_candidates(cause, effect, eligible, checks)

    def check_declared(
        self, cause: GraphNode, effect: GraphNode, declared: frozenset[GraphNode]
    ) -> IdentifiedEffect:
        """
        :param cause: The cause.
        :param effect: The effect.
        :param declared: The nodes declared to adjust for.
        :return: The declared set, if it meets the criterion in every member graph.
        :raises InvalidAdjustmentSet: If it fails in one.
        """
        for index, check in enumerate(self.checks_for(cause, effect)):
            reason = check.failure_reason(declared)
            if reason is not None:
                raise InvalidAdjustmentSet(cause, effect, declared, reason, index)
        return IdentifiedEffect(declared)

    def checks_for(self, cause: GraphNode, effect: GraphNode) -> list[AdjustmentCheck]:
        """
        :param cause: The cause.
        :param effect: The effect.
        :return: The adjustment criterion in every member graph.
        """
        return [
            AdjustmentCheck(member, cause, effect)
            for member in self.graph.member_graphs()
        ]

    def candidate_adjustments(
        self, pool: list[GraphNode]
    ) -> list[frozenset[GraphNode]]:
        """
        :param pool: The nodes that may be adjusted for, sorted by name.
        :return: Every set of at most :attr:`maximum_adjustment_size` of them, the
            smallest first, sets of one size in the order of their sorted names.
        """
        return [
            frozenset(adjustment)
            for size in range(self.maximum_adjustment_size + 1)
            for adjustment in itertools.combinations(pool, size)
        ]

    def under_supported_cells(
        self,
        cause: GraphNode,
        adjustment: frozenset[GraphNode],
        training_rows: pd.DataFrame | None,
    ) -> int:
        """
        :param cause: The cause.
        :param adjustment: The nodes to adjust for.
        :param training_rows: The rows the model was fitted on.
        :return: How many combinations of the cause's and the set's values hold some
            training rows but fewer than :attr:`minimum_rows_per_cell`; nothing if there
            are no rows, or they lack a column for one of the variables.
        """
        columns = [cause.name, *sorted(node.name for node in adjustment)]
        if training_rows is None or not set(columns) <= set(training_rows.columns):
            return 0
        cell_sizes = (
            self.discretizer.fit_transform(training_rows[columns])
            .groupby(columns)
            .size()
        )
        return int(((cell_sizes > 0) & (cell_sizes < self.minimum_rows_per_cell)).sum())

    def parent_set_candidates(
        self,
        cause: GraphNode,
        effect: GraphNode,
        eligible: set[GraphNode],
        checks: list[AdjustmentCheck],
    ) -> Identification:
        """
        :param cause: The cause.
        :param effect: The effect.
        :param eligible: The nodes the model can adjust for.
        :param checks: The adjustment criterion in every member graph.
        :return: One candidate per distinct parent set of the cause if the member
            graphs disagree on it, else a refusal.
        """
        parent_sets = {
            frozenset(check.member_graph.predecessors(cause)) for check in checks
        }
        if len(parent_sets) == 1:
            return UnidentifiedEffect(UnidentifiedReason.NO_VALID_SET_AMONG_ELIGIBLE)
        candidates = sorted(
            (parents for parents in parent_sets if effect not in parents),
            key=lambda parents: (len(parents), sorted(map(name_of, parents))),
        )
        if not all(parents <= eligible for parents in candidates):
            return UnidentifiedEffect(UnidentifiedReason.PARENT_SET_NOT_ELIGIBLE)
        return PartiallyIdentifiedEffect(
            candidate_adjustments=candidates,
            includes_member_without_causal_path=any(
                effect in parents for parents in parent_sets
            ),
        )
