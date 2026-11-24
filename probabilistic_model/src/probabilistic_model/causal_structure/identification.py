"""
Choosing the set of variables to adjust for from a causal graph, so that adjusting for
it gives the effect of a cause on an effect: the adjustment criterion of Perković et al.
(2015) in every directed acyclic graph a causal graph stands for, or their generalized
adjustment criterion (2018) in an ancestral graph.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property

import inspect

import pandas as pd
from typing_extensions import Generic

from krrood.patterns.classproperty import classproperty
from krrood.patterns.interface import Interface
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import recursive_subclasses

from probabilistic_model.causal_structure.ancestral_graph import AncestralGraph
from probabilistic_model.causal_structure.discretization import Discretizer
from probabilistic_model.causal_structure.exceptions import (
    InvalidAdjustmentSet,
    NoAdjustmentCriterionForGraph,
)
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    EdgeMarkedGraph,
    GraphEdge,
    GraphNode,
    MarkedGraphType,
    MemberGraph,
)

# %% the adjustment criterion in one graph


@dataclass(frozen=True)
class AdjustmentSet:
    """
    A set of nodes to adjust for.
    """

    nodes: frozenset[GraphNode] = frozenset()
    """
    The nodes.
    """

    @property
    def names(self) -> list[str]:
        """
        :return: The names of the nodes, sorted.
        """
        return sorted(node.name for node in self.nodes)

    def __lt__(self, other: AdjustmentSet) -> bool:
        """
        Order smaller sets first, and sets of one size by their sorted names.
        """
        return (len(self.nodes), self.names) < (len(other.nodes), other.names)


class InvalidAdjustmentReason(StrEnum):
    """
    Why a set of variables fails the adjustment criterion in a causal graph.
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

    NOT_AMENABLE = "not_amenable"
    """
    An ancestral graph leaves open whether a hidden variable causes the cause and a
    node on a possibly causal path, so no set identifies the effect.
    """

    SELECTION_BIAS_EDGE = "selection_bias_edge"
    """
    An ancestral graph has an edge that says the rows were selected, so no set
    identifies the effect.
    """

    NOT_ANCESTRAL = "not_ancestral"
    """
    An ancestral graph stands for no maximal ancestral graph, so no set identifies the
    effect.
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
        self, adjustment: AdjustmentSet
    ) -> InvalidAdjustmentReason | None:
        """
        :param adjustment: The nodes to adjust for.
        :return: The first condition of the adjustment criterion the set fails, or
            ``None`` if it meets them all.
        """
        if {self.cause, self.effect} & adjustment.nodes:
            return InvalidAdjustmentReason.CONTAINS_CAUSE_OR_EFFECT
        if self.forbidden_nodes & adjustment.nodes:
            return InvalidAdjustmentReason.CONTAINS_FORBIDDEN_NODE
        if not self.proper_backdoor_graph.is_directionally_separated(
            {self.cause}, {self.effect}, set(adjustment.nodes)
        ):
            return InvalidAdjustmentReason.LEAVES_NON_CAUSAL_PATH_OPEN
        return None


# %% what a graph says about an effect


class Identification(ABC):
    """
    What a causal graph says about the effect of a cause on an effect.
    """

    @property
    @abstractmethod
    def candidate_adjustments(self) -> list[AdjustmentSet]:
        """
        :return: Every set the effect may be identified by adjusting for; none if the
            effect is not identified.
        """

    @property
    def is_identified(self) -> bool:
        """
        :return: Whether a single set identifies the effect.
        """
        return False

    @property
    def includes_no_effect(self) -> bool:
        """
        :return: Whether, in some graph the causal graph stands for, the cause has no
            effect on the effect at all.
        """
        return False

    @property
    def correction(self) -> str | None:
        """
        :return: What to change so that the graph identifies the effect, where the
            identification itself says it; ``None`` otherwise.
        """
        return None


@dataclass
class IdentifiedEffect(Identification):
    """
    The effect is identified by adjusting for one set of variables.
    """

    adjustment: AdjustmentSet
    """
    The nodes to adjust for.
    """

    @property
    def candidate_adjustments(self) -> list[AdjustmentSet]:
        return [self.adjustment]

    @property
    def is_identified(self) -> bool:
        return True


@dataclass
class PartiallyIdentifiedEffect(Identification):
    """
    The graphs the causal graph stands for disagree on the cause's parents, so the
    effect is one of several, one per parent set (Maathuis, Kalisch and Bühlmann, 2009).
    """

    parent_sets: list[AdjustmentSet]
    """
    Each distinct parent set of the cause that does not hold the effect, the smallest
    first.
    """

    includes_member_without_causal_path: bool
    """
    Whether the effect is a parent of the cause in some member graph, where the cause
    has no effect on it at all.
    """

    @property
    def candidate_adjustments(self) -> list[AdjustmentSet]:
        return self.parent_sets

    @property
    def includes_no_effect(self) -> bool:
        return self.includes_member_without_causal_path


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

    NOT_AMENABLE = "not_amenable"
    """
    The ancestral graph leaves open whether a hidden variable causes both the cause
    and a node on a possibly causal path from it, which no set can adjust for.
    """

    SELECTION_BIAS_EDGE = "selection_bias_edge"
    """
    The ancestral graph has an edge that says the rows were selected by a variable
    both its ends cause, which the adjustment criterion does not cover.
    """

    NOT_ANCESTRAL = "not_ancestral"
    """
    The ancestral graph stands for no maximal ancestral graph: its directed edges form
    a cycle, a bidirected edge joins a node to its ancestor, or its ``o–o`` edges
    cannot be oriented without an unshielded collider.
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

    unsettled_edge: GraphEdge | None = None
    """
    For :attr:`UnidentifiedReason.NOT_AMENABLE`, the edge out of the cause that may
    hide a confounder; ``None`` otherwise.
    """

    @property
    def candidate_adjustments(self) -> list[AdjustmentSet]:
        return []

    @property
    def correction(self) -> str | None:
        """
        :return: For an edge that may hide a confounder, to record that confounder;
            ``None`` otherwise.
        """
        if self.unsettled_edge is None:
            return None
        return (
            f"The edge {self.unsettled_edge} out of the cause may hide a confounder of "
            f"it and the effect, which no marked set can adjust for. Record that "
            f"confounder as a variable and learn the structure again."
        )


# %% how a kind of graph checks an adjustment set


@dataclass(frozen=True, kw_only=True)
class AdjustmentFailure:
    """
    Why an adjustment set fails the criterion of a graph.
    """

    reason: InvalidAdjustmentReason
    """
    The first condition the set fails.
    """

    member_index: int | None = None
    """
    The position of a member graph the set fails in; ``None`` for a graph that stands
    for no list of member graphs.
    """


@dataclass(kw_only=True)
class AdjustmentCriterion(Generic[MarkedGraphType], SubClassSafeGeneric, ABC):
    """
    The adjustment criterion of one kind of causal graph, for one cause and one effect.

    A subclass binds the kind of graph it checks.
    """

    graph: MarkedGraphType
    """
    The causal graph.
    """

    cause: GraphNode
    """
    The cause.
    """

    effect: GraphNode
    """
    The effect.
    """

    @classproperty
    def graph_class(cls) -> type[EdgeMarkedGraph]:
        """
        :return: The kind of graph the criterion checks, as the subclass binds it.
        """
        return cls.get_type_of_generic_parameter(parameter=MarkedGraphType)

    @classmethod
    def from_graph(
        cls, *, graph: EdgeMarkedGraph, cause: GraphNode, effect: GraphNode
    ) -> AdjustmentCriterion:
        """
        :param graph: A causal graph.
        :param cause: The cause.
        :param effect: The effect.
        :return: The criterion of the graph's kind.
        :raises NoAdjustmentCriterionForGraph: If no criterion checks that kind.
        """
        for criterion in recursive_subclasses(AdjustmentCriterion):
            if not inspect.isabstract(criterion) and isinstance(
                graph, criterion.graph_class
            ):
                return criterion(graph=graph, cause=cause, effect=effect)
        raise NoAdjustmentCriterionForGraph(graph_kind=type(graph).__name__)

    @property
    @abstractmethod
    def precondition_failure(self) -> UnidentifiedEffect | None:
        """
        :return: Why the graph identifies the effect by no set at all; ``None`` if it
            may identify it by some set.
        """

    @property
    @abstractmethod
    def forbidden_nodes(self) -> set[GraphNode]:
        """
        :return: The cause, and every node no valid set may hold.
        """

    @abstractmethod
    def failure_of(self, *, adjustment: AdjustmentSet) -> AdjustmentFailure | None:
        """
        :param adjustment: The nodes to adjust for.
        :return: Why the set fails the criterion; ``None`` if it meets it.
        """

    @abstractmethod
    def identification_without_valid_set(
        self, *, eligible: set[GraphNode]
    ) -> Identification:
        """
        :param eligible: The nodes the model can adjust for.
        :return: What the graph says about the effect when no eligible set is valid.
        """


@dataclass(kw_only=True)
class MemberGraphAdjustmentCriterion(AdjustmentCriterion[CausalGraph]):
    """
    The adjustment criterion in every directed acyclic graph a causal graph stands for.
    """

    @cached_property
    def checks(self) -> list[AdjustmentCheck]:
        """
        :return: The adjustment criterion in every member graph.
        """
        return [
            AdjustmentCheck(member, self.cause, self.effect)
            for member in self.graph.member_graphs()
        ]

    @property
    def precondition_failure(self) -> UnidentifiedEffect | None:
        if not self.graph.has_member_graphs:
            return UnidentifiedEffect(UnidentifiedReason.NO_MEMBER_GRAPH)
        return None

    @cached_property
    def forbidden_nodes(self) -> set[GraphNode]:
        return set().union(*(check.forbidden_nodes for check in self.checks))

    def failure_of(self, *, adjustment: AdjustmentSet) -> AdjustmentFailure | None:
        for index, check in enumerate(self.checks):
            reason = check.failure_reason(adjustment)
            if reason is not None:
                return AdjustmentFailure(reason=reason, member_index=index)
        return None

    def identification_without_valid_set(
        self, *, eligible: set[GraphNode]
    ) -> Identification:
        """
        :return: One candidate per distinct parent set of the cause if the member
            graphs disagree on it, else a refusal.
        """
        parent_sets = {
            frozenset(check.member_graph.predecessors(self.cause))
            for check in self.checks
        }
        if len(parent_sets) == 1:
            return UnidentifiedEffect(UnidentifiedReason.NO_VALID_SET_AMONG_ELIGIBLE)
        candidates = sorted(
            AdjustmentSet(parents)
            for parents in parent_sets
            if self.effect not in parents
        )
        if not all(candidate.nodes <= eligible for candidate in candidates):
            return UnidentifiedEffect(UnidentifiedReason.PARENT_SET_NOT_ELIGIBLE)
        return PartiallyIdentifiedEffect(
            parent_sets=candidates,
            includes_member_without_causal_path=any(
                self.effect in parents for parents in parent_sets
            ),
        )


@dataclass(kw_only=True)
class GeneralizedAdjustmentCriterion(AdjustmentCriterion[AncestralGraph]):
    """
    The generalized adjustment criterion in an ancestral graph, which holds in every
    maximal ancestral graph it stands for, whatever hidden variables they leave out.
    """

    @cached_property
    def possibly_causal_paths(self) -> list[list[GraphNode]]:
        """
        :return: Every possibly directed path from the cause to the effect.
        """
        return self.graph.possibly_causal_paths(cause=self.cause, effect=self.effect)

    @cached_property
    def precondition_failure(self) -> UnidentifiedEffect | None:
        if self.graph.has_selection_bias_edge:
            return UnidentifiedEffect(UnidentifiedReason.SELECTION_BIAS_EDGE)
        if self.graph.representative_maximal_ancestral_graph is None:
            return UnidentifiedEffect(UnidentifiedReason.NOT_ANCESTRAL)
        for path in self.possibly_causal_paths:
            edge = self.graph.edge_between(path[0], path[1])
            if not (
                self.graph.is_parent_of(parent=self.cause, child=path[1])
                and self.graph.is_visible(edge=edge)
            ):
                return UnidentifiedEffect(
                    UnidentifiedReason.NOT_AMENABLE, unsettled_edge=edge
                )
        return None

    @cached_property
    def forbidden_nodes(self) -> set[GraphNode]:
        return self.graph.forbidden_nodes(
            possibly_causal_paths=self.possibly_causal_paths
        ) | {self.cause}

    @cached_property
    def back_door_graph(self) -> AncestralGraph:
        """
        :return: The graph's representative maximal ancestral graph without the edges
            from the cause into a node on a causal path to the effect.
        """
        return self.graph.representative_maximal_ancestral_graph.without_causal_first_edges(
            cause=self.cause, effect=self.effect
        )

    def failure_of(self, *, adjustment: AdjustmentSet) -> AdjustmentFailure | None:
        """
        :return: Why the set fails the criterion, or, by the member of the same name,
            why the graph identifies the effect by no set; ``None`` if the set meets
            the criterion.
        """
        if self.precondition_failure is not None:
            return AdjustmentFailure(
                reason=InvalidAdjustmentReason[self.precondition_failure.reason.name]
            )
        if {self.cause, self.effect} & adjustment.nodes:
            return AdjustmentFailure(
                reason=InvalidAdjustmentReason.CONTAINS_CAUSE_OR_EFFECT
            )
        if self.forbidden_nodes & adjustment.nodes:
            return AdjustmentFailure(
                reason=InvalidAdjustmentReason.CONTAINS_FORBIDDEN_NODE
            )
        if not self.back_door_graph.directed_graph_with_hidden_causes.is_directionally_separated(
            {self.cause}, {self.effect}, set(adjustment.nodes)
        ):
            return AdjustmentFailure(
                reason=InvalidAdjustmentReason.LEAVES_NON_CAUSAL_PATH_OPEN
            )
        return None

    def identification_without_valid_set(
        self, *, eligible: set[GraphNode]
    ) -> Identification:
        """
        :return: A refusal: the parent sets of the cause are no adjustment sets once
            hidden variables may cause both.
        """
        return UnidentifiedEffect(UnidentifiedReason.NO_VALID_SET_AMONG_ELIGIBLE)


# %% choosing an adjustment set


@dataclass
class AdjustmentSelector:
    """
    Chooses the set of variables to adjust for from a causal graph, and checks a set
    declared for it.
    """

    graph: EdgeMarkedGraph
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
        :return: The best valid adjustment set; else what the graph's criterion says
            without one, as :meth:`AdjustmentCriterion.identification_without_valid_set`
            says it.
        """
        criterion = AdjustmentCriterion.from_graph(
            graph=self.graph, cause=cause, effect=effect
        )
        if criterion.precondition_failure is not None:
            return criterion.precondition_failure
        valid = [
            adjustment
            for adjustment in self.candidate_adjustments(
                sorted(eligible - {cause, effect} - criterion.forbidden_nodes)
            )
            if criterion.failure_of(adjustment=adjustment) is None
        ]
        if valid:
            return IdentifiedEffect(
                min(
                    valid,
                    key=lambda adjustment: (
                        self.under_supported_cells(cause, adjustment, training_rows),
                        adjustment,
                    ),
                )
            )
        return criterion.identification_without_valid_set(eligible=eligible)

    def check_declared(
        self, cause: GraphNode, effect: GraphNode, declared: AdjustmentSet
    ) -> IdentifiedEffect:
        """
        :param cause: The cause.
        :param effect: The effect.
        :param declared: The nodes declared to adjust for.
        :return: The declared set, if it meets the graph's criterion.
        :raises InvalidAdjustmentSet: If it fails it.
        """
        failure = AdjustmentCriterion.from_graph(
            graph=self.graph, cause=cause, effect=effect
        ).failure_of(adjustment=declared)
        if failure is not None:
            raise InvalidAdjustmentSet(
                cause, effect, declared, failure.reason, failure.member_index
            )
        return IdentifiedEffect(declared)

    def candidate_adjustments(self, pool: list[GraphNode]) -> list[AdjustmentSet]:
        """
        :param pool: The nodes that may be adjusted for, sorted by name.
        :return: Every set of at most :attr:`maximum_adjustment_size` of them, the
            smallest first, sets of one size in the order of their sorted names.
        """
        return [
            AdjustmentSet(frozenset(adjustment))
            for size in range(self.maximum_adjustment_size + 1)
            for adjustment in itertools.combinations(pool, size)
        ]

    def under_supported_cells(
        self,
        cause: GraphNode,
        adjustment: AdjustmentSet,
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
        columns = [cause.name, *adjustment.names]
        if training_rows is None or not set(columns) <= set(training_rows.columns):
            return 0
        cell_sizes = (
            self.discretizer.fit_transform(training_rows[columns])
            .groupby(columns)
            .size()
        )
        return int(((cell_sizes > 0) & (cell_sizes < self.minimum_rows_per_cell)).sum())


# %% what to do with a declared adjustment set


class DeclaredAdjustmentPolicy(Interface):
    """
    What a causal graph's identification of an effect does with a set of nodes declared
    to adjust for.
    """

    @abstractmethod
    def identify(
        self,
        selector: AdjustmentSelector,
        cause: GraphNode,
        effect: GraphNode,
        declared: AdjustmentSet,
        eligible: set[GraphNode],
        training_rows: pd.DataFrame | None = None,
    ) -> Identification:
        """
        :param selector: Chooses and checks adjustment sets in the causal graph.
        :param cause: The cause.
        :param effect: The effect.
        :param declared: The nodes declared to adjust for.
        :param eligible: The nodes the model can adjust for.
        :param training_rows: The rows the model was fitted on, whose support ranks the
            valid adjustment sets.
        :return: What the graph says about the effect, given the declared set.
        """


@dataclass
class InvalidDeclaredAdjustmentRefusal(DeclaredAdjustmentPolicy):
    """
    Keeps a declared set that meets the causal graph's adjustment criterion and refuses
    any other.
    """

    def identify(
        self,
        selector: AdjustmentSelector,
        cause: GraphNode,
        effect: GraphNode,
        declared: AdjustmentSet,
        eligible: set[GraphNode],
        training_rows: pd.DataFrame | None = None,
    ) -> Identification:
        """
        :raises InvalidAdjustmentSet: If the declared set fails the criterion.
        """
        return selector.check_declared(cause, effect, declared)


@dataclass
class InvalidDeclaredAdjustmentRepair(DeclaredAdjustmentPolicy):
    """
    Keeps a declared set that meets the causal graph's adjustment criterion, and repairs
    any other: of the valid sets, those that drop the fewest declared nodes win,
    ranked among themselves as :meth:`AdjustmentSelector.identify` ranks its sets.
    """

    def identify(
        self,
        selector: AdjustmentSelector,
        cause: GraphNode,
        effect: GraphNode,
        declared: AdjustmentSet,
        eligible: set[GraphNode],
        training_rows: pd.DataFrame | None = None,
    ) -> Identification:
        criterion = AdjustmentCriterion.from_graph(
            graph=selector.graph, cause=cause, effect=effect
        )
        if criterion.precondition_failure is not None:
            return criterion.precondition_failure
        if criterion.failure_of(adjustment=declared) is None:
            return IdentifiedEffect(declared)
        forbidden = criterion.forbidden_nodes
        keepable = sorted(declared.nodes - {cause, effect} - forbidden)
        added = selector.candidate_adjustments(
            sorted(eligible - declared.nodes - {cause, effect} - forbidden)
        )
        valid = [
            adjustment
            for adjustment in self.sets_keeping(keepable, added)
            if criterion.failure_of(adjustment=adjustment) is None
        ]
        if not valid:
            return selector.identify(cause, effect, eligible, training_rows)
        return IdentifiedEffect(
            min(
                valid,
                key=lambda adjustment: (
                    len(declared.nodes - adjustment.nodes),
                    selector.under_supported_cells(cause, adjustment, training_rows),
                    adjustment,
                ),
            )
        )

    @staticmethod
    def sets_keeping(
        keepable: list[GraphNode], added: list[AdjustmentSet]
    ) -> list[AdjustmentSet]:
        """
        :param keepable: The declared nodes a set may keep.
        :param added: The sets of other nodes a set may add.
        :return: Every combination of some of the keepable nodes with one of the added
            sets.
        """
        return [
            AdjustmentSet(frozenset(kept) | addition.nodes)
            for addition in added
            for size in range(len(keepable), -1, -1)
            for kept in itertools.combinations(keepable, size)
        ]
