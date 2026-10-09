"""
Grounding one exchangeable relation of a relational probabilistic circuit at the class
circuit: which aggregation statistics a query leaves undetermined, and how the instances
of the relation are weighed at every mounting product node.
"""

from __future__ import annotations

import enum
import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from sortedcontainers import SortedSet
from typing_extensions import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar

from probabilistic_model.probabilistic_model import PartialPointType
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    InvalidMonteCarloSampleCountError,
    UndeterminedLatentsNotModeledError,
    UndeterminedLatentsNotPartitionedError,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    Unit,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    DenseEdgeValues,
)
from random_events.interval import Interval
from random_events.variable import Variable

if TYPE_CHECKING:
    from krrood.entity_query_language.query.match import Match
    from probabilistic_model.probabilistic_circuit.relational.rspn import (
        ExchangeableDistributionTemplate,
    )

logger = logging.getLogger(__name__)

GroundedPart = TypeVar("GroundedPart")
"""
What a grounder turns the instances of one exchangeable part into.
"""

GroundedCircuit = TypeVar("GroundedCircuit")
"""
The circuit a template is grounded into.
"""


class GroundingMode(enum.IntEnum):
    """
    How a grounded circuit represents the aggregation statistics a query leaves
    undetermined. The statistics stay variables of the grounded circuit.
    """

    SAMPLED = enum.auto()
    """
    One instance per distinct Monte-Carlo sample of the statistics, carrying the
    sampled values as point masses.
    """

    EXACT = enum.auto()
    """
    One instance per branch of the fitted partition over the statistics, carrying the
    branch.
    """


@dataclass
class GroundedPartTemplate(Generic[GroundedCircuit]):
    """
    The template of one child object, grounded for the query part of a child object
    whose query has the same shape.
    """

    circuit: GroundedCircuit
    """
    The grounded template, shared by every child object of the same shape.
    """

    grounded_prefix: str
    """
    The namespace of the child object the template was grounded for.
    """

    prefix: str
    """
    The namespace of the child object.
    """


@dataclass
class InstanceMixture:
    """
    The exchangeable instances a grounding mixes at every mounting product node: one
    instance per distinct assignment of the undetermined latents, weighted per node.
    """

    assignments: list[PartialPointType]
    """
    The distinct assignments of the undetermined latents, one instance each.
    """

    log_weights: DenseEdgeValues
    """
    The weight of every instance at every mounting node, shape (#mounting nodes,
    #instances), ``-inf`` where a node does not use an instance.
    """

    @classmethod
    def of_node_local_assignments(
        cls,
        node_local_assignments: list[tuple[list[PartialPointType], list[float]]],
        key_of: Callable[[PartialPointType], tuple[Any, ...]],
    ) -> InstanceMixture:
        """
        :param node_local_assignments: Per mounting node, the assignments it mixes and
            their log-weights.
        :param key_of: What tells two equal assignments apart from different ones.
        :return: The mixture over the distinct assignments of all nodes.
        """
        index_of_key: dict[tuple[Any, ...], int] = {}
        assignments: list[PartialPointType] = []
        for node_assignments, _ in node_local_assignments:
            for assignment in node_assignments:
                key = key_of(assignment)
                if key not in index_of_key:
                    index_of_key[key] = len(assignments)
                    assignments.append(assignment)
        log_weights = np.full((len(node_local_assignments), len(assignments)), -np.inf)
        for node, (node_assignments, node_log_weights) in enumerate(
            node_local_assignments
        ):
            for assignment, log_weight in zip(node_assignments, node_log_weights):
                log_weights[node, index_of_key[key_of(assignment)]] = log_weight
        return cls(assignments, log_weights)


@dataclass
class ExchangeablePartGrounder(ABC, Generic[GroundedPart]):
    """
    Grounds one exchangeable part at the mounting product nodes of a class circuit.

    Weighing the instances of the part at every mounting node is the same for every
    representation of the grounded circuit; how the instances become part of it is up
    to the subclass.
    """

    circuit: ProbabilisticCircuit
    """
    The working class circuit being extended.
    """

    product_nodes_to_extend: list[ProductUnit]
    """
    The class circuit's product nodes that mount the grounded instance(s).
    """

    template: ExchangeableDistributionTemplate
    """
    The fitted template whose exchangeable relation is being grounded.
    """

    query_parts: list[Match]
    """
    The query parts, one per child object in the relation.
    """

    determined_statistics: PartialPointType
    """
    Aggregation statistics determinable from the query.
    """

    undetermined_latents: SortedSet[Variable] = field(default_factory=SortedSet)
    """
    Latent variables the query leaves undetermined.
    """

    monte_carlo_sample_count: int = 10
    """
    Number of Monte-Carlo samples drawn to integrate out ``undetermined_latents``.
    """

    def ground(self, grounding_mode: GroundingMode) -> GroundedPart:
        """
        Ground the part with one instance per assignment of the aggregation statistics
        the query leaves undetermined, or with a single instance if it determines all.

        :param grounding_mode: How to represent the undetermined statistics, falling
            back to :attr:`GroundingMode.SAMPLED` if their fitted partition is not
            disjoint.
        :return: The grounded part.
        """
        if not self.undetermined_latents:
            return self.single_instance()
        if grounding_mode is GroundingMode.EXACT:
            try:
                mixture, branches = self.exact_partition_mixture()
            except UndeterminedLatentsNotPartitionedError:
                logger.warning(
                    "Exact-partition grounding for latents [%s] is not support-"
                    "deterministic; falling back to GroundingMode.SAMPLED.",
                    ", ".join(variable.name for variable in self.undetermined_latents),
                )
            else:
                return self.partition_mixture(mixture, branches)
        return self.sampled_mixture(self.monte_carlo_mixture())

    @abstractmethod
    def single_instance(self) -> GroundedPart:
        """
        :return: The part as one instance, conditioned on the determined statistics,
            that every mounting node multiplies.
        """
        raise NotImplementedError

    @abstractmethod
    def sampled_mixture(self, mixture: InstanceMixture) -> GroundedPart:
        """
        :param mixture: The sampled assignments of the undetermined statistics, with
            their weights at every mounting node.
        :return: The part as one instance per assignment, each carrying its sampled
            values as point masses, mixed at every mounting node.
        """
        raise NotImplementedError

    @abstractmethod
    def partition_mixture(
        self, mixture: InstanceMixture, branches: list[Unit]
    ) -> GroundedPart:
        """
        :param mixture: A representative assignment of every branch of the exact
            partition over the undetermined statistics, with the weights of the
            branches at every mounting node.
        :param branches: The branches of the partition.
        :return: The part as one instance per branch, each carrying its branch, mixed
            at every mounting node.
        """
        raise NotImplementedError

    def monte_carlo_mixture(self) -> InstanceMixture:
        """
        Sample the undetermined aggregation statistics and weigh every distinct sample
        locally to each mounting product node, then remove the undetermined latents from
        the class circuit, since the instances carry them from here on.

        :return: The sampled assignments and their weights at every mounting node.
        """
        sampled_assignments = self._sample_undetermined_latents()
        node_local_assignments = [
            self._node_local_assignments(product_node, sampled_assignments)
            for product_node in self.product_nodes_to_extend
        ]
        retained_variables = (
            SortedSet(self.circuit.variables) - self.undetermined_latents
        )
        self.circuit.restrict_to_variables_in_place(retained_variables)
        return InstanceMixture.of_node_local_assignments(
            node_local_assignments, self._assignment_key
        )

    def _assignment_key(self, assignment: PartialPointType) -> tuple[Any, ...]:
        """
        :param assignment: Values of the undetermined latents.
        :return: The values in the latents' sorted order, to tell assignments apart.
        """
        return tuple(assignment[variable] for variable in self.undetermined_latents)

    def exact_partition_mixture(self) -> tuple[InstanceMixture, list[Unit]]:
        """
        Weigh the branches of the undetermined latents' own exact partition locally to
        every mounting product node, then remove the undetermined latents from the class
        circuit, since the branches carry them from here on.

        Unlike Monte-Carlo sampling, this enumerates ``circuit``'s already-fitted,
        exact partition over ``undetermined_latents`` (the branches of
        ``circuit.marginal(undetermined_latents)``'s root) instead of drawing samples
        from it: reproducible across calls, and covers every value the model learned
        about rather than only whichever points got sampled.

        :return: One instance per branch, conditioned on a representative point of the
            branch, with the weights of the branches at every mounting node, and the
            branches.
        :raises UndeterminedLatentsNotModeledError: If ``circuit`` does not model
            ``undetermined_latents`` and thus has no partition over them.
        :raises UndeterminedLatentsNotPartitionedError: If that partition's branches
            are not pairwise disjoint, so exact-partition grounding would not be
            support-deterministic.
        """
        proposal = self.circuit.marginal(self.undetermined_latents)
        if proposal is None:
            raise UndeterminedLatentsNotModeledError(list(self.undetermined_latents))
        if not self._undetermined_latents_partition_disjointly(proposal):
            raise UndeterminedLatentsNotPartitionedError(
                list(self.undetermined_latents)
            )

        # the precondition just verified guarantees proposal.root is a SumUnit with at
        # least two branches
        branches = [branch for _, branch in proposal.root.log_weighted_subcircuits]
        # _undetermined_latents_partition_disjointly already called proposal.support
        # above, caching each branch's region on it as result_of_current_query
        branch_regions = [branch.result_of_current_query for branch in branches]

        # each node's weights must be read off circuit before undetermined_latents are
        # stripped from it below -- product_node stops modeling them afterward
        log_weights = np.array(
            [
                self._node_local_branch_log_probabilities(
                    product_node, self.undetermined_latents, branch_regions
                )
                for product_node in self.product_nodes_to_extend
            ]
        )

        retained_variables = (
            SortedSet(self.circuit.variables) - self.undetermined_latents
        )
        self.circuit.restrict_to_variables_in_place(retained_variables)

        assignments = [
            self._representative_value(latent_branch, self.undetermined_latents)
            for latent_branch in branches
        ]
        return InstanceMixture(assignments, log_weights), branches

    def _sample_undetermined_latents(
        self, product_node: Optional[ProductUnit] = None
    ) -> list[PartialPointType]:
        """
        Draw the distinct values of the undetermined latents to integrate over.

        Samples ``monte_carlo_sample_count`` joint assignments of the undetermined
        latents from the conditioned class circuit and deduplicates them, so that each
        distinct value is grounded only once.

        :param product_node: A mounting product node to sample the latents local to,
            instead of from the whole conditioned class circuit.
        :return: One value assignment per distinct sampled point.
        :raises InvalidMonteCarloSampleCountError: If the sample count is not positive.
        :raises UndeterminedLatentsNotModeledError: If the conditioned class circuit
            does not model the undetermined latents and thus cannot be sampled from.
        """
        if self.monte_carlo_sample_count < 1:
            raise InvalidMonteCarloSampleCountError(self.monte_carlo_sample_count)
        if product_node is None:
            proposal = self.circuit.marginal(self.undetermined_latents)
        else:
            proposal = ProbabilisticCircuit()
            proposal.mount(product_node)
            proposal = proposal.marginal(self.undetermined_latents)
        if proposal is None:
            raise UndeterminedLatentsNotModeledError(list(self.undetermined_latents))
        samples = proposal.sample(self.monte_carlo_sample_count)
        index_of_variable = proposal.variable_to_index_map
        unique_rows = {tuple(row) for row in samples.tolist()}
        return [
            {
                variable: row[index_of_variable[variable]]
                for variable in self.undetermined_latents
            }
            for row in (np.array(unique_row) for unique_row in unique_rows)
        ]

    @staticmethod
    def _node_local_latent_log_likelihoods(
        product_node: ProductUnit,
        undetermined_latents: SortedSet[Variable],
        latent_assignments: list[PartialPointType],
    ) -> list[float]:
        """
        Log-likelihoods of latent assignments local to a mounting product node.

        Marginalizes the subcircuit rooted at ``product_node`` to the undetermined
        latents once, then evaluates every assignment in a single batched pass.

        :param product_node: The mounting product node.
        :param undetermined_latents: The latent variables sampled by Monte-Carlo.
        :param latent_assignments: The sampled assignments of those latents.
        :return: One log-likelihood per assignment, in input order.
        """
        subcircuit = ProbabilisticCircuit()
        subcircuit.mount(product_node)
        subcircuit.marginal_in_place(undetermined_latents)
        index_of_variable = subcircuit.variable_to_index_map
        events = np.full((len(latent_assignments), len(index_of_variable)), np.nan)
        for row, assignment in enumerate(latent_assignments):
            for variable, value in assignment.items():
                events[row, index_of_variable[variable]] = value
        return [
            float(log_likelihood)
            for log_likelihood in subcircuit.log_likelihood(events)
        ]

    def _node_local_assignments(
        self, product_node: ProductUnit, sampled_assignments: list[PartialPointType]
    ) -> tuple[list[PartialPointType], list[float]]:
        """
        The latent assignments one mounting node integrates over, with their node-local
        log-likelihoods.

        The assignments sampled from the whole class circuit are used wherever the node
        gives any of them positive likelihood. A node whose own local latent marginal is
        narrow enough that none of them falls into it -- as happens when the class
        circuit was fit stratified over one of the latents -- draws its own samples from
        that local marginal instead, so it is never handed an instance carrying another
        node's latent value.

        :param product_node: The mounting product node.
        :param sampled_assignments: The assignments sampled from the whole circuit.
        :return: The node's assignments and their node-local log-likelihoods.
        """
        log_weights = self._node_local_latent_log_likelihoods(
            product_node, self.undetermined_latents, sampled_assignments
        )
        if any(log_weight > -np.inf for log_weight in log_weights):
            return sampled_assignments, log_weights
        local_assignments = self._sample_undetermined_latents(product_node)
        return local_assignments, self._node_local_latent_log_likelihoods(
            product_node, self.undetermined_latents, local_assignments
        )

    @staticmethod
    def _node_local_branch_log_probabilities(
        product_node: ProductUnit,
        undetermined_latents: SortedSet[Variable],
        branch_regions: list,
    ) -> list[float]:
        """
        Log-probability of each partition branch's region, local to a mounting product
        node.

        Different product nodes can correlate ``undetermined_latents`` with the
        variables that distinguish them, so each node's weights over the same global
        partition must be computed from its own local marginal, mirroring
        :meth:`_node_local_latent_log_likelihoods`'s per-node handling for the Monte-
        Carlo mixture.

        :param product_node: The mounting product node.
        :param undetermined_latents: The latent variables the partition covers.
        :param branch_regions: Each partition branch's own support region, in the same
            order as the branches being weighted.
        :return: One log-probability per region, in input order.
        """
        subcircuit = ProbabilisticCircuit()
        subcircuit.mount(product_node)
        subcircuit.marginal_in_place(undetermined_latents)
        with np.errstate(divide="ignore"):
            return [
                float(np.log(subcircuit.probability(region)))
                for region in branch_regions
            ]

    @staticmethod
    def _undetermined_latents_partition_disjointly(
        proposal: ProbabilisticCircuit,
    ) -> bool:
        """
        Check whether ``proposal`` is a genuine, pairwise-disjoint partition over the
        undetermined latents: a mixture of at least two branches, no two of which
        overlap.

        A single, undifferentiated branch fails this precondition rather than
        trivially passing it: grounding would still retain the latents as a real
        distribution, but every exchangeable instance would be grounded from the same
        representative point regardless of which latent value the region actually
        corresponds to, silently discarding any correlation between the latents and
        the rest of the circuit.

        :param proposal: ``circuit`` marginalized down to exactly the undetermined
            latents.
        :return:``True`` only if ``proposal``'s root is a mixture of at least two
            branches, every pair of which has non-overlapping support.
        """
        root = proposal.root
        if not isinstance(root, SumUnit) or len(root.subcircuits) < 2:
            return False
        _ = proposal.support
        branch_supports = [child.result_of_current_query for child in root.subcircuits]
        return all(
            left.intersection_with(right).is_empty()
            for left, right in itertools.combinations(branch_supports, 2)
        )

    @staticmethod
    def _representative_value(
        latent_branch: Unit, undetermined_latents: SortedSet[Variable]
    ) -> PartialPointType:
        """
        Extract one concrete point per undetermined latent from a partition branch.

        Uses each leaf's own mode, collapsed to a single point of that mode region.
        Any point within the branch's support would do for grounding purposes here,
        since the branch's actual probability mass is retained separately by mounting
        the branch itself, not narrowed by this choice. A single point is required
        because conditioning a leaf's distribution on a whole region -- rather than one
        point -- is not supported by :meth:`~probabilistic_model.distributions.distributions.ContinuousDistribution.log_conditional`,
        which every leaf here resolves to (including :class:`IntegerDistribution`,
        through its continuous base).

        :param latent_branch: One branch of ``undetermined_latents``' exact partition.
        :param undetermined_latents: The latent variables to extract a value for.
        :return: One conditioning point per variable in ``undetermined_latents``.
        """
        values = {}
        for leaf_node in latent_branch.leaves:
            if leaf_node.variable not in undetermined_latents:
                continue
            mode, _ = leaf_node.distribution.univariate_log_mode()
            values[leaf_node.variable] = (
                mode.simple_sets[0].lower
                if isinstance(mode, Interval)
                else next(iter(mode))
            )
        return values
