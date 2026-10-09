"""
Grounding a relational probabilistic circuit for a query in the circuit type it was
fitted in: conditioning the class circuit on the aggregation statistics a query
determines, and weighing the instances of every exchangeable relation at the mounting
nodes for the statistics it leaves undetermined.
"""

from __future__ import annotations

import enum
import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

import numpy as np
from random_events.product_algebra import Event
from sortedcontainers import SortedSet
from typing_extensions import TYPE_CHECKING, Any, Generic, Optional, Type, TypeVar

from krrood.parametrization.feature_extraction.aggregations import (
    compute_aggregation_statistics,
)
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from probabilistic_model.probabilistic_model import PartialPointType
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    CircuitNotFittedError,
    InvalidMonteCarloSampleCountError,
    UndeterminedLatentsNotModeledError,
    UndeterminedLatentsNotPartitionedError,
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
        RelationalProbabilisticCircuit,
    )

logger = logging.getLogger(__name__)

Circuit = TypeVar("Circuit")
"""
The circuit type a relational circuit is fitted and grounded in.
"""

MountingNode = TypeVar("MountingNode")
"""
A node of the class circuit that the instances of an exchangeable relation are mounted
at.
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
class GroundedPartTemplate(Generic[Circuit], SubClassSafeGeneric):
    """
    The template of one child object, grounded for the query part of a child object
    whose query has the same shape.
    """

    circuit: Circuit
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
class WeightedAssignments:
    """
    Assignments of the undetermined latents with their log-weights at one mounting
    node.
    """

    assignments: list[PartialPointType]
    """
    The assignments.
    """

    log_weights: list[float]
    """
    The log-weight of every assignment.
    """


@dataclass
class InstanceMixture:
    """
    The exchangeable instances a grounding mixes at every mounting node: one instance
    per distinct assignment of the undetermined latents, weighted per node.
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
        node_local_assignments: list[WeightedAssignments],
        latents: SortedSet[Variable],
    ) -> InstanceMixture:
        """
        :param node_local_assignments: The assignments every mounting node mixes.
        :param latents: The variables the assignments assign.
        :return: The mixture over the distinct assignments of all nodes.
        """

        def key_of(assignment: PartialPointType) -> tuple[Any, ...]:
            return tuple(assignment[variable] for variable in latents)

        index_of_key: dict[tuple[Any, ...], int] = {}
        assignments: list[PartialPointType] = []
        for node_assignments in node_local_assignments:
            for assignment in node_assignments.assignments:
                key = key_of(assignment)
                if key not in index_of_key:
                    index_of_key[key] = len(assignments)
                    assignments.append(assignment)
        log_weights = np.full((len(node_local_assignments), len(assignments)), -np.inf)
        for node, node_assignments in enumerate(node_local_assignments):
            for assignment, log_weight in zip(
                node_assignments.assignments, node_assignments.log_weights
            ):
                log_weights[node, index_of_key[key_of(assignment)]] = log_weight
        return cls(assignments, log_weights)


@dataclass
class PartitionMixture(InstanceMixture, Generic[Circuit], SubClassSafeGeneric):
    """
    A mixture with one instance per branch of the exact partition over the undetermined
    latents.
    """

    branches: list[Circuit] = field(default_factory=list)
    """
    The branch of every instance, as a circuit over the undetermined latents.
    """


@dataclass
class ExchangeablePartGrounder(
    Generic[Circuit, MountingNode], SubClassSafeGeneric, ABC
):
    """
    Grounds one exchangeable part into the class circuit, in place.

    Weighing the instances of the part at every mounting node is the same for every
    circuit type; the operations on the nodes of the class circuit and how the
    instances are mounted are up to the subclass.
    """

    circuit: Circuit
    """
    The working class circuit.
    """

    template: ExchangeableDistributionTemplate
    """
    The fitted template whose exchangeable relation is being grounded.
    """

    part_templates: list[GroundedPartTemplate[Circuit]]
    """
    The grounded template of every child object of the relation.
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

    product_nodes_to_extend: list[MountingNode] = field(
        init=False, default_factory=list
    )
    """
    The lowest product nodes of the class circuit that model every latent variable of
    the template, which mount the grounded instances.
    """

    def ground(self, grounding_mode: GroundingMode):
        """
        Condition the class circuit on the determined statistics and mount one instance
        per assignment of the statistics the query leaves undetermined, or a single
        instance if it determines all.

        :param grounding_mode: How to represent the undetermined statistics, falling
            back to :attr:`GroundingMode.SAMPLED` if their fitted partition is not
            disjoint.
        """
        self.condition_class_circuit()
        self.product_nodes_to_extend = self.mounting_nodes()
        if not self.undetermined_latents:
            self.single_instance()
            return
        if grounding_mode is GroundingMode.EXACT:
            try:
                mixture = self.exact_partition_mixture()
            except UndeterminedLatentsNotPartitionedError:
                logger.warning(
                    "Exact-partition grounding for latents [%s] is not support-"
                    "deterministic; falling back to GroundingMode.SAMPLED.",
                    ", ".join(variable.name for variable in self.undetermined_latents),
                )
            else:
                self.partition_mixture(mixture)
                return
        self.sampled_mixture(self.monte_carlo_mixture())

    # %% operations on the class circuit

    @abstractmethod
    def condition_class_circuit(self):
        """
        Condition the class circuit on the determined statistics in place, keeping its
        structure so that its product nodes stay the mounting points.

        Statistics the circuit deems impossible leave it as it is, together with what
        the exchangeable parts grounded before have mounted into it.
        """
        raise NotImplementedError

    @abstractmethod
    def mounting_nodes(self) -> list[MountingNode]:
        """
        :return: The lowest product nodes that model every latent variable of the
            template, none of them an ancestor of another.
        """
        raise NotImplementedError

    @abstractmethod
    def circuit_below(self, node: MountingNode) -> Circuit:
        """
        :param node: A node of the class circuit.
        :return: The subcircuit rooted at the node.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_undetermined_latents(self):
        """
        Remove the undetermined latents from the class circuit in place, keeping the
        mounting nodes.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def partition_branches(proposal: Circuit) -> list[Circuit]:
        """
        :param proposal: A circuit over the undetermined latents.
        :return: The subcircuits that the root of the circuit mixes, or nothing if the
            root does not mix.
        """
        raise NotImplementedError

    # %% mounting the instances

    @abstractmethod
    def single_instance(self):
        """
        Mount the part as one instance, conditioned on the determined statistics, that
        every mounting node multiplies.
        """
        raise NotImplementedError

    @abstractmethod
    def sampled_mixture(self, mixture: InstanceMixture):
        """
        Mount the part as one instance per sampled assignment of the undetermined
        statistics, each carrying its sampled values as point masses, mixed at every
        mounting node.

        :param mixture: The sampled assignments with their weights at every mounting
            node.
        """
        raise NotImplementedError

    @abstractmethod
    def partition_mixture(self, mixture: PartitionMixture[Circuit]):
        """
        Mount the part as one instance per branch of the exact partition over the
        undetermined statistics, each carrying its branch, mixed at every mounting
        node.

        :param mixture: A representative assignment of every branch, with the weights
            of the branches at every mounting node.
        """
        raise NotImplementedError

    # %% weighing the instances

    def monte_carlo_mixture(self) -> InstanceMixture:
        """
        Sample the undetermined aggregation statistics and weigh every distinct sample
        locally to each mounting node, then remove the undetermined latents from the
        class circuit, since the instances carry them from here on.

        :return: The sampled assignments and their weights at every mounting node.
        """
        sampled_assignments = self._sample_undetermined_latents()
        node_local_assignments = [
            self._node_local_assignments(node, sampled_assignments)
            for node in self.product_nodes_to_extend
        ]
        self.remove_undetermined_latents()
        return InstanceMixture.of_node_local_assignments(
            node_local_assignments, self.undetermined_latents
        )

    def exact_partition_mixture(self) -> PartitionMixture[Circuit]:
        """
        Weigh the branches of the undetermined latents' own exact partition locally to
        every mounting node, then remove the undetermined latents from the class
        circuit, since the branches carry them from here on.

        Unlike Monte-Carlo sampling, this enumerates the class circuit's already-fitted,
        exact partition over ``undetermined_latents`` (the branches of the root of its
        marginal over them) instead of drawing samples from it: reproducible across
        calls, and covers every value the model learned about rather than only
        whichever points got sampled.

        :return: One instance per branch, conditioned on a representative point of the
            branch, with the weights of the branches at every mounting node.
        :raises UndeterminedLatentsNotModeledError: If the class circuit does not model
            ``undetermined_latents`` and thus has no partition over them.
        :raises UndeterminedLatentsNotPartitionedError: If that partition's branches
            are not pairwise disjoint, so exact-partition grounding would not be
            support-deterministic.
        """
        proposal = self.circuit.marginal(self.undetermined_latents)
        if proposal is None:
            raise UndeterminedLatentsNotModeledError(list(self.undetermined_latents))
        branches = self.partition_branches(proposal)
        branch_regions = [branch.support for branch in branches]
        if not self._undetermined_latents_partition_disjointly(branch_regions):
            raise UndeterminedLatentsNotPartitionedError(
                list(self.undetermined_latents)
            )

        # each node's weights must be read off the class circuit before the
        # undetermined latents are removed from it below
        log_weights = np.array(
            [
                self._branch_log_probabilities(
                    self.circuit_below(node), self.undetermined_latents, branch_regions
                )
                for node in self.product_nodes_to_extend
            ]
        )
        self.remove_undetermined_latents()

        assignments = [
            self._representative_value(branch, self.undetermined_latents)
            for branch in branches
        ]
        return PartitionMixture(assignments, log_weights, branches)

    def _sample_undetermined_latents(
        self, node: Optional[MountingNode] = None
    ) -> list[PartialPointType]:
        """
        Draw the distinct values of the undetermined latents to integrate over.

        Samples ``monte_carlo_sample_count`` joint assignments of the undetermined
        latents from the conditioned class circuit and deduplicates them, so that each
        distinct value is grounded only once.

        :param node: A mounting node to sample the latents local to, instead of from
            the whole conditioned class circuit.
        :return: One value assignment per distinct sampled point.
        :raises InvalidMonteCarloSampleCountError: If the sample count is not positive.
        :raises UndeterminedLatentsNotModeledError: If the conditioned class circuit
            does not model the undetermined latents and thus cannot be sampled from.
        """
        if self.monte_carlo_sample_count < 1:
            raise InvalidMonteCarloSampleCountError(self.monte_carlo_sample_count)
        source = self.circuit if node is None else self.circuit_below(node)
        proposal = source.marginal(self.undetermined_latents)
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

    def _node_local_latent_log_likelihoods(
        self, node: MountingNode, latent_assignments: list[PartialPointType]
    ) -> list[float]:
        """
        Log-likelihoods of latent assignments local to a mounting node.

        Marginalizes the subcircuit rooted at ``node`` to the undetermined latents once,
        then evaluates every assignment in a single batched pass.

        :param node: The mounting node.
        :param latent_assignments: The sampled assignments of the undetermined latents.
        :return: One log-likelihood per assignment, in input order.
        """
        subcircuit = self.circuit_below(node)
        subcircuit.marginal_in_place(self.undetermined_latents)
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
        self, node: MountingNode, sampled_assignments: list[PartialPointType]
    ) -> WeightedAssignments:
        """
        The latent assignments one mounting node integrates over, with their node-local
        log-likelihoods.

        The assignments sampled from the whole class circuit are used wherever the node
        gives any of them positive likelihood. A node whose own local latent marginal is
        narrow enough that none of them falls into it -- as happens when the class
        circuit was fit stratified over one of the latents -- draws its own samples from
        that local marginal instead, so it is never handed an instance carrying another
        node's latent value.

        :param node: The mounting node.
        :param sampled_assignments: The assignments sampled from the whole circuit.
        :return: The node's assignments with their node-local log-likelihoods.
        """
        log_weights = self._node_local_latent_log_likelihoods(node, sampled_assignments)
        if any(log_weight > -np.inf for log_weight in log_weights):
            return WeightedAssignments(sampled_assignments, log_weights)
        local_assignments = self._sample_undetermined_latents(node)
        return WeightedAssignments(
            local_assignments,
            self._node_local_latent_log_likelihoods(node, local_assignments),
        )

    @staticmethod
    def _branch_log_probabilities(
        subcircuit: Circuit,
        undetermined_latents: SortedSet[Variable],
        branch_regions: list[Event],
    ) -> list[float]:
        """
        Log-probability of each partition branch's region, local to a mounting node.

        Different mounting nodes can correlate ``undetermined_latents`` with the
        variables that distinguish them, so each node's weights over the same global
        partition must be computed from its own local marginal, mirroring
        :meth:`_node_local_latent_log_likelihoods`'s per-node handling for the Monte-
        Carlo mixture.

        :param subcircuit: The subcircuit rooted at the mounting node.
        :param undetermined_latents: The latent variables the partition covers.
        :param branch_regions: Each partition branch's own support region, in the same
            order as the branches being weighted.
        :return: One log-probability per region, in input order.
        """
        subcircuit.marginal_in_place(undetermined_latents)
        with np.errstate(divide="ignore"):
            return [
                float(np.log(subcircuit.probability(region)))
                for region in branch_regions
            ]

    @staticmethod
    def _undetermined_latents_partition_disjointly(
        branch_regions: list[Event],
    ) -> bool:
        """
        Check whether the branches of the undetermined latents' marginal form a genuine,
        pairwise-disjoint partition: at least two branches, no two of which overlap.

        A single, undifferentiated branch fails this precondition rather than
        trivially passing it: grounding would still retain the latents as a real
        distribution, but every exchangeable instance would be grounded from the same
        representative point regardless of which latent value the region actually
        corresponds to, silently discarding any correlation between the latents and
        the rest of the circuit.

        :param branch_regions: The support of every branch.
        :return: ``True`` only if there are at least two branches, every pair of which
            has non-overlapping support.
        """
        if len(branch_regions) < 2:
            return False
        return all(
            left.intersection_with(right).is_empty()
            for left, right in itertools.combinations(branch_regions, 2)
        )

    @staticmethod
    def _representative_value(
        branch: Circuit, undetermined_latents: SortedSet[Variable]
    ) -> PartialPointType:
        """
        Extract one concrete point per undetermined latent from a partition branch.

        Uses the mode of the branch's marginal over each latent, collapsed to a single
        point of that mode region. Any point within the branch's support would do for
        grounding purposes here, since the branch's actual probability mass is retained
        separately by mounting the branch itself, not narrowed by this choice. A single
        point is required because conditioning a leaf's distribution on a whole region
        -- rather than one point -- is not supported by
        :meth:`~probabilistic_model.distributions.distributions.ContinuousDistribution.log_conditional`.

        :param branch: One branch of ``undetermined_latents``' exact partition.
        :param undetermined_latents: The latent variables to extract a value for.
        :return: One conditioning point per variable in ``undetermined_latents``.
        """
        values = {}
        for variable in undetermined_latents:
            mode, _ = branch.marginal([variable]).log_mode(check_determinism=False)
            region = mode.simple_sets[0][variable]
            values[variable] = (
                region.simple_sets[0].lower
                if isinstance(region, Interval)
                else next(iter(region))
            )
        return values


# %% grounding a relational circuit


@dataclass
class RelationalGrounding(Generic[Circuit], SubClassSafeGeneric, ABC):
    """
    Grounds a relational probabilistic circuit for queries in the circuit type it was
    fitted in.

    Determining the statistics of every exchangeable relation and grounding the
    templates of its child objects are the same for every circuit type; what grounds a
    relation is up to the subclass.
    """

    relational_circuit: RelationalProbabilisticCircuit
    """
    The fitted relational circuit to ground.
    """

    grounding_mode: GroundingMode = GroundingMode.SAMPLED
    """
    How to represent the aggregation statistics a query leaves undetermined, for the
    relations of the circuit and of every nested template.
    """

    @property
    @abstractmethod
    def part_grounder_type(self) -> Type[ExchangeablePartGrounder[Circuit, Any]]:
        """
        :return: What grounds one exchangeable relation.
        """
        raise NotImplementedError

    def ground(self, query: Match) -> Circuit:
        """
        :param query: An underspecified, resolved query whose structure determines which
            parts are grounded and how many child objects every exchangeable relation
            contains.
        :return: The grounded circuit over all variables implied by the query.
        :raises CircuitNotFittedError: If the relational circuit is not fitted.
        """
        relational_circuit = self.relational_circuit
        if relational_circuit.class_probabilistic_circuit is None:
            raise CircuitNotFittedError(relational_circuit.class_)
        circuit = relational_circuit.class_probabilistic_circuit.__deepcopy__()
        instance = query.construct_instance()
        for (
            exchangeable_part_name,
            template,
        ) in relational_circuit.exchangeable_distribution_templates.items():
            self.part_grounder(
                circuit, exchangeable_part_name, template, query, instance
            ).ground(self.grounding_mode)
        return circuit

    def part_grounder(
        self,
        circuit: Circuit,
        exchangeable_part_name: str,
        template: ExchangeableDistributionTemplate,
        query: Match,
        instance: Any,
    ) -> ExchangeablePartGrounder[Circuit, Any]:
        """
        Determine the aggregation statistics of one exchangeable part and ground the
        templates of its child objects.

        :param circuit: The working class circuit.
        :param exchangeable_part_name: Field name of the exchangeable relation.
        :param template: The fitted template of the relation.
        :param query: The grounding query.
        :param instance: The instance constructed from the query.
        :return: The grounder of the part.
        """
        aggregation_statistics = compute_aggregation_statistics(
            instance, exchangeable_part_name, template.latent_variables
        )
        determined_statistics = {
            variable: value
            for variable, value in aggregation_statistics.items()
            if self._is_concrete_statistic(variable, value)
        }
        undetermined_latents = SortedSet(
            variable
            for variable in template.latent_variables
            if variable not in determined_statistics
        )
        nested_grounding = replace(
            self, relational_circuit=template.template_distribution
        )
        return self.part_grounder_type(
            circuit=circuit,
            template=template,
            part_templates=template.grounded_templates_of_parts(
                query._kwargs_[exchangeable_part_name], nested_grounding.ground
            ),
            determined_statistics=determined_statistics,
            undetermined_latents=undetermined_latents,
            monte_carlo_sample_count=self.relational_circuit.monte_carlo_sample_count,
        )

    @staticmethod
    def _is_concrete_statistic(variable: Variable, value: Any) -> bool:
        """
        :param variable: The latent variable the value belongs to.
        :param value: The observed aggregation value, either a concrete point or a range.
        :return: Whether the value designates exactly one element of the variable's
            domain.
        """
        composite = variable.make_value(value)
        if isinstance(composite, Interval):
            return composite.is_singleton()
        return len(composite.simple_sets) == 1
