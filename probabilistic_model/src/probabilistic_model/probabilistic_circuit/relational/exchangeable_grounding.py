"""
Grounding relational probabilistic circuits for queries, independent of the circuit type.
"""

from __future__ import annotations

import copy
import enum
import itertools
from dataclasses import dataclass, field, replace

import numpy as np
from random_events.product_algebra import Event
from sortedcontainers import SortedSet
from typing_extensions import TYPE_CHECKING, Any, Generic, TypeVar

from krrood.parametrization.feature_extraction.aggregations import (
    compute_aggregation_statistics,
)
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from probabilistic_model.probabilistic_model import PartialPointType
from probabilistic_model.probabilistic_circuit.groundable_circuit import (
    ConditionedCopies,
    GroundableCircuit,
    NodeType,
    ProductsOfCopies,
)
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    CircuitNotFittedError,
    InvalidMonteCarloSampleCountError,
    PartCircuitGroundingFailedError,
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
    from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
        ExchangeableDistributionTemplate,
        RelationalProbabilisticCircuit,
    )


CircuitType = TypeVar("CircuitType", bound=GroundableCircuit)
"""
The circuit type a relational circuit is fitted and grounded in.
"""


class GroundingMode(enum.IntEnum):
    """
    How a grounded circuit represents the aggregation statistics a query leaves
    undetermined.
    """

    SAMPLED = enum.auto()
    """
    One instance per distinct sample of the statistics.
    """

    EXACT = enum.auto()
    """
    One instance per branch of the fitted partition over the statistics.
    """


@dataclass
class GroundedPartTemplate(Generic[CircuitType], SubClassSafeGeneric):
    """
    The grounded template of one child object.
    """

    circuit: CircuitType
    """
    The grounded template, shared by every child object whose query has the same shape.
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
    The instances of an exchangeable relation and their weights at every mounting node.
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
        index_of_key: dict[tuple[Any, ...], int] = {}
        assignments: list[PartialPointType] = []
        for node_assignments in node_local_assignments:
            for assignment in node_assignments.assignments:
                key = cls.key_of(assignment, latents)
                if key not in index_of_key:
                    index_of_key[key] = len(assignments)
                    assignments.append(assignment)
        log_weights = np.full((len(node_local_assignments), len(assignments)), -np.inf)
        for node, node_assignments in enumerate(node_local_assignments):
            for assignment, log_weight in zip(
                node_assignments.assignments, node_assignments.log_weights
            ):
                log_weights[node, index_of_key[cls.key_of(assignment, latents)]] = (
                    log_weight
                )
        return cls(assignments, log_weights)

    @staticmethod
    def key_of(
        assignment: PartialPointType, latents: SortedSet[Variable]
    ) -> tuple[Any, ...]:
        """
        :param assignment: An assignment of the latents.
        :param latents: The variables the assignment assigns.
        :return: The values of the assignment, in the order of the latents.
        """
        return tuple(assignment[variable] for variable in latents)


@dataclass
class PartitionMixture(InstanceMixture, Generic[CircuitType], SubClassSafeGeneric):
    """
    A mixture with one instance per branch of the exact partition over the undetermined
    latents.
    """

    branches: list[CircuitType] = field(default_factory=list)
    """
    The branch of every instance, as a circuit over the undetermined latents.
    """


@dataclass
class ExchangeablePartGrounder(Generic[NodeType], SubClassSafeGeneric):
    """
    Grounds one exchangeable part into the class circuit, in place, through the
    operations on nodes that every groundable circuit provides.
    """

    circuit: GroundableCircuit[NodeType]
    """
    The working class circuit.
    """

    template: ExchangeableDistributionTemplate
    """
    The fitted template whose exchangeable relation is being grounded.
    """

    part_templates: list[GroundedPartTemplate]
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
    Number of samples drawn of ``undetermined_latents``.
    """

    product_nodes_to_extend: list[NodeType] = field(init=False, default_factory=list)
    """
    The nodes of the class circuit that the instances are mounted at.
    """

    def ground(self, grounding_mode: GroundingMode):
        """
        Mount the instances of the part into the class circuit.

        :param grounding_mode: How to represent the undetermined statistics.
        :raises UndeterminedLatentsNotPartitionedError: If the grounding mode is
            ``EXACT`` and the fitted partition of the undetermined statistics is not
            disjoint.
        """
        self.circuit.condition_on_point_in_place(self.determined_statistics)
        self.product_nodes_to_extend = self.circuit.lowest_product_nodes_over(
            SortedSet(self.template.latent_variables)
        )
        if not self.undetermined_latents:
            self.single_instance()
            return
        if grounding_mode is GroundingMode.EXACT:
            self.partition_mixture(self.exact_partition_mixture())
        else:
            self.sampled_mixture(self.monte_carlo_mixture())

    def remove_undetermined_latents(self):
        """
        Remove the undetermined latents from the class circuit in place, keeping the
        mounting nodes.
        """
        self.product_nodes_to_extend = self.circuit.remove_variables_in_place(
            self.undetermined_latents, self.product_nodes_to_extend
        )

    # %% mounting the instances

    def single_instance(self):
        """
        Mount the part as one instance at every mounting node.
        """
        self.circuit.multiply_nodes_by_mixtures(
            self.product_nodes_to_extend,
            self.instances([self.determined_statistics]),
            None,
        )

    def sampled_mixture(self, mixture: InstanceMixture):
        """
        Mount the part as a mixture of one instance per sampled assignment, each
        carrying its assignment as point masses.

        :param mixture: The sampled assignments with their weights at every mounting
            node.
        """
        instances = self.instances(self.statistics_of(mixture))
        instances.point_masses = {
            variable: [assignment[variable] for assignment in mixture.assignments]
            for variable in self.undetermined_latents
        }
        self.circuit.multiply_nodes_by_mixtures(
            self.product_nodes_to_extend, instances, mixture.log_weights
        )

    def partition_mixture(self, mixture: PartitionMixture):
        """
        Mount the part as a mixture of one instance per branch of the exact partition,
        each carrying its branch.

        :param mixture: The branches with their weights at every mounting node.
        """
        instances = self.instances(self.statistics_of(mixture))
        instances.components = mixture.branches
        self.circuit.multiply_nodes_by_mixtures(
            self.product_nodes_to_extend, instances, mixture.log_weights
        )

    def statistics_of(self, mixture: InstanceMixture) -> list[PartialPointType]:
        """
        :param mixture: The assignments of the undetermined statistics.
        :return: The values of every aggregation statistic, one assignment per instance.
        """
        return [
            {**self.determined_statistics, **assignment}
            for assignment in mixture.assignments
        ]

    def instances(self, statistics: list[PartialPointType]) -> ProductsOfCopies:
        """
        :param statistics: The values of every aggregation statistic, one assignment per
            instance.
        :return: The instances, each multiplying the template of every child object
            conditioned on its statistics.
        :raises PartCircuitGroundingFailedError: If a template models nothing but the
            aggregation statistics.
        """
        conditioned_copies = []
        for part in self.part_templates:
            variables = SortedSet(
                variable
                for variable in part.circuit.variables
                if variable not in self.template.latent_variables
            )
            if not variables:
                raise PartCircuitGroundingFailedError(
                    self.template.template_distribution.class_
                )
            conditioned_copies.append(
                ConditionedCopies(
                    part.circuit,
                    variables,
                    {
                        variable: self.template.variable_of_part(variable, part)
                        for variable in variables
                    },
                )
            )
        return ProductsOfCopies(statistics, conditioned_copies)

    # %% weighing the instances

    def monte_carlo_mixture(self) -> InstanceMixture:
        """
        Sample the undetermined statistics and remove them from the class circuit.

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

    def exact_partition_mixture(self) -> PartitionMixture[CircuitType]:
        """
        Weigh the branches of the fitted partition over the undetermined statistics and
        remove the statistics from the class circuit.

        :return: The branches, a representative assignment of each, and their weights
            at every mounting node.
        :raises UndeterminedLatentsNotModeledError: If the class circuit does not model
            the undetermined statistics.
        :raises UndeterminedLatentsNotPartitionedError: If the regions of the branches
            are not pairwise disjoint.
        """
        proposal = self.circuit.marginal(self.undetermined_latents)
        if proposal is None:
            raise UndeterminedLatentsNotModeledError(list(self.undetermined_latents))
        branch_regions = self._distinct_component_regions(proposal)
        if not self._undetermined_latents_partition_disjointly(branch_regions):
            raise UndeterminedLatentsNotPartitionedError(
                list(self.undetermined_latents)
            )
        branches = [
            proposal.truncated(region).distribution for region in branch_regions
        ]

        # each node's weights must be read off the class circuit before the
        # undetermined latents are removed from it below
        log_weights = np.array(
            [
                self._branch_log_probabilities(
                    self.circuit.subcircuit_at(node),
                    self.undetermined_latents,
                    branch_regions,
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
        self, node: NodeType | None = None
    ) -> list[PartialPointType]:
        """
        :param node: A mounting node to sample below, instead of the whole class
            circuit.
        :return: The distinct sampled assignments of the undetermined latents.
        :raises InvalidMonteCarloSampleCountError: If the sample count is not positive.
        :raises UndeterminedLatentsNotModeledError: If the circuit does not model the
            undetermined latents.
        """
        if self.monte_carlo_sample_count < 1:
            raise InvalidMonteCarloSampleCountError(self.monte_carlo_sample_count)
        source = self.circuit if node is None else self.circuit.subcircuit_at(node)
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
        self, node: NodeType, latent_assignments: list[PartialPointType]
    ) -> list[float]:
        """
        :param node: A mounting node.
        :param latent_assignments: Assignments of the undetermined latents.
        :return: The log-likelihood of every assignment under the subcircuit of the
            node.
        """
        subcircuit = self.circuit.subcircuit_at(node)
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
        self, node: NodeType, sampled_assignments: list[PartialPointType]
    ) -> WeightedAssignments:
        """
        :param node: A mounting node.
        :param sampled_assignments: The assignments sampled from the whole circuit.
        :return: The assignments the node mixes, weighted by their likelihood below the
            node. These are its own samples if the node rules out all of
            ``sampled_assignments``.
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
        subcircuit: CircuitType,
        undetermined_latents: SortedSet[Variable],
        branch_regions: list[Event],
    ) -> list[float]:
        """
        :param subcircuit: The subcircuit below a mounting node.
        :param undetermined_latents: The variables the partition covers.
        :param branch_regions: The support of every branch.
        :return: The log-probability of every branch under the subcircuit.
        """
        subcircuit.marginal_in_place(undetermined_latents)
        with np.errstate(divide="ignore"):
            return [
                float(np.log(subcircuit.probability(region)))
                for region in branch_regions
            ]

    @staticmethod
    def _distinct_component_regions(proposal: CircuitType) -> list[Event]:
        """
        :param proposal: The class circuit marginalized to the undetermined statistics.
        :return: The distinct supports of the components its root mixes, or its own
            support if the root does not mix. Components that the fitted circuit only
            separates by other variables have the same support here.
        """
        components = proposal.root_mixture_components() or [proposal]
        result = []
        for component in components:
            if component.support not in result:
                result.append(component.support)
        return result

    @staticmethod
    def _undetermined_latents_partition_disjointly(
        branch_regions: list[Event],
    ) -> bool:
        """
        :param branch_regions: The support of every branch.
        :return: Whether no two of the regions overlap and every instance can be tied to
            its region: there are at least two regions, or the only one is a point.
        """
        if len(branch_regions) == 1:
            return branch_regions[0].is_singleton()
        return all(
            left.intersection_with(right).is_empty()
            for left, right in itertools.combinations(branch_regions, 2)
        )

    @staticmethod
    def _representative_value(
        branch: CircuitType, undetermined_latents: SortedSet[Variable]
    ) -> PartialPointType:
        """
        :param branch: A branch of the exact partition.
        :param undetermined_latents: The variables the partition covers.
        :return: A point in the mode of the branch, to condition the instance of the
            branch on.
        """
        values = {}
        for variable in undetermined_latents:
            mode = branch.marginal([variable]).log_mode(check_determinism=False).event
            region = mode.simple_sets[0][variable]

            values[variable] = (
                region.simple_sets[0].lower
                if isinstance(region, Interval)
                else next(iter(region))
            )
        return values


# %% grounding a relational circuit


@dataclass
class RelationalGrounding:
    """
    Grounds a relational probabilistic circuit for queries, in the circuit type it was
    fitted in.
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

    def ground(self, query: Match) -> GroundableCircuit:
        """
        :param query: An underspecified, resolved query whose structure determines which
            parts are grounded and how many child objects every exchangeable relation
            contains.
        :return: The grounded circuit over all variables implied by the query.
        :raises CircuitNotFittedError: If the relational circuit is not fitted.
        """
        relational_circuit = self.relational_circuit
        if relational_circuit.class_probabilistic_circuit is None:
            raise CircuitNotFittedError()
        circuit = copy.deepcopy(relational_circuit.class_probabilistic_circuit)
        if not relational_circuit.exchangeable_distribution_templates:
            return circuit
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
        circuit: GroundableCircuit,
        exchangeable_part_name: str,
        template: ExchangeableDistributionTemplate,
        query: Match,
        instance: Any,
    ) -> ExchangeablePartGrounder:
        """
        :param circuit: The working class circuit.
        :param exchangeable_part_name: Field name of the exchangeable relation.
        :param template: The fitted template of the relation.
        :param query: The grounding query.
        :param instance: The instance constructed from the query.
        :return: The grounder of the part, with the templates of its child objects
            grounded.
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
        return ExchangeablePartGrounder(
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
        return variable.make_value(value).is_singleton()
