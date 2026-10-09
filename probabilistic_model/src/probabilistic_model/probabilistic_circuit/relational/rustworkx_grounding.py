"""
Grounding a relational probabilistic circuit fitted as rustworkx circuits, by mounting
the instances of its exchangeable relations into the class circuit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from sortedcontainers import SortedSet
from typing_extensions import List, Type

from probabilistic_model.distributions.helper import make_dirac
from probabilistic_model.probabilistic_circuit.relational.exchangeable_grounding import (
    ExchangeablePartGrounder,
    InstanceMixture,
    PartitionMixture,
    RelationalGrounding,
)
from probabilistic_model.probabilistic_circuit.relational.helper import (
    find_lowest_product_nodes_that_model_variables,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    Unit,
    leaf,
)
from probabilistic_model.probabilistic_model import PartialPointType
from random_events.variable import Variable

# %% the latent statistics an instance carries


@dataclass
class RetainedLatents(ABC):
    """
    How the instances of a relation keep the aggregation statistics that the query
    leaves undetermined as variables of the grounded circuit.
    """

    variables: SortedSet[Variable]
    """
    The undetermined aggregation statistics.
    """

    @abstractmethod
    def factors_of(self, instance: int, circuit: ProbabilisticCircuit) -> List[Unit]:
        """
        :param instance: The index of an instance.
        :param circuit: The circuit to mount the factors into.
        :return: The mounted units that carry the statistics of the instance, one
            factor of the instance each.
        """
        raise NotImplementedError


@dataclass
class SampledLatents(RetainedLatents):
    """
    Every instance carries the sampled values it was grounded on, as point masses.
    """

    assignments: List[PartialPointType] = field(default_factory=list)
    """
    The sampled values of every instance.
    """

    def factors_of(self, instance: int, circuit: ProbabilisticCircuit) -> List[Unit]:
        return [
            leaf(make_dirac(variable, self.assignments[instance][variable]), circuit)
            for variable in self.variables
        ]


@dataclass
class PartitionBranches(RetainedLatents):
    """
    Every instance carries the branch of the latents' exact partition it was grounded
    on.
    """

    branches: List[ProbabilisticCircuit] = field(default_factory=list)
    """
    The branch of every instance.
    """

    def factors_of(self, instance: int, circuit: ProbabilisticCircuit) -> List[Unit]:
        root = self.branches[instance].root
        return [circuit.mount(root)[root.index]]


# %% mounting the instances into the class circuit


@dataclass
class RustworkxExchangeablePartGrounder(
    ExchangeablePartGrounder[ProbabilisticCircuit, ProductUnit]
):
    """
    Grounds one exchangeable part by mounting its instances into the rustworkx class
    circuit.
    """

    def condition_class_circuit(self):
        statistics = self.determined_statistics
        if not statistics:
            return
        statistics_circuit = self.circuit.marginal(statistics)
        if statistics_circuit is not None:
            conditioned, _ = statistics_circuit.log_conditional_in_place(statistics)
            if conditioned is None:
                return
        self.circuit.log_conditional_in_place(statistics, preserve_structure=True)

    def mounting_nodes(self) -> List[ProductUnit]:
        return find_lowest_product_nodes_that_model_variables(
            self.circuit, SortedSet(self.template.latent_variables)
        )

    def circuit_below(self, node: ProductUnit) -> ProbabilisticCircuit:
        result = ProbabilisticCircuit()
        result.mount(node)
        return result

    def remove_undetermined_latents(self):
        self.circuit.restrict_to_variables_in_place(
            SortedSet(self.circuit.variables) - self.undetermined_latents
        )

    @staticmethod
    def partition_branches(
        proposal: ProbabilisticCircuit,
    ) -> List[ProbabilisticCircuit]:
        root = proposal.root
        if not isinstance(root, SumUnit):
            return []
        branches = []
        for subcircuit in root.subcircuits:
            branch = ProbabilisticCircuit()
            branch.mount(subcircuit)
            branches.append(branch)
        return branches

    def single_instance(self):
        instance_root = self.mounted_instance(self.determined_statistics)
        for product_node in self.product_nodes_to_extend:
            product_node.add_subcircuit(instance_root)

    def sampled_mixture(self, mixture: InstanceMixture):
        self.mixed_part(
            mixture, SampledLatents(self.undetermined_latents, mixture.assignments)
        )

    def partition_mixture(self, mixture: PartitionMixture[ProbabilisticCircuit]):
        self.mixed_part(
            mixture, PartitionBranches(self.undetermined_latents, mixture.branches)
        )

    def mixed_part(self, mixture: InstanceMixture, retained_latents: RetainedLatents):
        """
        Make every mounting node multiply its own normalized sum unit over the
        instances.

        :param mixture: The assignments of the undetermined statistics, with their
            weights at every mounting node.
        :param retained_latents: How the instances keep the undetermined statistics.
        """
        instance_roots = []
        for instance, assignment in enumerate(mixture.assignments):
            instance_root = ProductUnit(probabilistic_circuit=self.circuit)
            instance_root.add_subcircuit(
                self.mounted_instance({**self.determined_statistics, **assignment})
            )
            for factor in retained_latents.factors_of(instance, self.circuit):
                instance_root.add_subcircuit(factor)
            instance_roots.append(instance_root)
        for product_node, log_weights in zip(
            self.product_nodes_to_extend, mixture.log_weights
        ):
            self._attach_mixture_to_node(
                product_node, instance_roots, log_weights.tolist()
            )

    def mounted_instance(self, aggregation_statistics: PartialPointType) -> Unit:
        """
        :param aggregation_statistics: Statistics to condition the instance on.
        :return: The root of the instance, mounted into the class circuit.
        """
        grounded = self.template.ground_instance(
            self.part_templates, aggregation_statistics
        )
        node_index_map = self.circuit.mount(grounded.root)
        return node_index_map[grounded.root.index]

    def _attach_mixture_to_node(
        self,
        product_node: ProductUnit,
        instance_roots: List[Unit],
        log_weights: List[float],
    ) -> None:
        """
        Attach a normalized sum unit over exchangeable instances to one node.

        Instances whose node-local likelihood is zero are skipped; at least one has a
        positive one, since :meth:`_node_local_assignments` draws a node's own samples
        otherwise. The instances are already mounted in ``circuit`` and shared across
        all mounting nodes; only the weighted sum-unit edges differ per node.

        :param product_node: The mounting product node to extend.
        :param instance_roots: The roots of the mounted exchangeable instances.
        :param log_weights: The node-local log-likelihood weight of each instance.
        """
        weighted_instances = [
            (instance_root, log_weight)
            for instance_root, log_weight in zip(instance_roots, log_weights)
            if log_weight > -np.inf
        ]
        sum_unit = SumUnit(probabilistic_circuit=self.circuit)
        product_node.add_subcircuit(sum_unit)
        for instance_root, log_weight in weighted_instances:
            sum_unit.add_subcircuit(instance_root, log_weight)
        sum_unit.normalize()


# %% grounding a relational circuit


@dataclass
class RustworkxGrounding(RelationalGrounding[ProbabilisticCircuit]):
    """
    Grounds a relational probabilistic circuit fitted as rustworkx circuits.
    """

    @property
    def part_grounder_type(self) -> Type[RustworkxExchangeablePartGrounder]:
        return RustworkxExchangeablePartGrounder
