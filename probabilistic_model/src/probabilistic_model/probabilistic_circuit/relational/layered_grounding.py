"""
Grounding a relational probabilistic circuit into a layered circuit, with the
exchangeable relations as layers.

The instances of one relation all come from the same fitted template, conditioned on
different values of the aggregation statistics. Conditioning a circuit on a point keeps
its structure and only changes its weights, so the conditioned templates are stacked
into one layer graph whose layers hold the nodes of every instance, instead of being
built one instance and one child object at a time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

import numpy as np
from sortedcontainers import SortedSet
from typing_extensions import TYPE_CHECKING, Dict, List, Optional, Type

from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    ConvertedCircuit,
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.distributions.helper import make_dirac
from probabilistic_model.probabilistic_model import PartialPointType
from probabilistic_model.probabilistic_circuit.relational.exchangeable_grounding import (
    ExchangeablePartGrounder,
    GroundedPartTemplate,
    InstanceMixture,
    PartitionMixture,
    RelationalGrounding,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    Unit,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.inner_layer_edge import (
    InnerLayerEdges,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.product_layer import (
    ProductLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.query_cache import QueryCache
from probabilistic_model.probabilistic_circuit.tensorized.stacked_copies import (
    AlignedCopiesStacker,
    StackedLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    StructuralQuery,
)
from random_events.variable import Variable

if TYPE_CHECKING:
    from probabilistic_model.probabilistic_circuit.relational.rspn import (
        ExchangeableDistributionTemplate,
    )


# %% the latent statistics an instance carries


@dataclass
class InstanceFactor:
    """
    One child layer of the layer of instances, with the node every instance multiplies.
    """

    layer: Layer
    """
    The child layer.
    """

    instances: NodeIndices
    """
    The instances that multiply a node of the child layer.
    """

    nodes: NodeIndices
    """
    The node of the child layer every one of ``instances`` multiplies.
    """


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
    def factors_in(
        self, variables: SortedSet[Variable], number_of_instances: int
    ) -> List[InstanceFactor]:
        """
        :param variables: The variables of the grounded circuit.
        :param number_of_instances: The number of instances.
        :return: The layers that carry the statistics, one factor of every instance.
        """
        raise NotImplementedError


@dataclass
class NoRetainedLatents(RetainedLatents):
    """
    The query determines every aggregation statistic, so there is nothing to retain.
    """

    def factors_in(
        self, variables: SortedSet[Variable], number_of_instances: int
    ) -> List[InstanceFactor]:
        return []


@dataclass
class SampledLatents(RetainedLatents):
    """
    Every instance carries the sampled values it was grounded on, as point masses.
    """

    assignments: List[PartialPointType] = field(default_factory=list)
    """
    The sampled values of every instance.
    """

    def factors_in(
        self, variables: SortedSet[Variable], number_of_instances: int
    ) -> List[InstanceFactor]:
        instances = np.arange(number_of_instances)
        return [
            InstanceFactor(
                LayeredProbabilisticCircuit.point_masses_layer(
                    variables.index(variable),
                    [
                        make_dirac(variable, assignment[variable])
                        for assignment in self.assignments
                    ],
                ),
                instances,
                instances,
            )
            for variable in self.variables
        ]


@dataclass
class PartitionBranches(RetainedLatents):
    """
    Every instance carries the branch of the latents' exact partition it was grounded
    on.
    """

    branches: List[Unit] = field(default_factory=list)
    """
    The branch of every instance.
    """

    def factors_in(
        self, variables: SortedSet[Variable], number_of_instances: int
    ) -> List[InstanceFactor]:
        factors = []
        for instance, branch in enumerate(self.branches):
            branch_circuit = ProbabilisticCircuit()
            branch_circuit.mount(branch)
            layered = RustworkxCircuitToLayeredCircuitConverter.convert(branch_circuit)
            layered.restore_variables(variables)
            factors.append(
                InstanceFactor(
                    layered.root,
                    np.array([instance]),
                    np.zeros(1, dtype=np.int64),
                )
            )
        return factors


# %% the instances of one relation


@dataclass
class LayeredExchangeableInstances:
    """
    The instances of one exchangeable relation as one layer: instance ``i`` multiplies
    one copy of the template per child object, conditioned on the ``i``-th assignment of
    the aggregation statistics, and whatever retains the undetermined statistics.
    """

    template: ExchangeableDistributionTemplate
    """
    The fitted template of the relation.
    """

    parts: List[GroundedPartTemplate[LayeredProbabilisticCircuit]]
    """
    The grounded template of every child object of the relation.
    """

    assignments: List[PartialPointType]
    """
    The values of every aggregation statistic, one assignment per instance.
    """

    retained_latents: RetainedLatents
    """
    How the instances keep the statistics the query leaves undetermined.
    """

    @property
    def variables(self) -> SortedSet[Variable]:
        """
        :return: The variables the instances add to the grounded circuit.
        """
        result = SortedSet(self.retained_latents.variables)
        for part in self.parts:
            result.update(
                self.template.variable_of_part(variable, part)
                for variable in part.circuit.variables
                if variable not in self.template.latent_variables
            )
        return result

    def conditioned_template(
        self, circuit: LayeredProbabilisticCircuit
    ) -> StackedLayer:
        """
        Condition the template of a child object on every assignment, remove the
        aggregation statistics and stack the results.

        :param circuit: The template of the child object.
        :return: The stacked templates, with node ``i`` of the root belonging to the
            ``i``-th assignment, over the variables of ``circuit``.
        """
        kept = np.array(
            [
                variable not in self.template.latent_variables
                for variable in circuit.variables
            ]
        )
        copies = []
        for assignment in self.assignments:
            conditioned = circuit.root.log_conditional_of_point(
                circuit.encoded_point(assignment),
                StructuralQuery(circuit.variables),
                cache=QueryCache(),
            )
            # an assignment the template deems impossible leaves the template as it is
            impossible = conditioned.log_probabilities[0] == -np.inf
            source = circuit.root if impossible else conditioned.layer
            copies.append(source.marginal(kept))
        stacked = AlignedCopiesStacker().stack(copies)
        stacked.layer.normalize()
        return stacked

    def instance_layer(self, variables: SortedSet[Variable]) -> ProductLayer:
        """
        :param variables: The variables of the grounded circuit.
        :return: The layer of instances, one node per assignment.
        """
        number_of_instances = len(self.assignments)
        instances = np.arange(number_of_instances)
        stacked_of_circuit: Dict[int, StackedLayer] = {}

        factors = []
        for part in self.parts:
            circuit = part.circuit
            if id(circuit) not in stacked_of_circuit:
                stacked_of_circuit[id(circuit)] = self.conditioned_template(circuit)
            stacked = stacked_of_circuit[id(circuit)]
            remap = np.array(
                [
                    (
                        -1
                        if variable in self.template.latent_variables
                        else variables.index(
                            self.template.variable_of_part(variable, part)
                        )
                    )
                    for variable in circuit.variables
                ],
                dtype=np.int64,
            )
            child_object = stacked.layer.__deepcopy__({})
            child_object.remap_variables(remap)
            root_of_instance = stacked.nodes_of_copy(
                instances, np.zeros(number_of_instances, dtype=np.int64)
            )
            factors.append(InstanceFactor(child_object, instances, root_of_instance))
        factors.extend(self.retained_latents.factors_in(variables, number_of_instances))

        edges = InnerLayerEdges(
            np.concatenate([factor.instances for factor in factors]),
            np.concatenate(
                [
                    np.full(len(factor.instances), factor_index, dtype=np.int64)
                    for factor_index, factor in enumerate(factors)
                ]
            ),
            np.concatenate([factor.nodes for factor in factors]),
        )
        return ProductLayer.from_edges(
            [factor.layer for factor in factors], edges, number_of_instances
        )


# %% attaching the instances to the class circuit


@dataclass
class LayeredExchangeablePart:
    """
    One exchangeable relation of a grounding, waiting to be attached to the layered
    class circuit.
    """

    instances: LayeredExchangeableInstances
    """
    The instances of the relation.
    """

    mounting_nodes: List[ProductUnit]
    """
    The product units of the class circuit that get the relation as a factor.
    """

    mixture: Optional[InstanceMixture] = None
    """
    The weights of the instances at every mounting node, or ``None`` when there is only
    one instance, which every mounting node multiplies.
    """

    def without_removed_mounting_nodes(
        self, circuit: ProbabilisticCircuit
    ) -> LayeredExchangeablePart:
        """
        :param circuit: The class circuit after conditioning on the statistics of every
            exchangeable relation.
        :return: The part without the mounting nodes that conditioning on the statistics
            of a later relation removed from the circuit, since nothing multiplies them
            anymore.
        """
        units_of_circuit = {id(unit) for unit in circuit.nodes()}
        kept = np.array(
            [id(unit) in units_of_circuit for unit in self.mounting_nodes], dtype=bool
        )
        mixture = (
            None
            if self.mixture is None
            else replace(self.mixture, log_weights=self.mixture.log_weights[kept])
        )
        mounting_nodes = [
            unit for unit, is_kept in zip(self.mounting_nodes, kept) if is_kept
        ]
        return LayeredExchangeablePart(self.instances, mounting_nodes, mixture)

    def attach_to(self, converted: ConvertedCircuit, variables: SortedSet[Variable]):
        """
        Make every mounting node multiply its mixture of the instances, in place.

        :param converted: The layered class circuit, with the layers its units became.
        :param variables: The variables of the grounded circuit, which the class circuit
            already refers to.
        """
        instance_layer = self.instances.instance_layer(variables)
        number_of_mounting_nodes = len(self.mounting_nodes)
        if self.mixture is None:
            attached = instance_layer
            child_nodes = np.zeros(number_of_mounting_nodes, dtype=np.int64)
        else:
            used = np.isfinite(self.mixture.log_weights)
            nodes, instances = np.nonzero(used)
            attached = SumLayer.from_edges(
                [instance_layer],
                InnerLayerEdges(nodes, np.zeros(len(nodes), dtype=np.int64), instances),
                self.mixture.log_weights[used],
                number_of_mounting_nodes,
            )
            attached.normalize_own()
            child_nodes = np.arange(number_of_mounting_nodes)

        nodes_per_layer: Dict[int, tuple[ProductLayer, List[int], List[int]]] = {}
        for mounting_index, unit in enumerate(self.mounting_nodes):
            target = converted.node_of(unit)
            layer, nodes, children = nodes_per_layer.setdefault(
                id(target.layer), (target.layer, [], [])
            )
            nodes.append(target.node)
            children.append(child_nodes[mounting_index])
        for layer, nodes, children in nodes_per_layer.values():
            layer.attach_child_layer(attached, np.array(nodes), np.array(children))


# %% weighing the instances at the class circuit


@dataclass
class LayeredExchangeablePartGrounder(
    ExchangeablePartGrounder[LayeredProbabilisticCircuit, LayeredExchangeablePart]
):
    """
    Grounds one exchangeable part into the layers of its instances, to be attached to
    the layered class circuit.
    """

    def single_instance(self) -> LayeredExchangeablePart:
        return LayeredExchangeablePart(
            self.instances(
                [self.determined_statistics], NoRetainedLatents(SortedSet())
            ),
            self.product_nodes_to_extend,
        )

    def sampled_mixture(self, mixture: InstanceMixture) -> LayeredExchangeablePart:
        return self.mixed_part(
            mixture, SampledLatents(self.undetermined_latents, mixture.assignments)
        )

    def partition_mixture(self, mixture: PartitionMixture) -> LayeredExchangeablePart:
        return self.mixed_part(
            mixture, PartitionBranches(self.undetermined_latents, mixture.branches)
        )

    def mixed_part(
        self, mixture: InstanceMixture, retained_latents: RetainedLatents
    ) -> LayeredExchangeablePart:
        """
        :param mixture: The assignments of the undetermined statistics, with their
            weights at every mounting node.
        :param retained_latents: How the instances keep the undetermined statistics.
        :return: The part as one instance per assignment, mixed at every mounting node.
        """
        assignments = [
            {**self.determined_statistics, **assignment}
            for assignment in mixture.assignments
        ]
        return LayeredExchangeablePart(
            self.instances(assignments, retained_latents),
            self.product_nodes_to_extend,
            mixture,
        )

    def instances(
        self,
        assignments: List[PartialPointType],
        retained_latents: RetainedLatents,
    ) -> LayeredExchangeableInstances:
        """
        :param assignments: The values of every aggregation statistic, one assignment
            per instance.
        :param retained_latents: How the instances keep the undetermined statistics.
        :return: The instances of the part.
        """
        return LayeredExchangeableInstances(
            self.template, self.part_templates, assignments, retained_latents
        )


# %% grounding a relational circuit


@dataclass
class LayeredGrounding(
    RelationalGrounding[LayeredProbabilisticCircuit, LayeredExchangeablePart]
):
    """
    Grounds a relational probabilistic circuit into a layered circuit.

    The instances of an exchangeable relation are the fitted template conditioned on
    different aggregation statistics, which only changes its weights, so they are built
    as one stack of layers rather than one circuit per instance and child object.
    """

    @property
    def part_grounder_type(self) -> Type[LayeredExchangeablePartGrounder]:
        return LayeredExchangeablePartGrounder

    def grounded_circuit(
        self, circuit: ProbabilisticCircuit, parts: List[LayeredExchangeablePart]
    ) -> LayeredProbabilisticCircuit:
        parts = [part.without_removed_mounting_nodes(circuit) for part in parts]
        converted = RustworkxCircuitToLayeredCircuitConverter.convert_with_layers(
            circuit
        )
        variables = SortedSet(converted.circuit.variables)
        for part in parts:
            variables.update(part.instances.variables)
        converted.circuit.restore_variables(variables)
        for part in parts:
            part.attach_to(converted, variables)
        converted.circuit.reset_scopes()
        return converted.circuit
