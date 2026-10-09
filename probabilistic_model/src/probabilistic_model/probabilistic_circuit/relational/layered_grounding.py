"""
Grounding a relational probabilistic circuit fitted as layered circuits, with the
exchangeable relations as layers.

The instances of one relation all come from the same fitted template, conditioned on
different values of the aggregation statistics. Conditioning a circuit on a point keeps
its structure and only changes its weights, so the conditioned templates are stacked
into one layer graph whose layers hold the nodes of every instance, instead of being
built one instance and one child object at a time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from sortedcontainers import SortedSet
from typing_extensions import TYPE_CHECKING, Dict, List, Optional, Type

from probabilistic_model.distributions.helper import make_dirac
from probabilistic_model.probabilistic_model import PartialPointType
from probabilistic_model.probabilistic_circuit.relational.exchangeable_grounding import (
    ExchangeablePartGrounder,
    GroundedPartTemplate,
    InstanceMixture,
    PartitionMixture,
    RelationalGrounding,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
    NodeMask,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    InnerLayer,
    Layer,
)
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


# %% the nodes of a layered circuit


@dataclass
class LayerNode:
    """
    One node of a layer of a layered circuit.
    """

    layer: Layer
    """
    The layer.
    """

    node: int
    """
    The index of the node in the layer.
    """

    def as_circuit(self, variables: SortedSet[Variable]) -> LayeredProbabilisticCircuit:
        """
        :param variables: The variables the layers refer to.
        :return: The circuit rooted at the node, sharing its layers.
        """
        root = SumLayer.from_edges(
            [self.layer],
            InnerLayerEdges(
                np.zeros(1, dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                np.array([self.node], dtype=np.int64),
            ),
            np.zeros(1),
            1,
        )
        return LayeredProbabilisticCircuit(SortedSet(variables), root)


def lowest_product_nodes_that_model_variables(
    circuit: LayeredProbabilisticCircuit, variables: SortedSet[Variable]
) -> List[LayerNode]:
    """
    :param circuit: The circuit to search.
    :param variables: The variables every returned node must model.
    :return: The product nodes that model all of ``variables`` and have no such node
        below them, so none of them is an ancestor of another.
    """
    variable_indices = [circuit.variables.index(variable) for variable in variables]
    result = []
    contains_found: Dict[int, NodeMask] = {}
    # children before parents
    for layer in reversed(circuit.layers):
        if not isinstance(layer, InnerLayer):
            contains_found[id(layer)] = np.zeros(layer.number_of_nodes, dtype=bool)
            continue
        edges = layer.inner_layer_edges
        found_below = np.zeros(len(edges), dtype=bool)
        for child_layer_index, child_layer in enumerate(layer.child_layers):
            of_child = edges.child_layer_indices == child_layer_index
            found_below[of_child] = contains_found[id(child_layer)][
                edges.child_nodes[of_child]
            ]
        below = np.zeros(layer.number_of_nodes, dtype=bool)
        below[edges.nodes[found_below]] = True
        if isinstance(layer, ProductLayer):
            covers = np.ones(layer.number_of_nodes, dtype=bool)
            for variable_index in variable_indices:
                models_variable = np.array(
                    [
                        variable_index in child_layer.variables
                        for child_layer in layer.child_layers
                    ],
                    dtype=bool,
                )
                covers_variable = np.zeros(layer.number_of_nodes, dtype=bool)
                covers_variable[
                    edges.nodes[models_variable[edges.child_layer_indices]]
                ] = True
                covers &= covers_variable
            found = covers & ~below
            result.extend(LayerNode(layer, int(node)) for node in np.flatnonzero(found))
            below |= found
        contains_found[id(layer)] = below
    return result


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

    branches: List[LayeredProbabilisticCircuit] = field(default_factory=list)
    """
    The branch of every instance.
    """

    def factors_in(
        self, variables: SortedSet[Variable], number_of_instances: int
    ) -> List[InstanceFactor]:
        factors = []
        for instance, branch in enumerate(self.branches):
            branch = branch.__deepcopy__()
            branch.restore_variables(variables)
            factors.append(
                InstanceFactor(
                    branch.root,
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


# %% mounting the instances into the class circuit


@dataclass
class LayeredExchangeablePartGrounder(
    ExchangeablePartGrounder[LayeredProbabilisticCircuit, LayerNode]
):
    """
    Grounds one exchangeable part by attaching the layer of its instances to the layered
    class circuit.
    """

    def condition_class_circuit(self):
        circuit = self.circuit
        statistics = {
            variable: value
            for variable, value in self.determined_statistics.items()
            if variable in circuit.variables
        }
        if not statistics:
            return
        query = StructuralQuery(circuit.variables)
        conditioned = circuit.root.log_conditional_of_point(
            circuit.encoded_point(statistics), query, cache=QueryCache()
        )
        if conditioned.log_probabilities[0] == -np.inf:
            return
        circuit.root = conditioned.layer.prune(query.log_probabilities)
        circuit.normalize()

    def mounting_nodes(self) -> List[LayerNode]:
        return lowest_product_nodes_that_model_variables(
            self.circuit, SortedSet(self.template.latent_variables)
        )

    def circuit_below(self, node: LayerNode) -> LayeredProbabilisticCircuit:
        return node.as_circuit(self.circuit.variables)

    def remove_undetermined_latents(self):
        circuit = self.circuit
        kept = np.array(
            [
                variable not in self.undetermined_latents
                for variable in circuit.variables
            ]
        )
        cache = QueryCache()
        circuit.root = circuit.root.marginal(kept, cache=cache)
        self.product_nodes_to_extend = [
            LayerNode(cache.result_of(ProductLayer.marginal, node.layer), node.node)
            for node in self.product_nodes_to_extend
        ]

    @staticmethod
    def partition_branches(
        proposal: LayeredProbabilisticCircuit,
    ) -> List[LayeredProbabilisticCircuit]:
        root = proposal.root
        if not isinstance(root, SumLayer) or root.number_of_nodes != 1:
            return []
        edges = root.inner_layer_edges
        return [
            LayerNode(root.child_layers[child_layer_index], int(child_node)).as_circuit(
                proposal.variables
            )
            for child_layer_index, child_node in zip(
                edges.child_layer_indices, edges.child_nodes
            )
        ]

    def single_instance(self):
        self.attach(
            self.instances(
                [self.determined_statistics], NoRetainedLatents(SortedSet())
            ),
            None,
        )

    def sampled_mixture(self, mixture: InstanceMixture):
        self.mixed_part(
            mixture, SampledLatents(self.undetermined_latents, mixture.assignments)
        )

    def partition_mixture(self, mixture: PartitionMixture[LayeredProbabilisticCircuit]):
        self.mixed_part(
            mixture, PartitionBranches(self.undetermined_latents, mixture.branches)
        )

    def mixed_part(self, mixture: InstanceMixture, retained_latents: RetainedLatents):
        """
        Make every mounting node multiply its own normalized mixture of the instances.

        :param mixture: The assignments of the undetermined statistics, with their
            weights at every mounting node.
        :param retained_latents: How the instances keep the undetermined statistics.
        """
        assignments = [
            {**self.determined_statistics, **assignment}
            for assignment in mixture.assignments
        ]
        self.attach(self.instances(assignments, retained_latents), mixture)

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

    def attach(
        self,
        instances: LayeredExchangeableInstances,
        mixture: Optional[InstanceMixture],
    ):
        """
        Make every mounting node multiply its mixture of the instances, in place.

        :param instances: The instances of the part.
        :param mixture: The weights of the instances at every mounting node, or ``None``
            when there is only one instance, which every mounting node multiplies.
        """
        variables = SortedSet(self.circuit.variables) | instances.variables
        self.circuit.restore_variables(variables)
        instance_layer = instances.instance_layer(variables)
        number_of_mounting_nodes = len(self.product_nodes_to_extend)
        if mixture is None:
            attached = instance_layer
            child_nodes = np.zeros(number_of_mounting_nodes, dtype=np.int64)
        else:
            used = np.isfinite(mixture.log_weights)
            nodes, instance_indices = np.nonzero(used)
            attached = SumLayer.from_edges(
                [instance_layer],
                InnerLayerEdges(
                    nodes, np.zeros(len(nodes), dtype=np.int64), instance_indices
                ),
                mixture.log_weights[used],
                number_of_mounting_nodes,
            )
            attached.normalize_own()
            child_nodes = np.arange(number_of_mounting_nodes)

        nodes_per_layer: Dict[int, tuple[ProductLayer, List[int], List[int]]] = {}
        for mounting_index, mounting_node in enumerate(self.product_nodes_to_extend):
            layer, nodes, children = nodes_per_layer.setdefault(
                id(mounting_node.layer), (mounting_node.layer, [], [])
            )
            nodes.append(mounting_node.node)
            children.append(child_nodes[mounting_index])
        for layer, nodes, children in nodes_per_layer.values():
            layer.attach_child_layer(attached, np.array(nodes), np.array(children))
        self.circuit.reset_scopes()


# %% grounding a relational circuit


@dataclass
class LayeredGrounding(RelationalGrounding[LayeredProbabilisticCircuit]):
    """
    Grounds a relational probabilistic circuit fitted as layered circuits.
    """

    @property
    def part_grounder_type(self) -> Type[LayeredExchangeablePartGrounder]:
        return LayeredExchangeablePartGrounder
