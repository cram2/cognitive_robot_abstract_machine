"""
Region graphs, the random partitions of variables that randomized and tensorized sum-
product networks (RAT-SPNs) are built from.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

import numpy as np
import rustworkx as rx
from random_events.interval import SimpleInterval
from random_events.variable import Continuous, Symbolic, Variable
from sortedcontainers import SortedSet
from typing_extensions import Iterable, Self

from probabilistic_model.graph import DirectedAcyclicGraph
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
from probabilistic_model.probabilistic_circuit.tensorized.folding import LayerFolding
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    DenseEdgeMatrix,
    SparseEntries,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.product_layer import (
    ProductLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.discrete_layer import (
    SymbolicLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table import (
    DenseProbabilityTable,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.learning.region_graph.exceptions import (
    MultipleRootsError,
    PredictedVariableInRegionGraphError,
    UnsupportedRegionVariableError,
)


# %% nodes


@dataclass(eq=False, kw_only=True)
class RegionGraphNode(ABC):
    """
    A node of a region graph, compared by identity.
    """

    layer: Layer | None = field(default=None, init=False, repr=False)
    """
    The layer that models this node in the circuit of the region graph, once
    :meth:`RegionGraph.to_probabilistic_circuit` built it: the fully factorized input
    distributions of a leaf region, the sum layer of any other region and the cross
    products of a partition.
    """


@dataclass(eq=False, kw_only=True)
class Region(RegionGraphNode):
    """
    A region in a region graph.

    A region is a set of variables. Refer to this paper
    https://ml-research.github.io/papers/peharz2019uai_ratspns.pdf
    .
    """

    variables: SortedSet
    """
    The variables in the region.
    """

    def random_partition(
        self, *, random_number_generator: np.random.Generator, number_of_parts: int = 2
    ) -> list[Self]:
        """
        :param random_number_generator: The generator that shuffles the variables.
        :param number_of_parts: The number of regions to split this region into, at
            most one per variable.
        :return: Non-empty regions of nearly equal size whose variables partition the
            variables of this region at random.
        """
        indices = random_number_generator.permutation(len(self.variables))
        return [
            Region(variables=SortedSet([self.variables[index] for index in split]))
            for split in np.array_split(
                indices, min(number_of_parts, len(self.variables))
            )
        ]

    def __repr__(self) -> str:
        """
        :return: The names of the variables of this region as a set.
        """
        return "{" + ", ".join([v.name for v in self.variables]) + "}"


@dataclass(eq=False, kw_only=True)
class Partition(RegionGraphNode):
    """
    A partition in a region graph.
    """


# %% region graph


@dataclass(eq=False, kw_only=True)
class RegionGraph(DirectedAcyclicGraph[RegionGraphNode]):
    """
    A region graph is a directed acyclic bipartite graph that represents a (potentially
    repeated) partition of variables, built as in Algorithm 1 of
    https://ml-research.github.io/papers/peharz2019uai_ratspns.pdf .
    """

    variables: SortedSet
    """
    The variables of the region graph.
    """

    partitions: int = 2
    """
    The number of regions every partition splits its region into.
    """

    depth: int = 2
    """
    The number of times each repetition splits the root region and the regions it
    produces.
    """

    repetitions: int = 2
    """
    The number of repetitions to create in the region graph.

    A repetition is another partitioning on the set of variables.
    """

    seed: int = 69
    """
    The seed of :attr:`random_number_generator`, which fixes the graph and the initial
    parameters of its circuits.
    """

    initial_weight_range: SimpleInterval = field(
        default_factory=lambda: SimpleInterval.from_data(lower=0.1, upper=1.0)
    )
    """
    The interval the initial weights of the sum units of a circuit are drawn uniformly
    from.
    """

    initial_location_range: SimpleInterval = field(
        default_factory=lambda: SimpleInterval.from_data(lower=-1.0, upper=1.0)
    )
    """
    The interval the initial means of the Gaussian input distributions of a circuit are
    drawn uniformly from.
    """

    initial_scale_range: SimpleInterval = field(
        default_factory=lambda: SimpleInterval.from_data(lower=0.5, upper=3.0)
    )
    """
    The interval the initial standard deviations of the Gaussian input distributions of
    a circuit are drawn uniformly from.
    """

    initial_probability_range: SimpleInterval = field(
        default_factory=lambda: SimpleInterval.from_data(lower=0.1, upper=1.0)
    )
    """
    The interval the initial probabilities of the categorical input distributions of a
    circuit are drawn uniformly from, before they are normalized.
    """

    random_number_generator: np.random.Generator = field(init=False, repr=False)
    """
    The generator that draws the random partitions and the initial parameters of the
    circuits.
    """

    def __post_init__(self) -> None:
        """
        Seed :attr:`random_number_generator` with :attr:`seed`.
        """
        self.random_number_generator = np.random.default_rng(self.seed)

    def create_random_region_graph(self) -> Self:
        """
        Split the root region :attr:`repetitions` times, in place.

        :return: This region graph.
        """
        root = Region(variables=self.variables)
        self.add_node(root)
        for repetition in range(self.repetitions):
            self.random_split(region=root, depth=self.depth)
        return self

    def regions(self) -> Iterable[Region]:
        """
        :return: An iterator over all regions.
        """
        for node in self.nodes():
            if isinstance(node, Region):
                yield node

    def partition_nodes(self) -> Iterable[Partition]:
        """
        :return: An iterator over all partitions.
        """
        for node in self.nodes():
            if isinstance(node, Partition):
                yield node

    def random_split(self, *, region: Region, depth: int) -> None:
        """
        Add a random partition of a region and split its regions again, until
        ``depth`` splits are made or a region holds a single variable, which stays a
        leaf.

        :param region: The region to split.
        :param depth: The number of splits left.
        """
        if depth < 1 or len(region.variables) < 2:
            return
        partition = Partition()
        self.add_edge(region, partition)
        for part in region.random_partition(
            random_number_generator=self.random_number_generator,
            number_of_parts=self.partitions,
        ):
            self.add_edge(partition, part)
            self.random_split(region=part, depth=depth - 1)

    @property
    def root(self) -> Region:
        """
        :return: The root of the region graph.
        """
        possible_roots = [node for node in self.nodes() if self.in_degree(node) == 0]
        if len(possible_roots) > 1:
            raise MultipleRootsError(roots=possible_roots)
        return possible_roots[0]

    # %% circuits

    def to_probabilistic_circuit(
        self,
        *,
        input_units: int = 5,
        sum_units: int = 5,
        predicted_variable: Symbolic | None = None,
    ) -> LayeredProbabilisticCircuit:
        """
        Convert the region graph to a randomly initialized RAT-SPN with numpy
        parameters, folded by :class:`LayerFolding`. With a predicted variable, the root region has one sum unit per category of it
        and the root mixes them, each multiplied by the indicator of its category, so
        that the circuit classifies; without one, the root region has a single sum unit.

        :param input_units: The number of input distributions of each leaf region.
        :param sum_units: The number of sum units of each inner region.
        :param predicted_variable: The symbolic variable the circuit classifies into
            its categories, if any.
        :return: The layered circuit.
        :raises PredictedVariableInRegionGraphError: If the predicted variable is one of
            :attr:`variables`.
        """
        if predicted_variable in self.variables:
            raise PredictedVariableInRegionGraphError(
                predicted_variable=predicted_variable
            )
        variables = SortedSet(self.variables)
        number_of_root_units = 1
        if predicted_variable is not None:
            variables = variables | SortedSet([predicted_variable])
            number_of_root_units = len(predicted_variable.domain.simple_sets)

        for layer in reversed(rx.bfs_layers(self.graph, [self.index_of(self.root)])):
            for node in map(self.graph.__getitem__, layer):
                children = self.successors(node)
                if isinstance(node, Partition):
                    node.layer = ProductLayer.cross_product(
                        child_layers=[child.layer for child in children]
                    )
                elif children:
                    node.layer = self.fully_connected_sum_layer(
                        children=children,
                        number_of_nodes=(
                            number_of_root_units
                            if self.in_degree(node) == 0
                            else sum_units
                        ),
                    )
                else:
                    node.layer = self.fully_factorized_input_layer(
                        region=node,
                        variables=variables,
                        input_units=input_units,
                    )

        root = self.root.layer
        if not self.successors(self.root):
            root = self.fully_connected_sum_layer(
                children=[self.root],
                number_of_nodes=number_of_root_units,
            )
        if predicted_variable is not None:
            root = self.mixture_of_categories(
                category_conditionals=root,
                predicted_variable=predicted_variable,
                variables=variables,
            )
        return LayeredProbabilisticCircuit(
            variables, LayerFolding(root=root).folded_root()
        )

    def fully_connected_sum_layer(
        self, *, children: list[RegionGraphNode], number_of_nodes: int
    ) -> SumLayer:
        """
        :param children: The nodes whose layers the sum units mix.
        :param number_of_nodes: The number of sum units.
        :return: The sum layer that connects every unit to every unit of every child,
            with dense edges and weights drawn from :attr:`initial_weight_range`.
        """
        child_layers = [child.layer for child in children]
        number_of_columns = sum(layer.number_of_nodes for layer in child_layers)
        rows, columns = np.nonzero(np.ones((number_of_nodes, number_of_columns)))
        weights = self.random_number_generator.uniform(
            self.initial_weight_range.lower,
            self.initial_weight_range.upper,
            len(rows),
        )
        return SumLayer.from_entries(
            child_layers,
            SparseEntries(np.log(weights), rows, columns),
            number_of_nodes,
            DenseEdgeMatrix,
        )

    def fully_factorized_input_layer(
        self,
        *,
        region: Region,
        variables: SortedSet,
        input_units: int,
    ) -> Layer:
        """
        :param region: A leaf region.
        :param variables: The variables of the circuit.
        :param input_units: The number of input distributions.
        :return: The layer whose unit ``i`` is the product of the ``i``-th univariate
            input distribution of every variable of the region.
        """
        univariate_layers = [
            self.input_layer(
                variable=variable, variables=variables, input_units=input_units
            )
            for variable in region.variables
        ]
        if len(univariate_layers) == 1:
            return univariate_layers[0]
        return ProductLayer.node_wise_product_of(univariate_layers)

    def input_layer(
        self, *, variable: Variable, variables: SortedSet, input_units: int
    ) -> GaussianLayer | SymbolicLayer:
        """
        :param variable: The variable of a leaf region.
        :param variables: The variables of the circuit.
        :param input_units: The number of input units.
        :return: Gaussians with means drawn from :attr:`initial_location_range` and
            standard deviations from :attr:`initial_scale_range` for a continuous
            variable, categorical distributions with probabilities drawn from
            :attr:`initial_probability_range` and normalized for a symbolic one.
        :raises UnsupportedRegionVariableError: If the variable is of another type.
        """
        variable_index = variables.index(variable)
        if isinstance(variable, Continuous):
            return GaussianLayer(
                np.full(input_units, variable_index, dtype=np.int64),
                self.random_number_generator.uniform(
                    self.initial_location_range.lower,
                    self.initial_location_range.upper,
                    input_units,
                ),
                np.log(
                    self.random_number_generator.uniform(
                        self.initial_scale_range.lower,
                        self.initial_scale_range.upper,
                        input_units,
                    )
                ),
            )
        if isinstance(variable, Symbolic):
            number_of_states = len(variable.domain.simple_sets)
            probabilities = self.random_number_generator.uniform(
                self.initial_probability_range.lower,
                self.initial_probability_range.upper,
                (input_units, number_of_states),
            )
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            return SymbolicLayer(
                np.full(input_units, variable_index, dtype=np.int64),
                np.arange(number_of_states),
                DenseProbabilityTable(np.log(probabilities)),
            )
        raise UnsupportedRegionVariableError(variable=variable)

    @staticmethod
    def mixture_of_categories(
        *,
        category_conditionals: Layer,
        predicted_variable: Symbolic,
        variables: SortedSet,
    ) -> SumLayer:
        """
        :param category_conditionals: The layer with one node per category of the
            predicted variable.
        :param predicted_variable: The variable the circuit predicts.
        :param variables: The variables of the circuit.
        :return: The root that mixes, with equal weights, the product of every category
            conditional node with the indicator of its category.
        """
        number_of_categories = len(predicted_variable.domain.simple_sets)
        with np.errstate(divide="ignore"):
            indicators = SymbolicLayer(
                np.full(
                    number_of_categories,
                    variables.index(predicted_variable),
                    dtype=np.int64,
                ),
                np.arange(number_of_categories),
                DenseProbabilityTable(np.log(np.eye(number_of_categories))),
            )
        products = ProductLayer.node_wise_product_of(
            [category_conditionals, indicators]
        )
        return SumLayer.from_entries(
            [products],
            SparseEntries(
                np.full(number_of_categories, -np.log(number_of_categories)),
                np.zeros(number_of_categories, dtype=np.int64),
                np.arange(number_of_categories),
            ),
            1,
            DenseEdgeMatrix,
        )
