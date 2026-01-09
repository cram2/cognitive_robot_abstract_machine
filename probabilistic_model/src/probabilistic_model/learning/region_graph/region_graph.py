from collections import deque

import networkx as nx
import numpy as np
from jax.experimental.sparse import BCOO
from random_events.variable import Continuous, Symbolic
from sortedcontainers import SortedSet
from typing_extensions import List, Self, Type, Iterable, Union

from ...distributions import GaussianDistribution
from ...probabilistic_circuit.jax import SparseSumLayer, ProductLayer, DenseSumLayer
from ...probabilistic_circuit.jax.discrete_layer import DiscreteLayer
from ...probabilistic_circuit.jax.gaussian_layer import GaussianLayer
from ...probabilistic_circuit.rx.probabilistic_circuit import ProbabilisticCircuit, SumUnit, ProductUnit, UnivariateContinuousLeaf
from ...probabilistic_circuit.jax.probabilistic_circuit import ProbabilisticCircuit as JPC, ClassificationCircuit
import jax.numpy as jnp
import jax.random


class Region:
    """
    A region in a region graph.
    """
    variables: SortedSet

    def __init__(self, variables: SortedSet):
        self.variables = variables

    def __hash__(self) -> int:
        return id(self)

    def random_partition(self, k=2) -> List[Self]:
        indices = np.arange(len(self.variables))
        np.random.shuffle(indices)
        partitions = [Region(SortedSet([self.variables[index] for index in split])) for split in np.array_split(indices, k)]
        return partitions

    def __repr__(self) -> str:
        return "{" + ", ".join([v.name for v in self.variables]) + "}"


class Partition:
    """
    A partition in a region graph.
    """
    def __hash__(self) -> int:
        return id(self)


class RegionGraph(nx.DiGraph):
    """
    A region graph is a directed acyclic bipartite graph that represents a (potentially repeated) partition of variables.
    Bluntly speaking, a region graph is an ensemble of random Latent-Variable trees.
    """

    variables: SortedSet
    """
    The variables of the region graph.
    """

    classes: int
    """
    The number of classes.
    If classes > 1, the circuit is a classifier, if 1 it is a generative model.
    """

    partitions: int
    """
    The number of partitions to create in each region.
    """

    depth: int
    """
    The depth of each repetition of the region graph.
    """

    repetitions: int
    """
    The number of repetitions to create in the region graph.
    A repetition is another partitioning on the set of variables.
    """

    def __init__(self, variables: SortedSet,
                 partitions: int = 2,
                 depth:int = 2,
                 repetitions:int = 2,
                 classes: int = 1,
                 ):
        super().__init__()
        self.classes = classes
        self.variables = variables
        self.partitions = partitions
        self.depth = depth
        self.repetitions = repetitions


    def create_random_region_graph(self):
        """
        Create a random region graph according to the classes parameters in-place.
        """
        root = Region(self.variables)
        self.add_node(root)
        for repetition in range(self.repetitions):
            self.recursive_split(root)
        return self

    def regions(self) -> Iterable[Region]:
        """
        :return: An iterator over all regions.
        """
        for node in self.nodes:
            if isinstance(node, Region):
                yield node

    def partition_nodes(self) -> Iterable[Partition]:
        """
        :return: An iterator over all partitions.
        """
        for node in self.nodes:
            if isinstance(node, Partition):
                yield node

    def recursive_split(self, node: Region):
        """
        Recursively split a region into partitions.

        :param node: The node to start inducing from.
        """
        root_partition = Partition()
        self.add_edge(node, root_partition)
        remaining_regions = deque([(node, self.depth, root_partition)])

        while remaining_regions:
            region, depth, partition = remaining_regions.popleft()


            if len(region.variables) == 1:
                continue

            if depth == 0:
                for variable in region.variables:
                    self.add_edge(partition, Region(SortedSet([variable])))
                continue

            new_regions = region.random_partition(self.partitions)
            for new_region in new_regions:
                self.add_edge(partition, new_region)
                new_partition = Partition()
                self.add_edge(new_region, new_partition)
                remaining_regions.append((new_region, depth - 1, new_partition))

    @property
    def root(self) -> Region:
        """
        :return: The root of the region graph.
        """
        possible_roots = [node for node in self.nodes() if self.in_degree(node) == 0]
        if len(possible_roots) > 1:
            raise ValueError(f"More than one root found. Possible roots are {possible_roots}")
        return possible_roots[0]


    def as_probabilistic_circuit(self, input_units: int = 5, sum_units: int = 5,
                                 key=jax.random.PRNGKey(69)) -> Union[JPC, ClassificationCircuit]:
        """
        Convert the region graph to a jax probabilistic circuit.
        :param input_units: The number of input units to use in each input layer.
        :param sum_units: The number of sum units to use in each sum layer.
        :param key: The random key to use for all trainable parameters.
        :return: The layered circuit in jax.
        """
        root = self.root

        # create nodes for each region
        for layer in reversed(list(nx.bfs_layers(self, root))):
            for node in layer:
                children = list(self.successors(node))
                parents = list(self.predecessors(node))
                if isinstance(node, Region):
                    # if the region is a leaf
                    if len(children) == 0:
                        variable = node.variables[0]
                        variable_index = self.variables.index(variable)
                        if isinstance(variable, Continuous):
                            location = jax.random.uniform(key, shape=(input_units,), minval=-1., maxval=1.)
                            log_scale = jnp.log(jax.random.uniform(key, shape=(input_units,), minval=0.5, maxval=3.))
                            node.layer = GaussianLayer(variable_index, location=location, log_scale=log_scale,
                                                       min_scale=jnp.full_like(location, 0.1))
                            node.layer.validate()
                        elif isinstance(variable, Symbolic):
                            log_probabilities = jax.random.uniform(key,
                                                                   shape=(input_units, len(variable.domain.simple_sets)),
                                                                   minval=0.1, maxval=1.)
                            log_probabilities = jnp.log(log_probabilities)
                            node.layer = DiscreteLayer(variable_index, log_probabilities=log_probabilities)
                        else:
                            raise ValueError(f"Variable {variable} is not supported.")

                    # if the region is root or in the middle
                    else:
                        # if the region is root
                        if len(parents) == 0:
                            sum_units = self.classes

                        log_weights = [jnp.log(jax.random.uniform(key, shape=(sum_units, child.layer.number_of_nodes),
                                                          minval=0.1, maxval=1.)) for child in children]
                        node.layer = DenseSumLayer([child.layer for child in children], log_weights=log_weights)
                        node.layer.validate()


                elif isinstance(node, Partition):
                    node_lengths = [child.layer.number_of_nodes for child in children]
                    assert (len(set(node_lengths)) == 1), "Node lengths must be all equal. Got {}".format(node_lengths)

                    edges = jnp.arange(node_lengths[0]).reshape(1, -1).repeat(len(children), axis=0)
                    sparse_edges = BCOO.fromdense(jnp.ones_like(edges))
                    sparse_edges.data = edges.flatten()
                    node.layer = ProductLayer([child.layer for child in children], sparse_edges)
                    node.layer.validate()

        if self.classes > 1:
            model = ClassificationCircuit(self.variables, root.layer)
        else:
            model = JPC(self.variables, root.layer)

        return model
