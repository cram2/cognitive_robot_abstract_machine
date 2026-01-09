from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import rustworkx as rx
from matplotlib import pyplot as plt
from random_events.variable import Symbolic, Variable
from typing_extensions import Optional, Any, Self, List, Tuple, Set, Iterable, Dict

from ..distributions import SymbolicDistribution
from ..distributions.helper import make_dirac
from ..probabilistic_circuit.rx.probabilistic_circuit import ProbabilisticCircuit, ProductUnit, SumUnit, leaf


@dataclass
class Node:
    """
    A node in the bayesian network
    These distributions do not inherit from probabilistic models,
    since inference in Bayesian Networks is intractable.
    For inference, convert the bayesian network to a probabilistic circuit.
    """

    bayesian_network: Optional[BayesianNetwork] = field(kw_only=True, repr=False, default=None)
    """
    The bayesian network this node is part of. 
    """

    index: Optional[int] = field(kw_only=True, default=None, repr=False)
    """
    The index of the node in the graph of its circuit.
    """

    product_units: Dict[Any, ProductUnit] = field(init=False, default_factory=dict, repr=False)
    """
    A dictionary from states of the variable to product units. Only needed during conversion to probabilistic circuits.
    """

    def __post_init__(self):
        if self.bayesian_network is not None:
            self.bayesian_network.add_node(self)

    def __hash__(self):
        if self.bayesian_network is not None and self.index is not None:
            return hash((self.index, id(self.bayesian_network)))
        else:
            return id(self)

    @property
    def parent(self) -> Node:
        return self.bayesian_network.predecessors(self)[0]

    @property
    def variables(self) -> Tuple[Variable, ...]:
        raise NotImplementedError

    def as_probabilistic_circuit(self, result: ProbabilisticCircuit):
        """
        Add this node to the probabilistic circuit.
        This also creates all the edges implied by this node.

        :param result: The probabilistic circuit to add the nodes to.
        """
        raise NotImplementedError


@dataclass
class BayesianNetwork:
    """
    Class for Bayesian Networks that are rooted, tree shaped and have univariate inner nodes.
    This class does not inherit from ProbabilisticModel since it cannot perform inference.
    Bayesian Networks can be converted to a probabilistic circuit which can perform inference.
    """

    graph: rx.PyDAG[Node] = field(default_factory=lambda: rx.PyDAG(multigraph=False))
    """
    The graph to check connectivity from.
    """

    def __len__(self):
        """
        Return the number of nodes in the graph.

        :return: The number of nodes in the graph.
        """
        return len(self.graph)

    def __iter__(self):
        """
        Return an iterator over the nodes in the graph.

        :return: An iterator over the nodes in the graph.
        """
        return iter(self.graph.nodes())

    @property
    def leaves(self) -> List[Node]:
        return [node for node in self.nodes() if len(self.successors(node)) == 0]

    def is_valid(self) -> bool:
        """
        Check if this graph is:

        - acyclic
        - connected

        :return: True if the graph is valid, False otherwise.
        """
        return rx.is_connected(self.graph) and (len(self.edges()) == (len(self.nodes()) - 1)) and self.root

    def add_node(self, node: Node):

        if node.bayesian_network is self and node.index is not None:
            return
        elif node.bayesian_network is not None and node.bayesian_network is not self:
            raise NotImplementedError("Cannot add a node that already belongs to another bayesian network.")

        node.index = self.graph.add_node(node)

        # write self as the nodes bn
        node.bayesian_network = self

    def add_nodes_from(self, nodes: Iterable[Node]):
        [self.add_node(node) for node in nodes]

    def add_edge(self, parent: Node, child: Node):
        self.add_node(parent)
        self.add_node(child)
        self.graph.add_edge(parent.index, child.index, None)

    def add_edges_from(self, edges: Iterable[Tuple[Node, Node]]):
        [self.add_edge(*edge) for edge in edges]

    def successors(self, node: Node) -> List[Node]:
        return self.graph.successors(node.index)

    def descendants(self, unit: Node) -> Set[Node]:
        return {self.graph[unit] for unit in rx.descendants(self.graph, unit.index)}

    def predecessors(self, unit: Node) -> List[Node]:
        return self.graph.predecessors(unit.index)

    def in_edges(self, node: Node) -> List[Tuple[Node, Node, Optional[float]]]:
        return [(self.graph.get_node_data(parent_index), node, edge_data,)
                for parent_index, _, edge_data in self.graph.in_edges(node.index)]

    def nodes(self) -> List[Node]:
        """
        Return an iterator over the nodes.

        :return: An iterator over the nodes.
        """
        return self.graph.nodes()

    def edges(self) -> List[Tuple[Node, Node]]:
        return [(self.graph[parent], self.graph[child]) for parent, child in self.graph.edge_list()]

    def in_degree(self, node: Node):
        return self.graph.in_degree(node.index)

    def has_edge(self, parent: Node, child: Node) -> bool:
        return self.graph.has_edge(parent.index, child.index)

    @property
    def root(self) -> Root:
        """
        The root of the circuit is the node with in-degree 0.
        This is the output node, that will perform the final computation.

        :return: The root of the circuit.
        """
        possible_roots = [node for node in self.nodes() if self.in_degree(node) == 0]
        if len(possible_roots) == 1:
            if not isinstance(possible_roots[0], Root):
                raise ValueError("The root is not an instance of Root.")
            return possible_roots[0]
        elif len(possible_roots) > 1:
            raise ValueError(f"More than one root found. Possible roots are {possible_roots}")
        else:
            raise ValueError(f"No root found.")

    def __eq__(self, other: Self):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__} with {len(self.nodes())} nodes and {len(self.edges())} edges"

    def as_probabilistic_circuit(self) -> ProbabilisticCircuit:
        """
        Convert the bayesian network to a probabilistic circuit.

        :return: The probabilistic circuit.
        """
        result = ProbabilisticCircuit()

        for node in rx.topological_sort(self.graph):
            node = self.graph[node]
            node.as_probabilistic_circuit(result)

        result.remove_unreachable_nodes(self.root.root)
        result.simplify()

        return result

    def plot(self):
        import rustworkx.visualization
        rustworkx.visualization.mpl_draw(self.graph, with_labels=True,
                                         labels=lambda node: ", ".join(v.name for v in node.variables))
        plt.show()


@dataclass
class Root(Node):
    distribution: SymbolicDistribution

    root: Optional[SumUnit] = field(init=False, repr=False, default=None)
    """
    The root of the circuit that is generated by the as_probabilistic_circuit method.
    """

    __hash__ = Node.__hash__

    @property
    def variable(self) -> Symbolic:
        return self.distribution.variable

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return self.distribution.variables

    def as_probabilistic_circuit(self, result: ProbabilisticCircuit):
        self.root = SumUnit(probabilistic_circuit=result)
        for value, probability in self.distribution.probabilities.items():
            prod = ProductUnit(probabilistic_circuit=result)
            distribution = leaf(make_dirac(self.variable, value, ), result)
            self.root.add_subcircuit(prod, np.log(probability), )
            prod.add_subcircuit(distribution)
            self.product_units[value] = prod


@dataclass
class ConditionalProbabilityTable(Node):
    """
    Conditional probability distribution for Bayesian Network nodes given their parents.
    The parent in this case must be exactly one node.
    """

    conditional_probability_distributions: Dict[Any, SymbolicDistribution] = field(default_factory=dict)
    __hash__ = Node.__hash__

    @property
    def variable(self) -> Symbolic:
        return list(self.conditional_probability_distributions.values())[0].variable

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return (self.variable,)

    def __repr__(self):
        return f"P({self.variable.name}|{self.parent.variable.name})"

    def to_tabulate(self) -> List[List[str]]:
        """
        Tabulate the truncated probability table.

        :return: A table with the truncated probability table that can be printed using tabulate.
        """
        table = [[self.parent.variable.name, self.variable.name, repr(self)]]

        parent_domain_hash_map = self.parent.variable.domain.hash_map
        own_domain_hash_map = self.variable.domain.hash_map

        for parent_hash, distribution in self.conditional_probability_distributions.items():
            for own_hash, probability in distribution.probabilities.items():
                table.append([str(parent_domain_hash_map[parent_hash]), str(own_domain_hash_map[own_hash]),
                              str(probability)])
        return table

    def as_probabilistic_circuit(self, result: ProbabilisticCircuit):
        for value in self.variable.domain:
            prod = ProductUnit(probabilistic_circuit=result)
            distribution = leaf(make_dirac(self.variable, value, ), result)
            prod.add_subcircuit(distribution)
            self.product_units[value.element] = prod

        parent = self.parent

        for key, conditional_distribution in self.conditional_probability_distributions.items():
            sum_unit = SumUnit(probabilistic_circuit=result)
            parent.product_units[key].add_subcircuit(sum_unit)

            for value, probability in conditional_distribution.probabilities.items():
                sum_unit.add_subcircuit(self.product_units[value], np.log(probability))


@dataclass
class ConditionalProbabilisticCircuit(Node):
    """
    Conditional probability distribution represented as Circuit for Bayesian Network nodes given their parents.
    """

    conditional_probability_distributions: Dict[int, ProbabilisticCircuit] = field(default_factory=dict)
    __hash__ = Node.__hash__

    @property
    def parent(self) -> ConditionalProbabilityTable:
        return super().parent

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return tuple(list(self.conditional_probability_distributions.values())[0].variables)

    def __repr__(self):
        return f"P({', '.join([v.name for v in self.variables])} | {self.parent.variable.name})"

    def as_probabilistic_circuit(self, result: ProbabilisticCircuit):
        parent = self.parent

        for key, conditional_distribution in self.conditional_probability_distributions.items():
            old_root = conditional_distribution.root
            node_remap = result.mount(old_root)
            root_in_result = node_remap[old_root.index]
            parent.product_units[key].add_subcircuit(root_in_result)
