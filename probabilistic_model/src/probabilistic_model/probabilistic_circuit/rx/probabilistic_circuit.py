from __future__ import annotations

import copy
import itertools
import math
import queue
import random
from abc import abstractmethod, ABC
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import IntEnum

import rustworkx as rx
import numpy as np
import rustworkx.visualization
from random_events.interval import SimpleInterval, Interval
from scipy.special import logsumexp
import tqdm
from random_events.product_algebra import VariableMap, SimpleEvent, Event
from random_events.set import Set
from random_events.utils import SubclassJSONSerializer
from random_events.variable import Variable, Symbolic, Continuous, Integer
from sortedcontainers import SortedSet
from typing_extensions import List, Optional, Any, Self, Dict, Tuple, Iterable, Callable, Union

from ...distributions import UnivariateDistribution, IntegerDistribution, SymbolicDistribution, DiscreteDistribution, \
    ContinuousDistribution
from ...distributions.helper import make_dirac
from ...error import IntractableError
from ...interfaces.drawio.drawio import DrawIOInterface, circled_product, circled_sum
from ...probabilistic_model import ProbabilisticModel, OrderType, CenterType, MomentType
from ...utils import MissingDict

class PlotAlignment(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1

@dataclass
class Unit(SubclassJSONSerializer, DrawIOInterface, ABC):
    """
    Class for all units of a probabilistic circuit.

    This class should not be used by users directly.
    Use :class:`ProbabilisticCircuit` as interface to users.
    """

    probabilistic_circuit: Optional[ProbabilisticCircuit] = field(kw_only=True, repr=False, default=None)
    """
    The circuit this component is part of. 
    """

    result_of_current_query: Any = field(init=False, default=None, repr=False)
    """
    The result of the current query. 
    """

    index: Optional[int] = field(kw_only=True, default=None, repr=False)
    """
    The index of the node in the graph of its circuit.
    """

    def __post_init__(self):
        if self.probabilistic_circuit is not None:
            self.probabilistic_circuit.add_node(self)

    @property
    def subcircuits(self) -> List[Unit]:
        """
        :return: The subcircuits of this unit.
        """
        return self.probabilistic_circuit.successors(self)

    @property
    def parents(self) -> List[InnerUnit]:
        """
        :return: The parents of this unit.
        """
        return self.probabilistic_circuit.predecessors(self)

    @abstractmethod
    def support(self):
        """
        Calculate the support of this unit.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_leaf(self):
        """
        :return: If this unit is a leaf unit.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def variables(self) -> SortedSet:
        raise NotImplementedError

    @property
    def leaves(self) -> List[LeafUnit]:
        """
        :return: The leaves of the circuit that are descendants of this node.
        """
        return [unit for unit in self.probabilistic_circuit.descendants(self) if unit.is_leaf]

    def update_variables(self, new_variables: VariableMap):
        """
        Update the variables of this unit and its descendants.

        :param new_variables: A map that maps the variables that should be replaced to their new variable.
        """
        for leaf in self.leaves:
            for variable in leaf.variables:
                if variable in new_variables:
                    leaf.distribution.variable = new_variables[leaf.variable]

    def connect_incoming_edges_to(self, other: Unit):
        """
        Connect all incoming edges to this unit to another unit.

        :param other: The other unit to connect the incoming edges to.
        """
        [self.probabilistic_circuit.add_edge(parent, other, data)
         for parent, _ ,data in self.probabilistic_circuit.in_edges(self)]

    def filter_variable_map_by_self(self, variable_map: VariableMap):
        """
        Filter a variable map by the variables of this unit.

        :param variable_map: The map to filter
        :return: The map filtered by the variables of this unit.
        """
        variables = self.variables
        return variable_map.__class__(
            {variable: value for variable, value in variable_map.items() if variable in variables})

    @property
    def impossible_condition_result(self) -> Tuple[Optional[Unit], float]:
        """
        :return: The result of an impossible truncated query.
        """
        return None, -np.inf

    def log_mode(self):
        raise NotImplementedError

    def __hash__(self):
        if self.probabilistic_circuit is not None and self.index is not None:
            return hash((self.index, id(self.probabilistic_circuit)))
        else:
            return id(self)

    def copy_without_graph(self):
        result = self.empty_copy()
        result.result_of_current_query = self.result_of_current_query
        return result

    def empty_copy(self) -> Self:
        """
        Creat a copy of this circuit without any subcircuits. Only the parameters should be copied.
        This is used whenever a new circuit has to be created during inference.

        :return: A copy of this circuit without any subcircuits that is not in this units graph.
        """
        return self.__class__()

    def simplify(self):
        """
        Simplify the circuit by removing nodes and redirected edges that have no impact in-place.
        Essentially, this method transforms the circuit into an alternating order of sum and product units.

        :return: The simplified circuit.
        """
        raise NotImplementedError()

    def sample(self, *args, **kwargs):
        """
        Draw samples from the circuit.

        For sampling, a node gets requested a number of samples from all his parents.
        The parents write into the `result_of_current_query` attribute a tuple describing the beginning index of the
        sampling and how many samples are requested.
        """
        raise NotImplementedError

    def marginal(self, *args, **kwargs) -> Optional[Self]:
        """
        Remove nodes that are not part of the marginal distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def moment(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def drawio_style(self) -> Dict[str, Any]:
        return {"style": self.drawio_label, "width": 30, "height": 30, }


@dataclass
class LeafUnit(Unit):
    """
    Class for Leaf units.
    """

    distribution: Optional[ProbabilisticModel]
    """
    The distribution contained in this leaf unit.
    """

    def __repr__(self):
        return f"leaf({repr(self.distribution)}"

    @property
    def drawio_label(self):
        return "ellipse;whiteSpace=wrap;html=1;aspect=fixed;"

    @property
    def drawio_style(self) -> Dict[str, Any]:
        return {"style": self.drawio_label, "width": 30, "height": 30, "label": self.distribution.abbreviated_symbol}

    @property
    def variables(self) -> Iterable[Variable]:
        return SortedSet(self.distribution.variables)

    @property
    def subcircuits(self) -> List[Unit]:
        return []

    @property
    def is_leaf(self):
        return True

    @property
    def leaves(self) -> List[LeafUnit]:
        return []

    def log_likelihood(self, events: np.array):
        self.result_of_current_query = self.distribution.log_likelihood(events)

    def cdf(self, events: np.array):
        self.result_of_current_query = self.distribution.cdf(events)

    def probability_of_simple_event(self, event: SimpleEvent):
        self.result_of_current_query = self.distribution.probability_of_simple_event(event)

    def support(self):
        self.result_of_current_query = self.distribution.support # .__deepcopy__()

    def simplify(self):
        if self.distribution is None:
            self.probabilistic_circuit.remove_node(self)

    def log_truncated_of_simple_event_in_place(self, event: SimpleEvent):
        self.distribution, self.result_of_current_query = self.distribution.log_truncated(event.as_composite_set())

    def moment(self, order, center, variable_to_index_map):
        result = np.zeros(len(variable_to_index_map))
        moment = self.distribution.moment(order, center)
        for variable in self.variables:
            result[variable_to_index_map[variable]] = moment[variable]
        self.result_of_current_query = result

    def sample(self, samples: np.array, variable_to_index_map: Dict[Variable, int]):
        """
        Sample from the distribution and write the samples into the samples array.
        :param samples: The array to write the samples into.
        :param variable_to_index_map: The map from variables to column indices in the samples array.
        """
        column_indices = [variable_to_index_map[variable] for variable in self.variables]
        for start_index, amount in self.result_of_current_query:
            samples[start_index:start_index + amount, column_indices] = self.distribution.sample(amount)

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        marginal = self.distribution.marginal(variables)
        if marginal is None:
            self.probabilistic_circuit.remove_node(self)
            return None
        else:
            self.distribution = marginal
            return self

    def log_mode(self):
        self.result_of_current_query = self.distribution.log_mode()

    def to_json(self):
        result = super().to_json()
        result["distribution"] = self.distribution.to_json()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        distribution = SubclassJSONSerializer.from_json(data["distribution"])
        return cls(distribution)

    def log_conditional_in_place(self, point: Dict[Variable, Any]):
        if any(variable for variable in self.variables if variable in point):
            self.distribution, self.result_of_current_query = self.distribution.log_conditional(point)
        else:
            self.result_of_current_query = 0.

    def copy_without_graph(self):
        return self.__class__(distribution = self.distribution.__deepcopy__())


class InnerUnit(Unit):
    """
    Class for inner units
    """

    @property
    def is_leaf(self):
        return False

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def log_forward(self, *args, **kwargs):
        raise NotImplementedError

    def marginal(self, *args, **kwargs) -> Optional[Self]:
        if len(self.subcircuits) == 0:
            self.probabilistic_circuit.remove_node(self)
            return None
        return self

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls()

    def add_subcircuit(self, subcircuit: Unit, log_weight: float = None):
        """
        Add a subcircuit to the subcircuits of this unit.

        .. note::

            This method does not normalize the edges to the subcircuits.


        :param subcircuit: The subcircuit to add.
        :param log_weight: The logarithmic weight of the subcircuit.
        Only needed if this is a sum unit
        """
        self.probabilistic_circuit.add_edge(self, subcircuit, log_weight)

class SumUnit(InnerUnit):
    _latent_variable: Optional[Symbolic] = None
    """
    The latent variable of this unit.
    This has to be here due to the rvalue/lvalue problem in random events.

    TODO remove this when RE is fixed
    """

    def __repr__(self):
        return "⊕"

    __hash__ = Unit.__hash__

    @property
    def representation(self) -> str:
        return "+"

    @property
    def drawio_label(self) -> str:
        return circled_sum

    @property
    def log_weighted_subcircuits(self) -> List[Tuple[float, Unit]]:
        """
        :return: The weighted subcircuits of this unit.
        """
        return [(self.probabilistic_circuit.graph.get_edge_data(self.index, subcircuit.index), subcircuit)
                for subcircuit in self.subcircuits]

    @property
    def variables(self) -> SortedSet:
        return self.subcircuits[0].variables

    @property
    def latent_variable(self) -> Symbolic:
        name = f"{hash(self)}.latent"
        subcircuit_enum = IntEnum(name, {str(hash(subcircuit)): index
                                         for index, subcircuit in enumerate(self.subcircuits)})
        result = Symbolic(name, Set.from_iterable(subcircuit_enum))
        self._latent_variable = result
        return result

    def forward(self, *args, **kwargs):
        self.result_of_current_query = np.sum(
            [np.exp(weight) * subcircuit.result_of_current_query for weight, subcircuit in self.log_weighted_subcircuits], axis=0)

    def log_forward(self, *args, **kwargs):
        result = [lw + s.result_of_current_query for lw, s in self.log_weighted_subcircuits]
        self.result_of_current_query = logsumexp(result, axis=0)
    moment = forward

    def log_forward_conditioning(self, *args, **kwargs):
        result = [lw + s.result_of_current_query for lw, s in self.log_weighted_subcircuits]

        # update weights according to bayes rule
        for new_weight, subcircuit in zip(result, self.subcircuits):
            self.probabilistic_circuit.add_edge(self, subcircuit, log_weight=new_weight)

        self.result_of_current_query = logsumexp(result, axis=0)

    def support(self):
        support = self.subcircuits[0].result_of_current_query
        for subcircuit in self.subcircuits[1:]:
            support |= subcircuit.result_of_current_query.__deepcopy__()
        self.result_of_current_query = support

    @property
    def log_weights(self) -> np.array:
        """
        :return: The log_weights of the subcircuits.
        """
        return np.array([weight for weight, _ in self.log_weighted_subcircuits])

    def sample(self, *args, **kwargs) -> np.array:
        weights, subcircuits = self.log_weights, self.subcircuits

        subcircuit_indices = list(range(len(subcircuits)))
        # for every sampling request
        for start_index, amount in self.result_of_current_query:

            # calculate the numbers of samples requested from the sub circuits
            counts = np.random.multinomial(amount, pvals=np.exp(weights))
            total = 0

            # shuffle the order to sample from the subcircuits to avoid bias
            random.shuffle(subcircuit_indices)

            # add the sampling requests to the subcircuits
            for index in subcircuit_indices:
                subcircuit = subcircuits[index]
                count = counts[index]
                subcircuit.result_of_current_query.append((start_index + total, count))
                total += count

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        result = cls()
        return result

    def mount_with_interaction_terms(self, other: Self, interaction_model: ProbabilisticModel):
        """
        Create a distribution that factorizes as follows:

        .. math::
            p(self.latent\_variable) \cdot p(self.variables | self.latent\_variable) \cdot
            p(other.latent\_variable | self.latent\_variable) \cdot p(other.variables | other.latent\_variable)

        where `self.latent_variable` and `other.latent_variable` are the results of the latent variable interpretation
        of mixture models.

        :param other: The other distribution to mount at this distribution children level.
        :param interaction_model: The interaction probabilities between both latent variables
        """
        assert set(self.variables).intersection(set(other.variables)) == set()
        assert set(interaction_model.variables) == {self.latent_variable, other.latent_variable}

        # load latent variables
        own_latent_variable = self.latent_variable
        other_latent_variable = other.latent_variable

        # load subircuits
        own_subcircuits = self.subcircuits
        other_subcircuits = other.subcircuits

        for own_index, own_subcircuit in enumerate(own_subcircuits):

            # create denominator of weight
            own_index = own_latent_variable.domain.simple_sets[0].element.__class__(own_index)
            condition = SimpleEvent({own_latent_variable: own_index}).as_composite_set()
            p_condition = interaction_model.probability(condition)

            # skip iterations that are impossible
            if p_condition == 0:
                continue

            # create proxy nodes for mounting
            proxy_product_node = ProductUnit()
            proxy_sum_node = other.empty_copy()
            self.probabilistic_circuit.add_nodes_from([proxy_product_node, proxy_sum_node])

            # remove edge to old child and replace it by product proxy
            self.probabilistic_circuit.remove_edge(self, own_subcircuit)
            self.add_subcircuit(proxy_product_node, np.log(p_condition))

            # mount current child on the product proxy
            proxy_product_node.add_subcircuit(own_subcircuit)

            # mount the proxy for the children from other in the product proxy
            proxy_product_node.add_subcircuit(proxy_sum_node)

            for other_index, other_subcircuit in enumerate(other_subcircuits):

                # create numerator of weight
                other_index = other_latent_variable.domain.simple_sets[0].element.__class__(other_index)
                query = SimpleEvent({other_latent_variable: other_index}).as_composite_set() & condition
                p_query = interaction_model.probability(query)

                # skip iterations that are impossible
                if p_query == 0:
                    continue

                # calculate truncated probability
                weight = p_query / p_condition

                # create edge from proxy to subcircuit
                proxy_sum_node.add_subcircuit(other_subcircuit, log_weight=np.log(weight))
            proxy_sum_node.normalize()

    def mount_from_bayesian_network(self, other: 'SumUnit'):
        """
        Mount a distribution from tge `to_probabilistic_circuit` method in bayesian networks.
        The distribution is mounted as follows:


        :param other: The other distribution to mount at this distribution children level.
        :return:
        """
        assert set(self.variables).intersection(set(other.variables)) == set()
        assert len(self.subcircuits) == len(other.subcircuits)
        # mount the other subcircuit

        for (own_weight, own_subcircuit), other_subcircuit in zip(self.log_weighted_subcircuits, other.subcircuits):
            # create proxy nodes for mounting
            proxy_product_node = ProductUnit()
            self.probabilistic_circuit.add_node(proxy_product_node)

            # remove edge to old child and replace it by product proxy
            self.probabilistic_circuit.remove_edge(self, own_subcircuit)
            self.add_subcircuit(proxy_product_node, np.log(own_weight))
            proxy_product_node.add_subcircuit(own_subcircuit)
            proxy_product_node.add_subcircuit(other_subcircuit)

    def simplify(self):

        # if this has only one child
        if len(self.subcircuits) == 1:

            # redirect every incoming edge to the child
            incoming_edges = list(self.probabilistic_circuit.in_edges(self))
            for parent, _, data in incoming_edges:
                self.probabilistic_circuit.add_edge(parent, self.subcircuits[0], data)

            # remove this node
            self.probabilistic_circuit.remove_node(self)

            return

        # for every subcircuit
        for weight, subcircuit in self.log_weighted_subcircuits:

            # if the weight is 0, skip this subcircuit
            if weight == -np.inf:
                # remove the edge
                self.probabilistic_circuit.remove_edge(self, subcircuit)

            # if the simplified subcircuit is of the same type as this
            if type(subcircuit) is type(self):

                # type hinting
                subcircuit: Self

                # mount the children of that circuit directly
                for sub_weight, sub_subcircuit in subcircuit.log_weighted_subcircuits:
                    new_weight = sub_weight + weight

                    # add an edge to that subcircuit
                    self.add_subcircuit(sub_subcircuit, new_weight, mount=False)

                # remove the old node
                self.probabilistic_circuit.remove_node(subcircuit)

    def normalize(self):
        """
        Normalize the log_weights of the subcircuits such that they sum up to 1 inplace.
        """
        total_weight = logsumexp(self.log_weights)
        for log_weight, subcircuit in self.log_weighted_subcircuits:
            self.probabilistic_circuit.graph.add_edge(self.index, subcircuit.index, log_weight - total_weight)

    def is_deterministic(self) -> bool:
        """
        :return: If this unit is deterministic or not.
        """
        # for every unique combination of subcircuits
        for subcircuit_a, subcircuit_b in itertools.combinations(self.subcircuits, 2):
            # check if they intersect
            if not subcircuit_a.result_of_current_query.intersection_with(
                    subcircuit_b.result_of_current_query).is_empty():
                print(subcircuit_a.result_of_current_query, subcircuit_b.result_of_current_query)
                return False

        # if none intersect, the subcircuit is deterministic
        return True

    def log_mode(self):
        log_maxima = [log_weight + subcircuit.result_of_current_query[1] for log_weight, subcircuit in
                      self.log_weighted_subcircuits]
        log_max = max(log_maxima)
        arg_log_maxima = [subcircuit.result_of_current_query[0] for lm, subcircuit in zip(log_maxima, self.subcircuits)
                          if lm == log_max]
        arg_log_max = arg_log_maxima[0]
        for event in arg_log_maxima[1:]:
            arg_log_max |= event
        self.result_of_current_query = (arg_log_max, log_max)

    def subcircuit_index_of_samples(self, samples: np.array) -> np.array:
        """
        :return: the index of the subcircuit where p(sample) > 0 and None if p(sample) = 0 for all subcircuits.
        """
        result = np.full(len(samples), np.nan)
        for index, subcircuit in enumerate(self.subcircuits):
            likelihood = subcircuit.log_likelihood(samples)
            result[likelihood > -np.inf] = index
        return result


class ProductUnit(InnerUnit):
    """
    Decomposable Product Units for Probabilistic Circuits
    """

    representation = "×"
    __hash__ = Unit.__hash__

    @property
    def drawio_label(self) -> str:
        return circled_product

    def __repr__(self):
        return "⊗"

    def forward(self, *args, **kwargs):
        self.result_of_current_query = math.prod(
            [subcircuit.result_of_current_query for subcircuit in self.subcircuits])

    def log_forward(self, *args, **kwargs):
        self.result_of_current_query = np.sum([subcircuit.result_of_current_query for subcircuit in self.subcircuits],
                                              axis=0)

    moment = log_forward

    @property
    def variables(self) -> SortedSet:
        result = SortedSet()
        for subcircuit in self.subcircuits:
            result = result.union(subcircuit.variables)
        return result

    def support(self):
        support: Event = self.subcircuits[0].result_of_current_query
        support.fill_missing_variables(self.variables)

        for subcircuit in self.subcircuits[1:]:
            support &= subcircuit.result_of_current_query

        self.result_of_current_query = support


    def is_decomposable(self):
        for index, subcircuit in enumerate(self.subcircuits):
            variables = subcircuit.variables
            for subcircuit_ in self.subcircuits[index + 1:]:
                if len(set(subcircuit_.variables).intersection(set(variables))) > 0:
                    return False
        return True

    def log_mode(self):
        arg_log_max, log_max = self.subcircuits[0].result_of_current_query
        arg_log_max.fill_missing_variables(self.variables)
        for subcircuit in self.subcircuits[1:]:
            arg_log_max = arg_log_max.intersection_with(subcircuit.result_of_current_query[0])
            log_max += subcircuit.result_of_current_query[1]
        self.result_of_current_query = arg_log_max, log_max

    def __copy__(self):
        return self.empty_copy()

    def simplify(self):

        # if this has only one child
        if len(self.subcircuits) == 1:
            self.connect_incoming_edges_to(self.subcircuits[0])
            self.probabilistic_circuit.remove_node(self)
            return

        # for every subcircuit
        for subcircuit in self.subcircuits:

            # if the simplified subcircuit is of the same type as this
            if type(subcircuit) is type(self):

                # type hinting
                subcircuit: Self

                # mount the children of that circuit directly
                for sub_subcircuit in subcircuit.subcircuits:
                    subcircuit.add_subcircuit(sub_subcircuit)

    def sample(self, *args, **kwargs):
        for start_index, amount in self.result_of_current_query:
            for subcircuit in self.subcircuits:
                subcircuit.result_of_current_query.append([start_index, amount])


@dataclass
class ProbabilisticCircuit(ProbabilisticModel, SubclassJSONSerializer):
    """
    Probabilistic Circuits as a directed, rooted, acyclic graph.

    The nodes of the graph are the units of the circuit.
    The edges of the graph indicate how the units are connected.
    The outgoing edges of a sum unit contain the log-log_weights of the subcircuits.
    """

    graph: rx.PyDAG[Unit] = field(default_factory= lambda: rx.PyDAG(multigraph=False))
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

    @classmethod
    def from_other(cls, other: Self) -> Self:
        result = cls()
        result.add_edges_and_nodes_from_circuit(other)
        return result

    @property
    def variables(self) -> SortedSet:
        return self.root.variables

    @property
    def variable_to_index_map(self) -> Dict[Variable, int]:
        return {variable: index for index, variable in enumerate(self.variables)}

    @property
    def layers(self) -> List[List[Unit]]:
        return rx.layers(self.graph, [self.root.index], index_output=False)

    @property
    def leaves(self) -> List[LeafUnit]:
        return self.root.leaves

    def is_valid(self) -> bool:
        """
        Check if this graph is:

        - acyclic
        - connected

        :return: True if the graph is valid, False otherwise.
        """
        return rx.is_directed_acyclic_graph(self.graph) and self.root

    def add_node(self, node: Unit):

        if node.probabilistic_circuit is self and node.index is not None:
            return
        elif node.probabilistic_circuit is not None and node.probabilistic_circuit is not self:
            raise NotImplementedError("Cannot add a node that already belongs to another circuit.")

        node.index = self.graph.add_node(node)

        # write self as the nodes' circuit
        node.probabilistic_circuit = self

    def add_nodes_from(self, units: Iterable[Unit]):
        [self.add_node(node) for node in units]

    def add_edge(self, parent: Unit, child: Unit, log_weight: Optional[float] = None):
        self.add_node(parent)
        self.add_node(child)
        self.graph.add_edge(parent.index, child.index, log_weight)

    def add_edges_from(self, edges: Iterable[Union[Tuple[Unit, Unit], Tuple[Unit, Unit, float]]]):
        [self.add_edge(*edge) for edge in edges]

    def successors(self, unit: Unit) -> List[Unit]:
        return self.graph.successors(unit.index)

    def descendants(self, unit: Unit) -> Set[Unit]:
        return {self.graph[unit] for unit in rx.descendants(self.graph, unit.index)}

    def predecessors(self, unit: Unit) -> List[InnerUnit]:
        return self.graph.predecessors(unit.index)

    def remove_node(self, unit: Unit):
        self.graph.remove_node(unit.index)
        unit.index = None
        unit.probabilistic_circuit = None

    def remove_nodes_from(self, units: Iterable[Unit]):
        [self.remove_node(unit) for unit in units]

    def remove_edge(self, parent: Unit, child: Unit):
        self.graph.remove_edge(parent.index, child.index)

    def remove_edges_from(self, edges: Iterable[Tuple[Unit, Unit]]):
        [self.remove_edge(*edge) for edge in edges]

    def in_edges(self, unit: Unit) -> List[Tuple[Unit, Unit, Optional[float]]]:
        return [(self.graph.get_node_data(parent_index), unit, edge_data,)
                for parent_index, _, edge_data in self.graph.in_edges(unit.index)]

    def add_from_subgraph(self, subgraph: rx.PyDAG[Unit]) -> Dict[int, Unit]:
        """
        Add nodes and edges from a subgraph to this circuit.

        :param subgraph: The subgraph to add nodes from.
        :return: A dictionary mapping the node indices in the subgraph to the new units in this circuit.
        """
        new_nodes = {node.index: node.copy_without_graph() for node in subgraph.nodes()}
        self.add_nodes_from(new_nodes.values())

        [self.graph.add_edge(new_nodes[parent].index, new_nodes[child].index,
                             subgraph.get_edge_data(parent, child)) for parent, child in subgraph.edge_list()]
        return new_nodes


    def nodes(self) -> List[Unit]:
        """
        Return an iterator over the nodes.

        :return: An iterator over the nodes.
        """
        return self.graph.nodes()

    def edges(self) -> List[Tuple[Unit, Unit, Optional[float]]]:
        return [(self.graph[parent], self.graph[child], self.graph.get_edge_data(parent, child))
                for parent, child in self.graph.edge_list()]

    def in_degree(self, unit: Unit):
        return self.graph.in_degree(unit.index)

    def has_edge(self, parent: Unit, child: Unit) -> bool:
        return self.graph.has_edge(parent.index, child.index)

    @property
    def root(self) -> Unit:
        """
        The root of the circuit is the node with in-degree 0.
        This is the output node, that will perform the final computation.

        :return: The root of the circuit.
        """
        possible_roots = [node for node in self.nodes() if self.in_degree(node) == 0]
        if len(possible_roots) == 1:
            return possible_roots[0]
        elif len(possible_roots) > 1:
            raise ValueError(f"More than one root found. Possible roots are {possible_roots}")
        else:
            raise ValueError(f"No root found.")

    def log_likelihood(self, events: np.array) -> np.array:
        variable_to_index_map = self.variable_to_index_map
        for layer in reversed(self.layers):
            for unit in layer:  # open all the procesess
                if unit.is_leaf:
                    unit: LeafUnit
                    unit.log_likelihood(events[:, [variable_to_index_map[variable] for variable in unit.variables]])
                else:
                    unit: InnerUnit
                    unit.log_forward()  # Synch trheads 1
        return self.root.result_of_current_query

    def cdf(self, events: np.array) -> np.array:
        variable_to_index_map = self.variable_to_index_map
        for layer in reversed(self.layers):
            for unit in layer:
                unit: LeafUnit
                if unit.is_leaf:
                    unit.cdf(events[:, [variable_to_index_map[variable] for variable in unit.variables]])
                else:
                    unit: InnerUnit
                    unit.forward()
        return self.root.result_of_current_query

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        for layer in reversed(self.layers):
            for unit in layer:
                if unit.is_leaf:
                    unit: LeafUnit
                    unit.probability_of_simple_event(event)
                else:
                    unit: InnerUnit
                    unit.forward()
        return self.root.result_of_current_query

    def log_mode(self, check_determinism: bool = True) -> Tuple[Event, float]:
        if check_determinism:
            if not self.is_deterministic():
                raise IntractableError("The circuit is not deterministic.")
        [unit.log_mode() for layer in reversed(self.layers) for unit in layer]
        return self.root.result_of_current_query

    def remove_unreachable_nodes(self, root: Unit):
        """
        Remove all nodes that are not reachable from the root.
        """
        reachable_nodes = self.descendants(root)
        unreachable_nodes = set(self.graph.nodes()) - (reachable_nodes | {root})
        self.remove_nodes_from(unreachable_nodes)

    def log_truncated_of_simple_event_in_place(self, simple_event: SimpleEvent) -> Tuple[Optional[Self], float]:
        """
        Construct the truncated circuit from a simple event.

        :param simple_event: The simple event to condition on.
        :return: The truncated circuit and the log-probability of the event
        """
        for layer in reversed(self.layers):
            for unit in layer:
                if unit.is_leaf:
                    unit: LeafUnit
                    unit.log_truncated_of_simple_event_in_place(simple_event)
                else:
                    unit: InnerUnit
                    unit.log_forward()

        root = self.root
        [self.remove_node(node) for node in self.nodes() if
         node.result_of_current_query == -np.inf]

        if root not in self.graph.nodes():
            return None, -np.inf

        # clean the circuit up
        self.remove_unreachable_nodes(root)
        self.simplify()
        self.normalize()

        return self, root.result_of_current_query

    def log_truncated_in_place(self, event: Event) -> Tuple[Optional[Self], float]:
        """
        Efficiently compute the truncated for an Event, batching as much as possible.
        """
        # skip trivial case
        if event.is_empty():
            self.graph.remove_nodes_from(list(self.graph.nodes()))
            return None, -np.inf


        # if the event is easy, don't create a proxy node
        elif len(event.simple_sets) == 1:
            result = self.log_truncated_of_simple_event_in_place(event.simple_sets[0])
            return result

        # Helper so every thread does its own deepcopy and truncation
        def _copy_and_truncate(simple_event):
            return self.__deepcopy__().log_truncated_of_simple_event_in_place(simple_event)

        with ThreadPoolExecutor(max_workers=min(32, len(event.simple_sets))) as executor:
            conditional_circuits = list(executor.map(_copy_and_truncate, event.simple_sets))

        # clear this circuit
        self.remove_nodes_from(list(self.graph.nodes()))

        # filtered out impossible conditionals
        conditional_circuits = [(conditional, log_probability) for conditional, log_probability in conditional_circuits
                                if log_probability > -np.inf]

        # if all conditionals are impossible
        if len(conditional_circuits) == 0:
            return None, -np.inf

        # create a new sum unit
        result = SumUnit(probabilistic_circuit=self)

        # add the conditionals to the sum unit
        for conditional, log_probability in conditional_circuits:
            root = conditional.root
            new_nodes = result.probabilistic_circuit.add_from_subgraph(conditional.graph)
            result.add_subcircuit(new_nodes[root.index], log_probability)

        result.log_forward()
        result.normalize()
        return self, result.result_of_current_query

    def log_truncated(self, event: Event) -> Tuple[Optional[Self], float]:
        result = copy.deepcopy(self)
        return result.log_truncated_in_place(event)

    def marginal_in_place(self, variables: Iterable[Variable]) -> Optional[Self]:
        result = [node.marginal(variables) for layer in reversed(self.layers) for node in layer][-1]
        if result is not None:
            self.remove_unreachable_nodes(result)
            self.simplify()
            return self
        else:
            return None

    def log_conditional_in_place(self, point: Dict[Variable, Any]) -> Tuple[Optional[Self], float]:

        # do forward pass
        for layer in reversed(self.layers):
            for unit in layer:
                if unit.is_leaf:
                    unit: LeafUnit
                    unit.log_conditional_in_place(point)
                elif isinstance(unit, SumUnit):
                    unit.log_forward_conditioning()
                elif isinstance(unit, ProductUnit):
                    unit.log_forward()
                else:
                    raise NotImplementedError()

        # clean the circuit up
        root = self.root
        [self.graph.remove_node(node) for layer in reversed(self.layers) for node in layer if
         node.result_of_current_query == -np.inf]

        if root not in self.graph.nodes():
            return None, -np.inf

        self.remove_unreachable_nodes(root)

        # simplify dirac parts
        remaining_variables = [v for v in self.variables if v not in point]


        self.marginal_in_place(remaining_variables)

        if len(remaining_variables) > 0:
            root = self.root

        # add dirac parts
        new_root = ProductUnit(probabilistic_circuit=self)

        if len(remaining_variables) > 0:
            new_root.add_subcircuit(root, False)

        for variable, value in point.items():
            new_root.add_subcircuit(leaf(make_dirac(variable, value), self))

        new_root.result_of_current_query = root.result_of_current_query

        self.simplify()
        self.normalize()

        return self, root.result_of_current_query

    def log_conditional(self, point: Dict[Variable, Any]) -> Tuple[Optional[Self], float]:
        result = self.__deepcopy__()
        return result.log_conditional_in_place(point)

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        result = self.__deepcopy__()
        return result.marginal_in_place(variables)

    def sample(self, amount: int) -> np.array:

        # initialize all results
        for node in self.graph.nodes():
            node.result_of_current_query = []

        variable_to_index_map = self.variable_to_index_map

        # initialize the sample arguments
        self.root.result_of_current_query.append((0, amount))

        # initialize the samples
        samples = np.full((amount, len(variable_to_index_map)), np.nan)

        # forward through the circuit to sample
        [node.sample(samples, variable_to_index_map) for layer in self.layers for node in layer]

        return samples

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        variable_to_index_map = self.variable_to_index_map
        [node.moment(order, center, variable_to_index_map) for layer in reversed(self.layers) for node in layer]
        return MomentType({variable: moment for variable, moment in
                           zip(variable_to_index_map.keys(), self.root.result_of_current_query)})

    def simplify(self) -> Self:
        """
        Simplify the circuit inplace.
        """
        [node.simplify() for layer in reversed(self.layers) for node in layer]
        return self

    @property
    def support(self) -> Event:
        [node.support() for layer in reversed(self.layers) for node in layer]
        return self.root.result_of_current_query

    def is_decomposable(self) -> bool:
        """
        Check if the whole circuit is decomposed.

        A circuit is decomposed if all its product units are decomposed.

        :return: if the whole circuit is decomposed
        """
        return all([subcircuit.is_decomposable() for subcircuit in self.leaves if isinstance(subcircuit, ProductUnit)])

    def __eq__(self, other: Self):
        raise NotImplementedError

    def empty_copy(self) -> Self:
        """
        Create a copy of this circuit without any nodes.
        Only the parameters should be copied.
        This is used whenever a new circuit has to be created during inference.

        :return: A copy of this circuit without any subcircuits that is not in this units graph.
        """
        return self.__class__()

    def __deepcopy__(self, memo=None) -> Self:
        """
        Deep copy of the circuit.

        :param memo: A dictionary that is used to keep track of objects that have already been copied.
        :return: A deep copy of the circuit.
        """
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]

        # Create a new empty circuit
        result = self.empty_copy()
        memo[id_self] = result

        # remap nodes to new copies
        remapped_indices = {node.index: node.copy_without_graph() for node in self.nodes()}

        # add copied nodes
        result.add_nodes_from(remapped_indices.values())

        # copy edges and edge data
        [result.graph.add_edge(remapped_indices[parent].index, remapped_indices[child].index,
                               self.graph.get_edge_data(parent, child)) for parent, child in self.graph.edge_list()]

        return result

    def to_json(self) -> Dict[str, Any]:
        # get super result
        result = super().to_json()

        index_to_node_map = {node.index: node.to_json() for node in self.nodes()}
        edges = [(parent.index, child.index, data) for parent, child, data in self.edges()]

        result["index_to_node_map"] = index_to_node_map
        result["edges"] = edges

        return result

    @classmethod
    def parameters_from_json(cls, data: Dict[str, Any]) -> Self:
        return cls()

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        result = cls.parameters_from_json(data)
        hash_remap: Dict[int, Unit] = dict()

        for index, node_data in data["index_to_node_map"].items():
            node = Unit.from_json(node_data)
            hash_remap[int(index)] = node
            result.add_node(node)

        [result.graph.add_edge(hash_remap[parent_index].index, hash_remap[child_index].index, data)
         for parent_index, child_index, data in data["edges"]]

        return result

    def update_variables(self, new_variables: VariableMap):
        """
        Update the variables of this unit and its descendants.

        :param new_variables: The new variables to set.
        """
        self.root.update_variables(new_variables)

    def is_deterministic(self) -> bool:
        """
        :return: Whether, this circuit is deterministic or not.
        """

        # calculate the support
        support = self.support

        # check for determinism of every node
        return all(node.is_deterministic() for node in self.graph.nodes() if isinstance(node, SumUnit))

    def normalize(self):
        """
        Normalize every sum node of this circuit in-place.
        """
        [node.normalize() for node in self.graph.nodes() if isinstance(node, SumUnit)]

    def add_edges_and_nodes_from_circuit(self, other: Self):
        """
        Add all edges and nodes from another circuit to this circuit.

        :param other: The other circuit to add.
        """
        self.add_nodes_from(other.graph.nodes())
        self.graph.add_edges_from(other.unweighted_edges)
        self.add_weighted_edges_from(other.log_weighted_edges, weight="log_weight")

    def add_weighted_edges_from(self, ebunch_to_add, weight = "log_weight", **attr):
        return self.graph.add_weighted_edges_from(ebunch_to_add, weight=weight, **attr)

    def subgraph_of(self, node: Unit) -> Self:
        """
        Create a subgraph with a node as root.

        :param node: The root of the subgraph.
        :return: The subgraph.
        """
        nodes_to_keep = list(nx.descendants(self.graph, node)) + [node]
        result = self.__class__()
        result.graph = self.graph.subgraph(nodes_to_keep)
        return result

    def fill_node_colors(self, node_colors: Dict[Unit, str]):
        """
        Fill the node colors for the structure plot.

        :param node_colors: The node colors to fill.
        """
        # fill the colors for the nodes
        if node_colors is None:
            node_colors = dict()
        for node in self.graph.nodes():
            if node not in node_colors:
                node_colors[node] = "black"
        return node_colors

    def bfs_layout(self, scale: float = 1., align: PlotAlignment = PlotAlignment.VERTICAL) -> Dict[int, np.array]:
        """
        Generate a bfs layout for this circuit.

        :return: A dict mapping the node indices to 2d coordinates.
        """
        layers = self.layers

        pos = None
        nodes = []
        width = len(layers)
        for i, layer in enumerate(layers):
            height = len(layer)
            xs = np.repeat(i, height)
            ys = np.arange(0, height, dtype=float)
            offset = ((width - 1) / 2, (height - 1) / 2)
            layer_pos = np.column_stack([xs, ys]) - offset
            if pos is None:
                pos = layer_pos
            else:
                pos = np.concatenate([pos, layer_pos])
            nodes.extend(layer)

        # Find max length over all dimensions
        pos -= pos.mean(axis=0)
        lim = np.abs(pos).max()  # max coordinate for all axes
        # rescale to (-scale, scale) in all directions, preserves aspect
        if lim > 0:
            pos *= scale / lim

        if align == PlotAlignment.HORIZONTAL:
            pos = pos[:, ::-1]  # swap x and y coords

        pos = dict(zip([node.index for node in nodes], pos))
        return pos


    def plot_structure(self, node_colors: Optional[Dict[Unit, str]] = None, variable_name_offset=0.2,
                       plot_inference=False,
                       inference_representation: Callable = lambda node: str(node.result_of_current_query),
                       inference_result_offset: float = -0.25):
        """
        Plot the structure of the circuit using matplotlib.

        :param node_colors: Optionally specified colors of the node.
        If nodes are not specified in the dictionary, they will be black.
        :param node_size: The size of the nodes
        :param variable_name_offset: The offset to the right of the variable names.
        :param plot_inference: If the results of the inference should be plotted.
        :param inference_representation: The representation of the inference results as a function from node to string.
        :param inference_result_offset: The vertical offset of the inference results.
        """

        # fill the colors for the nodes
        node_colors = self.fill_node_colors(node_colors)
        scale = 1.
        layers = self.layers

        # get the positions of the nodes
        positions = self.bfs_layout(scale=scale, align=PlotAlignment.VERTICAL)
        position_for_variable_name = {node: (x + variable_name_offset, y) for node, (x, y) in positions.items()}


        def node_labels(node: Unit) -> str:
            if isinstance(node, SumUnit):
                return "+"
            elif isinstance(node, ProductUnit):
                return "×"
            elif isinstance(node, LeafUnit):
                return str(node.distribution)
            else:
                raise NotImplementedError

        def edge_labels(data) -> str:
            if data is None:
                return ""
            else:
                return str(np.round(data, decimals=2))

        rustworkx.visualization.mpl_draw(self.graph, pos=positions, labels=node_labels, with_labels=True,
                                         edge_labels=edge_labels)


    def nodes_weights(self) -> dict:
        """
        :return: dict with keys as nodes and values as list of all the log_weights for the node.
        """
        node_weights = {hash(self.root): [1]}
        seen_nodes = set()
        seen_nodes.add(hash(self.root))

        to_visit_nodes = queue.Queue()

        to_visit_nodes.put(self.root)
        while not to_visit_nodes.empty():
            node = to_visit_nodes.get()
            succ_iter = self.graph.successors(node)
            for succ in succ_iter:
                if self.graph.has_edge(node, succ):
                    weight = self.graph.get_edge_data(node, succ).get("weight", 1)
                    node_weights[hash(succ)] = [old * weight for old in node_weights[hash(node)]] + node_weights.get(
                        hash(succ), [])
                    if hash(succ) not in seen_nodes:
                        seen_nodes.add(hash(succ))
                        to_visit_nodes.put(succ)
        return node_weights

    def replace_discrete_distribution_with_deterministic_sum(self):
        """
        splits the distribution into sum unit with all the discrete possibilities as leaf.
        """
        old_leafs = self.leaves
        for leaf in old_leafs:

            if isinstance(leaf, UnivariateDiscreteLeaf):
                leaf: UnivariateDiscreteLeaf
                sum_leaf = leaf.as_deterministic_sum()
                old_predecessors = list(self.graph.predecessors(leaf))
                for predecessor in old_predecessors:
                    weight = self.graph.get_edge_data(predecessor, leaf).get("log_weight", -1)
                    if weight == -1:
                        predecessor.add_subcircuit(sum_leaf)
                    else:
                        predecessor.add_subcircuit(sum_leaf, log_weight=weight)
                    self.graph.remove_edge(predecessor, leaf)
                self.graph.remove_node(leaf)

    def translate(self, translation: Dict[Variable, float]):
        for leaf in self.leaves:
            if any(v.is_numeric for v in leaf.variables):
                leaf.distribution.translate(translation)

    def scale(self, scale: Dict[Variable, float]):
        for leaf in self.leaves:
            if any(v.is_numeric for v in leaf.variables):
                leaf.distribution.scale(scale)

    def mount(self, other: Unit) -> Dict[int, Unit]:
        """
        Mount another unit including its descendants. There will be no edge from `self` to `other`.
        This will also remove the nodes in other and their descendants from their circuit.

        :param other: The other unit to mount.
        """
        if other.probabilistic_circuit is not None:
            descendants = other.probabilistic_circuit.descendants(other)
            descendants = descendants.union([other])
            subgraph = other.probabilistic_circuit.graph.subgraph([u.index for u in descendants])
            result = self.add_from_subgraph(subgraph)
            return result
        else:
            raise ValueError("Trying to mount a unit that doesn't belong to any probabilistic circuit.")

    def __repr__(self):
        return f"{self.__class__.__name__} with {len(self.nodes())} nodes and {len(self.edges())} edges"

class ShallowProbabilisticCircuit(ProbabilisticCircuit):
    """
    class for PC in shallow form, sum unit as root followed by product units which only have leafs as children.
    """

    @classmethod
    def from_probabilistic_circuit(cls, probabilistic_circuit: ProbabilisticCircuit):
        """
        Initialization function, to input a PC to create its shallow version.
        """
        result = cls()
        shallow_pc = probabilistic_circuit.__copy__()
        cls.shallowing(result, node=shallow_pc.root, presucc=None)
        result.add_nodes_from(shallow_pc.nodes)
        result.add_edges_from(shallow_pc.edges)
        result.add_weighted_edges_from(shallow_pc.log_weighted_edges)
        return result

    def shallowing(self, node: Unit, presucc: Unit | None):
        """
        This function transforms the PC into it shallow form, in place.
        This function uses recursion and need to be called on the root of the PC.
        :node: the Node in focus to be shallowed
        :presucc: the predecessor of the node of before shallowing.
        """
        probabilistic_circuit = node.probabilistic_circuit
        succ_list: List = list(probabilistic_circuit.successors(node))
        if not isinstance(node, SumUnit) and not isinstance(node, ProductUnit):
            sum_unit = SumUnit()
            product_unit = ProductUnit()
            probabilistic_circuit.add_node(sum_unit)
            probabilistic_circuit.add_node(product_unit)
            probabilistic_circuit.add_edge(product_unit, node)
            probabilistic_circuit.add_edge(sum_unit, product_unit, log_weight=0.)
            if presucc is not None:
                data = probabilistic_circuit.get_edge_data(presucc, node, {"weight": 1})
                probabilistic_circuit.add_edge(presucc, sum_unit, **data)
                probabilistic_circuit.remove_edge(presucc, node)
            return
        elif isinstance(node, SumUnit):
            for succ in succ_list:
                self.shallowing(succ, presucc=node)
            new_succ_list = list(probabilistic_circuit.successors(node))
            for sum_succ in new_succ_list:
                first_weight = probabilistic_circuit.get_edge_data(node, sum_succ, {"log_weight": 0}).get("log_weight", 0)
                for succ_succ in list(probabilistic_circuit.successors(sum_succ)):
                    second_weight = probabilistic_circuit.get_edge_data(sum_succ, succ_succ, {"log_weight": 0}).get(
                        "weight", 1)
                    probabilistic_circuit.add_edge(node, succ_succ, log_weight=first_weight + second_weight)
                probabilistic_circuit.remove_edge(node, sum_succ)
                if len(list(probabilistic_circuit.predecessors(sum_succ))) == 0:
                    self.remove_node_and_successor_structure(sum_succ)
            return
        elif isinstance(node, ProductUnit):
            for succ in succ_list:
                self.shallowing(succ, presucc=node)
            new_succ_list = list(probabilistic_circuit.successors(node))
            combination_li = list()
            sum_unit = SumUnit()
            probabilistic_circuit.add_node(sum_unit)
            if presucc is not None:
                data = probabilistic_circuit.get_edge_data(presucc, node, {"weight": 1})
                probabilistic_circuit.add_edge(presucc, sum_unit, **data)
                probabilistic_circuit.remove_edge(presucc, node)
                if len(list(probabilistic_circuit.predecessors(node))) == 0:
                    probabilistic_circuit.remove_node(node)
            elif presucc is None:
                # None only happen if this Instance is root
                probabilistic_circuit.remove_node(node)
            for sum_succ in new_succ_list:
                pro_li = []
                for pro_succ in list(probabilistic_circuit.successors(sum_succ)):
                    data = probabilistic_circuit.get_edge_data(sum_succ, pro_succ, {"weight": 1})
                    pro_li.append((pro_succ, data.get("weight", 1)))
                combination_li.append(pro_li)
            for combination in itertools.product(*combination_li):
                product_unit = ProductUnit()
                probabilistic_circuit.add_node(product_unit)
                total_weight = 1
                for pro_tuple in combination:
                    under_node, weight = pro_tuple[0], pro_tuple[1]
                    total_weight *= weight
                    under_succ_li = probabilistic_circuit.successors(under_node)
                    for under_succ in under_succ_li:
                        data = probabilistic_circuit.get_edge_data(under_node, under_succ, {"log_weight": 0.})
                        probabilistic_circuit.add_edge(product_unit, under_succ, **data)
                probabilistic_circuit.add_edge(sum_unit, product_unit, log_weight=total_weight)
            for sum_succ in new_succ_list:
                if len(list(probabilistic_circuit.predecessors(sum_succ))) == 0:
                    self.remove_node_and_successor_structure(sum_succ)
            return

        else:
            raise TypeError(f"{type(node)} is not supported")

    def events_of_higher_density_product(self, other: Self, own_pro_unit, other_pro_unit, tolerance: float = 10e-8):
        """
        Construct E_p of a product unit in a shallow context.
        :own_pro_unit: product unit which is part of E_p
        :other: other product unit which is part of E_p
        :tolerance: float as how close to zero is zero, because of imprecision.
        """
        # supp_own = own_pro_unit.support
        # supp_other = other_pro_unit.support
        # intersection: Event = (supp_own & supp_other)
        own_copy = self.subgraph_of(own_pro_unit)
        other_copy = other.subgraph_of(other_pro_unit)
        intersection = own_copy.support & other_copy.support

        if intersection.is_empty():
            return Event()

        center = np.array([assignment.simple_sets[0].center() for variable, assignment in
                           intersection.simple_sets[0].items()]).reshape(1, -1)

        likelihood_own = self.likelihood(center)
        likelihood_other = other.likelihood(center)
        diff = likelihood_own - likelihood_other
        if diff > tolerance:
            return intersection
        else:
            return Event()

    def events_of_higher_density_sum(self, other: Self, tolerance: float = 10e-8):
        """
        Construct E_p of a sum unit in a shallow context.
        :other: the other Root shallow PC node to create the E_p
        :tolerance: float as how close to zero is zero, because of imprecisions
        """
        progress_bar = tqdm.tqdm(total=len(self.root.subcircuits) * len(other.root.subcircuits))
        result = self.support - other.support
        for own_prod, other_prod in itertools.product(self.root.subcircuits, other.root.subcircuits):
            result |= self.events_of_higher_density_product(other, own_prod, other_prod, tolerance)
            progress_bar.update()
        return result

    def l1(self, other: Self, tolerance: float = 10e-8) -> float:
        """
        The L1 metric between shallow Circuits are calculated.
        It is important, that before the shallowing the PC replace_discrete_distribution_with_deterministic_sum called on.´
        :other: the other shallow PC which the L1 metric is calculated
        :tolerance: float as how close to zero is zero, because of imprecision for the Creation of E_p.
        """
        e = self.events_of_higher_density_sum(other, tolerance)
        p_e = self.probability(e)
        q_e = other.probability(e)

        return 2 * (p_e - q_e)

    def remove_node_and_successor_structure(self, node: Unit):
        """
        This is an assist function for pruning disconnected subgraphs from the PC.
        :node: the node that needs to be checked if to be pruned and its children.
        """
        probabilistic_circuit = node.probabilistic_circuit
        succ_list: List = list(probabilistic_circuit.successors(node))
        probabilistic_circuit.remove_node(node)
        for succ in succ_list:
            if len(list(probabilistic_circuit.predecessors(succ))) == 0:
                self.remove_node_and_successor_structure(succ)

@dataclass
class UnivariateLeaf(LeafUnit):

    @property
    def variable(self) -> Variable:
        return self.distribution.variables[0]

@dataclass
class UnivariateContinuousLeaf(UnivariateLeaf):
    distribution: Optional[ContinuousDistribution]

    __hash__ = Unit.__hash__

    def log_truncated_of_simple_event_in_place(self, event: SimpleEvent):
        return self.univariate_log_truncated_of_simple_event_in_place(event[self.variable])

    def univariate_log_truncated_of_simple_event_in_place(self, event: Interval):
        """
        Condition this distribution on a simple event in-place but use sum units to create conditions on composite
        intervals.
        :param event: The simple event to condition on.
        """

        # if it is a simple truncation
        if len(event.simple_sets) == 1:
            self.distribution, self.result_of_current_query = self.distribution.log_conditional_from_simple_interval(
                event.simple_sets[0])
            return self

        total_probability = 0.

        # calculate the truncated distribution as sum unit
        result = SumUnit(probabilistic_circuit=self.probabilistic_circuit)

        for simple_interval in event.simple_sets:
            current_conditional, current_log_probability = self.distribution.log_conditional_from_simple_interval(
                simple_interval)
            current_probability = np.exp(current_log_probability)

            if current_probability == 0:
                continue

            current_conditional = self.__class__(distribution=current_conditional,
                                                 probabilistic_circuit=self.probabilistic_circuit)
            result.add_subcircuit(current_conditional, np.log(current_probability))
            total_probability += current_probability

        # if the event is impossible
        if total_probability == 0:
            self.result_of_current_query = -np.inf
            self.distribution = None
            self.probabilistic_circuit.remove_node(result)
            return None

        # reroute the parent to the new sum unit
        self.connect_incoming_edges_to(result)

        # remove this node
        self.probabilistic_circuit.remove_node(self)

        # update result
        result.normalize()
        result.result_of_current_query = np.log(total_probability)
        return result

@dataclass
class UnivariateDiscreteLeaf(UnivariateLeaf):

    distribution: Optional[DiscreteDistribution]
    __hash__ = Unit.__hash__

    def as_deterministic_sum(self) -> SumUnit:
        """
        Convert this distribution to a deterministic sum unit that encodes the same distribution in-place.
        The result has as many children as the probability dictionary of this distribution.
        Each child encodes the value of the variable.

        :return: The deterministic sum unit that encodes the same distribution.
        """
        result = SumUnit(probabilistic_circuit=self.probabilistic_circuit)

        for element, probability in self.distribution.probabilities.items():
            result.add_subcircuit(leaf(self.distribution.__class__(self.variable, MissingDict(float, {element: 1.})),
                                                         self.probabilistic_circuit), np.log(probability))
        self.connect_incoming_edges_to(result)
        self.probabilistic_circuit.remove_node(self)
        return result

    @classmethod
    def from_mixture(cls, mixture: ProbabilisticCircuit):
        """
        Create a discrete distribution from a univariate mixture.

        :param mixture: The mixture to create the distribution from.
        :return: The discrete distribution.
        """
        assert len(mixture.variables) == 1, "Can only convert univariate sum units to discrete distributions."
        variable = mixture.variables[0]
        probabilities = MissingDict(float)

        for element in mixture.support.simple_sets[0][variable].simple_sets:
            probability = mixture.probability_of_simple_event(
                SimpleEvent({variable: element}))
            if isinstance(element, SimpleInterval):
                element = element.lower
            probabilities[hash(element)] = probability

        distribution_class = IntegerDistribution if isinstance(variable, Integer) else SymbolicDistribution
        distribution = distribution_class(variable, probabilities)
        return cls(distribution)


def leaf(distribution: UnivariateDistribution, probabilistic_circuit: Optional[ProbabilisticCircuit] = None) -> UnivariateLeaf:
    """
    Factory that creates the correct leaf from a distribution.

    :return: The leaf.
    """
    if isinstance(distribution.variable, Continuous):
        return UnivariateContinuousLeaf(distribution, probabilistic_circuit=probabilistic_circuit)
    else:
        return UnivariateDiscreteLeaf(distribution, probabilistic_circuit=probabilistic_circuit)
