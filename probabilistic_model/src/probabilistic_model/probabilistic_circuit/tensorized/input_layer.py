from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import tqdm
from random_events.interval import Bound, Interval, SimpleInterval
from random_events.product_algebra import Event, SimpleEvent, VariableMap
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import Any, Dict, List, Optional, Self, Tuple, Type

from probabilistic_model.distributions.distributions import (
    ContinuousDistribution,
    DiracDeltaDistribution,
    UnivariateDistribution,
)
from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer import (
    ForwardSampleAssignment,
    Layer,
    LayerConverter,
    LayerQuery,
    QueryCache,
    RustworkxUnitType,
    SumLayer,
    memoized,
)
from probabilistic_model.probabilistic_circuit.tensorized.rustworkx_conversion import (
    layer_class_of,
)
from probabilistic_model.probabilistic_circuit.tensorized.utils import SparseArray
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RustworkxProbabilisticCircuit,
    Unit,
    leaf,
)


def layer_of_distributions(
    variable_index: int, distributions: List[UnivariateDistribution]
) -> Layer:
    """
    Create the input layer that holds a list of distributions of the same type.

    :param variable_index: The index of the variable of the distributions.
    :param distributions: The distributions, all of the same type.
    :return: The input layer.
    """
    layer_class = layer_class_of(type(distributions[0]))
    return layer_class.from_distributions(variable_index, distributions)


def assemble_input_layer(
    variable_index: int,
    pieces: List[List[Tuple[UnivariateDistribution, float]]],
) -> Tuple[Layer, npt.NDArray]:
    """
    Assemble the result of a structural query on an input layer.

    Every node of the original layer contributes a list of pieces, each a distribution
    with its log-probability. A node contributes more than one piece when it was
    truncated to a composite set, and different nodes may contribute pieces of different
    types, for instance when truncating a Gaussian layer to an interval leaves some
    nodes untruncated.

    The result always has exactly as many nodes as the original layer, in the same
    order, so that the edges of the parents stay valid. If the pieces fit into a single
    input layer, that layer is returned directly; otherwise the pieces are grouped by
    type and a sum layer selects the pieces of every original node.

    :param variable_index: The index of the variable of the layer.
    :param pieces: The pieces per node of the original layer.
    :return: The new layer and the log-probabilities of its nodes.
    """
    log_probabilities = np.array(
        [
            (
                -np.inf
                if not node_pieces
                else float(
                    np.logaddexp.reduce(
                        [log_probability for _, log_probability in node_pieces]
                    )
                )
            )
            for node_pieces in pieces
        ]
    )

    types = {type(distribution) for node in pieces for distribution, _ in node}
    single_piece = all(len(node_pieces) == 1 for node_pieces in pieces)

    # the common case: every node stays a single distribution of a single type, so the
    # layer keeps its shape and no selecting sum layer is needed
    if single_piece and len(types) == 1:
        layer = layer_of_distributions(
            variable_index, [node_pieces[0][0] for node_pieces in pieces]
        )
        return layer, log_probabilities

    # group the pieces by type, remembering where each piece ended up
    distributions_by_type: Dict[Type, List[UnivariateDistribution]] = {}
    locations: List[List[Tuple[Type, int, float]]] = []
    for node_pieces in pieces:
        node_locations = []
        for distribution, log_probability in node_pieces:
            bucket = distributions_by_type.setdefault(type(distribution), [])
            node_locations.append((type(distribution), len(bucket), log_probability))
            bucket.append(distribution)
        locations.append(node_locations)

    ordered_types = list(distributions_by_type)
    child_layers = [
        layer_of_distributions(variable_index, distributions_by_type[type_])
        for type_ in ordered_types
    ]

    number_of_nodes = len(pieces)
    rows: List[List[int]] = [[] for _ in ordered_types]
    columns: List[List[int]] = [[] for _ in ordered_types]
    values: List[List[float]] = [[] for _ in ordered_types]

    for node, node_locations in enumerate(locations):
        for type_, column, log_probability in node_locations:
            bucket = ordered_types.index(type_)
            rows[bucket].append(node)
            columns[bucket].append(column)
            values[bucket].append(log_probability)

    log_weights = [
        SparseArray.from_coordinates(
            np.array(rows[bucket], dtype=np.int64),
            np.array(columns[bucket], dtype=np.int64),
            np.array(values[bucket], dtype=float),
            (number_of_nodes, child_layers[bucket].number_of_nodes),
        )
        for bucket in range(len(ordered_types))
    ]

    return SumLayer(child_layers, log_weights), log_probabilities


@dataclass(eq=False, repr=False)
class InputLayer(Layer[RustworkxUnitType], ABC):
    """
    Abstract base class for the input layers of a layered circuit.

    An input layer holds univariate distributions of one single type over one single
    variable, so that the likelihood of all of its nodes is evaluated without any
    branching.
    """

    variable: int
    """
    The index of the variable of this layer.
    """

    def __post_init__(self):
        self.variable = int(self.variable)

    @property
    def child_layers(self) -> List[Layer]:
        """
        :return: An empty list. An input layer is a leaf of the layer graph.
        """
        return []

    @property
    def variables(self) -> npt.NDArray:
        return np.array([self.variable], dtype=np.int64)

    def set_variables(self, value: npt.NDArray):
        """
        Overwrite the variable indices of this layer.

        :param value: The new indices.
        """
        [self.variable] = np.asarray(value, dtype=np.int64)

    def remap_variables(self, remap: npt.NDArray, cache: Optional[QueryCache] = None):
        self.variable = int(remap[self.variable])

    def column_of(self, x: npt.NDArray) -> npt.NDArray:
        """
        Select the column of the variable of this layer from an event array.

        :param x: The events with shape (#events, #variables of the circuit).
        :return: The column of this layer's variable.
        """
        return x[:, self.variable]

    # %% per node view

    @abstractmethod
    def node_distribution(
        self, index: int, variable: Variable
    ) -> UnivariateDistribution:
        """
        Materialize one node of this layer as a univariate distribution.

        :param index: The index of the node.
        :param variable: The variable of this layer.
        :return: The distribution of that node.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_distributions(
        cls, variable_index: int, distributions: List[UnivariateDistribution]
    ) -> Self:
        """
        Create a layer from a list of distributions of the type of this layer.

        :param variable_index: The index of the variable of the distributions.
        :param distributions: The distributions.
        :return: The layer.
        """
        raise NotImplementedError

    @abstractmethod
    def select_nodes(self, mask: npt.NDArray) -> Self:
        """
        Create a layer that only holds the nodes selected by a mask.

        :param mask: A boolean mask over the nodes of this layer.
        :return: The reduced layer.
        """
        raise NotImplementedError

    def node_distributions(self, variable: Variable) -> List[UnivariateDistribution]:
        """
        :param variable: The variable of this layer.
        :return: Every node of this layer as a univariate distribution.
        """
        return [
            self.node_distribution(index, variable)
            for index in range(self.number_of_nodes)
        ]

    # %% queries

    @memoized(LayerQuery.SUPPORT)
    def support_of_nodes(
        self, variables: SortedSet, cache: Optional[QueryCache] = None
    ) -> List[Event]:
        variable = variables[self.variable]
        return [
            distribution.support for distribution in self.node_distributions(variable)
        ]

    @memoized(LayerQuery.LOG_MODE)
    def log_mode_of_nodes(
        self, variables: SortedSet, cache: Optional[QueryCache] = None
    ) -> Tuple[List[Event], npt.NDArray]:
        variable = variables[self.variable]
        modes = [
            distribution.log_mode()
            for distribution in self.node_distributions(variable)
        ]
        return [mode for mode, _ in modes], np.array(
            [value for _, value in modes], dtype=float
        )

    @memoized(LayerQuery.MOMENT)
    def moment_of_nodes(
        self,
        order: npt.NDArray,
        center: npt.NDArray,
        requested: npt.NDArray,
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
    ) -> npt.NDArray:
        result = np.zeros((self.number_of_nodes, len(order)))
        if not requested[self.variable]:
            return result
        result[:, self.variable] = self.moment_of_nodes_own(
            int(order[self.variable]),
            float(center[self.variable]),
            variables[self.variable],
        )
        return result

    def moment_of_nodes_own(
        self, order: int, center: float, variable: Variable
    ) -> npt.NDArray:
        """
        Calculate the moment of the variable of this layer for every node.

        The fallback evaluates the nodes one by one through their distributions. Layers
        whose moment has a closed form that numpy can evaluate for all nodes at once
        override this.

        :param order: The order of the moment.
        :param center: The center of the moment.
        :param variable: The variable of this layer.
        :return: The moments with shape (#nodes,).
        """
        order_map = VariableMap({variable: order})
        center_map = VariableMap({variable: center})
        return np.array(
            [
                distribution.moment(order_map, center_map)[variable]
                for distribution in self.node_distributions(variable)
            ],
            dtype=float,
        )

    def sample_forward(
        self,
        assignment: ForwardSampleAssignment,
        samples: npt.NDArray,
        variables: SortedSet,
    ):
        own_assignment = assignment.rows_of(self)
        for node, rows_of_node in enumerate(own_assignment):
            if not rows_of_node:
                continue
            rows = np.concatenate(rows_of_node)
            samples[rows, self.variable] = self.sample_of_node(
                node, len(rows), variables
            )

    def sample_of_node(
        self, node: int, amount: int, variables: SortedSet
    ) -> npt.NDArray:
        """
        Draw samples from a single node of this layer.

        :param node: The index of the node.
        :param amount: The number of samples.
        :param variables: The variables of the circuit.
        :return: The samples with shape (amount,).
        """
        distribution = self.node_distribution(node, variables[self.variable])
        return distribution.sample(amount)[:, 0]

    # %% structural

    def truncate_node(
        self,
        distribution: UnivariateDistribution,
        assignment: AbstractCompositeSet,
        singleton_allowed: bool,
    ) -> List[Tuple[UnivariateDistribution, float]]:
        """
        Truncate a single node to the assignment of its variable.

        :param distribution: The distribution of the node.
        :param assignment: The assignment of the variable in the truncating event.
        :param singleton_allowed: Whether singletons are allowed.
        :return: The pieces the node is truncated into.
        """
        truncated, log_probability = distribution.log_truncated(
            SimpleEvent.from_data(
                {distribution.variable: assignment}
            ).as_composite_set(),
            singleton_allowed,
        )
        if truncated is None or log_probability == -np.inf:
            return [(distribution, -np.inf)]
        return [(truncated, float(log_probability))]

    def log_truncated_of_assignment(
        self, assignment: AbstractCompositeSet, singleton_allowed: bool
    ) -> Optional[Tuple[Layer, npt.NDArray]]:
        """
        Truncate every node of this layer at once.

        This is the vectorized counterpart of :meth:`truncate_node`. Returning ``None``
        means "not supported for this layer or this assignment", and the caller falls
        back to materializing every node as a distribution and truncating it one by one.
        Overriding this is what keeps a truncation from costing one python call per
        node, which is the dominant cost when truncating to an event with many simple
        sets.

        :param assignment: The assignment of the variable of this layer.
        :param singleton_allowed: Whether singletons are allowed.
        :return: The truncated layer and the log-probabilities of its nodes, or
            ``None``.
        """
        return None

    def log_truncated_of_simple_event(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        singleton_allowed: bool,
        cache: Optional[QueryCache] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        if cache is None:
            cache = QueryCache()
        if cache.has(LayerQuery.TRUNCATED, self):
            return cache.get(LayerQuery.TRUNCATED, self)

        variable = variables[self.variable]
        assignment = event[variable]

        vectorized = self.log_truncated_of_assignment(assignment, singleton_allowed)
        if vectorized is not None:
            layer, node_log_probabilities = vectorized
        else:
            pieces = [
                self.truncate_node(distribution, assignment, singleton_allowed)
                for distribution in self.node_distributions(variable)
            ]
            layer, node_log_probabilities = assemble_input_layer(self.variable, pieces)

        if log_probabilities is not None:
            log_probabilities[id(layer)] = node_log_probabilities

        return cache.set(LayerQuery.TRUNCATED, self, (layer, node_log_probabilities))

    @classmethod
    def concatenate(cls, layers: List[Self]) -> Optional[Self]:
        """
        Join layers of this type over the same variable into one layer.

        The nodes keep the order of the layers, so the nodes of ``layers[k]`` occupy one
        contiguous block. Only layers that :meth:`log_truncated_of_assignment` produced
        from the same layer are concatenated, which is why the shared parameters may be
        taken from the first one.

        :param layers: The layers to join.
        :return: The joined layer, or ``None`` if this layer type cannot be joined.
        """
        return None

    def log_truncated_of_simple_events(
        self,
        events: List[SimpleEvent],
        variables: SortedSet,
        singleton_allowed: bool,
        cache: Optional[QueryCache] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Optional[Tuple[Layer, npt.NDArray]]:
        if cache is None:
            cache = QueryCache()
        if cache.has(LayerQuery.BATCHED_TRUNCATED, self):
            return cache.get(LayerQuery.BATCHED_TRUNCATED, self)

        variable = variables[self.variable]

        truncated = []
        for event in events:
            result = self.log_truncated_of_assignment(
                event[variable], singleton_allowed
            )
            # the generic per-node path may split a node into several pieces or change
            # its type, neither of which keeps the block layout this pass relies on
            if result is None:
                return None
            truncated.append(result)

        layers = [layer for layer, _ in truncated]
        if len({type(layer) for layer in layers}) != 1:
            return None

        layer = type(layers[0]).concatenate(layers)
        if layer is None:
            return None

        node_log_probabilities = np.concatenate(
            [log_probability for _, log_probability in truncated]
        )
        log_probabilities[id(layer)] = node_log_probabilities

        return cache.set(
            LayerQuery.BATCHED_TRUNCATED, self, (layer, node_log_probabilities)
        )

    def log_conditional_of_point(
        self,
        point: Dict[Variable, Any],
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        if cache is None:
            cache = QueryCache()
        if cache.has(LayerQuery.CONDITIONAL, self):
            return cache.get(LayerQuery.CONDITIONAL, self)

        variable = variables[self.variable]

        if variable not in point:
            layer = self.__deepcopy__()
            node_log_probabilities = np.zeros(self.number_of_nodes)
        else:
            pieces = []
            for distribution in self.node_distributions(variable):
                conditional, log_probability = distribution.log_conditional(point)
                if conditional is None or log_probability == -np.inf:
                    pieces.append([(distribution, -np.inf)])
                else:
                    pieces.append([(conditional, float(log_probability))])
            layer, node_log_probabilities = assemble_input_layer(self.variable, pieces)

        if log_probabilities is not None:
            log_probabilities[id(layer)] = node_log_probabilities

        return cache.set(LayerQuery.CONDITIONAL, self, (layer, node_log_probabilities))

    def rebuild(
        self,
        needed: Dict[int, npt.NDArray],
        rebuilt: Dict[int, Optional[Layer]],
    ) -> Optional[Layer]:
        alive = needed[id(self)]
        if not alive.any():
            return None
        return self.select_nodes(alive)

    def marginal(
        self, kept: npt.NDArray, cache: Optional[QueryCache] = None
    ) -> Optional[Layer]:
        if not kept[self.variable]:
            return None
        return self.__deepcopy__()

    # %% conversion

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(
        cls,
        nodes: List[Unit],
        child_layers: List[LayerConverter],
        progress_bar: bool = False,
    ) -> LayerConverter:
        hash_remap = {hash(node): index for index, node in enumerate(nodes)}
        variable = nodes[0].variable
        iterator = (
            tqdm.tqdm(nodes, desc=f"Assembling input layer for {variable.name}")
            if progress_bar
            else nodes
        )
        layer = cls.from_distributions(
            nodes[0].probabilistic_circuit.variables.index(variable),
            [node.distribution for node in iterator],
        )
        return LayerConverter(layer, nodes, hash_remap)

    def to_rustworkx(
        self,
        variables: SortedSet,
        result: RustworkxProbabilisticCircuit,
        cache: Optional[QueryCache] = None,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Unit]:
        if cache is None:
            cache = QueryCache()
        if cache.has(LayerQuery.TO_RUSTWORKX, self):
            return cache.get(LayerQuery.TO_RUSTWORKX, self)

        variable = variables[self.variable]
        if progress_bar:
            progress_bar.set_postfix_str(f"Parsing input layer for {variable.name}")

        units = [
            leaf(distribution, result)
            for distribution in self.node_distributions(variable)
        ]

        if progress_bar:
            progress_bar.update(self.number_of_nodes)

        return cache.set(LayerQuery.TO_RUSTWORKX, self, units)


@dataclass(eq=False, repr=False)
class ContinuousLayer(InputLayer[RustworkxUnitType], ABC):
    """
    Abstract base class for the input layers of continuous univariate distributions.
    """

    def truncate_node(
        self,
        distribution: ContinuousDistribution,
        assignment: Interval,
        singleton_allowed: bool,
    ) -> List[Tuple[UnivariateDistribution, float]]:
        # a continuous node truncated to a composite interval becomes a mixture of the
        # truncations to its simple intervals, exactly as in the rx implementation
        pieces = []
        for simple_interval in assignment.simple_sets:
            truncated, log_probability = (
                distribution.log_conditional_from_simple_interval(
                    simple_interval, singleton_allowed
                )
            )
            if truncated is None or log_probability == -np.inf:
                continue
            pieces.append((truncated, float(log_probability)))

        if not pieces:
            return [(distribution, -np.inf)]
        return pieces

    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
    ) -> npt.NDArray:
        if cache is None:
            cache = QueryCache()
        if cache.has(LayerQuery.PROBABILITY_OF_SIMPLE_EVENT, self):
            return cache.get(LayerQuery.PROBABILITY_OF_SIMPLE_EVENT, self)

        interval: Interval = event[variables[self.variable]]
        result = np.zeros(self.number_of_nodes)
        for simple_interval in interval.simple_sets:
            bounds = np.array(
                [[simple_interval.lower], [simple_interval.upper]], dtype=float
            )
            values = self.cumulative_distribution_of_nodes_from_column(bounds[:, 0])
            result += values[1] - values[0]

        return cache.set(LayerQuery.PROBABILITY_OF_SIMPLE_EVENT, self, result)

    @memoized(LayerQuery.CUMULATIVE_DISTRIBUTION)
    def cumulative_distribution_of_nodes(
        self, x: npt.NDArray, cache: Optional[QueryCache] = None
    ) -> npt.NDArray:
        return self.cumulative_distribution_of_nodes_from_column(self.column_of(x))

    @abstractmethod
    def cumulative_distribution_of_nodes_from_column(
        self, x: npt.NDArray
    ) -> npt.NDArray:
        """
        Calculate the cumulative distribution function of every node for a column of
        values of the variable of this layer.

        :param x: The values with shape (#events,).
        :return: The values with shape (#events, #nodes).
        """
        raise NotImplementedError


@dataclass(eq=False, repr=False)
class ContinuousLayerWithFiniteSupport(ContinuousLayer[RustworkxUnitType], ABC):
    """
    Abstract base class for continuous input layers whose nodes have a finite support.
    """

    interval: npt.NDArray
    """
    The support of every node as an array of shape (#nodes, 2).

    The first column holds the lower bounds, the second the upper bounds.
    """

    # keyword-only so that a subclass can add required positional fields (such as
    # location and scale) after ``interval`` without violating dataclass field
    # ordering, which does not allow a required field to follow one that has a default
    bounds: Optional[npt.NDArray] = field(default=None, kw_only=True)
    """
    The kind of every bound as an array of shape (#nodes, 2) holding
    :class:`random_events.interval.Bound` values.

    The jax implementation nudges closed bounds outwards by one floating point step and
    treats every interval as open. Keeping the bounds instead costs one comparison and
    lets a layer describe exactly the same support as the circuit it was created from,
    which the support and mode queries compare against.
    """

    def __post_init__(self):
        super().__post_init__()
        self.interval = np.asarray(self.interval, dtype=float).reshape(-1, 2)
        if self.bounds is None:
            self.bounds = np.full(self.interval.shape, int(Bound.OPEN), dtype=np.int64)
        self.bounds = np.asarray(self.bounds, dtype=np.int64).reshape(-1, 2)

    @property
    def lower(self) -> npt.NDArray:
        """
        :return: The lower bounds of the supports of the nodes.
        """
        return self.interval[:, 0]

    @property
    def upper(self) -> npt.NDArray:
        """
        :return: The upper bounds of the supports of the nodes.
        """
        return self.interval[:, 1]

    @property
    def left_closed(self) -> npt.NDArray:
        """
        :return: Whether the lower bound of every node is included.
        """
        return self.bounds[:, 0] == int(Bound.CLOSED)

    @property
    def right_closed(self) -> npt.NDArray:
        """
        :return: Whether the upper bound of every node is included.
        """
        return self.bounds[:, 1] == int(Bound.CLOSED)

    @property
    def number_of_nodes(self) -> int:
        return len(self.interval)

    def simple_interval_of(self, index: int) -> SimpleInterval:
        """
        :param index: The index of a node.
        :return: The support of that node as simple interval.
        """
        return SimpleInterval.from_data(
            float(self.interval[index, 0]),
            float(self.interval[index, 1]),
            Bound(int(self.bounds[index, 0])),
            Bound(int(self.bounds[index, 1])),
        )

    def validate_own(self):
        if self.interval.shape != self.bounds.shape:
            raise ShapeMismatchError(self.interval.shape, self.bounds.shape)

    def included_condition(self, x: npt.NDArray) -> npt.NDArray:
        """
        Check whether values lie inside the support of every node.

        :param x: The values with shape (#events,).
        :return: A boolean array of shape (#events, #nodes).
        """
        column = np.asarray(x, dtype=float).reshape(-1, 1)

        # these arrays hold one entry per event per node, so the homogeneous cases get
        # their own path rather than evaluating both comparisons and selecting between
        # them
        left_closed = self.left_closed
        if left_closed.all():
            left = self.lower <= column
        elif not left_closed.any():
            left = self.lower < column
        else:
            left = np.where(left_closed, self.lower <= column, self.lower < column)

        right_closed = self.right_closed
        if right_closed.all():
            right = column <= self.upper
        elif not right_closed.any():
            right = column < self.upper
        else:
            right = np.where(right_closed, column <= self.upper, column < self.upper)

        return left & right

    def select_nodes(self, mask: npt.NDArray) -> Self:
        return self.__class__(
            self.variable, self.interval[mask], bounds=self.bounds[mask]
        )

    @classmethod
    def concatenate(cls, layers: List[Self]) -> Self:
        return cls(
            layers[0].variable,
            np.concatenate([layer.interval for layer in layers]),
            bounds=np.concatenate([layer.bounds for layer in layers]),
        )

    def apply_translation_own(self, translation: npt.NDArray):
        self.interval = self.interval + translation[self.variable]

    def apply_scaling_own(self, scaling: npt.NDArray):
        self.interval = self.interval * scaling[self.variable]

    def __deepcopy__(self, memo=None) -> Self:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        result = self.__class__(
            self.variable, self.interval.copy(), bounds=self.bounds.copy()
        )
        memo[id(self)] = result
        return result


@dataclass(eq=False, repr=False)
class DiracDeltaLayer(ContinuousLayer[DiracDeltaDistribution]):
    """
    A layer of Dirac delta distributions over one continuous variable.
    """

    location: npt.NDArray
    """
    The location of every node.
    """

    density_cap: npt.NDArray
    """
    The value that replaces the infinite density of every node.
    """

    tolerance: float = 1e-6
    """
    The tolerance with which a value is considered equal to the location.
    """

    def __post_init__(self):
        super().__post_init__()
        self.location = np.asarray(self.location, dtype=float).reshape(-1)
        self.density_cap = np.asarray(self.density_cap, dtype=float).reshape(-1)

    @property
    def number_of_nodes(self) -> int:
        return len(self.location)

    @property
    def number_of_own_parameters(self) -> int:
        return 2 * self.number_of_nodes

    def validate_own(self):
        if self.location.shape != self.density_cap.shape:
            raise ShapeMismatchError(self.location.shape, self.density_cap.shape)

    def node_distribution(
        self, index: int, variable: Variable
    ) -> DiracDeltaDistribution:
        return DiracDeltaDistribution(
            variable=variable,
            location=float(self.location[index]),
            density_cap=float(self.density_cap[index]),
            tolerance=self.tolerance,
        )

    @classmethod
    def from_distributions(
        cls, variable_index: int, distributions: List[DiracDeltaDistribution]
    ) -> Self:
        return cls(
            variable_index,
            np.array([distribution.location for distribution in distributions]),
            np.array([distribution.density_cap for distribution in distributions]),
            distributions[0].tolerance,
        )

    def select_nodes(self, mask: npt.NDArray) -> Self:
        return self.__class__(
            self.variable, self.location[mask], self.density_cap[mask], self.tolerance
        )

    @classmethod
    def concatenate(cls, layers: List[Self]) -> Self:
        return cls(
            layers[0].variable,
            np.concatenate([layer.location for layer in layers]),
            np.concatenate([layer.density_cap for layer in layers]),
            layers[0].tolerance,
        )

    @memoized(LayerQuery.LOG_LIKELIHOOD)
    def log_likelihood_of_nodes(
        self, x: npt.NDArray, cache: Optional[QueryCache] = None
    ) -> npt.NDArray:
        column = self.column_of(x).astype(float).reshape(-1, 1)
        hit = np.abs(column - self.location) < self.tolerance
        with np.errstate(divide="ignore"):
            return np.where(hit, np.log(self.density_cap), -np.inf)

    def cumulative_distribution_of_nodes_from_column(
        self, x: npt.NDArray
    ) -> npt.NDArray:
        column = np.asarray(x, dtype=float).reshape(-1, 1)
        return (column >= self.location - self.tolerance).astype(float)

    def moment_of_nodes_own(
        self, order: int, center: float, variable: Variable
    ) -> npt.NDArray:
        if order == 0:
            return np.ones(self.number_of_nodes)
        if order == 1:
            return self.location - center
        return np.zeros(self.number_of_nodes)

    def sample_of_node(
        self, node: int, amount: int, variables: SortedSet
    ) -> npt.NDArray:
        return np.full(amount, self.location[node])

    def log_truncated_of_assignment(
        self, assignment: Interval, singleton_allowed: bool
    ) -> Tuple[DiracDeltaLayer, npt.NDArray]:
        """
        Truncating a Dirac delta either keeps it unchanged or makes it impossible, so
        the whole layer is truncated by testing which locations the assignment contains.
        """
        inside = np.zeros(self.number_of_nodes, dtype=bool)
        for interval in assignment.simple_sets:
            left = (
                interval.lower <= self.location
                if interval.left == Bound.CLOSED
                else interval.lower < self.location
            )
            right = (
                self.location <= interval.upper
                if interval.right == Bound.CLOSED
                else self.location < interval.upper
            )
            inside |= left & right

        return self.__deepcopy__(), np.where(inside, 0.0, -np.inf)

    def apply_translation_own(self, translation: npt.NDArray):
        self.location = self.location + translation[self.variable]

    def apply_scaling_own(self, scaling: npt.NDArray):
        self.location = self.location * scaling[self.variable]

    def __deepcopy__(self, memo=None) -> DiracDeltaLayer:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        result = self.__class__(
            self.variable,
            self.location.copy(),
            self.density_cap.copy(),
            self.tolerance,
        )
        memo[id(self)] = result
        return result
