from __future__ import annotations

import dataclasses
import enum
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import rustworkx
import tqdm
from krrood.adapters import json_serializer
from krrood.adapters.json_serializer import SubclassJSONSerializer
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import (
    Any,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Self,
    Tuple,
    TypeVar,
)

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RustworkxProbabilisticCircuit,
    Unit,
)

RustworkxUnitType = TypeVar("RustworkxUnitType")
"""
The class of the ``probabilistic_model.probabilistic_circuit.rx`` package that a
:class:`Layer` subclass represents: a unit class for an inner layer, or the distribution
class of a leaf unit for an input layer.
"""


class LayerQuery(enum.Enum):
    """
    The passes over the layers whose per-layer result a :class:`QueryCache` holds.

    A pass evaluates its query once per layer, so the query it belongs to is one half of
    the key a cached result sits under and the layer it was evaluated for is the other.
    """

    LOG_LIKELIHOOD = enum.auto()
    CUMULATIVE_DISTRIBUTION = enum.auto()
    PROBABILITY_OF_SIMPLE_EVENT = enum.auto()
    SUPPORT = enum.auto()
    LOG_MODE = enum.auto()
    MOMENT = enum.auto()
    TRUNCATED = enum.auto()
    BATCHED_TRUNCATED = enum.auto()
    CONDITIONAL = enum.auto()
    MARGINAL = enum.auto()
    SIMPLIFY = enum.auto()
    REMAP_VARIABLES = enum.auto()
    TO_RUSTWORKX = enum.auto()


@dataclass(frozen=True)
class QueryCacheKey:
    """
    Which query, evaluated for which layer, an entry of a :class:`QueryCache` is the
    result of.
    """

    query: LayerQuery
    """
    The query that was evaluated.
    """

    layer_id: int
    """
    The :func:`id` of the layer it was evaluated for.

    Layers are keyed by identity rather than by value: a pass has to evaluate a layer
    that is the child of several parents once, and two layers that happen to hold equal
    parameters are still two layers with two results.
    """


@dataclass
class QueryCache:
    """
    The results one pass over the layers has computed so far.

    Layers form a directed acyclic graph, not a tree: a layer that is the child of
    several parents must only be evaluated once per pass. Every method that walks the
    graph takes this cache as a ``cache`` keyword argument and hands it down to the
    calls it makes on its own children; the top level caller may omit it and gets a
    fresh one.
    """

    results: Dict[QueryCacheKey, Any] = field(default_factory=dict)
    """
    The result of every query evaluated so far, per layer.
    """

    def has(self, query: LayerQuery, layer: Layer) -> bool:
        """
        :param query: The query to look up.
        :param layer: The layer to look it up for.
        :return: Whether the result is already in this cache.
        """
        return QueryCacheKey(query, id(layer)) in self.results

    def get(self, query: LayerQuery, layer: Layer) -> Any:
        """
        :param query: The query to look up.
        :param layer: The layer to look it up for.
        :return: The cached result.
        :raises KeyError: If the query was not evaluated for that layer yet.
        """
        return self.results[QueryCacheKey(query, id(layer))]

    def set(self, query: LayerQuery, layer: Layer, result: Any) -> Any:
        """
        Record the result of a query for a layer.

        :param query: The query that was evaluated.
        :param layer: The layer it was evaluated for.
        :param result: The result.
        :return: That same result, so that a caller can ``return cache.set(...)``.
        """
        self.results[QueryCacheKey(query, id(layer))] = result
        return result


def memoized(query: LayerQuery):
    """
    Memoize a bottom-up query of a layer by the identity of the layer.

    :param query: The query the wrapped method evaluates.
    :return: The decorator.
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, cache: Optional[QueryCache] = None, **kwargs):
            if cache is None:
                cache = QueryCache()
            if not cache.has(query, self):
                cache.set(query, self, method(self, *args, cache=cache, **kwargs))
            return cache.get(query, self)

        return wrapper

    return decorator


@dataclass
class Edge:
    """
    One edge of an inner layer: a node of that layer, and the node of one of its child
    layers that it points at.
    """

    node: int
    """
    The index of the node inside the layer the edge belongs to.
    """

    child_layer_index: int
    """
    The index of the child layer the edge points into, within
    :attr:`InnerLayer.child_layers`.
    """

    child_node: int
    """
    The index of the node inside that child layer.
    """


@dataclass
class ForwardSampleAssignment:
    """
    Bookkeeping for a top-down sampling pass over a circuit.

    A layer routes the output rows assigned to each of its nodes to the nodes of its
    child layers; a child layer that is shared by several parents accumulates rows from
    each of them before it is its own turn to route them further.
    """

    rows_by_node: Dict[int, List[List[npt.NDArray]]]
    """
    For every layer, indexed by its id, the row-index arrays assigned to each of its
    nodes so far.
    """

    @classmethod
    def for_layers(cls, layers: Iterable[Layer]) -> Self:
        """
        :param layers: Every layer that will be visited during the pass.
        :return: An assignment with an empty bucket for every node of every layer.
        """
        return cls(
            {id(layer): [[] for _ in range(layer.number_of_nodes)] for layer in layers}
        )

    def assign(self, layer: Layer, node: int, rows: npt.NDArray) -> None:
        """
        Route output rows to one node of a layer.

        :param layer: The layer the node belongs to.
        :param node: The index of the node within that layer.
        :param rows: The output rows drawn from that node.
        """
        self.rows_by_node[id(layer)][node].append(rows)

    def rows_of(self, layer: Layer) -> List[List[npt.NDArray]]:
        """
        :param layer: The layer to read the assignment of.
        :return: The row-index arrays assigned to every node of that layer so far, one
            list per node.
        """
        return self.rows_by_node[id(layer)]


@dataclass
class LayerWithDepth:
    """
    A layer of a circuit together with its distance from the root.
    """

    depth: int
    """
    The number of layers between the root layer and this layer.
    """

    layer: Layer
    """
    The layer at that depth.
    """


class Layer(
    Generic[RustworkxUnitType], SubClassSafeGeneric, SubclassJSONSerializer, ABC
):
    """
    Abstract base class for the layers of a layered probabilistic circuit.

    Every node of a layer has the same scope (set of variables) and, for input layers,
    the same type of distribution. The parameters of all nodes of a layer are stored in
    contiguous arrays, which is what allows every query to be evaluated for all nodes of
    a layer at once.

    Variables are referred to by their index in the ``variables`` of the owning
    :class:`probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit.LayeredProbabilisticCircuit`
    rather than by the variable objects themselves.

    Concrete subclasses bind :data:`RustworkxUnitType` to the ``rx`` class they represent,
    for instance ``ProductLayer(InnerLayer[ProductUnit])``; the conversion in
    :mod:`probabilistic_model.probabilistic_circuit.tensorized.rustworkx_conversion` reads
    it back with :meth:`get_generic_type_parameters` to find the layer class for a unit.
    """

    # %% structure

    child_layers: List[Layer]
    """
    The layers below this one: a field on :class:`InnerLayer`, an empty list on the
    input layers.

    An annotation rather than a property, so that the generated ``__init__`` of
    :class:`InnerLayer` can assign to it.
    """

    @property
    @abstractmethod
    def variables(self) -> npt.NDArray:
        """
        :return: The sorted indices of the variables in the scope of this layer.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def number_of_nodes(self) -> int:
        """
        :return: The number of nodes in this layer.
        """
        raise NotImplementedError

    @property
    def number_of_components(self) -> int:
        """
        :return: The number of components (nodes and edges) of the circuit rooted here.
        """
        return self.number_of_nodes

    @property
    def number_of_parameters(self) -> int:
        """
        :return: The number of parameters of the circuit rooted at this layer.
        """
        return sum(layer.number_of_own_parameters for layer in self.all_layers())

    @property
    @abstractmethod
    def number_of_own_parameters(self) -> int:
        """
        :return: The number of parameters stored in this layer alone.
        """
        raise NotImplementedError

    def validate(self):
        """
        Check that the parameter arrays of this layer and all its descendants have
        consistent shapes.

        :raises ShapeMismatchError: If a shape is inconsistent.
        """
        for layer in self.all_layers():
            layer.validate_own()

    @abstractmethod
    def validate_own(self):
        """
        Check the shapes of the parameters stored in this layer alone.

        :raises ShapeMismatchError: If a shape is inconsistent.
        """
        raise NotImplementedError

    def all_layers(self) -> List[Layer]:
        """
        :return: Every layer of the circuit rooted here, each exactly once, parents
            before children.
        """
        result: List[Layer] = []
        self._visit_once(result, set())
        return result

    def _visit_once(self, result: List[Layer], seen: set):
        """
        Append this layer and its descendants to ``result``, each exactly once.

        :param result: The list to append to, in visiting order.
        :param seen: The ids of the layers already visited.
        """
        if id(self) in seen:
            return
        seen.add(id(self))
        result.append(self)
        for child_layer in self.child_layers:
            child_layer._visit_once(result, seen)

    def all_layers_with_depth(self, depth: int = 0) -> List[LayerWithDepth]:
        """
        :param depth: The depth to report for this layer.
        :return: Every layer of the circuit rooted here with its depth. Layers that are
            reachable along several paths appear once per path, mirroring the jax
            implementation.
        """
        result = [LayerWithDepth(depth, self)]
        for child_layer in self.child_layers:
            result.extend(child_layer.all_layers_with_depth(depth + 1))
        return result

    def topological_layer_order(self) -> List[Layer]:
        """
        Order the layers of the circuit rooted here such that every layer appears after
        all of its parents.

        This is the order in which a top-down pass (such as sampling or :meth:`prune`)
        has to visit the layers so that a layer is only processed once every parent has
        contributed to it. A breadth-first order does not give that guarantee: a layer
        that several parents share is reached at the smallest of their distances from
        the root, which can be before a parent further down has been visited.

        :return: The layers in topological order.
        """
        layers = self.all_layers()
        graph = rustworkx.PyDiGraph()
        index_of = {
            id(layer): index
            for layer, index in zip(layers, graph.add_nodes_from(layers))
        }
        graph.add_edges_from_no_data(
            [
                (index_of[id(layer)], index_of[id(child_layer)])
                for layer in layers
                for child_layer in layer.child_layers
            ]
        )
        return [graph[index] for index in rustworkx.topological_sort(graph)]

    # %% queries

    @abstractmethod
    def log_likelihood_of_nodes(
        self, x: npt.NDArray, cache: Optional[QueryCache] = None
    ) -> npt.NDArray:
        """
        Calculate the log-likelihood of every node of this layer.

        :param x: The events with shape (#events, #variables of the circuit).
        :param cache: The shared cache of the current query.
        :return: The log-likelihoods with shape (#events, #nodes).
        """
        raise NotImplementedError

    @abstractmethod
    def cumulative_distribution_of_nodes(
        self, x: npt.NDArray, cache: Optional[QueryCache] = None
    ) -> npt.NDArray:
        """
        Calculate the cumulative distribution function of every node of this layer.

        :param x: The events with shape (#events, #variables of the circuit).
        :param cache: The shared cache of the current query.
        :return: The values with shape (#events, #nodes).
        """
        raise NotImplementedError

    @abstractmethod
    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
    ) -> npt.NDArray:
        """
        Calculate the probability of a simple event for every node of this layer.

        :param event: The simple event.
        :param variables: The variables of the circuit.
        :param cache: The shared cache of the current query.
        :return: The probabilities with shape (#nodes,).
        """
        raise NotImplementedError

    @abstractmethod
    def support_of_nodes(
        self, variables: SortedSet, cache: Optional[QueryCache] = None
    ) -> List[Event]:
        """
        Calculate the support of every node of this layer.

        :param variables: The variables of the circuit.
        :param cache: The shared cache of the current query.
        :return: One event per node.
        """
        raise NotImplementedError

    @abstractmethod
    def log_mode_of_nodes(
        self, variables: SortedSet, cache: Optional[QueryCache] = None
    ) -> Tuple[List[Event], npt.NDArray]:
        """
        Calculate the mode of every node of this layer.

        :param variables: The variables of the circuit.
        :param cache: The shared cache of the current query.
        :return: One event per node and the log-likelihoods of the modes.
        """
        raise NotImplementedError

    @abstractmethod
    def moment_of_nodes(
        self,
        order: npt.NDArray,
        center: npt.NDArray,
        requested: npt.NDArray,
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
    ) -> npt.NDArray:
        """
        Calculate the moment of every node of this layer.

        :param order: The order per variable of the circuit.
        :param center: The center per variable of the circuit.
        :param requested: A boolean mask of the variables the moment is requested for.
        :param cache: The shared cache of the current query.
        :return: The moments with shape (#nodes, #variables of the circuit).
        """
        raise NotImplementedError

    @abstractmethod
    def sample_forward(
        self,
        assignment: ForwardSampleAssignment,
        samples: npt.NDArray,
        variables: SortedSet,
    ):
        """
        Route the sample rows that the parents of this layer assigned to its nodes.

        :param assignment: The rows assigned to every node of every layer so far.
        :param samples: The array the input layers write their samples into.
        :param variables: The variables of the circuit.
        """
        raise NotImplementedError

    # %% structural

    @abstractmethod
    def log_truncated_of_simple_event(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        singleton_allowed: bool,
        cache: Optional[QueryCache] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        """
        Truncate every node of this layer to a simple event.

        The returned layer has exactly as many nodes, in the same order, as this layer,
        so that the edges of the parents stay valid. Nodes that became impossible are
        reported with a log-probability of ``-inf`` and are removed by the following
        :meth:`prune` pass.

        :param event: The simple event to truncate to.
        :param variables: The variables of the circuit.
        :param singleton_allowed: Whether singletons are allowed in the event.
        :param cache: The shared cache of the current query.
        :param log_probabilities: The map the per-node log-probabilities of the new
            layers are written into, keyed by the id of the new layer.
        :return: The truncated layer and the log-probabilities of its nodes.
        """
        raise NotImplementedError

    @abstractmethod
    def log_truncated_of_simple_events(
        self,
        events: List[SimpleEvent],
        variables: SortedSet,
        singleton_allowed: bool,
        cache: Optional[QueryCache] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Optional[Tuple[Layer, npt.NDArray]]:
        """
        Truncate this layer to several simple events at once.

        The result holds one copy of every node per event: the node ``i`` truncated to the
        ``k``-th event sits at index ``k * self.number_of_nodes + i``. Truncating to an
        event with many simple sets this way keeps the number of *layers* constant and
        grows the parameter blocks instead, where truncating once per simple set and
        mixing the results produces one set of layers per simple set and takes the layered
        representation apart.

        :param events: The simple events to truncate to.
        :param variables: The variables of the circuit.
        :param singleton_allowed: Whether singletons are allowed in the events.
        :param cache: The shared cache of the current query.
        :param log_probabilities: The map the per-node log-probabilities are written to.
        :return: The truncated layer and the log-probabilities of its nodes, or ``None``
            if this layer or one below it cannot be truncated this way and the caller has
            to fall back to truncating once per event.
        """
        raise NotImplementedError

    @abstractmethod
    def log_conditional_of_point(
        self,
        point: Dict[Variable, Any],
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        """
        Condition every node of this layer on a partial point.

        See :meth:`log_truncated_of_simple_event` for the contract of the result.

        :param point: The partial point.
        :param variables: The variables of the circuit.
        :param cache: The shared cache of the current query.
        :param log_probabilities: The map the per-node log-probabilities are written to.
        :return: The conditioned layer and the log-probabilities of its nodes.
        """
        raise NotImplementedError

    def alive_own(self, log_probabilities: Dict[int, npt.NDArray]) -> npt.NDArray:
        """
        Read back which nodes of this layer a structural query left possible.

        A structural pass such as :meth:`log_truncated_of_simple_event` keeps the node
        count of every layer it rewrites, so that the edges of the parents stay valid,
        and reports the nodes that became impossible with a log-probability of ``-inf``.
        A node of this layer is therefore still possible if its recorded log-probability
        is above ``-inf``. A layer that the pass recorded nothing for is one it did not
        rewrite, so none of its nodes became impossible.

        :param log_probabilities: The per-layer log-probabilities of the structural pass
            that created this layer, keyed by the id of the layer.
        :return: A boolean mask of the nodes of this layer that are still possible.
        """
        own = log_probabilities.get(id(self))
        if own is None:
            return np.ones(self.number_of_nodes, dtype=bool)
        return own > -np.inf

    def required_child_nodes(
        self, alive: npt.NDArray, log_probabilities: Dict[int, npt.NDArray]
    ) -> List[Tuple[Layer, npt.NDArray]]:
        """
        Determine which nodes of the direct children a set of live nodes still needs.

        :param alive: A boolean mask of the live nodes of this layer.
        :param log_probabilities: The per-layer log-probabilities of the structural
            pass.
        :return: One ``(child layer, mask)`` pair per child layer.
        """
        return []

    @abstractmethod
    def rebuild(
        self,
        needed: Dict[int, npt.NDArray],
        rebuilt: Dict[int, Optional[Layer]],
    ) -> Optional[Layer]:
        """
        Create the pruned version of this layer.

        :param needed: The live node mask of every layer, keyed by layer id.
        :param rebuilt: The already pruned child layers, keyed by the id of the original
            layer. A value of ``None`` marks a layer that lost all of its nodes.
        :return: The pruned layer, or ``None`` if no node survives.
        """
        raise NotImplementedError

    def prune(self, log_probabilities: Dict[int, npt.NDArray]) -> Optional[Layer]:
        """
        Remove every impossible and every unreachable node of the circuit rooted here.

        The pass first propagates liveness downwards in topological order, so that a
        layer shared by several parents is pruned once against the union of what its
        parents need, and then rebuilds the layers bottom-up.

        :param log_probabilities: The per-layer log-probabilities of the structural pass
            that created this circuit.
        :return: The pruned circuit, or ``None`` if the root became impossible.
        """
        order = self.topological_layer_order()

        needed: Dict[int, npt.NDArray] = {
            id(self): np.ones(self.number_of_nodes, dtype=bool)
        }
        for layer in order:
            alive = needed.get(
                id(layer), np.zeros(layer.number_of_nodes, dtype=bool)
            ) & layer.alive_own(log_probabilities)
            needed[id(layer)] = alive
            for child_layer, mask in layer.required_child_nodes(
                alive, log_probabilities
            ):
                if id(child_layer) in needed:
                    needed[id(child_layer)] = needed[id(child_layer)] | mask
                else:
                    needed[id(child_layer)] = mask

        rebuilt: Dict[int, Optional[Layer]] = {}
        for layer in reversed(order):
            rebuilt[id(layer)] = layer.rebuild(needed, rebuilt)

        return rebuilt[id(self)]

    @abstractmethod
    def marginal(
        self, kept: npt.NDArray, cache: Optional[QueryCache] = None
    ) -> Optional[Layer]:
        """
        Restrict this layer to a subset of the variables.

        :param kept: A boolean mask over the variables of the circuit.
        :param cache: The shared cache of the current pass.
        :return: The marginalized layer, or ``None`` if this layer models none of the
            kept variables.
        """
        raise NotImplementedError

    @abstractmethod
    def remap_variables(self, remap: npt.NDArray, cache: Optional[QueryCache] = None):
        """
        Rewrite the variable indices of this layer in-place.

        :param remap: An array that maps the old variable index to the new one.
        :param cache: The shared cache of the current pass.
        """
        raise NotImplementedError

    def simplify(self, cache: Optional[QueryCache] = None) -> Layer:
        """
        Remove layers that have no effect on the represented distribution.

        This collapses the identity sum and product layers that the structural queries
        introduce. Nested layers of the same type are left alone: merging them would
        have to fuse the parameter blocks of layers with different numbers of nodes.

        :param cache: The shared cache of the current pass.
        :return: The simplified layer.
        """
        return self

    def normalize(self):
        """
        Normalize the weights of every sum layer of the circuit rooted here in-place.
        """
        for layer in self.all_layers():
            layer.normalize_own()

    def normalize_own(self):
        """
        Normalize the parameters stored in this layer alone in-place.
        """

    def is_decomposable(self) -> bool:
        """
        Only a product layer can violate decomposability, so only those are asked.

        :return: Whether every product layer of the circuit rooted here is decomposable.
        """
        # imported here because product_layer imports this module
        from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.product_layer import (
            ProductLayer,
        )

        return all(
            layer.is_decomposable_own()
            for layer in self.all_layers()
            if isinstance(layer, ProductLayer)
        )

    def is_deterministic(self, variables: SortedSet) -> bool:
        """
        Only a sum layer can violate determinism, so only those are asked.

        :param variables: The variables of the circuit.
        :return: Whether every sum layer of the circuit rooted here is deterministic.
        """
        # imported here because sum_layer imports this module
        from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
            SumLayer,
        )

        cache = QueryCache()
        return all(
            layer.is_deterministic_own(variables, cache)
            for layer in self.all_layers()
            if isinstance(layer, SumLayer)
        )

    def apply_translation(self, translation: npt.NDArray):
        """
        Translate the circuit rooted here in-place.

        :param translation: The translation per variable of the circuit.
        """
        for layer in self.all_layers():
            layer.apply_translation_own(translation)

    def apply_translation_own(self, translation: npt.NDArray):
        """
        Translate the parameters of this layer alone in-place.
        """

    def apply_scaling(self, scaling: npt.NDArray):
        """
        Scale the circuit rooted here in-place.

        :param scaling: The scaling per variable of the circuit.
        """
        for layer in self.all_layers():
            layer.apply_scaling_own(scaling)

    def apply_scaling_own(self, scaling: npt.NDArray):
        """
        Scale the parameters of this layer alone in-place.
        """

    # %% conversion

    @classmethod
    @abstractmethod
    def create_layer_from_nodes_with_same_type_and_scope(
        cls,
        nodes: List[Unit],
        child_layers: List[LayerConverter],
        progress_bar: bool = False,
    ) -> LayerConverter:
        """
        Create a layer from units of a rustworkx circuit that share type and scope.

        :param nodes: The units.
        :param child_layers: The converters of the level below.
        :param progress_bar: Whether to show a progress bar.
        :return: The converter of the created layer.
        """
        raise NotImplementedError

    @abstractmethod
    def to_rustworkx(
        self,
        variables: SortedSet,
        result: RustworkxProbabilisticCircuit,
        cache: Optional[QueryCache] = None,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Unit]:
        """
        Create one unit of a rustworkx circuit per node of this layer.

        :param variables: The variables of the circuit.
        :param result: The circuit to write into.
        :param cache: The shared cache of the conversion.
        :param progress_bar: A progress bar to update.
        :return: The created units, in the order of the nodes of this layer.
        """
        raise NotImplementedError

    @abstractmethod
    def __deepcopy__(self, memo=None) -> Layer:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.number_of_nodes})"

    # %% serialization

    @classmethod
    def serialized_fields(cls) -> List[dataclasses.Field]:
        """
        :return: The dataclass fields that describe a layer, which are the ones its
            constructor takes. The caches a layer fills in on its own are declared with
            ``init=False`` and are left out.
        """
        return [field_ for field_ in dataclasses.fields(cls) if field_.init]

    def to_json(self, **kwargs) -> Dict[str, Any]:
        """
        Serialize this layer through the fields it declares.

        The parameter arrays, the child layers and the sparse edge and weight blocks are
        all types that :mod:`krrood.adapters.json_serializer` serializes generically, so
        a layer type is serializable by declaring its fields and none of them has to
        write a method per field.

        :param kwargs: Keyword arguments to hand on to the nested ``to_json`` calls.
        :return: The JSON dict.
        """
        result = super().to_json(**kwargs)
        for field_ in self.serialized_fields():
            result[field_.name] = json_serializer.to_json(
                getattr(self, field_.name), **kwargs
            )
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            **{
                field_.name: json_serializer.from_json(data[field_.name], **kwargs)
                for field_ in cls.serialized_fields()
                if field_.name in data
            }
        )


@dataclass(eq=False, repr=False)
class InnerLayer(Layer[RustworkxUnitType], ABC):
    """
    Abstract base class for the layers that have child layers.
    """

    child_layers: List[Layer]
    """
    The child layers of this layer.

    The list is not copied.
    """

    _variables_cache: Optional[npt.NDArray] = field(
        default=None, init=False, repr=False
    )
    """
    Cached indices of the variables in the scope of this layer.
    """

    def reset_variables(self):
        """
        Drop the cached scope of this layer so that it is recomputed on the next access.
        """
        self._variables_cache = None

    def remap_variables(self, remap: npt.NDArray, cache: Optional[QueryCache] = None):
        if cache is None:
            cache = QueryCache()
        if cache.has(LayerQuery.REMAP_VARIABLES, self):
            return
        cache.set(LayerQuery.REMAP_VARIABLES, self, True)
        for child_layer in self.child_layers:
            child_layer.remap_variables(remap, cache)
        self.reset_variables()

    @abstractmethod
    def iterate_edges(self) -> Iterator[Edge]:
        """
        :return: Yields every edge from a node of this layer to a node of one of its
            child layers.
        """
        raise NotImplementedError


@dataclass
class LayerConverter:
    """
    Bookkeeping for the conversion of a circuit of the ``rx`` package into a layered
    one.
    """

    layer: Layer
    """
    The created layer.
    """

    nodes: List[Unit]
    """
    The units the layer was created from, in the order of its nodes.
    """

    hash_remap: Dict[int, int]
    """
    A map from the hash of a unit to the index of its node in the layer.
    """
