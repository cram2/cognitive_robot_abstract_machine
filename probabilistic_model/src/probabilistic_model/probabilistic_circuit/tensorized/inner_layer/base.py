from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import tqdm
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


def memoized(name: str):
    """
    Memoize a bottom-up query of a layer by the identity of the layer.

    Layers form a directed acyclic graph, not a tree: a layer that is the child of
    several parents must only be evaluated once per query. The wrapped method receives a
    ``cache`` keyword argument that it has to hand down to the calls it makes on its own
    children; the top level caller may omit it.

    :param name: The namespace of this query inside the shared cache.
    :return: The decorator.
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, cache: Optional[Dict] = None, **kwargs):
            if cache is None:
                cache = {}
            key = (name, id(self))
            if key not in cache:
                cache[key] = method(self, *args, cache=cache, **kwargs)
            return cache[key]

        return wrapper

    return decorator


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

    def __getattr__(self, name: str) -> List[Layer]:
        """
        Fall back to no child layers for a layer that never declared any.

        :class:`Layer` is not itself a dataclass and declares no ``child_layers`` field
        or property, so that :class:`InnerLayer` is free to declare it as an ordinary
        required field without a base-class descriptor of the same name blocking that (a
        ``@property`` would, even a getter-only one, since assigning to it in
        ``InnerLayer``'s generated ``__init__`` would then hit its missing setter).
        Ordinary attribute lookup only reaches ``__getattr__`` when nothing set the
        attribute anywhere else, which is exactly the case for a layer without children.

        :param name: The attribute that plain lookup could not find.
        :return: An empty list, for ``child_layers`` only.
        :raises AttributeError: For every other name.
        """
        if name == "child_layers":
            return []
        raise AttributeError(name)

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
    def number_of_own_parameters(self) -> int:
        """
        :return: The number of parameters stored in this layer alone.
        """
        return 0

    def validate(self):
        """
        Check that the parameter arrays of this layer and all its descendants have
        consistent shapes.

        :raises ShapeMismatchError: If a shape is inconsistent.
        """
        for layer in self.all_layers():
            layer.validate_own()

    def validate_own(self):
        """
        Check the shapes of the parameters stored in this layer alone.
        """

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

    def all_layers_with_depth(self, depth: int = 0) -> List[Tuple[int, Layer]]:
        """
        :return: Every layer of the circuit rooted here with its depth. Layers that are
            reachable along several paths appear once per path, mirroring the jax
            implementation.
        """
        result = [(depth, self)]
        for child_layer in self.child_layers:
            result.extend(child_layer.all_layers_with_depth(depth + 1))
        return result

    def topological_layer_order(self) -> List[Layer]:
        """
        Order the layers of the circuit rooted here such that every layer appears after
        all of its parents.

        This is the order in which a top-down pass (such as sampling) has to visit the
        layers so that a layer is only processed once every parent has contributed to
        it.

        :return: The layers in topological order.
        """
        layers = self.all_layers()
        index_of = {id(layer): index for index, layer in enumerate(layers)}

        in_degree = [0] * len(layers)
        for layer in layers:
            for child_layer in layer.child_layers:
                in_degree[index_of[id(child_layer)]] += 1

        queue = [index for index, degree in enumerate(in_degree) if degree == 0]
        result = []
        while queue:
            index = queue.pop()
            result.append(layers[index])
            for child_layer in layers[index].child_layers:
                child_index = index_of[id(child_layer)]
                in_degree[child_index] -= 1
                if in_degree[child_index] == 0:
                    queue.append(child_index)

        return result

    # %% queries

    @abstractmethod
    def log_likelihood_of_nodes(
        self, x: npt.NDArray, cache: Optional[Dict] = None
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
        self, x: npt.NDArray, cache: Optional[Dict] = None
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
        cache: Optional[Dict] = None,
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
        self, variables: SortedSet, cache: Optional[Dict] = None
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
        self, variables: SortedSet, cache: Optional[Dict] = None
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
        cache: Optional[Dict] = None,
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
        cache: Optional[Dict] = None,
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
        cache: Optional[Dict] = None,
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
        cache: Optional[Dict] = None,
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
        :param log_probabilities: The per-layer log-probabilities of the structural pass
            that created this layer.
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
        self, kept: npt.NDArray, cache: Optional[Dict] = None
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
    def remap_variables(self, remap: npt.NDArray, cache: Optional[Dict] = None):
        """
        Rewrite the variable indices of this layer in-place.

        :param remap: An array that maps the old variable index to the new one.
        :param cache: The shared cache of the current pass.
        """
        raise NotImplementedError

    def simplify(self, cache: Optional[Dict] = None) -> Layer:
        """
        Remove layers that have no effect on the represented distribution.

        This collapses the identity sum and product layers that the structural queries
        introduce. Unlike the rustworkx implementation it does not merge nested layers
        of the same type, because in a layered circuit that would have to fuse the
        parameter blocks of layers with different numbers of nodes.

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
        :return: Whether every product layer of the circuit rooted here is decomposable.
        """
        return all(layer.is_decomposable_own() for layer in self.all_layers())

    def is_decomposable_own(self) -> bool:
        """
        :return: Whether this layer alone is decomposable.
        """
        return True

    def is_deterministic(self, variables: SortedSet) -> bool:
        """
        :param variables: The variables of the circuit.
        :return: Whether every sum layer of the circuit rooted here is deterministic.
        """
        cache: Dict = {}
        return all(
            layer.is_deterministic_own(variables, cache) for layer in self.all_layers()
        )

    def is_deterministic_own(self, variables: SortedSet, cache: Dict) -> bool:
        """
        :param variables: The variables of the circuit.
        :param cache: The shared cache of the supports computed so far.
        :return: Whether this layer alone is deterministic.
        """
        return True

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
        cache: Optional[Dict] = None,
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


@dataclass(eq=False, repr=False)
class InnerLayer(Layer[RustworkxUnitType], ABC):
    """
    Abstract base class for the layers that have child layers.
    """

    child_layers: List[Layer]
    """
    The child layers of this layer.
    """

    _variables_cache: Optional[npt.NDArray] = field(
        default=None, init=False, repr=False
    )
    """
    Cached indices of the variables in the scope of this layer.
    """

    def __post_init__(self):
        self.child_layers = list(self.child_layers)

    def reset_variables(self):
        """
        Drop the cached scope of this layer so that it is recomputed on the next access.
        """
        self._variables_cache = None

    def remap_variables(self, remap: npt.NDArray, cache: Optional[Dict] = None):
        if cache is None:
            cache = {}
        if id(self) in cache:
            return
        cache[id(self)] = True
        for child_layer in self.child_layers:
            child_layer.remap_variables(remap, cache)
        self.reset_variables()

    def to_json(self, **kwargs) -> Dict[str, Any]:
        result = super().to_json(**kwargs)
        result["child_layers"] = [
            child_layer.to_json(**kwargs) for child_layer in self.child_layers
        ]
        return result


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
