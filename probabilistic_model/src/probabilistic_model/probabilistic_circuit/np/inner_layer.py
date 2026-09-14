from __future__ import annotations

import functools
import inspect
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import tqdm
from krrood.adapters.json_serializer import SubclassJSONSerializer, recursive_subclasses
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Self,
    Tuple,
    Type,
)

from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.np.utils import (
    SparseArray,
    embedded_logsumexp,
    remap_indices,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RustworkxProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    Unit,
)


class BatchedTruncationUnsupported(Exception):
    """
    Raised when a layer cannot be truncated to several simple events in one pass.

    The circuit catches this and falls back to truncating once per simple event. It is an
    exception rather than a ``None`` return because the decision is made deep inside the
    recursion, by an input layer, and has to abort the whole pass.
    """


def import_layer_modules():
    """
    Import the modules that define the concrete layers.

    The lookup of a layer class walks the subclasses of :class:`Layer`, so the modules
    that define them have to have been imported. Doing that here rather than in the
    package ``__init__`` keeps the import of this module free of cycles: every layer
    module imports from this one.
    """
    from probabilistic_model.probabilistic_circuit.np import discrete_layer  # noqa: F401
    from probabilistic_model.probabilistic_circuit.np import gaussian_layer  # noqa: F401
    from probabilistic_model.probabilistic_circuit.np import input_layer  # noqa: F401
    from probabilistic_model.probabilistic_circuit.np import uniform_layer  # noqa: F401


def layer_class_of(clazz: Type) -> Type[Layer]:
    """
    Find the layer class that corresponds to a class of the rustworkx implementation.

    An exact match wins over an inherited one. That distinction matters because the
    distributions form their own hierarchy: a truncated Gaussian is a Gaussian, so
    matching by ``issubclass`` alone would put it into whichever of the two layers the
    subclass iteration happens to reach first.

    :param clazz: The unit class or the distribution class of a leaf unit.
    :return: The matching layer class.
    """
    import_layer_modules()

    candidates = [
        subclass
        for subclass in recursive_subclasses(Layer)
        if not inspect.isabstract(subclass)
    ]

    for subclass in candidates:
        if clazz in subclass.rustworkx_classes():
            return subclass

    for subclass in candidates:
        if issubclass(clazz, subclass.rustworkx_classes()):
            return subclass

    raise TypeError(f"Could not find a layer class for {clazz}")


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


class Layer(SubclassJSONSerializer, ABC):
    """
    Abstract base class for the layers of a layered probabilistic circuit.

    Every node of a layer has the same scope (set of variables) and, for input layers,
    the same type of distribution. The parameters of all nodes of a layer are stored in
    contiguous arrays, which is what allows every query to be evaluated for all nodes of
    a layer at once.

    Variables are referred to by their index in the ``variables`` of the owning
    :class:`probabilistic_model.probabilistic_circuit.np.probabilistic_circuit.ProbabilisticCircuit`
    rather than by the variable objects themselves.
    """

    # ------------------------------------------------------------------ structure

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
    def child_layers(self) -> List[Layer]:
        """
        :return: The child layers of this layer.
        """
        return []

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
        seen = set()

        def visit(layer: Layer):
            if id(layer) in seen:
                return
            seen.add(id(layer))
            result.append(layer)
            for child_layer in layer.child_layers:
                visit(child_layer)

        visit(self)
        return result

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
        layers so that a layer is only processed once every parent has contributed to it.

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

    # ------------------------------------------------------------------ queries

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
        assignment: Dict[int, List[List[npt.NDArray]]],
        samples: npt.NDArray,
        variables: SortedSet,
    ):
        """
        Route the sample rows that the parents of this layer assigned to its nodes.

        :param assignment: A map from the id of a layer to, per node of that layer, the
            list of row index arrays that were routed to it.
        :param samples: The array the input layers write their samples into.
        :param variables: The variables of the circuit.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ structural

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
    ) -> Tuple[Layer, npt.NDArray]:
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
        :return: The truncated layer and the log-probabilities of its nodes.
        :raises BatchedTruncationUnsupported: If a layer below cannot do this.
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
        :param log_probabilities: The per-layer log-probabilities of the structural pass.
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
        introduce. Unlike the rustworkx implementation it does not merge nested layers of
        the same type, because in a layered circuit that would have to fuse the parameter
        blocks of layers with different numbers of nodes.

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

    # ------------------------------------------------------------------ conversion

    @classmethod
    def rustworkx_classes(cls) -> Tuple[Type, ...]:
        """
        :return: The classes of the ``probabilistic_model.probabilistic_circuit.rx``
            package that this layer represents.
        """
        return tuple()

    @staticmethod
    def create_layers_from_nodes(
        nodes: List[Unit],
        child_layers: List[LayerConverter],
        progress_bar: bool = False,
    ) -> List[LayerConverter]:
        """
        Group a list of units of a rustworkx circuit into layers.

        :param nodes: The units that form one level of the rustworkx circuit.
        :param child_layers: The converters of the level below.
        :param progress_bar: Whether to show a progress bar.
        :return: One converter per created layer.
        """
        result = []

        def type_of(node: Unit) -> Type:
            return type(node.distribution) if node.is_leaf else type(node)

        # grouping is by exact type, not by ``isinstance``: a truncated Gaussian leaf is
        # an instance of the Gaussian distribution and would otherwise be pulled into the
        # Gaussian group, whose layer cannot hold it
        groups: Dict[Tuple[Type, Tuple], List[Unit]] = {}
        for node in nodes:
            groups.setdefault((type_of(node), tuple(node.variables)), []).append(node)

        for (node_type, _), group in groups.items():
            layer_type = layer_class_of(node_type)
            result.append(
                layer_type.create_layer_from_nodes_with_same_type_and_scope(
                    group, child_layers, progress_bar
                )
            )

        return result

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
class InnerLayer(Layer, ABC):
    """
    Abstract base class for the layers that have child layers.

    The field is named ``_child_layers`` rather than ``child_layers`` because
    :class:`Layer` already defines ``child_layers`` as a property (returning ``[]`` for
    layers without children); a dataclass field of the same name would pick that property
    up as its default through inherited attribute lookup, which breaks field ordering in
    every subclass that adds a required field afterwards.
    """

    _child_layers: List[Layer]

    _variables_cache: Optional[npt.NDArray] = field(default=None, init=False, repr=False)
    """
    Cached indices of the variables in the scope of this layer.
    """

    def __post_init__(self):
        self._child_layers = list(self._child_layers)

    @property
    def child_layers(self) -> List[Layer]:
        return self._child_layers

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


@dataclass(eq=False, repr=False)
class SumLayer(InnerLayer, ABC):
    """
    Abstract base class for layers of sum units.

    The weights of all sum units of a layer are grouped per child layer: the ``i``-th
    entry of :attr:`log_weights` holds, for every node of this layer, the logarithmic
    weights of the edges into the ``i``-th child layer. All nodes of a sum layer have the
    same scope, which is the scope of the child layers.
    """

    log_weights: List[Any]

    def __post_init__(self):
        super().__post_init__()
        self.log_weights = list(self.log_weights)

    @property
    def variables(self) -> npt.NDArray:
        if self._variables_cache is None:
            self._variables_cache = self.child_layers[0].variables
        return self._variables_cache

    @property
    def log_weighted_child_layers(self) -> Iterator[Tuple[Any, Layer]]:
        """
        :return: The log-weights and the child layers, zipped together.
        """
        return zip(self.log_weights, self.child_layers)

    @property
    @abstractmethod
    def log_normalization_constants(self) -> npt.NDArray:
        """
        :return: ``log(sum(exp(w)))`` over the weights of each node, shape (#nodes,).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def normalized_weights(self) -> Any:
        """
        :return: The weights of each node in linear space, normalized to sum to one.
        """
        raise NotImplementedError

    def validate_own(self):
        for log_weights in self.log_weights:
            if log_weights.shape[0] != self.number_of_nodes:
                raise ShapeMismatchError(self.number_of_nodes, log_weights.shape[0])

        for log_weights, child_layer in self.log_weighted_child_layers:
            if log_weights.shape[1] != child_layer.number_of_nodes:
                raise ShapeMismatchError(
                    child_layer.number_of_nodes, log_weights.shape[1]
                )

    @classmethod
    def rustworkx_classes(cls) -> Tuple[Type, ...]:
        return (SumUnit,)

    # ------------------------------------------------------------------ queries

    def _weighted_forward(self, child_results: List[npt.NDArray]) -> npt.NDArray:
        """
        Combine the results of the child layers of a linear (non-logarithmic) query whose
        results have the nodes in the last axis.

        :param child_results: The result per child layer, shape (..., #child nodes).
        :return: The result for the nodes of this layer, shape (..., #nodes).
        """
        weights = self.normalized_weights_per_child_layer()
        result = None
        for weight, child_result in zip(weights, child_results):
            contribution = child_result @ weight.T
            result = contribution if result is None else result + contribution
        return result

    def _weighted_forward_over_nodes(
        self, child_results: List[npt.NDArray]
    ) -> npt.NDArray:
        """
        Combine the results of the child layers of a query whose results have the nodes
        in the first axis, such as the moments.

        :param child_results: The result per child layer, shape (#child nodes, ...).
        :return: The result for the nodes of this layer, shape (#nodes, ...).
        """
        weights = self.normalized_weights_per_child_layer()
        result = None
        for weight, child_result in zip(weights, child_results):
            contribution = weight @ child_result
            result = contribution if result is None else result + contribution
        return result

    @abstractmethod
    def normalized_weights_per_child_layer(self) -> List[npt.NDArray]:
        """
        :return: The dense, normalized weight block per child layer with shape
            (#nodes, #nodes of the child layer).
        """
        raise NotImplementedError

    @memoized("log_likelihood")
    def log_likelihood_of_nodes(
        self, x: npt.NDArray, cache: Optional[Dict] = None
    ) -> npt.NDArray:
        child_results = [
            child_layer.log_likelihood_of_nodes(x, cache=cache)
            for child_layer in self.child_layers
        ]
        return self.log_weighted_sum(child_results)

    @abstractmethod
    def log_weighted_sum(self, child_results: List[npt.NDArray]) -> npt.NDArray:
        """
        Reduce the log-results of the child layers with the normalized log-weights.

        :param child_results: The log-results per child layer, with shape
            (..., #nodes of the child layer).
        :return: The log-result of the nodes of this layer with shape (..., #nodes).
        """
        raise NotImplementedError

    @memoized("cumulative_distribution")
    def cumulative_distribution_of_nodes(
        self, x: npt.NDArray, cache: Optional[Dict] = None
    ) -> npt.NDArray:
        child_results = [
            child_layer.cumulative_distribution_of_nodes(x, cache=cache)
            for child_layer in self.child_layers
        ]
        return self._weighted_forward(child_results)

    @memoized("probability_of_simple_event")
    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: Optional[Dict] = None,
    ) -> npt.NDArray:
        child_results = [
            child_layer.probability_of_simple_event_of_nodes(
                event, variables, cache=cache
            ).reshape(1, -1)
            for child_layer in self.child_layers
        ]
        return self._weighted_forward(child_results).reshape(-1)

    @memoized("support")
    def support_of_nodes(
        self, variables: SortedSet, cache: Optional[Dict] = None
    ) -> List[Event]:
        child_supports = [
            child_layer.support_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        result: List[Optional[Event]] = [None] * self.number_of_nodes
        for node, child_layer_index, child_node in self.edges():
            support = child_supports[child_layer_index][child_node]
            if result[node] is None:
                result[node] = support.__deepcopy__()
            else:
                result[node] = result[node] | support.__deepcopy__()

        return [Event() if support is None else support for support in result]

    @memoized("log_mode")
    def log_mode_of_nodes(
        self, variables: SortedSet, cache: Optional[Dict] = None
    ) -> Tuple[List[Event], npt.NDArray]:
        child_modes = [
            child_layer.log_mode_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        log_weights = self.normalized_log_weights_per_child_layer()

        best_value = np.full(self.number_of_nodes, -np.inf)
        candidates: List[List[Event]] = [[] for _ in range(self.number_of_nodes)]

        for node, child_layer_index, child_node in self.edges():
            log_weight = log_weights[child_layer_index][node, child_node]
            value = log_weight + child_modes[child_layer_index][1][child_node]
            mode = child_modes[child_layer_index][0][child_node]
            if value > best_value[node]:
                best_value[node] = value
                candidates[node] = [mode]
            elif value == best_value[node]:
                candidates[node].append(mode)

        modes = []
        for events in candidates:
            if not events:
                modes.append(Event())
                continue
            mode = events[0].__deepcopy__()
            for event in events[1:]:
                mode |= event.__deepcopy__()
            modes.append(mode)

        return modes, best_value

    @memoized("moment")
    def moment_of_nodes(
        self,
        order: npt.NDArray,
        center: npt.NDArray,
        requested: npt.NDArray,
        variables: SortedSet,
        cache: Optional[Dict] = None,
    ) -> npt.NDArray:
        child_results = [
            child_layer.moment_of_nodes(
                order, center, requested, variables, cache=cache
            )
            for child_layer in self.child_layers
        ]
        return self._weighted_forward_over_nodes(child_results)

    def sample_forward(
        self,
        assignment: Dict[int, List[List[npt.NDArray]]],
        samples: npt.NDArray,
        variables: SortedSet,
    ):
        own_assignment = assignment[id(self)]
        weights = self.normalized_weights_per_child_layer()

        # the (child layer, child node) target of every column of the concatenated
        # weight matrix, so that a single multinomial draw partitions the rows
        targets = []
        for child_layer_index, child_layer in enumerate(self.child_layers):
            for child_node in range(child_layer.number_of_nodes):
                targets.append((child_layer_index, child_node))

        concatenated = np.concatenate(weights, axis=1)

        for node, rows_of_node in enumerate(own_assignment):
            if not rows_of_node:
                continue
            rows = np.concatenate(rows_of_node)
            probabilities = concatenated[node]

            # guard against the accumulated floating point error of the normalization
            total = probabilities.sum()
            if total <= 0:
                continue
            probabilities = probabilities / total

            counts = np.random.multinomial(len(rows), pvals=probabilities)

            # shuffle so that the contiguous chunks handed to the children are an
            # unbiased partition of the rows
            np.random.shuffle(rows)

            offset = 0
            for count, (child_layer_index, child_node) in zip(counts, targets):
                if not count:
                    continue
                child_layer = self.child_layers[child_layer_index]
                assignment[id(child_layer)][child_node].append(
                    rows[offset : offset + count]
                )
                offset += count

    # ------------------------------------------------------------------ structural

    @abstractmethod
    def edges(self) -> Iterator[Tuple[int, int, int]]:
        """
        :return: Yields ``(node, child layer index, child node)`` for every edge of this
            layer.
        """
        raise NotImplementedError

    @abstractmethod
    def normalized_log_weights_per_child_layer(self) -> List[npt.NDArray]:
        """
        :return: The dense, normalized log-weight block per child layer.
        """
        raise NotImplementedError

    def is_deterministic_own(self, supports: List[List[Event]]) -> bool:
        """
        Check whether every node of this sum layer is deterministic.

        :param supports: The support of every node of every child layer.
        :return: Whether all nodes are deterministic.
        """
        supports_per_node: List[List[Event]] = [[] for _ in range(self.number_of_nodes)]
        for node, child_layer_index, child_node in self.edges():
            supports_per_node[node].append(supports[child_layer_index][child_node])

        for node_supports in supports_per_node:
            for index, support in enumerate(node_supports):
                for other in node_supports[index + 1 :]:
                    if not support.intersection_with(other).is_empty():
                        return False
        return True


@dataclass(eq=False, repr=False)
class SparseSumLayer(SumLayer):
    """
    A sum layer whose weights are stored sparsely.

    This is the layer that a circuit of the ``rx`` package is converted into: sum units
    there usually have few children, so the dense weight matrix would be mostly empty.
    """

    log_weights: List[SparseArray]

    _edge_gather: Optional[npt.NDArray] = field(default=None, init=False, repr=False)
    """
    Cached index matrix of :attr:`edge_gather`.
    """

    _edges_are_contiguous: bool = field(default=False, init=False, repr=False)
    """
    Whether the edges are stored node by node with the same number of edges per node,
    filled together with :attr:`_edge_gather`.
    """

    _edge_targets: Optional[Tuple[npt.NDArray, npt.NDArray]] = field(
        default=None, init=False, repr=False
    )
    """
    Cached targets of :attr:`edge_targets`.
    """

    @property
    def number_of_nodes(self) -> int:
        return self.log_weights[0].shape[0]

    @property
    def number_of_own_parameters(self) -> int:
        return sum(
            log_weights.number_of_stored_entries for log_weights in self.log_weights
        )

    @property
    def number_of_components(self) -> int:
        return sum(
            child_layer.number_of_components for child_layer in self.child_layers
        ) + sum(
            log_weights.number_of_stored_entries for log_weights in self.log_weights
        )

    @property
    def concatenated_rows(self) -> npt.NDArray:
        """
        :return: The node of every edge, with the child layers concatenated in order.
        """
        return np.concatenate([log_weights.rows for log_weights in self.log_weights])

    @property
    def concatenated_edge_log_weights(self) -> npt.NDArray:
        """
        :return: The weight of every edge, with the child layers concatenated in order.
        """
        return np.concatenate([log_weights.data for log_weights in self.log_weights])

    @property
    def edge_gather(self) -> npt.NDArray:
        """
        The positions of the edges of every node, as a rectangular index matrix of shape
        (#nodes, largest number of edges of a node).

        Rows of nodes with fewer edges are padded with the position one past the last
        edge, which the queries fill with ``-inf``. Gathering with this matrix turns the
        per-node reduction over a ragged set of edges into one reduction over the last
        axis of a rectangular array, which is what keeps the likelihood of a whole batch
        of events a handful of numpy calls instead of a loop over nodes or events.
        """
        if self._edge_gather is None:
            rows = self.concatenated_rows
            number_of_edges = len(rows)
            counts = np.bincount(rows, minlength=self.number_of_nodes)
            width = max(int(counts.max()) if len(counts) else 0, 1)

            gather = np.full(
                (self.number_of_nodes, width), number_of_edges, dtype=np.int64
            )
            # the position of every edge inside the row of its node
            order = np.argsort(rows, kind="stable")
            sorted_rows = rows[order]
            offsets = np.arange(number_of_edges) - np.repeat(
                np.concatenate([[0], np.cumsum(counts)[:-1]]), counts
            )
            gather[sorted_rows, offsets] = order
            self._edge_gather = gather

            # when every node has the same number of edges and the edges are already
            # stored node by node, grouping them is a reshape rather than a gather
            self._edges_are_contiguous = bool(
                number_of_edges == self.number_of_nodes * width
                and np.array_equal(
                    gather, np.arange(number_of_edges).reshape(-1, width)
                )
            )
        return self._edge_gather

    @property
    def edges_are_contiguous(self) -> bool:
        """
        :return: Whether :meth:`group_edges_by_node` can reshape instead of gather.
        """
        self.edge_gather  # fills the flag along with the gather matrix
        return self._edges_are_contiguous

    def group_edges_by_node(
        self, values: npt.NDArray, padding: float = -np.inf
    ) -> npt.NDArray:
        """
        Rearrange per-edge values into one row per node.

        :param values: Per-edge values with the edges in the last axis.
        :param padding: The value for nodes with fewer edges than the widest one.
        :return: The values with shape ``(..., #nodes, edges per node)``.
        """
        if self.edges_are_contiguous:
            return values.reshape(values.shape[:-1] + (self.number_of_nodes, -1))
        return self.pad_edges(values, padding)[..., self.edge_gather]

    def pad_edges(self, values: npt.NDArray, padding: float) -> npt.NDArray:
        """
        Append the slot that :attr:`edge_gather` pads with.

        :param values: Per-edge values with the edges in the last axis.
        :param padding: The value of the padding slot. ``-inf`` is neutral for a
            logarithmic reduction, ``0`` for a linear one.
        :return: The values with one extra entry in the last axis.
        """
        return np.concatenate(
            [values, np.full(values.shape[:-1] + (1,), padding)], axis=-1
        )

    @property
    def log_normalization_constants(self) -> npt.NDArray:
        gathered = self.group_edges_by_node(self.concatenated_edge_log_weights)
        return embedded_logsumexp(gathered, axis=-1)

    @property
    def normalized_edge_weights(self) -> npt.NDArray:
        """
        :return: The weight of every edge in linear space, normalized per node.
        """
        normalization = self.log_normalization_constants
        rows = self.concatenated_rows
        shifted = self.concatenated_edge_log_weights - normalization[rows]
        # a node whose weights are all -inf normalizes to nan; it is impossible, and the
        # prune pass removes it, so its weights are simply zero here
        return np.where(np.isfinite(shifted), np.exp(shifted), 0.0)

    @property
    def edge_targets(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        :return: The index of the child layer and the index of the node in it that every
            edge points to, in the order of the concatenated edges.
        """
        if self._edge_targets is None:
            self._edge_targets = (
                np.concatenate(
                    [
                        np.full(log_weights.number_of_stored_entries, index, np.int64)
                        for index, log_weights in enumerate(self.log_weights)
                    ]
                ),
                np.concatenate(
                    [log_weights.columns for log_weights in self.log_weights]
                ),
            )
        return self._edge_targets

    def _weighted_forward(self, child_results: List[npt.NDArray]) -> npt.NDArray:
        # the dense weight block of a layer with many nodes is mostly empty and can be
        # far larger than the circuit itself, so the linear queries reduce over the
        # stored edges instead of building it
        values = np.concatenate(
            [
                child_result[..., log_weights.columns]
                for log_weights, child_result in zip(self.log_weights, child_results)
            ],
            axis=-1,
        )
        values = values * self.normalized_edge_weights
        return self.group_edges_by_node(values, padding=0.0).sum(axis=-1)

    def _weighted_forward_over_nodes(
        self, child_results: List[npt.NDArray]
    ) -> npt.NDArray:
        values = np.concatenate(
            [
                child_result[log_weights.columns]
                for log_weights, child_result in zip(self.log_weights, child_results)
            ],
            axis=0,
        )
        values = values * self.normalized_edge_weights[:, None]
        padded = np.concatenate([values, np.zeros((1, values.shape[1]))], axis=0)
        return padded[self.edge_gather].sum(axis=1)

    def sample_forward(
        self,
        assignment: Dict[int, List[List[npt.NDArray]]],
        samples: npt.NDArray,
        variables: SortedSet,
    ):
        own_assignment = assignment[id(self)]
        gather = self.edge_gather
        # the padding slot gets a weight of zero, so it is never drawn
        weights = np.append(self.normalized_edge_weights, 0.0)
        child_layer_of_edge, child_node_of_edge = self.edge_targets

        for node, rows_of_node in enumerate(own_assignment):
            if not rows_of_node:
                continue
            rows = np.concatenate(rows_of_node)

            positions = gather[node]
            probabilities = weights[positions]

            # guard against the accumulated floating point error of the normalization
            total = probabilities.sum()
            if total <= 0:
                continue
            counts = np.random.multinomial(len(rows), pvals=probabilities / total)

            # shuffle so that the contiguous chunks handed to the children are an
            # unbiased partition of the rows
            np.random.shuffle(rows)

            offset = 0
            for count, position in zip(counts, positions):
                if not count:
                    continue
                child_layer = self.child_layers[child_layer_of_edge[position]]
                assignment[id(child_layer)][child_node_of_edge[position]].append(
                    rows[offset : offset + count]
                )
                offset += count

    @property
    def normalized_weights(self) -> List[SparseArray]:
        normalization = self.log_normalization_constants
        result = []
        for log_weights in self.log_weights:
            normalized = log_weights.copy()
            with np.errstate(invalid="ignore"):
                normalized.data = np.exp(
                    normalized.data - normalization[normalized.rows]
                )
            normalized.data = np.nan_to_num(normalized.data, nan=0.0)
            result.append(normalized)
        return result

    def normalized_weights_per_child_layer(self) -> List[npt.NDArray]:
        return [weights.to_dense(0.0) for weights in self.normalized_weights]

    def normalized_log_weights_per_child_layer(self) -> List[npt.NDArray]:
        normalization = self.log_normalization_constants
        result = []
        for log_weights in self.log_weights:
            normalized = log_weights.copy()
            normalized.data = normalized.data - normalization[normalized.rows]
            result.append(normalized.to_dense(-np.inf))
        return result

    def edges(self) -> Iterator[Tuple[int, int, int]]:
        for child_layer_index, log_weights in enumerate(self.log_weights):
            for node, child_node in log_weights.indices:
                yield int(node), child_layer_index, int(child_node)

    def weighted_child_values(
        self, log_weights: SparseArray, child_result: npt.NDArray
    ) -> npt.NDArray:
        """
        Take the value of the child node of every edge and add the weight of that edge.

        :param log_weights: The weights of the edges into one child layer.
        :param child_result: The result of that child layer, child nodes last.
        :return: One value per edge, the edges last.
        """
        columns = log_weights.columns
        # a sum layer usually points at every node of its child layer exactly once and in
        # order, in which case the gather is an identity copy of an array that has one
        # entry per event per node, and skipping it is worth the comparison
        if len(columns) == child_result.shape[-1] and np.array_equal(
            columns, np.arange(len(columns))
        ):
            return child_result + log_weights.data
        return child_result[..., columns] + log_weights.data

    def log_weighted_sum(self, child_results: List[npt.NDArray]) -> npt.NDArray:
        values = np.concatenate(
            [
                self.weighted_child_values(log_weights, child_result)
                for log_weights, child_result in zip(self.log_weights, child_results)
            ],
            axis=-1,
        )
        gathered = self.group_edges_by_node(values)
        return embedded_logsumexp(gathered, axis=-1) - self.log_normalization_constants

    def normalize_own(self):
        normalization = self.log_normalization_constants
        for log_weights in self.log_weights:
            log_weights.data = log_weights.data - normalization[log_weights.rows]

    def __deepcopy__(self, memo=None) -> SparseSumLayer:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        child_layers = [
            child_layer.__deepcopy__(memo) for child_layer in self.child_layers
        ]
        result = self.__class__(
            child_layers, [log_weights.copy() for log_weights in self.log_weights]
        )
        memo[id(self)] = result
        return result

    def to_json(self, **kwargs) -> Dict[str, Any]:
        result = super().to_json(**kwargs)
        result["log_weights"] = [
            log_weights.to_json() for log_weights in self.log_weights
        ]
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        child_layers = [
            Layer.from_json(child_layer, **kwargs)
            for child_layer in data["child_layers"]
        ]
        log_weights = [
            SparseArray.from_json(log_weights) for log_weights in data["log_weights"]
        ]
        return cls(child_layers, log_weights)

    # ------------------------------------------------------------------ structural

    def _structural_pass(
        self,
        child_results: List[Tuple[Layer, npt.NDArray]],
        log_probabilities: Dict[int, npt.NDArray],
    ) -> Tuple[Layer, npt.NDArray]:
        """
        Update the weights of this layer with the log-probabilities of its children.

        This is the layered equivalent of ``SumUnit.log_forward_conditioning``: the new
        weight of an edge is its old weight times the probability of the event under the
        child, and the probability of a node is the sum of its new weights.

        :param child_results: The new child layer and its node log-probabilities.
        :param log_probabilities: The map to record the result in.
        :return: The new layer and the log-probabilities of its nodes.
        """
        new_log_weights = []
        for log_weights, (_, child_log_probabilities) in zip(
            self.log_weights, child_results
        ):
            updated = log_weights.copy()
            updated.data = updated.data + child_log_probabilities[updated.columns]
            new_log_weights.append(updated)

        result = self.__class__(
            [child_layer for child_layer, _ in child_results], new_log_weights
        )
        # the probability of a node is the sum of its updated weights
        own_log_probabilities = result.log_normalization_constants
        log_probabilities[id(result)] = own_log_probabilities
        return result, own_log_probabilities

    def log_truncated_of_simple_event(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        singleton_allowed: bool,
        cache: Optional[Dict] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        if cache is None:
            cache = {}
        key = ("truncated", id(self))
        if key in cache:
            return cache[key]

        child_results = [
            child_layer.log_truncated_of_simple_event(
                event,
                variables,
                singleton_allowed,
                cache=cache,
                log_probabilities=log_probabilities,
            )
            for child_layer in self.child_layers
        ]
        result = self._structural_pass(child_results, log_probabilities)
        cache[key] = result
        return result

    def log_truncated_of_simple_events(
        self,
        events: List[SimpleEvent],
        variables: SortedSet,
        singleton_allowed: bool,
        cache: Optional[Dict] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        if cache is None:
            cache = {}
        key = ("batched truncated", id(self))
        if key in cache:
            return cache[key]

        number_of_events = len(events)
        number_of_nodes = self.number_of_nodes

        new_child_layers = []
        new_log_weights = []
        for log_weights, child_layer in self.log_weighted_child_layers:
            new_child_layer, child_log_probabilities = (
                child_layer.log_truncated_of_simple_events(
                    events,
                    variables,
                    singleton_allowed,
                    cache=cache,
                    log_probabilities=log_probabilities,
                )
            )
            new_child_layers.append(new_child_layer)

            # the block of event k is the original sparsity pattern shifted into its own
            # rows and columns
            number_of_entries = log_weights.number_of_stored_entries
            blocks = np.arange(number_of_events)
            rows = np.tile(log_weights.rows, number_of_events) + np.repeat(
                blocks * number_of_nodes, number_of_entries
            )
            columns = np.tile(log_weights.columns, number_of_events) + np.repeat(
                blocks * child_layer.number_of_nodes, number_of_entries
            )
            # the weight of an edge times the probability of the event under its child
            data = (
                np.tile(log_weights.data, number_of_events)
                + child_log_probabilities[columns]
            )

            new_log_weights.append(
                SparseArray.from_coordinates(
                    rows,
                    columns,
                    data,
                    (
                        number_of_events * number_of_nodes,
                        number_of_events * child_layer.number_of_nodes,
                    ),
                )
            )

        result = self.__class__(new_child_layers, new_log_weights)
        own_log_probabilities = result.log_normalization_constants
        log_probabilities[id(result)] = own_log_probabilities

        cache[key] = (result, own_log_probabilities)
        return cache[key]

    def log_conditional_of_point(
        self,
        point: Dict[Variable, Any],
        variables: SortedSet,
        cache: Optional[Dict] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        if cache is None:
            cache = {}
        key = ("conditional", id(self))
        if key in cache:
            return cache[key]

        child_results = [
            child_layer.log_conditional_of_point(
                point,
                variables,
                cache=cache,
                log_probabilities=log_probabilities,
            )
            for child_layer in self.child_layers
        ]
        result = self._structural_pass(child_results, log_probabilities)
        cache[key] = result
        return result

    def live_entries(
        self,
        alive: npt.NDArray,
        log_weights: SparseArray,
        child_layer: Layer,
        log_probabilities: Dict[int, npt.NDArray],
    ) -> npt.NDArray:
        """
        Determine the edges into one child layer that survive a prune.

        :param alive: The live nodes of this layer.
        :param log_weights: The weights of the edges into the child layer.
        :param child_layer: The child layer.
        :param log_probabilities: The per-layer log-probabilities of the structural pass.
        :return: A boolean mask over the stored weight entries.
        """
        mask = alive[log_weights.rows] & (log_weights.data > -np.inf)
        child_log_probabilities = log_probabilities.get(id(child_layer))
        if child_log_probabilities is not None:
            mask = mask & (child_log_probabilities[log_weights.columns] > -np.inf)
        return mask

    def required_child_nodes(
        self, alive: npt.NDArray, log_probabilities: Dict[int, npt.NDArray]
    ) -> List[Tuple[Layer, npt.NDArray]]:
        result = []
        for log_weights, child_layer in self.log_weighted_child_layers:
            mask = self.live_entries(
                alive, log_weights, child_layer, log_probabilities
            )
            needed = np.zeros(child_layer.number_of_nodes, dtype=bool)
            needed[log_weights.columns[mask]] = True
            result.append((child_layer, needed))
        return result

    def rebuild(
        self,
        needed: Dict[int, npt.NDArray],
        rebuilt: Dict[int, Optional[Layer]],
    ) -> Optional[Layer]:
        alive = needed[id(self)]
        if not alive.any():
            return None

        node_remap, number_of_nodes = remap_indices(alive)

        new_child_layers = []
        new_log_weights = []
        for log_weights, child_layer in self.log_weighted_child_layers:
            pruned_child = rebuilt.get(id(child_layer))
            if pruned_child is None:
                continue
            child_needed = needed[id(child_layer)]
            mask = (
                alive[log_weights.rows]
                & (log_weights.data > -np.inf)
                & child_needed[log_weights.columns]
            )
            if not mask.any():
                continue
            child_remap, number_of_child_nodes = remap_indices(child_needed)
            new_child_layers.append(pruned_child)
            new_log_weights.append(
                SparseArray.from_coordinates(
                    node_remap[log_weights.rows[mask]],
                    child_remap[log_weights.columns[mask]],
                    log_weights.data[mask],
                    (number_of_nodes, number_of_child_nodes),
                )
            )

        if not new_child_layers:
            return None

        return self.__class__(new_child_layers, new_log_weights)

    def marginal(
        self, kept: npt.NDArray, cache: Optional[Dict] = None
    ) -> Optional[Layer]:
        if cache is None:
            cache = {}
        key = ("marginal", id(self))
        if key in cache:
            return cache[key]

        new_child_layers = []
        new_log_weights = []
        for log_weights, child_layer in self.log_weighted_child_layers:
            marginal_child = child_layer.marginal(kept, cache)
            if marginal_child is None:
                continue
            new_child_layers.append(marginal_child)
            new_log_weights.append(log_weights.copy())

        result = (
            None
            if not new_child_layers
            else self.__class__(new_child_layers, new_log_weights)
        )
        cache[key] = result
        return result

    def simplify(self, cache: Optional[Dict] = None) -> Layer:
        if cache is None:
            cache = {}
        key = ("simplify", id(self))
        if key in cache:
            return cache[key]

        # placed before the recursion so that a cycle-free DAG with shared layers
        # resolves to the same object for every parent
        simplified_children = [
            child_layer.simplify(cache) for child_layer in self.child_layers
        ]
        result = self.__class__(
            simplified_children,
            [log_weights.copy() for log_weights in self.log_weights],
        )

        if result.is_identity():
            result = simplified_children[0]

        cache[key] = result
        return result

    def is_identity(self) -> bool:
        """
        :return: Whether this layer passes its single child layer through unchanged, so
            that it can be removed without changing the distribution.
        """
        if len(self.log_weights) != 1:
            return False
        log_weights = self.log_weights[0]
        if log_weights.shape[0] != log_weights.shape[1]:
            return False
        if log_weights.number_of_stored_entries != self.number_of_nodes:
            return False
        sorted_weights = log_weights.sort_indices()
        expected = np.arange(self.number_of_nodes)
        return bool(
            np.array_equal(sorted_weights.rows, expected)
            and np.array_equal(sorted_weights.columns, expected)
        )

    # ------------------------------------------------------------------ conversion

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(
        cls,
        nodes: List[SumUnit],
        child_layers: List[LayerConverter],
        progress_bar: bool = False,
    ) -> LayerConverter:
        hash_remap = {hash(node): index for index, node in enumerate(nodes)}
        variables = np.array(
            [
                nodes[0].probabilistic_circuit.variables.index(variable)
                for variable in nodes[0].variables
            ]
        )

        # only the child layers with the same scope can be children of these sum units
        filtered_child_layers = [
            child_layer
            for child_layer in child_layers
            if np.array_equal(child_layer.layer.variables, variables)
        ]

        used_child_layers = []
        log_weights = []
        for child_layer in filtered_child_layers:
            rows, columns, values = [], [], []
            for index, node in enumerate(
                tqdm.tqdm(nodes, desc="Assembling sum layer") if progress_bar else nodes
            ):
                for log_weight, subcircuit in node.log_weighted_subcircuits:
                    if hash(subcircuit) in child_layer.hash_remap:
                        rows.append(index)
                        columns.append(child_layer.hash_remap[hash(subcircuit)])
                        values.append(log_weight)

            # a candidate that none of these nodes points to is not a child layer
            if not rows:
                continue

            used_child_layers.append(child_layer.layer)
            log_weights.append(
                SparseArray.from_coordinates(
                    np.array(rows, dtype=np.int64),
                    np.array(columns, dtype=np.int64),
                    np.array(values, dtype=float),
                    (len(nodes), child_layer.layer.number_of_nodes),
                )
            )

        layer = cls(used_child_layers, log_weights)
        return LayerConverter(layer, nodes, hash_remap)

    def to_rustworkx(
        self,
        variables: SortedSet,
        result: RustworkxProbabilisticCircuit,
        cache: Optional[Dict] = None,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Unit]:
        if cache is None:
            cache = {}
        if id(self) in cache:
            return cache[id(self)]

        if progress_bar:
            progress_bar.set_postfix_str(
                f"Parsing sum layer of {[variables[i] for i in self.variables]}"
            )

        units = [
            SumUnit(probabilistic_circuit=result) for _ in range(self.number_of_nodes)
        ]
        child_units = [
            child_layer.to_rustworkx(variables, result, cache, progress_bar)
            for child_layer in self.child_layers
        ]

        for log_weights, child_layer_units in zip(self.log_weights, child_units):
            for (node, child_node), log_weight in zip(
                log_weights.indices, log_weights.data
            ):
                units[node].add_subcircuit(
                    child_layer_units[child_node], float(log_weight)
                )
                if progress_bar:
                    progress_bar.update()

        for unit in units:
            unit.normalize()

        cache[id(self)] = units
        return units


@dataclass(eq=False, repr=False)
class DenseSumLayer(SumLayer):
    """
    A sum layer whose weights are stored densely.

    This is the layout of the sum layers of a randomly initialized region graph, where
    every node is connected to every node of its child layers.
    """

    log_weights: List[npt.NDArray]

    def __post_init__(self):
        super().__post_init__()
        self.log_weights = [np.asarray(w, dtype=float) for w in self.log_weights]

    @property
    def number_of_nodes(self) -> int:
        return self.log_weights[0].shape[0]

    @property
    def number_of_own_parameters(self) -> int:
        return sum(math.prod(log_weights.shape) for log_weights in self.log_weights)

    @property
    def number_of_components(self) -> int:
        return sum(
            child_layer.number_of_components for child_layer in self.child_layers
        ) + sum(math.prod(log_weights.shape) for log_weights in self.log_weights)

    @property
    def concatenated_log_weights(self) -> npt.NDArray:
        """
        :return: The log-weights of all child layers side by side.
        """
        return np.concatenate(self.log_weights, axis=1)

    @property
    def log_normalization_constants(self) -> npt.NDArray:
        return embedded_logsumexp(self.concatenated_log_weights, axis=1)

    @property
    def normalized_weights(self) -> npt.NDArray:
        with np.errstate(invalid="ignore"):
            result = np.exp(
                self.concatenated_log_weights
                - self.log_normalization_constants.reshape(-1, 1)
            )
        return np.nan_to_num(result, nan=0.0)

    def normalized_weights_per_child_layer(self) -> List[npt.NDArray]:
        normalization = self.log_normalization_constants.reshape(-1, 1)
        result = []
        for log_weights in self.log_weights:
            with np.errstate(invalid="ignore"):
                weights = np.exp(log_weights - normalization)
            result.append(np.nan_to_num(weights, nan=0.0))
        return result

    def normalized_log_weights_per_child_layer(self) -> List[npt.NDArray]:
        normalization = self.log_normalization_constants.reshape(-1, 1)
        return [log_weights - normalization for log_weights in self.log_weights]

    def edges(self) -> Iterator[Tuple[int, int, int]]:
        for child_layer_index, log_weights in enumerate(self.log_weights):
            for node in range(log_weights.shape[0]):
                for child_node in range(log_weights.shape[1]):
                    yield node, child_layer_index, child_node

    def log_weighted_sum(self, child_results: List[npt.NDArray]) -> npt.NDArray:
        normalization = self.log_normalization_constants
        result = None
        for log_weights, child_result in zip(self.log_weights, child_results):
            # (..., 1, #child nodes) + (#nodes, #child nodes)
            combined = child_result[..., None, :] + log_weights
            contribution = embedded_logsumexp(combined, axis=-1)
            result = (
                contribution
                if result is None
                else np.logaddexp(result, contribution)
            )
        return result - normalization

    def normalize_own(self):
        normalization = self.log_normalization_constants.reshape(-1, 1)
        self.log_weights = [
            log_weights - normalization for log_weights in self.log_weights
        ]

    def __deepcopy__(self, memo=None) -> DenseSumLayer:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        child_layers = [
            child_layer.__deepcopy__(memo) for child_layer in self.child_layers
        ]
        result = self.__class__(
            child_layers, [log_weights.copy() for log_weights in self.log_weights]
        )
        memo[id(self)] = result
        return result

    def to_json(self, **kwargs) -> Dict[str, Any]:
        result = super().to_json(**kwargs)
        result["log_weights"] = [
            log_weights.tolist() for log_weights in self.log_weights
        ]
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        child_layers = [
            Layer.from_json(child_layer, **kwargs)
            for child_layer in data["child_layers"]
        ]
        log_weights = [np.asarray(w, dtype=float) for w in data["log_weights"]]
        return cls(child_layers, log_weights)

    def as_sparse(self) -> SparseSumLayer:
        """
        :return: An equivalent sum layer with sparsely stored weights.
        """
        return SparseSumLayer(
            list(self.child_layers),
            [
                SparseArray.from_dense(log_weights, fill_value=-np.inf)
                for log_weights in self.log_weights
            ],
        )

    # the structural queries are the same for both weight layouts; converting to the
    # sparse layout keeps them in one place and produces a circuit that is sparse
    # afterwards anyway, because truncation removes edges.
    #
    # as_sparse() allocates a new SparseSumLayer on every call, so its own internal
    # cache (keyed by the id of that fresh object) never gets a hit across separate
    # calls on this same DenseSumLayer. A shared DenseSumLayer reached via several
    # parents would otherwise redo the conversion and recompute its own contribution
    # once per parent -- and, down a chain of shared dense layers, that compounds into
    # one recomputation per path instead of one per node. Caching under id(self) here,
    # with the same key scheme the sparse layers use, restores the "evaluate a shared
    # layer once" guarantee for the dense layout too.
    def log_truncated_of_simple_event(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        singleton_allowed: bool,
        cache: Optional[Dict] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        if cache is None:
            cache = {}
        key = ("truncated", id(self))
        if key in cache:
            return cache[key]
        result = self.as_sparse().log_truncated_of_simple_event(
            event,
            variables,
            singleton_allowed,
            cache=cache,
            log_probabilities=log_probabilities,
        )
        cache[key] = result
        return result

    def log_truncated_of_simple_events(
        self,
        events: List[SimpleEvent],
        variables: SortedSet,
        singleton_allowed: bool,
        cache: Optional[Dict] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        if cache is None:
            cache = {}
        key = ("batched truncated", id(self))
        if key in cache:
            return cache[key]
        result = self.as_sparse().log_truncated_of_simple_events(
            events,
            variables,
            singleton_allowed,
            cache=cache,
            log_probabilities=log_probabilities,
        )
        cache[key] = result
        return result

    def log_conditional_of_point(
        self,
        point: Dict[Variable, Any],
        variables: SortedSet,
        cache: Optional[Dict] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        if cache is None:
            cache = {}
        key = ("conditional", id(self))
        if key in cache:
            return cache[key]
        result = self.as_sparse().log_conditional_of_point(
            point, variables, cache=cache, log_probabilities=log_probabilities
        )
        cache[key] = result
        return result

    def required_child_nodes(
        self, alive: npt.NDArray, log_probabilities: Dict[int, npt.NDArray]
    ) -> List[Tuple[Layer, npt.NDArray]]:
        result = []
        for log_weights, child_layer in self.log_weighted_child_layers:
            needed = (alive[:, None] & (log_weights > -np.inf)).any(axis=0)
            child_log_probabilities = log_probabilities.get(id(child_layer))
            if child_log_probabilities is not None:
                needed = needed & (child_log_probabilities > -np.inf)
            result.append((child_layer, needed))
        return result

    def rebuild(
        self,
        needed: Dict[int, npt.NDArray],
        rebuilt: Dict[int, Optional[Layer]],
    ) -> Optional[Layer]:
        alive = needed[id(self)]
        if not alive.any():
            return None

        new_child_layers = []
        new_log_weights = []
        for log_weights, child_layer in self.log_weighted_child_layers:
            pruned_child = rebuilt.get(id(child_layer))
            if pruned_child is None:
                continue
            child_needed = needed[id(child_layer)]
            new_child_layers.append(pruned_child)
            new_log_weights.append(log_weights[np.ix_(alive, child_needed)])

        if not new_child_layers:
            return None

        return self.__class__(new_child_layers, new_log_weights)

    def marginal(
        self, kept: npt.NDArray, cache: Optional[Dict] = None
    ) -> Optional[Layer]:
        if cache is None:
            cache = {}
        key = ("marginal", id(self))
        if key in cache:
            return cache[key]

        new_child_layers = []
        new_log_weights = []
        for log_weights, child_layer in self.log_weighted_child_layers:
            marginal_child = child_layer.marginal(kept, cache)
            if marginal_child is None:
                continue
            new_child_layers.append(marginal_child)
            new_log_weights.append(log_weights.copy())

        result = (
            None
            if not new_child_layers
            else self.__class__(new_child_layers, new_log_weights)
        )
        cache[key] = result
        return result

    def simplify(self, cache: Optional[Dict] = None) -> Layer:
        if cache is None:
            cache = {}
        key = ("simplify", id(self))
        if key in cache:
            return cache[key]
        result = self.__class__(
            [child_layer.simplify(cache) for child_layer in self.child_layers],
            [log_weights.copy() for log_weights in self.log_weights],
        )
        cache[key] = result
        return result

    @classmethod
    def rustworkx_classes(cls) -> Tuple[Type, ...]:
        # a rustworkx circuit is always converted into the sparse variant
        return tuple()

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(
        cls,
        nodes: List[Unit],
        child_layers: List[LayerConverter],
        progress_bar: bool = False,
    ) -> LayerConverter:
        raise NotImplementedError(
            "Dense sum layers are not created from rustworkx circuits. "
            "Use SparseSumLayer instead."
        )

    def to_rustworkx(
        self,
        variables: SortedSet,
        result: RustworkxProbabilisticCircuit,
        cache: Optional[Dict] = None,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Unit]:
        if cache is None:
            cache = {}
        if id(self) in cache:
            return cache[id(self)]
        units = self.as_sparse().to_rustworkx(variables, result, cache, progress_bar)
        cache[id(self)] = units
        return units


@dataclass(eq=False, repr=False)
class ProductLayer(InnerLayer):
    """
    A layer of decomposable product units.

    The edges are stored as a sparse integer matrix of shape (#child layers, #nodes).
    The value of the entry ``(l, n)`` is the index of the node in the ``l``-th child
    layer that the ``n``-th node of this layer multiplies. A node of a child layer may be
    referenced by several nodes of this layer.
    """

    edges: SparseArray

    @property
    def number_of_nodes(self) -> int:
        return self.edges.shape[1]

    @property
    def number_of_components(self) -> int:
        return (
            sum(child_layer.number_of_components for child_layer in self.child_layers)
            + self.edges.number_of_stored_entries
        )

    @property
    def variables(self) -> npt.NDArray:
        if self._variables_cache is None:
            self._variables_cache = np.unique(
                np.concatenate(
                    [child_layer.variables for child_layer in self.child_layers]
                )
            )
        return self._variables_cache

    def validate_own(self):
        if self.edges.shape != (len(self.child_layers), self.number_of_nodes):
            raise ShapeMismatchError(
                (len(self.child_layers), self.number_of_nodes), self.edges.shape
            )

    @classmethod
    def rustworkx_classes(cls) -> Tuple[Type, ...]:
        return (ProductUnit,)

    def is_decomposable_own(self) -> bool:
        seen = set()
        for child_layer in self.child_layers:
            variables = set(child_layer.variables.tolist())
            if seen & variables:
                return False
            seen |= variables
        return True

    # ------------------------------------------------------------------ queries

    def edges_of_child_layer(
        self, child_layer_index: int
    ) -> Tuple[npt.NDArray, npt.NDArray, bool]:
        """
        The edges into one child layer.

        :param child_layer_index: The index of the child layer.
        :return: The nodes of this layer, the nodes of the child layer they point to, and
            whether every node appears at most once. A decomposable product has at most
            one factor in each child layer, so the fast path is the normal one; the check
            keeps the reduction correct for a circuit that is not decomposable.
        """
        mask = self.edges.rows == child_layer_index
        nodes = self.edges.columns[mask]
        child_nodes = self.edges.data[mask].astype(np.int64)
        unique = len(np.unique(nodes)) == len(nodes)
        return nodes, child_nodes, unique

    def _gather_and_add(
        self, child_results: List[npt.NDArray], fill: float
    ) -> npt.NDArray:
        """
        Sum, per node, the results of the child nodes the edges point to.

        :param child_results: The result per child layer with the child nodes last.
        :param fill: The value of a node without any edge.
        :return: The summed result with the nodes of this layer last.
        """
        leading_shape = child_results[0].shape[:-1]
        result = np.zeros(leading_shape + (self.number_of_nodes,))
        touched = np.zeros(self.number_of_nodes, dtype=bool)

        for child_layer_index, child_result in enumerate(child_results):
            nodes, child_nodes, unique = self.edges_of_child_layer(child_layer_index)
            if len(nodes) == 0:
                continue
            gathered = child_result[..., child_nodes]
            if unique:
                result[..., nodes] += gathered
            else:
                np.add.at(result, (Ellipsis, nodes), gathered)
            touched[nodes] = True

        if not touched.all():
            result[..., ~touched] = fill
        return result

    @memoized("log_likelihood")
    def log_likelihood_of_nodes(
        self, x: npt.NDArray, cache: Optional[Dict] = None
    ) -> npt.NDArray:
        child_results = [
            child_layer.log_likelihood_of_nodes(x, cache=cache)
            for child_layer in self.child_layers
        ]
        return self._gather_and_add(child_results, fill=0.0)

    @memoized("cumulative_distribution")
    def cumulative_distribution_of_nodes(
        self, x: npt.NDArray, cache: Optional[Dict] = None
    ) -> npt.NDArray:
        child_results = [
            child_layer.cumulative_distribution_of_nodes(x, cache=cache)
            for child_layer in self.child_layers
        ]
        return self._gather_and_multiply(child_results)

    def _gather_and_multiply(self, child_results: List[npt.NDArray]) -> npt.NDArray:
        leading_shape = child_results[0].shape[:-1]
        result = np.ones(leading_shape + (self.number_of_nodes,))
        for child_layer_index, child_result in enumerate(child_results):
            nodes, child_nodes, unique = self.edges_of_child_layer(child_layer_index)
            if len(nodes) == 0:
                continue
            gathered = child_result[..., child_nodes]
            if unique:
                result[..., nodes] *= gathered
            else:
                np.multiply.at(result, (Ellipsis, nodes), gathered)
        return result

    @memoized("probability_of_simple_event")
    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: Optional[Dict] = None,
    ) -> npt.NDArray:
        child_results = [
            child_layer.probability_of_simple_event_of_nodes(
                event, variables, cache=cache
            ).reshape(1, -1)
            for child_layer in self.child_layers
        ]
        return self._gather_and_multiply(child_results).reshape(-1)

    @memoized("support")
    def support_of_nodes(
        self, variables: SortedSet, cache: Optional[Dict] = None
    ) -> List[Event]:
        child_supports = [
            child_layer.support_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        own_variables = {variables[index] for index in self.variables}
        result: List[Optional[Event]] = [None] * self.number_of_nodes

        for child_layer_index, node, child_node in self._edge_triples():
            support = child_supports[child_layer_index][child_node].__deepcopy__()
            if result[node] is None:
                support.fill_missing_variables(own_variables)
                result[node] = support
            else:
                result[node] = result[node] & support

        return [Event() if support is None else support for support in result]

    @memoized("log_mode")
    def log_mode_of_nodes(
        self, variables: SortedSet, cache: Optional[Dict] = None
    ) -> Tuple[List[Event], npt.NDArray]:
        child_modes = [
            child_layer.log_mode_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        own_variables = {variables[index] for index in self.variables}
        events: List[Optional[Event]] = [None] * self.number_of_nodes
        values = np.zeros(self.number_of_nodes)

        for child_layer_index, node, child_node in self._edge_triples():
            child_event = child_modes[child_layer_index][0][child_node].__deepcopy__()
            values[node] += child_modes[child_layer_index][1][child_node]
            if events[node] is None:
                child_event.fill_missing_variables(own_variables)
                events[node] = child_event
            else:
                events[node] = events[node].intersection_with(child_event)

        return [Event() if event is None else event for event in events], values

    def _edge_triples(self) -> Iterator[Tuple[int, int, int]]:
        """
        :return: Yields ``(child layer index, node, child node)`` for every edge.
        """
        for (child_layer_index, node), child_node in zip(
            self.edges.indices, self.edges.data
        ):
            yield int(child_layer_index), int(node), int(child_node)

    @memoized("moment")
    def moment_of_nodes(
        self,
        order: npt.NDArray,
        center: npt.NDArray,
        requested: npt.NDArray,
        variables: SortedSet,
        cache: Optional[Dict] = None,
    ) -> npt.NDArray:
        child_results = [
            child_layer.moment_of_nodes(
                order, center, requested, variables, cache=cache
            )
            for child_layer in self.child_layers
        ]
        # the moments of a decomposable product are the moments of the factor that owns
        # the variable, so summing the (zero padded) child moments is the right reduction
        result = np.zeros((self.number_of_nodes, len(order)))
        for child_layer_index, child_result in enumerate(child_results):
            nodes, child_nodes, unique = self.edges_of_child_layer(child_layer_index)
            if len(nodes) == 0:
                continue
            if unique:
                result[nodes] += child_result[child_nodes]
            else:
                np.add.at(result, nodes, child_result[child_nodes])
        return result

    def sample_forward(
        self,
        assignment: Dict[int, List[List[npt.NDArray]]],
        samples: npt.NDArray,
        variables: SortedSet,
    ):
        own_assignment = assignment[id(self)]

        rows_per_node = [
            np.concatenate(rows) if rows else None for rows in own_assignment
        ]

        for child_layer_index, node, child_node in self._edge_triples():
            rows = rows_per_node[node]
            if rows is None:
                continue
            child_layer = self.child_layers[child_layer_index]
            assignment[id(child_layer)][child_node].append(rows)

    # ------------------------------------------------------------------ structural

    def _structural_pass(
        self,
        child_results: List[Tuple[Layer, npt.NDArray]],
        log_probabilities: Dict[int, npt.NDArray],
    ) -> Tuple[Layer, npt.NDArray]:
        """
        Accumulate the log-probabilities of the children of every node.

        :param child_results: The new child layer and its node log-probabilities.
        :param log_probabilities: The map to record the result in.
        :return: The new layer and the log-probabilities of its nodes.
        """
        result = self.__class__(
            [child_layer for child_layer, _ in child_results], self.edges.copy()
        )

        own_log_probabilities = np.zeros(self.number_of_nodes)
        for child_layer_index, (_, child_log_probabilities) in enumerate(child_results):
            nodes, child_nodes, unique = self.edges_of_child_layer(child_layer_index)
            if len(nodes) == 0:
                continue
            if unique:
                own_log_probabilities[nodes] += child_log_probabilities[child_nodes]
            else:
                np.add.at(
                    own_log_probabilities, nodes, child_log_probabilities[child_nodes]
                )

        log_probabilities[id(result)] = own_log_probabilities
        return result, own_log_probabilities

    def log_truncated_of_simple_event(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        singleton_allowed: bool,
        cache: Optional[Dict] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        if cache is None:
            cache = {}
        key = ("truncated", id(self))
        if key in cache:
            return cache[key]

        child_results = [
            child_layer.log_truncated_of_simple_event(
                event,
                variables,
                singleton_allowed,
                cache=cache,
                log_probabilities=log_probabilities,
            )
            for child_layer in self.child_layers
        ]
        result = self._structural_pass(child_results, log_probabilities)
        cache[key] = result
        return result

    def log_truncated_of_simple_events(
        self,
        events: List[SimpleEvent],
        variables: SortedSet,
        singleton_allowed: bool,
        cache: Optional[Dict] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        if cache is None:
            cache = {}
        key = ("batched truncated", id(self))
        if key in cache:
            return cache[key]

        number_of_events = len(events)
        number_of_nodes = self.number_of_nodes
        number_of_entries = self.edges.number_of_stored_entries
        blocks = np.arange(number_of_events)

        new_child_layers = []
        child_log_probabilities = []
        for child_layer in self.child_layers:
            new_child_layer, child_log_probability = (
                child_layer.log_truncated_of_simple_events(
                    events,
                    variables,
                    singleton_allowed,
                    cache=cache,
                    log_probabilities=log_probabilities,
                )
            )
            new_child_layers.append(new_child_layer)
            child_log_probabilities.append(child_log_probability)

        # every edge is repeated once per event, pointing into that event's block of the
        # child layer
        node_counts = np.array(
            [child_layer.number_of_nodes for child_layer in self.child_layers]
        )
        rows = np.tile(self.edges.rows, number_of_events)
        columns = np.tile(self.edges.columns, number_of_events) + np.repeat(
            blocks * number_of_nodes, number_of_entries
        )
        data = np.tile(self.edges.data.astype(np.int64), number_of_events) + np.repeat(
            blocks, number_of_entries
        ) * np.tile(node_counts[self.edges.rows], number_of_events)

        edges = SparseArray.from_coordinates(
            rows, columns, data, (len(self.child_layers), number_of_events * number_of_nodes)
        )
        result = self.__class__(new_child_layers, edges)

        own_log_probabilities = np.zeros(number_of_events * number_of_nodes)
        for child_layer_index, child_log_probability in enumerate(
            child_log_probabilities
        ):
            mask = rows == child_layer_index
            np.add.at(
                own_log_probabilities,
                columns[mask],
                child_log_probability[data[mask]],
            )
        log_probabilities[id(result)] = own_log_probabilities

        cache[key] = (result, own_log_probabilities)
        return cache[key]

    def log_conditional_of_point(
        self,
        point: Dict[Variable, Any],
        variables: SortedSet,
        cache: Optional[Dict] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        if cache is None:
            cache = {}
        key = ("conditional", id(self))
        if key in cache:
            return cache[key]

        child_results = [
            child_layer.log_conditional_of_point(
                point,
                variables,
                cache=cache,
                log_probabilities=log_probabilities,
            )
            for child_layer in self.child_layers
        ]
        result = self._structural_pass(child_results, log_probabilities)
        cache[key] = result
        return result

    def required_child_nodes(
        self, alive: npt.NDArray, log_probabilities: Dict[int, npt.NDArray]
    ) -> List[Tuple[Layer, npt.NDArray]]:
        kept_edges = alive[self.edges.columns]
        result = []
        for child_layer_index, child_layer in enumerate(self.child_layers):
            mask = kept_edges & (self.edges.rows == child_layer_index)
            needed = np.zeros(child_layer.number_of_nodes, dtype=bool)
            needed[self.edges.data[mask].astype(np.int64)] = True
            result.append((child_layer, needed))
        return result

    def rebuild(
        self,
        needed: Dict[int, npt.NDArray],
        rebuilt: Dict[int, Optional[Layer]],
    ) -> Optional[Layer]:
        alive = needed[id(self)]
        if not alive.any():
            return None

        node_remap, number_of_nodes = remap_indices(alive)
        kept_edges = alive[self.edges.columns]

        new_child_layers = []
        new_rows = []
        new_columns = []
        new_data = []

        for child_layer_index, child_layer in enumerate(self.child_layers):
            mask = kept_edges & (self.edges.rows == child_layer_index)
            if not mask.any():
                continue

            pruned_child = rebuilt.get(id(child_layer))
            if pruned_child is None:
                # a factor of the product became impossible, so every node that
                # references it is impossible as well
                return None

            child_remap, _ = remap_indices(needed[id(child_layer)])
            new_rows.append(np.full(mask.sum(), len(new_child_layers), dtype=np.int64))
            new_columns.append(node_remap[self.edges.columns[mask]])
            new_data.append(child_remap[self.edges.data[mask].astype(np.int64)])
            new_child_layers.append(pruned_child)

        if not new_child_layers:
            return None

        edges = SparseArray.from_coordinates(
            np.concatenate(new_rows),
            np.concatenate(new_columns),
            np.concatenate(new_data),
            (len(new_child_layers), number_of_nodes),
        )
        return self.__class__(new_child_layers, edges)

    def marginal(
        self, kept: npt.NDArray, cache: Optional[Dict] = None
    ) -> Optional[Layer]:
        if cache is None:
            cache = {}
        key = ("marginal", id(self))
        if key in cache:
            return cache[key]

        new_child_layers = []
        new_rows = []
        new_columns = []
        new_data = []

        for child_layer_index, child_layer in enumerate(self.child_layers):
            marginal_child = child_layer.marginal(kept, cache)
            if marginal_child is None:
                continue
            mask = self.edges.rows == child_layer_index
            new_rows.append(np.full(mask.sum(), len(new_child_layers), dtype=np.int64))
            new_columns.append(self.edges.columns[mask])
            new_data.append(self.edges.data[mask])
            new_child_layers.append(marginal_child)

        if not new_child_layers:
            cache[key] = None
            return None

        edges = SparseArray.from_coordinates(
            np.concatenate(new_rows),
            np.concatenate(new_columns),
            np.concatenate(new_data),
            (len(new_child_layers), self.number_of_nodes),
        )
        result = self.__class__(new_child_layers, edges)
        cache[key] = result
        return result

    def simplify(self, cache: Optional[Dict] = None) -> Layer:
        if cache is None:
            cache = {}
        key = ("simplify", id(self))
        if key in cache:
            return cache[key]

        simplified_children = [
            child_layer.simplify(cache) for child_layer in self.child_layers
        ]
        result = self.__class__(simplified_children, self.edges.copy())

        if result.is_identity():
            result = simplified_children[0]

        cache[key] = result
        return result

    def is_identity(self) -> bool:
        """
        :return: Whether this layer forwards its single child layer unchanged.
        """
        if len(self.child_layers) != 1:
            return False
        if self.edges.number_of_stored_entries != self.number_of_nodes:
            return False
        if self.child_layers[0].number_of_nodes != self.number_of_nodes:
            return False
        sorted_edges = self.edges.sort_indices()
        expected = np.arange(self.number_of_nodes)
        return bool(
            np.array_equal(sorted_edges.columns, expected)
            and np.array_equal(sorted_edges.data.astype(np.int64), expected)
        )

    def __deepcopy__(self, memo=None) -> ProductLayer:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        child_layers = [
            child_layer.__deepcopy__(memo) for child_layer in self.child_layers
        ]
        result = self.__class__(child_layers, self.edges.copy())
        memo[id(self)] = result
        return result

    def to_json(self, **kwargs) -> Dict[str, Any]:
        result = super().to_json(**kwargs)
        result["edges"] = self.edges.to_json()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        child_layers = [
            Layer.from_json(child_layer, **kwargs)
            for child_layer in data["child_layers"]
        ]
        return cls(child_layers, SparseArray.from_json(data["edges"]))

    # ------------------------------------------------------------------ conversion

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(
        cls,
        nodes: List[ProductUnit],
        child_layers: List[LayerConverter],
        progress_bar: bool = False,
    ) -> LayerConverter:
        hash_remap = {hash(node): index for index, node in enumerate(nodes)}

        # only the candidates that at least one of these nodes points to become child
        # layers, so that the edge matrix has no empty rows
        used_child_layers: List[LayerConverter] = []
        row_of_child_layer: Dict[int, int] = {}

        rows, columns, values = [], [], []

        iterator = (
            tqdm.tqdm(nodes, desc="Assembling product layer") if progress_bar else nodes
        )
        for node_index, node in enumerate(iterator):
            subcircuit_hashes = {hash(subcircuit) for subcircuit in node.subcircuits}
            for child_layer_index, child_layer in enumerate(child_layers):
                for subcircuit_hash in subcircuit_hashes:
                    if subcircuit_hash not in child_layer.hash_remap:
                        continue
                    if child_layer_index not in row_of_child_layer:
                        row_of_child_layer[child_layer_index] = len(used_child_layers)
                        used_child_layers.append(child_layer)
                    rows.append(row_of_child_layer[child_layer_index])
                    columns.append(node_index)
                    values.append(child_layer.hash_remap[subcircuit_hash])

        edges = SparseArray.from_coordinates(
            np.array(rows, dtype=np.int64),
            np.array(columns, dtype=np.int64),
            np.array(values, dtype=np.int64),
            (len(used_child_layers), len(nodes)),
        )
        layer = cls([cl.layer for cl in used_child_layers], edges)
        return LayerConverter(layer, nodes, hash_remap)

    def to_rustworkx(
        self,
        variables: SortedSet,
        result: RustworkxProbabilisticCircuit,
        cache: Optional[Dict] = None,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Unit]:
        if cache is None:
            cache = {}
        if id(self) in cache:
            return cache[id(self)]

        if progress_bar:
            progress_bar.set_postfix_str(
                f"Parsing product layer of {[variables[i] for i in self.variables]}"
            )

        units = [
            ProductUnit(probabilistic_circuit=result)
            for _ in range(self.number_of_nodes)
        ]
        child_units = [
            child_layer.to_rustworkx(variables, result, cache, progress_bar)
            for child_layer in self.child_layers
        ]

        for child_layer_index, node, child_node in self._edge_triples():
            units[node].add_subcircuit(child_units[child_layer_index][child_node])
            if progress_bar:
                progress_bar.update()

        cache[id(self)] = units
        return units


@dataclass
class LayerConverter:
    """
    Bookkeeping for the conversion of a circuit of the ``rx`` package into a layered one.
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
