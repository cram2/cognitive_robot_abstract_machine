from __future__ import annotations

import math
from abc import abstractmethod, ABC
import equinox as eqx
import jax
from jax import numpy as jnp
from jax.experimental.sparse import BCOO, bcoo_concatenate
from jax.scipy.special import logsumexp
from jax.tree_util import tree_flatten
from jaxtyping import Int, Array
from krrood.adapters.json_serializer import SubclassJSONSerializer
from probabilistic_model.exceptions import ShapeMismatchError
from typing_extensions import List, Iterator, Tuple, Union, Dict, Any, Self, Optional

from probabilistic_model.probabilistic_circuit.jax.utils import copy_bcoo


class DifferentiableLayer(eqx.Module, SubclassJSONSerializer, ABC):
    """
    Abstract class for Layers of a layered circuit.

    Layers have the same scope (set of variables) for every node in them.
    """

    _variables: Optional[Array] = eqx.field(static=False, default=None)
    """
    The variable indices of the layer.
    """

    @property
    def variables(self) -> jax.Array:
        raise NotImplementedError

    def set_variables(self, value: jax.Array):
        raise NotImplementedError

    @abstractmethod
    def log_likelihood_of_nodes_single(self, x: Array) -> Array:
        """
        Calculate the log-likelihood of the distribution.

        :param x: The whole event, one value per variable of the circuit, which the
            layer indexes by the indices of its variables.
        :return: The log-likelihood of every node in the layer for x.
        """

    def log_likelihood_of_nodes(self, x: Array) -> Array:
        """
        Vectorized version of :meth:`log_likelihood_of_nodes_single`
        """
        return jax.vmap(self.log_likelihood_of_nodes_single)(x)

    def validate(self):
        """
        Validate the parameters and their layouts.
        """
        raise NotImplementedError

    @property
    def number_of_nodes(self) -> int:
        """
        :return: The number of nodes in the layer.
        """
        raise NotImplementedError

    def all_layers(self) -> List[DifferentiableLayer]:
        """
        :return: A list of all layers in the circuit.
        """
        return [self]

    def all_layers_with_depth(
        self, depth: int = 0
    ) -> List[Tuple[int, DifferentiableLayer]]:
        """
        :return: A list of tuples of all layers in the circuit with their depth.
        """
        return [(depth, self)]

    def __deepcopy__(
        self, memo: Optional[Dict[int, Any]] = None
    ) -> "DifferentiableLayer":
        """
        Create a deep copy of the layer.

        :param memo: A dictionary that is used to keep track of objects that have
            already been copied.
        """
        raise NotImplementedError

    def partition(self) -> Tuple[Any, Any]:
        """
        Partition the layer into the parameters and the static structure.

        :return: A tuple containing the parameters and the static structure as pytrees.
        """
        return eqx.partition(self, eqx.is_inexact_array)

    @property
    def number_of_trainable_parameters(self):
        """
        :return: The trainable parameters of the layer and all child layers.
        """
        parameters, _ = self.partition()
        flattened_parameters, _ = tree_flatten(parameters)
        number_of_parameters = sum(
            [len(parameter) for parameter in flattened_parameters]
        )
        return number_of_parameters

    @property
    def number_of_components(self) -> int:
        """
        :return: The number of components (leaves + edges) of the entire circuit
        """
        return self.number_of_nodes


class DifferentiableInnerLayer(DifferentiableLayer, ABC):
    """
    Abstract Base Class for inner layers.
    """

    child_layers: List[DifferentiableLayer]
    """
    The child layers of this layer.
    """

    def __init__(self, child_layers: List[DifferentiableLayer]):
        super().__init__()
        self.child_layers = child_layers
        self.variables  # initialize the variables of the layer

    def set_variables(self, value: jax.Array):
        raise AttributeError("Variables of inner layers are read-only.")

    def reset_variables(self):
        object.__setattr__(self, "_variables", None)

    def all_layers(self) -> List[DifferentiableLayer]:
        """
        :return: A list of all layers in the circuit.
        """
        result = [self]
        for child_layer in self.child_layers:
            result.extend(child_layer.all_layers())
        return result

    def all_layers_with_depth(
        self, depth: int = 0
    ) -> List[Tuple[int, DifferentiableLayer]]:
        """
        :return: A list of tuples of all layers in the circuit with their depth.
        """
        result = [(depth, self)]
        for child_layer in self.child_layers:
            result.extend(child_layer.all_layers_with_depth(depth + 1))
        return result

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["child_layers"] = [
            child_layer.to_json() for child_layer in self.child_layers
        ]
        return result


class DifferentiableInputLayer(DifferentiableLayer, ABC):
    """
    Abstract base class for univariate input units.

    Input layers contain only one type of distribution such that the vectorization of
    the log likelihood calculation works without bottleneck statements like if/else or
    loops.
    """

    def __init__(self, variable: int):
        super().__init__()
        self._variables = jnp.array([variable])

    @property
    def variables(self) -> jax.Array:
        return self._variables

    def set_variables(self, value: jax.Array):
        object.__setattr__(self, "_variables", value)

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["variable"] = self._variables[0].item()
        return result

    @property
    def variable(self):
        return self._variables[0].item()

    def log_likelihood_of_nodes_single(self, x: Array) -> Array:
        return self.log_likelihood_of_nodes_of_value(x[self._variables])

    @abstractmethod
    def log_likelihood_of_nodes_of_value(self, value: Array) -> Array:
        """
        Calculate the log-likelihood of every node for a value of the variable.

        :param value: The value of the variable of this layer, with shape (1,).
        :return: The log-likelihood of every node in the layer.
        """


class DifferentiableSumLayer(DifferentiableInnerLayer, ABC):
    log_weights: List[Union[jax.Array, BCOO]]
    child_layers: Union[
        List[DifferentiableProductLayer], List[DifferentiableInputLayer]
    ]

    def __init__(
        self,
        child_layers: List[DifferentiableLayer],
        log_weights: List[Union[jax.Array, BCOO]],
    ):
        super().__init__(child_layers)
        self.log_weights = log_weights

    def validate(self):
        for log_weights in self.log_weights:
            if not log_weights.shape[0] == self.number_of_nodes:
                raise ShapeMismatchError(self.number_of_nodes, log_weights.shape[0])

        for log_weights, child_layer in self.log_weighted_child_layers:
            if not log_weights.shape[1] == child_layer.number_of_nodes:
                raise ShapeMismatchError(
                    child_layer.number_of_nodes,
                    log_weights.shape[1],
                )

    @property
    def log_weighted_child_layers(self) -> Iterator[Tuple[BCOO, DifferentiableLayer]]:
        """
        :returns: Yields the log-weights and the child layers zipped together.
        """
        yield from zip(self.log_weights, self.child_layers)

    @property
    def variables(self) -> jax.Array:
        if self._variables is None:
            object.__setattr__(self, "_variables", self.child_layers[0].variables)
        return self._variables

    @property
    def number_of_nodes(self) -> int:
        return self.log_weights[0].shape[0]


class DifferentiableSparseSumLayer(DifferentiableSumLayer):
    log_weights: List[BCOO]

    @property
    def number_of_components(self) -> int:
        return sum(
            [child_layer.number_of_components for child_layer in self.child_layers]
        ) + sum([child_log_weights.nse for child_log_weights in self.log_weights])

    @property
    def concatenated_log_weights(self) -> BCOO:
        """
        :return: The concatenated log_weights of the child layers for each node.
        """
        return bcoo_concatenate(self.log_weights, dimension=1).sort_indices()

    @property
    def log_normalization_constants(self) -> jax.Array:
        result = self.concatenated_log_weights
        maximum = result.data.max()
        result.data = jnp.exp(result.data - maximum)
        result = result.sum(1).todense()
        return maximum + jnp.log(result)

    @property
    def normalized_weights(self):
        """
        :return: The normalized log_weights of the child layers for each node.
        """
        result = self.concatenated_log_weights
        log_normalization_constants = self.log_normalization_constants
        result.data = jnp.exp(
            result.data - log_normalization_constants[result.indices[:, 0]]
        )
        return result

    def log_likelihood_of_nodes_single(self, x: jax.Array) -> jax.Array:
        rows, weighted_log_likelihoods = [], []
        for log_weights, child_layer in self.log_weighted_child_layers:
            child_layer_log_likelihood = child_layer.log_likelihood_of_nodes_single(x)
            rows.append(log_weights.indices[:, 0])
            weighted_log_likelihoods.append(
                log_weights.data + child_layer_log_likelihood[log_weights.indices[:, 1]]
            )
        rows = jnp.concatenate(rows)
        weighted_log_likelihoods = jnp.concatenate(weighted_log_likelihoods)

        # a log-sum-exp per node, shifted by the largest entry of the node, so that a
        # likelihood below the smallest positive float does not vanish to zero
        maximum = jax.ops.segment_max(
            weighted_log_likelihoods, rows, num_segments=self.number_of_nodes
        )
        shift = jax.lax.stop_gradient(jnp.where(jnp.isfinite(maximum), maximum, 0.0))
        shifted_sum = jax.ops.segment_sum(
            jnp.exp(weighted_log_likelihoods - shift[rows]),
            rows,
            num_segments=self.number_of_nodes,
        )
        return jnp.log(shifted_sum) + shift - self.log_normalization_constants

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        child_layers = [
            child_layer.__deepcopy__(memo) for child_layer in self.child_layers
        ]
        log_weights = [copy_bcoo(log_weight) for log_weight in self.log_weights]
        result = self.__class__(child_layers, log_weights)
        memo[id_self] = result
        return result

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["log_weights"] = [
            (
                child_log_weights.data.tolist(),
                child_log_weights.indices.tolist(),
                child_log_weights.shape,
            )
            for child_log_weights in self.log_weights
        ]
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        child_layers = [
            DifferentiableLayer.from_json(child_layer)
            for child_layer in data["child_layers"]
        ]
        log_weights = [
            BCOO(
                (jnp.array(values), jnp.array(indices)),
                shape=shape,
                indices_sorted=True,
                unique_indices=True,
            )
            for values, indices, shape in data["log_weights"]
        ]
        return cls(child_layers, log_weights)


class DifferentiableDenseSumLayer(DifferentiableSumLayer):
    log_weights: List[jax.Array]
    child_layers: Union[
        List[DifferentiableProductLayer], List[DifferentiableInputLayer]
    ]

    @property
    def number_of_components(self) -> float:
        return sum(
            [child_layer.number_of_components for child_layer in self.child_layers]
        ) + sum(
            [
                math.prod(child_log_weights.shape)
                for child_log_weights in self.log_weights
            ]
        )

    @property
    def concatenated_log_weights(self) -> Array:
        """
        :return: The concatenated log_weights of the child layers for each node.
        """
        return jnp.concatenate(self.log_weights, axis=1)

    @property
    def log_normalization_constants(self) -> jax.Array:
        return logsumexp(self.concatenated_log_weights, 1)

    @property
    def normalized_weights(self):
        """
        :return: The normalized log_weights of the child layers for each node.
        """
        return jnp.exp(
            self.concatenated_log_weights
            - self.log_normalization_constants.reshape(-1, 1)
        )

    def log_likelihood_of_nodes_single(self, x: jax.Array) -> jax.Array:
        # stay in log space, so that a likelihood below the smallest positive float does
        # not vanish to zero
        log_likelihood_per_child_layer = jnp.stack(
            [
                logsumexp(
                    log_weights + child_layer.log_likelihood_of_nodes_single(x), 1
                )
                for log_weights, child_layer in self.log_weighted_child_layers
            ]
        )
        return (
            logsumexp(log_likelihood_per_child_layer, 0)
            - self.log_normalization_constants
        )

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        child_layers = [
            child_layer.__deepcopy__(memo) for child_layer in self.child_layers
        ]
        log_weights = [jnp.copy(log_weight) for log_weight in self.log_weights]
        result = self.__class__(child_layers, log_weights)
        memo[id_self] = result
        return result

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["log_weights"] = [
            child_log_weights.tolist() for child_log_weights in self.log_weights
        ]
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        child_layers = [
            DifferentiableLayer.from_json(child_layer)
            for child_layer in data["child_layers"]
        ]
        log_weights = [
            jnp.asarray(child_log_weights) for child_log_weights in data["log_weights"]
        ]
        return cls(child_layers, log_weights)


class DifferentiableProductLayer(DifferentiableInnerLayer):
    """
    A layer that represents the product of multiple other units.
    """

    child_layers: List[Union[DifferentiableSparseSumLayer, DifferentiableInputLayer]]
    """
    The child of a product layer is a list that contains groups sum units with the same
    scope or groups of input units with the same scope.
    """

    edges: Int[BCOO, "len(child_layers), number_of_nodes"] = eqx.field(static=True)
    """
    The edges consist of a sparse matrix containing integers.

    The first dimension describes the edges for each child layer. The second dimension
    describes the edges for each node in the child layer. The integers are interpreted
    in such a way that n-th value represents a edge (n, edges[n]).

    Nodes in the child layer can be mapped to by multiple nodes in this layer.

    The shape is (#child_layers, #nodes).
    """

    def __init__(self, child_layers: List[DifferentiableLayer], edges: BCOO):
        """
        Initialize the product layer.

        :param child_layers: The child layers of the product layer.
        :param edges: The edges of the product layer.
        """
        super().__init__(child_layers)
        self.edges = edges
        self.variables

    def validate(self):
        if not self.edges.shape == (len(self.child_layers), self.number_of_nodes):
            raise ShapeMismatchError(
                (len(self.child_layers), self.number_of_nodes), self.edges.shape
            )

    @property
    def number_of_nodes(self) -> int:
        return self.edges.shape[1]

    @property
    def number_of_components(self) -> int:
        return (
            sum([child_layer.number_of_components for child_layer in self.child_layers])
            + self.edges.nse
        )

    @DifferentiableLayer.variables.getter
    def variables(self) -> jax.Array:
        if self._variables is None:
            variables = jnp.concatenate(
                [child_layer.variables for child_layer in self.child_layers]
            )
            variables = jnp.unique(variables)
            object.__setattr__(self, "_variables", variables)
        return self._variables

    def log_likelihood_of_nodes_single(self, x: jax.Array) -> jax.Array:
        result = jnp.zeros(self.number_of_nodes, dtype=jnp.float32)

        for edges, layer in zip(self.edges, self.child_layers):
            # every layer reads the variables of its scope from the whole event
            log_likelihoods = layer.log_likelihood_of_nodes_single(x)  # (#child_nodes,)

            # gather the log-likelihoods of the child nodes the edges point at
            log_likelihoods = log_likelihoods[edges.data]  # (#edges,)

            # add the gathered values to the result where the edges define the indices
            result = result.at[edges.indices[:, 0]].add(log_likelihoods)

        return result

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        child_layers = [
            child_layer.__deepcopy__(memo) for child_layer in self.child_layers
        ]
        edges = copy_bcoo(self.edges)
        result = self.__class__(child_layers, edges)
        memo[id_self] = result
        return result

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["edges"] = (
            self.edges.data.tolist(),
            self.edges.indices.tolist(),
            self.edges.shape,
        )
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        child_layers = [
            DifferentiableLayer.from_json(child_layer)
            for child_layer in data["child_layers"]
        ]
        edges = BCOO(
            (jnp.array(data["edges"][0]), jnp.array(data["edges"][1])),
            shape=data["edges"][2],
            indices_sorted=True,
            unique_indices=True,
        )
        return cls(child_layers, edges)
