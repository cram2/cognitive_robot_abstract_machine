from __future__ import annotations

import inspect
import math
from abc import abstractmethod, ABC
from dataclasses import dataclass

import equinox as eqx
import jax
import tqdm
from jax import numpy as jnp
from jax.experimental.sparse import BCOO, bcoo_concatenate
from jax.scipy.special import logsumexp
from jax.tree_util import tree_flatten
from jaxtyping import Int
from random_events.utils import recursive_subclasses, SubclassJSONSerializer
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import List, Iterator, Tuple, Union, Type, Dict, Any, Self, Optional

from .utils import copy_bcoo
from ..rx.probabilistic_circuit import (SumUnit, ProductUnit, Unit, ProbabilisticCircuit as NXProbabilisticCircuit)


def inverse_class_of(clazz: Type[Unit]) -> Type[Layer]:
    for subclass in recursive_subclasses(Layer):
        if not inspect.isabstract(subclass):
            if issubclass(clazz, subclass.nx_classes()):
                return subclass

    raise TypeError(f"Could not find class for {clazz}")


class Layer(eqx.Module, SubclassJSONSerializer, ABC):
    """
    Abstract class for Layers of a layered circuit.

    Layers have the same scope (set of variables) for every node in them.
    """

    _variables: Optional[jnp.array] = eqx.field(static=False, default=None)
    """
    The variable indices of the layer.
    """

    @property
    def variables(self) -> jax.Array:
        raise NotImplementedError

    def set_variables(self, value: jax.Array):
        raise NotImplementedError

    @abstractmethod
    def log_likelihood_of_nodes_single(self, x: jnp.array) -> jnp.array:
        """
        Calculate the log-likelihood of the distribution.

        :param x: The input vector.
        :return: The log-likelihood of every node in the layer for x.
        """
        raise NotImplementedError

    def log_likelihood_of_nodes(self, x: jnp.array) -> jnp.array:
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

    def all_layers(self) -> List[Layer]:
        """
        :return: A list of all layers in the circuit.
        """
        return [self]

    def all_layers_with_depth(self, depth: int = 0) -> List[Tuple[int, Layer]]:
        """
        :return: A list of tuples of all layers in the circuit with their depth.
        """
        return [(depth, self)]

    def __deepcopy__(self, memo=None) -> 'Layer':
        """
        Create a deep copy of the layer.

        :param memo: A dictionary that is used to keep track of objects that have already been copied.
        """
        raise NotImplementedError

    @classmethod
    def nx_classes(cls) -> Tuple[Type, ...]:
        """
        :return: The tuple of matching classes of the layer in the probabilistic_model.probabilistic_circuit.rx package.
        """
        return tuple()

    def to_nx(self, variables: SortedSet[Variable], result: NXProbabilisticCircuit,
              progress_bar: Optional[tqdm.tqdm] = None, ) -> List[Unit]:
        """
        Convert the layer to a networkx circuit.
        For every node in this circuit, a corresponding node in the networkx circuit
        is created.
        The nodes all belong to the same circuit.

        :param variables: The variables of the circuit.
        :param result: The resulting circuit to write into
        :param progress_bar: A progress bar to show the progress.

        :return: The nodes of the networkx circuit.
        """
        raise NotImplementedError

    @staticmethod
    def create_layers_from_nodes(nodes: List[Unit], child_layers: List[NXConverterLayer], progress_bar: bool = True) -> \
    List[NXConverterLayer]:
        """
        Create a layer from a list of nodes.
        """
        result = []

        unique_types = set(type(node) if not node.is_leaf else type(node.distribution) for node in nodes)
        for unique_type in unique_types:
            nodes_of_current_type = [node for node in nodes if (
                isinstance(node, unique_type) if not node.is_leaf else isinstance(node.distribution, unique_type))]

            if nodes[0].is_leaf:
                unique_type = type(nodes_of_current_type[0].distribution)

            layer_type = inverse_class_of(unique_type)

            scopes = [tuple(node.variables) for node in nodes_of_current_type]
            unique_scopes = set(scopes)
            for scope in unique_scopes:
                nodes_of_current_type_and_scope = [node for node in nodes_of_current_type if
                                                   tuple(node.variables) == scope]

                layer = layer_type.create_layer_from_nodes_with_same_type_and_scope(nodes_of_current_type_and_scope,
                                                                                    child_layers, progress_bar)
                result.append(layer)

        return result

    @classmethod
    @abstractmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[Unit], child_layers: List[NXConverterLayer],
                                                         progress_bar: bool = True) -> NXConverterLayer:
        """
        Create a layer from a list of nodes with the same type and scope.
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
        number_of_parameters = sum([len(p) for p in flattened_parameters])
        return number_of_parameters

    @property
    def number_of_components(self) -> int:
        """
        :return: The number of components (leaves + edges) of the entire circuit
        """
        return self.number_of_nodes


class InnerLayer(Layer, ABC):
    """
    Abstract Base Class for inner layers
    """

    child_layers: List[Layer]
    """
    The child layers of this layer.
    """

    def __init__(self, child_layers: List[Layer]):
        super().__init__()
        self.child_layers = child_layers
        self.variables  # initialize the variables of the layer

    def set_variables(self, value: jnp.array):
        raise AttributeError("Variables of inner layers are read-only.")

    def reset_variables(self):
        object.__setattr__(self, "_variables", None)

    def all_layers(self) -> List[Layer]:
        """
        :return: A list of all layers in the circuit.
        """
        result = [self]
        for child_layer in self.child_layers:
            result.extend(child_layer.all_layers())
        return result

    def all_layers_with_depth(self, depth: int = 0) -> List[Tuple[int, Layer]]:
        """
        :return: A list of tuples of all layers in the circuit with their depth.
        """
        result = [(depth, self)]
        for child_layer in self.child_layers:
            result.extend(child_layer.all_layers_with_depth(depth + 1))
        return result

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["child_layers"] = [child_layer.to_json() for child_layer in self.child_layers]
        return result


class InputLayer(Layer, ABC):
    """
    Abstract base class for univariate input units.

    Input layers contain only one type of distribution such that the vectorization of the log likelihood
    calculation works without bottleneck statements like if/else or loops.
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


class SumLayer(InnerLayer, ABC):
    log_weights: List[Union[jax.array, BCOO]]
    child_layers: Union[List[[ProductLayer]], List[InputLayer]]

    def __init__(self, child_layers: List[Layer], log_weights: List[Union[jax.array, BCOO]]):
        super().__init__(child_layers)
        self.log_weights = log_weights

    def validate(self):
        for log_weights in self.log_weights:
            assert log_weights.shape[
                       0] == self.number_of_nodes, "The number of nodes must match the number of log_weights."

        for log_weights, child_layer in self.log_weighted_child_layers:
            assert log_weights.shape[
                       1] == child_layer.number_of_nodes, "The number of nodes must match the number of log_weights."
            assert (child_layer.variables == self.variables).all(), "The variables must match."

    @property
    def log_weighted_child_layers(self) -> Iterator[Tuple[BCOO, Layer]]:
        """
        :returns: Yields log log_weights and the child layers zipped together.
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


class SparseSumLayer(SumLayer):
    log_weights: List[BCOO]

    @classmethod
    def nx_classes(cls) -> Tuple[Type, ...]:
        return SumUnit,

    @property
    def number_of_components(self) -> int:
        return sum([cl.number_of_components for cl in self.child_layers]) + sum([lw.nse for lw in self.log_weights])

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
        z = self.log_normalization_constants
        result.data = jnp.exp(result.data - z[result.indices[:, 0]])
        return result

    def log_likelihood_of_nodes_single(self, x: jax.Array) -> jax.Array:
        result = jnp.zeros(self.number_of_nodes, dtype=jnp.float32)

        for log_weights, child_layer in self.log_weighted_child_layers:
            # get the log likelihoods of the child nodes
            child_layer_log_likelihood = child_layer.log_likelihood_of_nodes_single(x)

            # weight the log likelihood of the child nodes by the weight for each node of this layer
            cloned_log_weights = copy_bcoo(log_weights)  # clone the log_weights

            # multiply the log_weights with the child layer likelihood
            cloned_log_weights.data += child_layer_log_likelihood[cloned_log_weights.indices[:, 1]]
            cloned_log_weights.data = jnp.exp(cloned_log_weights.data)  # exponent log_weights
            result = result.at[cloned_log_weights.indices[:, 0]].add(cloned_log_weights.data, indices_are_sorted=False,
                                                                     unique_indices=False)

        return jnp.log(result) - self.log_normalization_constants

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        child_layers = [child_layer.__deepcopy__(memo) for child_layer in self.child_layers]
        log_weights = [copy_bcoo(log_weight) for log_weight in self.log_weights]
        result = self.__class__(child_layers, log_weights)
        memo[id_self] = result
        return result

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["log_weights"] = [(lw.data.tolist(), lw.indices.tolist(), lw.shape) for lw in self.log_weights]
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        child_layer = [Layer.from_json(child_layer) for child_layer in data["child_layers"]]
        log_weights = [BCOO((jnp.array(lw[0]), jnp.array(lw[1])), shape=lw[2], indices_sorted=True, unique_indices=True)
                       for lw in data["log_weights"]]
        return cls(child_layer, log_weights)

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[SumUnit],
                                                         child_layers: List[NXConverterLayer],
                                                         progress_bar: bool = True) -> NXConverterLayer:

        result_hash_remap = {hash(node): index for index, node in enumerate(nodes)}
        variables = jnp.array(
            [nodes[0].probabilistic_circuit.variables.index(variable) for variable in nodes[0].variables])

        number_of_nodes = len(nodes)

        # filter the child layers to only contain layers with the same scope as this one
        filtered_child_layers = [child_layer for child_layer in child_layers if
                                 (child_layer.layer.variables == variables).all()]
        log_weights = []

        # for every possible child layer
        for child_layer in filtered_child_layers:

            # initialize indices and values for sparse weight matrix
            indices = []
            values = []

            # gather indices and log log_weights
            for index, node in enumerate(
                    tqdm.tqdm(nodes, desc="Calculating log_weights for sum node") if progress_bar else nodes):
                for weight, subcircuit in node.log_weighted_subcircuits:
                    if hash(subcircuit) in child_layer.hash_remap:
                        indices.append((index, child_layer.hash_remap[hash(subcircuit)]))
                        values.append((weight))

            # assemble sparse log weight matrix
            log_weights.append(BCOO((jnp.array(values), jnp.array(indices)),
                                    shape=(number_of_nodes, child_layer.layer.number_of_nodes)))

        sum_layer = cls([cl.layer for cl in filtered_child_layers], log_weights)
        return NXConverterLayer(sum_layer, nodes, result_hash_remap)

    def to_nx(self, variables: SortedSet[Variable], result: NXProbabilisticCircuit,
              progress_bar: Optional[tqdm.tqdm] = None) -> List[Unit]:

        variables_ = [variables[i] for i in self.variables]

        if progress_bar:
            progress_bar.set_postfix_str(f"Parsing Sum Layer for variables {variables_}")

        units = [SumUnit(probabilistic_circuit=result) for _ in range(self.number_of_nodes)]

        child_layer_nx = [cl.to_nx(variables, result, progress_bar) for cl in self.child_layers]

        for log_weights, child_layer in zip(self.log_weights, child_layer_nx):

            # extract the log_weights for the child layer
            for ((row, col), log_weight) in zip(log_weights.indices, log_weights.data):
                units[row].add_subcircuit(child_layer[col], log_weight.item())
                if progress_bar:
                    progress_bar.update()

        [unit.normalize() for unit in units]

        return units


class DenseSumLayer(SumLayer):
    log_weights: List[jnp.array]
    child_layers: Union[List[[ProductLayer]], List[InputLayer]]

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[Unit], child_layers: List[NXConverterLayer],
                                                         progress_bar: bool = True) -> NXConverterLayer:
        raise NotImplementedError

    @property
    def number_of_components(self) -> int:
        return sum([cl.number_of_components for cl in self.child_layers]) + sum(
            [math.prod(lw.shape) for lw in self.log_weights])

    @classmethod
    def nx_classes(cls) -> Tuple[Type, ...]:
        return tuple()

    @property
    def concatenated_log_weights(self) -> jnp.array:
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
        return jnp.exp(self.concatenated_log_weights - self.log_normalization_constants.reshape(-1, 1))

    def log_likelihood_of_nodes_single(self, x: jax.Array) -> jax.Array:
        result = jnp.zeros(self.number_of_nodes, dtype=jnp.float32)

        for log_weights, child_layer in self.log_weighted_child_layers:
            # get the log likelihoods of the child nodes
            child_layer_log_likelihood = child_layer.log_likelihood_of_nodes_single(x)

            # weight the log likelihood of the child nodes by the weight for each node of this layer
            log_likelihood = log_weights + child_layer_log_likelihood
            log_likelihood = jnp.exp(logsumexp(log_likelihood, 1))
            result += log_likelihood

        return jnp.log(result) - self.log_normalization_constants

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        child_layers = [child_layer.__deepcopy__(memo) for child_layer in self.child_layers]
        log_weights = [jnp.copy(log_weight) for log_weight in self.log_weights]
        result = self.__class__(child_layers, log_weights)
        memo[id_self] = result
        return result

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["log_weights"] = [lw.tolist() for lw in self.log_weights]
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        child_layer = [Layer.from_json(child_layer) for child_layer in data["child_layers"]]
        log_weights = [jnp.asarray(lw) for lw in data["log_weights"]]
        return cls(child_layer, log_weights)

    def to_nx(self, variables: SortedSet[Variable], result: NXProbabilisticCircuit,
              progress_bar: Optional[tqdm.tqdm] = None) -> List[Unit]:

        variables_ = [variables[i] for i in self.variables]

        if progress_bar:
            progress_bar.set_postfix_str(f"Parsing Dense Sum Layer for variables {variables_}")

        units = [SumUnit(probabilistic_circuit=result) for _ in range(self.number_of_nodes)]

        child_layer_nx = [cl.to_nx(variables, result, progress_bar) for cl in self.child_layers]

        for log_weights, child_layer in zip(self.log_weights, child_layer_nx):
            # extract the log_weights for the child layer
            for row in range(log_weights.shape[0]):
                for col in range(log_weights.shape[1]):
                    units[row].add_subcircuit(child_layer[col], jnp.exp(log_weights[row, col]).item())

                    if progress_bar:
                        progress_bar.update()

        [unit.normalize() for unit in units]

        return units


class ProductLayer(InnerLayer):
    """
    A layer that represents the product of multiple other units.
    """

    child_layers: List[Union[SparseSumLayer, InputLayer]]
    """
    The child of a product layer is a list that contains groups sum units with the same scope or groups of input
    units with the same scope.
    """

    edges: Int[BCOO, "len(child_layers), number_of_nodes"] = eqx.field(static=True)
    """
    The edges consist of a sparse matrix containing integers.
    The first dimension describes the edges for each child layer.
    The second dimension describes the edges for each node in the child layer.
    The integers are interpreted in such a way that n-th value represents a edge (n, edges[n]).

    Nodes in the child layer can be mapped to by multiple nodes in this layer.

    The shape is (#child_layers, #nodes).
    """

    def __init__(self, child_layers: List[Layer], edges: BCOO):
        """
        Initialize the product layer.

        :param child_layers: The child layers of the product layer.
        :param edges: The edges of the product layer.
        """
        super().__init__(child_layers)
        self.edges = edges
        self.variables

    def validate(self):
        assert self.edges.shape == (len(self.child_layers), self.number_of_nodes), (
            f"The shape of the edges must be {(len(self.child_layers), self.number_of_nodes)} "
            f"but was {self.edges.shape}.")

    @property
    def number_of_nodes(self) -> int:
        return self.edges.shape[1]

    @classmethod
    def nx_classes(cls) -> Tuple[Type, ...]:
        return ProductUnit,

    @property
    def number_of_components(self) -> int:
        return sum([cl.number_of_components for cl in self.child_layers]) + self.edges.nse

    @Layer.variables.getter
    def variables(self) -> jax.Array:
        if self._variables is None:
            variables = jnp.concatenate([child_layer.variables for child_layer in self.child_layers])
            variables = jnp.unique(variables)
            object.__setattr__(self, "_variables", variables)
        return self._variables

    def log_likelihood_of_nodes_single(self, x: jax.Array) -> jax.Array:
        result = jnp.zeros(self.number_of_nodes, dtype=jnp.float32)

        for edges, layer in zip(self.edges, self.child_layers):
            # calculate the log likelihood over the columns of the child layer
            ll = layer.log_likelihood_of_nodes_single(x[layer.variables])  # shape: #child_nodes

            # gather the ll at the indices of the nodes that are required for the edges
            ll = ll[edges.data]  # shape: #len(edges.values())

            # add the gathered values to the result where the edges define the indices
            result = result.at[edges.indices[:, 0]].add(ll)

        return result

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        child_layers = [child_layer.__deepcopy__(memo) for child_layer in self.child_layers]
        edges = copy_bcoo(self.edges)
        result = self.__class__(child_layers, edges)
        memo[id_self] = result
        return result

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["edges"] = (self.edges.data.tolist(), self.edges.indices.tolist(), self.edges.shape)
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        child_layer = [Layer.from_json(child_layer) for child_layer in data["child_layers"]]
        edges = BCOO((jnp.array(data["edges"][0]), jnp.array(data["edges"][1])), shape=data["edges"][2],
                     indices_sorted=True, unique_indices=True)
        return cls(child_layer, edges)

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[Unit], child_layers: List[NXConverterLayer],
                                                         progress_bar: bool = True) -> NXConverterLayer:

        hash_remap = {hash(node): index for index, node in enumerate(nodes)}
        number_of_nodes = len(nodes)

        edge_indices = []
        edge_values = []
        if progress_bar:
            progress_bar = tqdm.tqdm(total=number_of_nodes, desc="Assembling Product Layer")
        # for every node in the nodes for this layer
        for node_index, node in enumerate(nodes):

            # for every child layer
            for child_layer_index, child_layer in enumerate(child_layers):
                cl_variables = SortedSet(
                    [node.probabilistic_circuit.variables[index] for index in child_layer.layer.variables])

                # for every subcircuit
                for subcircuit_index, subcircuit in enumerate(node.subcircuits):
                    # if the scopes are compatible
                    if cl_variables == subcircuit.variables:
                        # add the edge
                        edge_indices.append([child_layer_index, node_index])
                        edge_values.append(child_layer.hash_remap[hash(subcircuit)])
            if progress_bar:
                progress_bar.update(1)

        # assemble sparse edge tensor
        edges = (BCOO((jnp.array(edge_values), jnp.array(edge_indices)),
                      shape=(len(child_layers), number_of_nodes)).sort_indices().sum_duplicates(remove_zeros=False))
        layer = cls([cl.layer for cl in child_layers], edges)
        return NXConverterLayer(layer, nodes, hash_remap)

    def to_nx(self, variables: SortedSet[Variable], result: NXProbabilisticCircuit,
              progress_bar: Optional[tqdm.tqdm] = None) -> List[Unit]:

        if result is None:
            result = NXProbabilisticCircuit()

        variables_ = [variables[i] for i in self.variables]
        if progress_bar:
            progress_bar.set_postfix_str(f"Parsing Product Layer of variables {variables_}")

        units = [ProductUnit(probabilistic_circuit=result) for _ in range(self.number_of_nodes)]

        child_layer_nx = [cl.to_nx(variables, result, progress_bar) for cl in self.child_layers]
        for (row, col), data in zip(self.edges.indices, self.edges.data):
            units[col].add_subcircuit(child_layer_nx[row][data])

            if progress_bar:
                progress_bar.update()

        return units


@dataclass
class NXConverterLayer:
    """
    Class used for conversion from a probabilistic circuit in networkx to a layered circuit in jax.
    """
    layer: Layer
    nodes: List[Unit]
    hash_remap: Dict[int, int]
