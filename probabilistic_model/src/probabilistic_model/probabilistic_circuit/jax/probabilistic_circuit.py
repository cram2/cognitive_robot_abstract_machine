from __future__ import annotations

import collections
from typing import Dict, Any

import numpy as np
import optax
from jax.experimental.sparse import BCOO
from random_events.product_algebra import SimpleEvent
from random_events.utils import SubclassJSONSerializer
from random_events.variable import Variable, Symbolic
from sortedcontainers import SortedSet
from typing_extensions import Tuple, Self, List, Optional

from . import ProductLayer, SparseSumLayer, InputLayer, InnerLayer
from .discrete_layer import DiscreteLayer
from .inner_layer import Layer, NXConverterLayer
from ..rx.probabilistic_circuit import ProbabilisticCircuit as NXProbabilisticCircuit
import jax
import tqdm
import networkx as nx
import jax.numpy as jnp
import equinox as eqx


class ProbabilisticCircuit(SubclassJSONSerializer):
    """
    A probabilistic circuit as wrapper for a layered probabilistic model.
    """

    variables: SortedSet
    """
    The variables of the circuit.
    """

    root: Layer
    """
    The root layer of the circuit.
    """

    def __init__(self, variables: SortedSet, root: Layer):
        self.variables = variables
        self.root = root

    def log_likelihood(self, x: jax.Array) -> jax.Array:
        return self.root.log_likelihood_of_nodes(x)[:, 0]

    @classmethod
    def from_nx(cls, pc: NXProbabilisticCircuit, progress_bar: bool = False) -> ProbabilisticCircuit:
        """
        Convert a probabilistic circuit to a layered circuit.
        The result expresses the same distribution as `pc`.

        :param pc: The probabilistic circuit.
        :param progress_bar: Whether to show a progress bar.
        :return: The layered circuit.
        """

        # group nodes by depth
        layer_to_nodes_map = {index: layer for index, layer in enumerate(pc.layers)}
        reversed_layers_to_nodes_map = dict(reversed(layer_to_nodes_map.items()))

        # create layers from nodes
        child_layers: List[NXConverterLayer] = []
        for layer_index, nodes in (tqdm.tqdm(reversed_layers_to_nodes_map.items(), desc="Creating Layers") if progress_bar
                                                     else reversed_layers_to_nodes_map.items()):

            child_layers = Layer.create_layers_from_nodes(nodes, child_layers, progress_bar)
        root = child_layers[0].layer

        return cls(pc.variables, root)

    def to_nx(self, progress_bar: bool = True) -> NXProbabilisticCircuit:
        """
        Convert the probabilistic circuit to a networkx graph.

        :param progress_bar: Whether to show a progress bar.
        :return: The networkx graph.
        """
        if progress_bar:
            number_of_edges = self.root.number_of_components
            progress_bar = tqdm.tqdm(total=number_of_edges, desc="Converting to rx")
        else:
            progress_bar = None
        result = NXProbabilisticCircuit()
        self.root.to_nx(self.variables, result, progress_bar)
        return result

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["variables"] = [variable.to_json() for variable in self.variables]
        result["root"] = self.root.to_json()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variables = SortedSet(Variable.from_json(variable) for variable in data["variables"])
        root = Layer.from_json(data["root"])
        return cls(variables, root)

    def fit(self, data: jax.Array, epochs: int = 100,
            optimizer: Optional[optax.GradientTransformation] = None, **kwargs) -> None:
        """
        Fit the circuit to the data using generative training with the negative average log-likelihood as loss.

        :param data: The data.
        :param epochs: The number of epochs.
        :param optimizer: The optimizer to use.
        If `None`, the Adam optimizer with a learning rate of 1e-3 is used.
        """

        @eqx.filter_jit
        def loss(p, x):
            ll = p.log_likelihood_of_nodes(x)
            return -jnp.mean(ll)

        if optimizer is None:
            optimizer = optax.adam(1e-3)

        opt_state = optimizer.init(eqx.filter(self.root, eqx.is_inexact_array))

        progress_bar = tqdm.tqdm(range(epochs), desc="Fitting")

        for epoch in progress_bar:
            loss_value, grads = eqx.filter_value_and_grad(loss)(self.root, data)

            updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(self.root, eqx.is_inexact_array)
            )
            self.root = eqx.apply_updates(self.root, updates)
            progress_bar.set_postfix_str(f"Neg. Avg. LL.: {loss_value}")



class ClassificationCircuit(ProbabilisticCircuit):
    """
    A probabilistic circuit for classification.
    It is assumed that the root layer of the circuit has as many output units as there are classes.
    """

    def as_probabilistic_circuit(self, class_variable: Symbolic,
                                 class_probabilities: jnp.array = None) -> ProbabilisticCircuit:
        """
        Create a full probabilistic circuit from this classification circuit.
        This is done by adding meaning to the sum units of the root layer.
        The first sum unit is the first class in the variables' domain,
        the second sum unit is the second class, and so on.

        :param class_variable: The variable to use for interpretation
        :param class_probabilities: The probabilities of the classes.
        If `None`, the classes are assumed to be uniformly distributed.
        :return: The full probabilistic circuit.
        """


        assert len(class_variable.domain.simple_sets) == self.root.number_of_nodes, \
            "The number of classes must match the number of sum units."

        number_of_classes = self.root.number_of_nodes

        # construct the new variables and figure out which indices to shift
        new_variables = self.variables | SortedSet([class_variable])
        class_variable_index = new_variables.index(class_variable)

        # initialize class probabilities if not given
        if class_probabilities is None:
            class_probabilities = jnp.ones(number_of_classes) / number_of_classes

        copied_root = self.root.__deepcopy__()
        # update variable indices
        for layer in copied_root.all_layers():
            if isinstance(layer, InputLayer):
                updated_variable_indices = jnp.where(layer.variables >= class_variable_index, layer.variables + 1, layer.variables)
                layer.set_variables(updated_variable_indices)
            elif isinstance(layer, InnerLayer):
                layer.reset_variables()
            else:
                raise ValueError(f"Layer {layer} is not supported.")

        # create the new input layer
        distribution_layer = DiscreteLayer(class_variable_index, jnp.log(jnp.eye(number_of_classes)))

        # connect the new input layer with the respective sum units
        edges = jnp.array([jnp.arange(number_of_classes), jnp.arange(number_of_classes)]).flatten()
        sparse_edges = BCOO.fromdense(jnp.ones((2, number_of_classes), dtype=int))
        sparse_edges.data = edges
        product_layer = ProductLayer([copied_root, distribution_layer], sparse_edges)

        # create the new root layer
        root_weights = BCOO.fromdense(jnp.ones((1, number_of_classes), dtype=float))
        root_weights.data = jnp.log(class_probabilities)
        root = SparseSumLayer([product_layer], [root_weights])

        # set the variables again
        for layer in root.all_layers():
            layer.variables # trigger the setter

        return ProbabilisticCircuit(new_variables, root)

    def to_nx(self, progress_bar: bool = True) -> NXProbabilisticCircuit:
        raise NotImplementedError("ClassificationCircuit does not support to_nx. "
                                  "Call 'to_probabilistic_circuit' first.")

    def fit(self, data: jax.Array, labels: jax.Array, epochs: int = 100,
                       optimizer: Optional[optax.GradientTransformation] = None) -> None:
        """
        Fit the circuit to the data using generative training with the cross-entropy as loss.

        :param data: The data.
        :param labels: The labels.
        :param epochs: The number of epochs.
        :param optimizer: The optimizer to use.
        If `None`, the Adam optimizer with a learning rate of 1e-3 is used.
        """

        @eqx.filter_jit
        def loss(p, x, y):
            log_probs = p.log_likelihood_of_nodes(x)
            return -jnp.mean(log_probs[y])

        if optimizer is None:
            optimizer = optax.adam(1e-3)

        opt_state = optimizer.init(eqx.filter(self.root, eqx.is_inexact_array))

        progress_bar = tqdm.tqdm(range(epochs), desc="Fitting")

        for epoch in progress_bar:
            loss_value, grads = eqx.filter_value_and_grad(loss)(self.root, data, labels)

            updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(self.root, eqx.is_inexact_array)
            )
            self.root = eqx.apply_updates(self.root, updates)
            progress_bar.set_postfix_str(f"Cross Entropy: {loss_value}")