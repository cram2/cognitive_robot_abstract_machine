from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import optax
from jax.experimental.sparse import BCOO
from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json
from random_events.variable import Symbolic
from sortedcontainers import SortedSet
from typing_extensions import Self, Optional

from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    DifferentiableProductLayer,
    DifferentiableSparseSumLayer,
    DifferentiableInputLayer,
    DifferentiableInnerLayer,
    DifferentiableLayer,
)
from probabilistic_model.probabilistic_circuit.jax.discrete_layer import (
    DifferentiableDiscreteLayer,
)
import jax
import tqdm
import jax.numpy as jnp
import equinox as eqx


@dataclass
class DifferentiableLayeredCircuit(SubclassJSONSerializer):
    """
    A layered probabilistic circuit whose log-likelihood is differentiable in its
    parameters, for learning them by gradient descent.

    Only the log-likelihood, the loss of training, is computed here. Every other query
    is answered by the layered circuits of the ``tensorized`` package, which the
    ``jax_tensorized`` adapters convert this circuit into and back.
    """

    variables: SortedSet
    """
    The variables of the circuit.
    """

    root: DifferentiableLayer
    """
    The root layer of the circuit.
    """

    def log_likelihood(self, x: jax.Array) -> jax.Array:
        return self.root.log_likelihood_of_nodes(x)[:, 0]

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["variables"] = [to_json(variable) for variable in self.variables]
        result["root"] = self.root.to_json()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        variables = SortedSet(
            from_json(variable, **kwargs) for variable in data["variables"]
        )
        root = DifferentiableLayer.from_json(data["root"], **kwargs)
        return cls(variables, root)

    def fit(
        self,
        data: jax.Array,
        epochs: int = 100,
        optimizer: Optional[optax.GradientTransformation] = None,
        **kwargs,
    ) -> None:
        """
        Fit the circuit to the data using generative training with the negative average
        log-likelihood as loss.

        :param data: The data.
        :param epochs: The number of epochs.
        :param optimizer: The optimizer to use. If `None`, the Adam optimizer with a
            learning rate of 1e-3 is used.
        """

        @eqx.filter_jit
        def loss(root: DifferentiableLayer, x: jax.Array) -> jax.Array:
            return -jnp.mean(root.log_likelihood_of_nodes(x))

        if optimizer is None:
            optimizer = optax.adam(1e-3)

        optimizer_state = optimizer.init(eqx.filter(self.root, eqx.is_inexact_array))

        progress_bar = tqdm.tqdm(range(epochs), desc="Fitting")

        for _ in progress_bar:
            loss_value, gradients = eqx.filter_value_and_grad(loss)(self.root, data)

            updates, optimizer_state = optimizer.update(
                gradients, optimizer_state, eqx.filter(self.root, eqx.is_inexact_array)
            )
            self.root = eqx.apply_updates(self.root, updates)
            progress_bar.set_postfix_str(
                f"Negative average log-likelihood: {loss_value}"
            )


@dataclass
class ClassificationCircuit(DifferentiableLayeredCircuit):
    """
    A probabilistic circuit for classification.

    It is assumed that the root layer of the circuit has as many output units as there
    are classes.
    """

    def as_probabilistic_circuit(
        self, class_variable: Symbolic, class_probabilities: Optional[jax.Array] = None
    ) -> DifferentiableLayeredCircuit:
        """
        Create a full probabilistic circuit from this classification circuit.

        This is done by adding meaning to the sum units of the root layer. The first sum
        unit is the first class in the variables' domain, the second sum unit is the
        second class, and so on.

        :param class_variable: The variable to use for interpretation
        :param class_probabilities: The probabilities of the classes. If `None`, the
            classes are assumed to be uniformly distributed.
        :return: The full probabilistic circuit.
        """
        assert (
            len(class_variable.domain.simple_sets) == self.root.number_of_nodes
        ), "The number of classes must match the number of sum units."

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
            if isinstance(layer, DifferentiableInputLayer):
                updated_variable_indices = jnp.where(
                    layer.variables >= class_variable_index,
                    layer.variables + 1,
                    layer.variables,
                )
                layer.set_variables(updated_variable_indices)
            elif isinstance(layer, DifferentiableInnerLayer):
                layer.reset_variables()
            else:
                raise ValueError(f"Layer {layer} is not supported.")

        # create the new input layer
        distribution_layer = DifferentiableDiscreteLayer(
            class_variable_index, jnp.log(jnp.eye(number_of_classes))
        )

        # connect the new input layer with the respective sum units
        edges = jnp.array(
            [jnp.arange(number_of_classes), jnp.arange(number_of_classes)]
        ).flatten()
        sparse_edges = BCOO.fromdense(jnp.ones((2, number_of_classes), dtype=int))
        sparse_edges.data = edges
        product_layer = DifferentiableProductLayer(
            [copied_root, distribution_layer], sparse_edges
        )

        # create the new root layer
        root_weights = BCOO.fromdense(jnp.ones((1, number_of_classes), dtype=float))
        root_weights.data = jnp.log(class_probabilities)
        root = DifferentiableSparseSumLayer([product_layer], [root_weights])

        # set the variables again
        for layer in root.all_layers():
            layer.variables  # trigger the setter

        return DifferentiableLayeredCircuit(new_variables, root)

    def fit(
        self,
        data: jax.Array,
        labels: jax.Array,
        epochs: int = 100,
        optimizer: Optional[optax.GradientTransformation] = None,
    ) -> None:
        """
        Fit the circuit to the data using generative training with the cross-entropy as
        loss.

        :param data: The data.
        :param labels: The labels.
        :param epochs: The number of epochs.
        :param optimizer: The optimizer to use. If `None`, the Adam optimizer with a
            learning rate of 1e-3 is used.
        """

        @eqx.filter_jit
        def loss(root: DifferentiableLayer, x: jax.Array, y: jax.Array) -> jax.Array:
            return -jnp.mean(root.log_likelihood_of_nodes(x)[y])

        if optimizer is None:
            optimizer = optax.adam(1e-3)

        optimizer_state = optimizer.init(eqx.filter(self.root, eqx.is_inexact_array))

        progress_bar = tqdm.tqdm(range(epochs), desc="Fitting")

        for _ in progress_bar:
            loss_value, gradients = eqx.filter_value_and_grad(loss)(
                self.root, data, labels
            )

            updates, optimizer_state = optimizer.update(
                gradients, optimizer_state, eqx.filter(self.root, eqx.is_inexact_array)
            )
            self.root = eqx.apply_updates(self.root, updates)
            progress_bar.set_postfix_str(f"Cross Entropy: {loss_value}")
