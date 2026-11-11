from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import Self, Any

import optax
from jax.experimental.sparse import BCOO
from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json
from random_events.variable import Symbolic
from sortedcontainers import SortedSet

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

    def to_json(self) -> dict[str, Any]:
        result = super().to_json()
        result["variables"] = [to_json(variable) for variable in self.variables]
        result["root"] = self.root.to_json()
        return result

    @classmethod
    def _from_json(cls, *, data: dict[str, Any], **kwargs) -> Self:
        variables = SortedSet(
            from_json(variable, **kwargs) for variable in data["variables"]
        )
        root = DifferentiableLayer.from_json(data=data["root"], **kwargs)
        return cls(variables, root)

    def fit(
        self,
        data: jax.Array,
        epochs: int = 100,
        optimizer: optax.GradientTransformation | None = None,
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
        self.root = LikelihoodMaximization(self.root, optimizer).minimize(epochs, data)


@dataclass
class ClassificationCircuit(DifferentiableLayeredCircuit):
    """
    A probabilistic circuit for classification.

    It is assumed that the root layer of the circuit has as many output units as there
    are classes.
    """

    def as_probabilistic_circuit(
        self, class_variable: Symbolic, class_probabilities: jax.Array | None = None
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
        assert len(class_variable.domain.simple_sets) == self.root.number_of_nodes, (
            "The number of classes must match the number of sum units."
        )

        number_of_classes = self.root.number_of_nodes

        # construct the new variables and figure out which indices to shift
        new_variables = self.variables | SortedSet([class_variable])
        class_variable_index = new_variables.index(class_variable)

        # initialize class probabilities if not given
        if class_probabilities is None:
            class_probabilities = jnp.ones(number_of_classes) / number_of_classes

        copied_root = copy.deepcopy(self.root)
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
        optimizer: optax.GradientTransformation | None = None,
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
        self.root = ClassConditionalLikelihoodMaximization(
            self.root, optimizer
        ).minimize(epochs, data, labels)


# %% gradient descent


class TrainingState(eqx.Module):
    """
    The values one compiled training step reads and returns.
    """

    parameters: list[jax.Array]
    """
    The trainable arrays of the layer, flattened by
    :attr:`GradientDescent.parameter_structure`.
    """

    optimizer_state: optax.OptState
    """
    The state of the optimizer after the last step.
    """

    loss: jax.Array
    """
    The loss of the parameters before the last step.
    """


@dataclass
class GradientDescent(ABC):
    """
    Minimizes a loss of a layer by gradient descent, one compiled step per epoch.

    The optimizer receives the trainable arrays of the layer as a flat list.
    """

    layer: DifferentiableLayer
    """
    The layer to train, which is left unchanged.
    """

    optimizer: optax.GradientTransformation | None = None
    """
    The optimizer to use.

    If `None`, the Adam optimizer with a learning rate of 1e-3 is used.
    """

    fixed_part: DifferentiableLayer = field(init=False)
    """
    The layer without its trainable arrays.
    """

    parameter_structure: jax.tree_util.PyTreeDef = field(init=False)
    """
    The tree structure of the trainable arrays of the layer.
    """

    initial_parameters: list[jax.Array] = field(init=False)
    """
    The trainable arrays of the layer before training.
    """

    def __post_init__(self) -> None:
        if self.optimizer is None:
            self.optimizer = optax.adam(1e-3)
        # Static fields of the layers hold arrays, which pytree comparisons inside the
        # optimizer cannot handle, so training works on the flat list of trainable arrays.
        parameters, self.fixed_part = eqx.partition(self.layer, eqx.is_inexact_array)
        self.initial_parameters, self.parameter_structure = jax.tree_util.tree_flatten(
            parameters
        )

    @abstractmethod
    def loss(self, layer: DifferentiableLayer, *arguments: jax.Array) -> jax.Array:
        """
        :param layer: The layer to evaluate.
        :param arguments: The arrays passed to :meth:`minimize`.
        :return: The scalar loss of the layer on the arguments.
        """

    def layer_from_parameters(self, parameters: list[jax.Array]) -> DifferentiableLayer:
        """
        :param parameters: Trainable arrays in the order of :attr:`initial_parameters`.
        :return: The layer with its trainable arrays replaced by the parameters.
        """
        return eqx.combine(
            jax.tree_util.tree_unflatten(self.parameter_structure, parameters),
            self.fixed_part,
        )

    def loss_of_parameters(
        self, parameters: list[jax.Array], *arguments: jax.Array
    ) -> jax.Array:
        """
        :param parameters: Trainable arrays in the order of :attr:`initial_parameters`.
        :param arguments: The arrays passed to :meth:`minimize`.
        :return: The :meth:`loss` of the layer with the given trainable arrays.
        """
        return self.loss(self.layer_from_parameters(parameters), *arguments)

    def step(self, state: TrainingState, *arguments: jax.Array) -> TrainingState:
        """
        :param state: The state before the step.
        :param arguments: The arrays passed to :meth:`minimize`.
        :return: The state after one update of the optimizer.
        """
        loss, gradients = jax.value_and_grad(self.loss_of_parameters)(
            state.parameters, *arguments
        )
        updates, optimizer_state = self.optimizer.update(
            gradients, state.optimizer_state, state.parameters
        )
        return TrainingState(
            optax.apply_updates(state.parameters, updates), optimizer_state, loss
        )

    def minimize(self, epochs: int, *arguments: jax.Array) -> DifferentiableLayer:
        """
        :param epochs: The number of epochs.
        :param arguments: The arrays :meth:`loss` takes after the layer.
        :return: The trained layer.
        """
        compiled_step = jax.jit(self.step)
        state = TrainingState(
            self.initial_parameters,
            self.optimizer.init(self.initial_parameters),
            jnp.full((), jnp.nan, jnp.result_type(*self.initial_parameters)),
        )
        progress_bar = tqdm.tqdm(range(epochs), desc="Fitting")
        for _ in progress_bar:
            state = compiled_step(state, *arguments)
            progress_bar.set_postfix_str(f"Loss: {state.loss}")
        return self.layer_from_parameters(state.parameters)


@dataclass
class LikelihoodMaximization(GradientDescent):
    """
    Maximizes the average log-likelihood of the data under the nodes of the layer.
    """

    def loss(self, layer: DifferentiableLayer, data: jax.Array) -> jax.Array:
        return -jnp.mean(layer.log_likelihood_of_nodes(data))


@dataclass
class ClassConditionalLikelihoodMaximization(GradientDescent):
    """
    Maximizes the average log-likelihood of each row under the node of its class.
    """

    def loss(
        self, layer: DifferentiableLayer, data: jax.Array, labels: jax.Array
    ) -> jax.Array:
        return -jnp.mean(
            layer.log_likelihood_of_nodes(data)[jnp.arange(len(labels)), labels]
        )
