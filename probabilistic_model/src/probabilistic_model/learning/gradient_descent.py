"""
Training the parameters of a layered circuit by gradient descent, with a loss that is a
pure function of the parameter arrays so that :func:`jax.grad` and :func:`jax.jit`
apply.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tqdm
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from typing_extensions import Generic, TypeVar

LayerType = TypeVar("LayerType")
"""
The type of the root layer that gradient descent trains, such as
:class:`~probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base.Layer`.
"""

# %% gradient descent


class TrainingState(eqx.Module):
    """
    The values one compiled training step reads and returns.
    """

    parameters: list[jax.Array]
    """
    The parameter arrays of the circuit, in the order of its ``circuit_parameters``.
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
class GradientDescent(Generic[LayerType], SubClassSafeGeneric, ABC):
    """
    Minimizes a loss of the circuit rooted at a layer by gradient descent, one compiled
    step per epoch.

    The layer gives its parameters as the flat list ``circuit_parameters``, projects
    such a list onto valid parameters with ``projected_circuit_parameters`` and rebuilds
    itself from it with ``copy_with_circuit_parameters``; the optimizer works on that
    list.
    """

    layer: LayerType
    """
    The root of the circuit to train, which is left unchanged.
    """

    optimizer: optax.GradientTransformation | None = None
    """
    The optimizer to use.

    If `None`, the Adam optimizer with a learning rate of 1e-3 is used.
    """

    initial_parameters: list[jax.Array] = field(init=False)
    """
    The parameter arrays of the circuit before training.
    """

    def __post_init__(self) -> None:
        if self.optimizer is None:
            self.optimizer = optax.adam(1e-3)
        self.initial_parameters = list(self.layer.circuit_parameters)

    @abstractmethod
    def loss(self, layer: LayerType, *arguments: jax.Array) -> jax.Array:
        """
        :param layer: The root of the circuit to evaluate.
        :param arguments: The arrays passed to :meth:`minimize`.
        :return: The scalar loss of the circuit on the arguments.
        """

    def loss_of_parameters(
        self, parameters: list[jax.Array], *arguments: jax.Array
    ) -> jax.Array:
        """
        :param parameters: Arrays in the order of :attr:`initial_parameters`.
        :param arguments: The arrays passed to :meth:`minimize`.
        :return: The :meth:`loss` of the circuit with these parameter arrays.
        """
        return self.loss(
            self.layer.copy_with_circuit_parameters(parameters), *arguments
        )

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
        parameters = self.layer.projected_circuit_parameters(
            optax.apply_updates(state.parameters, updates)
        )
        return TrainingState(parameters, optimizer_state, loss)

    def minimize(self, epochs: int, *arguments: jax.Array) -> LayerType:
        """
        :param epochs: The number of epochs.
        :param arguments: The arrays :meth:`loss` takes after the circuit.
        :return: The root of the trained circuit.
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
        return self.layer.copy_with_circuit_parameters(state.parameters)


# %% losses


@dataclass
class LikelihoodMaximization(GradientDescent[LayerType]):
    """
    Maximizes the average log-likelihood of the data under the nodes of the root.
    """

    def loss(self, layer: LayerType, data: jax.Array) -> jax.Array:
        return -jnp.mean(layer.log_likelihood_of_nodes(data))


@dataclass
class ClassConditionalLikelihoodMaximization(GradientDescent[LayerType]):
    """
    Maximizes the average log-likelihood of each row under the node of its class.
    """

    def loss(self, layer: LayerType, data: jax.Array, labels: jax.Array) -> jax.Array:
        return -jnp.mean(
            layer.log_likelihood_of_nodes(data)[jnp.arange(len(labels)), labels]
        )
