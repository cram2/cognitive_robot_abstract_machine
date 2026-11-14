"""
Training the parameters of a layered circuit by gradient descent, with a loss that is a
pure function of the parameter arrays so that :func:`jax.grad` and :func:`jax.jit`
apply.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import functools
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax
import tqdm
import array_api_compat
from array_api_compat import array_namespace
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from random_events.variable import Symbolic
from sortedcontainers import SortedSet
from typing_extensions import Generic, Self, TypeVar

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
)

LayerType = TypeVar("LayerType")
"""
The type of the root layer that gradient descent trains, such as
:class:`~probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base.Layer`.
"""

# %% losses


@dataclass
class Loss(Generic[LayerType], SubClassSafeGeneric, ABC):
    """
    A scalar loss of the circuit rooted at a layer, computed on any array backend.
    """

    @abstractmethod
    def evaluate(self, layer: LayerType, *arguments: BackendArray) -> BackendArray:
        """
        :param layer: The root of the circuit to evaluate.
        :param arguments: The arrays the loss is computed on, such as data and labels.
        :return: The scalar loss of the circuit on the arguments.
        """
        raise NotImplementedError


@dataclass
class AverageNegativeLogLikelihood(Loss[LayerType]):
    """
    The negative average log-likelihood of the data under the nodes of the root.
    """

    def evaluate(self, layer: LayerType, data: BackendArray) -> BackendArray:
        log_likelihoods = layer.log_likelihood_of_nodes(data)
        return -array_namespace(log_likelihoods).mean(log_likelihoods)


@dataclass
class ClassConditionalAverageNegativeLogLikelihood(Loss[LayerType]):
    """
    The negative average log-likelihood of each row under the node of its class.
    """

    def evaluate(
        self, layer: LayerType, data: BackendArray, labels: BackendArray
    ) -> BackendArray:
        log_likelihoods = layer.log_likelihood_of_nodes(data)
        namespace = array_namespace(log_likelihoods)
        rows = namespace.arange(
            labels.shape[0], device=array_api_compat.device(log_likelihoods)
        )
        return -namespace.mean(log_likelihoods[rows, labels])


@dataclass(kw_only=True)
class CrossEntropy(Loss[LayerType]):
    """
    The average negative log-probability of the class of each row given its other
    values, by Bayes' rule over the joint likelihoods of the row with every class.

    This is the discriminative loss of a RAT-SPN classifier (Peharz et al., 2019,
    equation 4), with the class prior the circuit holds.
    """

    class_column: int
    """
    The column of the class variable in the encoded data.
    """

    number_of_classes: int
    """
    The number of classes, encoded as their positions in the domain of the class
    variable.
    """

    @classmethod
    def from_class_variable(
        cls, *, class_variable: Symbolic, variables: SortedSet
    ) -> Self:
        """
        :param class_variable: The variable whose domain elements are the classes.
        :param variables: The variables of the circuit, which include the class
            variable.
        :return: The cross entropy of the class variable of the circuit.
        """
        return cls(
            class_column=variables.index(class_variable),
            number_of_classes=len(class_variable.domain.simple_sets),
        )

    def evaluate(self, layer: LayerType, data: BackendArray) -> BackendArray:
        log_likelihoods = layer.log_likelihood_of_nodes(data)[:, 0]
        namespace = array_namespace(log_likelihoods)
        data = namespace.asarray(data, device=array_api_compat.device(log_likelihoods))
        is_class_column = (
            namespace.arange(data.shape[1], device=array_api_compat.device(data))
            == self.class_column
        )
        log_likelihoods_of_classes = [
            layer.log_likelihood_of_nodes(
                namespace.where(is_class_column, namespace.full_like(data, label), data)
            )[:, 0]
            for label in range(self.number_of_classes)
        ]
        marginal = functools.reduce(namespace.logaddexp, log_likelihoods_of_classes)
        return namespace.mean(marginal - log_likelihoods)


# %% gradient descent


@jax.tree_util.register_dataclass
@dataclass
class TrainingState:
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
class GradientDescent(Generic[LayerType], SubClassSafeGeneric):
    """
    Minimizes a loss of the circuit rooted at a layer by gradient descent with jax, one
    compiled step per epoch. A circuit with torch parameters is trained the same way,
    with the same losses, by
    :class:`~probabilistic_model.learning.torch_gradient_descent.TorchGradientDescent`.

    The layer gives its parameters as the flat list ``circuit_parameters``, projects
    such a list onto valid parameters with ``projected_circuit_parameters`` and rebuilds
    itself from it with ``copy_with_circuit_parameters``; the optimizer works on that
    list.
    """

    layer: LayerType
    """
    The root of the circuit to train, which is left unchanged.
    """

    loss: Loss[LayerType]
    """
    The loss to minimize.
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

    impossible: list[jax.Array] = field(init=False)
    """
    Where every parameter array is ``-inf`` before training, a log-probability or log-
    weight of zero probability, which training keeps.
    """

    def __post_init__(self) -> None:
        if self.optimizer is None:
            self.optimizer = optax.adam(1e-3)
        self.initial_parameters = list(self.layer.circuit_parameters)
        self.impossible = [
            jnp.isneginf(parameter) for parameter in self.initial_parameters
        ]

    def loss_of_parameters(
        self, parameters: list[jax.Array], *arguments: jax.Array
    ) -> jax.Array:
        """
        :param parameters: Arrays in the order of :attr:`initial_parameters`.
        :param arguments: The arrays passed to :meth:`minimize`.
        :return: The :attr:`loss` of the circuit with these parameter arrays.
        """
        return self.loss.evaluate(
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
        updated = [
            jnp.where(impossible, -jnp.inf, parameter)
            for impossible, parameter in zip(
                self.impossible, optax.apply_updates(state.parameters, updates)
            )
        ]
        parameters = self.layer.projected_circuit_parameters(updated)
        return TrainingState(parameters, optimizer_state, loss)

    def minimize(self, epochs: int, *arguments: jax.Array) -> LayerType:
        """
        :param epochs: The number of epochs.
        :param arguments: The arrays the :attr:`loss` takes after the circuit.
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
