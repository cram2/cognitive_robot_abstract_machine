"""
Training the parameters of a layered circuit by gradient descent with jax: the loss is a
pure function of the parameter arrays, so that :func:`jax.grad` and :func:`jax.jit`
apply.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass, field
from enum import StrEnum

import jax
import jax.numpy as jnp
import optax
from krrood_compute_resources.jax_memory import JaxDeviceMemory
from typing_extensions import Callable
from krrood_compute_resources.memory import DeviceMemory, HostMemory

from probabilistic_model.learning.exceptions import UnscheduledOptimizerError
from probabilistic_model.learning.gradient_descent import (
    GradientDescent,
    GradientDescentTraining,
    LayerType,
)
from probabilistic_model.learning.training_schedule import (
    LearningRatePlateau,
    MiniBatches,
)

# %% training state


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


# %% gradient descent


@dataclass
class JaxGradientDescent(GradientDescent[LayerType]):
    """
    Gradient descent with jax, one compiled step per batch.

    A circuit with torch parameters is trained the same way by
    :class:`~probabilistic_model.learning.torch_gradient_descent.TorchGradientDescent`.
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
        """
        Default the optimizer and record the parameters before training.
        """
        super().__post_init__()
        if self.optimizer is None:
            self.optimizer = optax.adam(1e-3)
        self.initial_parameters = list(self.layer.circuit_parameters)

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
            jnp.where(zero_probability, -jnp.inf, parameter)
            for zero_probability, parameter in zip(
                self.zero_probability_entries,
                optax.apply_updates(state.parameters, updates),
            )
        ]
        parameters = self.layer.projected_circuit_parameters(updated)
        return TrainingState(parameters, optimizer_state, loss)

    def start_training(
        self,
        *,
        arguments: tuple[jax.Array, ...],
        mini_batches: MiniBatches,
        learning_rate_plateau: LearningRatePlateau | None = None,
    ) -> JaxGradientDescentTraining[LayerType]:
        return JaxGradientDescentTraining(
            arguments=arguments,
            mini_batches=mini_batches,
            gradient_descent=self,
            learning_rate_plateau=learning_rate_plateau,
        )

    def device_memory(self) -> DeviceMemory:
        [device] = self.initial_parameters[0].devices()
        if device.platform == JaxPlatform.CPU:
            return HostMemory()
        return JaxDeviceMemory(device=device)

    def probe_step_bytes(self, *, batch: tuple[jax.Array, ...]) -> int:
        """
        :return: The bytes the step compiled for this batch allocates, as XLA reports
            them: its batch, its temporaries and its new state.
        """
        analysis = (
            jax.jit(self.step).lower(self.initial_state(), *batch).compile()
        ).memory_analysis()
        return (
            sum(argument.nbytes for argument in batch)
            + analysis.temp_size_in_bytes
            + analysis.output_size_in_bytes
            - analysis.alias_size_in_bytes
        )

    def initial_state(self) -> TrainingState:
        """
        :return: The state before the first step: the parameters before training and a
            fresh optimizer state.
        """
        return TrainingState(
            self.initial_parameters,
            self.optimizer.init(self.initial_parameters),
            jnp.full((), jnp.nan, jnp.result_type(*self.initial_parameters)),
        )


# %% training on mini-batches


@dataclass(kw_only=True)
class JaxGradientDescentTraining(GradientDescentTraining[LayerType]):
    """
    Mini-batch training by :class:`JaxGradientDescent`, one compiled step per batch.
    """

    gradient_descent: JaxGradientDescent[LayerType]
    """
    The gradient descent whose steps are taken.
    """

    compiled_step: Callable[..., TrainingState] = field(init=False)
    """
    :meth:`JaxGradientDescent.step`, compiled once for every shape of batch.
    """

    state: TrainingState = field(init=False)
    """
    The state after the last step.
    """

    initial_learning_rate: jax.Array | None = field(init=False, default=None)
    """
    The learning rate before any reduction, ``None`` without a schedule.
    """

    def __post_init__(self) -> None:
        """
        Compile the step and start from the state before training.

        :raises UnscheduledOptimizerError: If a schedule is given for an optimizer whose
            learning rate cannot be changed.
        """
        self.compiled_step = jax.jit(self.gradient_descent.step)
        self.state = self.gradient_descent.initial_state()
        if self.learning_rate_plateau is None:
            return
        if not isinstance(
            self.state.optimizer_state,
            (optax.InjectHyperparamsState, optax.InjectStatefulHyperparamsState),
        ):
            raise UnscheduledOptimizerError()
        self.initial_learning_rate = self.state.optimizer_state.hyperparams[
            "learning_rate"
        ]

    def scale_learning_rate(self, *, scale: float) -> None:
        self.state.optimizer_state.hyperparams["learning_rate"] = (
            self.initial_learning_rate * scale
        )

    def step_on_batch(self, *, batch: tuple[jax.Array, ...]) -> jax.Array:
        self.state = self.compiled_step(self.state, *batch)
        return self.state.loss

    def trained_root(self) -> LayerType:
        return self.gradient_descent.layer.copy_with_circuit_parameters(
            self.state.parameters
        )


# %% device memory


class JaxPlatform(StrEnum):
    """
    The platforms of jax devices that are told apart.
    """

    CPU = "cpu"
