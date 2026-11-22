"""
Training the parameters of a layered circuit by gradient descent with jax: the loss is a
pure function of the parameter arrays, so that :func:`jax.grad` and :func:`jax.jit`
apply.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax
from krrood.patterns.called_positionally import called_positionally
from typing_extensions import Callable

from probabilistic_model.learning.circuit_parameter_learning import LayerType
from probabilistic_model.learning.exceptions import UnscheduledOptimizerError
from probabilistic_model.learning.gradient_descent import (
    GradientDescent,
    GradientDescentTraining,
)
from probabilistic_model.learning.jax_circuit_parameter_learning import (
    BatchSplitDevices,
    JaxCircuitParameterLearning,
)
from probabilistic_model.learning.training_schedule import (
    LearningRatePlateau,
    MiniBatches,
)

# %% training state


@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class JaxGradientDescent(
    GradientDescent[LayerType], JaxCircuitParameterLearning[LayerType]
):
    """
    Gradient descent with jax, one compiled step per batch.

    A circuit with torch parameters is trained the same way by
    :class:`~probabilistic_model.learning.torch_gradient_descent.TorchGradientDescent`.
    """

    optimizer: optax.GradientTransformation | None = None
    """
    The optimizer to use, Adam with a learning rate of 1e-3 if ``None``.
    """

    def __post_init__(self) -> None:
        """
        Default the optimizer and record the parameters before training.
        """
        super().__post_init__()
        if self.optimizer is None:
            self.optimizer = optax.adam(1e-3)

    @called_positionally
    def loss_of_parameters(
        self, parameters: list[jax.Array], *arguments: jax.Array
    ) -> jax.Array:
        """
        :param parameters: Arrays in the order of :attr:`initial_parameters`.
        :param arguments: The arrays passed to :meth:`minimize`.
        :return: The :attr:`loss` of the circuit with these parameter arrays.
        """
        return self.loss.evaluate(
            self.layer.copy_with_circuit_parameters(parameters=parameters), *arguments
        )

    @called_positionally
    def step(self, state: TrainingState, *arguments: jax.Array) -> TrainingState:
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
        parameters = self.layer.projected_circuit_parameters(parameters=updated)
        return TrainingState(
            parameters=parameters, optimizer_state=optimizer_state, loss=loss
        )

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

    def initial_state(self) -> TrainingState:
        """
        :return: The state before the first step: the parameters before training and a
            fresh optimizer state.
        """
        return TrainingState(
            parameters=self.initial_parameters,
            optimizer_state=self.optimizer.init(self.initial_parameters),
            loss=jnp.full((), jnp.nan, jnp.result_type(*self.initial_parameters)),
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

    batch_split_devices: BatchSplitDevices | None = field(init=False, default=None)
    """
    The :attr:`JaxGradientDescent.training_devices` that the rows of a batch are split
    across and the state is copied to, ``None`` if there is only one.
    """

    def __post_init__(self) -> None:
        """
        Compile the step and start from the state before training, copied to every
        training device.

        :raises UnscheduledOptimizerError: If a schedule is given for an optimizer whose
            learning rate cannot be changed.
        """
        self.compiled_step = jax.jit(self.gradient_descent.step)
        self.state = self.gradient_descent.initial_state()
        self.batch_split_devices = BatchSplitDevices.from_devices(
            devices=self.gradient_descent.training_devices
        )
        if self.batch_split_devices is not None:
            self.state = self.batch_split_devices.replicated(state=self.state)
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
        if self.batch_split_devices is not None:
            batch = self.batch_split_devices.placed_batch(batch=batch)
        self.state = self.compiled_step(self.state, *batch)
        return self.state.loss

    def trained_root(self) -> LayerType:
        return self.gradient_descent.layer.copy_with_circuit_parameters(
            parameters=jax.device_put(
                self.state.parameters, self.gradient_descent.parameter_device
            )
        )
