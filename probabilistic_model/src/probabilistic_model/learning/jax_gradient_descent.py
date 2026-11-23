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
import numpy as np
import optax

from probabilistic_model.learning.circuit_parameter_learning import LayerType
from probabilistic_model.learning.exceptions import UnscheduledOptimizerError
from probabilistic_model.learning.gradient_descent import (
    GradientDescent,
    GradientDescentTraining,
)
from probabilistic_model.learning.jax_circuit_parameter_learning import (
    CompiledStep,
    JaxCircuitParameterLearning,
    JaxCircuitParameterTraining,
    JaxTrainingState,
)
from probabilistic_model.learning.training_schedule import (
    LearningRatePlateau,
    MiniBatches,
)

# %% training state


@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class TrainingState(JaxTrainingState):
    """
    The values one compiled gradient descent step reads and returns.
    """

    optimizer_state: optax.OptState
    """
    The state of the optimizer after the last step.
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

    def loss_of_parameters(
        self, *, parameters: list[jax.Array], arguments: tuple[jax.Array, ...]
    ) -> jax.Array:
        """
        :param parameters: Arrays in the order of :attr:`initial_parameters`.
        :param arguments: The arrays the :attr:`loss` takes after the circuit.
        :return: The :attr:`loss` of the circuit with these parameter arrays.
        """
        return self.loss.evaluate(
            self.layer.copy_with_circuit_parameters(parameters=parameters), *arguments
        )

    def step(
        self, *, state: TrainingState, batch: tuple[jax.Array, ...]
    ) -> TrainingState:
        loss, gradients = jax.value_and_grad(
            lambda parameters: self.loss_of_parameters(
                parameters=parameters, arguments=batch
            )
        )(state.parameters)
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
            learning=self,
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
            loss=self.loss_before_any_step(),
        )


# %% training on mini-batches


@dataclass(kw_only=True)
class JaxGradientDescentTraining(
    GradientDescentTraining[LayerType], JaxCircuitParameterTraining[LayerType]
):
    """
    Mini-batch training by :class:`JaxGradientDescent`, one compiled step per batch.
    """

    learning: JaxGradientDescent[LayerType]
    """
    The gradient descent whose steps are taken.
    """

    compiled_step: CompiledStep = field(init=False)
    """
    :meth:`JaxGradientDescent.step`, compiled.
    """

    initial_learning_rate: jax.Array | None = field(init=False, default=None)
    """
    The learning rate before any reduction, ``None`` without a schedule.
    """

    def __post_init__(self) -> None:
        """
        Compile the step.

        :raises UnscheduledOptimizerError: If a schedule is given for an optimizer whose
            learning rate cannot be changed.
        """
        super().__post_init__()
        self.compiled_step = CompiledStep.from_step(step=self.learning.step)
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
        return self.take_compiled_step(step=self.compiled_step, batch=batch)

    def step_on_rows(self, *, rows: np.ndarray) -> jax.Array:
        """
        On a single training device, the rows are gathered by the compiled step itself,
        see :meth:`MiniBatchTraining.step_on_rows`.
        """
        return self.take_compiled_step_on_rows(step=self.compiled_step, rows=rows)
