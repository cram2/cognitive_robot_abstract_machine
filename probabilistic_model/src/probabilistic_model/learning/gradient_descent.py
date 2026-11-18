"""
Training the parameters of a layered circuit by gradient descent, with a loss that is a
pure function of the parameter arrays so that :func:`jax.grad` and :func:`jax.jit`
apply.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import functools
from dataclasses import dataclass, field
from enum import StrEnum

import jax
import jax.numpy as jnp
import optax
import tqdm
import array_api_compat
from array_api_compat import array_namespace
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from random_events.variable import Symbolic
from sortedcontainers import SortedSet
from typing_extensions import Callable, Generic, Self, TypeVar

from krrood_compute_hardware.jax_memory import JaxDeviceMemory
from krrood_compute_hardware.memory import DeviceMemory, HostMemory

from probabilistic_model.learning.step_memory import ProbeMeasurement, StepMemory
from probabilistic_model.learning.exceptions import UnscheduledOptimizerError
from probabilistic_model.learning.training_schedule import (
    Convergence,
    ConvergenceHistory,
    LearningRatePlateau,
    MiniBatches,
    PlateauScheduledTraining,
)
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


@dataclass(kw_only=True)
class CrossEntropy(Loss[LayerType]):
    """
    The average negative log-probability of the category of the predicted variable in
    each row given its other values, the discriminative loss of a RAT-SPN classifier
    (Peharz et al., 2019, equation 4) with the prior the circuit holds.
    """

    predicted_column: int
    """
    The column of the predicted variable in the encoded data.
    """

    number_of_categories: int
    """
    The number of categories of the predicted variable, encoded as their positions in
    its domain.
    """

    @classmethod
    def from_predicted_variable(
        cls, *, predicted_variable: Symbolic, variables: SortedSet
    ) -> Self:
        """
        :param predicted_variable: The symbolic variable the circuit classifies into.
        :param variables: The variables of the circuit, which include the predicted
            variable.
        :return: The cross entropy of the predicted variable of the circuit.
        """
        return cls(
            predicted_column=variables.index(predicted_variable),
            number_of_categories=len(predicted_variable.domain.simple_sets),
        )

    def evaluate(self, layer: LayerType, data: BackendArray) -> BackendArray:
        log_likelihoods = layer.log_likelihood_of_nodes(data)[:, 0]
        namespace = array_namespace(log_likelihoods)
        data = namespace.asarray(data, device=array_api_compat.device(log_likelihoods))
        is_predicted_column = (
            namespace.arange(data.shape[1], device=array_api_compat.device(data))
            == self.predicted_column
        )
        log_likelihoods_of_categories = [
            layer.log_likelihood_of_nodes(
                namespace.where(
                    is_predicted_column, namespace.full_like(data, category), data
                )
            )[:, 0]
            for category in range(self.number_of_categories)
        ]
        marginal = functools.reduce(namespace.logaddexp, log_likelihoods_of_categories)
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


@dataclass(kw_only=True)
class TrainingRun(Generic[LayerType], SubClassSafeGeneric):
    """
    A circuit trained until convergence, with the losses of the epochs it took.
    """

    trained_root: LayerType
    """
    The root of the trained circuit.
    """

    history: ConvergenceHistory
    """
    The average loss of every epoch and whether training converged.
    """


@dataclass(kw_only=True)
class GradientDescentTraining(
    PlateauScheduledTraining, Generic[LayerType], SubClassSafeGeneric, ABC
):
    """
    Mini-batch training of the parameters of a circuit by a :class:`GradientDescent`.
    """

    @abstractmethod
    def trained_root(self) -> LayerType:
        """
        :return: The root of a copy of the circuit with the parameters after the last
            step.
        """
        raise NotImplementedError


@dataclass
class GradientDescent(Generic[LayerType], SubClassSafeGeneric, ABC):
    """
    Minimizes a loss of the circuit rooted at a layer by gradient descent, with the
    array library of its subclass.

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

    zero_probability_entries: list[BackendArray] = field(init=False)
    """
    Where every parameter array is ``-inf`` before training, a log-probability or log-
    weight of zero probability, which training keeps.
    """

    def __post_init__(self) -> None:
        """
        Record the :attr:`zero_probability_entries` of the circuit.
        """
        self.zero_probability_entries = [
            array_namespace(parameter).isinf(parameter) & (parameter < 0)
            for parameter in self.layer.circuit_parameters
        ]

    @abstractmethod
    def start_training(
        self,
        *,
        arguments: tuple[BackendArray, ...],
        mini_batches: MiniBatches,
        learning_rate_plateau: LearningRatePlateau | None = None,
    ) -> GradientDescentTraining[LayerType]:
        """
        :param arguments: The arrays the :attr:`loss` takes after the circuit, one row
            per training example each.
        :param mini_batches: How the rows are split into the batches of every epoch.
        :param learning_rate_plateau: The schedule of the learning rate, a constant
            learning rate if ``None``.
        :return: Training that starts from the parameters of the circuit.
        """
        raise NotImplementedError

    @abstractmethod
    def device_memory(self) -> DeviceMemory:
        """
        :return: The memory of the device the parameters of the circuit are stored on.
        """
        raise NotImplementedError

    @abstractmethod
    def probe_step_bytes(self, *, batch: tuple[BackendArray, ...]) -> int:
        """
        :param batch: The same rows of every training array.
        :return: The bytes one step on the batch allocates beyond what is in use before
            it; the parameters stay unchanged.
        """
        raise NotImplementedError

    def minimize(self, epochs: int, *arguments: BackendArray) -> LayerType:
        """
        :param epochs: The number of epochs, one step on all rows each.
        :param arguments: The arrays the :attr:`loss` takes after the circuit.
        :return: The root of the trained circuit.
        """
        training = self.start_training(arguments=arguments, mini_batches=MiniBatches())
        progress_bar = tqdm.tqdm(range(epochs), desc="Fitting")
        for _ in progress_bar:
            loss = training.step_on_batch(batch=arguments)
            progress_bar.set_postfix_str(f"Loss: {loss}")
        return training.trained_root()

    def minimize_until_converged(
        self,
        *arguments: BackendArray,
        convergence: Convergence | None = None,
        mini_batches: MiniBatches | None = None,
        learning_rate_plateau: LearningRatePlateau | None = None,
    ) -> TrainingRun[LayerType]:
        """
        Minimize the :attr:`loss` on shuffled mini-batches, one step per batch, until
        the average loss of an epoch converges.

        :param arguments: The arrays the :attr:`loss` takes after the circuit, one row
            per training example each.
        :param convergence: When training stops, :class:`Convergence` with its defaults
            if ``None``.
        :param mini_batches: How the rows are split into the batches of every epoch,
            :class:`MiniBatches` with its defaults, one batch of all rows, if ``None``.
        :param learning_rate_plateau: The schedule of the learning rate, a constant
            learning rate if ``None``.
        :return: The trained circuit and the losses of its epochs.
        """
        training = self.start_training(
            arguments=arguments,
            mini_batches=MiniBatches() if mini_batches is None else mini_batches,
            learning_rate_plateau=learning_rate_plateau,
        )
        history = (
            Convergence() if convergence is None else convergence
        ).history_until_converged(training=training)
        return TrainingRun(trained_root=training.trained_root(), history=history)

    def step_memory(self, *arguments: BackendArray, probe_rows: int = 64) -> StepMemory:
        """
        :param arguments: The arrays the :attr:`loss` takes after the circuit, one row
            per training example each.
        :param probe_rows: The rows of the smaller of the two batches a step is measured
            on; the larger has twice as many.
        :return: The memory one step allocates.
        """
        smaller_rows = max(1, min(probe_rows, arguments[0].shape[0] // 2))
        return StepMemory.from_two_probes(
            smaller=ProbeMeasurement(
                rows=smaller_rows,
                step_bytes=self.probe_step_bytes(
                    batch=tuple(argument[:smaller_rows] for argument in arguments)
                ),
            ),
            larger=ProbeMeasurement(
                rows=2 * smaller_rows,
                step_bytes=self.probe_step_bytes(
                    batch=tuple(argument[: 2 * smaller_rows] for argument in arguments)
                ),
            ),
        )


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

    def step_on_batch(self, *, batch: tuple[jax.Array, ...]) -> float:
        self.state = self.compiled_step(self.state, *batch)
        return float(self.state.loss)

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
