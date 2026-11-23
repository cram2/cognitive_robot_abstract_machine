"""
Learning the parameters of a layered circuit with a fixed structure on mini-batches, on
any array library that a subclass supports.
"""

from __future__ import annotations

# %% imports
from abc import ABC, abstractmethod
from dataclasses import dataclass

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood_compute_resources.memory import DeviceMemory
from typing_extensions import Generic, TypeVar

from probabilistic_model.learning.parameters.step_memory import (
    ProbeMeasurement,
    StepMemory,
)
from probabilistic_model.learning.parameters.training_schedule import (
    Convergence,
    ConvergenceHistory,
    LearningRatePlateau,
    MiniBatches,
    MiniBatchTraining,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
)

LayerType = TypeVar("LayerType")
"""
The type of the root layer whose parameters are learned, such as
:class:`~probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base.Layer`.
"""

# %% training runs


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
class CircuitParameterTraining(
    MiniBatchTraining, Generic[LayerType], SubClassSafeGeneric, ABC
):
    """
    Mini-batch training of the parameters of a circuit by a
    :class:`CircuitParameterLearning`.
    """

    learning: CircuitParameterLearning[LayerType]
    """
    The learning whose steps are taken and whose circuit is trained.
    """

    @abstractmethod
    def trained_root(self) -> LayerType:
        """
        :return: The root of a copy of the circuit with the parameters after the last
            step.
        """
        raise NotImplementedError


# %% learning


@dataclass(kw_only=True)
class CircuitParameterLearning(Generic[LayerType], SubClassSafeGeneric, ABC):
    """
    Learns the parameters of the circuit rooted at a layer on mini-batches of training
    arrays, with the array library of its subclass; the structure stays unchanged.
    """

    layer: LayerType
    """
    The root of the circuit to train, which is left unchanged.
    """

    def __post_init__(self) -> None:
        """
        Prepare nothing; a subclass that records something of the circuit before
        training extends it.
        """

    @abstractmethod
    def start_training(
        self,
        *,
        arguments: tuple[BackendArray, ...],
        mini_batches: MiniBatches,
        learning_rate_plateau: LearningRatePlateau | None = None,
    ) -> CircuitParameterTraining[LayerType]:
        """
        :param arguments: The training arrays, one row per training example each.
        :param mini_batches: How the rows are split into the batches of every epoch.
        :param learning_rate_plateau: The schedule of the learning rate, a constant
            learning rate if ``None``; a learner without a learning rate ignores it.
        :return: Training that starts from the parameters of the circuit.
        """
        raise NotImplementedError

    @abstractmethod
    def device_memory(self) -> DeviceMemory:
        """
        :return: The memory of the device the parameters of the circuit are stored on.
        """
        raise NotImplementedError

    def largest_even_batch_size(self, *, rows: int) -> int:
        """
        :param rows: The most rows a batch may have.
        :return: The most rows of at most ``rows`` that a step splits evenly across the
            devices it computes on, ``rows`` itself on one device.
        """
        return rows

    @abstractmethod
    def probe_step_bytes(self, *, batch: tuple[BackendArray, ...]) -> int:
        """
        :param batch: The same rows of every training array.
        :return: The bytes one step on the batch allocates beyond what is in use before
            it; the parameters stay unchanged.
        """
        raise NotImplementedError

    def minimize_until_converged(
        self,
        *arguments: BackendArray,
        convergence: Convergence | None = None,
        mini_batches: MiniBatches | None = None,
        learning_rate_plateau: LearningRatePlateau | None = None,
    ) -> TrainingRun[LayerType]:
        """
        Train on shuffled mini-batches, one step per batch, until the average loss of an
        epoch converges.

        :param arguments: The training arrays, one row per training example each.
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
        :param arguments: The training arrays, one row per training example each.
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
