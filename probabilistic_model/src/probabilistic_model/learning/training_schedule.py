"""
When gradient descent stops and which rows each of its steps sees, independent of the
array backend that trains.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
from array_api_compat import array_namespace, device
from typing_extensions import Iterator, Self

from krrood_compute_resources.memory import DeviceMemory

from probabilistic_model.learning.step_memory import StepMemory
from probabilistic_model.learning.exceptions import NotEnoughMemoryError

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
)

# %% convergence


@dataclass(kw_only=True)
class ConvergenceHistory:
    """
    The average loss of every epoch a training ran, and why it stopped.
    """

    epoch_losses: list[float]
    """
    The average loss over the rows of each epoch, in the order they ran.
    """

    converged: bool
    """
    Whether training stopped because the loss converged rather than at a limit.
    """

    duration: timedelta = timedelta()
    """
    The wall-clock time of the epochs.
    """

    epoch_durations: list[timedelta] = field(default_factory=list)
    """
    The wall-clock time of each epoch, in the order they ran.
    """

    @property
    def epochs(self) -> int:
        """
        :return: The number of epochs training ran.
        """
        return len(self.epoch_losses)

    @property
    def average_epoch_duration(self) -> timedelta | None:
        """
        :return: The mean duration of the epochs after the first, which also compiles
            the training step, the first epoch's alone if it is the only one, ``None``
            without epochs.
        """
        if not self.epoch_durations:
            return None
        durations = self.epoch_durations[1:] or self.epoch_durations
        return sum(durations, timedelta()) / len(durations)


@dataclass(kw_only=True)
class Convergence:
    """
    Stops training once the average loss of an epoch improved by less than
    :attr:`tolerance` for :attr:`patience` epochs in a row, or after
    :attr:`maximum_epochs` or :attr:`maximum_duration`.
    """

    tolerance: float = 1e-3
    """
    The improvement of the average loss below which an epoch counts as not improving.
    """

    patience: int = 3
    """
    The number of epochs in a row that must improve by less than :attr:`tolerance`.
    """

    maximum_epochs: int = 1000
    """
    The number of epochs after which training stops without having converged.
    """

    maximum_duration: timedelta | None = None
    """
    The wall-clock time after which no further epoch starts, no limit if ``None``.
    """

    def has_converged(self, *, epoch_losses: list[float]) -> bool:
        """
        :param epoch_losses: The average loss of every epoch so far.
        :return: Whether each of the last :attr:`patience` epochs improved the loss by
            less than :attr:`tolerance`.
        """
        if len(epoch_losses) <= self.patience:
            return False
        recent = epoch_losses[-(self.patience + 1) :]
        return all(
            earlier - later < self.tolerance
            for earlier, later in zip(recent, recent[1:])
        )

    def history_until_converged(
        self, *, training: MiniBatchTraining
    ) -> ConvergenceHistory:
        """
        :param training: The training to run epoch by epoch.
        :return: The losses of the epochs trained until convergence or the epoch limit.
        """
        epoch_losses: list[float] = []
        epoch_durations: list[timedelta] = []
        start = time.perf_counter()

        def elapsed() -> timedelta:
            """
            :return: The time since training started.
            """
            return timedelta(seconds=time.perf_counter() - start)

        while len(epoch_losses) < self.maximum_epochs and not self.is_out_of_time(
            elapsed=elapsed()
        ):
            epoch_start = time.perf_counter()
            epoch_losses.append(training.train_one_epoch())
            training.finish_epoch(average_loss=epoch_losses[-1])
            epoch_durations.append(timedelta(seconds=time.perf_counter() - epoch_start))
            if training.has_settled() and self.has_converged(epoch_losses=epoch_losses):
                return ConvergenceHistory(
                    epoch_losses=epoch_losses,
                    converged=True,
                    duration=elapsed(),
                    epoch_durations=epoch_durations,
                )
        return ConvergenceHistory(
            epoch_losses=epoch_losses,
            converged=False,
            duration=elapsed(),
            epoch_durations=epoch_durations,
        )

    def is_out_of_time(self, *, elapsed: timedelta) -> bool:
        """
        :param elapsed: The time training has taken so far.
        :return: Whether :attr:`maximum_duration` has passed.
        """
        return self.maximum_duration is not None and elapsed >= self.maximum_duration


# %% learning rate schedule


@dataclass(kw_only=True)
class LearningRatePlateau:
    """
    Divides the learning rate by :attr:`factor` whenever the average loss of an epoch
    has not improved relatively by :attr:`relative_tolerance` on the best loss for
    :attr:`patience` epochs, as :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`
    does.
    """

    factor: float = 0.1
    """
    The factor the learning rate is multiplied by at every reduction.
    """

    patience: int = 10
    """
    The number of epochs without improvement after which the learning rate is reduced.
    """

    relative_tolerance: float = 1e-4
    """
    The improvement on the best loss, relative to its magnitude, below which an epoch
    counts as not improving.
    """

    reductions_until_settled: int = 3
    """
    The number of reductions after which training may stop.
    """

    best_loss: float = field(default=math.inf, init=False)
    """
    The lowest average loss of an epoch so far.
    """

    epochs_without_improvement: int = field(default=0, init=False)
    """
    The number of epochs since the best loss, or since the last reduction.
    """

    reductions: int = field(default=0, init=False)
    """
    The number of reductions so far.
    """

    @property
    def scale(self) -> float:
        """
        :return: The factor the initial learning rate is multiplied by now.
        """
        return self.factor**self.reductions

    @property
    def is_settled(self) -> bool:
        """
        :return: Whether the learning rate was reduced often enough for training to
            stop.
        """
        return self.reductions >= self.reductions_until_settled

    def record(self, *, average_loss: float) -> None:
        """
        Count an epoch, reducing the learning rate if the loss stopped improving.

        :param average_loss: The average loss of the epoch.
        """
        if math.isinf(
            self.best_loss
        ) or self.best_loss - average_loss > self.relative_tolerance * abs(
            self.best_loss
        ):
            self.best_loss = average_loss
            self.epochs_without_improvement = 0
            return
        self.epochs_without_improvement += 1
        if self.epochs_without_improvement > self.patience:
            self.reductions += 1
            self.epochs_without_improvement = 0


# %% mini-batches


@dataclass(kw_only=True)
class MiniBatches:
    """
    Splits the rows of the training arrays into shuffled mini-batches, anew every epoch.
    """

    batch_size: int | None = None
    """
    The number of rows of every batch but the last, all rows if ``None``.
    """

    seed: int = 69
    """
    The seed of the shuffling.
    """

    random_number_generator: np.random.Generator = field(init=False, repr=False)
    """
    The generator that shuffles the rows of every epoch.
    """

    def __post_init__(self) -> None:
        """
        Seed :attr:`random_number_generator` with :attr:`seed`.
        """
        self.random_number_generator = np.random.default_rng(self.seed)

    @classmethod
    def fitting_in_memory(
        cls, *, memory: DeviceMemory, step_memory: StepMemory, seed: int = 69
    ) -> Self:
        """
        :param memory: The memory of the device that trains.
        :param step_memory: The memory one training step allocates.
        :param seed: The seed of the shuffling.
        :return: Mini-batches of the most rows whose steps fit in the free memory.
        :raises NotEnoughMemoryError: If not even a step on one row fits.
        """
        free_bytes = memory.free_bytes()
        batch_size = step_memory.rows_fitting_in(free_bytes=free_bytes)
        if batch_size < 1:
            raise NotEnoughMemoryError(
                free_bytes=free_bytes,
                fixed_bytes=step_memory.fixed_bytes,
                bytes_per_row=step_memory.bytes_per_row,
            )
        return cls(batch_size=batch_size, seed=seed)

    def row_indices_of_epoch(self, *, number_of_rows: int) -> Iterator[np.ndarray]:
        """
        :param number_of_rows: The number of rows of every training array.
        :return: The indices of the rows of every batch of one epoch, together every
            row once.
        """
        order = self.random_number_generator.permutation(number_of_rows)
        step = number_of_rows if self.batch_size is None else self.batch_size
        for start in range(0, number_of_rows, step):
            yield order[start : start + step]

    def batches_of_epoch(
        self, *, arguments: tuple[BackendArray, ...]
    ) -> Iterator[tuple[BackendArray, ...]]:
        """
        :param arguments: The training arrays, one row per training example each.
        :return: The batches of one epoch, each holding the same rows of every array.
        """
        for rows in self.row_indices_of_epoch(number_of_rows=arguments[0].shape[0]):
            yield self.selected_rows(arguments=arguments, rows=rows)

    @staticmethod
    def selected_rows(
        *, arguments: tuple[BackendArray, ...], rows: np.ndarray
    ) -> tuple[BackendArray, ...]:
        """
        :param arguments: The training arrays, one row per training example each.
        :param rows: The indices of some rows.
        :return: Those rows of every array, in the order of the indices.
        """
        return tuple(
            array_namespace(argument).take(
                argument,
                array_namespace(argument).asarray(rows, device=device(argument)),
                axis=0,
            )
            for argument in arguments
        )


# %% training on mini-batches


@dataclass(kw_only=True)
class MiniBatchTraining(ABC):
    """
    Training that updates parameters once per mini-batch of the training arrays.
    """

    arguments: tuple[BackendArray, ...]
    """
    The training arrays, one row per training example each.
    """

    mini_batches: MiniBatches
    """
    How the rows of :attr:`arguments` are split into the batches of every epoch.
    """

    @abstractmethod
    def step_on_batch(self, *, batch: tuple[BackendArray, ...]) -> float | BackendArray:
        """
        Update the parameters by one step on a batch.

        :param batch: The same rows of every training array.
        :return: The loss of the batch before the step, a float or a scalar array of the
            backend that may still be computed on its device.
        """

    def step_on_rows(self, *, rows: np.ndarray) -> float | BackendArray:
        """
        Update the parameters by one step on the batch of some rows of every training
        array, see :meth:`step_on_batch`.

        :param rows: The indices of the rows of the batch.
        :return: The loss of the batch before the step.
        """
        return self.step_on_batch(
            batch=self.mini_batches.selected_rows(arguments=self.arguments, rows=rows)
        )

    def finish_epoch(self, *, average_loss: float) -> None:
        """
        React to the end of an epoch, nothing by default.

        :param average_loss: The average loss of the epoch.
        """

    def has_settled(self) -> bool:
        """
        :return: Whether the training's own schedule lets it stop, always by default.
        """
        return True

    def train_one_epoch(self) -> float:
        """
        Take one step on every batch of an epoch.

        :return: The average loss over the rows of the epoch.
        """
        total_loss = 0.0
        for rows in self.mini_batches.row_indices_of_epoch(
            number_of_rows=self.arguments[0].shape[0]
        ):
            total_loss += self.step_on_rows(rows=rows) * rows.shape[0]
        return float(total_loss) / self.arguments[0].shape[0]


@dataclass(kw_only=True)
class PlateauScheduledTraining(MiniBatchTraining, ABC):
    """
    Mini-batch training whose learning rate a :class:`LearningRatePlateau` reduces, and
    which stops only once the plateau has settled.
    """

    learning_rate_plateau: LearningRatePlateau | None = None
    """
    The schedule of the learning rate, a constant learning rate if ``None``.
    """

    @abstractmethod
    def scale_learning_rate(self, *, scale: float) -> None:
        """
        :param scale: The factor the initial learning rate is multiplied by from now.
        """

    def finish_epoch(self, *, average_loss: float) -> None:
        if self.learning_rate_plateau is None:
            return
        self.learning_rate_plateau.record(average_loss=average_loss)
        self.scale_learning_rate(scale=self.learning_rate_plateau.scale)

    def has_settled(self) -> bool:
        return (
            self.learning_rate_plateau is None or self.learning_rate_plateau.is_settled
        )
