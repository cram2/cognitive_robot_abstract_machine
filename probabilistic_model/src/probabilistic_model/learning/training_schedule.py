"""
When gradient descent stops and which rows each of its steps sees, independent of the
array backend that trains.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from array_api_compat import array_namespace, device
from typing_extensions import Iterator

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
    Whether training stopped because the loss converged rather than at the epoch limit.
    """

    @property
    def epochs(self) -> int:
        """
        :return: The number of epochs training ran.
        """
        return len(self.epoch_losses)


@dataclass(kw_only=True)
class Convergence:
    """
    Stops training once the average loss of an epoch improved by less than
    :attr:`tolerance` for :attr:`patience` epochs in a row, or after
    :attr:`maximum_epochs`.
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
        while len(epoch_losses) < self.maximum_epochs:
            epoch_losses.append(training.train_one_epoch())
            if self.has_converged(epoch_losses=epoch_losses):
                return ConvergenceHistory(epoch_losses=epoch_losses, converged=True)
        return ConvergenceHistory(epoch_losses=epoch_losses, converged=False)


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

    def batches_of_epoch(
        self, *, arguments: tuple[BackendArray, ...]
    ) -> Iterator[tuple[BackendArray, ...]]:
        """
        :param arguments: The training arrays, one row per training example each.
        :return: The batches of one epoch, each holding the same rows of every array.
        """
        number_of_rows = arguments[0].shape[0]
        order = self.random_number_generator.permutation(number_of_rows)
        step = number_of_rows if self.batch_size is None else self.batch_size
        for start in range(0, number_of_rows, step):
            rows = order[start : start + step]
            yield tuple(
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
    def step_on_batch(self, *, batch: tuple[BackendArray, ...]) -> float:
        """
        Update the parameters by one step on a batch.

        :param batch: The same rows of every training array.
        :return: The loss of the batch before the step.
        """

    def train_one_epoch(self) -> float:
        """
        Take one step on every batch of an epoch.

        :return: The average loss over the rows of the epoch.
        """
        total_loss = 0.0
        for batch in self.mini_batches.batches_of_epoch(arguments=self.arguments):
            total_loss += self.step_on_batch(batch=batch) * batch[0].shape[0]
        return total_loss / self.arguments[0].shape[0]
