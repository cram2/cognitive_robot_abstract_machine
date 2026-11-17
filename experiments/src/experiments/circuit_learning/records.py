"""
The records of the circuit learning benchmark besides its measured rows: the runs it
consists of, how each run ended, and the circuit each training run learned.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

from experiments.circuit_learning.datasets import DatasetName
from experiments.circuit_learning.methods import BenchmarkMethods, LearningMethod
from experiments.circuit_learning.settings import ComputationSetting
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

# %% names


class RunKind(StrEnum):
    """
    What a run measures.
    """

    TRAINING = "training"
    QUERIES = "queries"


class RunStatus(StrEnum):
    """
    How a run ended.
    """

    FINISHED = "finished"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    OUT_OF_MEMORY = "out_of_memory"
    SKIPPED = "skipped"


# %% runs


@dataclass(kw_only=True, frozen=True)
class BenchmarkRun:
    """
    One run of the benchmark.
    """

    kind: RunKind
    """
    What the run measures.
    """

    dataset: DatasetName
    """
    The dataset.
    """

    method: LearningMethod
    """
    The method that learns, or learned, the circuit.
    """

    setting: ComputationSetting
    """
    The setting the run computes in.
    """

    @property
    def key(self) -> str:
        """
        :return: A name of the run unique within a benchmark, usable as a file name.
        """
        return (
            f"{self.kind}.{self.dataset}.{self.method.name.lower()}.{self.setting.key}"
        )

    def training_run(self) -> BenchmarkRun:
        """
        :return: The training run whose circuit a query run measures: the training of
            the same method on the same dataset in the method's first setting.
        """
        return BenchmarkRun(
            kind=RunKind.TRAINING,
            dataset=self.dataset,
            method=self.method,
            setting=BenchmarkMethods().find(name=self.method).settings()[0],
        )


@dataclass(kw_only=True)
class RunRecord:
    """
    How a run of the benchmark ended.
    """

    run: BenchmarkRun
    """
    The run.
    """

    status: RunStatus
    """
    How it ended.
    """

    duration: timedelta
    """
    The wall-clock time from its start to its end.
    """

    log_tail: str | None = None
    """
    The end of the output of a run that did not finish, ``None`` for one that did.
    """


@dataclass(kw_only=True)
class TrainedCircuit:
    """
    The circuit a training run learned, kept for the runs that measure its queries.
    """

    training_run: BenchmarkRun
    """
    The training run that learned it.
    """

    circuit: LayeredProbabilisticCircuit
    """
    The learned circuit, with host parameters.
    """
