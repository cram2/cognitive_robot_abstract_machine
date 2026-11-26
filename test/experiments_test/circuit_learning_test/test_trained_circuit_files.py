"""
Tests of the files that keep the circuits training runs learned for the runs that
measure their queries.
"""

from __future__ import annotations

# %% imports
from pathlib import Path

import numpy as np
import pytest

from experiments.circuit_learning.database import TrainedCircuitFiles
from experiments.circuit_learning.datasets import DatasetName
from experiments.circuit_learning.methods import BenchmarkMethods, LearningMethod
from experiments.circuit_learning.records import BenchmarkRun, RunKind
from experiments.circuit_learning.settings import ComputationSetting
from probabilistic_model.learning.structure.hidden_chow_liu_tree.hidden_chow_liu_tree import (
    HiddenChowLiuTree,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)


@pytest.fixture
def training_run() -> BenchmarkRun:
    """
    The training of the hidden Chow-Liu tree on NLTCS in its first setting.
    """
    return BenchmarkRun(
        kind=RunKind.TRAINING,
        dataset=DatasetName.NLTCS,
        method=LearningMethod.HIDDEN_CHOW_LIU_TREE,
        setting=BenchmarkMethods()
        .find(name=LearningMethod.HIDDEN_CHOW_LIU_TREE)
        .settings()[0],
    )


@pytest.fixture
def circuit(dataset) -> LayeredProbabilisticCircuit:
    """
    An untrained hidden Chow-Liu tree circuit of the dataset.
    """
    return HiddenChowLiuTree(
        variables=dataset.variables, hidden_states=3
    ).build_circuit(data=dataset.encoded(frame=dataset.train))


# %% files


def test_a_written_circuit_is_read_back_with_the_same_likelihoods(
    tmp_path: Path, training_run, circuit, dataset
):
    """
    The circuit read from the file of a training run gives every row the likelihood of
    the circuit written to it.
    """
    files = TrainedCircuitFiles(directory=tmp_path)
    rows = dataset.encoded(frame=dataset.test)

    files.write(training_run=training_run, circuit=circuit)
    read = files.read(training_run=training_run)

    np.testing.assert_allclose(
        read.log_likelihood(events=rows), circuit.log_likelihood(events=rows)
    )


def test_every_training_run_has_a_file_named_after_it(
    tmp_path: Path, training_run, circuit
):
    """
    The file of a training run is named after the run's key, in the directory.
    """
    files = TrainedCircuitFiles(directory=tmp_path / "circuits")

    files.write(training_run=training_run, circuit=circuit)

    assert files.path(training_run=training_run).parent == files.directory
    assert files.path(training_run=training_run).stem == training_run.key
    assert files.path(training_run=training_run).exists()


def test_a_query_run_reads_the_circuit_of_its_training_run(
    tmp_path: Path, training_run, circuit, dataset
):
    """
    A run that measures queries in another setting finds the circuit its training run
    wrote.
    """
    files = TrainedCircuitFiles(directory=tmp_path)
    files.write(training_run=training_run, circuit=circuit)
    query_run = BenchmarkRun(
        kind=RunKind.QUERIES,
        dataset=training_run.dataset,
        method=training_run.method,
        setting=ComputationSetting.host(),
    )
    rows = dataset.encoded(frame=dataset.test)

    read = files.read(training_run=query_run.training_run())

    np.testing.assert_allclose(
        read.log_likelihood(events=rows), circuit.log_likelihood(events=rows)
    )
