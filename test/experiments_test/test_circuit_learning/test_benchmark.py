"""
Tests for the circuit learning benchmark: its datasets, its measurements, the storage of
circuits and its runner, on a dataset placed in the cache so that nothing is downloaded.
"""

from __future__ import annotations

# %% imports
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.circuit_learning.datasets import (
    BinaryDatasetFile,
    BinaryDatasetSource,
    BinarySplit,
    DatasetName,
    MaskedAutoregressiveFlowSource,
)
from experiments.circuit_learning.measurements import (
    QueryMeasurement,
    TrainingMeasurement,
)
from experiments.circuit_learning.methods import (
    JointProbabilityTreeBaseline,
    LearningMethod,
    RandomTensorizedSumProductNetworkByExpectationMaximization,
    TrainingContext,
)
from experiments.circuit_learning.rows import (
    BenchmarkConfiguration,
    ConditioningSpeedResult,
    ExactQueriesResult,
    LikelihoodSpeedResult,
    Query,
    SamplingSpeedResult,
    TruncationSpeedResult,
)
from experiments.circuit_learning.runner import (
    BenchmarkRunner,
    JobKind,
    ResultFile,
    RunStatus,
)
from experiments.circuit_learning.settings import ComputationSetting
from experiments.circuit_learning.storage import CircuitFile
from probabilistic_model.learning.training_schedule import Convergence

# %% datasets in the cache


@pytest.fixture
def cache_directory(tmp_path: Path) -> Path:
    """
    A cache holding the three splits of a small binary dataset under the name of NLTCS.
    """
    generator = np.random.default_rng(0)
    for split, rows in zip(BinarySplit, (200, 50, 50)):
        bits = (generator.random((rows, 6)) < 0.4).astype(int)
        bits[:, 1] = bits[:, 0]
        path = tmp_path / "binary" / f"{BinaryDatasetFile.NLTCS}.{split}.data"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, bits, fmt="%d", delimiter=",")
    return tmp_path


@pytest.fixture
def dataset(cache_directory):
    """
    The small binary dataset of :func:`cache_directory`.
    """
    return BinaryDatasetSource(
        cache_directory=cache_directory, dataset=BinaryDatasetFile.NLTCS
    ).load()


def test_binary_source_reads_the_cached_splits_as_symbolic_variables(dataset):
    """
    A cached binary dataset loads as symbolic variables encoded by hash.
    """
    assert (len(dataset.train), len(dataset.validation), len(dataset.test)) == (
        200,
        50,
        50,
    )
    assert dataset.is_symbolic
    encoded = dataset.encoded(frame=dataset.train)
    assert set(np.unique(encoded)) <= {float(hash("0")), float(hash("1"))}


def test_tenths_split_off_as_the_paper_splits_them():
    """
    The last tenth becomes the test rows and the last tenth of the rest the validation
    rows.
    """
    train, validation, test = MaskedAutoregressiveFlowSource.split_off_tenths(
        data=np.arange(100).reshape(100, 1)
    )
    assert (len(train), len(validation), len(test)) == (81, 9, 10)
    assert test[0, 0] == 90 and validation[0, 0] == 81


# %% measurements


def test_training_measures_and_stores_the_circuit(dataset, tmp_path: Path):
    """
    Training a circuit reports its fit and stores it so that it loads with the same
    likelihoods.
    """
    circuit_file = CircuitFile(path=tmp_path / "circuit.pickle")
    result = TrainingMeasurement(
        dataset=dataset,
        method=JointProbabilityTreeBaseline(),
        setting=ComputationSetting.host(),
        context=TrainingContext(
            convergence=Convergence(maximum_epochs=2), memory_budget_bytes=2**30
        ),
        circuit_file=circuit_file,
    ).run()
    rows = dataset.encoded(frame=dataset.test)
    assert result.average_test_log_likelihood == pytest.approx(
        float(circuit_file.load().log_likelihood(rows).mean())
    )
    assert result.fraction_of_test_rows_without_support == 0.0
    assert result.epochs_until_convergence is None


def test_queries_on_the_host_measure_every_query(dataset, tmp_path: Path):
    """
    The queries of a circuit on the host give one row per query and size.
    """
    circuit = JointProbabilityTreeBaseline().learn(
        dataset=dataset,
        setting=ComputationSetting.host(),
        context=TrainingContext(convergence=Convergence(), memory_budget_bytes=2**30),
    )
    measurement = QueryMeasurement(
        dataset=dataset,
        configuration=BenchmarkConfiguration.from_setting(
            learning_method=LearningMethod.JOINT_PROBABILITY_TREE,
            dataset=DatasetName.NLTCS,
            setting=ComputationSetting.host(),
            seed=69,
        ),
        setting=ComputationSetting.host(),
        circuit=circuit.circuit,
        minimum_timing_seconds=0.0,
        query_sizes=(2, 5),
        repetitions=2,
    )
    rows = measurement.run()
    counts = {
        row_type: sum(isinstance(row, row_type) for row in rows)
        for row_type in (
            LikelihoodSpeedResult,
            SamplingSpeedResult,
            ConditioningSpeedResult,
            TruncationSpeedResult,
            ExactQueriesResult,
        )
    }
    assert counts == {
        LikelihoodSpeedResult: 1,
        SamplingSpeedResult: 1,
        ConditioningSpeedResult: 2,
        TruncationSpeedResult: 2,
        ExactQueriesResult: 1,
    }


def test_a_truncation_event_has_the_requested_disjoint_simple_sets(dataset):
    """
    The cells of distinct test rows form an event of as many simple sets as asked.
    """
    circuit = (
        JointProbabilityTreeBaseline()
        .learn(
            dataset=dataset,
            setting=ComputationSetting.host(),
            context=TrainingContext(
                convergence=Convergence(), memory_budget_bytes=2**30
            ),
        )
        .circuit
    )
    measurement = QueryMeasurement(
        dataset=dataset,
        configuration=BenchmarkConfiguration.from_setting(
            learning_method=LearningMethod.JOINT_PROBABILITY_TREE,
            dataset=DatasetName.NLTCS,
            setting=ComputationSetting.host(),
            seed=69,
        ),
        setting=ComputationSetting.host(),
        circuit=circuit,
    )
    event = measurement.cells_of_test_rows(circuit=circuit, number_of_cells=5)
    assert len(event.simple_sets) == 5


def test_a_deterministic_circuit_answers_most_likely_assignments_exactly(dataset):
    """
    A joint probability tree is deterministic, so it answers most likely assignments
    exactly.
    """
    circuit = (
        JointProbabilityTreeBaseline()
        .learn(
            dataset=dataset,
            setting=ComputationSetting.host(),
            context=TrainingContext(
                convergence=Convergence(), memory_budget_bytes=2**30
            ),
        )
        .circuit
    )
    result = QueryMeasurement(
        dataset=dataset,
        configuration=BenchmarkConfiguration.from_setting(
            learning_method=LearningMethod.JOINT_PROBABILITY_TREE,
            dataset=DatasetName.NLTCS,
            setting=ComputationSetting.host(),
            seed=69,
        ),
        setting=ComputationSetting.host(),
        circuit=circuit,
    ).exact_queries()
    assert Query.MOST_LIKELY_ASSIGNMENT in result.exact_queries


def test_expectation_maximization_trains_a_random_sum_product_network_until_convergence(
    dataset,
):
    """
    The RAT-SPN that expectation maximization learns converges on the training rows,
    each epoch at most as costly as the one before.
    """
    learned = RandomTensorizedSumProductNetworkByExpectationMaximization(
        depth=1, repetitions=2, units=2
    ).learn(
        dataset=dataset,
        setting=ComputationSetting.host(),
        context=TrainingContext(
            convergence=Convergence(maximum_epochs=200), memory_budget_bytes=2**30
        ),
    )
    losses = learned.history.epoch_losses

    assert learned.history.converged
    assert all(later <= earlier + 1e-9 for earlier, later in zip(losses, losses[1:]))


def test_expectation_maximization_learns_the_circuit_the_next_epoch_starts_from(
    dataset,
):
    """
    The learned layered circuit has the training log-likelihood before the epoch that
    one more epoch of training would take.
    """
    method = RandomTensorizedSumProductNetworkByExpectationMaximization(
        depth=1, repetitions=2, units=2
    )
    epochs = 3

    def learned_in(maximum_epochs: int):
        return method.learn(
            dataset=dataset,
            setting=ComputationSetting.host(),
            context=TrainingContext(
                convergence=Convergence(tolerance=0.0, maximum_epochs=maximum_epochs),
                memory_budget_bytes=2**30,
            ),
        )

    learned = learned_in(epochs)
    rows = dataset.encoded(frame=dataset.train)

    assert float(learned.circuit.log_likelihood(rows).mean()) == pytest.approx(
        -learned_in(epochs + 1).history.epoch_losses[-1]
    )


# %% runner


def test_the_runner_trains_and_queries_on_the_host(cache_directory, tmp_path: Path):
    """
    The runner trains a host method, then measures its queries in every host setting,
    recording every run as finished.
    """
    output = tmp_path / "results"
    runner = BenchmarkRunner(
        datasets=[DatasetName.NLTCS],
        methods=[LearningMethod.JOINT_PROBABILITY_TREE],
        output=output,
        cache_directory=cache_directory,
        gpu_slots=0,
        cpu_slots=2,
        poll_seconds=0.5,
    )
    host_jobs = [job for job in runner.jobs() if not job.setting.uses_gpu]
    runner.run_jobs(jobs=host_jobs)
    records = [
        json.loads(line)
        for line in (output / ResultFile.RUNS.value).read_text().splitlines()
    ]
    assert {record["status"] for record in records} == {RunStatus.FINISHED.value}
    assert sum(record["kind"] == JobKind.TRAINING.name for record in records) == 1
    assert len(records) == len(host_jobs)


def test_the_first_settings_alone_leave_out_every_other_training(cache_directory):
    """
    Restricted to the first settings, the runner trains each method once, in its first
    setting, and still measures the queries of every setting.
    """
    arguments = dict(
        datasets=[DatasetName.NLTCS],
        methods=[LearningMethod.RAT_SPN_GRADIENT_DESCENT],
        output=cache_directory / "results",
        cache_directory=cache_directory,
    )
    every_job = BenchmarkRunner(**arguments).jobs()
    first_jobs = BenchmarkRunner(**arguments, first_settings_only=True).jobs()

    assert [job for job in first_jobs if job.kind is JobKind.TRAINING] == [every_job[0]]
    assert [job for job in first_jobs if job.kind is JobKind.QUERIES] == [
        job for job in every_job if job.kind is JobKind.QUERIES
    ]
