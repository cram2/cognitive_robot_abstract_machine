"""
Tests of the limits a method of the circuit learning benchmark trains within.
"""

from __future__ import annotations

# %% imports
import math

import numpy as np
import pytest
from krrood_compute_resources.devices import Device

from experiments.circuit_learning.datasets import BenchmarkDataset
from experiments.circuit_learning.methods import (
    ExpectationMaximizationPhase,
    ParameterLearnerBuilder,
    RandomTensorizedSumProductNetworkByGradientDescent,
    TrainingContext,
)
from experiments.circuit_learning.settings import (
    ArrayLibrary,
    ComputationSetting,
    EdgeRepresentation,
    FloatingPointType,
)
from probabilistic_model.learning.parameters.training_schedule import Convergence
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)


@pytest.fixture
def circuit(dataset: BenchmarkDataset) -> LayeredProbabilisticCircuit:
    """
    The RAT-SPN of the gradient descent baseline over the dataset, with host parameters.
    """
    return RandomTensorizedSumProductNetworkByGradientDescent().random_circuit(
        dataset=dataset
    )


def central_processor_setting(
    *, edge_representation: EdgeRepresentation
) -> ComputationSetting:
    """
    :param edge_representation: How the edges are stored.
    :return: The jax setting on the central processor in single precision.
    """
    return ComputationSetting(
        array_library=ArrayLibrary.JAX,
        device=Device.CENTRAL_PROCESSOR,
        floating_point_type=FloatingPointType.FLOAT32,
        edge_representation=edge_representation,
    )


def context_fitting_rows(
    *,
    builder: ParameterLearnerBuilder,
    circuit: LayeredProbabilisticCircuit,
    setting: ComputationSetting,
    rows: np.ndarray,
    fitting_rows: int,
) -> TrainingContext:
    """
    :param builder: The builder of the learner whose step is measured.
    :param circuit: The circuit, with host parameters.
    :param setting: The setting that trains.
    :param rows: The encoded training rows.
    :param fitting_rows: The rows a step of the builder's learner should fit.
    :return: A training context whose memory budget fits exactly that many rows.
    """
    placed = setting.placed(circuit=circuit)
    step_memory = builder.learner(root=placed.root, setting=setting).step_memory(
        setting.asarray(rows=placed.encoded(rows))
    )
    return TrainingContext(
        convergence=Convergence(),
        memory_budget_bytes=math.ceil(
            step_memory.fixed_bytes + (fitting_rows + 0.5) * step_memory.bytes_per_row
        ),
    )


# %% rows per step


def test_the_rows_of_a_step_are_measured_with_the_edges_of_the_setting(
    dataset, circuit
):
    """
    The rows that fit in a step are those of the step with the setting's own edge
    representation, not of one with dense edges.
    """
    setting = central_processor_setting(edge_representation=EdgeRepresentation.BLOCK)
    builder = RandomTensorizedSumProductNetworkByGradientDescent()
    rows = dataset.encoded(frame=dataset.train)
    expected_rows = len(rows) // 4
    context = context_fitting_rows(
        builder=builder,
        circuit=circuit,
        setting=setting,
        rows=rows,
        fitting_rows=expected_rows,
    )

    assert (
        context.rows_per_step(
            circuit=circuit, setting=setting, rows=rows, learner_builder=builder
        )
        == expected_rows
    )


def test_the_rows_of_a_step_are_measured_with_the_learner_that_trains(dataset, circuit):
    """
    The rows that fit in a step of expectation maximization are those of its own step,
    not of a step of gradient descent.
    """
    setting = central_processor_setting(edge_representation=EdgeRepresentation.BLOCK)
    builder = ExpectationMaximizationPhase()
    rows = dataset.encoded(frame=dataset.train)
    expected_rows = len(rows) // 4
    context = context_fitting_rows(
        builder=builder,
        circuit=circuit,
        setting=setting,
        rows=rows,
        fitting_rows=expected_rows,
    )

    assert (
        context.rows_per_step(
            circuit=circuit, setting=setting, rows=rows, learner_builder=builder
        )
        == expected_rows
    )
