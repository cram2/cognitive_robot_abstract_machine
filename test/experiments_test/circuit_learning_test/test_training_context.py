"""
Tests of the limits a method of the circuit learning benchmark trains within.
"""

from __future__ import annotations

# %% imports
import math

from krrood_compute_resources.devices import Device

from experiments.circuit_learning.methods import (
    RandomTensorizedSumProductNetworkByGradientDescent,
    TrainingContext,
)
from experiments.circuit_learning.settings import (
    ArrayLibrary,
    ComputationSetting,
    EdgeRepresentation,
    FloatingPointType,
)
from probabilistic_model.learning.parameters.gradient_descent.gradient_descent import (
    AverageNegativeLogLikelihood,
)
from probabilistic_model.learning.parameters.gradient_descent.jax_gradient_descent import (
    JaxGradientDescent,
)
from probabilistic_model.learning.parameters.training_schedule import Convergence

# %% rows per step


def test_the_rows_of_a_step_are_measured_with_the_edges_of_the_setting(dataset):
    """
    The rows that fit in a step are those of the step with the setting's own edge
    representation, not of one with dense edges.
    """
    circuit = RandomTensorizedSumProductNetworkByGradientDescent().random_circuit(
        dataset=dataset
    )
    setting = ComputationSetting(
        array_library=ArrayLibrary.JAX,
        device=Device.CENTRAL_PROCESSOR,
        floating_point_type=FloatingPointType.FLOAT32,
        edge_representation=EdgeRepresentation.BLOCK,
    )
    rows = dataset.encoded(frame=dataset.train)
    placed = setting.placed(circuit=circuit)
    step_memory = JaxGradientDescent(
        layer=placed.root, loss=AverageNegativeLogLikelihood()
    ).step_memory(setting.asarray(rows=placed.encoded(rows)))
    expected_rows = len(rows) // 4
    context = TrainingContext(
        convergence=Convergence(),
        memory_budget_bytes=math.ceil(
            step_memory.fixed_bytes + (expected_rows + 0.5) * step_memory.bytes_per_row
        ),
    )

    assert (
        context.rows_per_step(circuit=circuit, setting=setting, rows=rows)
        == expected_rows
    )
