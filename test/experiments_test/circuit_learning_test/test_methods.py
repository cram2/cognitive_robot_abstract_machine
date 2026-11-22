"""
Tests of the learning methods of the circuit learning benchmark that train RAT-SPNs by
expectation maximization.
"""

from __future__ import annotations

# %% imports
import jax
import numpy as np
from krrood_compute_resources.devices import Device

from experiments.circuit_learning.methods import (
    RandomTensorizedSumProductNetworkByExpectationMaximization,
    RandomTensorizedSumProductNetworkByOnlineExpectationMaximization,
    TrainingContext,
)
from experiments.circuit_learning.settings import (
    ArrayLibrary,
    ComputationSetting,
    EdgeRepresentation,
    FloatingPointType,
)
from probabilistic_model.learning.training_schedule import Convergence

# %% settings


def test_expectation_maximization_learns_on_the_host_and_in_every_setting_of_the_grid():
    """
    Full-batch expectation maximization runs on the host and in every jax and torch
    setting; the online one only where it is tensorized.
    """
    grid = ComputationSetting.grid(
        array_libraries=(ArrayLibrary.JAX, ArrayLibrary.TORCH)
    )

    assert RandomTensorizedSumProductNetworkByExpectationMaximization().settings() == [
        ComputationSetting.host(),
        *grid,
    ]
    assert (
        RandomTensorizedSumProductNetworkByOnlineExpectationMaximization().settings()
        == grid
    )


# %% learning


def test_tensorized_expectation_maximization_learns_what_the_host_one_learns(dataset):
    """
    Two epochs of full-batch expectation maximization with jax in double precision give
    every row the log-likelihood that two iterations on the host give it.
    """
    method = RandomTensorizedSumProductNetworkByExpectationMaximization()
    context = TrainingContext(
        convergence=Convergence(maximum_epochs=2, patience=2),
        memory_budget_bytes=2**30,
    )
    jax_setting = ComputationSetting(
        array_library=ArrayLibrary.JAX,
        device=Device.CENTRAL_PROCESSOR,
        floating_point_type=FloatingPointType.FLOAT64,
        edge_representation=EdgeRepresentation.BLOCK,
    )
    rows = dataset.encoded(frame=dataset.train)
    with jax.enable_x64(True):
        tensorized = method.learn(
            dataset=dataset, setting=jax_setting, context=context
        ).circuit
        tensorized_log_likelihoods = np.asarray(
            tensorized.log_likelihood(events=jax_setting.asarray(rows=rows))
        )
    host = method.learn(
        dataset=dataset, setting=ComputationSetting.host(), context=context
    ).circuit

    np.testing.assert_allclose(
        tensorized_log_likelihoods, host.log_likelihood(events=rows), rtol=1e-8
    )
