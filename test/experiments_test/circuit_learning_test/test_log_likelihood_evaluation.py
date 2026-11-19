"""
Tests of the chunked log-likelihood evaluation of the circuit learning benchmark.
"""

from __future__ import annotations

# %% imports
import jax.numpy as jnp
import numpy as np
from random_events.variable import Continuous
from sortedcontainers import SortedSet

from experiments.circuit_learning.measurements import ChunkedLogLikelihoodEvaluation
from probabilistic_model.learning.region_graph.region_graph import RegionGraph

# %% jax circuits


def test_chunked_log_likelihoods_of_a_jax_circuit_are_those_of_its_host_circuit():
    """
    A jax circuit evaluated in chunks, the last one shorter, gives every row the log-
    likelihood its numpy circuit gives it.
    """
    variables = SortedSet([Continuous(f"x{index}") for index in range(4)])
    host = (
        RegionGraph(variables=variables, depth=1, repetitions=2)
        .create_random_region_graph()
        .to_probabilistic_circuit(input_units=2, sum_units=2)
    )
    rows = np.random.default_rng(0).normal(size=(10, 4))
    evaluation = ChunkedLogLikelihoodEvaluation(
        circuit=host.to_namespace(jnp), chunk_rows=4
    )
    np.testing.assert_allclose(
        evaluation.compute_log_likelihoods(rows=rows),
        host.log_likelihood(rows),
        rtol=1e-5,
    )
