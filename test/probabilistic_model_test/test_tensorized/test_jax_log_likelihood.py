"""
Tests for the compiled log-likelihood of a layered circuit with jax parameters.
"""

from __future__ import annotations

# %% imports
import jax
import jax.numpy as jnp
import numpy as np

from probabilistic_model.jax_log_likelihood import CompiledLogLikelihood


# %% compiled log-likelihood


def test_compiled_log_likelihood_is_the_log_likelihood_of_the_circuit(
    mixture_of_products_case,
):
    """
    The compiled log-likelihood of events is a jax array equal to the log-likelihood the
    circuit computes without compilation.
    """
    case = mixture_of_products_case
    with jax.enable_x64(True):
        circuit = case.circuit(jnp)
        events = case.events(jnp)
        compiled = CompiledLogLikelihood(circuit=circuit).log_likelihood(
            encoded_events=circuit.encoded(events)
        )
        assert isinstance(compiled, jax.Array)
        np.testing.assert_allclose(
            np.asarray(compiled),
            np.asarray(circuit.log_likelihood(events=events)),
            rtol=1e-12,
        )


def test_compiled_log_likelihood_of_a_second_shape_of_events(
    mixture_of_products_case,
):
    """
    Events of another number of rows than the first call still get the log-likelihood of
    the circuit.
    """
    case = mixture_of_products_case
    with jax.enable_x64(True):
        circuit = case.circuit(jnp)
        events = case.events(jnp)
        log_likelihood = CompiledLogLikelihood(circuit=circuit)
        log_likelihood.log_likelihood(encoded_events=circuit.encoded(events))
        np.testing.assert_allclose(
            np.asarray(
                log_likelihood.log_likelihood(
                    encoded_events=circuit.encoded(events[:2])
                )
            ),
            np.asarray(circuit.log_likelihood(events=events[:2])),
            rtol=1e-12,
        )
