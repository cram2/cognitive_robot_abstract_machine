"""
Tests of the normal distribution functions of every supported array backend.
"""

from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from probabilistic_model.probabilistic_circuit.tensorized import (
    normal_distribution_functions,
)
from probabilistic_model.probabilistic_circuit.tensorized.exceptions import (
    UnsupportedArrayBackendError,
)
from probabilistic_model.probabilistic_circuit.tensorized.normal_distribution_functions import (
    JaxNormalDistributionFunctions,
    ScipyNormalDistributionFunctions,
    NormalDistributionFunctions,
)

# %% fixtures


@pytest.fixture
def values() -> np.ndarray:
    """
    A column of values, with both infinities among them.
    """
    return np.array([[-np.inf], [-3.0], [-0.5], [0.0], [1.25], [40.0], [np.inf]])


@pytest.fixture
def location() -> np.ndarray:
    """
    The means of three normal distributions.
    """
    return np.array([-1.0, 0.0, 2.5])


@pytest.fixture
def scale() -> np.ndarray:
    """
    The standard deviations of three normal distributions.
    """
    return np.array([0.5, 1.0, 3.0])


@pytest.fixture
def float64_jax():
    """
    Let jax compute in double precision for the duration of a test.
    """
    with jax.enable_x64(True):
        yield


# %% the backend is chosen by the arrays


def test_numpy_arrays_get_the_scipy_functions(location):
    """
    Numpy parameters select the functions scipy computes.
    """
    assert isinstance(
        NormalDistributionFunctions.from_arrays(location),
        ScipyNormalDistributionFunctions,
    )


def test_jax_arrays_get_the_jax_functions(location):
    """
    Jax parameters select the functions jax computes.
    """
    assert isinstance(
        NormalDistributionFunctions.from_arrays(jnp.asarray(location)),
        JaxNormalDistributionFunctions,
    )


def test_arrays_of_an_unsupported_backend_raise(monkeypatch, location):
    """
    Arrays of a backend no subclass supports raise instead of falling back.
    """
    monkeypatch.setattr(
        normal_distribution_functions,
        normal_distribution_functions.array_namespace.__name__,
        lambda *arrays: ModuleType("unsupported"),
    )
    with pytest.raises(UnsupportedArrayBackendError):
        NormalDistributionFunctions.from_arrays(location)


# %% numpy


def test_numpy_log_density_is_that_of_scipy(values, location, scale):
    """
    On numpy the log-density equals that of :data:`scipy.stats.norm` bit for bit.
    """
    np.testing.assert_array_equal(
        NormalDistributionFunctions.from_arrays(location).log_density(
            values, location, scale
        ),
        norm.logpdf(values, loc=location, scale=scale),
    )


def test_numpy_cumulative_distribution_is_that_of_scipy(values, location, scale):
    """
    On numpy the distribution function equals that of :data:`scipy.stats.norm` bit for
    bit.
    """
    np.testing.assert_array_equal(
        NormalDistributionFunctions.from_arrays(location).cumulative_distribution(
            values, location, scale
        ),
        norm.cdf(values, loc=location, scale=scale),
    )


# %% jax


def test_jax_log_density_is_that_of_scipy(float64_jax, values, location, scale):
    """
    On jax, compiled, the log-density agrees with that of :data:`scipy.stats.norm`.
    """
    jax_location, jax_scale = jnp.asarray(location), jnp.asarray(scale)
    result = jax.jit(NormalDistributionFunctions.from_arrays(jax_location).log_density)(
        jnp.asarray(values), jax_location, jax_scale
    )
    np.testing.assert_allclose(
        np.asarray(result), norm.logpdf(values, loc=location, scale=scale)
    )


def test_jax_cumulative_distribution_is_that_of_scipy(
    float64_jax, values, location, scale
):
    """
    On jax, compiled, the distribution function agrees with that of
    :data:`scipy.stats.norm`.
    """
    jax_location, jax_scale = jnp.asarray(location), jnp.asarray(scale)
    result = jax.jit(
        NormalDistributionFunctions.from_arrays(jax_location).cumulative_distribution
    )(jnp.asarray(values), jax_location, jax_scale)
    np.testing.assert_allclose(
        np.asarray(result), norm.cdf(values, loc=location, scale=scale)
    )
