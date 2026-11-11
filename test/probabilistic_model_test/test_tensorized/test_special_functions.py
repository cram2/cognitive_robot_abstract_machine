from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from probabilistic_model.probabilistic_circuit.tensorized import special_functions
from probabilistic_model.probabilistic_circuit.tensorized.exceptions import (
    UnsupportedArrayBackendError,
)
from probabilistic_model.probabilistic_circuit.tensorized.special_functions import (
    JaxSpecialFunctions,
    ScipySpecialFunctions,
    SpecialFunctions,
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


def test_numpy_arrays_get_the_scipy_special_functions(location):
    assert isinstance(SpecialFunctions.from_arrays(location), ScipySpecialFunctions)


def test_jax_arrays_get_the_jax_special_functions(location):
    assert isinstance(
        SpecialFunctions.from_arrays(jnp.asarray(location)), JaxSpecialFunctions
    )


def test_arrays_of_an_unsupported_backend_raise(monkeypatch, location):
    monkeypatch.setattr(
        special_functions,
        special_functions.array_namespace.__name__,
        lambda *arrays: ModuleType("unsupported"),
    )
    with pytest.raises(UnsupportedArrayBackendError):
        SpecialFunctions.from_arrays(location)


# %% numpy


def test_numpy_normal_log_density_is_that_of_scipy(values, location, scale):
    np.testing.assert_array_equal(
        SpecialFunctions.from_arrays(location).normal_log_density(
            values, location, scale
        ),
        norm.logpdf(values, loc=location, scale=scale),
    )


def test_numpy_normal_cumulative_distribution_is_that_of_scipy(values, location, scale):
    np.testing.assert_array_equal(
        SpecialFunctions.from_arrays(location).normal_cumulative_distribution(
            values, location, scale
        ),
        norm.cdf(values, loc=location, scale=scale),
    )


# %% jax


def test_jax_normal_log_density_is_that_of_scipy(float64_jax, values, location, scale):
    jax_location, jax_scale = jnp.asarray(location), jnp.asarray(scale)
    result = jax.jit(SpecialFunctions.from_arrays(jax_location).normal_log_density)(
        jnp.asarray(values), jax_location, jax_scale
    )
    np.testing.assert_allclose(
        np.asarray(result), norm.logpdf(values, loc=location, scale=scale)
    )


def test_jax_normal_cumulative_distribution_is_that_of_scipy(
    float64_jax, values, location, scale
):
    jax_location, jax_scale = jnp.asarray(location), jnp.asarray(scale)
    result = jax.jit(
        SpecialFunctions.from_arrays(jax_location).normal_cumulative_distribution
    )(jnp.asarray(values), jax_location, jax_scale)
    np.testing.assert_allclose(
        np.asarray(result), norm.cdf(values, loc=location, scale=scale)
    )
