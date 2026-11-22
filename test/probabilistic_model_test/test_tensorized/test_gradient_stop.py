"""
Tests that values can be kept out of the gradient on every array backend that
differentiates automatically.
"""

from __future__ import annotations

# %% imports
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from array_api_compat import array_namespace

from probabilistic_model.probabilistic_circuit.tensorized.gradient_stop import (
    GradientStop,
    JaxGradientStop,
    TorchGradientStop,
)

# %% the backend is chosen by the arrays


def test_numpy_arrays_get_the_gradient_stop_that_keeps_them():
    """
    Numpy, which does not differentiate, selects the gradient stop that returns the
    values as they are.
    """
    values = np.array([1.0, -2.0])
    stop = GradientStop.from_arrays(values)
    assert type(stop) is GradientStop
    assert stop.stop_gradient(values=values) is values


def test_jax_arrays_get_the_jax_gradient_stop():
    """
    Jax arrays select the gradient stop of jax.
    """
    assert isinstance(GradientStop.from_arrays(jnp.zeros(2)), JaxGradientStop)


def test_torch_tensors_get_the_torch_gradient_stop():
    """
    Torch tensors select the gradient stop of torch.
    """
    torch = pytest.importorskip("torch")
    assert isinstance(GradientStop.from_arrays(torch.zeros(2)), TorchGradientStop)


# %% stopped values have no gradient


def test_stopped_jax_values_have_no_gradient():
    """
    The gradient of the stopped values times the values is the values: the stopped
    factor counts as a constant.
    """
    values = jnp.asarray([1.0, -2.0, 0.5])

    def total(x: jax.Array) -> jax.Array:
        """
        :return: The sum of the stopped values times the values.
        """
        return jnp.sum(GradientStop.from_arrays(x).stop_gradient(values=x) * x)

    np.testing.assert_allclose(np.asarray(jax.grad(total)(values)), np.asarray(values))


def test_stopped_torch_values_have_no_gradient():
    """
    The gradient of the stopped tensor times the tensor is the tensor: the stopped
    factor counts as a constant.
    """
    torch = pytest.importorskip("torch")
    values = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)
    namespace = array_namespace(values)
    namespace.sum(
        GradientStop.from_arrays(values).stop_gradient(values=values) * values
    ).backward()
    np.testing.assert_allclose(values.grad.numpy(), values.detach().numpy())
