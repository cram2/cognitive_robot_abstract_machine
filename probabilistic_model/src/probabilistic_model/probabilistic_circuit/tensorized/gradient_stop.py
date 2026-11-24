"""
Keeping values out of the gradient, for every array backend that differentiates
automatically; the Python Array API standard does not define it.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass
from types import ModuleType

from array_api_compat import array_namespace, is_jax_namespace, is_torch_namespace
from typing_extensions import Collection, Self

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
)

# %% gradient stop


@dataclass(kw_only=True)
class GradientStop:
    """
    Treats values as constants of automatic differentiation, for a backend that does
    not differentiate: the values stay as they are.
    """

    namespace: ModuleType
    """
    The Array API namespace of the backend.
    """

    @classmethod
    def from_arrays(cls, *, arrays: Collection[BackendArray]) -> Self:
        """
        :param arrays: Arrays of one backend.
        :return: The gradient stop of the subclass that supports that backend, this
            class if none does.
        """
        namespace = array_namespace(*arrays)
        for subclass in cls.__subclasses__():
            if subclass.supports(namespace=namespace):
                return subclass(namespace=namespace)
        return cls(namespace=namespace)

    @classmethod
    def supports(cls, *, namespace: ModuleType) -> bool:
        """
        :param namespace: An Array API namespace.
        :return: Whether this class stops the gradients of that backend.
        """
        return False

    def stop_gradient(self, *, values: BackendArray) -> BackendArray:
        """
        :param values: Arrays of the backend.
        :return: The same values, which automatic differentiation treats as constants.
        """
        return values


# %% backends


@dataclass(kw_only=True)
class JaxGradientStop(GradientStop):
    """
    The gradient stop of jax arrays.
    """

    @classmethod
    def supports(cls, *, namespace: ModuleType) -> bool:
        """
        :meth:`GradientStop.supports` for jax.
        """
        return is_jax_namespace(namespace)

    def stop_gradient(self, *, values: BackendArray) -> BackendArray:
        """
        :meth:`GradientStop.stop_gradient` for jax arrays.
        """
        # jax is optional, see the core boundary in pyproject.toml
        import jax

        return jax.lax.stop_gradient(values)


@dataclass(kw_only=True)
class TorchGradientStop(GradientStop):
    """
    The gradient stop of torch tensors.
    """

    @classmethod
    def supports(cls, *, namespace: ModuleType) -> bool:
        """
        :meth:`GradientStop.supports` for torch.
        """
        return is_torch_namespace(namespace)

    def stop_gradient(self, *, values: BackendArray) -> BackendArray:
        """
        :meth:`GradientStop.stop_gradient` for torch tensors, detached from their graph.
        """
        return values.detach()
